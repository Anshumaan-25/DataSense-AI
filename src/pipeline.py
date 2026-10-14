"""Bounded Self-Correction Repair Pipeline (State Machine) for Text-to-SQL."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
import duckdb
import pandas as pd

from src.database import get_connection, get_schema_info, execute_query, TPCH_TABLES
from src.validator import validate_sql, is_security_violation
from src.generator import generate_sql, generate_repaired_sql
from src.retriever import SchemaMetricRetriever
from src.ambiguity import AmbiguityClassifier, AmbiguityResult
from src.semantic import SemanticValidator, SemanticValidationResult


@dataclass
class PipelineResult:
    """Telemetry and outcome data from the self-correcting execution pipeline."""

    success: bool
    final_sql: str = ""
    df: Optional[pd.DataFrame] = None
    total_attempts: int = 1
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None
    is_security_violation: bool = False
    attempt_history: List[Dict[str, Any]] = field(default_factory=list)
    rag_context: str = ""
    rag_metadata: List[Dict[str, Any]] = field(default_factory=list)
    clarified_context: str = ""
    ambiguity_result: Optional[AmbiguityResult] = None
    semantic_validation: Optional[SemanticValidationResult] = None
    was_semantically_repaired: bool = False


def sanitize_error_message(raw_error: str) -> str:
    """Extract clean compiler or binder error messages, stripping Python tracebacks and ANSI codes."""
    if not raw_error:
        return ""

    # Remove ANSI escape color sequences
    clean_msg = re.sub(r"\x1b\[[0-9;]*[mGKH]", "", str(raw_error)).strip()

    # If it's a DuckDB error (e.g., Binder Error, Catalog Error, Parser Error)
    lines = clean_msg.splitlines()
    relevant_lines = []
    capture = False

    for line in lines:
        line_s = line.strip()
        # Look for the start of DuckDB compiler error keywords
        if any(
            err_prefix in line_s
            for err_prefix in (
                "Binder Error:",
                "Catalog Error:",
                "Parser Error:",
                "Permission Error:",
                "Conversion Error:",
                "Syntax Error:",
                "IO Error:",
            )
        ):
            capture = True
        if capture:
            # Stop if we hit generic python traceback markers
            if line_s.startswith("Traceback (most recent call last):"):
                break
            relevant_lines.append(line_s)

    if relevant_lines:
        result = "\n".join(relevant_lines)
    else:
        # Fallback to the first 4 non-empty lines
        result = "\n".join([l.strip() for l in lines if l.strip()][:4])

    return result[:600]


class SelfCorrectingPipeline:
    """Coordinating state machine that manages LLM generation, AST validation, and DuckDB self-repair."""

    def __init__(
        self,
        db_conn: Optional[duckdb.DuckDBPyConnection] = None,
        retriever: Optional[SchemaMetricRetriever] = None,
        generator_fn: Optional[Callable[..., str]] = None,
        repair_fn: Optional[Callable[..., str]] = None,
        classifier: Optional[AmbiguityClassifier] = None,
        semantic_validator: Optional[SemanticValidator] = None,
    ):
        """Initialize the pipeline with database, retriever, model functions, and validators.

        Args:
            db_conn: DuckDB connection instance.
            retriever: SchemaMetricRetriever instance.
            generator_fn: Function to generate initial SQL (defaults to generate_sql).
            repair_fn: Function to generate repaired SQL (defaults to generate_repaired_sql).
            classifier: AmbiguityClassifier instance.
            semantic_validator: SemanticValidator instance.
        """
        self.db_conn = db_conn if db_conn is not None else get_connection()
        self.retriever = retriever
        self.generator_fn = generator_fn if generator_fn is not None else generate_sql
        self.repair_fn = repair_fn if repair_fn is not None else generate_repaired_sql
        self.classifier = classifier if classifier is not None else AmbiguityClassifier()
        self.semantic_validator = (
            semantic_validator if semantic_validator is not None else SemanticValidator()
        )

    def check_ambiguity(
        self,
        user_question: str,
        schema_info: Optional[str] = None,
        rag_context: str = "",
        provider: Optional[str] = None,
    ) -> AmbiguityResult:
        """Classify user intent ambiguity and generate clarification options.

        Args:
            user_question: Natural language question submitted by user.
            schema_info: Database DDL or table schema summary (defaults to DB schema).
            rag_context: Business metric and filter rules retrieved via RAG.
            provider: LLM provider override ('gemini', 'openai').

        Returns:
            AmbiguityResult with 3-tier classification, reasoning, and structured options.
        """
        if schema_info is None:
            schema_info = get_schema_info(conn=self.db_conn, as_string=True)

        return self.classifier.classify_intent(
            user_question=user_question,
            schema_info=schema_info,
            rag_context=rag_context,
            provider=provider,
        )

    def execute_with_repair(
        self,
        user_question: str,
        max_retries: int = 3,
        provider: Optional[str] = None,
        clarified_context: str = "",
    ) -> PipelineResult:
        """Execute the self-correction loop with hard bounded retries.

        State Machine Transitions:
        1. Retrieve RAG Context (Metrics & Schema)
        2. Attempt Loop (attempt = 1 to max_retries):
           - Generate SQL (Initial or Repaired with preserved clarified context)
           - Check for Stagnation / Repeated Query
           - AST Safety Validation
             - If SECURITY VIOLATION: Fatal Abort (No retry)
             - If SYNTAX ERROR: Record & Retry
           - DuckDB Execution
             - If SUCCESS: Return PipelineResult(success=True)
             - If RUNTIME ERROR: Record & Retry
        3. If all attempts exhausted: Return PipelineResult(success=False)

        Args:
            user_question: The user's natural language question or direct SQL test.
            max_retries: Maximum number of attempts allowed (default 3).
            provider: LLM provider override ('gemini', 'openai').
            clarified_context: Optional user clarification of ambiguous intent.

        Returns:
            PipelineResult containing execution outcome and attempt telemetry.
        """
        start_time = time.perf_counter()

        if not user_question or not user_question.strip():
            return PipelineResult(
                success=False,
                error_message="Question is empty.",
                clarified_context=clarified_context,
            )

        # Step 1: Retrieve Schema & RAG Context
        schema_str = get_schema_info(conn=self.db_conn, as_string=True)
        rag_context = ""
        rag_metadata: List[Dict[str, Any]] = []

        if self.retriever is not None:
            try:
                rag_context, rag_metadata = self.retriever.retrieve(
                    user_question,
                    top_k_metrics=2,
                    top_k_filters=2,
                    top_k_tables=3,
                )
            except Exception:
                rag_context = ""
                rag_metadata = []

        # Check for direct SQL bypass test
        cleaned_input = user_question.strip()
        sql_keywords = ("SELECT ", "WITH ", "DROP ", "DELETE ", "UPDATE ", "INSERT ", "ALTER ", "CREATE ")
        is_direct_sql = cleaned_input.upper().startswith(sql_keywords)

        attempt_history: List[Dict[str, Any]] = []
        seen_queries: Set[str] = set()
        current_sql = ""
        last_error = ""

        # Step 2: Bounded Attempt Loop
        for attempt in range(1, max_retries + 1):
            # A. Generate or Repair SQL
            try:
                if attempt == 1:
                    if is_direct_sql:
                        current_sql = cleaned_input
                    else:
                        try:
                            current_sql = self.generator_fn(
                                user_question,
                                schema_str,
                                rag_context=rag_context,
                                clarified_context=clarified_context,
                                provider=provider,
                            )
                        except TypeError:
                            current_sql = self.generator_fn(
                                user_question,
                                schema_str,
                                rag_context=rag_context,
                                provider=provider,
                            )
                else:
                    last_record = attempt_history[-1]
                    try:
                        current_sql = self.repair_fn(
                            user_question=user_question,
                            schema_info=schema_str,
                            rag_context=rag_context,
                            failed_sql=last_record["sql"],
                            error_message=last_record["error"],
                            attempt_history=attempt_history,
                            clarified_context=clarified_context,
                            provider=provider,
                        )
                    except TypeError:
                        current_sql = self.repair_fn(
                            user_question=user_question,
                            schema_info=schema_str,
                            rag_context=rag_context,
                            failed_sql=last_record["sql"],
                            error_message=last_record["error"],
                            attempt_history=attempt_history,
                            provider=provider,
                        )
            except Exception as gen_exc:
                elapsed = (time.perf_counter() - start_time) * 1000
                attempt_history.append({
                    "attempt": attempt,
                    "sql": current_sql,
                    "error": str(gen_exc),
                    "error_type": "GENERATION_FAILURE",
                })
                return PipelineResult(
                    success=False,
                    final_sql=current_sql,
                    total_attempts=attempt,
                    execution_time_ms=elapsed,
                    error_message=f"Model generation error: {gen_exc}",
                    attempt_history=attempt_history,
                    rag_context=rag_context,
                    rag_metadata=rag_metadata,
                    clarified_context=clarified_context,
                )

            # B. Loop Stagnation & Cycle Detection
            normalized_sql = " ".join(current_sql.strip().split())
            if normalized_sql in seen_queries:
                elapsed = (time.perf_counter() - start_time) * 1000
                stagnation_msg = (
                    "Stagnation loop detected: the model repeated an identical SQL query "
                    "that previously failed."
                )
                attempt_history.append({
                    "attempt": attempt,
                    "sql": current_sql,
                    "error": stagnation_msg,
                    "error_type": "CYCLE_STAGNATION",
                })
                return PipelineResult(
                    success=False,
                    final_sql=current_sql,
                    total_attempts=attempt,
                    execution_time_ms=elapsed,
                    error_message=stagnation_msg,
                    attempt_history=attempt_history,
                    rag_context=rag_context,
                    rag_metadata=rag_metadata,
                    clarified_context=clarified_context,
                )
            seen_queries.add(normalized_sql)

            # C. AST Safety Validation
            is_valid, validation_msg = validate_sql(current_sql, allowed_tables=TPCH_TABLES)

            if not is_valid:
                # CRITICAL: Security Policy Violations MUST NEVER BE RETRIED
                if is_security_violation(validation_msg):
                    elapsed = (time.perf_counter() - start_time) * 1000
                    attempt_history.append({
                        "attempt": attempt,
                        "sql": current_sql,
                        "error": validation_msg,
                        "error_type": "SECURITY_VIOLATION",
                    })
                    return PipelineResult(
                        success=False,
                        final_sql=current_sql,
                        total_attempts=attempt,
                        execution_time_ms=elapsed,
                        error_message=validation_msg,
                        is_security_violation=True,
                        attempt_history=attempt_history,
                        rag_context=rag_context,
                        rag_metadata=rag_metadata,
                        clarified_context=clarified_context,
                    )

                # Benign syntax error -> Retriable
                attempt_history.append({
                    "attempt": attempt,
                    "sql": current_sql,
                    "error": validation_msg,
                    "error_type": "AST_SYNTAX",
                })
                last_error = validation_msg
                continue

            # D. Execution against DuckDB Engine
            df, query_err = execute_query(current_sql, conn=self.db_conn)

            if query_err is None and df is not None:
                # E. Semantic Validation
                sem_res = self.semantic_validator.validate_output(
                    user_question=user_question,
                    sql_query=current_sql,
                    df=df,
                    schema_info=schema_str,
                    rag_context=rag_context,
                    clarified_context=clarified_context,
                    provider=provider,
                )

                if not sem_res.is_valid:
                    sem_err_msg = f"{sem_res.reasoning} Suggested fix: {sem_res.suggested_fix}"
                    attempt_history.append({
                        "attempt": attempt,
                        "sql": current_sql,
                        "error": sem_err_msg,
                        "error_type": "SEMANTIC_VALIDATION",
                        "issue_type": sem_res.issue_type,
                    })
                    last_error = sem_err_msg
                    continue

                # SUCCESS! Both syntax and semantics are valid!
                elapsed = (time.perf_counter() - start_time) * 1000
                was_semantically_repaired = any(
                    rec.get("error_type") == "SEMANTIC_VALIDATION"
                    for rec in attempt_history
                )
                return PipelineResult(
                    success=True,
                    final_sql=current_sql,
                    df=df,
                    total_attempts=attempt,
                    execution_time_ms=elapsed,
                    attempt_history=attempt_history,
                    rag_context=rag_context,
                    rag_metadata=rag_metadata,
                    clarified_context=clarified_context,
                    semantic_validation=sem_res,
                    was_semantically_repaired=was_semantically_repaired,
                )

            # Runtime DuckDB error -> Check if engine-level security block
            clean_err = sanitize_error_message(query_err or "Unknown database error")
            if (
                "permission error" in clean_err.lower()
                or "disabled by configuration" in clean_err.lower()
            ):
                elapsed = (time.perf_counter() - start_time) * 1000
                attempt_history.append({
                    "attempt": attempt,
                    "sql": current_sql,
                    "error": clean_err,
                    "error_type": "SECURITY_VIOLATION",
                })
                return PipelineResult(
                    success=False,
                    final_sql=current_sql,
                    total_attempts=attempt,
                    execution_time_ms=elapsed,
                    error_message=clean_err,
                    is_security_violation=True,
                    attempt_history=attempt_history,
                    rag_context=rag_context,
                    rag_metadata=rag_metadata,
                    clarified_context=clarified_context,
                )

            # Benign DuckDB Runtime Binder/Parser Error -> Retriable
            attempt_history.append({
                "attempt": attempt,
                "sql": current_sql,
                "error": clean_err,
                "error_type": "DUCKDB_RUNTIME",
            })
            last_error = clean_err

        # Step 3: All Attempts Exhausted
        elapsed = (time.perf_counter() - start_time) * 1000
        return PipelineResult(
            success=False,
            final_sql=current_sql,
            total_attempts=max_retries,
            execution_time_ms=elapsed,
            error_message=f"Failed to auto-repair query after {max_retries} attempts. Last error: {last_error}",
            attempt_history=attempt_history,
            rag_context=rag_context,
            rag_metadata=rag_metadata,
            clarified_context=clarified_context,
        )
