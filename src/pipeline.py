"""Bounded Self-Correction Repair Pipeline (State Machine) for Text-to-SQL."""

from __future__ import annotations

import inspect
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
import duckdb
import pandas as pd

from src.database import (
    TPCH_TABLES,
    execute_query,
    get_connection,
    get_query_sample_rows,
    get_schema_info,
)
from src.validator import validate_sql, is_security_violation
from src.generator import generate_sql, generate_repaired_sql
from src.retriever import SchemaMetricRetriever
from src.ambiguity import AmbiguityClassifier, AmbiguityResult
from src.semantic import SemanticValidator, SemanticValidationResult
from src.llm import ProviderRequestError

logger = logging.getLogger(__name__)


def _call_model(fn: Callable, *args, **kwargs):
    """Preserve injected legacy callables without retrying errors raised inside them."""
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        kwargs = {
            key: value for key, value in kwargs.items()
            if key in parameters and parameters[key].kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
            )
        }
    return fn(*args, **kwargs)


def _needs_type_samples(error_message: str) -> bool:
    message = error_message.lower()
    return any(marker in message for marker in (
        "conversion error", "type mismatch", "cannot compare values of type",
        "cannot mix values of type", "no function matches the given name and argument types",
    ))


@dataclass
class PipelineAttempt:
    """Observed outcome of one real generation/repair attempt; no gold feedback."""

    attempt: int
    sql: str = ""
    success: bool = False
    df: Optional[pd.DataFrame] = None
    error_type: str = ""
    error_message: str = ""
    provider_failed: bool = False
    provider_error_code: Optional[int] = None
    quota_kind: Optional[str] = None


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
    attempts: List[PipelineAttempt] = field(default_factory=list)


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
            semantic_validator if semantic_validator is not None else SemanticValidator(db_conn=self.db_conn)
        )

    def check_ambiguity(
        self,
        user_question: str,
        schema_info: Optional[str] = None,
        rag_context: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> AmbiguityResult:
        """Classify user intent ambiguity and generate clarification options.

        Args:
            user_question: Natural language question submitted by user.
            schema_info: Database DDL or table schema summary (defaults to DB schema).
            rag_context: Business metric and filter rules retrieved via RAG.
            provider: LLM provider override ('groq', 'gemini', 'openai').

        Returns:
            AmbiguityResult with 3-tier classification, reasoning, and structured options.
        """
        if schema_info is None:
            schema_info = get_schema_info(conn=self.db_conn, as_string=True)

        if rag_context is None:
            rag_context = ""
            if self.retriever is not None:
                try:
                    rag_context, _ = self.retriever.retrieve(user_question)
                except Exception as exc:
                    logger.warning("Ambiguity classification has no retrieval context: %s", exc)

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
        """Run one bounded trajectory, retaining actual per-attempt outcomes.

        Only compiler, safety, and semantic feedback reaches repairs. Evaluation
        observes these outcomes after completion and never supplies gold SQL/results.
        """
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 1:
            raise ValueError("max_retries must be a positive integer.")
        start_time = time.perf_counter()
        result = PipelineResult(success=False, total_attempts=0, clarified_context=clarified_context)
        if not user_question or not user_question.strip():
            result.error_message = "Question is empty."
            return result

        schema_str = get_schema_info(conn=self.db_conn, as_string=True)
        if self.retriever is not None:
            try:
                result.rag_context, result.rag_metadata = self.retriever.retrieve(
                    user_question, top_k_metrics=2, top_k_filters=2, top_k_tables=3,
                )
            except Exception as exc:
                logger.warning("SQL generation has no retrieval context: %s", exc)

        cleaned_input = user_question.strip()
        sql_keywords = ("SELECT ", "WITH ", "DROP ", "DELETE ", "UPDATE ", "INSERT ", "ALTER ", "CREATE ")
        is_direct_sql = cleaned_input.upper().startswith(sql_keywords)
        seen_queries: Set[str] = set()

        for attempt in range(1, max_retries + 1):
            result.total_attempts = attempt
            outcome = PipelineAttempt(attempt=attempt)
            result.attempts.append(outcome)

            def record_failure(error_type: str, message: str, **detail) -> None:
                outcome.error_type = error_type
                outcome.error_message = message
                result.error_message = message
                result.attempt_history.append({
                    "attempt": attempt, "sql": outcome.sql,
                    "error": message, "error_type": error_type, **detail,
                })

            try:
                if attempt == 1:
                    current_sql = cleaned_input if is_direct_sql else _call_model(
                        self.generator_fn, user_question, schema_str,
                        rag_context=result.rag_context, clarified_context=clarified_context,
                        provider=provider,
                    )
                else:
                    last_record = result.attempt_history[-1]
                    table_samples = {}
                    if last_record["error_type"] == "DUCKDB_RUNTIME" and _needs_type_samples(last_record["error"]):
                        try:
                            table_samples = get_query_sample_rows(last_record["sql"], self.db_conn)
                        except Exception as exc:
                            logger.warning("Repair will proceed without table samples: %s", exc)
                    current_sql = _call_model(
                        self.repair_fn, user_question=user_question, schema_info=schema_str,
                        rag_context=result.rag_context, failed_sql=last_record["sql"],
                        error_message=last_record["error"], attempt_history=result.attempt_history,
                        clarified_context=clarified_context, provider=provider, table_samples=table_samples,
                    )
                if not isinstance(current_sql, str):
                    raise TypeError("The model must return a SQL string.")
            except Exception as exc:
                outcome.provider_failed = isinstance(exc, ProviderRequestError)
                outcome.provider_error_code = getattr(exc, "code", None)
                outcome.quota_kind = getattr(exc, "quota_kind", None)
                record_failure("GENERATION_FAILURE", str(exc))
                result.error_message = f"Model generation error: {exc}"
                break

            outcome.sql = current_sql
            result.final_sql = current_sql
            # Preserve whitespace inside literals: those queries can have different meanings.
            normalized_sql = current_sql.strip()
            if normalized_sql in seen_queries:
                record_failure("CYCLE_STAGNATION", "Stagnation loop detected: the model repeated an identical SQL query that previously failed.")
                break
            seen_queries.add(normalized_sql)

            is_valid, validation_msg = validate_sql(current_sql, allowed_tables=TPCH_TABLES)
            if not is_valid:
                result.is_security_violation = is_security_violation(validation_msg)
                record_failure("SECURITY_VIOLATION" if result.is_security_violation else "AST_SYNTAX", validation_msg)
                if result.is_security_violation:
                    break
                continue

            df, query_err = execute_query(current_sql, conn=self.db_conn)
            if query_err is not None or df is None:
                clean_err = sanitize_error_message(query_err or "Unknown database error")
                result.is_security_violation = any(marker in clean_err.lower() for marker in (
                    "permission error", "disabled by configuration",
                ))
                record_failure("SECURITY_VIOLATION" if result.is_security_violation else "DUCKDB_RUNTIME", clean_err)
                if result.is_security_violation:
                    break
                continue

            outcome.df = df
            sem_res = _call_model(
                self.semantic_validator.validate_output,
                user_question=user_question, sql_query=current_sql, df=df,
                schema_info=schema_str, rag_context=result.rag_context,
                rag_metadata=result.rag_metadata, clarified_context=clarified_context,
                provider=provider,
            )
            result.semantic_validation = sem_res
            if not sem_res.is_valid:
                record_failure("SEMANTIC_VALIDATION", f"{sem_res.reasoning} Suggested fix: {sem_res.suggested_fix}", issue_type=sem_res.issue_type)
                continue

            outcome.success = True
            result.success = True
            result.df = df
            result.error_message = None
            break
        else:
            result.error_message = f"Failed to auto-repair query after {max_retries} attempts. Last error: {result.error_message}"

        result.execution_time_ms = (time.perf_counter() - start_time) * 1000
        result.was_semantically_repaired = result.success and any(
            record["error_type"] == "SEMANTIC_VALIDATION" for record in result.attempt_history
        )
        return result
