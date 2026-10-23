"""Silent Execution & Semantic Validation Engine for DataSense AI.

Provides a fast-path 2-stage validation system:
- Stage 1: Deterministic fast heuristics (< 1ms) detecting casing mismatches,
           Cartesian explosions, catalog metric omissions, and aggregate mismatches.
- Stage 2: Selective lightweight LLM-as-a-Judge verifier on suspected anomalies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set
import pandas as pd

from src.config import settings


@dataclass
class SemanticValidationResult:
    """Outcome and diagnostics from semantic validation."""

    is_valid: bool = True
    issue_type: str = "NONE"  # 'NONE' | 'CASING_MISMATCH' | 'CARTESIAN_EXPLOSION' | 'CATALOG_METRIC_OMISSION' | 'AGGREGATE_MISMATCH' | 'SEMANTIC_JUDGE_REJECT'
    reasoning: str = ""
    suggested_fix: str = ""


# Known TPC-H constant values that require exact uppercase comparison
TPCH_UPPERCASE_CONSTANTS: Dict[str, str] = {
    # Nations
    "algeria": "ALGERIA",
    "argentina": "ARGENTINA",
    "brazil": "BRAZIL",
    "canada": "CANADA",
    "egypt": "EGYPT",
    "ethiopia": "ETHIOPIA",
    "france": "FRANCE",
    "germany": "GERMANY",
    "india": "INDIA",
    "indonesia": "INDONESIA",
    "iran": "IRAN",
    "iraq": "IRAQ",
    "japan": "JAPAN",
    "jordan": "JORDAN",
    "kenya": "KENYA",
    "morocco": "MOROCCO",
    "mozambique": "MOZAMBIQUE",
    "peru": "PERU",
    "china": "CHINA",
    "romania": "ROMANIA",
    "saudi arabia": "SAUDI ARABIA",
    "vietnam": "VIETNAM",
    "russia": "RUSSIA",
    "united kingdom": "UNITED KINGDOM",
    "united states": "UNITED STATES",
    # Regions
    "africa": "AFRICA",
    "america": "AMERICA",
    "asia": "ASIA",
    "europe": "EUROPE",
    "middle east": "MIDDLE EAST",
    # Order Priorities
    "urgent": "1-URGENT",
    "high": "2-HIGH",
    "medium": "3-MEDIUM",
    "not specified": "4-NOT SPECIFIED",
    "low": "5-LOW",
    "1-urgent": "1-URGENT",
    "2-high": "2-HIGH",
    "3-medium": "3-MEDIUM",
    "4-not specified": "4-NOT SPECIFIED",
    "5-low": "5-LOW",
    # Market Segments
    "automobile": "AUTOMOBILE",
    "building": "BUILDING",
    "furniture": "FURNITURE",
    "machinery": "MACHINERY",
    "household": "HOUSEHOLD",
    # Order Statuses
    "o": "O",
    "p": "P",
    "f": "F",
}


def extract_string_literals(sql: str) -> List[str]:
    """Extract string literals enclosed in single quotes from SQL."""
    if not sql:
        return []
    # Match strings inside '...'
    return re.findall(r"'([^']*)'", sql)


class SemanticValidator:
    """Validates executed SQL query outputs against semantic and data expectations."""

    def __init__(
        self,
        judge_fn: Optional[Callable[..., str]] = None,
        enable_llm_judge: bool = False,
    ):
        """Initialize validator with optional custom LLM judge function and judge toggle.

        Args:
            judge_fn: Optional custom callable to act as LLM judge.
            enable_llm_judge: Whether to run Stage 2 LLM judge on Stage 1 anomalies.
        """
        self.judge_fn = judge_fn
        self.enable_llm_judge = enable_llm_judge

    def validate_output(
        self,
        user_question: str,
        sql_query: str,
        df: pd.DataFrame,
        schema_info: str = "",
        rag_context: str = "",
        clarified_context: str = "",
        provider: Optional[str] = None,
    ) -> SemanticValidationResult:
        """Validate query output using 2-stage fast-path heuristics and selective LLM judge.

        Args:
            user_question: Natural language question asked by user.
            sql_query: Executed DuckDB SQL query.
            df: Resulting Pandas DataFrame from DuckDB.
            schema_info: Database schema/DDL.
            rag_context: Retrieved business metric formulas.
            clarified_context: Clarified user intent.
            provider: LLM provider override.

        Returns:
            SemanticValidationResult with validity flag, issue type, and actionable suggestions.
        """
        if df is None:
            return SemanticValidationResult(
                is_valid=False,
                issue_type="ZERO_ROWS",
                reasoning="Query returned no DataFrame (None).",
                suggested_fix="Verify query execution and schema table references.",
            )

        # -------------------------------------------------------------
        # STAGE 1: Fast Deterministic Heuristics (< 1ms)
        # -------------------------------------------------------------

        # 1. Zero-Row & Casing Check
        casing_result = self._check_zero_rows_and_casing(df, sql_query, user_question)
        if not casing_result.is_valid:
            return casing_result

        # 2. Cartesian Explosion / Unbounded Multi-Table Join Check
        cartesian_result = self._check_cartesian_explosion(df, sql_query)
        if not cartesian_result.is_valid:
            return cartesian_result

        # 3. Catalog Metric Omission Check (Phase 2 RAG synergy)
        metric_result = self._check_catalog_metric_omission(df, sql_query, rag_context)
        if not metric_result.is_valid:
            return metric_result

        # 4. Aggregation Mismatch Check
        agg_result = self._check_aggregate_mismatch(df, sql_query, user_question)
        if not agg_result.is_valid:
            return agg_result

        # -------------------------------------------------------------
        # STAGE 2: Fast-Path / Selective LLM Judge
        # If Stage 1 heuristics found zero anomalies, FAST PATH returns immediately!
        # -------------------------------------------------------------
        if not self.enable_llm_judge:
            return SemanticValidationResult(is_valid=True, issue_type="NONE")

        # Selective judge invocation
        return self._run_llm_judge(
            user_question=user_question,
            sql_query=sql_query,
            df=df,
            provider=provider,
        )

    def _check_zero_rows_and_casing(
        self,
        df: pd.DataFrame,
        sql_query: str,
        user_question: str,
    ) -> SemanticValidationResult:
        """Inspect zero-row results for case-sensitive string literal mismatches."""
        if len(df) > 0:
            return SemanticValidationResult(is_valid=True)

        literals = extract_string_literals(sql_query)
        for lit in literals:
            lower_lit = lit.strip().lower()
            if lower_lit in TPCH_UPPERCASE_CONSTANTS:
                correct_upper = TPCH_UPPERCASE_CONSTANTS[lower_lit]
                if lit != correct_upper:
                    return SemanticValidationResult(
                        is_valid=False,
                        issue_type="CASING_MISMATCH",
                        reasoning=(
                            f"Query executed successfully but returned 0 rows due to case-sensitive string "
                            f"literal '{lit}'. In DuckDB TPC-H tables, this value is stored in uppercase as '{correct_upper}'."
                        ),
                        suggested_fix=(
                            f"Change '{lit}' to '{correct_upper}' or use ILIKE / UPPER() for case-insensitive matching."
                        ),
                    )

        # Check if question was a negative existence check
        negative_check_keywords = ("are there any", "is there any", "check if any", "check if there")
        if any(kw in user_question.lower() for kw in negative_check_keywords):
            return SemanticValidationResult(is_valid=True)

        return SemanticValidationResult(is_valid=True)

    def _check_cartesian_explosion(
        self,
        df: pd.DataFrame,
        sql_query: str,
    ) -> SemanticValidationResult:
        """Detect multi-table queries that accidentally create Cartesian products."""
        clean_sql = sql_query.upper()

        # Check if multiple TPC-H tables are queried
        tpch_tables = ("CUSTOMER", "ORDERS", "LINEITEM", "PARTSUPP", "PART", "SUPPLIER", "NATION", "REGION")
        matched_tables = [t for t in tpch_tables if re.search(rf"\b{t}\b", clean_sql)]

        # Multi-table query with row explosion (> 1000 rows) and no aggregation or limit
        if len(matched_tables) >= 2:
            has_group = "GROUP BY" in clean_sql
            has_limit = "LIMIT " in clean_sql
            if len(df) > 1000 and not has_group and not has_limit:
                return SemanticValidationResult(
                    is_valid=False,
                    issue_type="CARTESIAN_EXPLOSION",
                    reasoning=(
                        f"Query produced {len(df):,} unaggregated rows across multiple tables ({', '.join(matched_tables)}), "
                        f"indicating a missing join predicate or unintended Cartesian product."
                    ),
                    suggested_fix=(
                        "Ensure explicit join keys are specified (e.g. orders.o_custkey = customer.c_custkey, "
                        "orders.o_orderkey = lineitem.l_orderkey). Add a LIMIT if an unaggregated sample was intended."
                    ),
                )

        return SemanticValidationResult(is_valid=True)

    def _check_catalog_metric_omission(
        self,
        df: pd.DataFrame,
        sql_query: str,
        rag_context: str,
    ) -> SemanticValidationResult:
        """Verify that specific formulas retrieved via RAG are present in the SQL."""
        if not rag_context or not rag_context.strip():
            return SemanticValidationResult(is_valid=True)

        norm_rag = re.sub(r"\s+", "", rag_context.lower())
        norm_sql = re.sub(r"\s+", "", sql_query.lower())

        # Check if net revenue formula was retrieved and expected
        if "net_revenue" in norm_rag and "1-l_discount" in norm_rag:
            # Check if SQL calculates lineitem price but omitted discount formula
            if "l_extendedprice" in norm_sql and "1-l_discount" not in norm_sql:
                return SemanticValidationResult(
                    is_valid=False,
                    issue_type="CATALOG_METRIC_OMISSION",
                    reasoning=(
                        "Query computed gross revenue (SUM(l_extendedprice)) instead of the retrieved "
                        "catalog metric formula for net revenue (SUM(l_extendedprice * (1 - l_discount)))."
                    ),
                    suggested_fix=(
                        "Replace SUM(l_extendedprice) with the exact net revenue formula: "
                        "SUM(l_extendedprice * (1 - l_discount))."
                    ),
                )

        return SemanticValidationResult(is_valid=True)

    def _check_aggregate_mismatch(
        self,
        df: pd.DataFrame,
        sql_query: str,
        user_question: str,
    ) -> SemanticValidationResult:
        """Detect scalar aggregation intent that produced a dump of unaggregated rows."""
        q_lower = user_question.lower()
        clean_sql = sql_query.upper()

        # Phrases indicating user wanted a single summary metric across the whole dataset
        scalar_summary_starters = (
            "what is the total",
            "what is the sum",
            "what is the average",
            "how much is the total",
            "what is the overall",
            "calculate the total",
        )

        is_scalar_intent = any(q_lower.startswith(prefix) for prefix in scalar_summary_starters)
        has_grouping_words = any(w in q_lower for w in (" per ", " by ", " each ", " for every ", "top ", "breakdown"))

        # If user wanted a scalar total, but query returned > 10 rows without GROUP BY or agg functions
        if is_scalar_intent and not has_grouping_words:
            has_agg_func = any(fn in clean_sql for fn in ("SUM(", "AVG(", "COUNT(", "MIN(", "MAX("))
            if not has_agg_func and len(df) > 10:
                return SemanticValidationResult(
                    is_valid=False,
                    issue_type="AGGREGATE_MISMATCH",
                    reasoning=(
                        f"User requested a scalar summary aggregation ('{user_question}'), but the query "
                        f"returned {len(df):,} unaggregated individual rows without an aggregate function."
                    ),
                    suggested_fix=(
                        "Wrap the target column in the appropriate aggregate function (e.g. SUM(...), AVG(...)) "
                        "to return a single scalar summary value."
                    ),
                )

        return SemanticValidationResult(is_valid=True)

    def _run_llm_judge(
        self,
        user_question: str,
        sql_query: str,
        df: pd.DataFrame,
        provider: Optional[str] = None,
    ) -> SemanticValidationResult:
        """Execute Stage 2 lightweight LLM-as-a-Judge verifier."""
        preview_rows = df.head(5).to_dict(orient="records")
        preview_json = json.dumps(preview_rows, default=str)

        judge_prompt = f"""You are a Senior SQL & Analytical Verifier.
Evaluate whether the following executed SQL query and its preview result logically answer the user's question.

User Question: {user_question}
SQL Query: {sql_query}
Row Count: {len(df)}
Preview Rows: {preview_json}

Output strictly a JSON object:
{{
  "is_valid": true | false,
  "issue_type": "NONE" | "SEMANTIC_JUDGE_REJECT",
  "reasoning": "Explanation of your verdict",
  "suggested_fix": "Actionable fix if invalid, or empty string"
}}"""

        try:
            if self.judge_fn is not None:
                raw_response = self.judge_fn(judge_prompt)
            else:
                active_provider = provider.lower() if provider else settings.get_effective_provider()
                if active_provider == "gemini":
                    if not settings.gemini_api_key:
                        return SemanticValidationResult(is_valid=True)
                    from google import genai
                    from google.genai import types

                    client = genai.Client(api_key=settings.gemini_api_key)
                    import time
                    last_err = None
                    raw_response = ""
                    for attempt in range(3):
                        try:
                            resp = client.models.generate_content(
                                model=settings.gemini_model,
                                contents=[judge_prompt],
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    temperature=0.0,
                                ),
                            )
                            raw_response = resp.text or ""
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                            if ("503" in str(e) or "429" in str(e) or "UNAVAILABLE" in str(e)) and attempt < 2:
                                time.sleep(1.5 * (attempt + 1))
                                continue
                            raise e
                    if last_err:
                        raise last_err
                elif active_provider == "openai":
                    if not settings.openai_api_key:
                        return SemanticValidationResult(is_valid=True)
                    from openai import OpenAI

                    client = OpenAI(api_key=settings.openai_api_key)
                    resp = client.chat.completions.create(
                        model=settings.openai_model,
                        messages=[{"role": "user", "content": judge_prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.0,
                    )
                    raw_response = resp.choices[0].message.content or ""
                else:
                    return SemanticValidationResult(is_valid=True)

            # Parse JSON
            clean_text = raw_response.strip()
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text, re.IGNORECASE)
            if match:
                clean_text = match.group(1).strip()
            data = json.loads(clean_text)

            is_valid = bool(data.get("is_valid", True))
            issue_type = str(data.get("issue_type", "NONE" if is_valid else "SEMANTIC_JUDGE_REJECT"))
            reasoning = str(data.get("reasoning", ""))
            suggested_fix = str(data.get("suggested_fix", ""))

            return SemanticValidationResult(
                is_valid=is_valid,
                issue_type=issue_type,
                reasoning=reasoning,
                suggested_fix=suggested_fix,
            )

        except Exception as exc:
            # Fail-safe fallback: verifier exceptions never block query execution
            return SemanticValidationResult(
                is_valid=True,
                issue_type="NONE",
                reasoning=f"Judge fallback on exception: {exc}",
            )
