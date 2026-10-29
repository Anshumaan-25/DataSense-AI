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
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Set
import duckdb
import pandas as pd
import sqlglot
import yaml
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, build_scope, traverse_scope

from src.config import settings
from src.llm import generate_gemini_text
from src.groq_client import generate_groq_text
from src.database import DistinctValueProfile, get_distinct_value_profile


@dataclass
class SemanticValidationResult:
    """Outcome and diagnostics from semantic validation."""

    is_valid: bool = True
    issue_type: str = "NONE"  # 'NONE' | 'CASING_MISMATCH' | 'CARTESIAN_EXPLOSION' | 'CATALOG_METRIC_OMISSION' | 'AGGREGATE_MISMATCH' | 'SEMANTIC_JUDGE_REJECT'
    reasoning: str = ""
    suggested_fix: str = ""
    judge_status: str = "NOT_RUN"


def _canonical_formula(node):
    """Compare AST structure without table aliases, parentheses, or operand order."""
    if isinstance(node, (exp.Paren, exp.Alias)):
        return _canonical_formula(node.this)
    if isinstance(node, exp.Column):
        return ("column", node.name.lower())
    if isinstance(node, exp.Literal):
        return ("string", node.this) if node.is_string else ("number", Decimal(node.this))
    if isinstance(node, (exp.Add, exp.Mul)):
        operands = []

        def flatten(part):
            while isinstance(part, exp.Paren):
                part = part.this
            if isinstance(part, type(node)):
                flatten(part.this)
                flatten(part.expression)
            else:
                operands.append(_canonical_formula(part))

        flatten(node)
        return (node.key, tuple(sorted(operands, key=repr)))
    if isinstance(node, exp.Expression):
        return (node.key, tuple(
            (key, _canonical_formula(value)) for key, value in sorted(node.args.items())
            if value is not None and key != "alias"
        ))
    if isinstance(node, list):
        return tuple(_canonical_formula(value) for value in node)
    return node


def _output_formulas(tree):
    """Resolve only output CTE/derived columns, preserving surrounding arithmetic."""
    root = build_scope(tree)

    def expand(scope, expression, seen):
        def substitute(node):
            if not isinstance(node, exp.Column):
                return node
            candidates = []
            for alias, (_, source) in scope.selected_sources.items():
                if not isinstance(source, Scope) or (node.table and node.table.lower() != alias.lower()):
                    continue
                for index, selected in enumerate(source.expression.selects):
                    name = source.outer_columns[index] if index < len(source.outer_columns) else selected.alias_or_name
                    if name.lower() == node.name.lower():
                        candidates.append((source, selected))
            if len(candidates) != 1:
                return node
            source, selected = candidates[0]
            key = (id(source), id(selected))
            return node if key in seen else expand(source, selected.unalias(), seen | {key})

        return expression.copy().transform(substitute, copy=False)

    def projections(scope):
        branches = getattr(scope, "set_operation_scopes", getattr(scope, "union_scopes", []))
        if branches:
            return [value for branch in branches for value in projections(branch)]
        output = []
        for selected in scope.expression.selects:
            if isinstance(selected, exp.Star) or (isinstance(selected, exp.Column) and selected.is_star):
                for alias, (_, source) in scope.selected_sources.items():
                    if isinstance(source, Scope) and (not isinstance(selected, exp.Column) or not selected.table or selected.table.lower() == alias.lower()):
                        output.extend(projections(source))
            else:
                expression = expand(scope, selected.unalias(), set())
                while isinstance(expression, (exp.Paren, exp.Round, exp.Cast)):
                    expression = expression.this
                output.append(expression)
        return output

    return projections(root) if root is not None else []


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
        db_conn: Optional[duckdb.DuckDBPyConnection] = None,
        distinct_value_profile: Optional[DistinctValueProfile] = None,
    ):
        """Initialize validator with optional custom LLM judge function and judge toggle.

        Args:
            judge_fn: Optional custom callable to act as LLM judge.
            enable_llm_judge: Whether to run Stage 2 LLM judge on Stage 1 anomalies.
            db_conn: Database supplying a cached, column-specific casing profile.
            distinct_value_profile: Explicit profile for standalone/offline validation.
        """
        self.judge_fn = judge_fn
        self.enable_llm_judge = enable_llm_judge
        self.db_conn = db_conn
        self.distinct_value_profile = distinct_value_profile
        catalog_path = settings.data_dir / "metrics.yaml"
        self.metrics = yaml.safe_load(catalog_path.read_text(encoding="utf-8")).get("metrics", [])

    def validate_output(
        self,
        user_question: str,
        sql_query: str,
        df: pd.DataFrame,
        schema_info: str = "",
        rag_context: str = "",
        clarified_context: str = "",
        provider: Optional[str] = None,
        rag_metadata: Optional[List[Dict[str, Any]]] = None,
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
        metric_result = self._check_catalog_metric_omission(
            df, sql_query, rag_context, user_question, clarified_context, rag_metadata,
        )
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
            return SemanticValidationResult(is_valid=True, issue_type="NONE", judge_status="DISABLED")

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
        # Scalar aggregates over an empty filtered input can return one NULL row.
        if not df.empty and not (len(df) == 1 and df.isna().all().all()):
            return SemanticValidationResult(is_valid=True)

        profile = self.distinct_value_profile
        if profile is None:
            profile = get_distinct_value_profile(self.db_conn) if self.db_conn is not None else {}
        if not profile:
            return SemanticValidationResult(is_valid=True)

        try:
            tree = sqlglot.parse_one(sql_query, read="duckdb")
        except sqlglot.errors.ParseError:
            return SemanticValidationResult(is_valid=True)

        for scope in traverse_scope(tree):
            for predicate in scope.expression.find_all(exp.EQ, exp.In):
                if predicate.find_ancestor(exp.Select) is not scope.expression:
                    continue
                if predicate.find_ancestor(exp.Not) is not None:
                    continue
                if isinstance(predicate, exp.EQ):
                    pairs = [(predicate.this, predicate.expression), (predicate.expression, predicate.this)]
                else:
                    pairs = [(predicate.this, value) for value in predicate.expressions]
                for column, literal in pairs:
                    if not isinstance(column, exp.Column) or not isinstance(literal, exp.Literal) or not literal.is_string:
                        continue
                    sources = [
                        source for alias, source in scope.sources.items()
                        if isinstance(source, exp.Table)
                        and (not column.table or alias.lower() == column.table.lower())
                        and not source.catalog and source.db.lower() in ("", "main")
                    ]
                    candidates = [
                        profile[source.name.lower()][column.name.lower()]
                        for source in sources
                        if column.name.lower() in profile.get(source.name.lower(), {})
                    ]
                    # Do not guess across ambiguous columns or case-colliding values.
                    if len(candidates) != 1 or literal.this in candidates[0]:
                        continue
                    matches = [value for value in candidates[0] if value.casefold() == literal.this.casefold()]
                    if len(matches) != 1:
                        continue
                    correct_value = matches[0]
                    return SemanticValidationResult(
                        is_valid=False,
                        issue_type="CASING_MISMATCH",
                        reasoning=(
                            f"Empty result may be caused by case-sensitive literal {literal.this!r} "
                            f"for {column.sql()}. The database profile contains {correct_value!r}."
                        ),
                        suggested_fix=f"Use the stored value {correct_value!r} for {column.sql()}, or use ILIKE / UPPER().",
                    )

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
        user_question: str = "",
        clarified_context: str = "",
        rag_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> SemanticValidationResult:
        """Require requested catalog formulas in output, rather than every RAG candidate.

        Structural equivalence supports aliases, redundant parentheses, numeric
        literal formatting, and associative/commutative addition/multiplication.
        It is not a general SQL algebra or join-correctness proof.
        """
        if (not rag_context and not rag_metadata) or user_question.lstrip().upper().startswith(("SELECT ", "WITH ")):
            return SemanticValidationResult(is_valid=True)
        catalog = {metric["id"]: metric for metric in self.metrics}
        if rag_metadata:
            available = [
                {**catalog.get(metric.get("id"), {}), **metric}
                for metric in rag_metadata if metric.get("kind") == "metric" and metric.get("formula")
            ]
        else:
            available = [metric for metric in self.metrics if re.search(
                rf"\b{re.escape(metric['id'])}\b", rag_context, re.IGNORECASE,
            )]

        def requested(text):
            normalized = text.lower().replace("_", " ")
            matches = []
            for metric in available:
                aliases = metric.get("aliases", [])
                if isinstance(aliases, str):
                    aliases = aliases.split(", ")
                terms = [metric.get("name", ""), metric["id"].replace("_", " "), *aliases]
                for term in terms:
                    if term:
                        for match in re.finditer(rf"(?<!\w){re.escape(term.lower())}(?!\w)", normalized):
                            matches.append((match.start(), match.end(), metric))
            # A more specific phrase ('gross revenue') wins over its contained
            # generic alias ('revenue'), while separate requested metrics remain.
            return {
                metric["id"]: metric for start, end, metric in matches
                if not any(other_start <= start and end <= other_end and other_end - other_start > end - start
                           for other_start, other_end, _ in matches)
            }

        selected = requested(clarified_context) or requested(user_question)
        if not selected:
            return SemanticValidationResult(is_valid=True)
        try:
            tree = sqlglot.parse_one(sql_query, read="duckdb")
            projected = {_canonical_formula(node) for node in _output_formulas(tree)}
            for metric in selected.values():
                expected = _canonical_formula(sqlglot.parse_one(metric["formula"], read="duckdb"))
                if expected not in projected:
                    explanation = f"Output omitted the requested catalog formula for {metric.get('name', metric['id'])}."
                    if metric["id"] == "net_revenue":
                        explanation += " Computing gross revenue does not satisfy net revenue."
                    return SemanticValidationResult(
                        is_valid=False, issue_type="CATALOG_METRIC_OMISSION",
                        reasoning=explanation,
                        suggested_fix=f"Use the exact catalog formula in the output: {metric['formula']}",
                    )
        except sqlglot.errors.ParseError:
            # Syntax validation is handled earlier; an unparseable catalog formula
            # cannot establish a semantic violation.
            return SemanticValidationResult(is_valid=True, reasoning="Catalog formula could not be verified.")
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
                        return SemanticValidationResult(is_valid=True, judge_status="UNAVAILABLE")
                    raw_response = generate_gemini_text(
                        judge_prompt, model_name=settings.gemini_model,
                        api_key=settings.gemini_api_key, response_mime_type="application/json",
                        config=settings,
                    )
                elif active_provider == "groq":
                    if not settings.groq_api_key:
                        return SemanticValidationResult(is_valid=True, judge_status="UNAVAILABLE")
                    raw_response = generate_groq_text(judge_prompt, model_name=settings.groq_model,
                        api_key=settings.groq_api_key, response_mime_type="application/json", config=settings)
                elif active_provider == "openai":
                    if not settings.openai_api_key:
                        return SemanticValidationResult(is_valid=True, judge_status="UNAVAILABLE")
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
                    return SemanticValidationResult(is_valid=True, judge_status="UNAVAILABLE")

            # Parse JSON
            clean_text = raw_response.strip()
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text, re.IGNORECASE)
            if match:
                clean_text = match.group(1).strip()
            data = json.loads(clean_text)

            if not isinstance(data, dict) or not isinstance(data.get("is_valid"), bool):
                raise ValueError("Semantic judge must return a JSON object with boolean is_valid.")
            is_valid = data["is_valid"]
            issue_type = str(data.get("issue_type", "NONE" if is_valid else "SEMANTIC_JUDGE_REJECT"))
            reasoning = str(data.get("reasoning", ""))
            suggested_fix = str(data.get("suggested_fix", ""))

            return SemanticValidationResult(
                is_valid=is_valid,
                issue_type=issue_type,
                reasoning=reasoning,
                suggested_fix=suggested_fix,
                judge_status="PASSED" if is_valid else "REJECTED",
            )

        except Exception as exc:
            # Fail-safe fallback: verifier exceptions never block query execution
            return SemanticValidationResult(
                is_valid=True,
                issue_type="NONE",
                reasoning=f"Judge fallback on exception: {exc}",
                judge_status="ERROR",
            )
