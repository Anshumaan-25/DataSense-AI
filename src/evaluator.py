"""Evaluation & Benchmarking Engine for DataSense AI — TPC-H Execution Accuracy (EX).

Architecture:
- DataFrameNormalizer:         Deterministic normalization for order-agnostic comparison.
- ExecutionAccuracyComparator: Shape + value comparison of two normalized DataFrames.
- BenchmarkRunner:             Orchestrates the full benchmark against SelfCorrectingPipeline.
- BenchmarkReport:             Structured telemetry with per-task and aggregate metrics.

Design Decisions:
- Ground truth SQL is EXECUTED LIVE against the same DuckDB instance at benchmark time —
  never stored as hardcoded expected values. This ensures scale-factor-agnostic EX scores.
- Pass@1 and Pass@N observe the same pipeline trajectory. Gold output never drives repair.
- Ambiguous queries (expected_ambiguity = 'HARD_AMBIGUITY') are EXCLUDED from EX measurement
  and tracked separately via Ambiguity Classifier Accuracy to avoid category errors.
- SOFT_DEFAULT queries ARE included in EX with classifier defaults preserved.
- Comparison policy and first-attempt SQL are exported for auditability.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.ambiguity import AmbiguityResult
from src.config import settings
from src.database import execute_query, get_connection, get_schema_info
from src.pipeline import SelfCorrectingPipeline
from src.retriever import SchemaMetricRetriever
from src.validator import validate_sql


# ---------------------------------------------------------------------------
# DataFrameNormalizer
# ---------------------------------------------------------------------------

class DataFrameNormalizer:
    """Legacy display normalization; EX uses original values via the comparator.

    Normalization steps (applied in sequence):
    1. Column names → lowercase, stripped of whitespace.
    2. Float64 columns → rounded to 2 decimal places.
    3. Object (string) columns → str.lower(), NaN preserved as NaN.
    4. Sort all rows by all columns (nullable-safe via string coercion on sort key).
    5. Reset index.
    """

    FLOAT_ROUND_DECIMALS = 2

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a canonical, order-agnostic version of df.

        Args:
            df: Input DataFrame to normalize.

        Returns:
            Normalized DataFrame with lowercase column names, rounded floats,
            lowercased strings, sorted rows, and reset integer index.

        Raises:
            TypeError: If df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        if df.empty:
            # Return empty frame with normalized column names only
            normalized_cols = [c.strip().lower() for c in df.columns]
            return pd.DataFrame(columns=normalized_cols)

        result = df.copy()

        # Step 1: Normalize column names
        result.columns = [str(c).strip().lower() for c in result.columns]

        # Step 2: Round float columns
        for col in result.columns:
            if pd.api.types.is_float_dtype(result[col]):
                result[col] = result[col].round(self.FLOAT_ROUND_DECIMALS)

        # Step 3: Lowercase string columns (object or StringDtype in pandas 2.x), NaN-safe
        for col in result.columns:
            if pd.api.types.is_string_dtype(result[col]) and not pd.api.types.is_bool_dtype(result[col]):
                result[col] = result[col].apply(
                    lambda x: x.lower() if isinstance(x, str) else x
                )

        # Step 4: Sort rows — use string coercion on sort key to avoid TypeError
        # on mixed-type or nullable columns
        try:
            sort_cols = list(result.columns)
            result = result.sort_values(
                by=sort_cols,
                key=lambda s: s.astype(str),
                ignore_index=True,
            )
        except Exception:
            # Last-resort fallback: convert entire frame to string for sorting
            str_proxy = result.astype(str)
            sort_order = str_proxy.sort_values(by=list(str_proxy.columns)).index
            result = result.loc[sort_order].reset_index(drop=True)

        # Step 5: Reset index
        result = result.reset_index(drop=True)
        return result


# ---------------------------------------------------------------------------
# ExecutionAccuracyComparator
# ---------------------------------------------------------------------------

class ExecutionAccuracyComparator:
    """Value-based EX with a global column mapping and duplicate-preserving rows.

    Default policy: aliases and row order are ignored; strings are case-sensitive;
    numbers use absolute tolerance 0.01 with zero relative tolerance. Raw values are
    compared without rounding, float coercion, or string coercion of cell types.
    """

    def __init__(self, atol: float = 0.01, rtol: float = 0.0,
                 ignore_column_names: bool = True, case_sensitive_strings: bool = True):
        if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)):
            raise ValueError("Numeric tolerances must be finite and non-negative.")
        self.atol = Decimal(str(atol))
        self.rtol = Decimal(str(rtol))
        self.ignore_column_names = ignore_column_names
        self.case_sensitive_strings = case_sensitive_strings

    def policy(self) -> Dict[str, Any]:
        return {
            "absolute_tolerance": str(self.atol), "relative_tolerance": str(self.rtol),
            "ignore_column_names": self.ignore_column_names,
            "case_sensitive_strings": self.case_sensitive_strings,
            "row_order": "unordered unless task.order_sensitive is true",
            "duplicates": "preserved", "column_mapping": "one global permutation",
        }

    def _key(self, value):
        if isinstance(value, (list, tuple, dict, set)):
            raise TypeError("EX comparison supports scalar cells.")
        if pd.isna(value):
            return ("null",)
        if isinstance(value, bool):
            return ("bool", value)
        if isinstance(value, (Real, Decimal)):
            return ("number", Decimal(str(value)))
        if isinstance(value, str):
            return ("string", value if self.case_sensitive_strings else value.casefold())
        return ("scalar", value)

    def _cells_equal(self, left, right) -> bool:
        a, b = self._key(left), self._key(right)
        if a == b:
            return True
        if a[0] != "number" or b[0] != "number":
            return False
        if not a[1].is_finite() or not b[1].is_finite():
            return False
        return abs(a[1] - b[1]) <= self.atol + self.rtol * max(abs(a[1]), abs(b[1]))

    def _rows_equal(self, left, right, ordered=False) -> bool:
        if len(left) != len(right):
            return False
        if ordered:
            return all(all(self._cells_equal(a, b) for a, b in zip(x, y)) for x, y in zip(left, right))
        left_keys = [tuple(self._key(value) for value in row) for row in left]
        right_keys = [tuple(self._key(value) for value in row) for row in right]
        if Counter(left_keys) == Counter(right_keys):
            return True
        groups_left, groups_right = defaultdict(list), defaultdict(list)
        for rows, keys, groups in ((left, left_keys, groups_left), (right, right_keys, groups_right)):
            for row, keys_for_row in zip(rows, keys):
                group = tuple(("number",) if key[0] == "number" else key for key in keys_for_row)
                groups[group].append(row)
        if groups_left.keys() != groups_right.keys():
            return False
        for group, generated in groups_left.items():
            truth = groups_right[group]
            if len(generated) != len(truth):
                return False
            adjacency = [
                [j for j, target in enumerate(truth)
                 if all(self._cells_equal(a, b) for a, b in zip(row, target))]
                for row in generated
            ]
            # Augmenting paths, rather than greedy pairing, preserve a valid match
            # when multiple numeric values lie within the same tolerance interval.
            matched_left, matched_right = {}, {}
            for start in range(len(generated)):
                queue, parent = deque([start]), {}
                found = None
                while queue and found is None:
                    index = queue.popleft()
                    for candidate in adjacency[index]:
                        if candidate in parent:
                            continue
                        parent[candidate] = index
                        if candidate not in matched_right:
                            found = candidate
                            break
                        queue.append(matched_right[candidate])
                if found is None:
                    return False
                while found is not None:
                    index = parent[found]
                    previous = matched_left.get(index)
                    matched_left[index] = found
                    matched_right[found] = index
                    found = previous
        return True

    def compare_outputs(self, df_generated: pd.DataFrame, df_truth: pd.DataFrame,
                        order_sensitive: bool = False) -> bool:
        if df_generated is None or df_truth is None:
            return False
        if not isinstance(df_generated, pd.DataFrame) or not isinstance(df_truth, pd.DataFrame):
            raise TypeError("EX comparison requires DataFrames.")
        if df_generated.shape != df_truth.shape:
            return False
        left = list(df_generated.itertuples(index=False, name=None))
        right = list(df_truth.itertuples(index=False, name=None))
        names_left = [str(name).strip().lower() for name in df_generated.columns]
        names_right = [str(name).strip().lower() for name in df_truth.columns]
        candidates = {}
        for target in range(df_truth.shape[1]):
            candidates[target] = [
                index for index in range(df_generated.shape[1])
                if (self.ignore_column_names or names_left[index] == names_right[target])
                and self._rows_equal([(row[index],) for row in left], [(row[target],) for row in right], order_sensitive)
            ]
            candidates[target].sort(key=lambda index: names_left[index] != names_right[target])
            if not candidates[target]:
                return False
        targets = sorted(candidates, key=lambda target: len(candidates[target]))

        def align(mapping, used):
            if len(mapping) == len(targets):
                return True
            target = targets[len(mapping)]
            tried = set()
            for index in candidates[target]:
                if index in used:
                    continue
                signature = tuple(self._key(row[index]) for row in left)
                if signature in tried:
                    continue
                tried.add(signature)
                proposal = mapping + [index]
                partial_left = [tuple(row[col] for col in proposal) for row in left]
                partial_right = [tuple(row[col] for col in targets[:len(proposal)]) for row in right]
                if self._rows_equal(partial_left, partial_right, order_sensitive) and align(proposal, used | {index}):
                    return True
            return False

        return align([], set())


# ---------------------------------------------------------------------------
# Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Per-query benchmark evaluation record."""

    task_id: str
    category: str
    question: str
    expected_ambiguity: str

    # EX evaluation fields (populated only for non-HARD_AMBIGUITY tasks)
    ex_eligible: bool = True          # False for HARD_AMBIGUITY tasks
    pass_at_1: Optional[bool] = None  # None if not eligible
    pass_at_n: Optional[bool] = None  # None if not eligible
    total_attempts: int = 0
    was_semantically_repaired: bool = False
    latency_ms: float = 0.0

    # Ambiguity classifier accuracy (populated for all tasks)
    ambiguity_predicted: Optional[str] = None
    ambiguity_correct: Optional[bool] = None

    # Diagnostic detail
    generated_sql: str = ""
    ground_truth_sql: str = ""
    error_message: str = ""
    pipeline_success: bool = False
    tags: List[str] = field(default_factory=list)
    first_sql: str = ""
    first_error_type: str = ""
    clarified_context: str = ""
    needs_clarification: bool = False
    classification_succeeded: bool = True
    evaluation_error: bool = False
    order_sensitive: bool = False
    provider_failed: bool = False
    provider_error_code: Optional[int] = None
    quota_kind: Optional[str] = None


@dataclass
class BenchmarkReport:
    """Aggregate benchmark results across all evaluated tasks."""

    total_tasks: int = 0
    ex_eligible_tasks: int = 0       # Excludes HARD_AMBIGUITY
    pass_at_1_count: int = 0
    pass_at_n_count: int = 0
    semantic_repair_count: int = 0
    total_latency_ms: float = 0.0

    # Ambiguity classification accuracy
    ambiguity_tasks: int = 0
    ambiguity_correct_count: int = 0

    task_results: List[TaskResult] = field(default_factory=list)

    # --- Derived metrics (computed on finalize()) ---
    pass_at_1_accuracy: float = 0.0
    pass_at_n_accuracy: float = 0.0
    self_correction_delta: float = 0.0  # pass@N - pass@1 (Δ EX gain from self-repair)
    avg_latency_ms: float = 0.0
    ambiguity_accuracy: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_error_count: int = 0
    expected_ex_eligible_tasks: Optional[int] = None

    @property
    def aborted(self) -> bool:
        return bool(self.metadata.get("abort_reason"))

    def finalize(self) -> "BenchmarkReport":
        """Compute aggregate accuracy metrics from task-level results. Returns self."""
        eligible = [t for t in self.task_results if t.ex_eligible]
        self.ex_eligible_tasks = self.expected_ex_eligible_tasks if self.expected_ex_eligible_tasks is not None else len(eligible)

        self.pass_at_1_count = sum(1 for t in eligible if t.pass_at_1 is True)
        self.pass_at_n_count = sum(1 for t in eligible if t.pass_at_n is True)
        self.semantic_repair_count = sum(1 for t in eligible if t.was_semantically_repaired)
        self.total_latency_ms = sum(t.latency_ms for t in self.task_results)

        amb_tasks = self.task_results
        self.ambiguity_tasks = len(amb_tasks)
        self.ambiguity_correct_count = sum(1 for t in amb_tasks if t.ambiguity_correct is True)
        self.evaluation_error_count = sum(t.evaluation_error for t in self.task_results)

        n = self.ex_eligible_tasks
        self.pass_at_1_accuracy = (self.pass_at_1_count / n * 100) if n > 0 else 0.0
        self.pass_at_n_accuracy = (self.pass_at_n_count / n * 100) if n > 0 else 0.0
        self.self_correction_delta = self.pass_at_n_accuracy - self.pass_at_1_accuracy
        self.avg_latency_ms = (self.total_latency_ms / len(self.task_results)) if self.task_results else 0.0
        self.ambiguity_accuracy = (
            (self.ambiguity_correct_count / self.ambiguity_tasks * 100)
            if self.ambiguity_tasks > 0 else 0.0
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full report to a JSON-serializable dictionary."""
        return {
            "summary": {
                "total_tasks": self.total_tasks,
                "ex_eligible_tasks": self.ex_eligible_tasks,
                "pass_at_1_accuracy_pct": None if self.aborted else round(self.pass_at_1_accuracy, 2),
                "pass_at_n_accuracy_pct": None if self.aborted else round(self.pass_at_n_accuracy, 2),
                "self_correction_delta_pct": None if self.aborted else round(self.self_correction_delta, 2),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "semantic_repair_count": self.semantic_repair_count,
                "ambiguity_classifier_accuracy_pct": None if self.aborted else round(self.ambiguity_accuracy, 2),
                "evaluation_error_count": self.evaluation_error_count,
                "run_status": "ABORTED_PROVIDER_ERROR" if self.aborted else "COMPLETED_WITH_ERRORS" if self.evaluation_error_count else "COMPLETED",
                "completed_tasks": len(self.task_results),
            },
            "metadata": self.metadata,
            "tasks": [
                {
                    "id": t.task_id,
                    "category": t.category,
                    "question": t.question,
                    "expected_ambiguity": t.expected_ambiguity,
                    "ex_eligible": t.ex_eligible,
                    "pass_at_1": t.pass_at_1,
                    "pass_at_n": t.pass_at_n,
                    "total_attempts": t.total_attempts,
                    "was_semantically_repaired": t.was_semantically_repaired,
                    "latency_ms": round(t.latency_ms, 2),
                    "ambiguity_predicted": t.ambiguity_predicted,
                    "ambiguity_correct": t.ambiguity_correct,
                    "generated_sql": t.generated_sql,
                    "ground_truth_sql": t.ground_truth_sql,
                    "pipeline_success": t.pipeline_success,
                    "error_message": t.error_message,
                    "tags": t.tags,
                    "first_sql": t.first_sql,
                    "first_error_type": t.first_error_type,
                    "clarified_context": t.clarified_context,
                    "needs_clarification": t.needs_clarification,
                    "classification_succeeded": t.classification_succeeded,
                    "evaluation_error": t.evaluation_error,
                    "order_sensitive": t.order_sensitive,
                    "provider_failed": t.provider_failed,
                    "provider_error_code": t.provider_error_code,
                    "quota_kind": t.quota_kind,
                }
                for t in self.task_results
            ],
        }


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Evaluate one observed pipeline trajectory per question, with UI defaults.

    Gold SQL is validated/executed before any LLM work. Gold outputs are visible
    only to the evaluator after generation and never drive the repair loop.
    """

    def __init__(self, db_conn=None, retriever: Optional[SchemaMetricRetriever] = None,
                 max_retries: int = 3, provider: Optional[str] = None,
                 comparator: Optional[ExecutionAccuracyComparator] = None):
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 1:
            raise ValueError("max_retries must be a positive integer.")
        self.db_conn = db_conn if db_conn is not None else get_connection()
        self.retriever = retriever
        self.max_retries = max_retries
        self.provider = provider
        self.comparator = comparator or ExecutionAccuracyComparator()
        self._pipeline = SelfCorrectingPipeline(db_conn=self.db_conn, retriever=retriever)

    def _classify_ambiguity(self, question: str) -> AmbiguityResult:
        return self._pipeline.check_ambiguity(user_question=question, provider=self.provider)

    def _prepare_tasks(self, tasks):
        if not isinstance(tasks, list):
            raise ValueError("Benchmark dataset must be a JSON list.")
        known_ids, references = set(), {}
        allowed_tables = set(get_schema_info(conn=self.db_conn))
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError("Every benchmark task must be an object.")
            task_id = task.get("id")
            if not isinstance(task_id, str) or not task_id.strip() or task_id in known_ids:
                raise ValueError("Benchmark IDs must be nonempty and unique.")
            known_ids.add(task_id)
            if not isinstance(task.get("question"), str) or not task["question"].strip():
                raise ValueError(f"{task_id}: question must be nonempty.")
            expected = task.get("expected_ambiguity", "CLEAR")
            if expected not in ("CLEAR", "SOFT_DEFAULT", "HARD_AMBIGUITY"):
                raise ValueError(f"{task_id}: invalid expected ambiguity.")
            if not isinstance(task.get("order_sensitive", False), bool):
                raise ValueError(f"{task_id}: order_sensitive must be boolean.")
            if expected == "HARD_AMBIGUITY":
                continue
            gold = task.get("ground_truth_sql")
            if not isinstance(gold, str) or not gold.strip():
                raise ValueError(f"{task_id}: EX-eligible tasks require ground_truth_sql.")
            valid, message = validate_sql(gold, allowed_tables=allowed_tables)
            if not valid:
                raise ValueError(f"{task_id}: unsafe/invalid ground truth SQL: {message}")
            frame, error = execute_query(gold, conn=self.db_conn)
            if error is not None or frame is None:
                raise ValueError(f"{task_id}: Ground truth SQL execution failed: {error}")
            references[task_id] = frame
        return references

    def run_benchmark(self, benchmark_file: str, progress_callback=None) -> BenchmarkReport:
        benchmark_path = Path(benchmark_file)
        with benchmark_path.open(encoding="utf-8") as handle:
            tasks = json.load(handle)
        references = self._prepare_tasks(tasks)
        provider = self.provider or settings.get_effective_provider()
        report = BenchmarkReport(total_tasks=len(tasks), expected_ex_eligible_tasks=len(references), metadata={
            "protocol": "single_trajectory_v1",
            "pass_at_1": "first observed pipeline attempt passed validation and matched gold",
            "pass_at_n": "final successful pipeline output matched gold within the attempt budget",
            "gold_feedback_to_pipeline": False,
            "max_attempts": self.max_retries,
            "provider": provider,
            "model": {"gemini": settings.gemini_model, "groq": settings.groq_model,
                "openai": settings.openai_model}.get(provider),
            "comparison": self.comparator.policy(),
            "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "retriever_enabled": self.retriever is not None,
        })
        for index, task in enumerate(tasks):
            record = TaskResult(
                task_id=task["id"], category=task.get("category", "unknown"),
                question=task["question"], expected_ambiguity=task.get("expected_ambiguity", "CLEAR"),
                ground_truth_sql=task.get("ground_truth_sql") or "", tags=task.get("tags", []),
                ex_eligible=task["id"] in references,
                order_sensitive=task.get("order_sensitive", False),
            )
            if record.ex_eligible:
                record.pass_at_1 = record.pass_at_n = False
            started = time.perf_counter()
            try:
                ambiguity = self._classify_ambiguity(record.question)
                record.ambiguity_predicted = ambiguity.ambiguity_type
                record.classification_succeeded = ambiguity.classification_succeeded
                record.provider_failed = ambiguity.provider_failed
                record.provider_error_code = ambiguity.provider_error_code
                record.quota_kind = ambiguity.quota_kind
                record.ambiguity_correct = ambiguity.classification_succeeded and ambiguity.ambiguity_type == record.expected_ambiguity
                if not ambiguity.classification_succeeded:
                    record.evaluation_error = True
                    record.error_message = ambiguity.reasoning
                record.needs_clarification = ambiguity.is_ambiguous
                if ambiguity.ambiguity_type == "SOFT_DEFAULT":
                    record.clarified_context = ambiguity.default_interpretation
                # A predicted HARD_AMBIGUITY pauses just as the assistant UI does.
                # Expected-HARD tasks remain classifier-only even if misclassified.
                if record.ex_eligible and ambiguity.classification_succeeded and not record.needs_clarification:
                    result = self._pipeline.execute_with_repair(
                        record.question, max_retries=self.max_retries,
                        provider=self.provider, clarified_context=record.clarified_context,
                    )
                    record.pipeline_success = result.success
                    record.total_attempts = result.total_attempts
                    record.generated_sql = result.final_sql
                    record.was_semantically_repaired = result.was_semantically_repaired
                    if not result.attempts:
                        raise ValueError("Pipeline returned no attempt telemetry; Pass@1 cannot be established.")
                    first = result.attempts[0]
                    record.first_sql, record.first_error_type = first.sql, first.error_type
                    truth = references[record.task_id]
                    record.pass_at_1 = first.success and first.df is not None and self.comparator.compare_outputs(
                        first.df, truth, order_sensitive=record.order_sensitive,
                    )
                    record.pass_at_n = result.success and result.df is not None and self.comparator.compare_outputs(
                        result.df, truth, order_sensitive=record.order_sensitive,
                    )
                    if result.error_message:
                        record.error_message = "; ".join(filter(None, (record.error_message, result.error_message)))
                    if any(attempt.error_type == "GENERATION_FAILURE" for attempt in result.attempts):
                        record.evaluation_error = True
                    for attempt in result.attempts:
                        if attempt.provider_failed:
                            record.provider_failed = True
                            record.provider_error_code = attempt.provider_error_code
                            record.quota_kind = attempt.quota_kind
            except Exception as exc:
                record.evaluation_error = True
                if record.ambiguity_correct is None:
                    record.ambiguity_correct = False
                    record.classification_succeeded = False
                record.error_message = "; ".join(filter(None, (record.error_message, str(exc))))
            record.latency_ms = (time.perf_counter() - started) * 1000
            report.task_results.append(record)
            if progress_callback is not None:
                progress_callback(index + 1, len(tasks), record.task_id)
            if record.provider_failed:
                report.metadata["abort_reason"] = {
                    "code": record.provider_error_code, "quota_kind": record.quota_kind,
                    "task_id": record.task_id, "message": record.error_message,
                }
                break
        return report.finalize()
