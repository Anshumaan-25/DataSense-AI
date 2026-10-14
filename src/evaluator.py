"""Evaluation & Benchmarking Engine for DataSense AI — TPC-H Execution Accuracy (EX).

Architecture:
- DataFrameNormalizer:         Deterministic normalization for order-agnostic comparison.
- ExecutionAccuracyComparator: Shape + value comparison of two normalized DataFrames.
- BenchmarkRunner:             Orchestrates the full benchmark against SelfCorrectingPipeline.
- BenchmarkReport:             Structured telemetry with per-task and aggregate metrics.

Design Decisions:
- Ground truth SQL is EXECUTED LIVE against the same DuckDB instance at benchmark time —
  never stored as hardcoded expected values. This ensures scale-factor-agnostic EX scores.
- Pass@1 is measured via a separate single-attempt pipeline call (max_retries=1), then the
  full repair loop runs independently. This doubles API cost for failed-at-1 queries but
  produces a clean, honest Pass@1 measurement without polluting the pipeline state.
- Ambiguous queries (expected_ambiguity = 'HARD_AMBIGUITY') are EXCLUDED from EX measurement
  and tracked separately via Ambiguity Classifier Accuracy to avoid category errors.
- SOFT_DEFAULT queries ARE included in EX — the pipeline applies defaults automatically.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.database import get_connection, init_tpch_db, execute_query
from src.pipeline import SelfCorrectingPipeline, PipelineResult
from src.retriever import SchemaMetricRetriever


# ---------------------------------------------------------------------------
# DataFrameNormalizer
# ---------------------------------------------------------------------------

class DataFrameNormalizer:
    """Normalizes a DataFrame into a canonical form for order-agnostic comparison.

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
    """Compares two DataFrames for exact execution accuracy (EX).

    Comparison contract:
    - Both DataFrames are normalized before comparison.
    - Shape (rows × columns) must be identical.
    - Column names (as sets) must be identical (order-agnostic).
    - Cell values must match element-wise after both frames are column-sorted.
    """

    def __init__(self) -> None:
        self._normalizer = DataFrameNormalizer()

    def compare_outputs(
        self,
        df_generated: pd.DataFrame,
        df_truth: pd.DataFrame,
    ) -> bool:
        """Return True if both DataFrames represent equivalent result sets.

        Args:
            df_generated: DataFrame produced by the LLM-generated SQL.
            df_truth:     DataFrame produced by the ground truth SQL.

        Returns:
            True if normalized shape, column names, and all cell values match exactly.
        """
        if df_generated is None or df_truth is None:
            return False

        norm_gen = self._normalizer.normalize(df_generated)
        norm_truth = self._normalizer.normalize(df_truth)

        # Shape guard
        if norm_gen.shape != norm_truth.shape:
            return False

        # Column name guard (set comparison — order-agnostic)
        if set(norm_gen.columns) != set(norm_truth.columns):
            return False

        # Align column order to truth before cell-level comparison
        try:
            norm_gen = norm_gen[norm_truth.columns]
        except KeyError:
            return False

        # Cell-level comparison (NaN == NaN treated as equal)
        try:
            return norm_gen.equals(norm_truth)
        except Exception:
            return False


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

    def finalize(self) -> "BenchmarkReport":
        """Compute aggregate accuracy metrics from task-level results. Returns self."""
        eligible = [t for t in self.task_results if t.ex_eligible]
        self.ex_eligible_tasks = len(eligible)

        self.pass_at_1_count = sum(1 for t in eligible if t.pass_at_1 is True)
        self.pass_at_n_count = sum(1 for t in eligible if t.pass_at_n is True)
        self.semantic_repair_count = sum(1 for t in eligible if t.was_semantically_repaired)
        self.total_latency_ms = sum(t.latency_ms for t in self.task_results)

        amb_tasks = [t for t in self.task_results if t.ambiguity_correct is not None]
        self.ambiguity_tasks = len(amb_tasks)
        self.ambiguity_correct_count = sum(1 for t in amb_tasks if t.ambiguity_correct is True)

        n = self.ex_eligible_tasks
        self.pass_at_1_accuracy = (self.pass_at_1_count / n * 100) if n > 0 else 0.0
        self.pass_at_n_accuracy = (self.pass_at_n_count / n * 100) if n > 0 else 0.0
        self.self_correction_delta = self.pass_at_n_accuracy - self.pass_at_1_accuracy
        self.avg_latency_ms = (self.total_latency_ms / self.total_tasks) if self.total_tasks > 0 else 0.0
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
                "pass_at_1_accuracy_pct": round(self.pass_at_1_accuracy, 2),
                "pass_at_n_accuracy_pct": round(self.pass_at_n_accuracy, 2),
                "self_correction_delta_pct": round(self.self_correction_delta, 2),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "semantic_repair_count": self.semantic_repair_count,
                "ambiguity_classifier_accuracy_pct": round(self.ambiguity_accuracy, 2),
            },
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
                }
                for t in self.task_results
            ],
        }


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Orchestrates the full TPC-H benchmark evaluation over all tasks.

    Workflow per task:
    1. Execute ground truth SQL live to get the reference DataFrame.
    2. Run ambiguity classifier and record classifier accuracy.
    3. For EX-eligible tasks (non-HARD_AMBIGUITY):
       a. Run pipeline with max_retries=1 → record Pass@1.
       b. Run pipeline with full max_retries → record Pass@N.
       c. Compare final generated DataFrame to ground truth DataFrame via EX comparator.
    4. Accumulate into BenchmarkReport and finalize aggregate metrics.
    """

    def __init__(
        self,
        db_conn=None,
        retriever: Optional[SchemaMetricRetriever] = None,
        max_retries: int = 3,
        provider: Optional[str] = None,
    ) -> None:
        """Initialize the benchmark runner.

        Args:
            db_conn:     DuckDB connection. Defaults to shared global connection.
            retriever:   SchemaMetricRetriever for RAG context. Optional.
            max_retries: Max self-repair attempts for Pass@N evaluation.
            provider:    LLM provider override ('gemini', 'openai', or None for auto).
        """
        self.db_conn = db_conn if db_conn is not None else get_connection()
        self.retriever = retriever
        self.max_retries = max_retries
        self.provider = provider
        self.comparator = ExecutionAccuracyComparator()

        # Shared pipeline instance (re-used across tasks)
        self._pipeline = SelfCorrectingPipeline(
            db_conn=self.db_conn,
            retriever=self.retriever,
        )

    def _execute_ground_truth(self, sql: str) -> Optional[pd.DataFrame]:
        """Execute the ground truth SQL and return its DataFrame, or None on error."""
        df, err = execute_query(sql, conn=self.db_conn)
        if err is not None:
            return None
        return df

    def _classify_ambiguity(self, question: str) -> Optional[str]:
        """Run the ambiguity classifier and return the predicted ambiguity_type string."""
        try:
            result = self._pipeline.check_ambiguity(
                user_question=question,
                provider=self.provider,
            )
            return result.ambiguity_type
        except Exception:
            return None

    def _run_single_attempt(
        self,
        question: str,
        clarified_context: str = "",
    ) -> PipelineResult:
        """Run pipeline capped at exactly 1 attempt for Pass@1 measurement."""
        return self._pipeline.execute_with_repair(
            user_question=question,
            max_retries=1,
            provider=self.provider,
            clarified_context=clarified_context,
        )

    def _run_full_repair(
        self,
        question: str,
        clarified_context: str = "",
    ) -> PipelineResult:
        """Run pipeline with full self-repair for Pass@N measurement."""
        return self._pipeline.execute_with_repair(
            user_question=question,
            max_retries=self.max_retries,
            provider=self.provider,
            clarified_context=clarified_context,
        )

    def run_benchmark(
        self,
        benchmark_file: str,
        progress_callback=None,
    ) -> BenchmarkReport:
        """Run the full benchmark suite and return a finalized BenchmarkReport.

        Args:
            benchmark_file:    Path to the JSON benchmark dataset file.
            progress_callback: Optional callable(current: int, total: int, task_id: str)
                               called after each task completes (used by Streamlit progress bar).

        Returns:
            Finalized BenchmarkReport with per-task results and aggregate metrics.

        Raises:
            FileNotFoundError: If benchmark_file does not exist.
            json.JSONDecodeError: If benchmark_file is malformed JSON.
        """
        benchmark_path = Path(benchmark_file)
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

        with open(benchmark_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        report = BenchmarkReport(total_tasks=len(tasks))
        total = len(tasks)

        for idx, task in enumerate(tasks):
            task_id = task.get("id", f"task_{idx}")
            question = task.get("question", "")
            category = task.get("category", "unknown")
            expected_ambiguity = task.get("expected_ambiguity", "CLEAR")
            ground_truth_sql = task.get("ground_truth_sql")
            tags = task.get("tags", [])

            task_result = TaskResult(
                task_id=task_id,
                category=category,
                question=question,
                expected_ambiguity=expected_ambiguity,
                ground_truth_sql=ground_truth_sql or "",
                tags=tags,
            )

            task_start = time.perf_counter()

            # ----------------------------------------------------------------
            # Step 1: Ambiguity Classifier Accuracy (all tasks)
            # ----------------------------------------------------------------
            predicted_ambiguity = self._classify_ambiguity(question)
            task_result.ambiguity_predicted = predicted_ambiguity
            if predicted_ambiguity is not None:
                task_result.ambiguity_correct = (
                    predicted_ambiguity == expected_ambiguity
                )

            # ----------------------------------------------------------------
            # Step 2: EX Evaluation (skip for HARD_AMBIGUITY tasks)
            # ----------------------------------------------------------------
            is_ex_eligible = (
                expected_ambiguity != "HARD_AMBIGUITY"
                and ground_truth_sql is not None
                and ground_truth_sql.strip() != ""
            )
            task_result.ex_eligible = is_ex_eligible

            if is_ex_eligible:
                # Execute ground truth live
                df_truth = self._execute_ground_truth(ground_truth_sql)

                if df_truth is None:
                    # Ground truth itself failed — skip EX for this task
                    task_result.error_message = "Ground truth SQL execution failed."
                    task_result.ex_eligible = False
                else:
                    # --- Pass@1 ---
                    try:
                        result_1 = self._run_single_attempt(question)
                        if result_1.success and result_1.df is not None:
                            task_result.pass_at_1 = self.comparator.compare_outputs(
                                result_1.df, df_truth
                            )
                        else:
                            task_result.pass_at_1 = False
                    except Exception as e:
                        task_result.pass_at_1 = False
                        task_result.error_message += f" Pass@1 error: {e}"

                    # --- Pass@N (full repair loop) ---
                    try:
                        result_n = self._run_full_repair(question)
                        task_result.pipeline_success = result_n.success
                        task_result.total_attempts = result_n.total_attempts
                        task_result.was_semantically_repaired = result_n.was_semantically_repaired
                        task_result.generated_sql = result_n.final_sql

                        if result_n.success and result_n.df is not None:
                            task_result.pass_at_n = self.comparator.compare_outputs(
                                result_n.df, df_truth
                            )
                        else:
                            task_result.pass_at_n = False
                            if result_n.error_message:
                                task_result.error_message += f" Pass@N error: {result_n.error_message}"
                    except Exception as e:
                        task_result.pass_at_n = False
                        task_result.error_message += f" Pass@N exception: {e}"

            task_result.latency_ms = (time.perf_counter() - task_start) * 1000
            report.task_results.append(task_result)

            if progress_callback is not None:
                try:
                    progress_callback(idx + 1, total, task_id)
                except Exception:
                    pass

        return report.finalize()
