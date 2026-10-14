"""Unit tests for the DataSense AI Evaluation & Benchmarking Engine.

Coverage:
- DataFrameNormalizer: float rounding, column name normalisation, string lowercasing,
  row sorting, NaN handling, empty DataFrames, mixed-type columns.
- ExecutionAccuracyComparator: identical frames, column reordering, shape mismatch,
  value mismatch, None inputs, NaN equality.
- BenchmarkRunner: ground truth execution, Pass@1 vs Pass@N logic, ambiguity
  classification accuracy, EX eligibility filtering for HARD_AMBIGUITY tasks.
- BenchmarkReport: finalize() aggregate metric derivations, to_dict() serialization.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pytest

from src.evaluator import (
    BenchmarkReport,
    BenchmarkRunner,
    DataFrameNormalizer,
    ExecutionAccuracyComparator,
    TaskResult,
)


# ─── DataFrameNormalizer ─────────────────────────────────────────────────────

class TestDataFrameNormalizer:

    def setup_method(self):
        self.norm = DataFrameNormalizer()

    def test_column_names_lowercased_and_stripped(self):
        df = pd.DataFrame({"  Name  ": ["Alice"], " AGE ": [30]})
        result = self.norm.normalize(df)
        assert list(result.columns) == ["name", "age"]

    def test_float_columns_rounded_to_2dp(self):
        df = pd.DataFrame({"revenue": [1234.5678, 99.999, 0.0]})
        result = self.norm.normalize(df)
        # After sort ascending: [0.0, 100.0, 1234.57]
        assert sorted(result["revenue"].tolist()) == [0.0, 100.0, 1234.57]

    def test_string_columns_lowercased(self):
        df = pd.DataFrame({"nation": ["GERMANY", "FRANCE", "Japan"]})
        result = self.norm.normalize(df)
        # After lowercasing and sort ascending
        assert list(result["nation"]) == ["france", "germany", "japan"]

    def test_string_nan_preserved(self):
        df = pd.DataFrame({"name": ["Alice", None, "Bob"]})
        result = self.norm.normalize(df)
        # After lowercase + sort: [None, 'alice', 'bob'] — None sorts first via str(None)='None'
        values = result["name"].tolist()
        assert len(values) == 3
        # At least one value should be None/NaN (the original None)
        has_null = any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values)
        assert has_null

    def test_rows_sorted_for_order_agnostic_comparison(self):
        df = pd.DataFrame({"value": [3, 1, 2]})
        result = self.norm.normalize(df)
        assert list(result["value"]) == [1, 2, 3]

    def test_empty_dataframe_returns_normalised_columns_only(self):
        df = pd.DataFrame(columns=["  OrderKey  ", "REVENUE"])
        result = self.norm.normalize(df)
        assert list(result.columns) == ["orderkey", "revenue"]
        assert len(result) == 0

    def test_integer_columns_unchanged(self):
        df = pd.DataFrame({"count": [5, 3, 10]})
        result = self.norm.normalize(df)
        assert sorted(result["count"].tolist()) == [3, 5, 10]

    def test_mixed_type_column_sorts_without_error(self):
        """Nullable integer column (Int64) must sort without TypeError."""
        df = pd.DataFrame({"qty": pd.array([3, None, 1], dtype="Int64")})
        result = self.norm.normalize(df)
        assert len(result) == 3

    def test_type_error_on_non_dataframe(self):
        with pytest.raises(TypeError):
            self.norm.normalize({"col": [1, 2]})


# ─── ExecutionAccuracyComparator ─────────────────────────────────────────────

class TestExecutionAccuracyComparator:

    def setup_method(self):
        self.cmp = ExecutionAccuracyComparator()

    def _df(self, data: dict) -> pd.DataFrame:
        return pd.DataFrame(data)

    def test_identical_frames_match(self):
        df = self._df({"revenue": [100.0, 200.0], "nation": ["GERMANY", "FRANCE"]})
        assert self.cmp.compare_outputs(df.copy(), df.copy()) is True

    def test_column_order_agnostic(self):
        """Same data, different column order → should still match."""
        df1 = self._df({"a": [1, 2], "b": ["x", "y"]})
        df2 = self._df({"b": ["x", "y"], "a": [1, 2]})
        assert self.cmp.compare_outputs(df1, df2) is True

    def test_row_order_agnostic(self):
        """Same data, different row order → should still match after sort."""
        df1 = self._df({"n": [3, 1, 2]})
        df2 = self._df({"n": [1, 2, 3]})
        assert self.cmp.compare_outputs(df1, df2) is True

    def test_float_rounding_tolerance(self):
        """Values that round to the same 2dp → match."""
        # 100.0049 → 100.0 and 100.0050 → 100.01 (banker's rounding on some systems)
        # Use values that definitively round to the same 2dp value
        df1 = self._df({"val": [100.001]})
        df2 = self._df({"val": [100.002]})
        # Both round to 100.0 at 2dp
        assert self.cmp.compare_outputs(df1, df2) is True

    def test_shape_mismatch_returns_false(self):
        df1 = self._df({"a": [1, 2, 3]})
        df2 = self._df({"a": [1, 2]})
        assert self.cmp.compare_outputs(df1, df2) is False

    def test_column_name_mismatch_returns_false(self):
        df1 = self._df({"revenue": [100.0]})
        df2 = self._df({"gross": [100.0]})
        assert self.cmp.compare_outputs(df1, df2) is False

    def test_value_mismatch_returns_false(self):
        df1 = self._df({"revenue": [100.0]})
        df2 = self._df({"revenue": [200.0]})
        assert self.cmp.compare_outputs(df1, df2) is False

    def test_none_input_returns_false(self):
        df = self._df({"a": [1]})
        assert self.cmp.compare_outputs(None, df) is False
        assert self.cmp.compare_outputs(df, None) is False
        assert self.cmp.compare_outputs(None, None) is False

    def test_nan_equality(self):
        """NaN == NaN should be treated as equal (pandas .equals() behavior)."""
        df1 = self._df({"val": [float("nan")]})
        df2 = self._df({"val": [float("nan")]})
        assert self.cmp.compare_outputs(df1, df2) is True

    def test_string_case_agnostic(self):
        """After normalisation, 'GERMANY' and 'germany' should match."""
        df1 = self._df({"nation": ["GERMANY"]})
        df2 = self._df({"nation": ["germany"]})
        assert self.cmp.compare_outputs(df1, df2) is True


# ─── BenchmarkReport ─────────────────────────────────────────────────────────

class TestBenchmarkReport:

    def _make_task(self, ex_eligible=True, pass_at_1=True, pass_at_n=True,
                   semantic=False, amb_correct=True) -> TaskResult:
        t = TaskResult(
            task_id="t1", category="test", question="Q?",
            expected_ambiguity="CLEAR",
            ex_eligible=ex_eligible,
            pass_at_1=pass_at_1,
            pass_at_n=pass_at_n,
            total_attempts=1 if pass_at_1 else 2,
            was_semantically_repaired=semantic,
            latency_ms=100.0,
            ambiguity_predicted="CLEAR",
            ambiguity_correct=amb_correct,
        )
        return t

    def test_finalize_pass_at_1_accuracy(self):
        report = BenchmarkReport(total_tasks=2)
        report.task_results = [
            self._make_task(pass_at_1=True, pass_at_n=True),
            self._make_task(pass_at_1=False, pass_at_n=True),
        ]
        report.finalize()
        assert report.pass_at_1_accuracy == 50.0
        assert report.pass_at_n_accuracy == 100.0

    def test_finalize_self_correction_delta(self):
        report = BenchmarkReport(total_tasks=2)
        report.task_results = [
            self._make_task(pass_at_1=True, pass_at_n=True),
            self._make_task(pass_at_1=False, pass_at_n=True),
        ]
        report.finalize()
        # Pass@1=50%, Pass@N=100%, Δ=50%
        assert report.self_correction_delta == 50.0

    def test_finalize_hard_ambiguity_excluded_from_ex(self):
        report = BenchmarkReport(total_tasks=2)
        report.task_results = [
            self._make_task(ex_eligible=True, pass_at_n=True),
            TaskResult(
                task_id="t_amb", category="ambiguity_classification_only",
                question="Show top customers", expected_ambiguity="HARD_AMBIGUITY",
                ex_eligible=False, pass_at_1=None, pass_at_n=None,
                ambiguity_predicted="HARD_AMBIGUITY", ambiguity_correct=True, latency_ms=50.0,
            ),
        ]
        report.finalize()
        assert report.ex_eligible_tasks == 1
        assert report.pass_at_n_accuracy == 100.0

    def test_finalize_ambiguity_accuracy(self):
        report = BenchmarkReport(total_tasks=3)
        report.task_results = [
            self._make_task(amb_correct=True),
            self._make_task(amb_correct=True),
            self._make_task(amb_correct=False),
        ]
        report.finalize()
        assert abs(report.ambiguity_accuracy - 66.67) < 0.01

    def test_finalize_avg_latency(self):
        report = BenchmarkReport(total_tasks=2)
        report.task_results = [
            self._make_task(),
            self._make_task(),
        ]
        # Both have latency_ms=100.0
        report.finalize()
        assert report.avg_latency_ms == 100.0

    def test_to_dict_has_required_keys(self):
        report = BenchmarkReport(total_tasks=1)
        report.task_results = [self._make_task()]
        report.finalize()
        d = report.to_dict()
        assert "summary" in d
        assert "tasks" in d
        summary_keys = [
            "total_tasks", "ex_eligible_tasks", "pass_at_1_accuracy_pct",
            "pass_at_n_accuracy_pct", "self_correction_delta_pct",
            "avg_latency_ms", "semantic_repair_count",
        ]
        for k in summary_keys:
            assert k in d["summary"], f"Missing key: {k}"

    def test_empty_report_finalize_no_division_error(self):
        report = BenchmarkReport(total_tasks=0)
        report.finalize()
        assert report.pass_at_1_accuracy == 0.0
        assert report.pass_at_n_accuracy == 0.0
        assert report.avg_latency_ms == 0.0


# ─── BenchmarkRunner (integration with in-memory DuckDB) ─────────────────────

class TestBenchmarkRunner:
    """Integration tests using a real in-memory DuckDB and mocked pipeline."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Provision minimal TPC-H-like schema in a fresh in-memory DuckDB."""
        self.conn = duckdb.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE nation (n_nationkey INTEGER, n_name VARCHAR, n_regionkey INTEGER);"
        )
        self.conn.execute(
            "INSERT INTO nation VALUES (1, 'GERMANY', 1), (2, 'FRANCE', 1), (3, 'BRAZIL', 2);"
        )
        self.conn.execute(
            "CREATE TABLE customer (c_custkey INTEGER, c_name VARCHAR, c_acctbal DOUBLE, c_nationkey INTEGER);"
        )
        self.conn.execute(
            "INSERT INTO customer VALUES (1, 'Alice', 5000.50, 1), (2, 'Bob', 3000.25, 2);"
        )

    def _make_benchmark_file(self, tasks: list) -> str:
        """Write tasks to a temp JSON file and return its path."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(tasks, tmp)
        tmp.flush()
        return tmp.name

    def _make_runner(self, generator_fn=None, repair_fn=None) -> BenchmarkRunner:
        """Create a BenchmarkRunner with a mocked pipeline generator."""
        runner = BenchmarkRunner(db_conn=self.conn, retriever=None, max_retries=2)
        if generator_fn or repair_fn:
            from src.pipeline import SelfCorrectingPipeline
            runner._pipeline = SelfCorrectingPipeline(
                db_conn=self.conn,
                generator_fn=generator_fn or (lambda *a, **kw: "SELECT 1;"),
                repair_fn=repair_fn or (lambda *a, **kw: "SELECT 1;"),
            )
        return runner

    def test_ex_eligible_pass_at_n_correct_sql(self):
        """Generator produces correct SQL → Pass@1 and Pass@N both True."""
        def gen(q, schema, **kw):
            return "SELECT COUNT(*) AS nation_count FROM nation;"

        tasks = [{
            "id": "t01",
            "category": "simple",
            "question": "How many nations?",
            "ground_truth_sql": "SELECT COUNT(*) AS nation_count FROM nation;",
            "expected_ambiguity": "CLEAR",
            "tags": [],
        }]
        path = self._make_benchmark_file(tasks)
        runner = self._make_runner(generator_fn=gen)

        with patch.object(runner, "_classify_ambiguity", return_value="CLEAR"):
            report = runner.run_benchmark(path)

        assert report.ex_eligible_tasks == 1
        assert report.pass_at_n_count == 1
        assert report.pass_at_n_accuracy == 100.0

    def test_hard_ambiguity_excluded_from_ex(self):
        """HARD_AMBIGUITY task should not count toward EX metrics."""
        tasks = [{
            "id": "t_amb",
            "category": "ambiguity_classification_only",
            "question": "Show top customers",
            "ground_truth_sql": None,
            "expected_ambiguity": "HARD_AMBIGUITY",
            "tags": [],
        }]
        path = self._make_benchmark_file(tasks)
        runner = self._make_runner()

        with patch.object(runner, "_classify_ambiguity", return_value="HARD_AMBIGUITY"):
            report = runner.run_benchmark(path)

        assert report.ex_eligible_tasks == 0
        assert report.pass_at_n_count == 0
        # Ambiguity classification accuracy should be measured
        assert report.ambiguity_tasks == 1
        assert report.ambiguity_accuracy == 100.0

    def test_wrong_sql_fails_ex(self):
        """Generator returning wrong SQL → Pass@N False."""
        def gen(q, schema, **kw):
            # Returns only GERMANY, ground truth returns all 3 nations
            return "SELECT COUNT(*) AS nation_count FROM nation WHERE n_name = 'GERMANY';"

        tasks = [{
            "id": "t02",
            "category": "simple",
            "question": "How many nations?",
            "ground_truth_sql": "SELECT COUNT(*) AS nation_count FROM nation;",
            "expected_ambiguity": "CLEAR",
            "tags": [],
        }]
        path = self._make_benchmark_file(tasks)
        runner = self._make_runner(generator_fn=gen)

        with patch.object(runner, "_classify_ambiguity", return_value="CLEAR"):
            report = runner.run_benchmark(path)

        assert report.pass_at_n_count == 0
        assert report.pass_at_n_accuracy == 0.0

    def test_pass_at_1_vs_pass_at_n_distinction(self):
        """Pass@1=False when 1-attempt result mismatches GT; Pass@N=True when full repair matches."""
        tasks = [{
            "id": "t03",
            "category": "repair",
            "question": "How many nations total?",
            "ground_truth_sql": "SELECT COUNT(*) AS nation_count FROM nation;",
            "expected_ambiguity": "CLEAR",
            "tags": [],
        }]
        path = self._make_benchmark_file(tasks)
        runner = self._make_runner()

        # Pass@1: generator returns count=1 (wrong — GT returns 3)
        wrong_df = pd.DataFrame({"nation_count": [1]})
        # Pass@N: repaired pipeline returns count=3 (correct)
        correct_df = pd.DataFrame({"nation_count": [3]})

        from src.pipeline import PipelineResult
        pass1_result = PipelineResult(success=True, final_sql="SELECT ...",
                                      df=wrong_df, total_attempts=1)
        passn_result = PipelineResult(success=True, final_sql="SELECT COUNT(*) AS nation_count FROM nation;",
                                      df=correct_df, total_attempts=2,
                                      was_semantically_repaired=False)

        with patch.object(runner, "_classify_ambiguity", return_value="CLEAR"), \
             patch.object(runner, "_run_single_attempt", return_value=pass1_result), \
             patch.object(runner, "_run_full_repair", return_value=passn_result):
            report = runner.run_benchmark(path)

        t = report.task_results[0]
        assert t.pass_at_1 is False, "Expected Pass@1 to be False (wrong count)"
        assert t.pass_at_n is True,  "Expected Pass@N to be True (correct count after repair)"
        assert report.self_correction_delta == 100.0

    def test_missing_benchmark_file_raises(self):
        runner = self._make_runner()
        with pytest.raises(FileNotFoundError):
            runner.run_benchmark("/tmp/nonexistent_benchmark_xyz.json")

    def test_progress_callback_called_for_each_task(self):
        """Progress callback should be invoked once per task."""
        calls = []

        def cb(current, total, task_id):
            calls.append((current, total, task_id))

        tasks = [
            {
                "id": f"t{i}",
                "category": "simple",
                "question": "How many nations?",
                "ground_truth_sql": "SELECT COUNT(*) AS n FROM nation;",
                "expected_ambiguity": "CLEAR",
                "tags": [],
            }
            for i in range(3)
        ]
        path = self._make_benchmark_file(tasks)
        runner = self._make_runner(
            generator_fn=lambda q, schema, **kw: "SELECT COUNT(*) AS n FROM nation;"
        )

        with patch.object(runner, "_classify_ambiguity", return_value="CLEAR"):
            runner.run_benchmark(path, progress_callback=cb)

        assert len(calls) == 3
        assert calls[-1][0] == 3  # last call current == total

    def test_ground_truth_failure_marks_task_ineligible(self):
        """If ground truth SQL itself fails, task should be marked ineligible."""
        tasks = [{
            "id": "t_bad_gt",
            "category": "simple",
            "question": "Broken ground truth",
            "ground_truth_sql": "SELECT nonexistent_col FROM nation;",
            "expected_ambiguity": "CLEAR",
            "tags": [],
        }]
        path = self._make_benchmark_file(tasks)
        runner = self._make_runner()

        with patch.object(runner, "_classify_ambiguity", return_value="CLEAR"):
            report = runner.run_benchmark(path)

        assert report.task_results[0].ex_eligible is False
        assert "Ground truth SQL execution failed" in report.task_results[0].error_message
