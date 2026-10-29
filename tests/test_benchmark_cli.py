"""Exercise the real CLI/runner path with a local database and injected providers."""

import json
import sys
from dataclasses import replace
from unittest.mock import MagicMock

import duckdb
import pytest

from scripts import run_benchmark as cli
from src.ambiguity import AmbiguityClassifier
from src.evaluator import BenchmarkRunner


@pytest.fixture
def cli_environment(tmp_path, monkeypatch):
    dataset = tmp_path / "benchmark.json"
    dataset.write_text(json.dumps([{
        "id": "cli", "question": "How many nations?",
        "ground_truth_sql": "SELECT COUNT(*) AS count FROM nation",
    }]))
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--benchmark", str(dataset),
        "--output", str(output), "--no-retriever", "--read-only"])
    monkeypatch.setattr(cli, "settings", replace(cli.settings, llm_provider="gemini", gemini_api_key="test"))
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE nation AS SELECT 'GERMANY' AS n_name")
        connection = MagicMock(return_value=conn)
        monkeypatch.setattr(cli, "get_connection", connection)
        monkeypatch.setattr(cli, "init_tpch_db", lambda *a, **kw: conn)

        def runner_factory(**kwargs):
            runner = BenchmarkRunner(**kwargs)
            runner._pipeline.classifier = AmbiguityClassifier(classifier_fn=lambda prompt: '{"ambiguity_type":"CLEAR"}')
            runner._pipeline.generator_fn = lambda *a, **kw: "SELECT COUNT(*) AS total FROM nation"
            return runner

        monkeypatch.setattr(cli, "BenchmarkRunner", runner_factory)
        yield output, connection, runner_factory


def test_cli_writes_auditable_report_and_respects_read_only(cli_environment):
    output, connection, _ = cli_environment
    assert cli.main() == 0
    connection.assert_called_once_with(read_only=True)
    report = json.loads(output.read_text())
    assert report["summary"]["pass_at_1_accuracy_pct"] == 100.0
    assert report["summary"]["run_status"] == "COMPLETED"
    assert report["metadata"]["protocol"] == "single_trajectory_v1"
    assert report["tasks"][0]["first_sql"] == "SELECT COUNT(*) AS total FROM nation"


def test_cli_accepts_groq_override_and_records_actual_model(cli_environment, monkeypatch):
    from src import evaluator
    output, _, _ = cli_environment
    policy = replace(cli.settings, llm_provider="groq", groq_api_key="test", groq_model="qwen/qwen3.8-27b")
    monkeypatch.setattr(cli, "settings", policy)
    monkeypatch.setattr(evaluator, "settings", policy)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--provider", "groq"])
    assert cli.main() == 0
    report = json.loads(output.read_text())
    assert report["metadata"]["provider"] == "groq"
    assert report["metadata"]["model"] == "qwen/qwen3.8-27b"


def test_cli_returns_failure_status_when_provider_generation_fails(cli_environment, monkeypatch):
    output, _, make_runner = cli_environment

    def unavailable_runner(**kwargs):
        runner = make_runner(**kwargs)
        runner._pipeline.generator_fn = MagicMock(side_effect=RuntimeError("provider unavailable"))
        return runner

    monkeypatch.setattr(cli, "BenchmarkRunner", unavailable_runner)
    assert cli.main() == 2
    report = json.loads(output.read_text())
    assert report["summary"]["evaluation_error_count"] == 1
    assert report["summary"]["run_status"] == "COMPLETED_WITH_ERRORS"


def test_cli_does_not_silently_disable_a_failed_retriever(cli_environment, monkeypatch):
    output, _, _ = cli_environment
    monkeypatch.setattr(sys, "argv", [value for value in sys.argv if value != "--no-retriever"])
    monkeypatch.setattr(cli, "SchemaMetricRetriever", MagicMock(side_effect=RuntimeError("index unavailable")))
    assert cli.main() == 1
    assert not output.exists()


@pytest.mark.parametrize("problem", ["missing_key", "invalid_budget"])
def test_cli_rejects_invalid_setup_before_opening_a_database(cli_environment, monkeypatch, problem):
    _, connection, _ = cli_environment
    if problem == "missing_key":
        monkeypatch.setattr(cli, "settings", replace(cli.settings, gemini_api_key=""))
    else:
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--retries", "0"])
    assert cli.main() == 1
    connection.assert_not_called()


def test_cli_rejects_malformed_dataset_before_database_setup(cli_environment):
    _, connection, _ = cli_environment
    from pathlib import Path
    dataset = Path(sys.argv[sys.argv.index("--benchmark") + 1])
    dataset.write_text("{broken json")
    assert cli.main() == 1
    connection.assert_not_called()


def test_cli_aborts_on_daily_quota_without_publishing_zero_accuracy(cli_environment, monkeypatch, capsys):
    from pathlib import Path
    from src.llm import GeminiRequestError
    output, _, make_runner = cli_environment
    dataset = Path(sys.argv[sys.argv.index("--benchmark") + 1])
    tasks = json.loads(dataset.read_text())
    dataset.write_text(json.dumps([*tasks, {**tasks[0], "id": "second"}]))
    generate = MagicMock(side_effect=GeminiRequestError("model", 1, 429, "Daily quota exhausted", quota_kind="daily"))

    def quota_runner(**kwargs):
        runner = make_runner(**kwargs)
        runner._pipeline.generator_fn = generate
        return runner

    monkeypatch.setattr(cli, "BenchmarkRunner", quota_runner)
    assert cli.main() == 2
    generate.assert_called_once()
    report = json.loads(output.read_text())
    assert report["summary"]["run_status"] == "ABORTED_PROVIDER_ERROR"
    assert report["summary"]["completed_tasks"] == 1
    assert report["summary"]["pass_at_n_accuracy_pct"] is None
    assert "Accuracy is unmeasured" in capsys.readouterr().out
