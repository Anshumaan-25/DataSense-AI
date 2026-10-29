"""Tests for Bounded Self-Correction Repair Pipeline (State Machine)."""

import duckdb
import pytest
import pandas as pd
from src.database import init_tpch_db
from src.pipeline import SelfCorrectingPipeline, PipelineResult, sanitize_error_message


@pytest.fixture(scope="module")
def memory_db():
    """In-memory DuckDB connection for fast pipeline tests."""
    conn = duckdb.connect(":memory:")
    init_tpch_db(conn=conn, scale_factor=0.01)
    return conn


def test_sanitize_error_message():
    """Verify error sanitizer extracts DuckDB compiler messages and strips tracebacks."""
    raw_duckdb_err = """
    Traceback (most recent call last):
      File "/tmp/test.py", line 12, in <module>
    duckdb.BinderException: Binder Error: Referenced column "c_cust_id" not found in FROM clause!
    Candidate: "c_custkey"
    LINE 1: SELECT c_cust_id FROM customer
                   ^
    """
    clean = sanitize_error_message(raw_duckdb_err)
    assert "Binder Error: Referenced column" in clean
    assert "Candidate: \"c_custkey\"" in clean
    assert "Traceback" not in clean


def test_first_attempt_success(memory_db):
    """Verify successful query on attempt 1 completes without triggering repairs."""
    pipeline = SelfCorrectingPipeline(
        db_conn=memory_db,
        generator_fn=lambda q, s, rag_context="", provider=None: "SELECT c_name FROM customer LIMIT 5;",
    )
    result = pipeline.execute_with_repair("Who are 5 customers?")
    assert result.success is True
    assert result.total_attempts == 1
    assert result.is_security_violation is False
    assert len(result.attempt_history) == 0
    assert isinstance(result.df, pd.DataFrame)
    assert len(result.df) == 5


def test_repair_on_duckdb_runtime_error(memory_db):
    """Verify DuckDB runtime binder error (column typo) triggers self-correction and succeeds on attempt 2."""
    call_counts = {"gen": 0, "repair": 0}

    def mock_generator(q, s, rag_context="", provider=None):
        call_counts["gen"] += 1
        return "SELECT c_namee FROM customer LIMIT 5;"  # Deliberate column typo

    def mock_repair(user_question, schema_info, rag_context, failed_sql, error_message, attempt_history, provider=None):
        call_counts["repair"] += 1
        assert "c_namee" in failed_sql
        assert "Binder Error" in error_message or "not found" in error_message.lower()
        return "SELECT c_name FROM customer LIMIT 5;"  # Fixed SQL

    pipeline = SelfCorrectingPipeline(
        db_conn=memory_db,
        generator_fn=mock_generator,
        repair_fn=mock_repair,
    )
    result = pipeline.execute_with_repair("Show 5 customer names")

    assert result.success is True
    assert result.total_attempts == 2
    assert call_counts["gen"] == 1
    assert call_counts["repair"] == 1
    assert len(result.attempt_history) == 1
    assert result.attempt_history[0]["error_type"] == "DUCKDB_RUNTIME"
    assert "c_namee" in result.attempt_history[0]["sql"]
    assert isinstance(result.df, pd.DataFrame)


def test_repair_on_ast_syntax_error(memory_db):
    """Verify malformed syntax rejected by AST validator is repaired on attempt 2."""
    def mock_generator(q, s, rag_context="", provider=None):
        return "SELECT c_name FROM customer WHERE ;"  # Syntax error

    def mock_repair(user_question, schema_info, rag_context, failed_sql, error_message, attempt_history, provider=None):
        assert "syntax error" in error_message.lower() or "failed to parse" in error_message.lower()
        return "SELECT c_name FROM customer WHERE c_acctbal > 0 LIMIT 5;"

    pipeline = SelfCorrectingPipeline(
        db_conn=memory_db,
        generator_fn=mock_generator,
        repair_fn=mock_repair,
    )
    result = pipeline.execute_with_repair("Show positive balance customers")

    assert result.success is True
    assert result.total_attempts == 2
    assert len(result.attempt_history) == 1
    assert result.attempt_history[0]["error_type"] == "AST_SYNTAX"


def test_security_violation_never_retried(memory_db):
    """CRITICAL: Verify security policy violation triggers immediate fatal abort with zero retries."""
    repair_called = False

    def mock_generator(q, s, rag_context="", provider=None):
        return "DROP TABLE customer;"

    def mock_repair(*args, **kwargs):
        nonlocal repair_called
        repair_called = True
        return "SELECT * FROM customer;"

    pipeline = SelfCorrectingPipeline(
        db_conn=memory_db,
        generator_fn=mock_generator,
        repair_fn=mock_repair,
    )
    result = pipeline.execute_with_repair("DROP TABLE customer;")

    assert result.success is False
    assert result.is_security_violation is True
    assert result.total_attempts == 1
    assert repair_called is False  # MUST NEVER BE RETRIED
    assert result.attempt_history[0]["error_type"] == "SECURITY_VIOLATION"


def test_file_read_security_never_retried(memory_db):
    """CRITICAL: Verify file read attempt triggers immediate fatal abort with zero retries."""
    repair_called = False

    def mock_generator(q, s, rag_context="", provider=None):
        return "SELECT * FROM read_csv('/etc/passwd');"

    def mock_repair(*args, **kwargs):
        nonlocal repair_called
        repair_called = True
        return "SELECT * FROM customer;"

    pipeline = SelfCorrectingPipeline(
        db_conn=memory_db,
        generator_fn=mock_generator,
        repair_fn=mock_repair,
    )
    result = pipeline.execute_with_repair("Dump password file")

    assert result.success is False
    assert result.is_security_violation is True
    assert result.total_attempts == 1
    assert repair_called is False  # MUST NEVER BE RETRIED


def test_max_retries_capped(memory_db):
    """Verify repair loop strictly terminates when max_retries limit is reached."""
    repair_attempts = 0

    def mock_generator(q, s, rag_context="", provider=None):
        return "SELECT non_existent_col_0 FROM customer;"

    def mock_repair(user_question, schema_info, rag_context, failed_sql, error_message, attempt_history, provider=None):
        nonlocal repair_attempts
        repair_attempts += 1
        return f"SELECT non_existent_col_{repair_attempts} FROM customer;"

    pipeline = SelfCorrectingPipeline(
        db_conn=memory_db,
        generator_fn=mock_generator,
        repair_fn=mock_repair,
    )
    result = pipeline.execute_with_repair("Query invalid cols", max_retries=3)

    assert result.success is False
    assert result.total_attempts == 3
    assert repair_attempts == 2  # Attempt 1 = gen, Attempt 2 = repair 1, Attempt 3 = repair 2
    assert len(result.attempt_history) == 3
    assert "Failed to auto-repair" in result.error_message


def test_cycle_stagnation_detection(memory_db):
    """Verify loop stagnation is detected when the model repeats identical SQL."""
    def mock_generator(q, s, rag_context="", provider=None):
        return "SELECT invalid_col FROM customer;"

    def mock_repair(user_question, schema_info, rag_context, failed_sql, error_message, attempt_history, provider=None):
        # Repeats the exact same failed SQL
        return "SELECT invalid_col FROM customer;"

    pipeline = SelfCorrectingPipeline(
        db_conn=memory_db,
        generator_fn=mock_generator,
        repair_fn=mock_repair,
    )
    result = pipeline.execute_with_repair("Query invalid col", max_retries=3)

    assert result.success is False
    assert result.total_attempts == 2  # Aborts on attempt 2 when stagnation detected
    assert "stagnation" in result.error_message.lower()
    assert result.attempt_history[-1]["error_type"] == "CYCLE_STAGNATION"


@pytest.mark.parametrize("failed_sql", [
    "SELECT CAST(raw_value AS INTEGER) FROM orders",
    "SELECT * FROM orders WHERE raw_value > 10",
])
def test_type_repairs_receive_only_referenced_sample_rows(failed_sql):
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE orders AS SELECT 'USD 20' AS raw_value FROM range(5)")
        conn.execute("CREATE TABLE customer AS SELECT 'unrelated' AS secret")
        captured = []

        def repair(**kwargs):
            captured.append(kwargs)
            return "SELECT raw_value FROM orders LIMIT 3"

        result = SelfCorrectingPipeline(
            db_conn=conn, generator_fn=lambda *a, **kw: failed_sql, repair_fn=repair,
        ).execute_with_repair("Show order values", clarified_context="Keep original currency text")
        assert result.success
        assert result.total_attempts == 2
        assert captured[0]["table_samples"] == {"orders": [{"raw_value": "USD 20"}] * 3}
        assert captured[0]["clarified_context"] == "Keep original currency text"


def test_ordinary_binder_errors_do_not_sample_tables(memory_db, monkeypatch):
    from src import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "get_query_sample_rows", lambda *a, **kw: pytest.fail("unnecessary sampling"))
    result = SelfCorrectingPipeline(
        db_conn=memory_db,
        generator_fn=lambda *a, **kw: "SELECT missing_column FROM orders",
        repair_fn=lambda **kw: "SELECT o_orderkey FROM orders LIMIT 1",
    ).execute_with_repair("Show one order")
    assert result.success


def test_security_rejections_do_not_sample_tables(memory_db, monkeypatch):
    from src import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "get_query_sample_rows", lambda *a, **kw: pytest.fail("security violation sampled"))
    result = SelfCorrectingPipeline(db_conn=memory_db).execute_with_repair("SELECT * FROM read_csv('/etc/passwd')")
    assert result.is_security_violation
    assert result.total_attempts == 1


def test_sample_failure_does_not_prevent_repair(monkeypatch):
    from src import pipeline as pipeline_module

    def unavailable(*args, **kwargs):
        raise RuntimeError("sample unavailable")

    monkeypatch.setattr(pipeline_module, "get_query_sample_rows", unavailable)
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE orders AS SELECT 'USD 20' AS raw_value")

        def repair(**kwargs):
            assert kwargs["table_samples"] == {}
            return "SELECT raw_value FROM orders"

        result = SelfCorrectingPipeline(
            db_conn=conn,
            generator_fn=lambda *a, **kw: "SELECT CAST(raw_value AS INTEGER) FROM orders",
            repair_fn=repair,
        ).execute_with_repair("Show order values")
        assert result.success


def test_internal_model_type_error_is_not_reinvoked(memory_db):
    calls = []

    def generator(*args, **kwargs):
        calls.append(1)
        raise TypeError("failure inside the model function")

    result = SelfCorrectingPipeline(db_conn=memory_db, generator_fn=generator).execute_with_repair("Show orders")
    assert not result.success
    assert len(calls) == 1
    assert "failure inside the model function" in result.error_message


def test_attempt_telemetry_keeps_initial_failure_and_actual_success(memory_db):
    result = SelfCorrectingPipeline(
        db_conn=memory_db,
        generator_fn=lambda *a, **kw: "SELECT missing_column FROM orders",
        repair_fn=lambda **kw: "SELECT o_orderkey FROM orders LIMIT 1",
    ).execute_with_repair("Show one order")
    assert [outcome.attempt for outcome in result.attempts] == [1, 2]
    assert result.attempts[0].error_type == "DUCKDB_RUNTIME"
    assert result.attempts[0].df is None and not result.attempts[0].success
    assert result.attempts[1].success and len(result.attempts[1].df) == 1


def test_generation_failure_records_an_attempt_without_fake_sql(memory_db):
    def unavailable(*args, **kwargs):
        raise RuntimeError("provider unavailable")
    result = SelfCorrectingPipeline(db_conn=memory_db, generator_fn=unavailable).execute_with_repair("Show orders")
    assert len(result.attempts) == 1
    assert result.attempts[0].sql == ""
    assert result.attempts[0].error_type == "GENERATION_FAILURE"


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_attempt_budgets_are_rejected(memory_db, budget):
    with pytest.raises(ValueError, match="positive integer"):
        SelfCorrectingPipeline(db_conn=memory_db).execute_with_repair("Show orders", max_retries=budget)
