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
