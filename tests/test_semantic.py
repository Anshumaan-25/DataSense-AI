"""Unit tests for Semantic Validator and auto-repair integration."""

import pytest
import pandas as pd
import duckdb
from src.semantic import (
    SemanticValidator,
    SemanticValidationResult,
    extract_string_literals,
)
from src.pipeline import SelfCorrectingPipeline


def test_extract_string_literals():
    """Verify regex extraction of SQL string literals."""
    sql = "SELECT * FROM customer WHERE c_mktsegment = 'BUILDING' AND c_comment LIKE '%special%';"
    literals = extract_string_literals(sql)
    assert literals == ["BUILDING", "%special%"]


def test_check_zero_rows_casing_mismatch():
    """Verify zero-row query with lowercase constant triggers CASING_MISMATCH."""
    validator = SemanticValidator()
    df_empty = pd.DataFrame(columns=["c_name", "n_name"])
    sql = "SELECT c_name, n_name FROM customer JOIN nation ON c_nationkey = n_nationkey WHERE n_name = 'germany';"

    result = validator.validate_output(
        user_question="What are the customers in Germany?",
        sql_query=sql,
        df=df_empty,
    )

    assert result.is_valid is False
    assert result.issue_type == "CASING_MISMATCH"
    assert "GERMANY" in result.reasoning
    assert "GERMANY" in result.suggested_fix


def test_check_zero_rows_valid_negative_check():
    """Verify negative existence checks returning 0 rows are treated as valid."""
    validator = SemanticValidator()
    df_empty = pd.DataFrame(columns=["o_orderkey"])
    sql = "SELECT o_orderkey FROM orders WHERE o_totalprice < 0;"

    result = validator.validate_output(
        user_question="Are there any orders with a negative total price?",
        sql_query=sql,
        df=df_empty,
    )

    assert result.is_valid is True
    assert result.issue_type == "NONE"


def test_check_cartesian_explosion():
    """Verify multi-table unaggregated query producing excessive rows triggers CARTESIAN_EXPLOSION."""
    validator = SemanticValidator()
    # Mock a dataframe with 1500 rows from a cross join
    df_large = pd.DataFrame({"c_custkey": range(1500), "o_orderkey": range(1500)})
    sql = "SELECT c_custkey, o_orderkey FROM customer, orders;"

    result = validator.validate_output(
        user_question="Show customer orders",
        sql_query=sql,
        df=df_large,
    )

    assert result.is_valid is False
    assert result.issue_type == "CARTESIAN_EXPLOSION"
    assert "Cartesian product" in result.reasoning
    assert "join keys" in result.suggested_fix


def test_check_catalog_metric_omission():
    """Verify query computing gross revenue instead of retrieved net revenue formula triggers CATALOG_METRIC_OMISSION."""
    validator = SemanticValidator()
    df_result = pd.DataFrame({"gross": [100000.0]})
    # SQL computes gross revenue without discount formula
    sql = "SELECT SUM(l_extendedprice) as gross FROM lineitem;"
    rag_context = "net_revenue: SUM(l_extendedprice * (1 - l_discount))"

    result = validator.validate_output(
        user_question="What is the net revenue?",
        sql_query=sql,
        df=df_result,
        rag_context=rag_context,
    )

    assert result.is_valid is False
    assert result.issue_type == "CATALOG_METRIC_OMISSION"
    assert "gross revenue" in result.reasoning
    assert "1 - l_discount" in result.suggested_fix


def test_check_aggregate_mismatch():
    """Verify scalar total question returning unaggregated list triggers AGGREGATE_MISMATCH."""
    validator = SemanticValidator()
    # 25 unaggregated rows
    df_rows = pd.DataFrame({"l_extendedprice": [100.0] * 25})
    sql = "SELECT l_extendedprice FROM lineitem LIMIT 25;"

    result = validator.validate_output(
        user_question="What is the total revenue of lineitems?",
        sql_query=sql,
        df=df_rows,
    )

    assert result.is_valid is False
    assert result.issue_type == "AGGREGATE_MISMATCH"
    assert "scalar summary aggregation" in result.reasoning
    assert "SUM(" in result.suggested_fix


def test_fast_path_bypasses_llm_judge():
    """Verify valid queries pass through fast-path with zero judge calls."""
    judge_called = False

    def mock_judge(prompt):
        nonlocal judge_called
        judge_called = True
        return '{"is_valid": true}'

    validator = SemanticValidator(judge_fn=mock_judge, enable_llm_judge=False)
    df_valid = pd.DataFrame({"total": [5000.0]})
    sql = "SELECT count(*) FROM customer;"

    result = validator.validate_output(
        user_question="How many customers are there?",
        sql_query=sql,
        df=df_valid,
    )

    assert result.is_valid is True
    assert result.issue_type == "NONE"
    assert judge_called is False  # Fast path avoided judge call entirely!


def test_pipeline_semantic_auto_repair_flow():
    """Verify pipeline catches semantic failure on Attempt 1 and auto-repairs on Attempt 2."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE nation (n_nationkey BIGINT, n_name VARCHAR);")
    conn.execute("INSERT INTO nation VALUES (1, 'GERMANY'), (2, 'FRANCE');")

    captured_attempts = []

    def staged_generator(user_question, schema_str, rag_context="", clarified_context="", provider=None):
        # Attempt 1: Lowercase 'germany' which will yield 0 rows in DuckDB!
        captured_attempts.append("GEN")
        return "SELECT n_name FROM nation WHERE n_name = 'germany';"

    def staged_repair(user_question, schema_info, rag_context="", failed_sql="", error_message="", attempt_history=None, clarified_context="", provider=None):
        # Attempt 2: Repaired to uppercase 'GERMANY'
        captured_attempts.append("REPAIR")
        assert "GERMANY" in error_message or "CASING_MISMATCH" in str(attempt_history)
        return "SELECT n_name FROM nation WHERE n_name = 'GERMANY';"

    pipeline = SelfCorrectingPipeline(
        db_conn=conn,
        generator_fn=staged_generator,
        repair_fn=staged_repair,
    )

    result = pipeline.execute_with_repair(
        user_question="Show nation Germany",
        max_retries=3,
        clarified_context="Filter by Germany",
    )

    assert result.success is True
    assert result.total_attempts == 2
    assert result.was_semantically_repaired is True
    assert result.clarified_context == "Filter by Germany"
    assert len(result.df) == 1
    assert result.df.iloc[0]["n_name"] == "GERMANY"
    assert any(rec.get("error_type") == "SEMANTIC_VALIDATION" for rec in result.attempt_history)
