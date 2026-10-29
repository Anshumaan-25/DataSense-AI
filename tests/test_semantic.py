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
    validator = SemanticValidator(distinct_value_profile={"nation": {"n_name": ("GERMANY",)}})
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


@pytest.mark.parametrize("sql", [
    "SELECT status FROM orders WHERE status = 'readytoship'",
    "SELECT o.status FROM orders o WHERE o.status IN ('readytoship')",
    "SELECT status FROM orders WHERE 'readytoship' = status",
    "WITH x AS (SELECT status FROM orders WHERE status = 'readytoship') SELECT * FROM x",
])
def test_dynamic_casing_uses_mixed_case_database_values(sql):
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE orders AS SELECT 'ReadyToShip' AS status")
        validator = SemanticValidator(db_conn=conn)
        result = validator.validate_output("Show ready orders", sql, conn.execute(sql).df())
        assert result.issue_type == "CASING_MISMATCH"
        assert "ReadyToShip" in result.suggested_fix


@pytest.mark.parametrize("sql", [
    "SELECT status FROM orders WHERE status = 'unknown'",
    "SELECT status FROM orders WHERE description = 'readytoship'",
    "SELECT 'readytoship' FROM orders WHERE 1 = 0",
    "SELECT status FROM orders WHERE LOWER(status) = 'readytoship' AND 1 = 0",
    "SELECT status FROM orders WHERE status NOT IN ('readytoship') AND 1 = 0",
    "SELECT status FROM orders WHERE status = 'ReadyToShip' AND 1 = 0",
])
def test_dynamic_casing_does_not_guess_unrelated_or_valid_literals(sql):
    profile = {"orders": {"status": ("ReadyToShip",), "description": ("Other",)}}
    result = SemanticValidator(distinct_value_profile=profile).validate_output(
        "Show orders", sql, pd.DataFrame(columns=["status"]),
    )
    assert result.is_valid


def test_case_collisions_are_not_guessed():
    validator = SemanticValidator(distinct_value_profile={"orders": {"status": ("Ready", "READY")}})
    result = validator.validate_output(
        "Show ready orders", "SELECT status FROM orders WHERE status = 'ready'", pd.DataFrame(columns=["status"]),
    )
    assert result.is_valid


def test_casing_detects_null_scalar_aggregate_with_escaped_literal():
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE orders AS SELECT 'O''Reilly' AS publisher, 10 AS amount")
        sql = "SELECT SUM(amount) FROM orders WHERE publisher = 'o''reilly'"
        result = SemanticValidator(db_conn=conn).validate_output("Total amount", sql, conn.execute(sql).df())
        assert result.issue_type == "CASING_MISMATCH"
        assert "O'Reilly" in result.suggested_fix


@pytest.mark.parametrize("metric,expression", [
    ("net_revenue", "SUM(l.l_extendedprice * (1.0 - l.l_discount))"),
    ("net_revenue", "ROUND(SUM((1 - l.l_discount) * l.l_extendedprice), 2)"),
    ("gross_revenue", "SUM(l.l_extendedprice)"),
    ("discount_loss", "SUM(l.l_discount * l.l_extendedprice)"),
    ("tax_amount", "SUM(l.l_tax * ((1 - l.l_discount) * l.l_extendedprice))"),
    ("item_margin", "SUM(l.l_extendedprice * (1 - l.l_discount) - l.l_quantity * ps.ps_supplycost)"),
])
def test_catalog_formulas_accept_aliases_wrappers_and_commutative_operands(metric, expression):
    validator = SemanticValidator()
    definition = next(item for item in validator.metrics if item["id"] == metric)
    result = validator.validate_output(
        f"What is the {definition['name']}?", f"SELECT {expression} AS result FROM lineitem l JOIN partsupp ps ON l.l_partkey = ps.ps_partkey",
        pd.DataFrame({"result": [10.0]}), rag_context=metric,
    )
    assert result.is_valid, result.reasoning


@pytest.mark.parametrize("sql", [
    "WITH revenue AS (SELECT SUM(l.l_extendedprice * (1 - l.l_discount)) AS amount FROM lineitem l) SELECT r.amount FROM revenue r",
    "SELECT r.amount FROM (SELECT SUM(l.l_extendedprice * (1 - l.l_discount)) AS amount FROM lineitem l) r",
    "WITH revenue(amount) AS (SELECT SUM(l.l_extendedprice * (1 - l.l_discount)) FROM lineitem l) SELECT * FROM revenue",
])
def test_metric_formulas_follow_output_lineage(sql):
    result = SemanticValidator().validate_output(
        "What is net revenue?", sql, pd.DataFrame({"revenue": [10.0]}), rag_context="net_revenue",
    )
    assert result.is_valid, result.reasoning


@pytest.mark.parametrize("sql", [
    "SELECT SUM(l_extendedprice * (1 + l_discount)) FROM lineitem",
    "SELECT SUM(l_extendedprice * (1 - l_discount)) + 1 FROM lineitem",
    "SELECT SUM(l_extendedprice) FROM lineitem -- SUM(l_extendedprice * (1 - l_discount))",
    "SELECT SUM(l_extendedprice), 'SUM(l_extendedprice * (1 - l_discount))' AS decoy FROM lineitem",
    "WITH unused AS (SELECT SUM(l_extendedprice * (1 - l_discount)) AS total FROM lineitem) SELECT SUM(l_extendedprice) FROM lineitem",
    "SELECT SUM(l_extendedprice) FROM lineitem WHERE (SELECT SUM(l_extendedprice * (1 - l_discount)) FROM lineitem) > 0",
    "WITH revenue AS (SELECT SUM(l_extendedprice * (1 - l_discount)) AS total FROM lineitem) SELECT total + 1 FROM revenue",
])
def test_metric_check_rejects_wrong_arithmetic_and_non_output_decoys(sql):
    result = SemanticValidator().validate_output(
        "What is net revenue?", sql, pd.DataFrame({"result": [10.0]}), rag_context="net_revenue",
    )
    assert not result.is_valid
    assert result.issue_type == "CATALOG_METRIC_OMISSION"


@pytest.mark.parametrize("question,clarification,sql", [
    ("What is gross revenue?", "", "SELECT SUM(l_extendedprice) FROM lineitem"),
    ("How many orders?", "", "SELECT COUNT(*) FROM orders"),
    ("What is net revenue?", "Use gross revenue", "SELECT SUM(l_extendedprice) FROM lineitem"),
    ("Compare net revenue and gross revenue", "", "SELECT SUM(l_extendedprice * (1 - l_discount)), SUM(l_extendedprice) FROM lineitem"),
])
def test_retrieval_candidates_are_not_all_mandatory_metrics(question, clarification, sql):
    result = SemanticValidator().validate_output(
        question, sql, pd.DataFrame({"result": [10.0]}),
        rag_context="net_revenue gross_revenue", clarified_context=clarification,
    )
    assert result.is_valid, result.reasoning


def test_ast_metric_validation_uses_retrieved_custom_formula():
    metadata = [{"kind": "metric", "id": "freight_cost", "name": "Freight Cost", "formula": "SUM(shipping_cost)"}]
    validator = SemanticValidator()
    for sql, expected in (("SELECT SUM(o.shipping_cost) FROM orders o", True), ("SELECT COUNT(*) FROM orders", False)):
        result = validator.validate_output(
            "What is freight cost?", sql, pd.DataFrame({"cost": [10.0]}), rag_metadata=metadata,
        )
        assert result.is_valid is expected


@pytest.mark.parametrize("response,status,valid", [
    ('{"is_valid": true}', "PASSED", True),
    ('{"is_valid": false}', "REJECTED", False),
    ('{"is_valid": "false"}', "ERROR", True),
    ('not json', "ERROR", True),
])
def test_semantic_judge_status_distinguishes_verdicts_from_fallbacks(response, status, valid):
    result = SemanticValidator(enable_llm_judge=True, judge_fn=lambda prompt: response).validate_output(
        "How many orders?", "SELECT COUNT(*) FROM orders", pd.DataFrame({"count": [2]}),
    )
    assert result.is_valid is valid
    assert result.judge_status == status
