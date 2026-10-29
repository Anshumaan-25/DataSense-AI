"""Unit tests for Ambiguity Classifier and Clarification Pipeline."""

import pytest
import duckdb
from src.ambiguity import (
    AmbiguityClassifier,
    AmbiguityResult,
    parse_ambiguity_json,
)
from src.pipeline import SelfCorrectingPipeline, PipelineResult


@pytest.mark.parametrize("raw", ["", "not json", "[]", "{}", '{"ambiguity_type": "OTHER"}', '{"ambiguity_type": "SOFT_DEFAULT"}', '{"ambiguity_type": "HARD_AMBIGUITY"}'])
def test_invalid_classifications_are_marked_as_failures(raw):
    assert parse_ambiguity_json(raw).classification_succeeded is False


def test_json_with_surrounding_text_is_parsed_without_claiming_a_fallback():
    result = parse_ambiguity_json('Analysis: {"ambiguity_type": "CLEAR"} Done.')
    assert result.ambiguity_type == "CLEAR"
    assert result.classification_succeeded


def test_parse_ambiguity_json_valid():
    """Verify parsing of valid JSON output from LLM."""
    raw = """
    ```json
    {
      "ambiguity_type": "HARD_AMBIGUITY",
      "reasoning": "Top customers could mean by revenue, orders, or balance.",
      "clarification_prompt": "How would you like to rank top customers?",
      "options": [
        {"label": "By Net Revenue", "details": "SUM(l_extendedprice * (1 - l_discount))"},
        {"label": "By Account Balance", "details": "c_acctbal DESC"}
      ],
      "default_interpretation": "Rank by account balance descending limit 10"
    }
    ```
    """
    res = parse_ambiguity_json(raw)
    assert res.is_ambiguous is True
    assert res.ambiguity_type == "HARD_AMBIGUITY"
    assert "revenue" in res.reasoning.lower()
    assert len(res.options) == 2
    assert res.options[0]["label"] == "By Net Revenue"
    assert "SUM(l_extendedprice" in res.options[0]["details"]
    assert res.default_interpretation == "Rank by account balance descending limit 10"


def test_parse_ambiguity_json_soft_default():
    """Verify parsing of SOFT_DEFAULT sets is_ambiguous = False."""
    raw = """
    {
      "ambiguity_type": "SOFT_DEFAULT",
      "reasoning": "Standard sort and limit can be applied.",
      "clarification_prompt": "",
      "options": [],
      "default_interpretation": "ORDER BY o_orderdate DESC LIMIT 10"
    }
    """
    res = parse_ambiguity_json(raw)
    assert res.is_ambiguous is False
    assert res.ambiguity_type == "SOFT_DEFAULT"
    assert res.default_interpretation == "ORDER BY o_orderdate DESC LIMIT 10"


def test_parse_ambiguity_json_fallback_on_invalid():
    """Verify graceful fallback to CLEAR when LLM outputs broken JSON."""
    raw = "I am not sure how to answer this query properly."
    res = parse_ambiguity_json(raw)
    assert res.is_ambiguous is False
    assert res.ambiguity_type == "CLEAR"
    assert "fallback" in res.reasoning.lower() or "empty" in res.reasoning.lower()


def test_clear_intent_classification():
    """Verify unambiguous query is classified as CLEAR."""
    mock_response = """
    {
      "ambiguity_type": "CLEAR",
      "reasoning": "Explicit customer key and metric requested.",
      "clarification_prompt": "",
      "options": [],
      "default_interpretation": ""
    }
    """
    classifier = AmbiguityClassifier(classifier_fn=lambda prompt: mock_response)
    result = classifier.classify_intent(
        user_question="What is the account balance of customer#000000001?",
        schema_info="CREATE TABLE customer (c_custkey BIGINT, c_acctbal DOUBLE);",
    )
    assert result.is_ambiguous is False
    assert result.ambiguity_type == "CLEAR"


def test_soft_default_classification():
    """Verify mildly underspecified query is classified as SOFT_DEFAULT without user interruption."""
    mock_response = """
    {
      "ambiguity_type": "SOFT_DEFAULT",
      "reasoning": "Unspecified row limit and ordering, safe default exists.",
      "clarification_prompt": "",
      "options": [],
      "default_interpretation": "SELECT * FROM orders ORDER BY o_orderdate DESC LIMIT 10;"
    }
    """
    classifier = AmbiguityClassifier(classifier_fn=lambda prompt: mock_response)
    result = classifier.classify_intent(
        user_question="Show orders",
        schema_info="CREATE TABLE orders (o_orderkey BIGINT, o_orderdate DATE);",
    )
    assert result.is_ambiguous is False
    assert result.ambiguity_type == "SOFT_DEFAULT"
    assert "LIMIT 10" in result.default_interpretation


def test_hard_ambiguity_classification():
    """Verify conflicting business metrics trigger HARD_AMBIGUITY with catalog options."""
    mock_response = """
    {
      "ambiguity_type": "HARD_AMBIGUITY",
      "reasoning": "Top customers has multiple divergent meanings.",
      "clarification_prompt": "How would you like to rank the top customers?",
      "options": [
        {"label": "By Net Revenue", "details": "SUM(l_extendedprice * (1 - l_discount))"},
        {"label": "By Total Orders", "details": "COUNT(o_orderkey)"},
        {"label": "By Account Balance", "details": "c_acctbal"}
      ],
      "default_interpretation": "Rank by c_acctbal DESC LIMIT 5"
    }
    """
    classifier = AmbiguityClassifier(classifier_fn=lambda prompt: mock_response)
    result = classifier.classify_intent(
        user_question="Show top customers",
        schema_info="CREATE TABLE customer (c_custkey BIGINT, c_name VARCHAR, c_acctbal DOUBLE);",
        rag_context="net_revenue: SUM(l_extendedprice * (1 - l_discount))",
    )
    assert result.is_ambiguous is True
    assert result.ambiguity_type == "HARD_AMBIGUITY"
    assert len(result.options) == 3
    assert result.options[0]["label"] == "By Net Revenue"


def test_fallback_on_classifier_exception():
    """Verify that an exception in classifier execution triggers safe fallback to CLEAR."""
    def broken_fn(prompt):
        raise RuntimeError("API quota exhausted or network error")

    classifier = AmbiguityClassifier(classifier_fn=broken_fn)
    result = classifier.classify_intent(
        user_question="Any question",
        schema_info="CREATE TABLE customer (c_custkey BIGINT);",
    )
    assert result.is_ambiguous is False
    assert result.ambiguity_type == "CLEAR"
    assert "fallback" in result.reasoning.lower()


def test_pipeline_check_ambiguity_integration():
    """Verify SelfCorrectingPipeline.check_ambiguity integration."""
    mock_response = """
    {
      "ambiguity_type": "HARD_AMBIGUITY",
      "reasoning": "Unspecified metric for parts.",
      "clarification_prompt": "How would you like to rank parts?",
      "options": [{"label": "By Price", "details": "p_retailprice"}],
      "default_interpretation": "ORDER BY p_retailprice DESC LIMIT 10"
    }
    """
    classifier = AmbiguityClassifier(classifier_fn=lambda prompt: mock_response)
    pipeline = SelfCorrectingPipeline(
        db_conn=duckdb.connect(":memory:"),
        classifier=classifier,
    )

    amb_res = pipeline.check_ambiguity("Which parts sell best?")
    assert amb_res.is_ambiguous is True
    assert amb_res.ambiguity_type == "HARD_AMBIGUITY"


def test_pipeline_preserves_clarified_context_across_repair_loop():
    """Verify clarified_context is passed to generator and preserved across repair retries."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE customer (c_custkey BIGINT, c_name VARCHAR, c_acctbal DOUBLE);")
    conn.execute("INSERT INTO customer VALUES (1, 'Alice', 5000.0), (2, 'Bob', 3000.0);")

    captured_clarifications = []

    def staged_generator(user_question, schema_str, rag_context="", clarified_context="", provider=None):
        captured_clarifications.append(("GEN", clarified_context))
        # Deliberate column typo to trigger attempt 2
        return "SELECT c_custkey, c_namee FROM customer LIMIT 5;"

    def staged_repair(user_question, schema_info, rag_context="", failed_sql="", error_message="", attempt_history=None, clarified_context="", provider=None):
        captured_clarifications.append(("REPAIR", clarified_context))
        # Correct the typo
        return "SELECT c_custkey, c_name FROM customer LIMIT 5;"

    pipeline = SelfCorrectingPipeline(
        db_conn=conn,
        generator_fn=staged_generator,
        repair_fn=staged_repair,
    )

    user_clarification = "Rank by Account Balance (c_acctbal DESC)"
    result = pipeline.execute_with_repair(
        user_question="Show top customers",
        max_retries=3,
        clarified_context=user_clarification,
    )

    assert result.success is True
    assert result.total_attempts == 2
    assert result.clarified_context == user_clarification
    assert ("GEN", user_clarification) in captured_clarifications
    assert ("REPAIR", user_clarification) in captured_clarifications
