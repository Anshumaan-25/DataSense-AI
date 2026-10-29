"""Tests for SQL generator extraction and fallback behavior."""

import pytest
from src.generator import extract_sql_from_response, generate_sql


def test_extract_sql_from_markdown_block():
    """Verify clean SQL extraction from ```sql ... ``` fences."""
    raw = """
    Here is the query:
    ```sql
    SELECT c_name, c_acctbal
    FROM customer
    ORDER BY c_acctbal DESC
    LIMIT 5;
    ```
    I hope this helps!
    """
    extracted = extract_sql_from_response(raw)
    assert extracted.startswith("SELECT c_name")
    assert extracted.endswith("LIMIT 5;")
    assert "Here is the query" not in extracted


def test_extract_sql_from_generic_code_block():
    """Verify extraction from generic ``` code blocks."""
    raw = "```\nSELECT * FROM orders\n```"
    extracted = extract_sql_from_response(raw)
    assert extracted == "SELECT * FROM orders"


def test_extract_sql_plain_text():
    """Verify plain SQL without code fences is extracted cleanly."""
    raw = "SELECT count(*) FROM lineitem;"
    extracted = extract_sql_from_response(raw)
    assert extracted == "SELECT count(*) FROM lineitem;"


def test_generate_sql_missing_keys_raises(monkeypatch):
    """Verify generate_sql raises ValueError when no API keys are configured."""
    from src import generator
    from src.config import Settings
    monkeypatch.setattr(generator, "settings", Settings(gemini_api_key="", openai_api_key="", groq_api_key=""))
    with pytest.raises(ValueError) as exc_info:
        generate_sql("Who are the top customers?", "CREATE TABLE customer (c_custkey BIGINT);")
    assert "API key" in str(exc_info.value)


@pytest.mark.parametrize("provider", ["gemini", "openai", "groq"])
def test_repair_prompt_includes_samples_and_preserves_grounded_context(monkeypatch, provider):
    from src import generator
    from src.config import Settings

    monkeypatch.setattr(generator, "settings", Settings(gemini_api_key="test", openai_api_key="test", groq_api_key="test"))
    prompts = []

    def fake_provider(**kwargs):
        prompts.append(kwargs)
        return "```sql\nSELECT raw_value FROM orders\n```"

    monkeypatch.setattr(generator, f"_call_{provider}", fake_provider)
    sql = generator.generate_repaired_sql(
        user_question="Show order values", schema_info="CREATE TABLE orders (raw_value VARCHAR)",
        rag_context="Use net_revenue formula", clarified_context="Keep original currency text",
        failed_sql="SELECT CAST(raw_value AS INTEGER) FROM orders", error_message="Conversion Error",
        attempt_history=[{"attempt": 1, "sql": "SELECT CAST(raw_value AS INTEGER) FROM orders", "error_type": "DUCKDB_RUNTIME", "error": "Conversion Error"}],
        provider=provider, table_samples={"orders": [{"raw_value": "USD 20"}]},
    )
    assert sql == "SELECT raw_value FROM orders"
    prompt = prompts[0]["user_prompt"]
    assert "TABLE SAMPLE ROWS" in prompt and '"raw_value": "USD 20"' in prompt
    assert "Use net_revenue formula" in prompt and "Keep original currency text" in prompt
    assert "PREVIOUS FAILED ATTEMPTS" in prompt and "Conversion Error" in prompt
    assert "untrusted data" in prompts[0]["system_prompt"]
