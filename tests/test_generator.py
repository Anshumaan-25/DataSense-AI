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


def test_generate_sql_missing_keys_raises():
    """Verify generate_sql raises ValueError when no API keys are configured."""
    with pytest.raises(ValueError) as exc_info:
        generate_sql("Who are the top customers?", "CREATE TABLE customer (c_custkey BIGINT);")
    assert "API key" in str(exc_info.value)
