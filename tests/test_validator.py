"""Tests for sqlglot AST query safety validator."""

import pytest
from src.validator import validate_sql


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM customer LIMIT 10;",
        "SELECT c_name, c_acctbal FROM customer WHERE c_acctbal > 1000 ORDER BY c_acctbal DESC;",
        "SELECT o_orderpriority, count(*) as order_count FROM orders GROUP BY o_orderpriority;",
        "WITH high_bal AS (SELECT * FROM customer WHERE c_acctbal > 5000) SELECT * FROM high_bal;",
        "SELECT c_custkey FROM customer UNION ALL SELECT o_custkey FROM orders;",
        "(SELECT 1 as id, 'test' as name);",
    ],
)
def test_validate_sql_valid_queries(query):
    """Verify that read-only queries pass validation."""
    is_valid, msg = validate_sql(query)
    assert is_valid is True
    assert "Valid SELECT query" in msg


@pytest.mark.parametrize(
    "prohibited_query, expected_keyword",
    [
        ("DROP TABLE customer;", "DROP"),
        ("DROP TABLE IF EXISTS orders;", "DROP"),
        ("DELETE FROM customer WHERE c_acctbal < 0;", "DELETE"),
        ("UPDATE customer SET c_acctbal = 0;", "UPDATE"),
        ("INSERT INTO customer VALUES (1, 'Test', 'Addr', 1, '123', 0.0, 'MKT', 'Note');", "INSERT"),
        ("ALTER TABLE customer ADD COLUMN is_vip BOOLEAN;", "ALTER"),
        ("CREATE TABLE test_table (id INT);", "CREATE"),
        ("PRAGMA version;", "PRAGMA"),
        ("ATTACH ':memory:';", "ATTACH"),
        ("COPY customer TO 'customer.csv';", "COPY"),
    ],
)
def test_validate_sql_blocks_prohibited_statements(prohibited_query, expected_keyword):
    """Verify that destructive or administrative SQL operations are rejected."""
    is_valid, msg = validate_sql(prohibited_query)
    assert is_valid is False
    assert expected_keyword in msg.upper()


def test_validate_sql_blocks_multi_statements():
    """Verify that chained query attacks are detected and blocked."""
    chained_query = "SELECT * FROM customer; DROP TABLE orders;"
    is_valid, msg = validate_sql(chained_query)
    assert is_valid is False
    assert "Multiple SQL statements" in msg


def test_validate_sql_empty():
    """Verify empty queries return invalid."""
    is_valid, msg = validate_sql("   ")
    assert is_valid is False
    assert "empty" in msg.lower()


def test_validate_sql_syntax_error():
    """Verify syntax errors return invalid with syntax description."""
    is_valid, msg = validate_sql("SELECT FROM WHERE ;")
    assert is_valid is False
    assert "syntax error" in msg.lower() or "failed to parse" in msg.lower()


@pytest.mark.parametrize(
    "file_read_query, keyword",
    [
        ("SELECT * FROM read_csv('/etc/passwd');", "read"),
        ("SELECT * FROM read_csv_auto('/etc/shadow');", "read_csv_auto"),
        ("SELECT * FROM read_parquet('data/*.parquet');", "read"),
        ("SELECT * FROM read_json('config.json');", "read_json"),
        ("SELECT * FROM read_text('/etc/hosts');", "read_text"),
        ("SELECT * FROM glob('/*');", "glob"),
        ("SELECT read_text('/etc/passwd');", "read_text"),
    ],
)
def test_validate_sql_blocks_file_reading_functions(file_read_query, keyword):
    """Verify that AST validator blocks table functions that read local files."""
    is_valid, msg = validate_sql(file_read_query)
    assert is_valid is False
    assert (
        "forbidden" in msg.lower()
        or "prohibited" in msg.lower()
        or keyword.lower() in msg.lower()
    )


@pytest.mark.parametrize(
    "file_path_query",
    [
        "SELECT * FROM 'data/test.parquet';",
        "SELECT * FROM '/etc/passwd';",
        "SELECT * FROM './data/tpch.duckdb';",
        "SELECT * FROM 'users.csv';",
    ],
)
def test_validate_sql_blocks_file_path_replacement_scans(file_path_query):
    """Verify that AST validator blocks direct file path replacement scans."""
    is_valid, msg = validate_sql(file_path_query)
    assert is_valid is False
    assert "file path" in msg.lower() or "prohibited" in msg.lower()


def test_validate_sql_table_whitelist():
    """Verify that table whitelist allows legitimate tables and blocks unauthorized ones."""
    allowed = ["customer", "orders"]

    # Allowed table
    ok1, _ = validate_sql("SELECT * FROM customer;", allowed_tables=allowed)
    assert ok1 is True

    # CTE with allowed table
    ok2, _ = validate_sql(
        "WITH temp_cte AS (SELECT * FROM customer) SELECT * FROM temp_cte;",
        allowed_tables=allowed,
    )
    assert ok2 is True

    # Unauthorized table
    ok3, msg3 = validate_sql("SELECT * FROM lineitem;", allowed_tables=allowed)
    assert ok3 is False
    assert "unauthorized table" in msg3.lower()

