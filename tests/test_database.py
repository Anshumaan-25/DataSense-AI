"""Tests for DuckDB database connection, TPC-H initialization, and schema extractor."""

import duckdb
import pytest
import pandas as pd
from src.database import (
    get_connection,
    init_tpch_db,
    tpch_tables_exist,
    get_schema_info,
    execute_query,
    TPCH_TABLES,
)


@pytest.fixture
def memory_db():
    """Create an in-memory DuckDB connection with scale factor 0.01 for fast testing."""
    conn = duckdb.connect(":memory:")
    init_tpch_db(conn=conn, scale_factor=0.01)
    return conn


def test_tpch_initialization(memory_db):
    """Verify that all standard TPC-H tables are generated in DuckDB."""
    assert tpch_tables_exist(memory_db) is True
    tables = [row[0].lower() for row in memory_db.execute("SHOW TABLES;").fetchall()]
    for expected in TPCH_TABLES:
        assert expected in tables


def test_schema_info_dict(memory_db):
    """Verify get_schema_info returns a properly structured dictionary."""
    schema = get_schema_info(conn=memory_db, as_string=False)
    assert isinstance(schema, dict)
    assert "customer" in schema
    assert "orders" in schema
    assert "lineitem" in schema

    # Check customer columns
    col_names = [col["name"] for col in schema["customer"]]
    assert "c_custkey" in col_names
    assert "c_name" in col_names
    assert "c_acctbal" in col_names


def test_schema_info_string(memory_db):
    """Verify get_schema_info as string produces valid DDL prompt format."""
    schema_str = get_schema_info(conn=memory_db, as_string=True)
    assert isinstance(schema_str, str)
    assert "CREATE TABLE customer" in schema_str
    assert "CREATE TABLE orders" in schema_str


def test_execute_query_success(memory_db):
    """Verify execute_query returns a valid DataFrame on success."""
    sql = "SELECT c_custkey, c_name, c_acctbal FROM customer ORDER BY c_acctbal DESC LIMIT 5;"
    df, err = execute_query(sql, conn=memory_db)
    assert err is None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "c_custkey" in df.columns


def test_execute_query_error(memory_db):
    """Verify execute_query gracefully handles syntax or nonexistent table errors."""
    sql = "SELECT * FROM nonexistent_table_xyz;"
    df, err = execute_query(sql, conn=memory_db)
    assert df is None
    assert err is not None
    assert "nonexistent_table_xyz" in err


def test_external_access_disabled_blocks_file_reads(memory_db):
    """Verify DuckDB C++ engine blocks file system reads when external access is disabled."""
    sql = "SELECT * FROM read_csv('/etc/passwd');"
    df, err = execute_query(sql, conn=memory_db)
    assert df is None
    assert err is not None
    assert "disabled by configuration" in err.lower() or "permission error" in err.lower()

