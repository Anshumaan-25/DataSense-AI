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
    get_distinct_value_profile,
    profile_distinct_values,
    get_query_sample_rows,
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


def test_initialization_registers_profile_after_lockdown(memory_db, monkeypatch):
    from src import database

    monkeypatch.setattr(database, "profile_distinct_values", lambda *a, **kw: pytest.fail("profile rescanned"))
    profile = get_distinct_value_profile(memory_db)
    assert "GERMANY" in profile["nation"]["n_name"]
    assert "1-URGENT" in profile["orders"]["o_orderpriority"]
    assert memory_db.execute("SELECT current_setting('enable_external_access')").fetchone() == (False,)


def test_profiling_excludes_high_cardinality_nulls_and_long_values():
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE events (status VARCHAR, description VARCHAR, note VARCHAR, number INTEGER)")
        conn.execute("INSERT INTO events VALUES ('Ready', 'one', NULL, 1), ('Ready', 'two', 'long text', 2), ('Done', 'three', NULL, 3)")
        profile = profile_distinct_values(conn, max_distinct_values=2, max_value_length=5)
        assert profile == {"events": {"status": ("Done", "Ready")}}


def test_profiles_are_isolated_and_can_be_refreshed():
    with duckdb.connect(":memory:") as first, duckdb.connect(":memory:") as second:
        for conn, value in ((first, "Ready"), (second, "READY")):
            conn.execute("CREATE TABLE events (status VARCHAR)")
            conn.execute("INSERT INTO events VALUES (?)", [value])
        assert get_distinct_value_profile(first)["events"]["status"] == ("Ready",)
        assert get_distinct_value_profile(second)["events"]["status"] == ("READY",)
        first.execute("UPDATE events SET status = 'Done'")
        assert profile_distinct_values(first)["events"]["status"] == ("Done",)
        assert get_distinct_value_profile(second)["events"]["status"] == ("READY",)


def test_profiling_quotes_identifiers():
    with duckdb.connect(":memory:") as conn:
        conn.execute('CREATE TABLE "odd""table" ("odd""column" VARCHAR)')
        conn.execute('INSERT INTO "odd""table" VALUES (\'MixedCase\')')
        assert profile_distinct_values(conn) == {'odd"table': {'odd"column': ("MixedCase",)}}


@pytest.mark.parametrize("limits", [{"max_distinct_values": 0}, {"max_value_length": 0}])
def test_profiling_rejects_invalid_limits(limits):
    with duckdb.connect(":memory:") as conn, pytest.raises(ValueError):
        profile_distinct_values(conn, **limits)


def test_sample_rows_are_bounded_and_only_include_referenced_physical_tables():
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE orders AS SELECT i AS order_id, repeat('x', 500) AS note FROM range(10) t(i)")
        conn.execute("CREATE TABLE nation AS SELECT 'GERMANY' AS n_name")
        samples = get_query_sample_rows("SELECT * FROM orders", conn, row_limit=100)
        assert set(samples) == {"orders"}
        assert len(samples["orders"]) == 3
        assert len(samples["orders"][0]["note"]) == 257
        # A CTE named orders must not cause the unrelated physical orders table to be sampled.
        samples = get_query_sample_rows("WITH orders AS (SELECT * FROM nation) SELECT * FROM orders", conn)
        assert samples == {"nation": [{"n_name": "GERMANY"}]}


@pytest.mark.parametrize("sql", [
    "DROP TABLE orders", "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM orders; DROP TABLE orders", "SELECT * FROM private_data",
])
def test_sampling_rejects_unsafe_or_unauthorized_queries(sql):
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE orders (id INTEGER)")
        conn.execute("CREATE TABLE private_data AS SELECT 'secret' AS value")
        assert get_query_sample_rows(sql, conn) == {}
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone() == (0,)


def test_profiling_and_sampling_support_read_only_connections(tmp_path):
    path = str(tmp_path / "readonly.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE nation AS SELECT 'MixedCase' AS n_name")
    with duckdb.connect(path, read_only=True) as conn:
        conn.execute("SET enable_external_access = false")
        assert profile_distinct_values(conn)["nation"]["n_name"] == ("MixedCase",)
        assert get_query_sample_rows("SELECT * FROM nation", conn) == {"nation": [{"n_name": "MixedCase"}]}

