"""DuckDB connection manager, TPC-H database initialization, and schema extractor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
import duckdb
import pandas as pd

from src.config import settings

# Global connection cache for singleton usage
_GLOBAL_CONN: Optional[duckdb.DuckDBPyConnection] = None

TPCH_TABLES = [
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]


def get_connection(
    db_path: Optional[str] = None, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """Get or initialize a DuckDB connection.

    Args:
        db_path: Path to the database file or ':memory:'. Defaults to settings.duckdb_path.
        read_only: Connect in read-only mode if True.

    Returns:
        DuckDBPyConnection instance.
    """
    global _GLOBAL_CONN

    target_path = db_path if db_path is not None else settings.duckdb_path

    # In-memory connections shouldn't use disk directories
    if target_path != ":memory:":
        target_path_obj = Path(target_path).resolve()
        target_path_obj.parent.mkdir(parents=True, exist_ok=True)
        # If an empty placeholder file exists, remove it so DuckDB can initialize cleanly
        if target_path_obj.is_file() and target_path_obj.stat().st_size == 0:
            target_path_obj.unlink()

    if _GLOBAL_CONN is None or target_path == ":memory:":
        conn = duckdb.connect(database=target_path, read_only=read_only)
        if target_path != ":memory:":
            _GLOBAL_CONN = conn
        return conn

    return _GLOBAL_CONN


def tpch_tables_exist(conn: duckdb.DuckDBPyConnection) -> bool:
    """Check if all standard TPC-H tables exist in the database."""
    try:
        query = """
        SELECT LOWER(table_name)
        FROM information_schema.tables
        WHERE table_schema = 'main';
        """
        existing = {row[0] for row in conn.execute(query).fetchall()}
        return all(table in existing for table in TPCH_TABLES)
    except Exception:
        return False


def init_tpch_db(
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    scale_factor: Optional[float] = None,
    force_rebuild: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB with TPC-H benchmark data if not already populated.

    Args:
        conn: Optional existing connection. If None, uses get_connection().
        scale_factor: TPC-H scale factor (default from settings: 0.1).
        force_rebuild: If True, regenerates data even if tables exist.

    Returns:
        The DuckDB connection with TPC-H tables populated.
    """
    active_conn = conn if conn is not None else get_connection()
    sf = scale_factor if scale_factor is not None else settings.tpch_scale_factor

    if force_rebuild or not tpch_tables_exist(active_conn):
        # Install and load tpch extension, then generate dataset
        active_conn.execute("INSTALL tpch;")
        active_conn.execute("LOAD tpch;")
        active_conn.execute(f"CALL dbgen(sf={sf});")

    # Lock down external filesystem and network access for zero-trust security
    active_conn.execute("SET enable_external_access = false;")

    return active_conn


def get_schema_info(
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    as_string: bool = False,
) -> dict[str, list[dict[str, str]]] | str:
    """Extract table names, column names, and data types from the database.

    Args:
        conn: DuckDB connection. Defaults to active connection.
        as_string: If True, returns a formatted prompt-ready DDL string.

    Returns:
        Structured dictionary of schema or formatted prompt string.
    """
    active_conn = conn if conn is not None else get_connection()

    query = """
    SELECT
        table_name,
        column_name,
        data_type
    FROM information_schema.columns
    WHERE table_schema = 'main'
    ORDER BY table_name, ordinal_position;
    """
    rows = active_conn.execute(query).fetchall()

    schema_dict: dict[str, list[dict[str, str]]] = {}
    for table_name, column_name, data_type in rows:
        table_key = table_name.lower()
        if table_key not in schema_dict:
            schema_dict[table_key] = []
        schema_dict[table_key].append({
            "name": column_name.lower(),
            "type": str(data_type).upper(),
        })

    if as_string:
        return format_schema_for_prompt(schema_dict)

    return schema_dict


def format_schema_for_prompt(schema_dict: dict[str, list[dict[str, str]]]) -> str:
    """Convert structured schema dictionary into concise DDL representation for LLM prompts."""
    lines: list[str] = []
    lines.append("-- DuckDB TPC-H Database Schema:")
    for table_name, columns in sorted(schema_dict.items()):
        col_defs = ", ".join(f"{c['name']} {c['type']}" for c in columns)
        lines.append(f"CREATE TABLE {table_name} ({col_defs});")
    return "\n".join(lines)


def execute_query(
    sql_query: str,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Execute a SQL query against DuckDB and return a DataFrame or an error message.

    Args:
        sql_query: SQL string to execute.
        conn: Optional DuckDB connection. Defaults to get_connection().

    Returns:
        Tuple of (DataFrame, None) on success, or (None, error_message) on failure.
    """
    active_conn = conn if conn is not None else get_connection()

    try:
        df = active_conn.execute(sql_query).df()
        return df, None
    except Exception as exc:
        return None, str(exc)
