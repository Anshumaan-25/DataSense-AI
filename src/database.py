"""DuckDB connection manager, TPC-H database initialization, and schema extractor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
from weakref import WeakKeyDictionary
import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from src.config import settings
from src.validator import validate_sql

logger = logging.getLogger(__name__)

# Keep profiles isolated by connection without retaining closed/discarded databases.
DistinctValueProfile = dict[str, dict[str, tuple[str, ...]]]
_VALUE_PROFILES: WeakKeyDictionary = WeakKeyDictionary()

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

    # Read-only profiling also works when initializing an existing, locked-down database.
    profile_distinct_values(active_conn)

    return active_conn


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def profile_distinct_values(
    conn: duckdb.DuckDBPyConnection,
    max_distinct_values: int = 100,
    max_value_length: int = 256,
) -> DistinctValueProfile:
    """Refresh a connection's low-cardinality text values for semantic casing checks.

    At most threshold + 1 distinct values are fetched per column. High-cardinality
    columns and oversized strings are excluded from the in-memory profile.
    """
    if max_distinct_values < 1 or max_value_length < 1:
        raise ValueError("Profiling limits must be positive.")

    columns = conn.execute("""
        SELECT c.table_name, c.column_name
        FROM information_schema.columns c
        JOIN information_schema.tables t
          ON c.table_catalog = t.table_catalog
         AND c.table_schema = t.table_schema AND c.table_name = t.table_name
        WHERE c.table_catalog = current_database() AND c.table_schema = 'main'
          AND t.table_type = 'BASE TABLE'
          AND (c.data_type LIKE 'VARCHAR%' OR c.data_type LIKE 'CHAR%'
               OR c.data_type = 'TEXT')
        ORDER BY c.table_name, c.ordinal_position
    """).fetchall()

    profile: DistinctValueProfile = {}
    for table_name, column_name in columns:
        table_sql = _quote_identifier(table_name)
        column_sql = _quote_identifier(column_name)
        try:
            rows = conn.execute(
                f'SELECT DISTINCT {column_sql} FROM "main".{table_sql} '
                f'WHERE {column_sql} IS NOT NULL LIMIT ?',
                [max_distinct_values + 1],
            ).fetchall()
        except duckdb.Error as exc:
            logger.warning("Could not profile %s.%s: %s", table_name, column_name, exc)
            continue
        if len(rows) > max_distinct_values:
            continue
        values = tuple(sorted(row[0] for row in rows if len(row[0]) <= max_value_length))
        if values:
            profile.setdefault(table_name.lower(), {})[column_name.lower()] = values

    _VALUE_PROFILES[conn] = profile
    return profile


def get_distinct_value_profile(conn: duckdb.DuckDBPyConnection) -> DistinctValueProfile:
    """Reuse initialization profiling, or lazily profile a caller-supplied database."""
    if conn not in _VALUE_PROFILES:
        return profile_distinct_values(conn)
    return _VALUE_PROFILES[conn]


def get_query_sample_rows(
    sql_query: str,
    conn: duckdb.DuckDBPyConnection,
    row_limit: int = 3,
    allowed_tables: Optional[set[str] | list[str]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Sample only authorized physical tables referenced by a validated query.

    Generated SQL is parsed, never executed here. Quoted SELECTs against known
    main-schema tables return at most three rows per table, with bounded strings.
    CTE aliases and table functions are not sampled as physical tables.
    """
    allowed = {name.lower() for name in (TPCH_TABLES if allowed_tables is None else allowed_tables)}
    is_valid, _ = validate_sql(sql_query, allowed_tables=allowed)
    if not is_valid or row_limit < 1:
        return {}
    tree = sqlglot.parse_one(sql_query, read="duckdb")
    referenced = {
        source.name.lower()
        for scope in traverse_scope(tree)
        for source in scope.sources.values()
        if isinstance(source, exp.Table) and source.name.lower() in allowed
        and not source.catalog and source.db.lower() in ("", "main")
    }
    known_tables = get_schema_info(conn=conn)
    samples: dict[str, list[dict[str, Any]]] = {}
    for table_name in sorted(referenced.intersection(known_tables)):
        try:
            frame = conn.execute(
                f'SELECT * FROM "main".{_quote_identifier(table_name)} LIMIT ?',
                [min(row_limit, 3)],
            ).df()
        except duckdb.Error as exc:
            logger.warning("Could not sample %s: %s", table_name, exc)
            continue
        records = frame.to_dict(orient="records")
        for record in records:
            for key, value in record.items():
                if isinstance(value, (str, bytes, bytearray)) and len(value) > 256:
                    record[key] = str(value[:256]) + "…"
        samples[table_name] = records
    return samples


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
