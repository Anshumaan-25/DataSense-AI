"""SQL AST query safety validator using sqlglot."""

from __future__ import annotations

import re
from typing import Optional, Set, Tuple
import sqlglot
from sqlglot import exp, errors

# Expressions that modify data, schema, or system configuration
FORBIDDEN_EXPRESSIONS = (
    exp.Drop,
    exp.Delete,
    exp.Update,
    exp.Insert,
    exp.Alter,
    exp.Create,
    exp.Command,
    exp.Pragma,
    exp.Attach,
    exp.Copy,
    exp.TruncateTable,
    exp.ReadCSV,
    exp.ReadParquet,
)

# Functions that perform local file system or network reads
FORBIDDEN_FUNCTION_NAMES = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "read_json",
    "read_json_auto",
    "read_ndjson",
    "read_text",
    "read_blob",
    "scan_csv",
    "scan_parquet",
    "glob",
}

# File extensions commonly used in replacement scans
FILE_EXTENSIONS = (
    ".csv",
    ".parquet",
    ".json",
    ".jsonl",
    ".ndjson",
    ".txt",
    ".tsv",
    ".duckdb",
    ".db",
    ".sqlite",
)


def validate_sql(
    sql_query: str,
    allowed_tables: Optional[Set[str] | list[str]] = None,
) -> Tuple[bool, str]:
    """Validate that a SQL query is strictly read-only and safe for execution.

    Requirements enforced:
    - Must be a valid SQL statement parseable by sqlglot (DuckDB dialect).
    - Contains exactly one statement (rejects chained multi-statement injections).
    - Must be a read-only query (SELECT, WITH ... SELECT, UNION, or Subquery).
    - Explicitly blocks any data modification or schema alteration (DROP, DELETE,
      UPDATE, INSERT, ALTER, CREATE, PRAGMA, COPY, ATTACH).
    - Strictly blocks any external file or network reads (read_csv, read_parquet,
      read_text, glob, and direct file path replacement scans like '/etc/passwd').
    - Optionally validates that referenced tables match an allowed table whitelist.

    Args:
        sql_query: The SQL query string to validate.
        allowed_tables: Optional collection of allowed table names.

    Returns:
        A tuple of (is_valid, reason_or_message).
    """
    if not sql_query or not sql_query.strip():
        return False, "Query is empty."

    cleaned_query = sql_query.strip()

    try:
        # Parse all statements in the query using DuckDB dialect
        parsed_statements = [
            stmt for stmt in sqlglot.parse(cleaned_query, read="duckdb") if stmt is not None
        ]
    except errors.ParseError as exc:
        clean_msg = re.sub(r"\x1b\[[0-9;]*[mGKH]", "", str(exc))
        return False, f"SQL syntax error: {clean_msg}"
    except Exception as exc:
        return False, f"Failed to parse SQL: {str(exc)}"

    if not parsed_statements:
        return False, "No valid SQL statement found."

    if len(parsed_statements) > 1:
        return (
            False,
            f"Multiple SQL statements detected ({len(parsed_statements)}). "
            "Only a single read-only query is permitted.",
        )

    root_stmt = parsed_statements[0]

    # Verify that root statement is a read-only query
    if not isinstance(root_stmt, (exp.Query, exp.Subquery)):
        stmt_type = type(root_stmt).__name__.upper()
        return (
            False,
            f"Prohibited statement type '{stmt_type}'. Only read-only queries (SELECT / WITH) are allowed.",
        )

    # 1. Deep check for forbidden statement types and native file read expressions
    for forbidden_type in FORBIDDEN_EXPRESSIONS:
        for match in root_stmt.find_all(forbidden_type):
            action_name = match.key.upper() if hasattr(match, "key") and match.key else type(match).__name__.upper()
            return (
                False,
                f"Prohibited operation '{action_name}' detected. Modifications and external file operations are strictly forbidden.",
            )

    # 2. Check for file-reading functions (e.g., read_csv_auto, read_text, glob)
    for func in root_stmt.find_all(exp.Func):
        fname = (func.name or "").lower()
        if fname in FORBIDDEN_FUNCTION_NAMES or fname.startswith(("read_", "scan_")):
            return (
                False,
                f"Prohibited file-reading function '{fname}' detected. Direct file system access is forbidden.",
            )

    # 3. Check for replacement scans or file paths in table identifiers
    for tbl in root_stmt.find_all(exp.Table):
        tbl_str = (tbl.name or str(tbl.this or "")).strip("\"'")
        if any(sep in tbl_str for sep in ("/", "\\")) or any(
            tbl_str.lower().endswith(ext) for ext in FILE_EXTENSIONS
        ):
            return (
                False,
                f"Prohibited file path reference '{tbl_str}' detected. Reading external files is forbidden.",
            )

    # 4. Optional Table Whitelist verification
    if allowed_tables is not None:
        normalized_allowed = {t.lower() for t in allowed_tables}

        # Extract CTE aliases so they are treated as valid table references
        cte_aliases: Set[str] = {
            cte.alias_or_name.lower()
            for cte in root_stmt.find_all(exp.CTE)
            if cte.alias_or_name
        }

        for tbl in root_stmt.find_all(exp.Table):
            tbl_name = (tbl.name or "").lower()
            if tbl_name and tbl_name not in cte_aliases and tbl_name not in normalized_allowed:
                return (
                    False,
                    f"Unauthorized table reference '{tbl_name}'. Allowed tables: {sorted(normalized_allowed)}.",
                )

    return True, "Valid SELECT query"
