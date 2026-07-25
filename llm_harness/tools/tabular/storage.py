"""Tabular storage and fingerprinting helpers."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, cast

SQLITE_FILENAME = "tabular.sqlite"
SQLITE_CONTENTS_TABLE = "_tabular_contents"
SQLITE_SOURCES_TABLE = "_tabular_sources"
INSERT_BATCH_SIZE = 1000
LOCK_POLL_SECONDS = 0.1
LOCK_TIMEOUT_SECONDS = 10.0
DEFAULT_ROOT_DIR = Path.cwd().resolve()
DATE_PATTERNS = ("%Y-%m-%d", "%Y/%m/%d")
DATETIME_PATTERNS = (
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S",
)
INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
REAL_PATTERN = re.compile(r"^[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$")
SQLITE_SOURCES_COLUMNS = (
    "source_path",
    "source_format",
    "source_sheet_name",
    "source_table_name",
    "fingerprint",
    "content_id",
)
SQLITE_SOURCES_UNIQUE_COLUMNS = (
    "source_path",
    "source_sheet_name",
    "source_table_name",
)
SQLITE_SOURCES_COLUMN_SET = frozenset(SQLITE_SOURCES_COLUMNS)


def _tabular_dimensions(rows: list[list[str]]) -> tuple[int, int]:
    """Return the row and column counts for loaded rows."""
    return len(rows), max((len(row) for row in rows), default=0)


def _update_hash_rows(
    hasher: hashlib._Hash,
    rows: Iterable[list[str]],
    *,
    row_limit: int,
) -> None:
    """Update a hash from a bounded set of normalized rows."""
    for row in islice(rows, row_limit):
        for cell in row:
            hasher.update(cell.encode("utf-8"))
            hasher.update(b"\x1f")
        hasher.update(b"\x1e")


def fingerprint_from_samples(
    *,
    row_count: int,
    column_count: int,
    top_rows: list[list[str]],
    bottom_rows: list[list[str]],
    header_candidates: list[dict[str, Any]],
    max_sample_rows: int,
) -> str:
    """Build a cheap deterministic fingerprint from bounded samples."""
    hasher = hashlib.sha256()
    hasher.update(f"rows:{row_count}|cols:{column_count}".encode())
    hasher.update(b"\x1b")
    _update_hash_rows(
        hasher,
        (candidate["values"] for candidate in header_candidates),
        row_limit=max_sample_rows,
    )
    hasher.update(b"\x1d")
    _update_hash_rows(hasher, top_rows, row_limit=max_sample_rows)
    if bottom_rows:
        hasher.update(b"\x1c")
        _update_hash_rows(hasher, bottom_rows, row_limit=max_sample_rows)
    return hasher.hexdigest()


def fingerprint(
    rows: list[list[str]],
    *,
    max_sample_rows: int,
    header_candidates: list[dict[str, Any]],
) -> str:
    """Build a cheap deterministic fingerprint for routing and cache hints."""
    row_count, column_count = _tabular_dimensions(rows)
    return fingerprint_from_samples(
        row_count=row_count,
        column_count=column_count,
        top_rows=rows[:max_sample_rows],
        bottom_rows=rows[-max_sample_rows:] if row_count > max_sample_rows else [],
        header_candidates=header_candidates,
        max_sample_rows=max_sample_rows,
    )


def _content_id(columns: list[str], rows: list[list[str]]) -> str:
    """Build an exact content identifier from ordered columns and rows."""
    hasher = hashlib.sha256()
    _update_hash_rows(hasher, [columns], row_limit=1)
    hasher.update(b"\x1a")
    _update_hash_rows(hasher, rows, row_limit=len(rows))
    return hasher.hexdigest()


def resolve_root_dir(*, root_dir: str | Path | None = None) -> Path:
    """Resolve the tabular workspace root, defaulting to the current working directory."""
    return (
        Path.cwd().resolve()
        if root_dir is None
        else Path(root_dir).expanduser().resolve()
    )


def sqlite_database_path(*, root_dir: str | Path | None = None) -> Path:
    """Return the shared SQLite path for extracted tabular data."""
    resolved_root = resolve_root_dir(root_dir=root_dir)
    data_dir = resolved_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / SQLITE_FILENAME


@contextmanager
def sqlite_write_lock(database_path: Path):
    """Serialize SQLite writers with a lightweight lock file."""
    lock_path = database_path.with_suffix(f"{database_path.suffix}.lock")
    start_time = time.monotonic()

    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as error:
            if time.monotonic() - start_time >= LOCK_TIMEOUT_SECONDS:
                raise TimeoutError(
                    f"Timed out waiting for SQLite lock: {lock_path}"
                ) from error
            time.sleep(LOCK_POLL_SECONDS)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def quote_identifier(identifier: str) -> str:
    """Quote a SQL identifier safely."""
    return '"' + identifier.replace('"', '""') + '"'


def _db_column_names(columns: list[str]) -> list[str]:
    """Return unique SQL-safe column names while preserving order."""
    seen: dict[str, int] = {}
    normalized: list[str] = []

    for index, column in enumerate(columns, start=1):
        base_name = column or f"column_{index}"
        suffix = seen.get(base_name, 0) + 1
        seen[base_name] = suffix
        normalized.append(base_name if suffix == 1 else f"{base_name}_{suffix}")

    return normalized


def _create_sqlite_sources_table(
    connection: sqlite3.Connection, *, table_name: str = SQLITE_SOURCES_TABLE
) -> None:
    """Create the source-linkage table for extracted content."""
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {quote_identifier(table_name)} (
            source_path TEXT NOT NULL,
            source_format TEXT NOT NULL,
            source_sheet_name TEXT NOT NULL,
            source_table_name TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            content_id TEXT NOT NULL,
            UNIQUE(source_path, source_sheet_name, source_table_name)
        )
        """
    )


def _sqlite_table_columns(connection: sqlite3.Connection, table_name: str) -> list[str]:
    """Return the current column names for a SQLite table."""
    return [
        cast(str, row[1])
        for row in connection.execute(
            f"PRAGMA table_info({quote_identifier(table_name)})"
        ).fetchall()
    ]


def _sqlite_unique_indexes(
    connection: sqlite3.Connection, table_name: str
) -> list[tuple[str, ...]]:
    """Return unique-index column tuples for a SQLite table."""
    unique_indexes: list[tuple[str, ...]] = []
    for _, index_name, is_unique, *_ in connection.execute(
        f"PRAGMA index_list({quote_identifier(table_name)})"
    ).fetchall():
        if not is_unique:
            continue
        columns = tuple(
            cast(str, row[2])
            for row in connection.execute(
                f"PRAGMA index_info({quote_identifier(cast(str, index_name))})"
            ).fetchall()
        )
        unique_indexes.append(columns)
    return unique_indexes


def _ensure_sqlite_sources_table_schema(connection: sqlite3.Connection) -> None:
    """Create or recreate `_tabular_sources` with the expected schema."""
    columns = _sqlite_table_columns(connection, SQLITE_SOURCES_TABLE)
    if not columns:
        _create_sqlite_sources_table(connection)
        return

    existing_columns = set(columns)
    has_expected_columns = existing_columns == SQLITE_SOURCES_COLUMN_SET
    has_expected_unique_key = SQLITE_SOURCES_UNIQUE_COLUMNS in _sqlite_unique_indexes(
        connection, SQLITE_SOURCES_TABLE
    )
    if has_expected_columns and has_expected_unique_key:
        return

    connection.execute(f"DROP TABLE {quote_identifier(SQLITE_SOURCES_TABLE)}")
    _create_sqlite_sources_table(connection)


def _ensure_sqlite_catalog(connection: sqlite3.Connection) -> None:
    """Create the shared SQLite catalog tables when missing."""
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SQLITE_CONTENTS_TABLE} (
            content_id TEXT PRIMARY KEY,
            table_name TEXT NOT NULL,
            source_format TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            column_schema_json TEXT NOT NULL
        )
        """
    )
    _ensure_sqlite_sources_table_schema(connection)


def _clean_numeric_text(value: str) -> str:
    """Normalize a numeric-looking string for safe SQL casting."""
    return value.strip().replace(",", "")


def _is_integer_value(value: str) -> bool:
    """Return whether a string safely represents an integer."""
    return bool(INTEGER_PATTERN.fullmatch(_clean_numeric_text(value)))


def _is_real_value(value: str) -> bool:
    """Return whether a string safely represents a real number."""
    normalized = _clean_numeric_text(value)
    return "." in normalized and bool(REAL_PATTERN.fullmatch(normalized))


def _matches_datetime_patterns(value: str, patterns: tuple[str, ...]) -> bool:
    """Return whether a string matches one of the accepted datetime patterns."""
    for pattern in patterns:
        try:
            time.strptime(value.strip(), pattern)
            return True
        except ValueError:
            continue
    return False


def _infer_column_type(values: Iterable[str]) -> str:
    """Infer a conservative SQLite affinity for a column."""
    non_empty_values = [value.strip() for value in values if value.strip()]
    if not non_empty_values:
        return "TEXT"
    if all(_is_integer_value(value) for value in non_empty_values):
        return "INTEGER"
    if all(
        _is_integer_value(value) or _is_real_value(value) for value in non_empty_values
    ):
        return "REAL"
    if all(
        _matches_datetime_patterns(value, DATETIME_PATTERNS)
        for value in non_empty_values
    ):
        return "DATETIME"
    if all(
        _matches_datetime_patterns(value, DATE_PATTERNS) for value in non_empty_values
    ):
        return "DATE"
    return "TEXT"


def _typed_view_expression(column_name: str, column_type: str) -> str:
    """Build a typed SQLite view expression for a raw text column."""
    identifier = quote_identifier(column_name)
    trimmed = f"TRIM({identifier})"
    empty_to_null = f"NULLIF({trimmed}, '')"
    normalized_numeric = f"REPLACE({trimmed}, ',', '')"
    normalized_date = f"REPLACE({trimmed}, '/', '-')"

    if column_type == "INTEGER":
        return f"CASE WHEN {trimmed} = '' THEN NULL ELSE CAST({normalized_numeric} AS INTEGER) END AS {identifier}"
    if column_type == "REAL":
        return f"CASE WHEN {trimmed} = '' THEN NULL ELSE CAST({normalized_numeric} AS REAL) END AS {identifier}"
    if column_type == "DATE":
        return f"CASE WHEN {trimmed} = '' THEN NULL ELSE date({normalized_date}) END AS {identifier}"
    if column_type == "DATETIME":
        return f"CASE WHEN {trimmed} = '' THEN NULL ELSE datetime({normalized_date}) END AS {identifier}"
    return f"{empty_to_null} AS {identifier}"


def _create_typed_sqlite_view(
    connection: sqlite3.Connection,
    *,
    table_name: str,
    db_columns: list[str],
    rows: list[list[str]],
) -> tuple[str, dict[str, str]]:
    """Create or replace a typed view alongside a raw extracted table."""
    typed_view_name = f"{table_name}_typed"
    inferred_types = {
        column_name: _infer_column_type(
            row[column_index] if column_index < len(row) else "" for row in rows
        )
        for column_index, column_name in enumerate(db_columns)
    }
    select_sql = ", ".join(
        _typed_view_expression(column_name, inferred_types[column_name])
        for column_name in db_columns
    )
    connection.execute(f"DROP VIEW IF EXISTS {quote_identifier(typed_view_name)}")
    connection.execute(
        f"""
        CREATE VIEW {quote_identifier(typed_view_name)} AS
        SELECT {select_sql}
        FROM {quote_identifier(table_name)}
        """
    )
    return typed_view_name, inferred_types


def _content_table_name(content_id: str) -> str:
    """Build the physical SQLite table name for a content identifier."""
    return f"content_{content_id[:16]}"


def _content_schema_json(columns: list[str], db_columns: list[str]) -> str:
    """Serialize source and database column mappings for catalog storage."""
    return json.dumps(
        {
            "source_columns": columns,
            "db_columns": db_columns,
        },
        separators=(",", ":"),
    )


def _create_sqlite_table(
    connection: sqlite3.Connection,
    *,
    table_name: str,
    columns: list[str],
    rows: list[list[str]],
) -> None:
    """Create and populate a SQLite table from normalized text rows."""
    column_sql = ", ".join(f"{quote_identifier(column)} TEXT" for column in columns)
    connection.execute(f"CREATE TABLE {quote_identifier(table_name)} ({column_sql})")
    placeholder_sql = ", ".join("?" for _ in columns)
    insert_sql = (
        f"INSERT INTO {quote_identifier(table_name)} VALUES ({placeholder_sql})"
    )

    for start_index in range(0, len(rows), INSERT_BATCH_SIZE):
        batch = rows[start_index : start_index + INSERT_BATCH_SIZE]
        connection.executemany(insert_sql, batch)


def _load_or_reuse_content_table(
    connection: sqlite3.Connection,
    *,
    columns: list[str],
    rows: list[list[str]],
    source_format: str,
) -> tuple[str, str, list[str], str]:
    """Ensure a raw content table exists and return its storage metadata."""
    content_id = _content_id(columns, rows)
    table_name = _content_table_name(content_id)
    db_columns = _db_column_names(columns)
    existing = connection.execute(
        f"SELECT table_name FROM {SQLITE_CONTENTS_TABLE} WHERE content_id = ?",
        [content_id],
    ).fetchone()
    if existing is not None:
        return content_id, cast(str, existing[0]), db_columns, "reused"

    _create_sqlite_table(
        connection,
        table_name=table_name,
        columns=db_columns,
        rows=rows,
    )
    connection.execute(
        f"""
        INSERT INTO {SQLITE_CONTENTS_TABLE}
        (content_id, table_name, source_format, row_count, column_schema_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            content_id,
            table_name,
            source_format,
            len(rows),
            _content_schema_json(columns, db_columns),
        ],
    )
    return content_id, table_name, db_columns, "loaded"


def _register_sqlite_source(
    connection: sqlite3.Connection,
    *,
    source_path: str,
    source_format: str,
    source_sheet_name: str,
    source_table_name: str,
    fingerprint: str,
    content_id: str,
) -> None:
    """Record how a source artifact maps to extracted content."""
    connection.execute(
        f"""
        INSERT OR REPLACE INTO {SQLITE_SOURCES_TABLE}
        (source_path, source_format, source_sheet_name, source_table_name, fingerprint, content_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            source_path,
            source_format,
            source_sheet_name,
            source_table_name,
            fingerprint,
            content_id,
        ],
    )


def load_tables_into_sqlite(
    recovered: dict[str, Any],
    *,
    root_dir: str | Path | None = None,
    fingerprint: str,
) -> dict[str, Any]:
    """Load recovered tables into a shared SQLite database."""
    database_path = sqlite_database_path(root_dir=root_dir)
    loaded_tables: list[dict[str, Any]] = []
    source_path = recovered["path"]
    source_format = recovered["format"]
    source_sheet_name = recovered.get("sheet_name") or ""

    with sqlite_write_lock(database_path):
        connection = sqlite3.connect(str(database_path))
        try:
            _ensure_sqlite_catalog(connection)

            for table in recovered["tables"]:
                columns = list(table["columns"])
                rows = list(table["rows"])
                content_id, table_name, db_columns, load_status = (
                    _load_or_reuse_content_table(
                        connection,
                        columns=columns,
                        rows=rows,
                        source_format=source_format,
                    )
                )
                typed_view_name, typed_columns = _create_typed_sqlite_view(
                    connection,
                    table_name=table_name,
                    db_columns=db_columns,
                    rows=rows,
                )
                _register_sqlite_source(
                    connection,
                    source_path=source_path,
                    source_format=source_format,
                    source_sheet_name=source_sheet_name,
                    source_table_name=table["name"],
                    fingerprint=fingerprint,
                    content_id=content_id,
                )

                loaded_tables.append(
                    {
                        "source_name": table["name"],
                        "content_id": content_id,
                        "table_name": table_name,
                        "typed_view_name": typed_view_name,
                        "row_count": len(rows),
                        "columns": columns,
                        "db_columns": db_columns,
                        "typed_columns": typed_columns,
                        "load_status": load_status,
                    }
                )
            connection.commit()
        finally:
            connection.close()

    return {
        "database_path": str(database_path),
        "tables": loaded_tables,
    }
