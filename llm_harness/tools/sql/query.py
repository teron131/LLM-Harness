"""Standalone SQLite query and schema-navigation helpers."""

from __future__ import annotations

from contextlib import closing
from datetime import date, datetime, time as datetime_time
from decimal import Decimal
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import zip_longest
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, cast

from ..tabular.storage import (
    SQLITE_CONTENTS_TABLE,
    SQLITE_SOURCES_TABLE,
    quote_identifier,
    sqlite_database_path,
    sqlite_write_lock,
)

MAX_QUERY_ROWS = 200
MAX_DESCRIBE_SAMPLE_ROWS = 5
MAX_REPAIR_CANDIDATES = 3
MAX_TEXT_VALUE_HINTS = 5
MAX_SUGGESTED_TARGETS = 5
MAX_SOURCE_PATH_PREVIEW = 3
MAX_COLUMN_PREVIEW = 8
MAX_REASON_PREVIEW = 3
_MISSING = object()
_READ_ONLY_SQL_PREFIXES = ("SELECT", "WITH", "EXPLAIN")
_VIEW_SQL_PREFIXES = ("SELECT", "WITH")
_LEADING_SQL_COMMENT = re.compile(r"\A(?:\s+|--[^\n]*(?:\n|\Z)|/\*.*?\*/)*", re.DOTALL)
_TEXT_TYPE_MARKERS = ("CHAR", "CLOB", "TEXT", "VARCHAR")
_TEXT_HINT_NAME_MARKERS = (
    "category",
    "code",
    "description",
    "group",
    "id",
    "identifier",
    "key",
    "kind",
    "label",
    "name",
    "segment",
    "status",
    "type",
)
_TARGET_MASTER_SQL = """
SELECT name, type, sql
FROM sqlite_master
WHERE type IN ('table', 'view')
  AND name NOT LIKE 'sqlite_%'
ORDER BY type, name
"""
_TARGET_BY_NAME_SQL = """
SELECT name, type, sql
FROM sqlite_master
WHERE type IN ('table', 'view') AND name = ?
"""
_AMBIGUOUS_COLUMN_ERROR_PREFIX = "ambiguous column name:"
_MISSING_COLUMN_ERROR_PREFIX = "no such column:"
_MISSING_TABLE_ERROR_PREFIX = "no such table:"
_SUGGESTION_STOP_WORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "me",
    "of",
    "on",
    "show",
    "the",
    "to",
    "what",
    "which",
    "with",
}
_VIEW_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _jsonable_value(value: object) -> object:
    """Convert SQL query results into JSON-friendly values."""
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "hex": value.hex(),
            "length": len(value),
        }
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return value


def _zip_exact(left: list[str], right: tuple[Any, ...]) -> list[tuple[str, Any]]:
    """Zip two sequences and fail if their lengths differ."""
    pairs: list[tuple[str, Any]] = []
    for left_item, right_item in zip_longest(left, right, fillvalue=_MISSING):
        if left_item is _MISSING or right_item is _MISSING:
            raise ValueError(
                "SQL result row width did not match the reported column metadata."
            )
        pairs.append((cast(str, left_item), right_item))
    return pairs


def _error_result(
    *,
    database_path: str | Path | None,
    error_type: str,
    message: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build a stable error payload for tool callers."""
    payload: dict[str, Any] = {
        "status": "error",
        "error_type": error_type,
        "message": message,
    }
    if database_path is not None:
        payload["database_path"] = str(database_path)
    payload.update(extra)
    return payload


def _preview_list(
    items: list[Any],
    *,
    max_items: int,
) -> tuple[list[Any], bool]:
    """Return a bounded list preview plus truncation state."""
    safe_max_items = max(0, max_items)
    return items[:safe_max_items], len(items) > safe_max_items


def _query_summary(
    *,
    row_count: int,
    column_count: int,
    truncated: bool,
) -> str:
    """Build one compact summary for a SQL query result."""
    summary = f"Returned {row_count} row(s) across {column_count} column(s)."
    if truncated:
        summary += " Result rows were truncated."
    return summary


def _query_result_payload(
    *,
    database_path: Path,
    cursor: sqlite3.Cursor,
    max_rows: int,
) -> dict[str, Any]:
    """Build the public result payload for one executed SQL cursor."""
    description = cursor.description
    if not description:
        return {
            "database_path": str(database_path),
            "status": "ok",
            "max_rows": max_rows,
            "row_count": 0,
            "truncated": False,
            "columns": [],
            "rows": [],
            "summary": _query_summary(
                row_count=0,
                column_count=0,
                truncated=False,
            ),
        }

    column_names, original_columns = _normalized_column_names(
        [cast(Any, column[0]) for column in description]
    )
    raw_rows = cursor.fetchmany(max_rows + 1)
    truncated = len(raw_rows) > max_rows
    rows = [
        {
            column_name: _jsonable_value(value)
            for column_name, value in _zip_exact(column_names, row)
        }
        for row in raw_rows[:max_rows]
    ]
    payload = {
        "database_path": str(database_path),
        "status": "ok",
        "max_rows": max_rows,
        "row_count": len(rows),
        "truncated": truncated,
        "columns": column_names,
        "rows": rows,
        "summary": _query_summary(
            row_count=len(rows),
            column_count=len(column_names),
            truncated=truncated,
        ),
    }
    if column_names != original_columns:
        payload["original_columns"] = original_columns
    return payload


def _target_summary(
    *,
    name: str,
    target_type: str,
    kind: str,
    row_count: int | None,
    column_names: list[str],
    source_paths: list[str],
    reasons: list[str] | None = None,
) -> str:
    """Build one compact summary for a target suggestion or listing."""
    summary_parts = [
        f"{name} ({kind}, {target_type})",
        f"{len(column_names)} column(s)",
    ]
    if row_count is not None:
        summary_parts.append(f"{row_count} row(s)")
    if source_paths:
        summary_parts.append(f"{len(source_paths)} source file(s)")
    if reasons:
        summary_parts.append("matched " + ", ".join(reasons[:MAX_REASON_PREVIEW]))
    return "; ".join(summary_parts)


def _target_column_names(target: dict[str, Any]) -> list[str]:
    """Return the catalog column names for one target."""
    return [
        cast(str, column["name"])
        for column in cast(list[dict[str, Any]], target["columns"])
    ]


def _target_list_item(target: dict[str, Any]) -> dict[str, Any]:
    """Build one public target listing item from catalog metadata."""
    name = cast(str, target["name"])
    target_type = cast(str, target["type"])
    kind = cast(str, target["kind"])
    row_count = cast(int | None, target["row_count"])
    source_paths = cast(list[str], target["source_paths"])
    source_path_preview, source_paths_truncated = _preview_list(
        source_paths,
        max_items=MAX_SOURCE_PATH_PREVIEW,
    )
    column_names = _target_column_names(target)
    return {
        "name": name,
        "type": target_type,
        "kind": kind,
        "row_count": row_count,
        "source_path_count": len(source_paths),
        "source_path_preview": source_path_preview,
        "source_paths_truncated": source_paths_truncated,
        "summary": _target_summary(
            name=name,
            target_type=target_type,
            kind=kind,
            row_count=row_count,
            column_names=column_names,
            source_paths=source_paths,
        ),
    }


def _target_suggestion(
    target: dict[str, Any],
    *,
    tokens: list[str],
) -> dict[str, Any] | None:
    """Build one scored target suggestion from catalog metadata."""
    name = cast(str, target["name"])
    target_type = cast(str, target["type"])
    kind = cast(str, target["kind"])
    row_count = cast(int | None, target["row_count"])
    column_names = _target_column_names(target)
    source_paths = cast(list[str], target["source_paths"])
    search_text = _target_search_text(
        name=name,
        target_type=target_type,
        kind=kind,
        column_names=column_names,
        source_paths=source_paths,
        create_sql=cast(str | None, target["create_sql"]),
    )
    score, reasons = _target_score(
        tokens=tokens,
        name=name,
        column_names=column_names,
        source_paths=source_paths,
        search_text=search_text,
    )
    if score <= 0:
        return None

    score += _kind_bias(kind)
    column_preview, columns_truncated = _preview_list(
        column_names,
        max_items=MAX_COLUMN_PREVIEW,
    )
    source_path_preview, source_paths_truncated = _preview_list(
        source_paths,
        max_items=MAX_SOURCE_PATH_PREVIEW,
    )
    return {
        "name": name,
        "type": target_type,
        "kind": kind,
        "score": score,
        "reasons": reasons[:MAX_REASON_PREVIEW],
        "column_count": len(column_names),
        "column_preview": column_preview,
        "columns_truncated": columns_truncated,
        "source_path_count": len(source_paths),
        "source_path_preview": source_path_preview,
        "source_paths_truncated": source_paths_truncated,
        "row_count": row_count,
        "summary": _target_summary(
            name=name,
            target_type=target_type,
            kind=kind,
            row_count=row_count,
            column_names=column_names,
            source_paths=source_paths,
            reasons=reasons,
        ),
    }


def _normalized_column_names(
    column_names: list[str | None],
) -> tuple[list[str], list[str]]:
    """Return stable, unique column names for row dictionaries."""
    seen: set[str] = set()
    normalized: list[str] = []
    originals: list[str] = []
    for index, raw_name in enumerate(column_names, start=1):
        base_name = raw_name or f"column_{index}"
        originals.append(base_name)
        suffix = 1
        candidate_name = base_name
        while candidate_name in seen:
            suffix += 1
            candidate_name = f"{base_name}__{suffix}"
        seen.add(candidate_name)
        normalized.append(candidate_name)
    return normalized, originals


def _leading_sql_keyword(sql: str) -> str:
    """Extract the first SQL keyword after whitespace and comments."""
    stripped = _LEADING_SQL_COMMENT.sub("", sql, count=1).lstrip()
    if not stripped:
        return ""
    return stripped.split(None, 1)[0].upper()


def _is_read_only_sql(sql: str) -> bool:
    """Return whether a SQL statement looks read-only."""
    return _leading_sql_keyword(sql) in _READ_ONLY_SQL_PREFIXES


def _is_view_sql(sql: str) -> bool:
    """Return whether a SQL statement can be embedded in CREATE VIEW AS."""
    return _leading_sql_keyword(sql) in _VIEW_SQL_PREFIXES


def _normalized_sql(sql: str) -> str:
    """Normalize one SQL statement for validation and embedding."""
    return sql.strip().rstrip(";").strip()


def _is_valid_view_name(view_name: str) -> bool:
    """Return whether a SQLite view name is safe to create."""
    return bool(_VIEW_NAME_PATTERN.fullmatch(view_name))


def _sqlite_object_names(connection: sqlite3.Connection) -> set[str]:
    """Return all user-visible table and view names in a SQLite database."""
    return {
        cast(str, row[0])
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
    }


def _sample_rows(
    connection: sqlite3.Connection,
    *,
    target_name: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch a small row preview for one table or view."""
    safe_limit = max(0, min(limit, MAX_DESCRIBE_SAMPLE_ROWS))
    if safe_limit == 0:
        return []

    cursor = connection.execute(
        f"SELECT * FROM {quote_identifier(target_name)} LIMIT ?",
        [safe_limit],
    )
    description = cursor.description or []
    column_names, _ = _normalized_column_names(
        [cast(Any, column[0]) for column in description]
    )
    preview_rows = []
    for row in cursor.fetchall():
        preview_rows.append(
            {
                column_name: _jsonable_value(value)
                for column_name, value in _zip_exact(column_names, row)
            }
        )
    return preview_rows


def _fetch_catalog_metadata(
    connection: sqlite3.Connection,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    """Load shared tabular catalog metadata for list, describe, and suggest helpers."""
    content_query = f"SELECT content_id, table_name, source_format, row_count, column_schema_json FROM {SQLITE_CONTENTS_TABLE}"
    content_rows = {
        cast(str, row[1]): {
            "content_id": cast(str, row[0]),
            "source_format": cast(str, row[2]),
            "row_count": cast(int, row[3]),
            "content_schema": json.loads(cast(str, row[4])),
        }
        for row in connection.execute(content_query).fetchall()
    }
    source_rows: dict[str, list[dict[str, Any]]] = {}
    source_query = f"""
    SELECT content_id, source_path, source_format, source_sheet_name, source_table_name, fingerprint
    FROM {SQLITE_SOURCES_TABLE}
    ORDER BY source_path, source_table_name
    """
    for row in connection.execute(source_query).fetchall():
        content_id = cast(str, row[0])
        source_rows.setdefault(content_id, []).append(
            {
                "source_path": cast(str, row[1]),
                "source_format": cast(str, row[2]),
                "source_sheet_name": cast(str, row[3]),
                "source_table_name": cast(str, row[4]),
                "fingerprint": cast(str, row[5]),
            }
        )
    return content_rows, source_rows


def _requested_database_path(
    *,
    root_dir: str | Path | None = None,
    database_path: str | Path | None = None,
) -> Path:
    """Return the requested SQLite path before existence checks."""
    return (
        sqlite_database_path(root_dir=root_dir)
        if database_path is None
        else Path(database_path)
    )


def _open_read_only_connection(database_path: Path) -> sqlite3.Connection:
    """Open one SQLite connection in read-only mode."""
    return sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True)


def _catalog_state(
    connection: sqlite3.Connection,
) -> tuple[
    bool,
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    """Return whether the tabular catalog exists plus its cached metadata."""
    if not _has_catalog(connection):
        return False, {}, {}
    content_rows, source_rows = _fetch_catalog_metadata(connection)
    return True, content_rows, source_rows


def _has_catalog(connection: sqlite3.Connection) -> bool:
    """Return whether the shared tabular catalog exists in this database."""
    object_names = _sqlite_object_names(connection)
    return (
        SQLITE_CONTENTS_TABLE in object_names and SQLITE_SOURCES_TABLE in object_names
    )


def _base_table_name(name: str, kind: str) -> str:
    """Map a view name back to its catalog table name."""
    if kind == "typed_content_view":
        return name.removesuffix("_typed")
    return name


def _target_source_paths(
    *,
    name: str,
    kind: str,
    content_rows: dict[str, dict[str, Any]],
    source_rows: dict[str, list[dict[str, Any]]],
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[str],
]:
    """Return catalog metadata, source mappings, and source paths for one target."""
    content_metadata = content_rows.get(_base_table_name(name, kind))
    if content_metadata is None:
        return None, [], []
    source_mappings = source_rows.get(cast(str, content_metadata["content_id"]), [])
    source_paths = list(
        dict.fromkeys(cast(str, mapping["source_path"]) for mapping in source_mappings)
    )
    return content_metadata, source_mappings, source_paths


def _database_cache_key(
    database_path: Path,
) -> tuple[
    str,
    int,
    int,
]:
    """Return a cache key that invalidates when the SQLite file changes."""
    resolved_path = database_path.resolve()
    stat_result = resolved_path.stat()
    return str(resolved_path), stat_result.st_mtime_ns, stat_result.st_size


@lru_cache(maxsize=8)
def _cached_database_catalog(
    resolved_path: str,
    mtime_ns: int,
    size_bytes: int,
) -> dict[str, Any]:
    """Load one cached SQLite catalog snapshot."""
    del mtime_ns, size_bytes
    database_path = Path(resolved_path)
    with closing(_open_read_only_connection(database_path)) as connection:
        has_catalog, content_rows, source_rows = _catalog_state(connection)
        targets = []
        targets_by_name: dict[str, dict[str, Any]] = {}
        for master_row in connection.execute(_TARGET_MASTER_SQL).fetchall():
            name = cast(str, master_row[0])
            target_type = cast(str, master_row[1])
            create_sql = master_row[2]
            kind = classify_target(name)
            columns = [
                {
                    "name": cast(str, row[1]),
                    "type": cast(str, row[2]),
                    "not_null": bool(row[3]),
                    "default_value": row[4],
                    "primary_key_position": cast(int, row[5]),
                }
                for row in connection.execute(
                    f"PRAGMA table_info({quote_identifier(name)})"
                ).fetchall()
            ]
            content_metadata, source_mappings, source_paths = _target_source_paths(
                name=name,
                kind=kind,
                content_rows=content_rows,
                source_rows=source_rows,
            )
            target_info = {
                "name": name,
                "type": target_type,
                "kind": kind,
                "create_sql": create_sql,
                "columns": columns,
                "content_id": None
                if content_metadata is None
                else content_metadata["content_id"],
                "content_schema": None
                if content_metadata is None
                else content_metadata["content_schema"],
                "row_count": None
                if content_metadata is None
                else content_metadata["row_count"],
                "source_mappings": source_mappings,
                "source_paths": source_paths,
            }
            targets.append(target_info)
            targets_by_name[name] = target_info
        return {
            "has_catalog": has_catalog,
            "targets": targets,
            "targets_by_name": targets_by_name,
        }


def _database_catalog(database_path: Path) -> dict[str, Any]:
    """Return the cached database catalog for one SQLite file."""
    return _cached_database_catalog(*_database_cache_key(database_path))


def _looks_like_text_column(column_name: str, declared_type: str) -> bool:
    """Return whether a column is a good candidate for distinct text-value hints."""
    normalized_type = declared_type.upper()
    normalized_name_tokens = set(re.findall(r"[a-z0-9]+", column_name.lower()))
    if any(marker in normalized_type for marker in _TEXT_TYPE_MARKERS):
        return True
    return any(marker in normalized_name_tokens for marker in _TEXT_HINT_NAME_MARKERS)


def _text_value_hints(
    connection: sqlite3.Connection,
    *,
    target_name: str,
    columns: list[dict[str, Any]],
    max_columns: int,
    max_values: int,
) -> dict[str, list[str]]:
    """Fetch a few distinct example values for useful text-like columns."""
    hints: dict[str, list[str]] = {}
    safe_max_columns = max(0, max_columns)
    safe_max_values = max(0, min(max_values, MAX_TEXT_VALUE_HINTS))
    if safe_max_columns == 0 or safe_max_values == 0:
        return hints

    candidate_columns = [
        cast(str, column["name"])
        for column in columns
        if _looks_like_text_column(
            cast(str, column["name"]),
            cast(str, column["type"]),
        )
    ][:safe_max_columns]

    for column_name in candidate_columns:
        rows = connection.execute(
            f"""
            SELECT DISTINCT CAST({quote_identifier(column_name)} AS TEXT)
            FROM {quote_identifier(target_name)}
            WHERE {quote_identifier(column_name)} IS NOT NULL
              AND TRIM(CAST({quote_identifier(column_name)} AS TEXT)) != ''
            LIMIT ?
            """,
            [safe_max_values],
        ).fetchall()
        values = [cast(str, row[0]) for row in rows if row and row[0] is not None]
        if values:
            hints[column_name] = values
    return hints


def _tokenize_query(text: str) -> list[str]:
    """Tokenize a natural-language query for lightweight target suggestion."""
    tokens = [
        token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) >= 2
    ]
    return [token for token in tokens if token not in _SUGGESTION_STOP_WORDS]


def _identifier_tokens(value: str) -> set[str]:
    """Split one identifier into comparable lowercase tokens."""
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _identifier_similarity(reference: str, candidate: str) -> float:
    """Score one candidate identifier against a missing identifier."""
    normalized_reference = re.sub(r"[^a-z0-9]+", "", reference.lower())
    normalized_candidate = re.sub(r"[^a-z0-9]+", "", candidate.lower())
    if not normalized_reference or not normalized_candidate:
        return 0.0
    if normalized_reference == normalized_candidate:
        return 100.0

    score = 0.0
    if (
        normalized_reference in normalized_candidate
        or normalized_candidate in normalized_reference
    ):
        score += 40.0

    reference_tokens = _identifier_tokens(reference)
    candidate_tokens = _identifier_tokens(candidate)
    if reference_tokens and candidate_tokens:
        shared_tokens = reference_tokens & candidate_tokens
        score += 20.0 * len(shared_tokens)
        if shared_tokens == reference_tokens == candidate_tokens:
            score += 20.0

    score += (
        SequenceMatcher(a=normalized_reference, b=normalized_candidate).ratio() * 20.0
    )
    return score


def _rank_identifier_candidates(
    identifier: str, candidates: list[str], *, max_matches: int
) -> list[str]:
    """Return the best schema identifier matches for a missing target or column."""
    if not identifier or max_matches <= 0:
        return []

    scored_candidates = sorted(
        (
            (_identifier_similarity(identifier, candidate), candidate)
            for candidate in dict.fromkeys(candidates)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return [candidate for score, candidate in scored_candidates if score > 0][
        :max_matches
    ]


def _error_identifier(error_message: str, prefix: str) -> str:
    """Extract the identifier payload from a SQLite error message."""
    identifier = error_message[len(prefix) :].strip()
    if not identifier:
        return ""
    unqualified = identifier.rsplit(".", 1)[-1]
    return unqualified.strip('"`[]')


def _format_repair_candidates(
    candidates: list[dict[str, Any]],
    *,
    include_targets: bool,
) -> str:
    """Format repair candidates into one compact human-readable string."""
    parts = []
    for candidate in candidates:
        name = cast(str, candidate["name"])
        if include_targets:
            targets = cast(list[str], candidate.get("targets", []))
            target_suffix = f" on {', '.join(targets)}" if targets else ""
            parts.append(f"{name}{target_suffix}")
        else:
            parts.append(name)
    return ", ".join(parts)


def _repair_result(
    *,
    kind: str,
    identifier: str,
    candidates: list[dict[str, Any]],
    message: str,
) -> list[dict[str, Any]]:
    """Wrap one repair suggestion in the stable public payload shape."""
    return [
        {
            "kind": kind,
            "identifier": identifier,
            "candidates": candidates,
            "message": message,
        }
    ]


def _target_search_text(
    *,
    name: str,
    target_type: str,
    kind: str,
    column_names: list[str],
    source_paths: list[str],
    create_sql: str | None,
) -> str:
    """Build a search blob for one target."""
    parts = [
        name.replace("_", " "),
        target_type,
        kind.replace("_", " "),
        " ".join(column_names),
        " ".join(source_paths),
        create_sql or "",
    ]
    return " ".join(parts).lower()


def _target_score(
    *,
    tokens: list[str],
    name: str,
    column_names: list[str],
    source_paths: list[str],
    search_text: str,
) -> tuple[int, list[str]]:
    """Score one target against a lightweight NL query."""
    score = 0
    reasons: list[str] = []
    lowered_name = name.lower()
    lowered_columns = [column.lower() for column in column_names]
    lowered_sources = [source.lower() for source in source_paths]

    for token in tokens:
        if token in lowered_name:
            score += 5
            reasons.append(f"name matched '{token}'")
            continue
        matching_columns = [column for column in lowered_columns if token in column]
        if matching_columns:
            score += 3
            reasons.append(f"column matched '{token}'")
            continue
        if any(token in source for source in lowered_sources):
            score += 2
            reasons.append(f"source matched '{token}'")
            continue
        if token in search_text:
            score += 1
            reasons.append(f"context matched '{token}'")

    if tokens and all(token in lowered_name for token in tokens):
        score += 2
        reasons.append("all tokens matched target name")
    return score, reasons


def _kind_bias(kind: str) -> int:
    """Prefer stable views over raw storage artifacts during target suggestion."""
    if kind == "typed_content_view":
        return 3
    if kind == "view_or_table":
        return 1
    if kind == "raw_content_table":
        return -2
    if kind == "internal_catalog":
        return -5
    return 0


def resolve_db_path(
    *,
    root_dir: str | Path | None = None,
    database_path: str | Path | None = None,
) -> Path:
    """Resolve the SQLite database path and ensure it exists."""
    resolved_path = _requested_database_path(
        root_dir=root_dir, database_path=database_path
    )
    if not resolved_path.exists():
        raise ValueError(f"SQLite database does not exist: {resolved_path}")
    return resolved_path


def classify_target(name: str) -> str:
    """Classify a SQLite target using current naming conventions."""
    if name in (SQLITE_CONTENTS_TABLE, SQLITE_SOURCES_TABLE):
        return "internal_catalog"
    if name.startswith("content_") and name.endswith("_typed"):
        return "typed_content_view"
    if name.startswith("content_"):
        return "raw_content_table"
    return "view_or_table"


def _save_view_validation_error(
    *,
    requested_path: Path,
    normalized_sql: str,
    normalized_view_name: str,
    replace: bool,
) -> dict[str, Any] | None:
    """Return a stable save-view validation error, if the request is invalid."""
    if not normalized_sql:
        return _error_result(
            database_path=requested_path,
            error_type="empty_sql",
            message="SQL query must not be empty.",
            view_name=normalized_view_name,
            replace=replace,
        )
    if not normalized_view_name:
        return _error_result(
            database_path=requested_path,
            error_type="empty_view_name",
            message="View name must not be empty.",
            replace=replace,
        )
    if not _is_valid_view_name(normalized_view_name):
        return _error_result(
            database_path=requested_path,
            error_type="invalid_view_name",
            message="View name must start with a letter or underscore and contain only letters, digits, and underscores.",
            view_name=normalized_view_name,
            replace=replace,
        )
    if normalized_view_name.startswith("sqlite_"):
        return _error_result(
            database_path=requested_path,
            error_type="reserved_view_name",
            message="View name must not start with 'sqlite_'.",
            view_name=normalized_view_name,
            replace=replace,
        )
    if not _is_view_sql(normalized_sql):
        return _error_result(
            database_path=requested_path,
            error_type="disallowed_sql",
            message="Only read-only SELECT and WITH queries can be saved as views.",
            view_name=normalized_view_name,
            replace=replace,
        )
    return None


def run_query(
    sql: str,
    *,
    root_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    max_rows: int = MAX_QUERY_ROWS,
) -> dict[str, Any]:
    """Run SQL against a SQLite database and return a bounded result."""
    requested_path = _requested_database_path(
        root_dir=root_dir, database_path=database_path
    )
    safe_max_rows = max(1, max_rows)
    normalized_sql = _normalized_sql(sql)
    try:
        if not normalized_sql:
            return _error_result(
                database_path=requested_path,
                error_type="empty_sql",
                message="SQL query must not be empty.",
                max_rows=safe_max_rows,
            )
        # This is a quick UX guard. The read-only SQLite connection is the actual safety boundary.
        if not _is_read_only_sql(normalized_sql):
            return _error_result(
                database_path=requested_path,
                error_type="disallowed_sql",
                message="Only read-only SELECT, WITH, and EXPLAIN queries are allowed.",
                max_rows=safe_max_rows,
            )

        resolved_path = resolve_db_path(
            root_dir=root_dir,
            database_path=database_path,
        )
        with closing(_open_read_only_connection(resolved_path)) as connection:
            cursor = connection.execute(sql)
            return _query_result_payload(
                database_path=resolved_path,
                cursor=cursor,
                max_rows=safe_max_rows,
            )
    except ValueError as exc:
        return _error_result(
            database_path=requested_path,
            error_type="missing_database",
            message=str(exc),
            max_rows=safe_max_rows,
        )
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return _error_result(
            database_path=requested_path,
            error_type="sql_execution_error",
            message=str(exc),
            max_rows=safe_max_rows,
        )


def save_view(
    sql: str,
    view_name: str,
    *,
    root_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Save one read-only SQL query as a named SQLite view."""
    requested_path = _requested_database_path(
        root_dir=root_dir, database_path=database_path
    )
    normalized_sql = _normalized_sql(sql)
    normalized_view_name = view_name.strip()
    try:
        validation_error = _save_view_validation_error(
            requested_path=requested_path,
            normalized_sql=normalized_sql,
            normalized_view_name=normalized_view_name,
            replace=replace,
        )
        if validation_error is not None:
            return validation_error

        resolved_path = resolve_db_path(
            root_dir=root_dir,
            database_path=database_path,
        )
        with (
            sqlite_write_lock(resolved_path),
            closing(sqlite3.connect(str(resolved_path))) as connection,
        ):
            existing_row = connection.execute(
                """
                SELECT type
                FROM sqlite_master
                WHERE name = ?
                """,
                [normalized_view_name],
            ).fetchone()
            if existing_row is not None:
                existing_type = cast(str, existing_row[0])
                if existing_type != "view":
                    return _error_result(
                        database_path=resolved_path,
                        error_type="name_conflict",
                        message=f"SQLite object already exists and is not a view: {normalized_view_name}",
                        view_name=normalized_view_name,
                        replace=replace,
                    )
                if not replace:
                    return _error_result(
                        database_path=resolved_path,
                        error_type="view_exists",
                        message=f"SQLite view already exists: {normalized_view_name}",
                        view_name=normalized_view_name,
                        replace=replace,
                    )
                connection.execute(
                    f"DROP VIEW IF EXISTS {quote_identifier(normalized_view_name)}"
                )

            connection.execute(
                f"CREATE VIEW {quote_identifier(normalized_view_name)} AS {normalized_sql}"
            )
            connection.commit()

        description = describe_target(
            normalized_view_name,
            root_dir=root_dir,
            database_path=resolved_path,
        )
        return {
            "database_path": str(resolved_path),
            "status": "ok",
            "view_name": normalized_view_name,
            "replace": replace,
            "saved_sql": normalized_sql,
            "target": description,
        }
    except ValueError as exc:
        return _error_result(
            database_path=requested_path,
            error_type="missing_database",
            message=str(exc),
            view_name=normalized_view_name,
            replace=replace,
        )
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return _error_result(
            database_path=requested_path,
            error_type="sql_execution_error",
            message=str(exc),
            view_name=normalized_view_name,
            replace=replace,
        )


def list_targets(
    *,
    root_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    include_internal: bool = False,
) -> dict[str, Any]:
    """List queryable SQLite tables and views."""
    requested_path = _requested_database_path(
        root_dir=root_dir, database_path=database_path
    )
    try:
        resolved_path = resolve_db_path(
            root_dir=root_dir,
            database_path=database_path,
        )
        catalog = _database_catalog(resolved_path)
        items = []
        for target in cast(list[dict[str, Any]], catalog["targets"]):
            kind = cast(str, target["kind"])
            if not include_internal and kind == "internal_catalog":
                continue
            items.append(_target_list_item(target))

        return {
            "database_path": str(resolved_path),
            "status": "ok",
            "has_tabular_catalog": catalog["has_catalog"],
            "target_count": len(items),
            "targets": items,
            "summary": f"Listed {len(items)} queryable target(s).",
        }
    except ValueError as exc:
        return _error_result(
            database_path=requested_path,
            error_type="missing_database",
            message=str(exc),
        )
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return _error_result(
            database_path=requested_path,
            error_type="sql_execution_error",
            message=str(exc),
        )


def describe_target(
    target_name: str,
    *,
    root_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    sample_rows: int = 3,
    text_value_hints: int = 3,
) -> dict[str, Any]:
    """Describe a single SQLite table or view."""
    requested_path = _requested_database_path(
        root_dir=root_dir, database_path=database_path
    )
    try:
        resolved_path = resolve_db_path(
            root_dir=root_dir,
            database_path=database_path,
        )
        catalog = _database_catalog(resolved_path)
        target_info = cast(
            dict[str, Any] | None, catalog["targets_by_name"].get(target_name)
        )
        if target_info is None:
            return _error_result(
                database_path=resolved_path,
                error_type="missing_target",
                message=f"SQLite target does not exist: {target_name}",
                target_name=target_name,
            )

        name = cast(str, target_info["name"])
        target_type = cast(str, target_info["type"])
        kind = cast(str, target_info["kind"])
        columns = cast(list[dict[str, Any]], target_info["columns"])
        source_mappings = cast(list[dict[str, Any]], target_info["source_mappings"])
        source_paths = cast(list[str], target_info["source_paths"])
        with closing(_open_read_only_connection(resolved_path)) as connection:
            sample_row_items = _sample_rows(
                connection,
                target_name=name,
                limit=sample_rows,
            )
            text_value_hint_map = _text_value_hints(
                connection,
                target_name=name,
                columns=columns,
                max_columns=text_value_hints,
                max_values=MAX_TEXT_VALUE_HINTS,
            )

            return {
                "database_path": str(resolved_path),
                "status": "ok",
                "has_tabular_catalog": catalog["has_catalog"],
                "name": name,
                "type": target_type,
                "kind": kind,
                "row_count": target_info["row_count"],
                "columns": columns,
                "sample_rows": sample_row_items,
                "text_value_hints": text_value_hint_map,
                "create_sql": target_info["create_sql"],
                "content_id": target_info["content_id"],
                "content_schema": target_info["content_schema"],
                "source_mappings": source_mappings,
                "source_path_count": len(source_paths),
                "source_path_preview": source_paths[:MAX_SOURCE_PATH_PREVIEW],
                "summary": _target_summary(
                    name=name,
                    target_type=target_type,
                    kind=kind,
                    row_count=cast(int | None, target_info["row_count"]),
                    column_names=[cast(str, column["name"]) for column in columns],
                    source_paths=source_paths,
                ),
            }
    except ValueError as exc:
        return _error_result(
            database_path=requested_path,
            error_type="missing_database",
            message=str(exc),
            target_name=target_name,
        )
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return _error_result(
            database_path=requested_path,
            error_type="sql_execution_error",
            message=str(exc),
            target_name=target_name,
        )


def suggest_targets(
    question: str,
    *,
    root_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    include_internal: bool = False,
    max_results: int = MAX_SUGGESTED_TARGETS,
) -> dict[str, Any]:
    """Suggest likely tables or views for a natural-language question."""
    requested_path = _requested_database_path(
        root_dir=root_dir, database_path=database_path
    )
    safe_max_results = max(1, max_results)
    try:
        tokens = _tokenize_query(question)
        if not tokens:
            return _error_result(
                database_path=requested_path,
                error_type="empty_question",
                message="Question must include at least one meaningful search token.",
                max_results=safe_max_results,
            )

        resolved_path = resolve_db_path(
            root_dir=root_dir,
            database_path=database_path,
        )
        catalog = _database_catalog(resolved_path)
        suggestions = []
        for target in cast(list[dict[str, Any]], catalog["targets"]):
            kind = cast(str, target["kind"])
            if not include_internal and kind == "internal_catalog":
                continue
            suggestion = _target_suggestion(
                target,
                tokens=tokens,
            )
            if suggestion is None:
                continue
            suggestions.append(suggestion)

        suggestions.sort(
            key=lambda item: (-cast(int, item["score"]), cast(str, item["name"]))
        )
        top_suggestions = suggestions[:safe_max_results]
        return {
            "database_path": str(resolved_path),
            "status": "ok",
            "has_tabular_catalog": catalog["has_catalog"],
            "question": question,
            "tokens": tokens,
            "suggestion_count": len(top_suggestions),
            "suggestions": top_suggestions,
            "summary": f"Suggested {len(top_suggestions)} target(s) for {len(tokens)} search token(s).",
        }
    except ValueError as exc:
        return _error_result(
            database_path=requested_path,
            error_type="missing_database",
            message=str(exc),
            question=question,
            max_results=safe_max_results,
        )
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return _error_result(
            database_path=requested_path,
            error_type="sql_execution_error",
            message=str(exc),
            question=question,
            max_results=safe_max_results,
        )


def suggest_sql_error_repair(
    error_message: str,
    *,
    available_targets: list[str],
    target_columns: dict[str, list[str]],
    max_matches: int = MAX_REPAIR_CANDIDATES,
) -> list[dict[str, Any]]:
    """Return deterministic schema-aware repair hints for common SQLite errors."""
    safe_max_matches = max(1, max_matches)
    lowered_error = error_message.strip().lower()

    if lowered_error.startswith(_MISSING_COLUMN_ERROR_PREFIX):
        missing_column = _error_identifier(error_message, _MISSING_COLUMN_ERROR_PREFIX)
        columns_by_name: dict[str, list[str]] = {}
        for target_name, columns in target_columns.items():
            for column_name in columns:
                columns_by_name.setdefault(column_name, []).append(target_name)

        candidate_columns = _rank_identifier_candidates(
            missing_column,
            list(columns_by_name),
            max_matches=safe_max_matches,
        )
        if not candidate_columns:
            return []

        candidates = [
            {
                "name": column_name,
                "targets": sorted(columns_by_name[column_name]),
            }
            for column_name in candidate_columns
        ]
        return _repair_result(
            kind="missing_column",
            identifier=missing_column,
            candidates=candidates,
            message=(
                f"Column `{missing_column}` was not found. Closest inspected columns: {_format_repair_candidates(candidates, include_targets=True)}."
            ),
        )

    if lowered_error.startswith(_MISSING_TABLE_ERROR_PREFIX):
        missing_target = _error_identifier(error_message, _MISSING_TABLE_ERROR_PREFIX)
        candidate_targets = _rank_identifier_candidates(
            missing_target,
            available_targets,
            max_matches=safe_max_matches,
        )
        if not candidate_targets:
            return []

        candidates = [{"name": target_name} for target_name in candidate_targets]
        return _repair_result(
            kind="missing_target",
            identifier=missing_target,
            candidates=candidates,
            message=(
                f"Target `{missing_target}` was not found. Closest inspected targets: {_format_repair_candidates(candidates, include_targets=False)}."
            ),
        )

    if lowered_error.startswith(_AMBIGUOUS_COLUMN_ERROR_PREFIX):
        ambiguous_column = _error_identifier(
            error_message, _AMBIGUOUS_COLUMN_ERROR_PREFIX
        )
        matching_targets = sorted(
            target_name
            for target_name, columns in target_columns.items()
            if ambiguous_column in columns
        )
        if not matching_targets:
            return []

        candidates = [{"name": ambiguous_column, "targets": matching_targets}]
        return _repair_result(
            kind="ambiguous_column",
            identifier=ambiguous_column,
            candidates=candidates,
            message=f"Column `{ambiguous_column}` is ambiguous. Qualify it with one of: {', '.join(matching_targets)}.",
        )

    return []
