"""Public SQL tool wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.tools import tool

from ..tabular.storage import resolve_root_dir
from .query import (
    MAX_QUERY_ROWS,
    MAX_SUGGESTED_TARGETS,
    describe_target,
    list_targets,
    run_query,
    save_view,
    suggest_targets,
)


def make_sql_tools(*, root_dir: str | Path | None = None):
    """Create standalone SQL navigation and query tools."""
    resolved_root_dir = resolve_root_dir(root_dir=root_dir)

    @tool(parse_docstring=True)
    def sql_list(
        database_path: str | None = None,
        include_internal: bool = False,
    ) -> dict[str, Any]:
        """List available tables and views in a SQLite database.

        Args:
            database_path: Optional path to a specific SQLite file. Defaults to the shared SQLite cache.
            include_internal: Whether to include internal catalog tables such as `_tabular_contents`.
        """
        return list_targets(
            root_dir=resolved_root_dir,
            database_path=database_path,
            include_internal=include_internal,
        )

    @tool(parse_docstring=True)
    def sql_describe(
        target_name: str,
        database_path: str | None = None,
        sample_rows: int = 3,
        text_value_hints: int = 3,
    ) -> dict[str, Any]:
        """Describe one table or view in a SQLite database.

        Args:
            target_name: Exact table or view name to inspect.
            database_path: Optional path to a specific SQLite file. Defaults to the shared SQLite cache.
            sample_rows: Number of preview rows to include in the description payload.
            text_value_hints: Number of text-like columns to include distinct example values for.
        """
        return describe_target(
            target_name,
            root_dir=resolved_root_dir,
            database_path=database_path,
            sample_rows=sample_rows,
            text_value_hints=text_value_hints,
        )

    @tool(parse_docstring=True)
    def sql_suggest(
        question: str,
        database_path: str | None = None,
        max_results: int = MAX_SUGGESTED_TARGETS,
        include_internal: bool = False,
    ) -> dict[str, Any]:
        """Suggest likely tables or views for a natural-language question.

        Args:
            question: Natural-language question or intent to match against target names and schema.
            database_path: Optional path to a specific SQLite file. Defaults to the shared SQLite cache.
            max_results: Maximum number of likely targets to return.
            include_internal: Whether to include internal catalog tables such as `_tabular_contents`.
        """
        return suggest_targets(
            question,
            root_dir=resolved_root_dir,
            database_path=database_path,
            max_results=max_results,
            include_internal=include_internal,
        )

    @tool(parse_docstring=True)
    def sql_query(
        sql: str,
        database_path: str | None = None,
        max_rows: int = MAX_QUERY_ROWS,
    ) -> dict[str, Any]:
        """Run SQL against a SQLite database.

        Args:
            sql: SQL statement to run.
            database_path: Optional path to a specific SQLite file. Defaults to the shared SQLite cache.
            max_rows: Maximum number of result rows to return for row-producing queries.
        """
        return run_query(
            sql,
            root_dir=resolved_root_dir,
            database_path=database_path,
            max_rows=max_rows,
        )

    @tool(parse_docstring=True)
    def sql_save(
        sql: str,
        view_name: str,
        database_path: str | None = None,
        replace: bool = False,
    ) -> dict[str, Any]:
        """Save a read-only SQL query as a named SQLite view.

        Args:
            sql: Read-only SQL statement to save as a view definition.
            view_name: Name for the SQLite view to create.
            database_path: Optional path to a specific SQLite file. Defaults to the shared SQLite cache.
            replace: Whether to replace an existing view with the same name.
        """
        return save_view(
            sql,
            view_name,
            root_dir=resolved_root_dir,
            database_path=database_path,
            replace=replace,
        )

    return [
        sql_list,
        sql_describe,
        sql_suggest,
        sql_query,
        sql_save,
    ]
