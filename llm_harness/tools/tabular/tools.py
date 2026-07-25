"""Public tabular orchestration and tool wiring."""

from __future__ import annotations

from pathlib import Path
import statistics
from typing import Any

from langchain.tools import tool

from .ingestion import (
    MAX_FULL_EXTRACT_BYTES,
    MAX_FULL_PROFILE_BYTES,
    MAX_METADATA_ROWS,
    MAX_SAMPLE_ROWS,
    count_non_empty,
    is_blank,
    load_rows,
    preview_dimensions,
    stream_csv_profile,
    stream_csv_window,
    tabular_dimensions,
    tabular_summary,
    tabular_summary_from_counts,
)
from .segmentation import (
    compute_region_boxes,
    header_candidates,
    profile_region_boxes,
    segment_tabular_blocks,
)
from .storage import (
    fingerprint,
    fingerprint_from_samples,
    load_tables_into_sqlite,
    resolve_root_dir,
)


def _inspect_tabular_file(
    path: Path,
    *,
    start_row: int = 1,
    limit: int = 5,
    start_col: int = 1,
    end_col: int | None = None,
    sheet: str | None = None,
) -> dict[str, Any]:
    """Return a bounded raw grid window from a tabular file."""
    safe_start = max(1, start_row)
    safe_limit = max(1, limit)
    safe_start_col = max(1, start_col)
    if path.suffix.lower() == ".csv":
        selected_rows, summary = stream_csv_window(
            path,
            start_row=safe_start,
            limit=safe_limit,
            start_col=safe_start_col,
            end_col=end_col,
        )
        summary_payload = tabular_summary_from_counts(
            row_count=None,
            column_count=None,
            format_info={
                "format": summary["format"],
                "encoding": summary["encoding"],
                "delimiter": summary["delimiter"],
                "quotechar": summary["quotechar"],
                "sheet_names": [],
                "read_mode": summary["read_mode"],
            },
        )
    else:
        rows, format_info = load_rows(path, sheet=sheet)
        row_count, column_count = tabular_dimensions(rows)
        safe_end_col = column_count if end_col is None else max(safe_start_col, end_col)
        end_row = safe_start + safe_limit - 1
        selected_source_rows = rows[safe_start - 1 : end_row]
        selected_rows = [
            row[safe_start_col - 1 : safe_end_col] for row in selected_source_rows
        ]
        summary_payload = tabular_summary_from_counts(
            row_count=row_count,
            column_count=column_count,
            format_info={
                **format_info,
                "read_mode": "full_layout",
            },
        )

    preview_row_count, preview_column_count = preview_dimensions(selected_rows)

    return {
        "path": str(path),
        **summary_payload,
        "preview_row_count": preview_row_count,
        "preview_column_count": preview_column_count,
        "start_row": safe_start,
        "end_row": safe_start + preview_row_count - 1
        if selected_rows
        else safe_start - 1,
        "start_col": safe_start_col,
        "end_col": safe_start_col + preview_column_count - 1
        if selected_rows
        else safe_start_col - 1,
        "rows": selected_rows,
    }


def _profile_tabular_file(
    path: Path,
    *,
    max_sample_rows: int = MAX_SAMPLE_ROWS,
    sheet: str | None = None,
) -> dict[str, Any]:
    """Profile a tabular file with read-only structural hints."""
    if path.suffix.lower() == ".csv" and path.stat().st_size > MAX_FULL_PROFILE_BYTES:
        summary = stream_csv_profile(path, max_sample_rows=max_sample_rows)
        profile_fingerprint = fingerprint_from_samples(
            row_count=summary["row_count"],
            column_count=summary["column_count"],
            top_rows=summary["top_rows"],
            bottom_rows=summary["bottom_rows"],
            header_candidates=[],
            max_sample_rows=max_sample_rows,
        )
        return {
            "path": str(path),
            **{
                key: value
                for key, value in summary.items()
                if key not in {"top_rows", "bottom_rows"}
            },
            "fingerprint": profile_fingerprint,
        }

    rows, format_info = load_rows(path, sheet=sheet)
    region_boxes = compute_region_boxes(rows)
    detected_header_candidates = header_candidates(rows, region_boxes=region_boxes)
    non_empty_counts = [count_non_empty(row) for row in rows if not is_blank(row)]

    return {
        "path": str(path),
        **tabular_summary(rows, format_info),
        "fingerprint": fingerprint(
            rows,
            max_sample_rows=max_sample_rows,
            header_candidates=detected_header_candidates,
        ),
        "non_empty_row_count": len(non_empty_counts),
        "blank_row_count": len(rows) - len(non_empty_counts),
        "max_non_empty_cells_in_row": max(non_empty_counts, default=0),
        "median_non_empty_cells_per_non_blank_row": statistics.median(non_empty_counts)
        if non_empty_counts
        else 0,
        "sample_rows": rows[:max_sample_rows],
        "header_candidates": detected_header_candidates,
        "regions": profile_region_boxes(region_boxes),
    }


def _recover_tabular_blocks(
    path: Path,
    *,
    sample_rows: int = MAX_SAMPLE_ROWS,
    metadata_rows: int = MAX_METADATA_ROWS,
    sheet: str | None = None,
) -> dict[str, Any]:
    """Recover metadata and table blocks for future extraction backends."""
    rows, format_info = load_rows(path, sheet=sheet)
    metadata, tables = segment_tabular_blocks(
        rows,
        table_sample_rows=sample_rows,
        metadata_sample_rows=metadata_rows,
    )

    return {
        "path": str(path),
        **tabular_summary(rows, format_info),
        "metadata": metadata,
        "tables": tables,
    }


def _extract_tabular_file(
    path: Path,
    *,
    root_dir: str | Path | None = None,
    sample_rows: int = MAX_SAMPLE_ROWS,
    metadata_rows: int = MAX_METADATA_ROWS,
    sheet: str | None = None,
) -> dict[str, Any]:
    """Extract tables and load them into the shared SQLite cache."""
    if path.suffix.lower() == ".csv" and path.stat().st_size > MAX_FULL_EXTRACT_BYTES:
        raise ValueError(
            f"CSV extraction currently requires a full in-memory layout pass and is capped at {MAX_FULL_EXTRACT_BYTES} bytes for safety: {path}"
        )

    profile = _profile_tabular_file(path, max_sample_rows=sample_rows, sheet=sheet)
    recovered = _recover_tabular_blocks(
        path,
        sample_rows=sample_rows,
        metadata_rows=metadata_rows,
        sheet=sheet,
    )
    loaded = load_tables_into_sqlite(
        recovered,
        root_dir=root_dir,
        fingerprint=profile["fingerprint"],
    )

    return {
        "path": recovered["path"],
        "format": recovered["format"],
        "sheet_name": recovered.get("sheet_name"),
        "status": "loaded",
        "target_backend": "sqlite",
        "database_path": loaded["database_path"],
        "fingerprint": profile["fingerprint"],
        "recovered_table_count": len(recovered["tables"]),
        "recovered_metadata_block_count": len(recovered["metadata"]),
        "tables": loaded["tables"],
    }


def make_tabular_tools(*, root_dir: str | Path | None = None):
    """Create tabular inspect/profile/extract/query tools for CSV and XLSX files."""
    resolved_root_dir = resolve_root_dir(root_dir=root_dir)

    @tool(parse_docstring=True)
    def inspect_tabular(
        path: str,
        start_row: int = 1,
        limit: int = 5,
        start_col: int = 1,
        end_col: int | None = None,
        sheet: str | None = None,
    ) -> dict[str, Any]:
        """Inspect a CSV or XLSX file with a bounded raw grid window.

        Args:
            path: Path to the CSV or XLSX file to inspect.
            start_row: One-based row number where the preview window begins.
            limit: Maximum number of rows to return in the preview window.
            start_col: One-based column number where the preview window begins.
            end_col: Optional one-based column number where the preview window ends.
            sheet: Optional worksheet name for XLSX files. When omitted, the first sheet is used.
        """
        return _inspect_tabular_file(
            Path(path),
            start_row=start_row,
            limit=limit,
            start_col=start_col,
            end_col=end_col,
            sheet=sheet,
        )

    @tool(parse_docstring=True)
    def profile_tabular(
        path: str, max_sample_rows: int = MAX_SAMPLE_ROWS, sheet: str | None = None
    ) -> dict[str, Any]:
        """Profile a CSV or XLSX file with read-only structural hints.

        Args:
            path: Path to the CSV or XLSX file to profile.
            max_sample_rows: Maximum number of top rows to include in the profile sample.
            sheet: Optional worksheet name for XLSX files. When omitted, the first sheet is used.
        """
        return _profile_tabular_file(
            Path(path),
            max_sample_rows=max_sample_rows,
            sheet=sheet,
        )

    @tool(parse_docstring=True)
    def extract_tabular(
        path: str,
        sample_rows: int = MAX_SAMPLE_ROWS,
        metadata_rows: int = MAX_METADATA_ROWS,
        sheet: str | None = None,
    ) -> dict[str, Any]:
        """Extract tables from a CSV or XLSX file into the shared SQLite cache.

        Args:
            path: Path to the CSV or XLSX file to extract.
            sample_rows: Maximum number of rows to inspect while preparing extraction.
            metadata_rows: Maximum number of metadata rows to inspect while preparing extraction.
            sheet: Optional worksheet name for XLSX files. When omitted, the first sheet is used.
        """
        return _extract_tabular_file(
            Path(path),
            root_dir=resolved_root_dir,
            sample_rows=sample_rows,
            metadata_rows=metadata_rows,
            sheet=sheet,
        )

    return [
        inspect_tabular,
        profile_tabular,
        extract_tabular,
    ]
