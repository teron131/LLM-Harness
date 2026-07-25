"""Table and metadata segmentation logic for normalized row grids."""

from __future__ import annotations

import statistics
from typing import Any, TypedDict, cast

import numpy as np
from scipy import ndimage

from .ingestion import count_non_empty, is_blank, preview_rows


class RegionBox(TypedDict):
    """Represent one rectangular region in a segmented table image."""

    row_start: int
    row_end: int
    column_start: int
    column_end: int


type RegionSlice = tuple[slice, slice]


def _looks_like_header(row: list[str]) -> bool:
    """Heuristically detect whether a row looks like a header."""
    non_empty_cells = [cell.strip() for cell in row if cell.strip()]
    if len(non_empty_cells) < 2:
        return False

    alpha_like = sum(any(char.isalpha() for char in cell) for cell in non_empty_cells)
    numeric_like = sum(cell.replace(".", "", 1).isdigit() for cell in non_empty_cells)
    return alpha_like >= max(2, len(non_empty_cells) - 1) and numeric_like <= 1


def _looks_like_header_prefix_row(row: list[str]) -> bool:
    """Detect short prefix rows that often appear above a stronger header."""
    non_empty_cells = [cell.strip() for cell in row if cell.strip()]
    if len(non_empty_cells) < 2:
        return False
    short_cells = sum(len(cell) <= 10 for cell in non_empty_cells)
    return short_cells >= max(2, round(len(non_empty_cells) * 0.75))


def _active_columns(row: list[str]) -> list[int]:
    """Return indexes of non-empty cells in a row."""
    return [index for index, cell in enumerate(row) if cell.strip()]


def _collect_followers(
    rows: list[list[str]],
    header_index: int,
    *,
    limit: int = 3,
) -> list[list[str]]:
    """Collect the non-blank rows immediately following a candidate header."""
    followers: list[list[str]] = []
    for index in range(header_index + 1, len(rows)):
        row = rows[index]
        if is_blank(row):
            break
        followers.append(row)
        if len(followers) >= limit:
            break
    return followers


def _is_table_header_at(
    rows: list[list[str]],
    header_index: int,
) -> bool:
    """Check whether a row can anchor a table block."""
    row = rows[header_index]
    if is_blank(row) or not _looks_like_header(row):
        return False

    followers = _collect_followers(rows, header_index)
    if not followers:
        return False

    row_non_empty = count_non_empty(row)
    expected_width = max(len(row), *(len(follower) for follower in followers))
    consistent_followers = [
        follower
        for follower in followers
        if len(follower) == expected_width and count_non_empty(follower) >= 2
    ]
    min_followers = 2 if row_non_empty <= 3 else 1
    return len(consistent_followers) >= min_followers


def _has_stronger_header_ahead(
    rows: list[list[str]],
    header_index: int,
    *,
    window: int = 6,
) -> bool:
    """Check whether a better table header appears shortly after this row."""
    current_non_empty = count_non_empty(rows[header_index])
    current_row = rows[header_index]
    end_index = min(len(rows), header_index + window + 1)
    for index in range(header_index + 1, end_index):
        row = rows[index]
        if is_blank(row):
            continue
        stronger_non_empty = count_non_empty(row)
        coverage_gain = sum(
            not current_cell.strip() and bool(future_cell.strip())
            for current_cell, future_cell in zip(current_row, row, strict=False)
        )
        if _looks_like_header_prefix_row(current_row) and _is_table_header_at(
            rows, index
        ):
            return True
        if _is_table_header_at(rows, index) and (
            stronger_non_empty >= current_non_empty + 4 or coverage_gain >= 2
        ):
            return True
    return False


def _data_row_threshold(rows: list[list[str]], header_index: int) -> int:
    """Estimate a minimum non-empty-cell threshold for follower rows."""
    followers = _collect_followers(rows, header_index, limit=5)
    non_empty_counts = [
        count_non_empty(row) for row in followers if count_non_empty(row) > 0
    ]
    if not non_empty_counts:
        return 2
    median_non_empty = statistics.median(non_empty_counts)
    return max(2, round(median_non_empty * 0.35))


def _build_component_mask(rows: list[list[str]]) -> np.ndarray:
    """Build a binary occupancy mask for non-empty cells."""
    row_count = len(rows)
    column_count = max((len(row) for row in rows), default=0)
    mask = np.zeros((row_count, column_count), dtype=bool)

    for row_index, row in enumerate(rows):
        for column_index in _active_columns(row):
            mask[row_index, column_index] = True

    if not mask.size:
        return mask

    mask = ndimage.binary_closing(mask, structure=np.ones((2, 1), dtype=bool))
    mask = ndimage.binary_fill_holes(mask)
    return mask


def compute_region_boxes(rows: list[list[str]]) -> list[RegionBox]:
    """Compute connected region boxes from the occupancy mask."""
    component_mask = _build_component_mask(rows)
    if not component_mask.size:
        return []

    labeled, component_count = ndimage.label(
        component_mask,
        structure=np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
    )
    slices = ndimage.find_objects(labeled)
    region_boxes: list[RegionBox] = []

    for label_id in range(1, component_count + 1):
        component_slice = cast(RegionSlice | None, slices[label_id - 1])
        if component_slice is None:
            continue
        row_slice, column_slice = component_slice
        region_boxes.append(
            RegionBox(
                row_start=row_slice.start,
                row_end=row_slice.stop - 1,
                column_start=column_slice.start,
                column_end=column_slice.stop - 1,
            )
        )

    return region_boxes


def _find_header_region_boxes(
    region_boxes: list[RegionBox],
    *,
    header_index: int,
    header_active_columns: list[int],
) -> list[RegionBox]:
    """Find region boxes that intersect the candidate header row."""
    if not header_active_columns:
        return []

    return sorted(
        [
            box
            for box in region_boxes
            if box["row_start"] <= header_index <= box["row_end"]
            and any(
                box["column_start"] <= column <= box["column_end"]
                for column in header_active_columns
            )
        ],
        key=lambda box: (box["column_start"], box["row_start"]),
    )


def _matches_table_span(
    row: list[str],
    *,
    column_start: int,
    column_end: int,
    anchor_column: int,
    required_columns: set[int],
    minimum_cells: int,
) -> bool:
    """Check whether a row fits an established table span."""
    active_columns = [
        column
        for column in _active_columns(row)
        if column_start <= column <= column_end
    ]
    if len(active_columns) < minimum_cells:
        return False
    if anchor_column not in active_columns:
        return False
    return bool(required_columns.intersection(active_columns))


def _supported_columns_in_box(
    rows: list[list[str]],
    *,
    row_start: int,
    row_end: int,
    column_start: int,
    column_end: int,
) -> set[int]:
    """Find columns with repeated support inside a region box."""
    support_counts = dict.fromkeys(range(column_start, column_end + 1), 0)
    for index in range(row_start, row_end + 1):
        row = rows[index]
        if is_blank(row):
            continue
        for column in _active_columns(row):
            if column_start <= column <= column_end:
                support_counts[column] += 1
    return {column for column, count in support_counts.items() if count >= 2}


def _extend_table(
    rows: list[list[str]],
    header_index: int,
    *,
    region_box: RegionBox | None = None,
) -> int | None:
    """Extend a table downward from its header until the structure breaks."""
    followers = _collect_followers(rows, header_index, limit=5)
    full_header_active_columns = _active_columns(rows[header_index])
    expected_width = max(
        len(rows[header_index]), *(len(follower) for follower in followers)
    )
    column_start = (
        region_box["column_start"]
        if region_box
        else (min(full_header_active_columns) if full_header_active_columns else 0)
    )
    column_end = (
        region_box["column_end"]
        if region_box
        else (
            max(full_header_active_columns)
            if full_header_active_columns
            else expected_width - 1
        )
    )
    header_active_columns = [
        column
        for column in full_header_active_columns
        if column_start <= column <= column_end
    ]
    if not header_active_columns:
        return None
    max_row_end = region_box["row_end"] if region_box else len(rows) - 1
    anchor_column = column_start
    supported_columns = (
        _supported_columns_in_box(
            rows,
            row_start=header_index + 1,
            row_end=max_row_end,
            column_start=column_start,
            column_end=column_end,
        )
        if region_box
        else set()
    )
    required_columns = (supported_columns - {anchor_column}) or set(
        header_active_columns[1:] or header_active_columns
    )
    minimum_cells = 2 if region_box else _data_row_threshold(rows, header_index)
    end_index = header_index

    for index in range(header_index + 1, min(len(rows), max_row_end + 1)):
        row = rows[index]
        if is_blank(row):
            break
        if len(row) != expected_width:
            break
        if not _matches_table_span(
            row,
            column_start=column_start,
            column_end=column_end,
            anchor_column=anchor_column,
            required_columns=required_columns,
            minimum_cells=minimum_cells,
        ):
            break
        end_index = index

    if end_index == header_index:
        return None
    return end_index


def _build_metadata_block(
    rows: list[list[str]],
    start_index: int,
    end_index: int,
    *,
    sample_rows: int | None,
) -> dict[str, Any]:
    """Build a metadata block summary for a row range."""
    block_rows = rows[start_index : end_index + 1]
    return {
        "row_start": start_index + 1,
        "row_end": end_index + 1,
        "row_count": len(block_rows),
        "rows": preview_rows(block_rows, sample_rows),
    }


def _flush_metadata_block(
    metadata: list[dict[str, Any]],
    rows: list[list[str]],
    metadata_start: int | None,
    end_index: int,
    *,
    sample_rows: int | None,
) -> None:
    """Append the current metadata block when a range is open."""
    if metadata_start is None or end_index < metadata_start:
        return
    metadata.append(
        _build_metadata_block(
            rows,
            metadata_start,
            end_index,
            sample_rows=sample_rows,
        )
    )


def _normalize_column_names(header: list[str]) -> list[str]:
    """Normalize header cells into non-empty column names."""
    column_names: list[str] = []
    for index, header_cell in enumerate(header):
        primary = header_cell.strip()
        if index == 0 and not primary:
            column_names.append("row_label")
            continue
        if primary:
            column_names.append(primary)
            continue
        column_names.append(f"column_{index + 1}")
    return column_names


def _build_table_block(
    rows: list[list[str]],
    start_index: int,
    end_index: int,
    *,
    region_box: RegionBox | None,
    table_index: int,
    sample_rows: int | None,
) -> dict[str, Any]:
    """Build a table block summary for a detected region."""
    column_start = region_box["column_start"] if region_box else 0
    column_end = (
        region_box["column_end"]
        if region_box
        else max((len(row) for row in rows[start_index : end_index + 1]), default=0) - 1
    )
    header = rows[start_index][column_start : column_end + 1]
    data_rows = [
        row[column_start : column_end + 1]
        for row in rows[start_index + 1 : end_index + 1]
    ]

    return {
        "name": f"table_{table_index}",
        "row_start": start_index + 1,
        "row_end": end_index + 1,
        "row_count": len(data_rows),
        "header_row": start_index + 1,
        "column_start": column_start + 1,
        "column_end": column_end + 1,
        "columns": _normalize_column_names(header),
        "rows": preview_rows(data_rows, sample_rows),
    }


def _build_tables_for_header(
    rows: list[list[str]],
    header_index: int,
    *,
    region_boxes: list[RegionBox],
    table_sample_rows: int | None,
    table_index: int,
) -> list[dict[str, Any]]:
    """Build all table blocks that share the same header row."""
    header_region_boxes = _find_header_region_boxes(
        region_boxes,
        header_index=header_index,
        header_active_columns=_active_columns(rows[header_index]),
    )
    built_tables: list[dict[str, Any]] = []

    for region_box in header_region_boxes or [None]:
        table_end = _extend_table(rows, header_index, region_box=region_box)
        if table_end is None:
            continue
        built_tables.append(
            _build_table_block(
                rows,
                header_index,
                table_end,
                region_box=region_box,
                table_index=table_index + len(built_tables),
                sample_rows=table_sample_rows,
            )
        )

    return built_tables


def _has_stronger_header_successor(
    rows: list[list[str]],
    header_index: int,
    *,
    row_end: int,
    window: int = 6,
) -> bool:
    """Return whether a stronger header appears shortly after this row."""
    current_row = rows[header_index]
    current_non_empty = count_non_empty(current_row)
    last_index = min(row_end, header_index + window)

    for future_index in range(header_index + 1, last_index + 1):
        future_row = rows[future_index]
        if is_blank(future_row) or not _is_table_header_at(rows, future_index):
            continue
        future_non_empty = count_non_empty(future_row)
        coverage_gain = sum(
            not current_cell.strip() and bool(future_cell.strip())
            for current_cell, future_cell in zip(current_row, future_row, strict=False)
        )
        if future_non_empty >= current_non_empty + 2 or coverage_gain >= 2:
            return True

    return False


def _candidate_header_index(
    rows: list[list[str]],
    *,
    row_start: int,
    row_end: int,
) -> int | None:
    """Select the topmost stable header row for a region."""
    fallback_index: int | None = None

    for row_index in range(max(0, row_start - 1), row_end + 1):
        row = rows[row_index]
        if is_blank(row) or not _is_table_header_at(rows, row_index):
            continue
        if fallback_index is None:
            fallback_index = row_index
        if not _has_stronger_header_successor(rows, row_index, row_end=row_end):
            return row_index

    return fallback_index


def header_candidates(
    rows: list[list[str]],
    *,
    region_boxes: list[RegionBox],
) -> list[dict[str, Any]]:
    """Collect the topmost likely header row for each detected region."""
    candidates: list[dict[str, Any]] = []
    seen_rows: set[int] = set()

    for box in region_boxes:
        header_index = _candidate_header_index(
            rows,
            row_start=box["row_start"],
            row_end=box["row_end"],
        )
        if header_index is None or header_index in seen_rows:
            continue

        row = rows[header_index]
        seen_rows.add(header_index)
        candidates.append(
            {
                "row": header_index + 1,
                "non_empty_cells": count_non_empty(row),
                "has_stronger_header_ahead": _has_stronger_header_ahead(
                    rows, header_index
                ),
                "values": row,
            }
        )

    return candidates


def profile_region_boxes(region_boxes: list[RegionBox]) -> list[dict[str, int]]:
    """Convert region boxes into read-only profile hints."""
    return [
        {
            "row_start": box["row_start"] + 1,
            "row_end": box["row_end"] + 1,
            "column_start": box["column_start"] + 1,
            "column_end": box["column_end"] + 1,
        }
        for box in region_boxes
    ]


def segment_tabular_blocks(
    rows: list[list[str]],
    *,
    table_sample_rows: int | None,
    metadata_sample_rows: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split loaded rows into metadata blocks and table blocks."""
    region_boxes = compute_region_boxes(rows)
    metadata: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    metadata_start: int | None = None
    index = 0
    table_index = 1

    while index < len(rows):
        row = rows[index]
        if is_blank(row):
            _flush_metadata_block(
                metadata,
                rows,
                metadata_start,
                index - 1,
                sample_rows=metadata_sample_rows,
            )
            metadata_start = None
            index += 1
            continue

        if not _is_table_header_at(rows, index) or _has_stronger_header_ahead(
            rows, index
        ):
            if metadata_start is None:
                metadata_start = index
            index += 1
            continue

        built_tables = _build_tables_for_header(
            rows,
            index,
            region_boxes=region_boxes,
            table_sample_rows=table_sample_rows,
            table_index=table_index,
        )
        if not built_tables:
            if metadata_start is None:
                metadata_start = index
            index += 1
            continue

        _flush_metadata_block(
            metadata,
            rows,
            metadata_start,
            index - 1,
            sample_rows=metadata_sample_rows,
        )
        metadata_start = None
        tables.extend(built_tables)
        table_index += len(built_tables)
        index = max(table["row_end"] for table in built_tables)

    _flush_metadata_block(
        metadata,
        rows,
        metadata_start,
        len(rows) - 1,
        sample_rows=metadata_sample_rows,
    )

    return metadata, tables
