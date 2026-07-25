"""Match stage for Python image stats."""

from __future__ import annotations

from typing import Any, cast

from ...utils import as_record
from ..matcher import ImageMatchMappedModel, get_image_match_model_mapping
from .types import ImageSourceData, ImageUnionRow


def _merge_image_row(
    source_data: ImageSourceData,
    mapped_model: ImageMatchMappedModel,
) -> ImageUnionRow:
    """Merge the image row."""
    best_match = mapped_model.get("best_match")
    best_match_payload = cast(dict[str, Any] | None, best_match)
    arena_model_name = best_match.get("arena_model") if best_match else None
    arena_model_name = arena_model_name if isinstance(arena_model_name, str) else None
    return {
        "artificial_analysis_slug": mapped_model.get("artificial_analysis_slug")
        if isinstance(mapped_model.get("artificial_analysis_slug"), str)
        else None,
        "artificial_analysis_name": mapped_model.get("artificial_analysis_name")
        if isinstance(mapped_model.get("artificial_analysis_name"), str)
        else None,
        "artificial_analysis_provider": mapped_model.get("artificial_analysis_provider")
        if isinstance(mapped_model.get("artificial_analysis_provider"), str)
        else None,
        "best_match": best_match_payload,
        "candidates": cast(list[dict[str, Any]], mapped_model.get("candidates") or []),
        "artificial_analysis": source_data["artificial_analysis_models_by_slug"].get(
            mapped_model.get("artificial_analysis_slug")
        )
        if isinstance(mapped_model.get("artificial_analysis_slug"), str)
        else None,
        "arena_ai": source_data["arena_models_by_name"].get(arena_model_name)
        if arena_model_name
        else None,
    }


def build_matched_rows(source_data: ImageSourceData) -> list[ImageUnionRow]:
    """Build matched rows for Matching-stage image stats selection."""
    mapping = get_image_match_model_mapping(
        {
            "artificial_analysis_models": source_data[
                "artificial_analysis_payload"
            ].get("data")
            or [],
            "arena_models": source_data["arena_payload"].get("rows") or [],
        }
    )
    mapped_models = [
        cast(ImageMatchMappedModel, model)
        for model in mapping.get("models") or []
        if isinstance(model, dict)
    ]
    matched_rows = [_merge_image_row(source_data, model) for model in mapped_models]
    matched_arena_names = {
        arena_model_name
        for model in mapped_models
        for arena_model_name in [as_record(model.get("best_match")).get("arena_model")]
        if isinstance(arena_model_name, str)
    }
    unmatched_arena_rows: list[ImageUnionRow] = [
        {
            "artificial_analysis_slug": None,
            "artificial_analysis_name": None,
            "artificial_analysis_provider": None,
            "best_match": None,
            "candidates": [],
            "artificial_analysis": None,
            "arena_ai": model,
        }
        for model in source_data["arena_payload"].get("rows") or []
        if isinstance(model, dict) and model.get("model") not in matched_arena_names
    ]
    return matched_rows + unmatched_arena_rows
