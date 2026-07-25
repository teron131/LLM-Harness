"""Final projection stage for Python image stats."""

from __future__ import annotations

import re
from typing import Any

from ...utils import as_record, mean_or_none
from .types import ImageUnionRow


def _to_model_id(value: str) -> str:
    """Normalize a model record to its id for Final-stage image stats selection."""
    normalized = re.sub(r"[._:/\s]+", "-", value.lower())
    normalized = re.sub(r"[^a-z0-9-]+", "", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-")


def _provider_from_arena_provider(value: Any) -> str | None:
    """Resolve the provider for Final-stage image stats selection."""
    if not isinstance(value, str):
        return None
    left = value.split("·")[0].strip()
    return left or None


def _build_logo(model: dict[str, Any], provider: str | None) -> str:
    """Build the logo field for Final-stage image stats selection."""
    artificial_analysis = as_record(model.get("artificial_analysis"))
    model_creator = as_record(artificial_analysis.get("model_creator"))
    logo_slug = model_creator.get("slug")
    if isinstance(logo_slug, str) and logo_slug:
        return f"https://artificialanalysis.ai/img/logos/{logo_slug}_small.svg"
    return f"https://models.dev/logos/{(provider or 'unknown').lower()}.svg"


def _pick_aa_percentiles(model: dict[str, Any]) -> dict[str, Any] | None:
    """Pick the aa percentiles."""
    percentiles = as_record(
        as_record(model.get("artificial_analysis")).get("percentiles")
    )
    return percentiles or None


def _pick_arena_percentiles(model: dict[str, Any]) -> dict[str, Any] | None:
    """Pick the arena percentiles."""
    percentiles = as_record(as_record(model.get("arena_ai")).get("percentiles"))
    return percentiles or None


def _pick_aa_scores(model: dict[str, Any]) -> dict[str, Any] | None:
    """Pick the aa scores."""
    weighted_scores = as_record(
        as_record(model.get("artificial_analysis")).get("weighted_scores")
    )
    return weighted_scores or None


def _pick_arena_scores(model: dict[str, Any]) -> dict[str, Any] | None:
    """Pick the arena scores."""
    weighted_scores = as_record(as_record(model.get("arena_ai")).get("weighted_scores"))
    return weighted_scores or None


def _map_union_model_to_selected(union_model: ImageUnionRow) -> dict[str, Any]:
    """Map a source model into the selected Final-stage image stats selection payload."""
    model = dict(union_model)
    artificial_analysis = as_record(model.get("artificial_analysis"))
    arena = as_record(model.get("arena_ai"))
    artificial_analysis_scores = _pick_aa_scores(model)
    arena_scores = _pick_arena_scores(model)
    artificial_analysis_percentiles = _pick_aa_percentiles(model)
    arena_percentiles = _pick_arena_percentiles(model)
    best_match = as_record(model.get("best_match"))
    inferred_id = _to_model_id(
        (
            artificial_analysis.get("slug")
            if isinstance(artificial_analysis.get("slug"), str)
            else None
        )
        or (
            artificial_analysis.get("name")
            if isinstance(artificial_analysis.get("name"), str)
            else None
        )
        or (arena.get("model") if isinstance(arena.get("model"), str) else None)
        or (
            best_match.get("arena_model")
            if isinstance(best_match.get("arena_model"), str)
            else None
        )
        or "unknown"
    )
    model_creator_name = as_record(artificial_analysis.get("model_creator")).get("name")
    provider = (
        model_creator_name
        if isinstance(model_creator_name, str)
        else _provider_from_arena_provider(arena.get("provider"))
    )
    photorealistic_score = mean_or_none(
        [
            artificial_analysis_scores.get("photorealistic")
            if artificial_analysis_scores
            else None,
            arena_scores.get("photorealistic") if arena_scores else None,
        ]
    )
    illustrative_score = mean_or_none(
        [
            artificial_analysis_scores.get("illustrative")
            if artificial_analysis_scores
            else None,
            arena_scores.get("illustrative") if arena_scores else None,
        ]
    )
    contextual_score = mean_or_none(
        [
            artificial_analysis_scores.get("contextual")
            if artificial_analysis_scores
            else None,
            arena_scores.get("contextual") if arena_scores else None,
        ]
    )
    overall_score = mean_or_none(
        [photorealistic_score, illustrative_score, contextual_score]
    )
    photorealistic_percentile = mean_or_none(
        [
            artificial_analysis_percentiles.get("photorealistic_percentile")
            if artificial_analysis_percentiles
            else None,
            arena_percentiles.get("photorealistic_percentile")
            if arena_percentiles
            else None,
        ]
    )
    illustrative_percentile = mean_or_none(
        [
            artificial_analysis_percentiles.get("illustrative_percentile")
            if artificial_analysis_percentiles
            else None,
            arena_percentiles.get("illustrative_percentile")
            if arena_percentiles
            else None,
        ]
    )
    contextual_percentile = mean_or_none(
        [
            artificial_analysis_percentiles.get("contextual_percentile")
            if artificial_analysis_percentiles
            else None,
            arena_percentiles.get("contextual_percentile")
            if arena_percentiles
            else None,
        ]
    )
    overall_percentile = mean_or_none(
        [photorealistic_percentile, illustrative_percentile, contextual_percentile]
    )
    return {
        "id": inferred_id or None,
        "name": (
            artificial_analysis.get("name")
            if isinstance(artificial_analysis.get("name"), str)
            else None
        )
        or (
            artificial_analysis.get("slug")
            if isinstance(artificial_analysis.get("slug"), str)
            else None
        )
        or (arena.get("model") if isinstance(arena.get("model"), str) else None)
        or (
            best_match.get("arena_model")
            if isinstance(best_match.get("arena_model"), str)
            else None
        ),
        "provider": provider,
        "logo": _build_logo(model, provider),
        "release_date": artificial_analysis.get("release_date")
        if isinstance(artificial_analysis.get("release_date"), str)
        else None,
        "sources": {
            "artificial_analysis": bool(artificial_analysis),
            "arena_ai": bool(arena),
        },
        "source_scores": {
            "artificial_analysis": artificial_analysis_scores,
            "arena_ai": arena_scores,
        },
        "source_percentiles": {
            "artificial_analysis": artificial_analysis_percentiles,
            "arena_ai": arena_percentiles,
        },
        "scores": {
            "photorealistic_score": photorealistic_score,
            "illustrative_score": illustrative_score,
            "contextual_score": contextual_score,
            "overall_score": overall_score,
        },
        "percentiles": {
            "photorealistic_percentile": photorealistic_percentile,
            "illustrative_percentile": illustrative_percentile,
            "contextual_percentile": contextual_percentile,
            "overall_percentile": overall_percentile,
        },
    }


def build_final_models(
    union_models: list[ImageUnionRow],
    model_id: str | None,
) -> list[dict[str, Any]]:
    """Build the final Final-stage image stats selection payload."""
    selected_models = sorted(
        [_map_union_model_to_selected(union_model) for union_model in union_models],
        key=lambda model: (
            model["scores"]["overall_score"]
            if model["scores"]["overall_score"] is not None
            else float("-inf")
        ),
        reverse=True,
    )
    if model_id is None:
        return selected_models
    return [model for model in selected_models if model.get("id") == model_id]
