"""Source stage for Python image stats."""

from __future__ import annotations

from ..sources.arena_ai import get_arena_ai_image_stats
from ..sources.artificial_analysis import get_artificial_analysis_image_stats
from .types import ImageSourceData


def fetch_source_data() -> ImageSourceData:
    """Fetch source rows for Source-stage image stats selection."""
    artificial_analysis_payload = get_artificial_analysis_image_stats()
    arena_payload = get_arena_ai_image_stats()
    artificial_analysis_models = artificial_analysis_payload.get("data") or []
    arena_models = arena_payload.get("rows") or []
    return {
        "artificial_analysis_payload": artificial_analysis_payload,
        "arena_payload": arena_payload,
        "artificial_analysis_models_by_slug": {
            model["slug"]: model
            for model in artificial_analysis_models
            if isinstance(model, dict) and isinstance(model.get("slug"), str)
        },
        "arena_models_by_name": {
            model["model"]: model
            for model in arena_models
            if isinstance(model, dict) and isinstance(model.get("model"), str)
        },
    }
