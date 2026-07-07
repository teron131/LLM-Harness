"""Source fetch stage for native Python LLM stats."""

from __future__ import annotations

from typing import Any, cast

from ..shared import FALLBACK_PROVIDER_IDS, PRIMARY_PROVIDER_ID, model_slug_from_model_id
from ..sources.artificial_analysis_api import get_artificial_analysis_stats
from ..sources.artificial_analysis_scraper import (
    get_artificial_analysis_scraped_evals_only_stats,
)
from ..sources.models_dev import get_models_dev_stats
from .types import LlmSourceData, LlmSourceDataModel


def _dedupe_preferred_provider_models(models_dev_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe the preferred provider models."""
    preferred_models = [model for model in models_dev_models if model.get("provider_id") == PRIMARY_PROVIDER_ID or model.get("provider_id") in FALLBACK_PROVIDER_IDS]
    with_priority = sorted(
        preferred_models,
        key=lambda model: 0 if model.get("provider_id") == PRIMARY_PROVIDER_ID else 1,
    )
    by_model_id: dict[str, dict[str, Any]] = {}
    for model in with_priority:
        model_id = model.get("model_id")
        if isinstance(model_id, str) and model_id not in by_model_id:
            by_model_id[model_id] = model
    return list(by_model_id.values())


def fetch_source_data() -> LlmSourceData:
    """Fetch source rows for Source-stage LLM stats selection."""
    artificial_analysis_payload = get_artificial_analysis_stats()
    scraped_payload = get_artificial_analysis_scraped_evals_only_stats()
    models_dev_payload = get_models_dev_stats()
    preferred_models_dev_models = _dedupe_preferred_provider_models(
        models_dev_payload.get("models") or [],
    )
    scraped_rows = [row for row in scraped_payload.get("data") or [] if isinstance(row, dict)]
    return cast(
        LlmSourceData,
        LlmSourceDataModel.model_validate(
            {
                "artificial_analysis_payload": artificial_analysis_payload,
                "scraped_payload": scraped_payload,
                "models_dev_payload": models_dev_payload,
                "scraped_rows": scraped_rows,
                "preferred_models_dev_models": preferred_models_dev_models,
                "models_dev_by_id": {model["model_id"]: model for model in preferred_models_dev_models if isinstance(model.get("model_id"), str)},
                "api_by_slug": {
                    model["slug"]: model for model in artificial_analysis_payload.get("models") or [] if isinstance(model, dict) and isinstance(model.get("slug"), str)
                },
                "scraped_by_slug": {slug: row for row in scraped_rows if (slug := model_slug_from_model_id(row.get("model_id"))) is not None},
            }
        ).model_dump(),
    )
