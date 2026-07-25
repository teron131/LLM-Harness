"""Match stage for native Python LLM stats."""

from __future__ import annotations

from typing import Any

from ..matcher import get_scraper_fallback_match_diagnostics
from ..shared import as_record, model_slug_from_model_id, normalize_provider_model_id
from .types import LlmSourceData


def _has_token(model_id: str, token: str) -> bool:
    """Return whether Matching-stage LLM stats selection has a matching token."""
    return token in model_id


def _canonical_model_id(
    model_id: Any, provider_id: Any, fallback_model_id: Any
) -> str | None:
    """Helper for canonical model id."""
    if isinstance(model_id, str) and "/" in model_id:
        return model_id
    if isinstance(provider_id, str) and isinstance(model_id, str):
        return f"{provider_id}/{model_id}"
    if isinstance(provider_id, str) and isinstance(fallback_model_id, str):
        return f"{provider_id}/{fallback_model_id}"
    return model_id if isinstance(model_id, str) else None


def _has_variant_conflict(
    artificial_analysis_slug: str,
    matched_model_id: str,
    matcher_options: dict[str, Any],
) -> bool:
    """Return whether Matching-stage LLM stats selection has a variant conflict."""
    aa = normalize_provider_model_id(artificial_analysis_slug)
    matched = normalize_provider_model_id(matched_model_id)
    return any(
        _has_token(aa, token) != _has_token(matched, token)
        for token in matcher_options.get("variant_tokens", [])
    )


def _build_matched_row_from_scraped_model(
    scraped_model: dict[str, Any],
    api_model: dict[str, Any] | None,
    matched_model_id: str,
    models_dev_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the matched row from scraped model."""
    artificial_analysis_model_id = (
        scraped_model.get("model_id")
        if isinstance(scraped_model.get("model_id"), str)
        else None
    )
    artificial_analysis_slug = model_slug_from_model_id(artificial_analysis_model_id)
    evaluations = as_record(scraped_model.get("evaluations"))
    intelligence = as_record(scraped_model.get("intelligence"))
    intelligence_index_cost = as_record(scraped_model.get("intelligence_index_cost"))
    logo = (
        scraped_model.get("logo")
        if isinstance(scraped_model.get("logo"), str)
        else None
    )
    api_evaluations = as_record((api_model or {}).get("evaluations"))
    api_intelligence = as_record((api_model or {}).get("intelligence"))
    api_intelligence_index_cost = as_record(
        (api_model or {}).get("intelligence_index_cost")
    )
    matched_models_dev = models_dev_by_id.get(matched_model_id)
    matched_model_fields = as_record(as_record(matched_models_dev).get("model"))
    canonical_id = _canonical_model_id(
        matched_model_fields.get("id") or matched_model_id,
        as_record(matched_models_dev).get("provider_id"),
        as_record(matched_models_dev).get("model_id"),
    )
    row = {
        key: value
        for key, value in matched_model_fields.items()
        if key not in {"id", "name", "family", "model_id", "slug"}
    }
    return {
        "id": canonical_id,
        "provider_id": as_record(matched_models_dev).get("provider_id"),
        "openrouter_id": matched_model_fields.get("id"),
        "name": matched_model_fields.get("name")
        if isinstance(matched_model_fields.get("name"), str)
        else artificial_analysis_model_id,
        "aa_id": artificial_analysis_model_id,
        "aa_slug": artificial_analysis_slug,
        "family": matched_model_fields.get("family"),
        "logo": logo,
        **row,
        "evaluations": {**api_evaluations, **evaluations},
        "intelligence": {**api_intelligence, **intelligence},
        "intelligence_index_cost": {
            **api_intelligence_index_cost,
            **intelligence_index_cost,
        },
    }


def build_matched_payload(
    source_data: LlmSourceData,
    matcher_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the matched Matching-stage LLM stats selection payload."""
    matcher_options = matcher_options or {
        "variant_tokens": ["flash-lite", "flash", "pro", "nano", "mini", "lite"]
    }
    models_dev_models = source_data["preferred_models_dev_models"]
    diagnostics = get_scraper_fallback_match_diagnostics(
        {
            "scraped_rows": source_data["scraped_rows"],
            "models_dev_models": models_dev_models,
        }
    )
    rows = []
    for matched_model in diagnostics.get("models", []):
        matched_model_id = as_record(matched_model.get("best_match")).get("model_id")
        if not isinstance(matched_model_id, str) or not matched_model_id:
            continue
        if _has_variant_conflict(
            matched_model["artificial_analysis_slug"], matched_model_id, matcher_options
        ):
            continue
        scraped_slug = matched_model["artificial_analysis_slug"]
        scraped_model = source_data["scraped_by_slug"].get(scraped_slug)
        if not scraped_model:
            continue
        rows.append(
            _build_matched_row_from_scraped_model(
                scraped_model,
                source_data["api_by_slug"].get(scraped_slug),
                matched_model_id,
                source_data["models_dev_by_id"],
            )
        )
    return {
        "artificial_analysis_fetched_at_epoch_seconds": source_data[
            "artificial_analysis_payload"
        ].get("fetched_at_epoch_seconds"),
        "models_dev_fetched_at_epoch_seconds": source_data["models_dev_payload"].get(
            "fetched_at_epoch_seconds"
        ),
        "models": rows,
    }
