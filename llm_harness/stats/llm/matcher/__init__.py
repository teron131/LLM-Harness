"""Public native Python LLM matcher APIs."""

from __future__ import annotations

from typing import Any, cast

from ..sources.artificial_analysis_api import get_artificial_analysis_stats
from ..sources.artificial_analysis_scraper import (
    get_artificial_analysis_scraped_evals_only_stats,
)
from ..sources.models_dev import get_models_dev_stats
from .pipeline import run_matcher, split_preferred_provider_models, unique_model_count
from .source_model import (
    build_source_models_from_artificial_analysis,
    build_source_models_from_scraped_rows,
)
from .types import (
    LlmMatchModelMappingOptions,
    LlmMatchModelMappingPayload,
    LlmScraperFallbackMatchDiagnosticsPayload,
)

DEFAULT_MAX_CANDIDATES = 5


def get_match_model_mapping(
    options: LlmMatchModelMappingOptions | None = None,
) -> LlmMatchModelMappingPayload:
    """Return match model mapping."""
    options = options or {}
    max_candidates = options.get("max_candidates", DEFAULT_MAX_CANDIDATES)
    artificial_analysis_stats = (
        {
            "fetched_at_epoch_seconds": None,
            "models": options.get("artificial_analysis_models"),
        }
        if options.get("artificial_analysis_models") is not None
        else get_artificial_analysis_stats()
    )
    models_dev_stats = (
        {
            "fetched_at_epoch_seconds": None,
            "models": options.get("models_dev_models"),
        }
        if options.get("models_dev_models") is not None
        else get_models_dev_stats()
    )
    models_dev_models = cast(list[dict[str, Any]], models_dev_stats.get("models") or [])
    artificial_analysis_models = cast(list[dict[str, Any]], artificial_analysis_stats.get("models") or [])
    provider_pools = split_preferred_provider_models(models_dev_models)
    total_scoped_models = unique_model_count(provider_pools["primary"] + provider_pools["fallback"])
    source_models = build_source_models_from_artificial_analysis(artificial_analysis_models)
    matcher_output = run_matcher(source_models, provider_pools, max_candidates)
    return cast(
        LlmMatchModelMappingPayload,
        {
            "artificial_analysis_fetched_at_epoch_seconds": artificial_analysis_stats.get("fetched_at_epoch_seconds")
            if isinstance(artificial_analysis_stats.get("fetched_at_epoch_seconds"), int)
            else None,
            "models_dev_fetched_at_epoch_seconds": models_dev_stats.get("fetched_at_epoch_seconds") if isinstance(models_dev_stats.get("fetched_at_epoch_seconds"), int) else None,
            "total_artificial_analysis_models": len(matcher_output["models"]),
            "total_models_dev_models": total_scoped_models,
            "max_candidates": max_candidates,
            "void_mode": "maxmin_half",
            "void_threshold": matcher_output["void_threshold"],
            "voided_count": matcher_output["voided_count"],
            "models": matcher_output["models"],
        },
    )


def get_scraper_fallback_match_diagnostics(
    options: LlmMatchModelMappingOptions | None = None,
) -> LlmScraperFallbackMatchDiagnosticsPayload:
    """Return scraper fallback match diagnostics."""
    options = options or {}
    max_candidates = options.get("max_candidates", DEFAULT_MAX_CANDIDATES)
    scraped_stats = (
        {
            "fetched_at_epoch_seconds": None,
            "data": options.get("scraped_rows"),
        }
        if options.get("scraped_rows") is not None
        else get_artificial_analysis_scraped_evals_only_stats()
    )
    models_dev_stats = (
        {
            "fetched_at_epoch_seconds": None,
            "models": options.get("models_dev_models"),
        }
        if options.get("models_dev_models") is not None
        else get_models_dev_stats()
    )
    models_dev_models = cast(list[dict[str, Any]], models_dev_stats.get("models") or [])
    scraped_rows = cast(list[dict[str, Any]], scraped_stats.get("data") or [])
    provider_pools = split_preferred_provider_models(models_dev_models)
    total_scoped_models = unique_model_count(provider_pools["primary"] + provider_pools["fallback"])
    source_models = build_source_models_from_scraped_rows(scraped_rows)
    matcher_output = run_matcher(source_models, provider_pools, max_candidates)
    return cast(
        LlmScraperFallbackMatchDiagnosticsPayload,
        {
            "scraped_fetched_at_epoch_seconds": scraped_stats.get("fetched_at_epoch_seconds") if isinstance(scraped_stats.get("fetched_at_epoch_seconds"), int) else None,
            "models_dev_fetched_at_epoch_seconds": models_dev_stats.get("fetched_at_epoch_seconds") if isinstance(models_dev_stats.get("fetched_at_epoch_seconds"), int) else None,
            "total_scraped_models": len(scraped_rows),
            "total_models_dev_models": total_scoped_models,
            "max_candidates": max_candidates,
            "pre_void_matched_count": matcher_output["pre_void_matched_count"],
            "pre_void_unmatched_count": matcher_output["pre_void_unmatched_count"],
            "void_mode": "maxmin_half",
            "void_threshold": matcher_output["void_threshold"],
            "voided_count": matcher_output["voided_count"],
            "matched_count": matcher_output["matched_count"],
            "unmatched_count": matcher_output["unmatched_count"],
            "models": matcher_output["models"],
        },
    )
