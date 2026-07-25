"""Internal native Python matcher pipeline."""

from __future__ import annotations

from ..shared import FALLBACK_PROVIDER_IDS, PRIMARY_PROVIDER_ID, normalize_model_token
from .scoring import compare_candidates, has_first_token_match, score_candidate
from .tokenize import split_base_model_id

VOID_THRESHOLD_RANGE_RATIO = 0.35


def unique_model_count(models_dev_models: list[dict]) -> int:
    """Helper for unique model count."""
    return len({model.get("model_id") for model in models_dev_models})


def _has_exact_slug_fallback_candidate(
    source_slug: str, fallback_candidates: list[dict]
) -> bool:
    """Return whether any fallback candidate matches the source slug exactly."""
    normalized_source_slug = normalize_model_token(source_slug)
    if not normalized_source_slug:
        return False
    for candidate in fallback_candidates:
        candidate_slug = normalize_model_token(
            split_base_model_id(candidate["model_id"])
        )
        if candidate_slug and candidate_slug == normalized_source_slug:
            return True
    return False


def split_preferred_provider_models(models_dev_models: list[dict]) -> dict:
    """Split the preferred provider models."""
    return {
        "primary": [
            model
            for model in models_dev_models
            if model.get("provider_id") == PRIMARY_PROVIDER_ID
        ],
        "fallback": [
            model
            for model in models_dev_models
            if model.get("provider_id") in FALLBACK_PROVIDER_IDS
        ],
    }


def _collect_candidates_for_source_slug(
    source_slug: str, models_dev_models: list[dict]
) -> list[dict]:
    """Helper for collect candidates for source slug."""
    if not source_slug:
        return []
    candidates = []
    for models_dev_model in models_dev_models:
        model_name = (
            models_dev_model.get("model", {}).get("name")
            if isinstance(models_dev_model.get("model"), dict)
            else ""
        )
        model_name = model_name if isinstance(model_name, str) else ""
        if not has_first_token_match(
            source_slug, models_dev_model.get("model_id", ""), model_name
        ):
            continue
        candidate_score = score_candidate(
            source_slug, models_dev_model.get("model_id", ""), model_name
        )
        if candidate_score <= 0:
            continue
        candidates.append(
            {
                "model_id": models_dev_model.get("model_id", ""),
                "provider_id": models_dev_model.get("provider_id", ""),
                "provider_name": models_dev_model.get("provider_name", ""),
                "model_name": model_name or None,
                "score": candidate_score,
            }
        )
    return sorted(candidates, key=compare_candidates)


def _select_preferred_candidates_for_slug(
    source_slug: str, provider_pools: dict
) -> list[dict]:
    """Select the preferred candidates for slug."""
    primary_candidates = _collect_candidates_for_source_slug(
        source_slug, provider_pools["primary"]
    )
    fallback_candidates = _collect_candidates_for_source_slug(
        source_slug, provider_pools["fallback"]
    )
    if not primary_candidates:
        return fallback_candidates
    if _has_exact_slug_fallback_candidate(source_slug, fallback_candidates):
        return fallback_candidates
    return primary_candidates


def _apply_maxmin_half_void(models: list[dict]) -> tuple[float | None, int]:
    """Apply the maxmin half void."""
    scores = sorted(
        model["best_match"]["score"]
        for model in models
        if model.get("best_match") is not None
    )
    if not scores:
        return None, 0
    min_score = scores[0]
    max_score = scores[-1]
    threshold = min_score + (max_score - min_score) * VOID_THRESHOLD_RANGE_RATIO
    voided = 0
    for model in models:
        score = (
            model["best_match"]["score"]
            if model.get("best_match") is not None
            else None
        )
        if score is not None and score < threshold:
            model["best_match"] = None
            if isinstance(model.get("candidates"), list):
                model["candidates"] = []
            voided += 1
    return threshold, voided


def run_matcher(
    source_models: list[dict], provider_pools: dict, max_candidates: int
) -> dict:
    """Helper for run matcher."""
    models = []
    for source_model in source_models:
        candidates = _select_preferred_candidates_for_slug(
            source_model["source_slug"], provider_pools
        )[:max_candidates]
        models.append(
            {
                "artificial_analysis_slug": source_model["source_slug"],
                "artificial_analysis_name": source_model["source_name"],
                "artificial_analysis_release_date": source_model["source_release_date"],
                "best_match": candidates[0] if candidates else None,
                "candidates": candidates,
            }
        )
    pre_void_matched_count = len(
        [model for model in models if model["best_match"] is not None]
    )
    pre_void_unmatched_count = len(models) - pre_void_matched_count
    void_threshold, voided_count = _apply_maxmin_half_void(models)
    matched_count = len([model for model in models if model["best_match"] is not None])
    unmatched_count = len(models) - matched_count
    return {
        "models": models,
        "void_threshold": void_threshold,
        "voided_count": voided_count,
        "pre_void_matched_count": pre_void_matched_count,
        "pre_void_unmatched_count": pre_void_unmatched_count,
        "matched_count": matched_count,
        "unmatched_count": unmatched_count,
    }
