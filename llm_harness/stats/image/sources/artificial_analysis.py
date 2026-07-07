"""Artificial Analysis image source (native Python implementation)."""

from __future__ import annotations

from typing import Any, TypedDict

from ...utils import fetch_with_timeout, now_epoch_seconds, percentile_rank

TEXT_TO_IMAGE_URL = "https://artificialanalysis.ai/api/v2/data/media/text-to-image?include_categories=true"
REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_MIN_MODEL_AGE_DAYS = 365
GROUPS = {
    "Photorealistic": [
        "General & Photorealistic",
        "People: Portraits",
        "Physical Spaces",
        "Nature & Landscapes",
        "Vintage & Retro",
        "People: Groups & Activities",
    ],
    "Illustrative": [
        "Cartoon & Illustration",
        "Anime",
        "Futuristic & Sci-Fi",
        "Fantasy & Mythical",
        "Traditional Art",
    ],
    "Contextual": [
        "Text & Typography",
        "UI/UX Design",
        "Commercial",
        "Graphic Design & Digital Rendering",
    ],
}
GROUP_BY_CATEGORY_LABEL = {label: group_name for group_name, labels in GROUPS.items() for label in labels}


class ArtificialAnalysisImageOptions(TypedDict, total=False):
    """Artificial Analysis image source options."""

    api_key: str
    min_model_age_days: int


def _parse_release_date_to_utc(release_date: str | None) -> float | None:
    """Parse the release date to utc."""
    if not release_date:
        return None
    if len(release_date) == 7:
        year_text, month_text = release_date.split("-")
        try:
            year = int(year_text)
            month = int(month_text)
        except ValueError:
            return None
        return float(__import__("datetime").datetime(year, month, 1).timestamp())
    if len(release_date) == 10:
        try:
            return float(__import__("datetime").datetime.fromisoformat(release_date).timestamp())
        except ValueError:
            return None
    return None


def _is_older_than_days(release_date: str | None, min_age_days: int) -> bool:
    """Return whether a release date is older than the requested day window."""
    release_ts = _parse_release_date_to_utc(release_date)
    if release_ts is None:
        return False
    age_seconds = now_epoch_seconds() - release_ts
    return age_seconds > min_age_days * 24 * 60 * 60


def _detect_group(category: dict[str, Any]) -> str | None:
    """Detect the group."""
    style = category.get("style_category")
    subject = category.get("subject_matter_category")
    if isinstance(style, str) and style in GROUP_BY_CATEGORY_LABEL:
        return GROUP_BY_CATEGORY_LABEL[style]
    if isinstance(subject, str) and subject in GROUP_BY_CATEGORY_LABEL:
        return GROUP_BY_CATEGORY_LABEL[subject]
    return None


def _init_accumulator() -> dict[str, dict[str, float]]:
    """Initialize an accumulator for Artificial Analysis image benchmark source."""
    return {
        "Photorealistic": {"weighted_elo_sum": 0.0, "appearance_sum": 0.0},
        "Illustrative": {"weighted_elo_sum": 0.0, "appearance_sum": 0.0},
        "Contextual": {"weighted_elo_sum": 0.0, "appearance_sum": 0.0},
    }


def _frequency_weighted_elo(weighted_elo_sum: float, appearance_sum: float) -> float | None:
    """Helper for frequency weighted elo."""
    if appearance_sum <= 0:
        return None
    return round(weighted_elo_sum / appearance_sum, 4)


def _to_aggregated_fields(accumulator: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Convert raw rows into aggregated fields for Artificial Analysis image benchmark source."""
    photorealistic = _frequency_weighted_elo(
        accumulator["Photorealistic"]["weighted_elo_sum"],
        accumulator["Photorealistic"]["appearance_sum"],
    )
    illustrative = _frequency_weighted_elo(
        accumulator["Illustrative"]["weighted_elo_sum"],
        accumulator["Illustrative"]["appearance_sum"],
    )
    contextual = _frequency_weighted_elo(
        accumulator["Contextual"]["weighted_elo_sum"],
        accumulator["Contextual"]["appearance_sum"],
    )
    total_known_appearances = sum(bucket["appearance_sum"] for bucket in accumulator.values())
    grouped_overall = _frequency_weighted_elo(
        sum(bucket["weighted_elo_sum"] for bucket in accumulator.values()),
        total_known_appearances,
    )
    return {
        "weighted_scores": {
            "photorealistic": photorealistic,
            "illustrative": illustrative,
            "contextual": contextual,
            "grouped_overall": grouped_overall,
        },
        "aggregated_frequencies": {
            "appearances": {
                "photorealistic": int(accumulator["Photorealistic"]["appearance_sum"]),
                "illustrative": int(accumulator["Illustrative"]["appearance_sum"]),
                "contextual": int(accumulator["Contextual"]["appearance_sum"]),
            },
            "total_known_appearances": int(total_known_appearances),
        },
    }


def _enrich_payload(raw_payload: dict[str, Any], min_model_age_days: int) -> dict[str, Any]:
    """Enrich the selected Artificial Analysis image benchmark source payload."""
    raw_models = raw_payload.get("data")
    all_models = raw_models if isinstance(raw_models, list) else []
    models = [model for model in all_models if isinstance(model, dict) and not _is_older_than_days(model.get("release_date"), min_model_age_days)]
    global_accumulator = _init_accumulator()
    enriched_models: list[dict[str, Any]] = []
    for model in models:
        local_accumulator = _init_accumulator()
        categories = model.get("categories")
        categories = categories if isinstance(categories, list) else []
        for category in categories:
            if not isinstance(category, dict):
                continue
            group = _detect_group(category)
            if group is None:
                continue
            try:
                elo = float(category.get("elo"))
                appearances = float(category.get("appearances"))
            except (TypeError, ValueError):
                continue
            if appearances <= 0:
                continue
            local_accumulator[group]["weighted_elo_sum"] += elo * appearances
            local_accumulator[group]["appearance_sum"] += appearances
            global_accumulator[group]["weighted_elo_sum"] += elo * appearances
            global_accumulator[group]["appearance_sum"] += appearances
        enriched_models.append(
            {
                **model,
                **_to_aggregated_fields(local_accumulator),
                "percentiles": {
                    "photorealistic_percentile": None,
                    "illustrative_percentile": None,
                    "contextual_percentile": None,
                    "grouped_overall_percentile": None,
                },
            }
        )
    enriched_models.sort(
        key=lambda model: model["weighted_scores"]["grouped_overall"] if model["weighted_scores"]["grouped_overall"] is not None else float("-inf"),
        reverse=True,
    )
    photorealistic_values = [model["weighted_scores"]["photorealistic"] for model in enriched_models]
    illustrative_values = [model["weighted_scores"]["illustrative"] for model in enriched_models]
    contextual_values = [model["weighted_scores"]["contextual"] for model in enriched_models]
    grouped_overall_values = [model["weighted_scores"]["grouped_overall"] for model in enriched_models]
    data = []
    for model in enriched_models:
        data.append(
            {
                **model,
                "percentiles": {
                    "photorealistic_percentile": percentile_rank(photorealistic_values, model["weighted_scores"]["photorealistic"]),
                    "illustrative_percentile": percentile_rank(illustrative_values, model["weighted_scores"]["illustrative"]),
                    "contextual_percentile": percentile_rank(contextual_values, model["weighted_scores"]["contextual"]),
                    "grouped_overall_percentile": percentile_rank(grouped_overall_values, model["weighted_scores"]["grouped_overall"]),
                },
            }
        )
    return {
        "filter": {
            "release_date_excluded_if_older_than_days": min_model_age_days,
            "total_models_before_filter": len(all_models),
            "total_models_after_filter": len(models),
        },
        "grouping_version": "v1",
        "grouped_taxonomy": GROUPS,
        "global_aggregates": _to_aggregated_fields(global_accumulator),
        "data": data,
    }


def _failure_payload(min_model_age_days: int) -> dict[str, Any]:
    """Helper for failure payload."""
    return {
        "fetched_at_epoch_seconds": None,
        "status_code": None,
        "endpoint": TEXT_TO_IMAGE_URL,
        **_enrich_payload({}, min_model_age_days),
    }


def get_artificial_analysis_image_stats(
    options: ArtificialAnalysisImageOptions | None = None,
) -> dict[str, Any]:
    """Fetch and enrich Artificial Analysis text-to-image data."""
    options = options or {}
    api_key = options.get("api_key") or __import__("os").getenv("ARTIFICIALANALYSIS_API_KEY")
    min_model_age_days = options.get("min_model_age_days", DEFAULT_MIN_MODEL_AGE_DAYS)
    if not api_key:
        return _failure_payload(min_model_age_days)
    try:
        response = fetch_with_timeout(
            TEXT_TO_IMAGE_URL,
            headers={"x-api-key": api_key},
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return {
            "fetched_at_epoch_seconds": now_epoch_seconds(),
            "status_code": response.status_code,
            "endpoint": TEXT_TO_IMAGE_URL,
            **_enrich_payload(response.json(), min_model_age_days),
        }
    except Exception:
        return _failure_payload(min_model_age_days)
