"""Artificial Analysis LLM source ported from the JS stats pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import math
import os
from typing import Any, TypedDict

import httpx

MODELS_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
LOOKBACK_DAYS = 365
REQUEST_TIMEOUT_SECONDS = 30.0
BENCHMARK_KEYS = ("hle", "terminalbench_hard", "lcr", "ifbench", "scicode")
SCORE_WEIGHTS = {
    "intelligence": 0.3,
    "benchmark_bias": 0.3,
    "price": 0.2,
    "speed": 0.2,
}


class ArtificialAnalysisOptions(TypedDict, total=False):
    """Options for Artificial Analysis API source."""

    api_key: str


def _as_finite_float(value: Any) -> float | None:
    """Helper for as finite float."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _is_positive_finite(value: Any) -> bool:
    """Return whether the current value is valid for Artificial Analysis API source."""
    numeric = _as_finite_float(value)
    return numeric is not None and numeric > 0


def _signed_log(value: Any, *, invert: bool = False) -> float | None:
    """Helper for signed log."""
    numeric = _as_finite_float(value)
    if numeric is None or numeric <= 0:
        return None
    return -math.log10(numeric) if invert else math.log10(numeric)


def _mean(values: list[float | None]) -> float | None:
    """Helper for mean."""
    finite_values = [
        value for value in values if value is not None and math.isfinite(value)
    ]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def _weighted_mean(pairs: list[tuple[float | None, float]]) -> float | None:
    """Compute a finite-aware aggregate for Artificial Analysis API source."""
    valid_pairs = [
        (value, weight)
        for value, weight in pairs
        if value is not None
        and math.isfinite(value)
        and math.isfinite(weight)
        and weight > 0
    ]
    if not valid_pairs:
        return None
    weighted_sum = sum(value * weight for value, weight in valid_pairs)
    weight_sum = sum(weight for _, weight in valid_pairs)
    if weight_sum == 0:
        return None
    return weighted_sum / weight_sum


def _percentile_rank(values: list[Any], value: Any) -> float | None:
    """Compute a finite-aware aggregate for Artificial Analysis API source."""
    numeric_value = _as_finite_float(value)
    if numeric_value is None:
        return None
    finite_values = [_as_finite_float(item) for item in values]
    finite_values = [item for item in finite_values if item is not None]
    if not finite_values:
        return None
    less_or_equal_count = sum(1 for item in finite_values if item <= numeric_value)
    return (less_or_equal_count / len(finite_values)) * 100


def _remove_ids(value: Any) -> Any:
    """Helper for remove ids."""
    if isinstance(value, list):
        return [_remove_ids(item) for item in value]
    if isinstance(value, dict):
        return {key: _remove_ids(child) for key, child in value.items() if key != "id"}
    return value


def _compute_scores(filtered_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute the scores."""
    scored_models: list[dict[str, Any]] = []
    for model in filtered_models:
        evaluations = model.get("evaluations") or {}
        pricing = model.get("pricing") or {}
        intelligence = _as_finite_float(
            evaluations.get("artificial_analysis_intelligence_index"),
        )
        coding = _as_finite_float(evaluations.get("artificial_analysis_coding_index"))
        blended_price = pricing.get("price_1m_blended_3_to_1")
        ttfa = model.get("median_time_to_first_answer_token")
        tps = model.get("median_output_tokens_per_second")
        intelligence_score = (
            (2 * intelligence) + coding
            if intelligence is not None and coding is not None
            else None
        )
        benchmark_bias_score = _mean(
            [
                _as_finite_float(evaluations.get(key))
                if _is_positive_finite(evaluations.get(key))
                else None
                for key in BENCHMARK_KEYS
            ]
        )
        price_score = _signed_log(blended_price, invert=True)
        speed_score = _mean([_signed_log(ttfa, invert=True), _signed_log(tps)])
        scored_models.append(
            {
                **model,
                "scores": {
                    "overall_score": _weighted_mean(
                        [
                            (intelligence_score, SCORE_WEIGHTS["intelligence"]),
                            (benchmark_bias_score, SCORE_WEIGHTS["benchmark_bias"]),
                            (price_score, SCORE_WEIGHTS["price"]),
                            (speed_score, SCORE_WEIGHTS["speed"]),
                        ]
                    ),
                    "intelligence_score": intelligence_score,
                    "benchmark_bias_score": benchmark_bias_score,
                    "price_score": price_score,
                    "speed_score": speed_score,
                },
            }
        )
    return scored_models


def _rank_and_enrich_models(
    models: list[dict[str, Any]],
    cutoff_date: str,
) -> list[dict[str, Any]]:
    """Rank the and enrich models."""
    filtered_models = [
        model
        for model in models
        if (model.get("release_date") or "") >= cutoff_date
        and _is_positive_finite(
            (model.get("pricing") or {}).get("price_1m_blended_3_to_1")
        )
        and _is_positive_finite(
            (model.get("pricing") or {}).get("price_1m_input_tokens")
        )
        and _is_positive_finite(
            (model.get("pricing") or {}).get("price_1m_output_tokens")
        )
        and _is_positive_finite(model.get("median_time_to_first_answer_token"))
        and _is_positive_finite(model.get("median_output_tokens_per_second"))
    ]
    scored_models = []
    for model in _compute_scores(filtered_models):
        raw_scores = model.get("scores")
        scores = raw_scores if isinstance(raw_scores, dict) else {}
        overall_score = scores.get("overall_score")
        if isinstance(overall_score, int | float) and math.isfinite(
            float(overall_score)
        ):
            scored_models.append(model)
    ranked = sorted(
        scored_models,
        key=lambda model: float(
            model["scores"]["overall_score"]
            if isinstance(model.get("scores"), dict)
            else 0
        ),
        reverse=True,
    )
    overall_values = [
        (model.get("scores") or {}).get("overall_score") for model in ranked
    ]
    intelligence_values = [
        (model.get("scores") or {}).get("intelligence_score") for model in ranked
    ]
    speed_values = [(model.get("scores") or {}).get("speed_score") for model in ranked]
    price_values = [(model.get("scores") or {}).get("price_score") for model in ranked]
    return [
        {
            **model,
            "percentiles": {
                "overall_percentile": _percentile_rank(
                    overall_values,
                    (model.get("scores") or {}).get("overall_score"),
                ),
                "intelligence_percentile": _percentile_rank(
                    intelligence_values,
                    (model.get("scores") or {}).get("intelligence_score"),
                ),
                "speed_percentile": _percentile_rank(
                    speed_values,
                    (model.get("scores") or {}).get("speed_score"),
                ),
                "price_percentile": _percentile_rank(
                    price_values,
                    (model.get("scores") or {}).get("price_score"),
                ),
            },
        }
        for model in ranked
    ]


def _fetch_models(api_key: str | None) -> dict[str, Any]:
    """Return models."""
    if not api_key:
        raise ValueError("Missing ARTIFICIALANALYSIS_API_KEY.")
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = client.get(MODELS_URL, headers={"x-api-key": api_key})
    response.raise_for_status()
    payload = response.json()
    return {
        "fetched_at_epoch_seconds": int(datetime.now(UTC).timestamp()),
        "status_code": response.status_code,
        "models": [_remove_ids(model) for model in payload.get("data", [])],
    }


def get_artificial_analysis_stats(
    options: ArtificialAnalysisOptions | None = None,
) -> dict[str, Any]:
    """Fetch, rank, and enrich Artificial Analysis model stats."""
    options = options or {}
    try:
        api_key = options.get("api_key") or os.getenv("ARTIFICIALANALYSIS_API_KEY")
        source_payload = _fetch_models(api_key)
        cutoff_date = (
            (datetime.now(UTC) - timedelta(days=LOOKBACK_DAYS)).date().isoformat()
        )
        return {
            "fetched_at_epoch_seconds": source_payload["fetched_at_epoch_seconds"],
            "status_code": source_payload["status_code"],
            "models": _rank_and_enrich_models(
                source_payload.get("models", []), cutoff_date
            ),
        }
    except Exception:
        return {
            "fetched_at_epoch_seconds": None,
            "status_code": None,
            "models": [],
        }
