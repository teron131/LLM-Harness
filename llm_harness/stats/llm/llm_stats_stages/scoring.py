"""Score helpers for native Python LLM stats."""

from __future__ import annotations

from statistics import median
from typing import Any, cast

import polars as pl

from ..shared import as_finite_number, as_record


def percentile_rank(values: list[Any], value: Any) -> float | None:
    """Compute a finite-aware aggregate for Scoring LLM stats selection."""
    numeric_value = as_finite_number(value)
    if numeric_value is None:
        return None
    finite_values = [
        item for item in (as_finite_number(v) for v in values) if item is not None
    ]
    if not finite_values:
        return None
    rank = (
        pl.DataFrame({"value": finite_values})
        .select(((pl.col("value") <= numeric_value).sum() / pl.len()) * 100)
        .item()
    )
    return float(rank) if isinstance(rank, int | float) else None


def mean_of_finite(values: list[float | None]) -> float | None:
    """Compute a finite-aware aggregate for Scoring LLM stats selection."""
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    mean = pl.Series("value", finite_values).mean()
    return float(mean) if isinstance(mean, int | float) else None


def _metric_value(model: dict[str, Any], key: str) -> float | None:
    """Helper for metric value."""
    intelligence = as_record(model.get("intelligence"))
    evaluations = as_record(model.get("evaluations"))
    if key == "omniscience_nonhallucination_rate":
        return as_finite_number(
            intelligence.get("omniscience_nonhallucination_rate")
        ) or as_finite_number(intelligence.get("omniscience_hallucination_rate"))
    return as_finite_number(intelligence.get(key)) or as_finite_number(
        evaluations.get(key)
    )


def blended_price_value(cost_like: Any, scoring_config: dict[str, Any]) -> float | None:
    """Helper for blended price value."""
    cost = as_record(cost_like)
    input_cost = as_finite_number(cost.get("input"))
    output_cost = as_finite_number(cost.get("output"))
    weighted_input = as_finite_number(cost.get("weighted_input"))
    weighted_output = as_finite_number(cost.get("weighted_output"))
    cache_read = as_finite_number(cost.get("cache_read"))
    cache_write = as_finite_number(cost.get("cache_write"))
    if input_cost is None or output_cost is None or input_cost <= 0 or output_cost <= 0:
        return None
    input_ratio = (
        as_finite_number(scoring_config.get("weighted_price_input_ratio")) or 0.75
    )
    output_ratio = (
        as_finite_number(scoring_config.get("weighted_price_output_ratio")) or 0.25
    )
    if weighted_input is not None or weighted_output is not None:
        effective_input = weighted_input if weighted_input is not None else input_cost
        effective_output = (
            weighted_output if weighted_output is not None else output_cost
        )
        return input_ratio * effective_input + output_ratio * effective_output
    cache_weighted_input = cache_read if cache_read is not None else input_cost
    cache_weighted_output = (
        0.1 * cache_write + 0.9 * output_cost
        if cache_write is not None
        else output_cost
    )
    return (
        input_ratio * (input_ratio * cache_weighted_input + output_ratio * input_cost)
        + output_ratio * cache_weighted_output
    )


def derive_speed_output_token_anchors(
    openrouter_speed_by_id: dict[str, dict[str, Any]], scoring_config: dict[str, Any]
) -> list[int]:
    """Derive the speed output token anchors."""
    implied_token_usages: list[float] = []
    for speed in openrouter_speed_by_id.values():
        throughput = as_finite_number(speed.get("throughput_tokens_per_second_median"))
        latency = as_finite_number(speed.get("latency_seconds_median"))
        e2e_latency = as_finite_number(speed.get("e2e_latency_seconds_median"))
        if (
            throughput is None
            or throughput <= 0
            or latency is None
            or e2e_latency is None
        ):
            continue
        generation_seconds = e2e_latency - latency
        if generation_seconds <= 0:
            continue
        implied_token_usages.append(generation_seconds * throughput)
    implied_token_usages.sort()
    default_anchors = cast(
        list[int],
        list(
            scoring_config.get(
                "default_speed_output_token_anchors", [200, 500, 1000, 2000, 8000]
            )
        ),
    )
    if not implied_token_usages:
        return default_anchors
    quantiles = [
        0,
        *cast(
            list[float], scoring_config.get("speed_anchor_quantiles", [0.25, 0.5, 0.75])
        ),
        1,
    ]
    token_series = pl.Series("tokens", implied_token_usages)
    anchors = []
    for quantile in quantiles:
        if quantile == 0:
            anchors.append(implied_token_usages[0])
        elif quantile == 1:
            anchors.append(implied_token_usages[-1])
        else:
            quantile_value = token_series.quantile(quantile, interpolation="linear")
            if quantile_value is not None:
                anchors.append(float(quantile_value))
    source_min = anchors[0]
    source_max = anchors[-1]
    if source_max <= source_min:
        return default_anchors
    range_min = (
        as_finite_number(scoring_config.get("speed_output_token_range_min")) or 200
    )
    range_max = (
        as_finite_number(scoring_config.get("speed_output_token_range_max")) or 8000
    )
    return [
        round(
            range_min
            + ((anchor - source_min) / (source_max - source_min))
            * (range_max - range_min)
        )
        for anchor in anchors
    ]


def build_scores(
    model: dict[str, Any],
    cost: Any,
    speed: dict[str, Any],
    speed_output_token_anchors: list[int],
    scoring_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Build score fields for Scoring LLM stats selection."""
    intelligence_index = _metric_value(model, "intelligence_index") or _metric_value(
        model, "artificial_analysis_intelligence_index"
    )
    agentic_index = _metric_value(model, "agentic_index") or _metric_value(
        model, "artificial_analysis_agentic_index"
    )
    intelligence_benchmark_mean = mean_of_finite(
        [
            _metric_value(model, key)
            for key in scoring_config.get("intelligence_benchmark_keys", [])
        ]
    )
    agentic_benchmark_mean = mean_of_finite(
        [
            _metric_value(model, key)
            for key in scoring_config.get("agentic_benchmark_keys", [])
        ]
    )
    intelligence_score = (
        (intelligence_index + intelligence_benchmark_mean * 100) / 2
        if intelligence_index is not None and intelligence_benchmark_mean is not None
        else None
    )
    agentic_score = (
        (agentic_index + agentic_benchmark_mean * 100) / 2
        if agentic_index is not None and agentic_benchmark_mean is not None
        else None
    )
    blended_price = blended_price_value(cost, scoring_config)
    latency = as_finite_number(speed.get("latency_seconds_median"))
    throughput = as_finite_number(speed.get("throughput_tokens_per_second_median"))
    e2e_latency = as_finite_number(speed.get("e2e_latency_seconds_median"))
    price_score = (
        1 / blended_price if blended_price is not None and blended_price > 0 else None
    )
    imagined_speed_score = mean_of_finite(
        [
            target / (latency + target / throughput)
            if latency is not None and throughput is not None and throughput > 0
            else None
            for target in speed_output_token_anchors
        ]
    )
    representative_target_tokens = (
        median(sorted(speed_output_token_anchors))
        if speed_output_token_anchors
        else None
    )
    observed_e2e_speed_score = (
        representative_target_tokens / e2e_latency
        if representative_target_tokens is not None
        and e2e_latency is not None
        and e2e_latency > 0
        else None
    )
    speed_score = mean_of_finite([imagined_speed_score, observed_e2e_speed_score])
    if all(
        score is None
        for score in (intelligence_score, agentic_score, price_score, speed_score)
    ):
        return None
    return {
        "intelligence_score": intelligence_score,
        "agentic_score": agentic_score,
        "speed_score": speed_score,
        "price_score": price_score,
    }


def attach_relative_scores(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach relative scores for Scoring LLM stats selection."""
    if not models:
        return []

    score_frame = pl.DataFrame(
        [
            {
                "row_index": row_index,
                "intelligence_score": as_finite_number(
                    as_record(model.get("scores")).get("intelligence_score")
                ),
                "agentic_score": as_finite_number(
                    as_record(model.get("scores")).get("agentic_score")
                ),
                "speed_score": as_finite_number(
                    as_record(model.get("scores")).get("speed_score")
                ),
                "price_score": as_finite_number(
                    as_record(model.get("scores")).get("price_score")
                ),
            }
            for row_index, model in enumerate(models)
        ]
    )

    def min_max_relative(column_name: str) -> pl.Expr:
        """Score a value relative to the observed min-max range."""
        return (
            pl.when(pl.col(column_name).is_null() | (pl.col(column_name).count() == 0))
            .then(None)
            .when(pl.col(column_name).max() == pl.col(column_name).min())
            .then(100.0)
            .otherwise(
                (
                    (pl.col(column_name) - pl.col(column_name).min())
                    / (pl.col(column_name).max() - pl.col(column_name).min())
                )
                * 100
            )
            .alias(f"{column_name}_relative")
        )

    def percentile_relative(column_name: str) -> pl.Expr:
        """Score a value relative to the observed percentile range."""
        return (
            pl.when(pl.col(column_name).is_null() | (pl.col(column_name).count() == 0))
            .then(None)
            .otherwise(
                (pl.col(column_name).rank(method="max") / pl.col(column_name).count())
                * 100
            )
            .alias(f"{column_name}_relative")
        )

    relative_frame = score_frame.with_columns(
        [
            min_max_relative("intelligence_score"),
            min_max_relative("agentic_score"),
            percentile_relative("speed_score"),
            percentile_relative("price_score"),
        ]
    ).with_columns(
        pl.mean_horizontal(
            "intelligence_score_relative",
            "agentic_score_relative",
            "speed_score_relative",
            "price_score_relative",
        ).alias("overall_score_relative")
    )

    relative_by_index = {
        row["row_index"]: row
        for row in relative_frame.select(
            "row_index",
            "intelligence_score_relative",
            "agentic_score_relative",
            "speed_score_relative",
            "price_score_relative",
            "overall_score_relative",
        ).to_dicts()
    }
    return [
        {
            **model,
            "relative_scores": {
                "intelligence_score": relative_by_index[row_index][
                    "intelligence_score_relative"
                ],
                "agentic_score": relative_by_index[row_index]["agentic_score_relative"],
                "speed_score": relative_by_index[row_index]["speed_score_relative"],
                "price_score": relative_by_index[row_index]["price_score_relative"],
                "overall_score": relative_by_index[row_index]["overall_score_relative"],
            },
        }
        for row_index, model in enumerate(models)
    ]
