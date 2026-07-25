"""OpenRouter enrichment stage for native Python LLM stats."""

from __future__ import annotations

from typing import Any

from ..shared import (
    PRIMARY_PROVIDER_ID,
    as_finite_number,
    as_record,
    model_slug_from_model_id,
    normalize_provider_model_id,
)
from ..sources.openrouter_scraper import get_openrouter_scraped_stats
from .scoring import derive_speed_output_token_anchors

EPHEMERAL_SUFFIXES = ("-adaptive",)


def _normalize_openrouter_speed(performance: Any) -> dict[str, Any]:
    """Normalize the openrouter speed."""
    parsed = as_record(performance)
    return {
        "throughput_tokens_per_second_median": as_finite_number(
            parsed.get("throughput_tokens_per_second_median")
        ),
        "latency_seconds_median": as_finite_number(
            parsed.get("latency_seconds_median")
        ),
        "e2e_latency_seconds_median": as_finite_number(
            parsed.get("e2e_latency_seconds_median")
        ),
    }


def _normalize_openrouter_pricing(pricing: Any) -> dict[str, Any]:
    """Normalize the openrouter pricing."""
    parsed = as_record(pricing)
    return {
        "weighted_input": as_finite_number(parsed.get("weighted_input_price_per_1m")),
        "weighted_output": as_finite_number(parsed.get("weighted_output_price_per_1m")),
    }


def _has_intelligence_cost(row: dict[str, Any]) -> bool:
    """Return whether the row includes an intelligence-cost signal."""
    intelligence_index_cost = as_record(row.get("intelligence_index_cost"))
    return as_finite_number(intelligence_index_cost.get("total_cost")) is not None


def _has_score_signal(row: dict[str, Any]) -> bool:
    """Return whether the row includes any usable score signal."""
    scores = as_record(row.get("scores"))
    return any(
        as_finite_number(scores.get(key)) is not None
        for key in ("intelligence_score", "agentic_score", "speed_score", "price_score")
    )


def _reasoning_effort_priority(aa_slug: str | None, canonical_slug: str | None) -> int:
    """Helper for reasoning effort priority."""
    if aa_slug is None or canonical_slug is None:
        return 0
    normalized_aa_slug = normalize_provider_model_id(aa_slug)
    normalized_canonical_slug = normalize_provider_model_id(canonical_slug)
    for suffix in EPHEMERAL_SUFFIXES:
        if normalized_aa_slug == f"{normalized_canonical_slug}{suffix}":
            return 6
    if normalized_aa_slug == normalized_canonical_slug:
        return 5
    for suffix, priority in (
        ("-xhigh", 5),
        ("-high", 4),
        ("-medium", 3),
        ("-low", 2),
        ("-minimal", 1),
    ):
        if normalized_aa_slug == f"{normalized_canonical_slug}{suffix}":
            return priority
    return 0


def _row_priority(row: dict[str, Any], normalized_id: str) -> int:
    """Helper for row priority."""
    provider_id = row.get("provider_id")
    openrouter_boost = 1_000_000 if provider_id == PRIMARY_PROVIDER_ID else 0
    intelligence_cost_boost = 1_000 if _has_intelligence_cost(row) else 0
    score_signal_boost = 10 if _has_score_signal(row) else 0
    aa_slug = row.get("aa_slug") if isinstance(row.get("aa_slug"), str) else None
    canonical_slug = model_slug_from_model_id(normalized_id)
    reasoning_effort_boost = (
        _reasoning_effort_priority(aa_slug, canonical_slug) * 10_000_000
    )
    return (
        reasoning_effort_boost
        + openrouter_boost
        + intelligence_cost_boost
        + score_signal_boost
    )


def _dedupe_rows_prefer_openrouter(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe the rows prefer openrouter."""
    grouped_by_normalized_id: dict[str, list[dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        row_record = as_record(row)
        row_id = row_record.get("id") if isinstance(row_record.get("id"), str) else None
        if not row_id:
            passthrough.append(row_record)
            continue
        key = normalize_provider_model_id(row_id)
        grouped_by_normalized_id.setdefault(key, []).append(row_record)

    deduped_rows: list[dict[str, Any]] = []
    for normalized_id, group in grouped_by_normalized_id.items():
        winner = sorted(
            group, key=lambda row: _row_priority(row, normalized_id), reverse=True
        )[0]
        merged_intelligence_index_cost = dict(
            as_record(winner.get("intelligence_index_cost"))
        )
        for candidate in group:
            candidate_cost = as_record(candidate.get("intelligence_index_cost"))
            for key, value in candidate_cost.items():
                if (
                    merged_intelligence_index_cost.get(key) is None
                    and value is not None
                ):
                    merged_intelligence_index_cost[key] = value
        deduped_rows.append(
            {
                **winner,
                "intelligence_index_cost": merged_intelligence_index_cost,
            }
        )
    return passthrough + deduped_rows


def _backfill_free_model_costs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backfill the free model costs."""
    non_free_cost_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_record = as_record(row)
        row_id = row_record.get("id") if isinstance(row_record.get("id"), str) else None
        if not row_id or row_id.endswith(":free"):
            continue
        cost = as_record(row_record.get("cost"))
        input_cost = as_finite_number(cost.get("input"))
        output_cost = as_finite_number(cost.get("output"))
        if (
            input_cost is not None
            and input_cost > 0
            and output_cost is not None
            and output_cost > 0
        ):
            non_free_cost_by_id[row_id] = cost

    enriched_rows = []
    for row in rows:
        row_record = as_record(row)
        row_id = row_record.get("id") if isinstance(row_record.get("id"), str) else None
        if not row_id or not row_id.endswith(":free"):
            enriched_rows.append(row_record)
            continue
        base_id = row_id[: -len(":free")]
        base_cost = non_free_cost_by_id.get(base_id)
        if not base_cost:
            enriched_rows.append(row_record)
            continue
        enriched_rows.append({**row_record, "cost": dict(base_cost)})
    return enriched_rows


def enrich_rows(
    matched_rows: list[dict[str, Any]],
    openrouter_config: dict[str, Any] | None = None,
    scoring_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch OpenRouter enrichments for matched rows and return late-bound maps."""
    openrouter_config = openrouter_config or {"speed_concurrency": 8}
    scoring_config = scoring_config or {}
    deduped_rows = _dedupe_rows_prefer_openrouter(matched_rows)
    rows = _backfill_free_model_costs(deduped_rows)
    model_ids = [
        row_id for row in rows if isinstance((row_id := row.get("id")), str) and row_id
    ]
    openrouter_payload = (
        get_openrouter_scraped_stats(
            {
                "model_ids": model_ids,
                "concurrency": int(openrouter_config.get("speed_concurrency", 8)),
            }
        )
        if model_ids
        else {"models": []}
    )
    openrouter_speed_by_id = {
        model["id"]: _normalize_openrouter_speed(model.get("performance"))
        for model in openrouter_payload.get("models", [])
        if isinstance(model, dict) and isinstance(model.get("id"), str)
    }
    openrouter_pricing_by_id = {
        model["id"]: _normalize_openrouter_pricing(model.get("pricing"))
        for model in openrouter_payload.get("models", [])
        if isinstance(model, dict) and isinstance(model.get("id"), str)
    }
    speed_output_token_anchors = derive_speed_output_token_anchors(
        openrouter_speed_by_id,
        scoring_config,
    )
    return {
        "rows": rows,
        "openrouter_speed_by_id": openrouter_speed_by_id,
        "openrouter_pricing_by_id": openrouter_pricing_by_id,
        "speed_output_token_anchors": speed_output_token_anchors,
    }
