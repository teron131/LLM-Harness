"""Final projection stage for native Python LLM stats."""

from __future__ import annotations

from typing import Any

from ..shared import as_finite_number, as_record
from .scoring import attach_relative_scores, blended_price_value, build_scores

EMPTY_OPENROUTER_PRICING = {"weighted_input": None, "weighted_output": None}
MIN_INTELLIGENCE_COST_TOKEN_THRESHOLD = 1_000_000
INTELLIGENCE_COST_TOTAL_COST_KEY = "intelligence_index_cost_total_cost"
INTELLIGENCE_COST_TOTAL_TOKENS_KEY = "intelligence_index_cost_total_tokens"


def _provider_from_id(model_id: Any) -> str | None:
    """Resolve the provider for Final-stage LLM stats selection."""
    if not isinstance(model_id, str) or "/" not in model_id:
        return None
    return model_id.split("/", 1)[0]


def _provider_from_model(model: dict[str, Any]) -> str | None:
    """Resolve the provider for Final-stage LLM stats selection."""
    return _provider_from_id(model.get("id")) or (model.get("provider_id") if isinstance(model.get("provider_id"), str) else None)


def _build_logo(model: dict[str, Any], provider: str | None) -> str:
    """Build the logo field for Final-stage LLM stats selection."""
    if isinstance(model.get("logo"), str) and model.get("logo"):
        return model["logo"]
    logo_slug = as_record(model.get("model_creator")).get("slug")
    if isinstance(logo_slug, str) and logo_slug:
        return f"https://artificialanalysis.ai/img/logos/{logo_slug}_small.svg"
    return f"https://models.dev/logos/{provider or 'unknown'}.svg"


def _build_speed(model: dict[str, Any], model_id: str | None, openrouter_speed_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the speed score for Final-stage LLM stats selection."""
    openrouter_speed = openrouter_speed_by_id.get(model_id, {}) if model_id else {}
    throughput = as_finite_number(openrouter_speed.get("throughput_tokens_per_second_median")) or as_finite_number(model.get("median_output_tokens_per_second"))
    latency = as_finite_number(openrouter_speed.get("latency_seconds_median")) or as_finite_number(model.get("median_time_to_first_token_seconds"))
    e2e_latency = as_finite_number(openrouter_speed.get("e2e_latency_seconds_median")) or as_finite_number(model.get("median_time_to_first_answer_token")) or latency
    return {
        "throughput_tokens_per_second_median": throughput,
        "latency_seconds_median": latency,
        "e2e_latency_seconds_median": e2e_latency,
    }


def _build_cost(model: dict[str, Any], openrouter_pricing: dict[str, Any], scoring_config: dict[str, Any]) -> dict[str, Any] | None:
    """Build the cost score for Final-stage LLM stats selection."""
    base_cost = as_record(model.get("cost"))
    cleaned_cost = {key: value for key, value in base_cost.items() if value is not None}
    weighted_input = as_finite_number(openrouter_pricing.get("weighted_input"))
    weighted_output = as_finite_number(openrouter_pricing.get("weighted_output"))
    if weighted_input is not None:
        cleaned_cost["weighted_input"] = weighted_input
    if weighted_output is not None:
        cleaned_cost["weighted_output"] = weighted_output
    blended_price = blended_price_value(cleaned_cost, scoring_config)
    if blended_price is not None:
        cleaned_cost["blended_price"] = blended_price
    return cleaned_cost or None


def _build_intelligence(model: dict[str, Any]) -> dict[str, Any] | None:
    """Build the intelligence score for Final-stage LLM stats selection."""
    intelligence = dict(as_record(model.get("intelligence")))
    nonhallucination_rate = as_finite_number(intelligence.get("omniscience_hallucination_rate"))
    if nonhallucination_rate is not None:
        intelligence["omniscience_nonhallucination_rate"] = nonhallucination_rate
        intelligence.pop("omniscience_hallucination_rate", None)
    intelligence.pop(INTELLIGENCE_COST_TOTAL_COST_KEY, None)
    intelligence.pop(INTELLIGENCE_COST_TOTAL_TOKENS_KEY, None)
    return intelligence or None


def _build_intelligence_index_cost(model: dict[str, Any]) -> dict[str, Any] | None:
    """Build the intelligence index cost for Final-stage LLM stats selection."""
    from_row = as_record(model.get("intelligence_index_cost"))
    from_intelligence = as_record(model.get("intelligence"))
    total_cost = as_finite_number(from_row.get("total_cost")) or as_finite_number(from_intelligence.get(INTELLIGENCE_COST_TOTAL_COST_KEY))
    total_tokens = as_finite_number(from_row.get("total_tokens")) or as_finite_number(from_intelligence.get(INTELLIGENCE_COST_TOTAL_TOKENS_KEY))
    normalized = {
        **from_row,
        "total_cost": total_cost,
        "total_tokens": total_tokens if total_tokens is not None and total_tokens >= MIN_INTELLIGENCE_COST_TOKEN_THRESHOLD else None,
    }
    cleaned = {key: value for key, value in normalized.items() if value is not None}
    return cleaned or None


def _project_final_model(
    row: dict[str, Any],
    openrouter_speed_by_id: dict[str, dict[str, Any]],
    openrouter_pricing_by_id: dict[str, dict[str, Any]],
    speed_output_token_anchors: list[int],
    scoring_config: dict[str, Any],
) -> dict[str, Any]:
    """Helper for project final model."""
    model = as_record(row)
    provider = _provider_from_model(model)
    model_id = model.get("id") if isinstance(model.get("id"), str) else None
    speed = _build_speed(model, model_id, openrouter_speed_by_id)
    pricing = openrouter_pricing_by_id.get(model_id, EMPTY_OPENROUTER_PRICING) if model_id else EMPTY_OPENROUTER_PRICING
    cost = _build_cost(model, pricing, scoring_config)
    return {
        "id": model_id,
        "name": model.get("name") if isinstance(model.get("name"), str) else None,
        "provider": provider,
        "logo": _build_logo(model, provider),
        "attachment": model.get("attachment") if isinstance(model.get("attachment"), bool) else None,
        "reasoning": model.get("reasoning") if isinstance(model.get("reasoning"), bool) else None,
        "release_date": model.get("release_date") if isinstance(model.get("release_date"), str) else None,
        "modalities": model.get("modalities"),
        "open_weights": model.get("open_weights") if isinstance(model.get("open_weights"), bool) else None,
        "cost": cost,
        "context_window": model.get("limit"),
        "speed": speed,
        "intelligence": _build_intelligence(model),
        "intelligence_index_cost": _build_intelligence_index_cost(model),
        "evaluations": as_record(model.get("evaluations")) or None,
        "scores": build_scores(model, cost, speed, speed_output_token_anchors, scoring_config),
        "relative_scores": None,
    }


def build_final_payload(
    rows: list[dict[str, Any]],
    *,
    model_id: str | None = None,
    fetched_at_epoch_seconds: int | None = None,
    openrouter_speed_by_id: dict[str, dict[str, Any]] | None = None,
    openrouter_pricing_by_id: dict[str, dict[str, Any]] | None = None,
    speed_output_token_anchors: list[int] | None = None,
    scoring_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the final Final-stage LLM stats selection payload."""
    openrouter_speed_by_id = openrouter_speed_by_id or {}
    openrouter_pricing_by_id = openrouter_pricing_by_id or {}
    speed_output_token_anchors = speed_output_token_anchors or [200, 500, 1000, 2000, 8000]
    scoring_config = scoring_config or {}
    models = [
        _project_final_model(
            row,
            openrouter_speed_by_id,
            openrouter_pricing_by_id,
            speed_output_token_anchors,
            scoring_config,
        )
        for row in rows
    ]
    models = attach_relative_scores(models)

    def intelligence_sort_score(model: dict[str, Any]) -> float:
        """Sort missing intelligence scores below real zero scores."""
        score = as_finite_number(as_record(model.get("relative_scores")).get("intelligence_score"))
        return score if score is not None else float("-inf")

    models.sort(
        key=lambda model: (
            -intelligence_sort_score(model),
            model.get("id") or "",
        )
    )
    if model_id is not None:
        models = [row for row in models if row.get("id") == model_id]
    return {
        "fetched_at_epoch_seconds": fetched_at_epoch_seconds,
        "models": models,
    }
