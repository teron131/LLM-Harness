"""Native Python port of the OpenRouter scraper stats source."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import random
import re
from statistics import median
import time
from typing import Any, TypedDict, cast
from urllib.parse import urlencode

import httpx

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/frontend/models"
OPENROUTER_THROUGHPUT_URL = "https://openrouter.ai/api/frontend/stats/throughput-comparison"
OPENROUTER_LATENCY_URL = "https://openrouter.ai/api/frontend/stats/latency-comparison"
OPENROUTER_E2E_LATENCY_URL = "https://openrouter.ai/api/frontend/stats/latency-e2e-comparison"
OPENROUTER_EFFECTIVE_PRICING_URL = "https://openrouter.ai/api/frontend/stats/effective-pricing"

DEFAULT_TIMEOUT_MS = 30_000
DEFAULT_CONCURRENCY = 8
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY_MS = 300
_ROUTE_SUFFIX_PATTERN = re.compile(r":[a-z0-9._-]+$", re.IGNORECASE)
_SLUG_SPLIT_PATTERN = re.compile(r"[-._/]+")


class OpenRouterScraperOptions(TypedDict, total=False):
    """Options for OpenRouter scraper model stats."""

    model_ids: list[str]
    timeout_ms: int
    concurrency: int
    max_retries: int
    retry_base_delay_ms: int


class OpenRouterModelOptions(TypedDict, total=False):
    """Options for OpenRouter scraper model stats."""

    model_id: str
    timeout_ms: int
    concurrency: int
    max_retries: int
    retry_base_delay_ms: int


class OpenRouterPerformanceSummary(TypedDict):
    """Summary data for OpenRouter scraper model stats."""

    throughput_tokens_per_second_median: float | None
    latency_seconds_median: float | None
    e2e_latency_seconds_median: float | None


class OpenRouterPricingSummary(TypedDict):
    """Summary data for OpenRouter scraper model stats."""

    weighted_input_price_per_1m: float | None
    weighted_output_price_per_1m: float | None


class OpenRouterScrapedModel(TypedDict):
    """Pydantic model for OpenRouter scraper model stats."""

    id: str
    permaslug: str | None
    performance: OpenRouterPerformanceSummary
    pricing: OpenRouterPricingSummary


def _sanitize_model_id(model_id: str) -> str:
    """Sanitize the model id."""
    return _ROUTE_SUFFIX_PATTERN.sub("", model_id.strip().lower())


def _as_finite_number(value: object) -> float | None:
    """Convert the input into a finite number for OpenRouter scraper model stats."""
    if value is None or isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _finite_numbers(values: list[object]) -> list[float]:
    """Helper for finite numbers."""
    return [numeric_value for value in values if (numeric_value := _as_finite_number(value)) is not None]


def _average(values: list[float]) -> float | None:
    """Helper for average."""
    if not values:
        return None
    return sum(values) / len(values)


def _to_daily_averaged_values(
    response: object,
    *,
    scale_to_seconds: bool,
) -> list[float]:
    """Helper for to daily averaged values."""
    if not isinstance(response, dict):
        return []
    data = response.get("data")
    if not isinstance(data, list):
        return []

    averaged_values: list[float] = []
    for point in data:
        if not isinstance(point, dict):
            continue
        y = point.get("y")
        if not isinstance(y, dict):
            continue
        values = _finite_numbers(list(y.values()))
        daily_average = _average(values)
        if daily_average is None:
            continue
        averaged_values.append(daily_average / 1000 if scale_to_seconds else daily_average)
    return averaged_values


def _summarize_performance(stats: dict[str, object] | None = None) -> OpenRouterPerformanceSummary:
    """Summarize the performance."""
    safe_stats = stats or {}
    throughput_values = _to_daily_averaged_values(safe_stats.get("throughput"), scale_to_seconds=False)
    latency_values = _to_daily_averaged_values(safe_stats.get("latency"), scale_to_seconds=True)
    e2e_latency_values = _to_daily_averaged_values(safe_stats.get("latency_e2e"), scale_to_seconds=True)
    return {
        "throughput_tokens_per_second_median": median(throughput_values) if throughput_values else None,
        "latency_seconds_median": median(latency_values) if latency_values else None,
        "e2e_latency_seconds_median": median(e2e_latency_values) if e2e_latency_values else None,
    }


def _summarize_pricing(response: object) -> OpenRouterPricingSummary:
    """Summarize the pricing."""
    data = response.get("data") if isinstance(response, dict) else None
    safe_data = data if isinstance(data, dict) else {}
    return {
        "weighted_input_price_per_1m": _as_finite_number(safe_data.get("weightedInputPrice")),
        "weighted_output_price_per_1m": _as_finite_number(safe_data.get("weightedOutputPrice")),
    }


def _empty_scraped_model(model_id: str) -> OpenRouterScrapedModel:
    """Helper for empty scraped model."""
    return {
        "id": model_id,
        "permaslug": None,
        "performance": _summarize_performance(),
        "pricing": _summarize_pricing(None),
    }


def _sleep_ms(delay_ms: int) -> None:
    """Helper for sleep ms."""
    time.sleep(max(0, delay_ms) / 1000)


def _fetch_json_with_retry(
    client: httpx.Client,
    *,
    url: str,
    max_retries: int,
    retry_base_delay_ms: int,
) -> object:
    """Return json with retry."""
    attempts = max(1, max_retries)
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            response = client.get(url)
            if response.status_code == 200:
                return response.json()
            status = response.status_code
            if (status == 429 or status >= 500) and attempt < attempts - 1:
                backoff_ms = retry_base_delay_ms * (2**attempt) + random.randint(0, 99)
                _sleep_ms(backoff_ms)
                continue
            response.raise_for_status()
        except Exception as error:
            last_error = error
            if attempt < attempts - 1:
                backoff_ms = retry_base_delay_ms * (2**attempt) + random.randint(0, 99)
                _sleep_ms(backoff_ms)
                continue
            break

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"OpenRouter request failed: {url}")


def _build_permaslug_lookup(models: object) -> dict[str, str]:
    """Build the permaslug lookup."""
    if not isinstance(models, list):
        return {}

    permaslug_by_slug: dict[str, str] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        slug = model.get("slug")
        permaslug = model.get("permaslug")
        if not isinstance(slug, str) or not isinstance(permaslug, str):
            continue
        sanitized_slug = _sanitize_model_id(slug)
        trimmed_permaslug = permaslug.strip()
        if not sanitized_slug or not trimmed_permaslug:
            continue
        permaslug_by_slug[sanitized_slug] = trimmed_permaslug
    return permaslug_by_slug


def _has_meaningful_performance(performance: OpenRouterPerformanceSummary) -> bool:
    """Return whether the current value is valid for OpenRouter scraper model stats."""
    return (
        performance["throughput_tokens_per_second_median"] is not None or performance["latency_seconds_median"] is not None or performance["e2e_latency_seconds_median"] is not None
    )


def _has_meaningful_pricing(pricing: OpenRouterPricingSummary) -> bool:
    """Return whether the current value is valid for OpenRouter scraper model stats."""
    weighted_input = pricing["weighted_input_price_per_1m"]
    weighted_output = pricing["weighted_output_price_per_1m"]
    return (weighted_input is not None and weighted_input > 0) or (weighted_output is not None and weighted_output > 0)


def _split_slug_tokens(value: str) -> list[str]:
    """Split OpenRouter scraper model stats into normalized tokens."""
    return [token for token in _SLUG_SPLIT_PATTERN.split(value.lower()) if token]


def _token_overlap_score(target_tokens: list[str], candidate_tokens: list[str]) -> float:
    """Helper for token overlap score."""
    if not target_tokens:
        return 0.0
    target_set = set(target_tokens)
    candidate_set = set(candidate_tokens)
    overlap_count = sum(1 for token in target_set if token in candidate_set)
    return overlap_count / len(target_set)


def _build_slug_fallback_candidates(model_id: str, available_slugs: list[str]) -> list[str]:
    """Build fallback candidates for OpenRouter scraper model stats."""
    normalized = _sanitize_model_id(model_id)
    if "/" not in normalized:
        return [normalized]

    provider, model_name = normalized.split("/", 1)
    if not provider or not model_name:
        return [normalized]

    target_tokens = _split_slug_tokens(model_name)
    core_prefix = "-".join(target_tokens[:2])
    scored_candidates: list[tuple[str, float, int]] = []
    for slug in available_slugs:
        if not slug.startswith(f"{provider}/") or slug == normalized:
            continue
        candidate_model = slug[len(provider) + 1 :]
        candidate_tokens = _split_slug_tokens(candidate_model)
        overlap_score = _token_overlap_score(target_tokens, candidate_tokens)
        prefix_score = 0.2 if core_prefix and candidate_model.startswith(core_prefix) else 0.0
        score = overlap_score + prefix_score
        if score < 0.6:
            continue
        scored_candidates.append((slug, score, abs(len(candidate_model) - len(model_name))))

    scored_candidates.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [normalized, *[slug for slug, _, _ in scored_candidates[:8]]]


def _resolve_permaslug_candidates(
    model_id: str,
    available_slugs: list[str],
    permaslug_by_slug: dict[str, str],
) -> list[str]:
    """Resolve permaslug candidates for OpenRouter scraper model stats."""
    resolved_candidates: list[str] = []
    seen: set[str] = set()
    for slug_candidate in _build_slug_fallback_candidates(model_id, available_slugs):
        permaslug = permaslug_by_slug.get(slug_candidate)
        if not permaslug or permaslug in seen:
            continue
        seen.add(permaslug)
        resolved_candidates.append(permaslug)
    return resolved_candidates


def _fetch_performance_for_permaslug(
    permaslug: str,
    *,
    timeout_ms: int,
    max_retries: int,
    retry_base_delay_ms: int,
) -> tuple[dict[str, object], object]:
    """Return performance for permaslug."""
    query = urlencode({"permaslug": permaslug})
    with httpx.Client(timeout=timeout_ms / 1000) as client:
        throughput = _fetch_json_with_retry(
            client,
            url=f"{OPENROUTER_THROUGHPUT_URL}?{query}",
            max_retries=max_retries,
            retry_base_delay_ms=retry_base_delay_ms,
        )
        latency = _fetch_json_with_retry(
            client,
            url=f"{OPENROUTER_LATENCY_URL}?{query}",
            max_retries=max_retries,
            retry_base_delay_ms=retry_base_delay_ms,
        )
        latency_e2e = _fetch_json_with_retry(
            client,
            url=f"{OPENROUTER_E2E_LATENCY_URL}?{query}",
            max_retries=max_retries,
            retry_base_delay_ms=retry_base_delay_ms,
        )
        effective_pricing = _fetch_json_with_retry(
            client,
            url=f"{OPENROUTER_EFFECTIVE_PRICING_URL}?{query}",
            max_retries=max_retries,
            retry_base_delay_ms=retry_base_delay_ms,
        )

    performance: dict[str, object] = {
        "throughput": throughput,
        "latency": latency,
        "latency_e2e": latency_e2e,
    }
    return performance, effective_pricing


def _fetch_best_available_model_stats(
    model_id: str,
    *,
    available_slugs: list[str],
    permaslug_by_slug: dict[str, str],
    timeout_ms: int,
    max_retries: int,
    retry_base_delay_ms: int,
) -> OpenRouterScrapedModel:
    """Return best available model stats."""
    permaslug_candidates = _resolve_permaslug_candidates(model_id, available_slugs, permaslug_by_slug)
    if not permaslug_candidates:
        return _empty_scraped_model(model_id)

    first_resolved: OpenRouterScrapedModel | None = None
    for permaslug in permaslug_candidates:
        try:
            performance_response, pricing_response = _fetch_performance_for_permaslug(
                permaslug,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                retry_base_delay_ms=retry_base_delay_ms,
            )
            performance = _summarize_performance(performance_response)
            pricing = _summarize_pricing(pricing_response)
            resolved_model: OpenRouterScrapedModel = {
                "id": model_id,
                "permaslug": permaslug,
                "performance": performance,
                "pricing": pricing,
            }
            if first_resolved is None:
                first_resolved = resolved_model
            if _has_meaningful_performance(performance) or _has_meaningful_pricing(pricing):
                return resolved_model
        except Exception:
            continue

    return first_resolved or _empty_scraped_model(model_id)


def get_openrouter_scraped_stats(
    options: OpenRouterScraperOptions | None = None,
) -> dict[str, Any]:
    """Return openrouter scraped stats."""
    options = options or {}
    raw_model_ids = options.get("model_ids") or []
    unique_model_ids = list(dict.fromkeys(model_id.strip() for model_id in raw_model_ids if isinstance(model_id, str) and model_id.strip()))
    timeout_ms = int(options.get("timeout_ms") or DEFAULT_TIMEOUT_MS)
    concurrency = max(1, int(options.get("concurrency") or DEFAULT_CONCURRENCY))
    max_retries = max(1, int(options.get("max_retries") or DEFAULT_MAX_RETRIES))
    retry_base_delay_ms = int(options.get("retry_base_delay_ms") or DEFAULT_RETRY_BASE_DELAY_MS)

    empty_payload = {
        "fetched_at_epoch_seconds": int(time.time()),
        "total_requested_models": len(unique_model_ids),
        "total_resolved_models": 0,
        "models": [_empty_scraped_model(model_id) for model_id in unique_model_ids],
    }
    if not unique_model_ids:
        return empty_payload

    try:
        with httpx.Client(timeout=timeout_ms / 1000) as client:
            model_directory = _fetch_json_with_retry(
                client,
                url=OPENROUTER_MODELS_URL,
                max_retries=max_retries,
                retry_base_delay_ms=retry_base_delay_ms,
            )
        data = model_directory.get("data") if isinstance(model_directory, dict) else None
        permaslug_by_slug = _build_permaslug_lookup(data)
        available_slugs = list(permaslug_by_slug.keys())
        if not available_slugs:
            return empty_payload

        worker_count = min(concurrency, len(unique_model_ids))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            models = list(
                executor.map(
                    lambda model_id: _fetch_best_available_model_stats(
                        model_id,
                        available_slugs=available_slugs,
                        permaslug_by_slug=permaslug_by_slug,
                        timeout_ms=timeout_ms,
                        max_retries=max_retries,
                        retry_base_delay_ms=retry_base_delay_ms,
                    ),
                    unique_model_ids,
                )
            )
        return {
            "fetched_at_epoch_seconds": int(time.time()),
            "total_requested_models": len(unique_model_ids),
            "total_resolved_models": sum(1 for model in models if model["permaslug"] is not None),
            "models": models,
        }
    except Exception:
        return empty_payload


def get_openrouter_model_stats(
    options: OpenRouterModelOptions | None = None,
) -> dict[str, Any]:
    """Return openrouter model stats."""
    model_id = (options or {}).get("model_id")
    if not isinstance(model_id, str):
        return {}

    scraper_options: OpenRouterScraperOptions = {"model_ids": [model_id]}
    if options:
        for option_key in ("timeout_ms", "concurrency", "max_retries", "retry_base_delay_ms"):
            option_value = options.get(option_key)
            if option_value is not None:
                scraper_options[option_key] = option_value

    payload = get_openrouter_scraped_stats(scraper_options)
    models = payload.get("models")
    if isinstance(models, list) and models:
        first_model = models[0]
        if isinstance(first_model, dict):
            return first_model
    return cast(dict[str, Any], _empty_scraped_model(model_id))
