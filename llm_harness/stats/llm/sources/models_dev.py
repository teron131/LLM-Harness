"""models.dev LLM source ported from the JS stats pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, TypedDict

import httpx
import polars as pl

from ...utils import as_finite_number, now_epoch_seconds

MODELS_DEV_URL = "https://models.dev/api.json"
LOOKBACK_DAYS = 365
REQUEST_TIMEOUT_SECONDS = 30.0


class ModelsDevOptions(TypedDict):
    """Reserved options for future extension."""


class ModelsDevOutputPayload(TypedDict):
    """Payload for Models.dev source recent model stats."""

    fetched_at_epoch_seconds: int | None
    status_code: int | None
    models: list[dict[str, Any]]


def _iso_date_days_ago(days: int) -> str:
    """Helper for iso date days ago."""
    return (datetime.now(UTC) - timedelta(days=days)).date().isoformat()


def _is_recent_date(iso_date: str | None, cutoff_iso_date: str) -> bool:
    """Return whether the current value is valid for Models.dev source recent model stats."""
    return bool(iso_date) and iso_date >= cutoff_iso_date


def _fetch_models_dev() -> dict[str, Any]:
    """Return models dev."""
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = client.get(MODELS_DEV_URL)
    response.raise_for_status()
    return {
        "fetched_at_epoch_seconds": now_epoch_seconds(),
        "status_code": response.status_code,
        "payload": response.json(),
    }


def _flatten_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Helper for flatten models."""
    rows: list[dict[str, Any]] = []
    for provider_id, provider in payload.items():
        if not isinstance(provider, dict):
            continue
        provider_name = provider.get("name") or provider_id
        models = provider.get("models") or {}
        if not isinstance(models, dict):
            continue
        for model_id, model in models.items():
            if not isinstance(model, dict):
                continue
            rows.append(
                {
                    "provider_id": provider_id,
                    "provider_name": provider_name,
                    "model_id": model.get("id") or model_id,
                    "model": model,
                }
            )
    return rows


def _rank_recent_models(
    models: list[dict[str, Any]],
    cutoff_iso_date: str,
) -> list[dict[str, Any]]:
    """Rank the recent models."""
    ranking_rows = [
        {
            "row_index": row_index,
            "release_date": (row.get("model") or {}).get("release_date"),
            "output_cost": as_finite_number(
                ((row.get("model") or {}).get("cost") or {}).get("output")
            ),
        }
        for row_index, row in enumerate(models)
        if _is_recent_date(
            (row.get("model") or {}).get("release_date"), cutoff_iso_date
        )
    ]
    if not ranking_rows:
        return []
    ranked_indices = (
        pl.DataFrame(ranking_rows)
        .with_columns(pl.col("output_cost").fill_null(float("inf")))
        .sort(
            by=["output_cost", "release_date"],
            descending=[False, True],
        )
        .get_column("row_index")
        .to_list()
    )
    return [models[row_index] for row_index in ranked_indices]


def get_models_dev_stats(
    _options: ModelsDevOptions | None = None,
) -> ModelsDevOutputPayload:
    """Fetch, flatten, and rank recent models from models.dev."""
    try:
        source_payload = _fetch_models_dev()
        cutoff_iso_date = _iso_date_days_ago(LOOKBACK_DAYS)
        ranked = _rank_recent_models(
            _flatten_models(source_payload.get("payload") or {}),
            cutoff_iso_date,
        )
        return {
            "fetched_at_epoch_seconds": source_payload["fetched_at_epoch_seconds"],
            "status_code": source_payload["status_code"],
            "models": ranked,
        }
    except Exception:
        return {
            "fetched_at_epoch_seconds": None,
            "status_code": None,
            "models": [],
        }
