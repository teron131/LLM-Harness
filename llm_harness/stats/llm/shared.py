"""Shared helpers for native Python LLM stats modules."""

from __future__ import annotations

import math
import re
from typing import Any

PRIMARY_PROVIDER_ID = "openrouter"
FALLBACK_PROVIDER_IDS = {"openai", "google", "anthropic"}


def as_record(value: Any) -> dict[str, Any]:
    """Return mapping-like source payloads as mutable records owned by stats stages."""
    return value if isinstance(value, dict) else {}


def as_finite_number(value: Any) -> float | None:
    """Return finite numeric source values from source payload fields."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if math.isfinite(numeric_value) else None


def normalize_model_token(value: str) -> str:
    """Normalize a model token for matching."""
    normalized = value.lower()
    normalized = re.sub(r"[._:\s]+", "-", normalized)
    normalized = re.sub(r"[^a-z0-9/-]+", "", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-/")


def model_slug_from_model_id(model_id: Any) -> str | None:
    """Derive a model slug from a model id."""
    if not isinstance(model_id, str) or not model_id:
        return None
    slug = model_id.split("/")[-1]
    return slug or None


def normalize_provider_model_id(model_id: str) -> str:
    """Normalize a provider/model identifier."""
    slash_index = model_id.find("/")
    if slash_index <= 0:
        return re.sub(r"-+", "-", model_id.lower().replace(".", "-"))
    provider = model_id[:slash_index].lower()
    base_model_id = re.sub(
        r"-+",
        "-",
        model_id[slash_index + 1 :].lower().replace(".", "-"),
    )
    return f"{provider}/{base_model_id}"
