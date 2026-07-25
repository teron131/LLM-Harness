"""Public image stats API and staged pipeline helpers."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from .cache import (
    DEFAULT_OUTPUT_PATH,
    current_epoch_seconds,
    load_image_stats_selected_from_cache,
    save_image_stats_selected_to_path,
)
from .final_stage import build_final_models
from .match_stage import build_matched_rows
from .source_stage import fetch_source_data
from .types import (
    ImageStatsSelectedOptions,
    ImageStatsSelectedOptionsModel,
    ImageStatsSelectedPayload,
    ImageStatsSelectedPayloadModel,
)

__all__ = [
    "ImageStatsSelectedOptions",
    "ImageStatsSelectedPayload",
    "get_image_stats_selected",
    "save_image_stats_selected",
]


def save_image_stats_selected(
    payload: ImageStatsSelectedPayload,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Save the selected Image stats selection payload."""
    validated_payload = ImageStatsSelectedPayloadModel.model_validate(payload)
    save_image_stats_selected_to_path(validated_payload.model_dump(), output_path)


def get_image_stats_selected(
    options: ImageStatsSelectedOptions | None = None,
) -> ImageStatsSelectedPayload:
    """Build the final selected image stats payload with cache-first list mode."""
    options_model = ImageStatsSelectedOptionsModel.model_validate(options or {})
    model_id = options_model.id
    try:
        if model_id is None:
            cached_payload = load_image_stats_selected_from_cache(DEFAULT_OUTPUT_PATH)
            if cached_payload is not None:
                return cast(
                    ImageStatsSelectedPayload,
                    ImageStatsSelectedPayloadModel.model_validate(
                        cached_payload
                    ).model_dump(),
                )
        source_data = fetch_source_data()
        matched_rows = build_matched_rows(source_data)
        models = build_final_models(matched_rows, model_id)
        fetched_at = current_epoch_seconds()
        if model_id is not None:
            return cast(
                ImageStatsSelectedPayload,
                ImageStatsSelectedPayloadModel.model_validate(
                    {
                        "fetched_at_epoch_seconds": fetched_at,
                        "models": models,
                    }
                ).model_dump(),
            )
        list_payload = cast(
            ImageStatsSelectedPayload,
            ImageStatsSelectedPayloadModel.model_validate(
                {
                    "fetched_at_epoch_seconds": fetched_at,
                    "models": models,
                }
            ).model_dump(),
        )
        save_image_stats_selected(list_payload, DEFAULT_OUTPUT_PATH)
        return list_payload
    except Exception:
        return cast(
            ImageStatsSelectedPayload,
            ImageStatsSelectedPayloadModel(
                fetched_at_epoch_seconds=None,
                models=[],
            ).model_dump(),
        )
