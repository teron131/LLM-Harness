"""Build matcher source rows from Artificial Analysis API rows or scraper rows."""

from __future__ import annotations

from ..shared import as_record, model_slug_from_model_id


def build_source_models_from_artificial_analysis(
    artificial_analysis_models: list[dict],
) -> list[dict]:
    """Build the source models from artificial analysis."""
    return [
        {
            "source_slug": model.get("slug")
            if isinstance(model.get("slug"), str)
            else "",
            "source_name": model.get("name")
            if isinstance(model.get("name"), str)
            else None,
            "source_release_date": model.get("release_date")
            if isinstance(model.get("release_date"), str)
            else None,
        }
        for model in artificial_analysis_models
    ]


def build_source_models_from_scraped_rows(scraped_rows: list[dict]) -> list[dict]:
    """Build the source models from scraped rows."""
    rows: list[dict] = []
    for scraped_row in scraped_rows:
        row = as_record(scraped_row)
        model_id = row.get("model_id") if isinstance(row.get("model_id"), str) else None
        rows.append(
            {
                "source_slug": model_slug_from_model_id(model_id) or "",
                "source_name": model_id,
                "source_release_date": None,
            }
        )
    return rows
