"""Native Python scraper for Artificial Analysis leaderboard rows."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import re
from typing import Any, TypedDict, cast

import httpx

DEFAULT_SCRAPE_URL = "https://artificialanalysis.ai/leaderboards/models"
DEFAULT_TIMEOUT_MS = 30_000
ROW_DETECTION_KEY = "intelligence_index"
SPARSE_COLUMN_NULL_RATIO = 0.5
MODEL_SEARCH_BACKTRACK_CHARS = 20_000
MIN_INTELLIGENCE_COST_TOKEN_THRESHOLD = 1_000_000
NEXT_FLIGHT_CHUNK_REGEX = re.compile(
    r'self\.__next_f\.push\(\[1,"([\s\S]*?)"\]\)</script>'
)

EVALUATION_KEY_HINT_REGEX = re.compile(
    r"(index|bench|mmlu|gpqa|hle|aime|math|vision|omniscience|ifbench|gdpval|lcr|arc|musr|humanity)",
    re.IGNORECASE,
)
NON_EVALUATION_KEY_REGEX = re.compile(
    r"(token|time|speed|price|cost|window|modality|reasoning_model|release_date|display_order|deprecated|deleted|commercial_allowed|frontier_model|is_open_weights|logo|url|license|creator|host|slug|name|id$|^id$|model_|timescale|response|performance|voice|image|audio|video|text)",
    re.IGNORECASE,
)
EVALUATION_EXCLUDED_KEYS = {
    "omniscience",
    "omniscience_accuracy",
    "omniscience_hallucination_rate",
    "intelligence_index_is_estimated",
    "intelligence_index",
    "agentic_index",
    "coding_index",
    "intelligence_index_per_m_output_tokens",
    "intelligence_index_cost",
}
NO_COLUMN_VALUE = object()

ARTIFICIAL_ANALYSIS_EVALS_ONLY_COLUMNS = [
    "model_id",
    "logo",
    "intelligence",
    "intelligence_index_cost",
    "evaluations",
]


class ArtificialAnalysisScraperOptions(TypedDict, total=False):
    """Options for Artificial Analysis scraper."""

    url: str
    timeout_ms: int
    timeoutMs: int
    flatten: bool
    drop_mostly_null_columns: bool
    dropMostlyNullColumns: bool
    selected_columns: list[str]
    selectedColumns: list[str]


class ArtificialAnalysisScrapedRawPayload(TypedDict):
    """Payload for Artificial Analysis scraper."""

    fetched_at_epoch_seconds: int | None
    data: list[dict[str, Any]]


def _now_epoch_seconds() -> int:
    """Return the current Unix epoch time in seconds."""
    return int(datetime.now(UTC).timestamp())


def _as_record(value: Any) -> dict[str, Any]:
    """Convert the input into a plain record for Artificial Analysis scraper."""
    return value if isinstance(value, dict) else {}


def _decode_flight_chunk(raw: str) -> str:
    """Helper for decode flight chunk."""
    try:
        return json.loads(f'"{raw}"')
    except Exception:
        return raw


def _to_absolute_aa_logo_url(value: Any) -> str | None:
    """Helper for to absolute aa logo url."""
    if not isinstance(value, str) or not value:
        return None
    if value.startswith(("http://", "https://")):
        return value
    normalized = value if value.startswith("/") else f"/{value}"
    return f"https://artificialanalysis.ai{normalized}"


def _pick_evaluations(row: dict[str, Any]) -> dict[str, Any]:
    """Pick the evaluations."""
    evaluations: dict[str, Any] = {}
    for key, value in row.items():
        if key in EVALUATION_EXCLUDED_KEYS:
            continue
        if not EVALUATION_KEY_HINT_REGEX.search(key):
            continue
        if NON_EVALUATION_KEY_REGEX.search(key):
            continue
        if isinstance(value, (int, float, bool)) and not isinstance(value, complex):
            evaluations[key] = value
    return evaluations


def _pick_intelligence(row: dict[str, Any]) -> dict[str, Any]:
    """Pick the intelligence."""
    omniscience_breakdown = _as_record(row.get("omniscience_breakdown"))
    omniscience_total = _as_record(omniscience_breakdown.get("total"))
    intelligence = {
        "intelligence_index": row.get("intelligence_index")
        if isinstance(row.get("intelligence_index"), (int, float))
        else None,
        "agentic_index": row.get("agentic_index")
        if isinstance(row.get("agentic_index"), (int, float))
        else None,
        "coding_index": row.get("coding_index")
        if isinstance(row.get("coding_index"), (int, float))
        else None,
        "omniscience_index": row.get("omniscience")
        if isinstance(row.get("omniscience"), (int, float))
        else None,
        "omniscience_accuracy": None,
        "omniscience_nonhallucination_rate": None,
    }
    if isinstance(omniscience_total.get("accuracy"), (int, float)):
        intelligence["omniscience_accuracy"] = omniscience_total["accuracy"]
    if isinstance(omniscience_total.get("hallucination_rate"), (int, float)):
        intelligence["omniscience_nonhallucination_rate"] = omniscience_total[
            "hallucination_rate"
        ]
    return intelligence


def _pick_intelligence_index_cost(row: dict[str, Any]) -> dict[str, Any]:
    """Pick the intelligence index cost."""
    intelligence_token_counts = _as_record(row.get("intelligence_index_token_counts"))
    intelligence_index_cost = _as_record(row.get("intelligence_index_cost"))
    input_tokens = (
        intelligence_token_counts.get("input_tokens")
        if isinstance(intelligence_token_counts.get("input_tokens"), (int, float))
        else row.get("total_input_tokens_api")
        if isinstance(row.get("total_input_tokens_api"), (int, float))
        else row.get("input_tokens")
        if isinstance(row.get("input_tokens"), (int, float))
        else None
    )
    output_tokens = (
        intelligence_token_counts.get("output_tokens")
        if isinstance(intelligence_token_counts.get("output_tokens"), (int, float))
        else None
    )
    answer_tokens = (
        intelligence_token_counts.get("answer_tokens")
        if isinstance(intelligence_token_counts.get("answer_tokens"), (int, float))
        else None
    )
    reasoning_tokens = (
        intelligence_token_counts.get("reasoning_tokens")
        if isinstance(intelligence_token_counts.get("reasoning_tokens"), (int, float))
        else None
    )
    output_from_parts = (
        (answer_tokens or 0) + (reasoning_tokens or 0)
        if (answer_tokens or 0) + (reasoning_tokens or 0) > 0
        else None
    )
    total_tokens = (
        output_tokens
        if isinstance(output_tokens, (int, float))
        else output_from_parts
        if isinstance(output_from_parts, (int, float))
        else row.get("total_answer_tokens_api")
        if isinstance(row.get("total_answer_tokens_api"), (int, float))
        else row.get("output_tokens")
        if isinstance(row.get("output_tokens"), (int, float))
        else None
    )
    return {
        "input_cost": intelligence_index_cost.get("input_cost")
        if isinstance(intelligence_index_cost.get("input_cost"), (int, float))
        else None,
        "reasoning_cost": intelligence_index_cost.get("reasoning_cost")
        if isinstance(intelligence_index_cost.get("reasoning_cost"), (int, float))
        else None,
        "output_cost": intelligence_index_cost.get("output_cost")
        if isinstance(intelligence_index_cost.get("output_cost"), (int, float))
        else None,
        "total_cost": intelligence_index_cost.get("total_cost")
        if isinstance(intelligence_index_cost.get("total_cost"), (int, float))
        else None,
        "input_tokens": input_tokens,
        "reasoning_tokens": reasoning_tokens,
        "answer_tokens": answer_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
        if isinstance(total_tokens, (int, float))
        and total_tokens >= MIN_INTELLIGENCE_COST_TOKEN_THRESHOLD
        else None,
    }


def _normalize_undefined_to_null(value: Any) -> Any:
    """Normalize the undefined to null."""
    if isinstance(value, list):
        return [_normalize_undefined_to_null(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_undefined_to_null(nested_value)
            for key, nested_value in value.items()
        }
    return value


def _extract_flight_corpus(page_html: str) -> str:
    """Extract the flight corpus."""
    return "\n".join(
        _decode_flight_chunk(match.group(1) or "")
        for match in NEXT_FLIGHT_CHUNK_REGEX.finditer(page_html)
    )


def _find_object_end(corpus: str, start_index: int) -> int:
    """Find the object end."""
    depth = 0
    in_string = False
    escaping = False

    for index in range(start_index, len(corpus)):
        char = corpus[index]
        if in_string:
            if escaping:
                escaping = False
            elif char == "\\":
                escaping = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _parse_json_object(value: str) -> dict[str, Any] | None:
    """Parse the json object."""
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _get_row_identifier(row: dict[str, Any]) -> str | None:
    """Return row identifier."""
    for key in ("id", "model_id", "slug"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return None


def _flatten_expanded_row(row: dict[str, Any]) -> dict[str, Any]:
    """Helper for flatten expanded row."""
    timescale_data = _as_record(row.get("timescaleData"))
    response_time_metrics = _as_record(row.get("end_to_end_response_time_metrics"))
    performance_by_prompt_length = row.get("performanceByPromptLength")
    first_performance_row = (
        _as_record(performance_by_prompt_length[0])
        if isinstance(performance_by_prompt_length, list)
        and performance_by_prompt_length
        else {}
    )

    flattened_row = dict(row)
    for source in (timescale_data, response_time_metrics):
        for key, value in source.items():
            if flattened_row.get(key) is None and value is not None:
                flattened_row[key] = value

    if (
        flattened_row.get("prompt_length_type_default") is None
        and first_performance_row.get("prompt_length_type") is not None
    ):
        flattened_row["prompt_length_type_default"] = first_performance_row[
            "prompt_length_type"
        ]

    return flattened_row


def _is_null_like(value: Any) -> bool:
    """Return whether a scraped value should be treated as null-like."""
    return (
        value is None
        or value == ""
        or value == "$undefined"
        or (isinstance(value, list) and len(value) == 0)
    )


def _drop_mostly_null_columns(
    rows: list[dict[str, Any]],
    null_ratio_threshold: float,
) -> list[dict[str, Any]]:
    """Helper for drop mostly null columns."""
    if not rows:
        return rows

    columns: list[str] = []
    seen_columns: set[str] = set()
    for row in rows:
        for column in row:
            if column not in seen_columns:
                seen_columns.add(column)
                columns.append(column)

    columns_to_drop: set[str] = set()
    for column in columns:
        null_like_count = sum(1 for row in rows if _is_null_like(row.get(column)))
        if null_like_count / len(rows) > null_ratio_threshold:
            columns_to_drop.add(column)

    if not columns_to_drop:
        return rows

    return [
        {
            column: value
            for column, value in row.items()
            if column not in columns_to_drop
        }
        for row in rows
    ]


def _slugify_provider_name(value: str) -> str:
    """Helper for slugify provider name."""
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", value.lower()))


def _get_provider_slug(row: dict[str, Any], creator: dict[str, Any]) -> str | None:
    """Return provider slug."""
    provider_name = (
        creator.get("name")
        if isinstance(creator.get("name"), str)
        else row.get("provider")
        if isinstance(row.get("provider"), str)
        else None
    )
    if provider_name is None:
        return None
    return _slugify_provider_name(provider_name)


def _build_row_selection_context(row: dict[str, Any]) -> dict[str, Any]:
    """Build the row selection context."""
    creator = _as_record(row.get("creator"))
    model_creators = _as_record(row.get("model_creators"))
    provider_slug = _get_provider_slug(row, creator)
    model_slug = (
        row.get("slug")
        if isinstance(row.get("slug"), str) and row.get("slug")
        else None
    )
    creator_slug = (
        model_creators.get("slug")
        if isinstance(model_creators.get("slug"), str) and model_creators.get("slug")
        else provider_slug
    )
    model_url = row.get("model_url")
    model_url_slug = (
        model_url.removeprefix("/models/") if isinstance(model_url, str) else None
    )
    return {
        "creator": creator,
        "model_creators": model_creators,
        "provider_slug": provider_slug,
        "model_slug": model_slug,
        "creator_slug": creator_slug,
        "model_url_slug": model_url_slug,
    }


def _select_modalities(row: dict[str, Any], modality_type: str) -> list[str]:
    """Select the modalities."""
    return [
        name
        for key, name in (
            (f"{modality_type}_modality_text", "text"),
            (f"{modality_type}_modality_image", "image"),
            (f"{modality_type}_modality_video", "video"),
            (f"{modality_type}_modality_speech", "speech"),
        )
        if row.get(key)
    ]


def _select_reasoning_flag(row: dict[str, Any]) -> bool | None:
    """Select the reasoning flag."""
    if isinstance(row.get("reasoning_model"), bool):
        return row["reasoning_model"]
    if isinstance(row.get("isReasoning"), bool):
        return row["isReasoning"]
    return None


def _get_selected_column_value(
    column: str,
    row: dict[str, Any],
    context: dict[str, Any],
) -> Any:
    """Return selected column value."""
    creator = context["creator"]
    model_creators = context["model_creators"]
    provider_slug = context["provider_slug"]
    model_slug = context["model_slug"]
    creator_slug = context["creator_slug"]
    model_url_slug = context["model_url_slug"]

    if column == "id":
        if provider_slug and model_slug:
            return f"{provider_slug}/{model_slug}"
        return model_slug or row.get("id")
    if column == "model_url":
        return row.get("model_url") or row.get("id")
    if column == "model_id":
        if creator_slug and model_url_slug:
            return f"{creator_slug}/{model_url_slug}"
        return model_url_slug or row.get("model_url")
    if column == "name":
        return (
            row.get("short_name")
            or row.get("shortName")
            or row.get("name")
            or row.get("slug")
        )
    if column == "provider":
        return (
            provider_slug
            or creator.get("name")
            or model_creators.get("name")
            or row.get("model_creator_id")
            or row.get("creator_name")
        )
    if column == "logo":
        return _to_absolute_aa_logo_url(
            row.get("logo_small_url")
            or row.get("logo_url")
            or row.get("logoSmall")
            or row.get("logo_small")
            or model_creators.get("logo_small_url")
            or model_creators.get("logo_url")
            or model_creators.get("logo_small")
            or model_creators.get("logo")
            or creator.get("logo_small_url")
            or creator.get("logo_url")
            or creator.get("logo_small")
            or creator.get("logo")
        )
    if column == "attachment":
        return bool(
            row.get("input_modality_image")
            or row.get("input_modality_video")
            or row.get("input_modality_speech")
        )
    if column in {"reasoning", "reasoning_model"}:
        return _select_reasoning_flag(row)
    if column == "input_modalities":
        return _select_modalities(row, "input")
    if column == "output_modalities":
        return _select_modalities(row, "output")
    if column == "release_date":
        return (
            row.get("release_date")
            if isinstance(row.get("release_date"), str)
            else None
        )
    if column == "input_tokens":
        intelligence_token_counts = _as_record(
            row.get("intelligence_index_token_counts")
        )
        return (
            intelligence_token_counts.get("input_tokens")
            or row.get("total_input_tokens_api")
            or row.get("input_tokens")
        )
    if column == "output_tokens":
        intelligence_token_counts = _as_record(
            row.get("intelligence_index_token_counts")
        )
        answer_tokens = (
            intelligence_token_counts.get("answer_tokens")
            if isinstance(intelligence_token_counts.get("answer_tokens"), (int, float))
            else None
        )
        reasoning_tokens = (
            intelligence_token_counts.get("reasoning_tokens")
            if isinstance(
                intelligence_token_counts.get("reasoning_tokens"), (int, float)
            )
            else None
        )
        output_from_parts = (
            (answer_tokens or 0) + (reasoning_tokens or 0)
            if (answer_tokens or 0) + (reasoning_tokens or 0) > 0
            else None
        )
        return (
            intelligence_token_counts.get("output_tokens")
            or output_from_parts
            or row.get("total_answer_tokens_api")
            or row.get("output_tokens")
        )
    if column == "median_speed":
        return row.get("median_output_speed") or _as_record(
            row.get("timescaleData")
        ).get("median_output_speed")
    if column == "median_time":
        return row.get("median_time_to_first_chunk") or _as_record(
            row.get("timescaleData")
        ).get("median_time_to_first_chunk")
    if column == "evaluations":
        return _pick_evaluations(row)
    if column == "intelligence":
        return _pick_intelligence(row)
    if column == "intelligence_index_cost":
        return _pick_intelligence_index_cost(row)
    return NO_COLUMN_VALUE


def _select_columns(
    rows: list[dict[str, Any]],
    selected_columns: list[str],
) -> list[dict[str, Any]]:
    """Select the columns."""
    keep_columns = [
        column for column in selected_columns if isinstance(column, str) and column
    ]
    if not keep_columns:
        return rows

    selected_rows: list[dict[str, Any]] = []
    for row in rows:
        selected_row: dict[str, Any] = {}
        context = _build_row_selection_context(row)
        for column in keep_columns:
            column_value = _get_selected_column_value(column, row, context)
            if column_value is NO_COLUMN_VALUE:
                selected_row[column] = _normalize_undefined_to_null(row.get(column))
            else:
                selected_row[column] = _normalize_undefined_to_null(column_value)
        selected_rows.append(selected_row)
    return selected_rows


def _extract_rows_from_corpus(corpus: str) -> list[dict[str, Any]]:
    """Extract the rows from corpus."""
    detection_token = f'"{ROW_DETECTION_KEY}":'
    rows_by_id: dict[str, dict[str, Any]] = {}
    cursor = 0

    while True:
        hit_index = corpus.find(detection_token, cursor)
        if hit_index == -1:
            break
        cursor = hit_index + len(detection_token)
        search_start = max(0, hit_index - MODEL_SEARCH_BACKTRACK_CHARS)

        for back_index in range(hit_index, search_start - 1, -1):
            if corpus[back_index] != "{":
                continue
            end_index = _find_object_end(corpus, back_index)
            if end_index == -1 or end_index < hit_index:
                continue
            row = _parse_json_object(corpus[back_index : end_index + 1])
            if row is None or ROW_DETECTION_KEY not in row:
                continue
            row_id = _get_row_identifier(row)
            if row_id is None:
                continue
            rows_by_id[row_id] = row
            break

    return list(rows_by_id.values())


def _get_option(
    options: ArtificialAnalysisScraperOptions,
    snake_key: str,
    camel_key: str,
) -> Any:
    """Return option."""
    option_values = cast(dict[str, Any], options)
    if snake_key in option_values:
        return option_values[snake_key]
    if camel_key in option_values:
        return option_values[camel_key]
    return None


def process_artificial_analysis_scraped_rows(
    rows: list[dict[str, Any]],
    options: ArtificialAnalysisScraperOptions | None = None,
) -> list[dict[str, Any]]:
    """Helper for process artificial analysis scraped rows."""
    options = options or {}
    if not isinstance(rows, list):
        return []

    safe_rows = [row for row in rows if isinstance(row, dict)]
    should_flatten = _get_option(options, "flatten", "flatten")
    should_drop_mostly_null_columns = _get_option(
        options,
        "drop_mostly_null_columns",
        "dropMostlyNullColumns",
    )
    selected_columns = _get_option(options, "selected_columns", "selectedColumns")

    normalized_rows = (
        [_flatten_expanded_row(row) for row in safe_rows]
        if should_flatten is not False
        else safe_rows
    )
    cleaned_rows = (
        _drop_mostly_null_columns(normalized_rows, SPARSE_COLUMN_NULL_RATIO)
        if should_drop_mostly_null_columns is not False
        else normalized_rows
    )
    return _select_columns(cleaned_rows, list(selected_columns or []))


def get_artificial_analysis_scraped_raw_stats(
    options: ArtificialAnalysisScraperOptions | None = None,
) -> ArtificialAnalysisScrapedRawPayload:
    """Return artificial analysis scraped raw stats."""
    options = options or {}
    try:
        url = options.get("url") or DEFAULT_SCRAPE_URL
        timeout_ms = (
            _get_option(options, "timeout_ms", "timeoutMs") or DEFAULT_TIMEOUT_MS
        )
        with httpx.Client(timeout=float(timeout_ms) / 1000.0) as client:
            response = client.get(url)
        response.raise_for_status()
        page_html = response.text
        corpus = _extract_flight_corpus(page_html)
        data = _extract_rows_from_corpus(corpus)
        return {
            "fetched_at_epoch_seconds": _now_epoch_seconds(),
            "data": data,
        }
    except Exception:
        return {
            "fetched_at_epoch_seconds": None,
            "data": [],
        }


def get_artificial_analysis_scraped_stats(
    options: ArtificialAnalysisScraperOptions | None = None,
) -> ArtificialAnalysisScrapedRawPayload:
    """Return artificial analysis scraped stats."""
    raw_payload = get_artificial_analysis_scraped_raw_stats(options)
    return {
        "fetched_at_epoch_seconds": raw_payload["fetched_at_epoch_seconds"],
        "data": process_artificial_analysis_scraped_rows(
            raw_payload["data"],
            options,
        ),
    }


def get_artificial_analysis_scraped_evals_only_stats(
    options: ArtificialAnalysisScraperOptions | None = None,
) -> ArtificialAnalysisScrapedRawPayload:
    """Return artificial analysis scraped evals only stats."""
    merged_options = cast(ArtificialAnalysisScraperOptions, dict(options or {}))
    merged_options["selected_columns"] = list(ARTIFICIAL_ANALYSIS_EVALS_ONLY_COLUMNS)
    return get_artificial_analysis_scraped_stats(merged_options)
