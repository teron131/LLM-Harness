"""Arena AI image source (native Python implementation)."""

from __future__ import annotations

import re
from typing import Any, TypedDict

from ...utils import (
    as_finite_number,
    fetch_with_timeout,
    now_epoch_seconds,
    percentile_rank,
)

ARENA_AI_BASE_URL = "https://arena.ai/leaderboard/text-to-image"
REQUEST_TIMEOUT_SECONDS = 30.0
MIN_VALID_ROWS = 20
MIN_VALID_CATEGORIES = 4
ARENA_AI_DEFAULT_CATEGORY_SLUGS = [
    "commercial-design",
    "3d-modeling",
    "cartoon",
    "photorealistic",
    "art",
    "portraits",
    "text-rendering",
]
ARENA_AI_GROUPED_CATEGORY_SLUGS = {
    "photorealistic": ["photorealistic", "portraits"],
    "illustrative": ["cartoon", "art"],
    "contextual": ["commercial-design", "3d-modeling", "text-rendering"],
}
ARENA_AI_GROUP_BY_SLUG = {
    slug: group_name
    for group_name, slugs in ARENA_AI_GROUPED_CATEGORY_SLUGS.items()
    for slug in slugs
}


class ArenaAiImageOptions(TypedDict, total=False):
    """Options for Arena AI image benchmark source."""

    category_slugs: list[str]
    min_valid_rows: int
    min_valid_categories: int


def _detect_challenge(html: str) -> bool:
    """Detect the challenge."""
    return bool(
        re.search(
            r"challenge-platform|__CF\$cv\$params|Verify you are human|Security Verification",
            html,
            re.IGNORECASE,
        )
    )


def _extract_leaderboard_rows(html: str) -> list[dict[str, Any]]:
    """Extract the leaderboard rows."""
    title_matches = list(re.finditer(r'<a[^>]*\btitle="([^"]+)"', html))
    rows: list[dict[str, Any]] = []
    for index, match in enumerate(title_matches):
        title = match.group(1) if match.group(1) else ""
        start = match.start()
        end = (
            title_matches[index + 1].start()
            if index + 1 < len(title_matches)
            else min(start + 4000, len(html))
        )
        row_block = html[start:end]
        score_match = re.search(r">(\d{3,4})</span><span[^>]*>±(\d+)</span>", row_block)
        votes_match = re.search(
            r"±\d+</span>(?:.|\n){0,1200}?>([\d,]{2,})</span>", row_block
        )
        provider_match = re.search(r">([^<]+ · [^<]+)</span>", row_block)
        rows.append(
            {
                "model": title,
                "provider": provider_match.group(1) if provider_match else None,
                "score": as_finite_number(
                    score_match.group(1) if score_match else None
                ),
                "ci95": f"±{score_match.group(2)}" if score_match else None,
                "votes": as_finite_number(votes_match.group(1).replace(",", ""))
                if votes_match
                else None,
            }
        )
    return rows


def _fetch_category(base_url: str, category_slug: str) -> dict[str, Any]:
    """Return category."""
    source_url = f"{base_url}/{category_slug}"
    try:
        response = fetch_with_timeout(
            source_url,
            headers={
                "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
                "accept-language": "en-US,en;q=0.9",
            },
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        html = response.text
        rows_with_score = [
            row for row in _extract_leaderboard_rows(html) if row["score"] is not None
        ]
        return {
            "fetched_at_epoch_seconds": now_epoch_seconds(),
            "category_slug": category_slug,
            "source_url": source_url,
            "final_url": str(response.url),
            "status_code": response.status_code,
            "challenge_detected": _detect_challenge(html),
            "rows_with_score": len(rows_with_score),
            "rows": rows_with_score,
        }
    except Exception:
        return {
            "fetched_at_epoch_seconds": now_epoch_seconds(),
            "category_slug": category_slug,
            "source_url": source_url,
            "final_url": None,
            "status_code": None,
            "challenge_detected": False,
            "rows_with_score": 0,
            "rows": [],
        }


def _round4(value: float) -> float:
    """Helper for round4."""
    return round(value, 4)


def _weighted_score_or_average(
    weighted_sum: float, votes_sum: float, score_sum: float, count: int | float
) -> float | None:
    """Helper for weighted score or average."""
    if votes_sum > 0:
        return _round4(weighted_sum / votes_sum)
    if count > 0:
        return _round4(score_sum / count)
    return None


def _build_grouped_scores(category_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the grouped scores."""
    grouped = {
        "photorealistic": {
            "score_weighted_sum": 0.0,
            "votes_sum": 0.0,
            "score_sum": 0.0,
            "count": 0,
        },
        "illustrative": {
            "score_weighted_sum": 0.0,
            "votes_sum": 0.0,
            "score_sum": 0.0,
            "count": 0,
        },
        "contextual": {
            "score_weighted_sum": 0.0,
            "votes_sum": 0.0,
            "score_sum": 0.0,
            "count": 0,
        },
    }
    for category_slug, row in category_rows.items():
        group_name = ARENA_AI_GROUP_BY_SLUG.get(category_slug)
        if not group_name:
            continue
        score = row.get("score") or 0
        votes = row.get("votes") or 0
        grouped[group_name]["score_weighted_sum"] += score * votes
        grouped[group_name]["votes_sum"] += votes
        grouped[group_name]["score_sum"] += score
        grouped[group_name]["count"] += 1
    grouped_total_votes = sum(bucket["votes_sum"] for bucket in grouped.values())
    weighted_scores = {
        "photorealistic": _weighted_score_or_average(
            grouped["photorealistic"]["score_weighted_sum"],
            grouped["photorealistic"]["votes_sum"],
            grouped["photorealistic"]["score_sum"],
            grouped["photorealistic"]["count"],
        ),
        "illustrative": _weighted_score_or_average(
            grouped["illustrative"]["score_weighted_sum"],
            grouped["illustrative"]["votes_sum"],
            grouped["illustrative"]["score_sum"],
            grouped["illustrative"]["count"],
        ),
        "contextual": _weighted_score_or_average(
            grouped["contextual"]["score_weighted_sum"],
            grouped["contextual"]["votes_sum"],
            grouped["contextual"]["score_sum"],
            grouped["contextual"]["count"],
        ),
        "grouped_overall": _weighted_score_or_average(
            sum(bucket["score_weighted_sum"] for bucket in grouped.values()),
            grouped_total_votes,
            sum(bucket["score_sum"] for bucket in grouped.values()),
            sum(bucket["count"] for bucket in grouped.values()),
        ),
    }
    return {
        "weighted_scores": weighted_scores,
        "grouped_votes": {
            "photorealistic": int(grouped["photorealistic"]["votes_sum"]),
            "illustrative": int(grouped["illustrative"]["votes_sum"]),
            "contextual": int(grouped["contextual"]["votes_sum"]),
            "total": int(grouped_total_votes),
        },
    }


def _build_aggregated_rows(
    category_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the aggregated rows."""
    by_model: dict[str, dict[str, Any]] = {}
    for payload in category_payloads:
        for index, row in enumerate(payload.get("rows", [])):
            if not isinstance(row, dict):
                continue
            aggregate = by_model.setdefault(
                row["model"],
                {
                    "model": row["model"],
                    "provider": row.get("provider"),
                    "category_rows": {},
                    "category_count": 0,
                    "votes_sum": 0.0,
                    "score_sum": 0.0,
                    "score_weighted_sum": 0.0,
                    "rank_sum": 0.0,
                },
            )
            aggregate["category_rows"][payload["category_slug"]] = {
                "rank": index + 1,
                "score": row.get("score"),
                "ci95": row.get("ci95"),
                "votes": row.get("votes"),
            }
            aggregate["category_count"] += 1
            aggregate["rank_sum"] += index + 1
            aggregate["score_sum"] += row.get("score") or 0
            votes = row.get("votes") or 0
            aggregate["votes_sum"] += votes
            aggregate["score_weighted_sum"] += (row.get("score") or 0) * votes
    aggregated_rows: list[dict[str, Any]] = []
    for aggregate in by_model.values():
        average_score = (
            _round4(aggregate["score_sum"] / aggregate["category_count"])
            if aggregate["category_count"] > 0
            else None
        )
        vote_weighted_score = (
            _round4(aggregate["score_weighted_sum"] / aggregate["votes_sum"])
            if aggregate["votes_sum"] > 0
            else average_score
        )
        average_rank = (
            _round4(aggregate["rank_sum"] / aggregate["category_count"])
            if aggregate["category_count"] > 0
            else None
        )
        grouped = _build_grouped_scores(aggregate["category_rows"])
        aggregated_rows.append(
            {
                "model": aggregate["model"],
                "provider": aggregate["provider"],
                "category_count": aggregate["category_count"],
                "average_score": average_score,
                "vote_weighted_score": vote_weighted_score,
                "average_rank": average_rank,
                "votes_sum": int(aggregate["votes_sum"]),
                "categories": aggregate["category_rows"],
                "weighted_scores": grouped["weighted_scores"],
                "grouped_votes": grouped["grouped_votes"],
                "percentiles": {
                    "vote_weighted_percentile": None,
                    "photorealistic_percentile": None,
                    "illustrative_percentile": None,
                    "contextual_percentile": None,
                    "grouped_overall_percentile": None,
                },
            }
        )
    aggregated_rows.sort(
        key=lambda row: (
            row["vote_weighted_score"]
            if row["vote_weighted_score"] is not None
            else float("-inf"),
            row["category_count"],
        ),
        reverse=True,
    )
    vote_weighted_values = [row["vote_weighted_score"] for row in aggregated_rows]
    photorealistic_values = [
        row["weighted_scores"]["photorealistic"] for row in aggregated_rows
    ]
    illustrative_values = [
        row["weighted_scores"]["illustrative"] for row in aggregated_rows
    ]
    contextual_values = [
        row["weighted_scores"]["contextual"] for row in aggregated_rows
    ]
    grouped_overall_values = [
        row["weighted_scores"]["grouped_overall"] for row in aggregated_rows
    ]
    return [
        {
            **row,
            "percentiles": {
                "vote_weighted_percentile": percentile_rank(
                    vote_weighted_values, row["vote_weighted_score"]
                ),
                "photorealistic_percentile": percentile_rank(
                    photorealistic_values, row["weighted_scores"]["photorealistic"]
                ),
                "illustrative_percentile": percentile_rank(
                    illustrative_values, row["weighted_scores"]["illustrative"]
                ),
                "contextual_percentile": percentile_rank(
                    contextual_values, row["weighted_scores"]["contextual"]
                ),
                "grouped_overall_percentile": percentile_rank(
                    grouped_overall_values, row["weighted_scores"]["grouped_overall"]
                ),
            },
        }
        for row in aggregated_rows
    ]


def get_arena_ai_image_stats(
    options: ArenaAiImageOptions | None = None,
) -> dict[str, Any]:
    """Fetch and aggregate Arena text-to-image leaderboard categories."""
    options = options or {}
    category_slugs = options.get("category_slugs") or ARENA_AI_DEFAULT_CATEGORY_SLUGS
    min_valid_rows = options.get("min_valid_rows", MIN_VALID_ROWS)
    min_valid_categories = options.get("min_valid_categories", MIN_VALID_CATEGORIES)
    categories = [_fetch_category(ARENA_AI_BASE_URL, slug) for slug in category_slugs]
    valid_categories = [
        category
        for category in categories
        if category["rows_with_score"] >= min_valid_rows
    ]
    rows = _build_aggregated_rows(valid_categories)
    return {
        "fetched_at_epoch_seconds": now_epoch_seconds(),
        "base_url": ARENA_AI_BASE_URL,
        "category_slugs": list(category_slugs),
        "categories": categories,
        "grouped_category_slugs": ARENA_AI_GROUPED_CATEGORY_SLUGS,
        "valid_categories": [
            category["category_slug"] for category in valid_categories
        ],
        "total_valid_categories": len(valid_categories),
        "total_models_aggregated": len(rows),
        "scrape_feasible_now": len(valid_categories) >= min_valid_categories,
        "rows": rows,
    }
