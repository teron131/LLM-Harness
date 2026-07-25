"""Native Python image matcher aligned with the JS image stats pipeline."""

from __future__ import annotations

import re
from typing import Any, TypedDict

from .sources.arena_ai import get_arena_ai_image_stats
from .sources.artificial_analysis import get_artificial_analysis_image_stats

DEFAULT_MAX_CANDIDATES = 3
PROVIDER_MATCH_REWARD = 2
MIN_ACCEPTED_CANDIDATE_SCORE = 3
VOID_THRESHOLD_RANGE_RATIO = 0.12
TOKEN_COVERAGE_WEIGHT = 8
QUALIFIER_MATCH_WEIGHT = 2.5
QUALIFIER_MISS_PENALTY = 2
MAX_QUALIFIER_PENALTY = 6
RANK_PROXIMITY_RADIUS = 10
RANK_PROXIMITY_MAX_BONUS = 3
TOP_RANK_PROTECTION_COUNT = 20
TOP_RANK_PROTECTION_MARGIN = 0.6
TOP_RANK_PROTECTION_THRESHOLD_DELTA = 0.6
VERSION_EXACT_BONUS = 10
VERSION_MAJOR_EXACT_BONUS = 4
VERSION_MAJOR_MISMATCH_PENALTY = 8
VERSION_MINOR_MISMATCH_PENALTY_SCALE = 1.25
VERSION_MINOR_MISMATCH_PENALTY_MAX = 4
VERSION_MISSING_PENALTY = 1.5
STRUCTURED_VERSION_EXACT_BONUS = 10
STRUCTURED_VERSION_MISMATCH_PENALTY = 6
VERSION_FAMILY_GUARD_PENALTY = 5
FAMILY_OVERLAP_WEIGHT = 5
NOISE_TOKENS = {
    "image",
    "images",
    "model",
    "models",
    "generate",
    "generation",
    "preview",
    "version",
    "ver",
    "ai",
    "the",
    "and",
    "for",
    "with",
}
QUALIFIER_TOKENS = {
    "ultra",
    "max",
    "pro",
    "mini",
    "dev",
    "fast",
    "flash",
    "standard",
    "flex",
    "turbo",
    "lite",
    "instruct",
    "high",
    "low",
    "medium",
    "plus",
    "base",
}
PROVIDER_NOISE_TOKENS = {
    "openai",
    "google",
    "alibaba",
    "tencent",
    "bytedance",
    "black",
    "forest",
    "labs",
    "microsoft",
    "xai",
    "recraft",
    "ideogram",
    "leonardo",
}


class ImageMatchCandidate(TypedDict):
    """Candidate model for Image model matching."""

    arena_model: str
    arena_provider: str | None
    score: float


class ImageMatchMappedModel(TypedDict):
    """Pydantic model for Image model matching."""

    artificial_analysis_slug: str | None
    artificial_analysis_name: str | None
    artificial_analysis_provider: str | None
    best_match: ImageMatchCandidate | None
    candidates: list[ImageMatchCandidate]


class ImageMatchModelMappingPayload(TypedDict):
    """Payload for Image model matching."""

    artificial_analysis_fetched_at_epoch_seconds: int | None
    arena_ai_fetched_at_epoch_seconds: int | None
    total_artificial_analysis_models: int
    total_arena_ai_models: int
    max_candidates: int
    void_threshold: float | None
    voided_count: int
    models: list[ImageMatchMappedModel]


class ImageMatchModelMappingOptions(TypedDict, total=False):
    """Options for Image model matching."""

    max_candidates: int
    artificial_analysis_models: list[dict[str, Any]]
    arena_models: list[dict[str, Any]]


def _as_record(value: Any) -> dict[str, Any]:
    """Convert the input into a plain record for Image model matching."""
    return value if isinstance(value, dict) else {}


def _get_model_creator_name(model: dict[str, Any]) -> str | None:
    """Return model creator name."""
    name = _as_record(model.get("model_creator")).get("name")
    return name if isinstance(name, str) else None


def _normalize_model_name(value: str) -> str:
    """Normalize the model name."""
    normalized = value.lower()
    normalized = re.sub(r"[\[\]()]", " ", normalized)
    normalized = re.sub(r"[._:/]+", "-", normalized)
    normalized = re.sub(r"[^a-z0-9-]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-")


def _split_tokens(value: str) -> list[str]:
    """Split Image model matching into normalized tokens."""
    tokens: list[str] = []
    for token in _normalize_model_name(value).split("-"):
        for part in re.split(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)", token):
            if part:
                tokens.append(part)
    return tokens


def _provider_prefix(provider: str | None) -> str | None:
    """Resolve the provider for Image model matching."""
    if not provider:
        return None
    left = provider.split("·", 1)[0].strip().lower()
    return left or None


def _rank_proximity_bonus(
    artificial_analysis_rank: int | None,
    arena_rank: int | None,
) -> float:
    """Rank the proximity bonus."""
    if artificial_analysis_rank is None or arena_rank is None:
        return 0.0
    gap = abs(artificial_analysis_rank - arena_rank)
    if gap > RANK_PROXIMITY_RADIUS:
        return 0.0
    return (
        (RANK_PROXIMITY_RADIUS - gap + 1) / (RANK_PROXIMITY_RADIUS + 1)
    ) * RANK_PROXIMITY_MAX_BONUS


def _common_prefix_length(left: str, right: str) -> int:
    """Helper for common prefix length."""
    max_length = min(len(left), len(right))
    index = 0
    while index < max_length and left[index] == right[index]:
        index += 1
    return index


def _to_numeric_token(token: str) -> int | None:
    """Helper for to numeric token."""
    if not re.fullmatch(r"\d+", token):
        return None
    numeric = int(token)
    return numeric if numeric >= 0 else None


def _extract_structured_versions(value: str) -> list[str]:
    """Extract the structured versions."""
    matches = re.findall(r"\b\d+\.\d+\b|\b\d+-\d+\b", value.lower())
    seen: set[str] = set()
    ordered: list[str] = []
    for match in matches:
        if match not in seen:
            seen.add(match)
            ordered.append(match)
    return ordered


def _token_similarity(left: str, right: str) -> float:
    """Score token overlap for Image model matching."""
    if left == right:
        return 1.0
    left_numeric = _to_numeric_token(left)
    right_numeric = _to_numeric_token(right)
    if left_numeric is not None and right_numeric is not None:
        gap = abs(left_numeric - right_numeric)
        return max(0.0, 1 - (gap / max(1, left_numeric, right_numeric)))
    if left in right or right in left:
        shorter = max(1, min(len(left), len(right)))
        return min(0.85, shorter / max(len(left), len(right)))
    prefix = _common_prefix_length(left, right)
    if prefix >= 2:
        return (prefix / max(1, min(len(left), len(right)))) * 0.7
    return 0.0


def _aligned_token_score(left_tokens: list[str], right_tokens: list[str]) -> float:
    """Score token overlap for Image model matching."""
    memo: dict[tuple[int, int], float] = {}

    def solve(left_index: int, right_index: int) -> float:
        """Solve the image-matching assignment problem."""
        key = (left_index, right_index)
        if key in memo:
            return memo[key]
        if left_index >= len(left_tokens) or right_index >= len(right_tokens):
            memo[key] = 0.0
            return 0.0
        match = _token_similarity(
            left_tokens[left_index],
            right_tokens[right_index],
        ) + solve(left_index + 1, right_index + 1)
        skip_left = solve(left_index + 1, right_index)
        skip_right = solve(left_index, right_index + 1)
        best = max(match, skip_left, skip_right)
        memo[key] = best
        return best

    return solve(0, 0)


def _set_jaccard(left_tokens: list[str], right_tokens: list[str]) -> float:
    """Helper for set jaccard."""
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    if not left_set and not right_set:
        return 0.0
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return (intersection / union) if union > 0 else 0.0


def _positional_exact_matches(left_tokens: list[str], right_tokens: list[str]) -> int:
    """Helper for positional exact matches."""
    limit = min(len(left_tokens), len(right_tokens))
    return sum(1 for index in range(limit) if left_tokens[index] == right_tokens[index])


def _is_distinctive_token(token: str) -> bool:
    """Return whether a token is distinctive enough to help matching."""
    return (
        len(token) >= 3
        and token not in NOISE_TOKENS
        and not re.fullmatch(
            r"\d+",
            token,
        )
    )


def _distinctive_coverage(left_tokens: list[str], right_tokens: list[str]) -> float:
    """Helper for distinctive coverage."""
    left_distinctive = [token for token in left_tokens if _is_distinctive_token(token)]
    if not left_distinctive:
        return 0.0
    right_set = set(right_tokens)
    matched = sum(1 for token in left_distinctive if token in right_set)
    return matched / len(left_distinctive)


def _qualifier_signals(
    left_tokens: list[str],
    right_tokens: list[str],
) -> tuple[float, float]:
    """Helper for qualifier signals."""
    left_qualifiers = [token for token in left_tokens if token in QUALIFIER_TOKENS]
    if not left_qualifiers:
        return 0.0, 0.0
    right_set = set(right_tokens)
    matched = sum(1 for token in left_qualifiers if token in right_set)
    missed = len(left_qualifiers) - matched
    return (
        matched * QUALIFIER_MATCH_WEIGHT,
        min(MAX_QUALIFIER_PENALTY, missed * QUALIFIER_MISS_PENALTY),
    )


def _get_family_anchor_tokens(name: str) -> list[str]:
    """Return family anchor tokens."""
    tokens = [
        token
        for token in _split_tokens(name)
        if _is_distinctive_token(token)
        and token not in QUALIFIER_TOKENS
        and token not in PROVIDER_NOISE_TOKENS
    ]
    seen: set[str] = set()
    anchors: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            anchors.append(token)
    return anchors


def _compute_name_similarity(left: str, right: str) -> float:
    """Compute the name similarity."""
    left_normalized = _normalize_model_name(left)
    right_normalized = _normalize_model_name(right)
    if not left_normalized or not right_normalized:
        return 0.0

    left_tokens = _split_tokens(left)
    right_tokens = _split_tokens(right)
    aligned = _aligned_token_score(left_tokens, right_tokens) / max(
        1,
        max(len(left_tokens), len(right_tokens)),
    )
    jaccard = _set_jaccard(left_tokens, right_tokens)
    positional = _positional_exact_matches(left_tokens, right_tokens) / max(
        1,
        min(len(left_tokens), len(right_tokens)),
    )
    containment = (
        1.0
        if (left_normalized in right_normalized or right_normalized in left_normalized)
        else 0.0
    )
    exact = 1.0 if left_normalized == right_normalized else 0.0
    coverage = _distinctive_coverage(left_tokens, right_tokens)
    qualifier_bonus, qualifier_penalty = _qualifier_signals(
        left_tokens,
        right_tokens,
    )

    left_family_anchors = _get_family_anchor_tokens(left)
    right_family_anchors = _get_family_anchor_tokens(right)
    right_family_anchor_set = set(right_family_anchors)
    family_overlap_count = sum(
        1 for token in left_family_anchors if token in right_family_anchor_set
    )
    family_overlap = (
        family_overlap_count / max(len(left_family_anchors), len(right_family_anchors))
        if max(len(left_family_anchors), len(right_family_anchors)) > 0
        else 0.0
    )
    has_family_signal = bool(left_family_anchors and right_family_anchors)
    has_family_overlap = family_overlap_count > 0

    left_version = [
        value
        for value in (_to_numeric_token(token) for token in left_tokens)
        if value is not None
    ][:2]
    right_version = [
        value
        for value in (_to_numeric_token(token) for token in right_tokens)
        if value is not None
    ][:2]
    version_bonus = 0.0
    version_penalty = 0.0
    if left_version or right_version:
        if has_family_signal and not has_family_overlap:
            version_penalty += VERSION_FAMILY_GUARD_PENALTY
        elif not left_version or not right_version:
            version_penalty += VERSION_MISSING_PENALTY
        else:
            left_major = left_version[0]
            right_major = right_version[0]
            if left_major != right_major:
                version_penalty += VERSION_MAJOR_MISMATCH_PENALTY
            else:
                version_bonus += VERSION_MAJOR_EXACT_BONUS
                if len(left_version) > 1 and len(right_version) > 1:
                    left_minor = left_version[1]
                    right_minor = right_version[1]
                    if left_minor == right_minor:
                        version_bonus += VERSION_EXACT_BONUS
                    else:
                        version_penalty += min(
                            VERSION_MINOR_MISMATCH_PENALTY_MAX,
                            abs(left_minor - right_minor)
                            * VERSION_MINOR_MISMATCH_PENALTY_SCALE,
                        )

    left_structured_versions = _extract_structured_versions(left)
    right_structured_versions = _extract_structured_versions(right)
    if left_structured_versions or right_structured_versions:
        if has_family_signal and not has_family_overlap:
            version_penalty += VERSION_FAMILY_GUARD_PENALTY
        elif not left_structured_versions or not right_structured_versions:
            version_penalty += VERSION_MISSING_PENALTY
        else:
            if left_structured_versions[0] == right_structured_versions[0]:
                version_bonus += STRUCTURED_VERSION_EXACT_BONUS
            else:
                version_penalty += STRUCTURED_VERSION_MISMATCH_PENALTY

    weighted = (
        exact * 10
        + aligned * 8
        + jaccard * 6
        + positional * 5
        + containment * 2
        + coverage * TOKEN_COVERAGE_WEIGHT
        + family_overlap * FAMILY_OVERLAP_WEIGHT
        + qualifier_bonus
        - qualifier_penalty
        - version_penalty
        + version_bonus
    )
    return round(weighted, 4)


def _get_artificial_analysis_names(model: dict[str, Any]) -> list[str]:
    """Return artificial analysis names."""
    names: list[str] = []
    name = model.get("name")
    slug = model.get("slug")
    if isinstance(name, str) and name:
        names.append(name)
    if isinstance(slug, str) and slug:
        names.append(slug)
    return names or [""]


def _compute_artificial_analysis_name_score(
    model: dict[str, Any],
    arena_model_name: str,
) -> float:
    """Compute the artificial analysis name score."""
    display_name = model.get("name") if isinstance(model.get("name"), str) else ""
    slug_name = model.get("slug") if isinstance(model.get("slug"), str) else ""
    if display_name and slug_name and display_name != slug_name:
        display_score = _compute_name_similarity(display_name, arena_model_name)
        slug_score = _compute_name_similarity(slug_name, arena_model_name)
        return round((display_score * 0.8) + (slug_score * 0.2), 4)
    if display_name:
        return _compute_name_similarity(display_name, arena_model_name)
    if slug_name:
        return _compute_name_similarity(slug_name, arena_model_name)
    return 0.0


def _has_family_anchor_overlap(
    artificial_analysis_model: dict[str, Any],
    arena_model_name: str,
) -> bool:
    """Return whether family-anchor tokens overlap between the compared rows."""
    aa_anchors = [
        token
        for name in _get_artificial_analysis_names(artificial_analysis_model)
        for token in _get_family_anchor_tokens(name)
    ]
    if not aa_anchors:
        return True
    arena_anchor_set = set(_get_family_anchor_tokens(arena_model_name))
    return any(token in arena_anchor_set for token in aa_anchors)


def _compute_candidate_score(
    artificial_analysis_model: dict[str, Any],
    arena_model: dict[str, Any],
    artificial_analysis_rank: int | None,
    arena_rank: int | None,
) -> float:
    """Compute the candidate score."""
    base_score = _compute_artificial_analysis_name_score(
        artificial_analysis_model,
        str(arena_model.get("model") or ""),
    )
    aa_provider = _get_model_creator_name(artificial_analysis_model)
    aa_provider = aa_provider.lower() if aa_provider else None
    arena_provider = _provider_prefix(
        arena_model.get("provider")
        if isinstance(arena_model.get("provider"), str)
        else None
    )
    provider_match_bonus = (
        PROVIDER_MATCH_REWARD
        if aa_provider
        and arena_provider
        and (aa_provider in arena_provider or arena_provider in aa_provider)
        else 0.0
    )
    score = (
        base_score
        + provider_match_bonus
        + _rank_proximity_bonus(artificial_analysis_rank, arena_rank)
    )
    return round(score, 4)


def _is_accepted_best_candidate(candidates: list[ImageMatchCandidate]) -> bool:
    """Return whether the best candidate passes the acceptance checks."""
    best = candidates[0] if candidates else None
    if not best or best["score"] < MIN_ACCEPTED_CANDIDATE_SCORE:
        return False
    second = candidates[1] if len(candidates) > 1 else None
    if second is None:
        return True
    return not (best["score"] - second["score"] < 0.75 and best["score"] < 9)


def _is_accepted_best_candidate_for_rank(
    artificial_analysis_model: dict[str, Any],
    candidates: list[ImageMatchCandidate],
    artificial_analysis_rank: int | None,
) -> bool:
    """Return whether the ranked best candidate passes the acceptance checks."""
    best = candidates[0] if candidates else None
    if best and not _has_family_anchor_overlap(
        artificial_analysis_model,
        best["arena_model"],
    ):
        return False
    if _is_accepted_best_candidate(candidates):
        return True
    if (
        artificial_analysis_rank is not None
        and artificial_analysis_rank <= TOP_RANK_PROTECTION_COUNT
    ):
        if not best or best["score"] < MIN_ACCEPTED_CANDIDATE_SCORE:
            return False
        second = candidates[1] if len(candidates) > 1 else None
        margin = (
            best["score"] - second["score"]
            if second is not None
            else TOP_RANK_PROTECTION_MARGIN
        )
        if margin >= TOP_RANK_PROTECTION_MARGIN:
            return True
    return False


def _apply_dynamic_void(
    models: list[ImageMatchMappedModel],
) -> tuple[float | None, int]:
    """Apply the dynamic void."""
    scores = sorted(
        best_match["score"]
        for model in models
        if (best_match := model.get("best_match")) is not None
    )
    if not scores:
        return None, 0
    min_score = scores[0]
    max_score = scores[-1]
    threshold = min_score + ((max_score - min_score) * VOID_THRESHOLD_RANGE_RATIO)
    voided = 0
    for row_index, model in enumerate(models):
        best_match = model.get("best_match")
        score = best_match["score"] if best_match is not None else None
        top_candidate = model["candidates"][0] if model["candidates"] else None
        second_candidate = (
            model["candidates"][1] if len(model["candidates"]) > 1 else None
        )
        margin = (
            top_candidate["score"] - second_candidate["score"]
            if top_candidate is not None and second_candidate is not None
            else None
        )
        is_protected_top_rank = (
            row_index < TOP_RANK_PROTECTION_COUNT
            and score is not None
            and score >= threshold - TOP_RANK_PROTECTION_THRESHOLD_DELTA
            and (margin is None or margin >= TOP_RANK_PROTECTION_MARGIN)
        )
        if is_protected_top_rank:
            continue
        if score is not None and score < threshold:
            model["best_match"] = None
            voided += 1
    return round(threshold, 4), voided


def _map_model(
    artificial_analysis_model: dict[str, Any],
    arena_models: list[dict[str, Any]],
    max_candidates: int,
    artificial_analysis_rank: int | None,
) -> ImageMatchMappedModel:
    """Map a source model into the selected Image model matching payload."""
    scored_candidates: list[ImageMatchCandidate] = sorted(
        [
            {
                "arena_model": str(arena_model.get("model") or ""),
                "arena_provider": (
                    arena_model.get("provider")
                    if isinstance(arena_model.get("provider"), str)
                    else None
                ),
                "score": _compute_candidate_score(
                    artificial_analysis_model,
                    arena_model,
                    artificial_analysis_rank,
                    arena_index + 1,
                ),
            }
            for arena_index, arena_model in enumerate(arena_models)
        ],
        key=lambda candidate: candidate["score"],
        reverse=True,
    )
    top_candidates = scored_candidates[:max_candidates]
    best_candidate = (
        top_candidates[0]
        if _is_accepted_best_candidate_for_rank(
            artificial_analysis_model,
            top_candidates,
            artificial_analysis_rank,
        )
        else None
    )
    return {
        "artificial_analysis_slug": (
            artificial_analysis_model.get("slug")
            if isinstance(artificial_analysis_model.get("slug"), str)
            else None
        ),
        "artificial_analysis_name": (
            artificial_analysis_model.get("name")
            if isinstance(artificial_analysis_model.get("name"), str)
            else None
        ),
        "artificial_analysis_provider": _get_model_creator_name(
            artificial_analysis_model,
        ),
        "best_match": best_candidate,
        "candidates": top_candidates,
    }


def get_image_match_model_mapping(
    options: ImageMatchModelMappingOptions | None = None,
) -> ImageMatchModelMappingPayload:
    """Build candidate mappings from Artificial Analysis image rows to Arena rows."""
    options = options or {}
    max_candidates = int(options.get("max_candidates") or DEFAULT_MAX_CANDIDATES)
    artificial_analysis_payload = (
        {
            "fetched_at_epoch_seconds": None,
            "data": options.get("artificial_analysis_models") or [],
        }
        if options.get("artificial_analysis_models") is not None
        else get_artificial_analysis_image_stats()
    )
    arena_payload = (
        {
            "fetched_at_epoch_seconds": None,
            "rows": options.get("arena_models") or [],
        }
        if options.get("arena_models") is not None
        else get_arena_ai_image_stats()
    )

    artificial_analysis_models = artificial_analysis_payload.get("data") or []
    arena_models = arena_payload.get("rows") or []
    models = [
        _map_model(model, arena_models, max_candidates, index + 1)
        for index, model in enumerate(artificial_analysis_models)
        if isinstance(model, dict)
    ]
    void_threshold, voided_count = _apply_dynamic_void(models)
    return {
        "artificial_analysis_fetched_at_epoch_seconds": artificial_analysis_payload.get(
            "fetched_at_epoch_seconds",
        ),
        "arena_ai_fetched_at_epoch_seconds": arena_payload.get(
            "fetched_at_epoch_seconds",
        ),
        "total_artificial_analysis_models": len(artificial_analysis_models),
        "total_arena_ai_models": len(arena_models),
        "max_candidates": max_candidates,
        "void_threshold": void_threshold,
        "voided_count": voided_count,
        "models": models,
    }
