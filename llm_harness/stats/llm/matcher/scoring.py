"""Candidate scoring helpers for native Python LLM matcher."""

from __future__ import annotations

from ..shared import normalize_model_token
from .tokenize import (
    common_prefix_length,
    first_parsed_number,
    is_numeric_token,
    parse_active_b_token,
    parse_b_scale_token,
    parsed_numeric_tokens,
    split_base_model_id,
    split_base_model_tokens,
    split_tokens,
)

TOKEN_PREFIX_WEIGHTS = (5, 4, 3, 2, 1)
TOKEN_PREFIX_REWARD_MULTIPLIER = 2
NUMERIC_EXACT_MATCH_REWARD = 2
NUMERIC_CLOSENESS_REWARD_SCALE = 0.1
NUMERIC_ALL_EQUAL_REWARD = 0.2
VARIANT_SUFFIX_REWARD = 2
COVERAGE_EXACT_REWARD = 4
COVERAGE_MISSING_BASE_PENALTY = 1
B_SCALE_EXACT_REWARD = 3
B_SCALE_MISMATCH_PENALTY = 4
B_SCALE_MISSING_PENALTY = 2
ACTIVE_B_EXACT_REWARD = 2
ACTIVE_B_MISMATCH_PENALTY = 2
CHAR_PREFIX_REWARD_SCALE = 0.03
LENGTH_GAP_PENALTY_SCALE = 0.005


def _weighted_token_prefix_score(
    left_tokens: list[str], right_tokens: list[str]
) -> int:
    """Score token overlap for LLM model matching score."""
    score = 0
    for idx, (left, right) in enumerate(zip(left_tokens, right_tokens, strict=False)):
        if left != right:
            break
        score += TOKEN_PREFIX_WEIGHTS[idx] if idx < len(TOKEN_PREFIX_WEIGHTS) else 0
    return score


def _numeric_match_reward(source_slug: str, candidate_model_id: str) -> int:
    """Helper for numeric match reward."""
    source_tokens = split_tokens(source_slug)
    candidate_tokens = split_base_model_tokens(candidate_model_id)
    for idx in range(min(len(source_tokens), len(candidate_tokens))):
        source_value = (
            parsed_numeric_tokens([source_tokens[idx]])[0]
            if parsed_numeric_tokens([source_tokens[idx]])
            else None
        )
        candidate_value = (
            parsed_numeric_tokens([candidate_tokens[idx]])[0]
            if parsed_numeric_tokens([candidate_tokens[idx]])
            else None
        )
        if source_value is not None and candidate_value is not None:
            return NUMERIC_EXACT_MATCH_REWARD if source_value == candidate_value else 0
    return 0


def _numeric_closeness_reward(source_slug: str, candidate_model_id: str) -> float:
    """Helper for numeric closeness reward."""
    source_numbers = parsed_numeric_tokens(split_tokens(source_slug))
    candidate_numbers = parsed_numeric_tokens(
        split_base_model_tokens(candidate_model_id)
    )
    for idx in range(max(len(source_numbers), len(candidate_numbers))):
        source_value = source_numbers[idx] if idx < len(source_numbers) else None
        candidate_value = (
            candidate_numbers[idx] if idx < len(candidate_numbers) else None
        )
        if source_value is None or candidate_value is None:
            return 0.0
        if source_value == candidate_value:
            continue
        return NUMERIC_CLOSENESS_REWARD_SCALE / (
            1 + abs(source_value - candidate_value)
        )
    return NUMERIC_ALL_EQUAL_REWARD


def _candidate_scale_value(
    candidate_model_id: str, candidate_model_name: str, parser
) -> int | None:
    """Helper for candidate scale value."""
    return first_parsed_number(
        split_base_model_tokens(candidate_model_id), parser
    ) or first_parsed_number(split_tokens(candidate_model_name), parser)


def _b_scale_reward_or_penalty(
    source_slug: str, candidate_model_id: str, candidate_model_name: str
) -> int:
    """Helper for b scale reward or penalty."""
    source_b_scale = first_parsed_number(split_tokens(source_slug), parse_b_scale_token)
    if source_b_scale is None:
        return 0
    candidate_b_scale = _candidate_scale_value(
        candidate_model_id, candidate_model_name, parse_b_scale_token
    )
    if candidate_b_scale is None:
        return -B_SCALE_MISSING_PENALTY
    if candidate_b_scale == source_b_scale:
        return B_SCALE_EXACT_REWARD
    return -B_SCALE_MISMATCH_PENALTY


def _has_hard_b_scale_mismatch(
    source_slug: str, candidate_model_id: str, candidate_model_name: str
) -> bool:
    """Return whether LLM model matching score has a variant conflict."""
    source_b_scale = first_parsed_number(split_tokens(source_slug), parse_b_scale_token)
    if source_b_scale is None:
        return False
    candidate_b_scale = _candidate_scale_value(
        candidate_model_id, candidate_model_name, parse_b_scale_token
    )
    return candidate_b_scale is not None and candidate_b_scale != source_b_scale


def _active_b_reward_or_penalty(
    source_slug: str, candidate_model_id: str, candidate_model_name: str
) -> int:
    """Helper for active b reward or penalty."""
    source_active_b = first_parsed_number(
        split_tokens(source_slug), parse_active_b_token
    )
    if source_active_b is None:
        return 0
    candidate_active_b = _candidate_scale_value(
        candidate_model_id, candidate_model_name, parse_active_b_token
    )
    if candidate_active_b is None:
        return 0
    if candidate_active_b == source_active_b:
        return ACTIVE_B_EXACT_REWARD
    return -ACTIVE_B_MISMATCH_PENALTY


def _same_variant_reward(
    source_slug: str, candidate_model_id: str, candidate_model_name: str
) -> int:
    """Helper for same variant reward."""
    source_last_token = (
        split_tokens(source_slug)[-1] if split_tokens(source_slug) else None
    )
    if not source_last_token or is_numeric_token(source_last_token):
        return 0
    base_last = (
        split_base_model_tokens(candidate_model_id)[-1]
        if split_base_model_tokens(candidate_model_id)
        else None
    )
    name_last = (
        split_tokens(candidate_model_name)[-1]
        if split_tokens(candidate_model_name)
        else None
    )
    return VARIANT_SUFFIX_REWARD if source_last_token in {base_last, name_last} else 0


def _coverage_reward_or_penalty(
    source_slug: str, candidate_model_id: str, candidate_model_name: str
) -> int:
    """Helper for coverage reward or penalty."""
    source_set = set(split_tokens(source_slug))
    base_set = set(split_base_model_tokens(candidate_model_id))
    name_set = set(split_tokens(candidate_model_name))

    def compare_sets(candidate_set: set[str]) -> int:
        """Compare predicted and expected result sets."""
        if not source_set:
            return 0
        missing_count = len(
            [token for token in source_set if token not in candidate_set]
        )
        if missing_count > 0:
            return -COVERAGE_MISSING_BASE_PENALTY - missing_count
        if len(candidate_set) == len(source_set):
            return COVERAGE_EXACT_REWARD
        return 0

    return max(compare_sets(base_set), compare_sets(name_set))


def has_first_token_match(
    source_slug: str, candidate_model_id: str, candidate_model_name: str
) -> bool:
    """Return whether LLM model matching score has a matching token."""
    source_tokens = split_tokens(source_slug)
    if not source_tokens:
        return False
    first = source_tokens[0]
    base_tokens = split_base_model_tokens(candidate_model_id)
    name_tokens = split_tokens(candidate_model_name)
    return first == (base_tokens[0] if base_tokens else None) or first == (
        name_tokens[0] if name_tokens else None
    )


def score_candidate(
    source_slug: str, candidate_model_id: str, candidate_model_name: str
) -> float:
    """Score the candidate."""
    normalized_source_slug = normalize_model_token(source_slug)
    normalized_model_base = normalize_model_token(
        split_base_model_id(candidate_model_id)
    )
    normalized_model_name = normalize_model_token(candidate_model_name)
    source_tokens = split_tokens(source_slug)
    model_base_tokens = split_base_model_tokens(candidate_model_id)
    model_name_tokens = split_tokens(candidate_model_name)
    base_prefix_length = common_prefix_length(
        normalized_source_slug, normalized_model_base
    )
    name_prefix_length = common_prefix_length(
        normalized_source_slug, normalized_model_name
    )
    max_prefix = max(base_prefix_length, name_prefix_length)
    if _has_hard_b_scale_mismatch(
        source_slug, candidate_model_id, candidate_model_name
    ):
        return 0.0
    weighted_token_score = max(
        _weighted_token_prefix_score(source_tokens, model_base_tokens),
        _weighted_token_prefix_score(source_tokens, model_name_tokens),
    )
    return (
        (weighted_token_score * TOKEN_PREFIX_REWARD_MULTIPLIER)
        + _numeric_match_reward(source_slug, candidate_model_id)
        + _numeric_closeness_reward(source_slug, candidate_model_id)
        + _same_variant_reward(source_slug, candidate_model_id, candidate_model_name)
        + _b_scale_reward_or_penalty(
            source_slug, candidate_model_id, candidate_model_name
        )
        + _active_b_reward_or_penalty(
            source_slug, candidate_model_id, candidate_model_name
        )
        + _coverage_reward_or_penalty(
            source_slug, candidate_model_id, candidate_model_name
        )
        + (max_prefix * CHAR_PREFIX_REWARD_SCALE)
        - (
            abs(len(normalized_source_slug) - len(normalized_model_base))
            * LENGTH_GAP_PENALTY_SCALE
        )
    )


def compare_candidates(candidate: dict) -> tuple[float, str]:
    """Compare the candidates."""
    return (-candidate["score"], candidate["model_id"])
