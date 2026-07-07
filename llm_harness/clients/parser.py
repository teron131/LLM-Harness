"""Response parsing utilities with metadata extraction."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast, overload

from langchain.messages import AIMessage
from pydantic import BaseModel, ConfigDict


class StructuredOutput(BaseModel):
    """Parse the raw message from the structured output response.

    >>> StructuredOutput = ChatOpenAI.with_structured_output(Schema, include_raw=True)
    """

    model_config = ConfigDict(extra="ignore")
    raw: AIMessage
    parsed: BaseModel


def _to_int(value: object) -> int:
    """Coerce provider metadata values to token counts."""
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, str) and not value.strip():
        return 0
    if not isinstance(value, str | int | float):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value: object) -> float:
    """Coerce provider metadata values to costs."""
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, str) and not value.strip():
        return 0.0
    if not isinstance(value, str | int | float):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_metadata(ai_message: AIMessage) -> tuple[int, int, float]:
    """Return (input_tokens, output_tokens, cost) from an AIMessage.

    - Prefer `AIMessage.usage_metadata` (LangChain standard; works for Gemini).
    - Fall back to `response_metadata["token_usage"]` (OpenAI-compatible providers; includes cost here).
    - Fall back to legacy Gemini shapes under `response_metadata["usage_metadata"]`.

    Cost may be unavailable for some providers; returns 0.0.
    """
    usage_metadata = getattr(ai_message, "usage_metadata", None)
    if isinstance(usage_metadata, dict) and usage_metadata:
        return (
            _to_int(usage_metadata.get("input_tokens")),
            _to_int(usage_metadata.get("output_tokens")),
            0.0,
        )

    response_metadata = getattr(ai_message, "response_metadata", None)
    if not isinstance(response_metadata, dict) or not response_metadata:
        return 0, 0, 0.0

    token_usage = response_metadata.get("token_usage")
    if isinstance(token_usage, dict) and token_usage:
        return (
            _to_int(token_usage.get("prompt_tokens")),
            _to_int(token_usage.get("completion_tokens")),
            _to_float(token_usage.get("cost")),
        )

    legacy_usage = response_metadata.get("usage_metadata")
    if isinstance(legacy_usage, dict) and legacy_usage:
        input_tokens = _to_int(legacy_usage.get("prompt_token_count") or legacy_usage.get("input_token_count"))
        output_tokens = _to_int(legacy_usage.get("candidates_token_count") or legacy_usage.get("output_token_count"))
        return input_tokens, output_tokens, 0.0

    return 0, 0, 0.0


def _extract_reasoning(content_blocks: list[Any]) -> str | None:
    """Extract reasoning from response content_blocks."""
    if not content_blocks:
        return None

    first_block = content_blocks[0]
    if not isinstance(first_block, dict):
        return None
    if reasoning := first_block.get("reasoning"):
        return str(reasoning)

    # Check for nested content structure (e.g. from some providers)
    if (
        (extras := first_block.get("extras"))
        and isinstance(extras, dict)
        and (nested_content := extras.get("content"))
        and isinstance(nested_content, list)
        and nested_content
        and isinstance(nested_content[-1], dict)
    ):
        text = nested_content[-1].get("text")
        return str(text) if text is not None else None

    return None


def _extract_text(content_blocks: list[Any]) -> str:
    """Return the final text block from a LangChain message."""
    if not content_blocks:
        return ""
    last_block = content_blocks[-1]
    if isinstance(last_block, dict):
        text = last_block.get("text")
        return str(text) if text is not None else ""
    return ""


def parse_invoke(
    response: AIMessage,
    include_reasoning: bool = False,
) -> str | tuple[str | None, str]:
    """Parse response to extract answer and optionally reasoning."""
    answer = _extract_text(response.content_blocks)
    if include_reasoning:
        reasoning = _extract_reasoning(response.content_blocks)
        return reasoning, answer
    return answer


@overload
def parse_batch(responses: list[AIMessage], include_reasoning: bool = False) -> list[str]: ...


@overload
def parse_batch(responses: list[AIMessage], include_reasoning: bool) -> list[str] | list[tuple[str | None, str]]: ...


def parse_batch(
    responses: list[AIMessage],
    include_reasoning: bool = False,
) -> list[str] | list[tuple[str | None, str]]:
    """Parse batched responses, optionally with reasoning."""
    if include_reasoning:
        return cast(
            list[tuple[str | None, str]],
            [parse_invoke(response, include_reasoning=True) for response in responses],
        )
    return cast(
        list[str],
        [parse_invoke(response, include_reasoning=False) for response in responses],
    )


def get_stream_generator(
    stream: Generator[AIMessage],
    include_reasoning: bool = False,
) -> Generator[str | tuple[str | None, str | None]]:
    """Yield streaming chunks, optionally with reasoning."""
    reasoning_yielded = False

    for chunk in stream:
        if not (blocks := getattr(chunk, "content_blocks", None)):
            continue

        if include_reasoning and not reasoning_yielded and (reasoning := _extract_reasoning(blocks)):
            reasoning_yielded = True
            yield (reasoning, None)

        if answer := _extract_text(blocks):
            yield (None, answer) if include_reasoning else answer


def parse_stream(
    stream: Generator[AIMessage],
    include_reasoning: bool = False,
) -> str | tuple[str | None, str]:
    """Print streamed chunks and return the final result."""
    reasoning = None
    answer_parts: list[str] = []

    for item in get_stream_generator(stream, include_reasoning):
        if isinstance(item, tuple):
            reasoning_chunk, answer_chunk = item
            if reasoning_chunk is not None:
                reasoning = reasoning_chunk
                print(f"Reasoning: {reasoning}", flush=True)
            if answer_chunk is not None:
                answer_parts.append(answer_chunk)
                print(answer_chunk, end="", flush=True)
        else:
            answer_parts.append(item)
            print(item, end="", flush=True)

    answer = "".join(answer_parts)
    return (reasoning, answer) if include_reasoning else answer
