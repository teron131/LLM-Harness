"""YouTube video summarization using Google Gemini multimodal API.

This module provides a direct interface to Gemini's native video analysis
capabilities, allowing you to pass YouTube URLs directly to the model.
"""

import logging
import os
from typing import Literal

from google import genai
from google.genai import types

from ...clients.usage import track_usage
from .prompts import get_gemini_summary_prompt
from .schemas import Summary

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3-flash-preview"

# Pricing in USD per million tokens
USD_PER_M_TOKENS_BY_MODEL = {
    "gemini-3-flash-preview": {"input": 0.5, "output": 3},
    "gemini-3-pro-preview": {"input": 2, "output": 12},
}


def _calculate_cost(
    model: str,
    prompt_tokens: int,
    total_tokens: int,
) -> float:
    """Calculate estimated cost based on usage."""
    pricing = USD_PER_M_TOKENS_BY_MODEL.get(model)
    if not pricing:
        return 0.0

    output_tokens = max(0, total_tokens - prompt_tokens)
    estimated_usd = (prompt_tokens / 1_000_000) * pricing["input"] + (
        output_tokens / 1_000_000
    ) * pricing["output"]
    return estimated_usd


def analyze_video_url(
    video_url: str,
    *,
    model: str = DEFAULT_MODEL,
    thinking_level: Literal["minimal", "low", "medium", "high"] = "medium",
    target_language: str = "auto",
    api_key: str | None = None,
    timeout: int = 600,
) -> Summary | None:
    """Analyze a YouTube video using Gemini's multimodal API.

    Args:
        video_url: YouTube URL (e.g., "https://youtu.be/MiUHjLxm3V0")
        model: Gemini model to use
        thinking_level: Level of reasoning ("minimal", "low", "medium", "high")
        target_language: Target output language ("auto", "en", "zh-TW", or other language code)
        api_key: Gemini API key (defaults to GOOGLE_API_KEY or GEMINI_API_KEY env var)
        timeout: Request timeout in seconds

    Returns:
        Summary object or None if analysis fails
    """
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY")

    client = genai.Client(api_key=api_key, http_options={"timeout": timeout})

    system_prompt = get_gemini_summary_prompt(
        target_language=target_language,
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part(file_data=types.FileData(file_uri=video_url)),
                types.Part(text=system_prompt),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel(thinking_level)
                ),
                response_mime_type="application/json",
                response_schema=Summary,
            ),
        )

        if not response.text:
            logger.warning("Empty response from Gemini API")
            return None

        # Parse the response into our Pydantic model
        analysis = Summary.model_validate_json(response.text)

        # Log usage and cost information
        usage = getattr(response, "usage_metadata", None)
        if usage:
            logger.info(f"Usage metadata: {usage}")

            prompt_token_count = getattr(usage, "prompt_token_count", None)
            total_token_count = getattr(usage, "total_token_count", None)
            if prompt_token_count is not None and total_token_count is not None:
                cost = _calculate_cost(model, prompt_token_count, total_token_count)
                output_tokens = total_token_count - prompt_token_count
                track_usage(
                    input_tokens=prompt_token_count,
                    output_tokens=output_tokens,
                    cost=cost,
                )

                if cost > 0:
                    logger.info(f"Estimated cost: ${cost:.4f} USD")
                    logger.info(
                        f"Tokens - Input: {prompt_token_count}, Output: {output_tokens}"
                    )

        return analysis

    except Exception as e:
        logger.exception(f"Failed to analyze video: {e}")
        return None


def summarize_video(
    video_url: str,
    *,
    model: str = DEFAULT_MODEL,
    thinking_level: Literal["minimal", "low", "medium", "high"] = "medium",
    target_language: str = "auto",
    api_key: str | None = None,
) -> Summary | None:
    """Convenience function to summarize a YouTube video.

    Args:
        video_url: YouTube URL
        model: Gemini model to use
        thinking_level: Reasoning level
        target_language: Target output language ("auto", "en", "zh-TW")
        api_key: Optional API key override

    Returns:
        Summary with overview and chapters
    """
    return analyze_video_url(
        video_url,
        model=model,
        thinking_level=thinking_level,
        target_language=target_language,
        api_key=api_key,
    )
