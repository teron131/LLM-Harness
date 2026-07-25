"""This module uses the Scrape Creators or Supadata API to scrape a YouTube video transcript.

The API result is wrapped by YouTubeScrapperResult object.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
import httpx
from langchain.tools import tool
from pydantic import BaseModel, ConfigDict

from ...utils.youtube_utils import (
    clean_text,
    clean_youtube_url,
    extract_video_id,
    is_youtube_url,
)

load_dotenv()

# API Endpoints
SCRAPECREATORS_ENDPOINT = "https://api.scrapecreators.com/v1/youtube/video/transcript"
SUPADATA_ENDPOINT = "https://api.supadata.ai/v1/transcript"

DEFAULT_TIMEOUT_SEC = 30


def _get_api_key(name: str) -> str | None:
    """Return api key."""
    value = os.getenv(name)
    if not value:
        return None
    value = value.strip()
    return value or None


class Channel(BaseModel):
    """Parsed YouTube channel metadata."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    url: str | None = None
    handle: str | None = None
    title: str | None = None


class TranscriptSegment(BaseModel):
    """Single transcript segment returned by the YouTube scraper."""

    model_config = ConfigDict(extra="ignore")

    text: str | None = None
    startMs: float | None = None
    endMs: float | None = None
    startTimeText: str | None = None


class YouTubeScrapperResult(BaseModel):
    """Result payload returned by the YouTube scraper."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    success: bool | None = None
    credits_remaining: float | None = None
    type: str | None = None
    transcript: list[TranscriptSegment] | None = None
    transcript_only_text: str | None = None
    title: str | None = None
    description: str | None = None
    thumbnail: str | None = None
    url: str | None = None
    id: str | None = None
    viewCountInt: int | None = None
    likeCountInt: int | None = None
    publishDate: str | None = None
    publishDateText: str | None = None
    channel: Channel | None = None
    durationFormatted: str | None = None
    keywords: list[str] | None = None
    videoId: str | None = None
    captionTracks: list[dict[str, Any]] | None = None
    language: str | None = None
    availableLangs: list[str] | None = None

    @property
    def parsed_transcript(self) -> str | None:
        """Return cleaned transcript text or None if unavailable."""
        if self.transcript:
            return clean_text(" ".join(seg.text for seg in self.transcript if seg.text))
        if self.transcript_only_text and self.transcript_only_text.strip():
            return clean_text(self.transcript_only_text)
        return None

    @property
    def has_transcript(self) -> bool:
        """Check if video has a transcript available."""
        return bool(
            self.transcript
            or (self.transcript_only_text and self.transcript_only_text.strip())
        )


def _fetch_scrape_creators(video_url: str) -> YouTubeScrapperResult | None:
    """Fetch transcript from Scrape Creators API."""
    api_key = _get_api_key("SCRAPECREATORS_API_KEY")
    if not api_key:
        return None

    try:
        response = httpx.get(
            SCRAPECREATORS_ENDPOINT,
            headers={"x-api-key": api_key},
            params={"url": video_url},
            timeout=DEFAULT_TIMEOUT_SEC,
        )
    except httpx.RequestError:
        return None

    if response.status_code in {401, 403}:
        return None
    if not response.is_success:
        return None

    try:
        data: dict[str, Any] = response.json()
    except ValueError:
        return None

    try:
        return YouTubeScrapperResult.model_validate(data)
    except Exception:
        return None


def _fetch_supadata(video_url: str) -> YouTubeScrapperResult | None:
    """Fetch transcript from Supadata API."""
    api_key = _get_api_key("SUPADATA_API_KEY")
    if not api_key:
        return None

    try:
        response = httpx.get(
            SUPADATA_ENDPOINT,
            headers={"x-api-key": api_key},
            params={"url": video_url, "lang": "en", "text": "true", "mode": "auto"},
            timeout=DEFAULT_TIMEOUT_SEC,
        )
    except httpx.RequestError:
        return None

    if response.status_code in {401, 403}:
        return None
    if response.status_code == 202:
        # Supadata may return an async jobId for large videos.
        return None
    if not response.is_success:
        return None

    try:
        data: dict[str, Any] = response.json()
    except ValueError:
        return None

    content = data.get("content")
    transcript_only_text: str | None = content if isinstance(content, str) else None
    transcript: list[TranscriptSegment] | None = None
    if transcript_only_text is None and isinstance(content, list):
        transcript = []
        for item in content:
            if not isinstance(item, dict):
                continue
            transcript.append(
                TranscriptSegment(
                    text=item.get("text"),
                    startMs=item.get("offset"),
                    endMs=(item.get("offset", 0) or 0) + (item.get("duration", 0) or 0),
                    startTimeText=None,
                )
            )

    video_id = extract_video_id(video_url)

    return YouTubeScrapperResult(
        url=video_url,
        transcript=transcript,
        transcript_only_text=transcript_only_text,
        videoId=video_id,
        language=data.get("lang"),
        availableLangs=data.get("availableLangs"),
        success=True,
        type="video",
    )


def scrape_youtube(youtube_url: str) -> YouTubeScrapperResult:
    """Scrape a YouTube video and return the transcript.

    Tries Scrape Creators first, then Supadata.

    Args:
        youtube_url: The YouTube video URL to scrape

    Returns:
        YouTubeScrapperResult: Parsed video data including transcript and metadata
    """
    if not is_youtube_url(youtube_url):
        raise ValueError("Invalid YouTube URL")

    youtube_url = clean_youtube_url(youtube_url)

    # Try Scrape Creators first
    result = _fetch_scrape_creators(youtube_url)
    if result and result.has_transcript:
        return result

    # Fallback to Supadata
    result = _fetch_supadata(youtube_url)
    if result and result.has_transcript:
        return result

    if not _get_api_key("SCRAPECREATORS_API_KEY") and not _get_api_key(
        "SUPADATA_API_KEY"
    ):
        raise ValueError("No API keys found for Scrape Creators or Supadata")

    raise ValueError("Failed to fetch transcript from available providers")


def get_transcript(youtube_url: str) -> str:
    """Scrape and return the parsed transcript text, raising an error if unavailable."""
    result = scrape_youtube(youtube_url)
    if not result.has_transcript:
        raise ValueError("Video has no transcript")
    transcript = result.parsed_transcript
    if not transcript:
        raise ValueError("Transcript is empty")
    return transcript


@tool(parse_docstring=True)
def scrape_youtube_tool(youtube_url: str) -> str:
    """Scrape a YouTube video and return the transcript.

    Args:
        youtube_url: The YouTube video URL to scrape.

    Returns:
        Parsed transcript text.
    """
    return get_transcript(youtube_url)
