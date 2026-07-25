"""Multimodal input utilities for OpenAI Chat Completions API.

Creates content blocks compatible with OpenAI-compatible chat endpoints.
"""

import base64
from pathlib import Path
from typing import Any, cast

from langchain.messages import HumanMessage

# Supported file types: extension -> (category, mime_type)
SUPPORTED_EXTENSIONS: dict[str, tuple[str, str]] = {
    # Images
    ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"),
    ".png": ("image", "image/png"),
    ".gif": ("image", "image/gif"),
    ".webp": ("image", "image/webp"),
    # Videos (mapped to image_url for vision models)
    ".mp4": ("video", "video/mp4"),
    ".mpeg": ("video", "video/mpeg"),
    ".mov": ("video", "video/quicktime"),
    ".webm": ("video", "video/webm"),
    # Audio
    ".mp3": ("audio", "audio/mpeg"),
    ".wav": ("audio", "audio/wav"),
    # Documents
    ".pdf": ("file", "application/pdf"),
    # Text
    ".txt": ("text", "text/plain"),
    ".md": ("text", "text/markdown"),
}


def _encode_base64(data: bytes) -> str:
    """Encode bytes to base64 string."""
    return base64.b64encode(data).decode("utf-8")


def _create_text_block(text: str) -> dict[str, Any]:
    """Create text content block for Chat Completions API."""
    return {"type": "text", "text": text}


def _create_image_block(data_url: str) -> dict[str, Any]:
    """Create image content block for Chat Completions API."""
    return {"type": "image_url", "image_url": {"url": data_url}}


def _create_file_block(filename: str, data_url: str) -> dict[str, Any]:
    """Create file content block for Chat Completions API."""
    return {"type": "file", "file": {"filename": filename, "file_data": data_url}}


def _create_audio_block(encoded_data: str, format: str) -> dict[str, Any]:
    """Create audio content block for Chat Completions API."""
    return {
        "type": "input_audio",
        "input_audio": {"data": encoded_data, "format": format},
    }


class MediaMessage(HumanMessage):
    """HumanMessage with media content for Chat Completions API.

    Supports file paths or raw bytes for images. Auto-detects file types from paths.

    Args:
        paths: File path(s) or bytes. Can be a single path/bytes or a list.
        description: Optional text description to append after media.
        label_pages: If True, adds "Page N:" labels before each media item.
        mime_type: MIME type for raw bytes (default: "image/jpeg").

    Example:
        >>> MediaMessage("image.png", "What's in this image?")
        >>> MediaMessage([b"page1", b"page2"], "Describe these pages", label_pages=True)
        >>> MediaMessage(paths=["doc.pdf"], description="Extract text from this PDF")
    """

    def __init__(
        self,
        paths: str | Path | bytes | list[str | Path | bytes] | None = None,
        media: str | Path | bytes | list[str | Path | bytes] | None = None,
        description: str = "",
        label_pages: bool = False,
        mime_type: str = "image/jpeg",
    ):
        # Support both 'paths' and 'media' for backward compatibility
        """Initialize the multimodal chat message payload."""
        media_input = paths if paths is not None else media
        if media_input is None:
            raise ValueError("Either 'paths' or 'media' must be provided")

        # Normalize to list
        items = (
            [media_input]
            if isinstance(media_input, (str, Path, bytes))
            else list(media_input)
        )

        content_blocks: list[dict[str, Any]] = []
        for idx, item in enumerate(items, 1):
            blocks = (
                self._from_bytes(item, mime_type)
                if isinstance(item, bytes)
                else self._from_path(Path(item))
            )

            if label_pages and blocks:
                content_blocks.append(_create_text_block(f"Page {idx}:"))

            content_blocks.extend(blocks)

        if description:
            content_blocks.append(_create_text_block(description))

        super().__init__(content=cast(list[str | dict], content_blocks))

    def _from_bytes(self, data: bytes, mime_type: str) -> list[dict[str, Any]]:
        """Create content blocks from raw bytes (assumes image)."""
        data_url = f"data:{mime_type};base64,{_encode_base64(data)}"
        return [_create_image_block(data_url)]

    def _from_path(self, path: Path) -> list[dict[str, Any]]:
        """Create content blocks from a file path."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
            raise ValueError(f"Unsupported extension: {suffix}. Supported: {supported}")

        category, mime_type = SUPPORTED_EXTENSIONS[suffix]

        if category == "text":
            return [_create_text_block(path.read_text(encoding="utf-8"))]

        # Binary content - read once
        encoded = _encode_base64(path.read_bytes())
        data_url = f"data:{mime_type};base64,{encoded}"

        if category in ("image", "video"):
            return [_create_image_block(data_url)]

        if category == "file":
            return [_create_file_block(path.name, data_url)]

        if category == "audio":
            fmt = "wav" if suffix == ".wav" else "mp3"
            return [_create_audio_block(encoded, fmt)]

        return []
