"""Lightweight web page loading helpers using simple HTTP fetches."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from html import unescape
import os
import re

import httpx
from langchain.tools import tool

DEFAULT_TIMEOUT_SEC = 20.0
MAX_OUTPUT_CHARS = 12_000
REQUEST_HEADERS = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36")}


def _clean_text(text: str) -> str:
    """Collapse noisy whitespace and trim empty lines."""
    cleaned_lines: list[str] = []
    previous_line = ""
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if len(line) < 2 or line == previous_line:
            continue
        cleaned_lines.append(line)
        previous_line = line
    return "\n".join(cleaned_lines)


def _extract_html_text(html: str, url: str) -> str | None:
    """Strip obvious boilerplate tags and return readable page text."""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = unescape(title_match.group(1)).strip() if title_match else ""

    cleaned_html = re.sub(r"(?is)<(script|style|noscript|svg|iframe).*?>.*?</\1>", " ", html)
    cleaned_html = re.sub(r"(?i)<br\s*/?>", "\n", cleaned_html)
    cleaned_html = re.sub(r"(?i)</(p|div|section|article|main|li|h[1-6])>", "\n", cleaned_html)
    text = unescape(re.sub(r"(?s)<[^>]+>", " ", cleaned_html))
    text = _clean_text(text)
    if not text:
        return None
    if title and title.lower() not in text[:200].lower():
        text = f"{title}\n\n{text}"
    return f"URL: {url}\n{text[:MAX_OUTPUT_CHARS]}"


def _fetch_url(url: str) -> str | None:
    """Fetch one URL and return a simple text representation."""
    try:
        response = httpx.get(
            url,
            headers=REQUEST_HEADERS,
            follow_redirects=True,
            timeout=DEFAULT_TIMEOUT_SEC,
        )
        response.raise_for_status()
    except Exception:
        return None

    content_type = response.headers.get("content-type", "").lower()
    if "html" in content_type or not content_type:
        return _extract_html_text(response.text, url)
    if content_type.startswith("text/"):
        text = _clean_text(response.text)
        return f"URL: {url}\n{text[:MAX_OUTPUT_CHARS]}" if text else None
    return None


def webloader(urls: str | list[str]) -> list[str | None]:
    """Fetch web pages and return simple text extracts."""
    url_list = [urls] if isinstance(urls, str) else urls
    if not url_list:
        return []

    max_workers = max(1, min(len(url_list), os.cpu_count() or 1, 10))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_fetch_url, url_list))


@tool(parse_docstring=True)
def webloader_tool(urls: list[str]) -> list[str]:
    """Load simple web content extracts for the given URLs.

    Args:
        urls: One or more URLs to fetch and convert into plain text.
    """
    return [content for content in webloader(urls) if content is not None]
