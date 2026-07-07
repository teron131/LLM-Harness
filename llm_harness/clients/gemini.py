"""Google Gemini utilities (caching, chat client, embeddings)."""

import logging
import os
from pathlib import Path
import time

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash-preview-09-2025"


def _resolve_gemini_api_key(
    *,
    error_message: str = "GEMINI_API_KEY (or GOOGLE_API_KEY) must be set",
) -> str:
    """Resolve the Gemini API key from the environment."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(error_message)
    os.environ.setdefault("GOOGLE_API_KEY", api_key)
    return api_key


def ChatGemini(
    model: str,
    *,
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    """Initialize a Gemini chat model.

    Args:
        model: Gemini model name (e.g., "gemini-2.5-flash")
        temperature: Sampling temperature (0.0-2.0)
        **kwargs: Additional arguments passed to ChatGoogleGenerativeAI
    """
    _resolve_gemini_api_key()
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, **kwargs)


def GeminiEmbeddings(
    model: str = "models/text-embedding-004",
    **kwargs,
) -> Embeddings:
    """Initialize Gemini embeddings model.

    Args:
        model: Gemini embedding model name
        **kwargs: Additional arguments passed to GoogleGenerativeAIEmbeddings
    """
    _resolve_gemini_api_key()
    if not model.startswith("models/"):
        model = f"models/{model}"
    return GoogleGenerativeAIEmbeddings(model=model, **kwargs)


def create_gemini_cache(
    file_path: str | Path,
    model: str = DEFAULT_MODEL,
    *,
    api_key: str | None = None,
) -> str:
    """Uploads a file and returns the cache name."""
    api_key = api_key or _resolve_gemini_api_key(error_message="API key not found")

    client = genai.Client(api_key=api_key)
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Uploading {path.name}...")
    file_ref = client.files.upload(file=str(path))

    if file_ref.state is None or file_ref.name is None:
        raise RuntimeError("Upload did not return a usable file reference")

    while file_ref.state.name == "PROCESSING":
        time.sleep(1)
        file_ref = client.files.get(name=file_ref.name)
        if file_ref.state is None or file_ref.name is None:
            raise RuntimeError("Uploaded file status did not include state or name")

    if file_ref.state.name == "FAILED":
        raise RuntimeError(f"Upload failed: {file_ref.state.name}")

    logger.info("Creating cache...")
    cache = client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(contents=[file_ref]),
    )
    if cache.name is None:
        raise RuntimeError("Cache creation did not return a cache name")
    return cache.name
