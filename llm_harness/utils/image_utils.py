"""Image loading helpers."""

import base64
import io

import httpx
from PIL import Image


def _load_image(image_source: str) -> Image.Image:
    """Load an image from a URL or local file."""
    if image_source.startswith(("http://", "https://")):
        with httpx.Client() as client:
            response = client.get(image_source)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
    return Image.open(image_source)


def _resize_image_proportionally(
    image: Image.Image,
    max_size: tuple[int, int],
) -> Image.Image:
    """Resize an image proportionally to fit within max_size dimensions."""
    ratio = min(max_size[0] / image.width, max_size[1] / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _image_to_base64(
    image: Image.Image,
    format: str = "JPEG",
) -> str:
    """Convert an image to a base64 string.

    Args:
        image (Image.Image): PIL Image to convert
        format (str): Image format to use (default: PNG)

    Returns:
        str: Base64-encoded image string
    """
    with io.BytesIO() as image_buffer:
        image.save(image_buffer, format=format)
        image_bytes = image_buffer.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")


def load_image_base64(
    image_source: str,
    max_size: tuple[int, int] = (768, 768),
    format: str = "JPEG",
) -> str:
    """Load and resize an image from a URL, local file, or PIL Image and return the result as a base64 string.

    Args:
        image_source (str or Image.Image): URL, local file path, or PIL Image of the image to resize.
        max_size (tuple): Desired maximum size of the image as (width, height).

    Returns:
        str: Base64 string of the resized image.
    """
    image = _load_image(image_source)
    resized_image = _resize_image_proportionally(image, max_size)
    return _image_to_base64(resized_image, format)
