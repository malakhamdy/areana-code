"""Safe image decoding and aspect-ratio-preserving input normalization."""
from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import PROCESSING_CANVAS_HEIGHT, PROCESSING_CANVAS_WIDTH
from .schemas import TransformationMetadata

MAX_IMAGE_PIXELS = 40_000_000


class ImageValidationError(ValueError):
    pass


def decode_image(data: bytes | bytearray | BinaryIO | np.ndarray) -> np.ndarray:
    """Decode to RGB uint8 without modifying or retaining the source image."""
    if isinstance(data, np.ndarray):
        image = data.copy()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ImageValidationError("Expected a grayscale, RGB, or RGBA image.")
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        _validate_dimensions(image)
        return image

    raw = data.read() if hasattr(data, "read") else bytes(data)
    if not raw:
        raise ImageValidationError("The uploaded file is empty.")
    try:
        with Image.open(BytesIO(raw)) as pil:
            pil = ImageOps.exif_transpose(pil)
            if pil.width * pil.height > MAX_IMAGE_PIXELS:
                raise ImageValidationError("Image exceeds the 40 megapixel safety limit.")
            image = np.asarray(pil.convert("RGB"), dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError("The file is not a supported image.") from exc
    _validate_dimensions(image)
    return image


def _validate_dimensions(image: np.ndarray) -> None:
    height, width = image.shape[:2]
    if width < 160 or height < 100:
        raise ImageValidationError(
            f"Image is too small ({width}×{height}); minimum is 160×100."
        )
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageValidationError("Image exceeds the 40 megapixel safety limit.")


def letterbox_image(
    original_rgb: np.ndarray,
    canvas_size: tuple[int, int] = (PROCESSING_CANVAS_WIDTH, PROCESSING_CANVAS_HEIGHT),
    pad_color: tuple[int, int, int] = (32, 38, 35),
) -> tuple[np.ndarray, TransformationMetadata]:
    """Resize once with a single scale and pad. X/Y are never stretched independently."""
    canvas_width, canvas_height = canvas_size
    original_height, original_width = original_rgb.shape[:2]
    scale = min(canvas_width / original_width, canvas_height / original_height)
    content_width = max(1, int(round(original_width * scale)))
    content_height = max(1, int(round(original_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(
        original_rgb, (content_width, content_height), interpolation=interpolation
    )
    padding_x = (canvas_width - content_width) // 2
    padding_y = (canvas_height - content_height) // 2
    canvas = np.full((canvas_height, canvas_width, 3), pad_color, dtype=np.uint8)
    canvas[
        padding_y : padding_y + content_height,
        padding_x : padding_x + content_width,
    ] = resized
    metadata = TransformationMetadata(
        original_width=original_width,
        original_height=original_height,
        processing_width=canvas_width,
        processing_height=canvas_height,
        scale=float(scale),
        padding_x=padding_x,
        padding_y=padding_y,
        content_width=content_width,
        content_height=content_height,
    )
    return canvas, metadata


def processing_to_original_points(
    points: np.ndarray, metadata: TransformationMetadata
) -> np.ndarray:
    result = np.asarray(points, dtype=np.float32).copy()
    result[:, 0] = (result[:, 0] - metadata.padding_x) / metadata.scale
    result[:, 1] = (result[:, 1] - metadata.padding_y) / metadata.scale
    result[:, 0] = np.clip(result[:, 0], 0, metadata.original_width - 1)
    result[:, 1] = np.clip(result[:, 1], 0, metadata.original_height - 1)
    return result
