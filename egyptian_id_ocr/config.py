"""Configuration constants for canonical Egyptian ID processing."""
from __future__ import annotations

from dataclasses import dataclass
import os

# ISO/IEC 7810 ID-1 ratio is 85.60 / 53.98 = 1.5858. Egyptian IDs use this form.
CANONICAL_CARD_WIDTH = 1280
CANONICAL_CARD_HEIGHT = 808
CANONICAL_ASPECT_RATIO = CANONICAL_CARD_WIDTH / CANONICAL_CARD_HEIGHT
PROCESSING_CANVAS_WIDTH = 1600
PROCESSING_CANVAS_HEIGHT = 1200


@dataclass(frozen=True)
class OCRConfig:
    """Paddle model settings. Models are downloaded by Paddle only on first use."""

    recognition_model: str = "arabic_PP-OCRv5_mobile_rec"
    detection_model: str = "PP-OCRv5_server_det"
    device: str = "auto"
    score_threshold: float = 0.15
    max_variants_per_field: int = 3

    @classmethod
    def from_env(cls) -> "OCRConfig":
        return cls(
            recognition_model=os.getenv(
                "EGYID_RECOGNITION_MODEL", "arabic_PP-OCRv5_mobile_rec"
            ),
            detection_model=os.getenv(
                "EGYID_DETECTION_MODEL", "PP-OCRv5_server_det"
            ),
            device=os.getenv("EGYID_DEVICE", "auto"),
            max_variants_per_field=int(os.getenv("EGYID_MAX_OCR_VARIANTS", "3")),
        )


FIELD_COLORS = {
    "national_id": (34, 197, 94),
    "name": (247, 144, 9),
    "address": (59, 130, 246),
    "barcode": (168, 85, 247),
    "portrait": (236, 72, 153),
}
