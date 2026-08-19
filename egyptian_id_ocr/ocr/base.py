"""OCR protocol and non-crashing unavailable engine."""
from __future__ import annotations

from typing import Protocol
import numpy as np

from ..schemas import OCRSpan


class OCREngine(Protocol):
    name: str

    def recognize(self, image_rgb: np.ndarray) -> list[OCRSpan]: ...


class UnavailableOCREngine:
    name = "unavailable"

    def __init__(self, reason: str = "OCR engine is not installed.") -> None:
        self.reason = reason

    def recognize(self, image_rgb: np.ndarray) -> list[OCRSpan]:
        return []
