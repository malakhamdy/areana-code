"""Typed data contracts shared by every pipeline stage.

The dataclasses deliberately keep raw OCR and normalized values separate. Images are
kept in PipelineArtifacts, not in the JSON result, to avoid accidental persistence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CoordinateSpace(str, Enum):
    ORIGINAL_IMAGE = "original_image"
    PROCESSING_CANVAS = "processing_canvas"
    CANONICAL_CARD = "canonical_card"
    FIELD_CROP = "field_crop"


class DocumentSide(str, Enum):
    FRONT = "front"
    BACK = "back"
    UNKNOWN = "unknown"


class FieldStatus(str, Enum):
    NOT_PRESENT = "NOT_PRESENT"
    NOT_EXTRACTED = "NOT_EXTRACTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    EXTRACTED = "EXTRACTED"
    VALIDATED = "VALIDATED"
    VERIFIED = "VERIFIED"
    CROSS_VALIDATED = "CROSS_VALIDATED"
    CONFLICT = "CONFLICT"


class NIDValidationCode(str, Enum):
    VALID = "VALID"
    INVALID_FORMAT = "INVALID_FORMAT"
    INVALID_DATE = "INVALID_DATE"
    INVALID_GOVERNORATE = "INVALID_GOVERNORATE"
    INVALID_CHECKSUM = "INVALID_CHECKSUM"
    INVALID_STRUCTURE = "INVALID_STRUCTURE"


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    width: int
    height: int
    coordinate_space: CoordinateSpace = CoordinateSpace.CANONICAL_CARD

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def as_xyxy(self) -> list[int]:
        return [self.x, self.y, self.x2, self.y2]

    def normalized(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        return (
            self.x / image_width,
            self.y / image_height,
            self.width / image_width,
            self.height / image_height,
        )


@dataclass
class TransformationMetadata:
    original_width: int
    original_height: int
    processing_width: int
    processing_height: int
    scale: float
    padding_x: int
    padding_y: int
    content_width: int
    content_height: int


@dataclass
class ImageQuality:
    width: int
    height: int
    blur_score: float
    sharpness_score: float
    contrast: float
    brightness: float
    glare_fraction: float
    shadow_fraction: float
    noise_score: float
    overall_score: float
    sufficient: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class CardDetection:
    corners: list[list[float]]
    confidence: float
    method: str
    coordinate_space: CoordinateSpace = CoordinateSpace.PROCESSING_CANVAS
    warnings: list[str] = field(default_factory=list)


@dataclass
class SideClassification:
    side: DocumentSide
    confidence: float
    cues: dict[str, float | int | bool] = field(default_factory=dict)


@dataclass
class LocalizedField:
    field: str
    bbox: BBox
    localization_confidence: float
    method: str
    expected_type: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class OCRSpan:
    text: str
    confidence: float
    bbox: list[list[float]] | None = None
    coordinate_space: CoordinateSpace = CoordinateSpace.FIELD_CROP


@dataclass
class OCRCandidate:
    raw: str
    normalized: str
    confidence: float
    preprocessing_variant: str
    engine: str
    spans: list[OCRSpan] = field(default_factory=list)
    score: float = 0.0
    validation: dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldResult:
    field: str
    raw: str | None
    normalized: str | None
    source: str
    bbox: BBox | None
    ocr_engine: str | None
    preprocessing_variant: str | None
    ocr_confidence: float
    localization_confidence: float
    validation_confidence: float
    cross_source_confidence: float
    final_confidence: float
    status: FieldStatus
    candidates: list[OCRCandidate] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DocumentResult:
    is_egyptian_id: bool
    side: DocumentSide
    side_confidence: float
    card_detection_confidence: float
    template_id: str
    template_confidence: float
    image_quality: ImageQuality
    fields: dict[str, FieldResult]
    derived: dict[str, FieldResult]
    cross_field_validation: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _enum_values(asdict(self))


def _enum_values(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_enum_values(item) for item in value]
    if isinstance(value, tuple):
        return [_enum_values(item) for item in value]
    return value
