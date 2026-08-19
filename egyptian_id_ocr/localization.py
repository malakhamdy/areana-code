"""Scale-invariant field localization on the canonical rectified card."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import clip_bbox, normalized_bbox_to_pixels
from .schemas import BBox, CoordinateSpace, LocalizedField
from .templates import CardTemplate, FieldTemplate


def localize_fields(
    canonical_rgb: np.ndarray, template: CardTemplate
) -> dict[str, LocalizedField]:
    height, width = canonical_rgb.shape[:2]
    localized: dict[str, LocalizedField] = {}
    for name, specification in template.fields.items():
        search = normalized_bbox_to_pixels(specification.search_region, width, height)
        if specification.expected_type in {"image", "pdf417"}:
            final = search
            confidence = 0.84 if specification.expected_type == "image" else 0.68
            method = "canonical_normalized_region"
            warnings: list[str] = []
        else:
            final, ink_confidence = _refine_with_ink(
                canonical_rgb, search, specification
            )
            confidence = 0.64 + 0.28 * ink_confidence
            method = "canonical_template+ink_projection"
            warnings = []
            if ink_confidence < 0.18:
                warnings.append(
                    "Little text evidence was found inside the normalized search region."
                )
        localized[name] = LocalizedField(
            field=name,
            bbox=clip_bbox(final, width, height),
            localization_confidence=round(float(np.clip(confidence, 0, 0.98)), 4),
            method=method,
            expected_type=specification.expected_type,
            warnings=warnings,
        )
    return localized


def _refine_with_ink(
    image: np.ndarray, search: BBox, specification: FieldTemplate
) -> tuple[BBox, float]:
    region = image[search.y : search.y2, search.x : search.x2]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    # Black-hat suppresses the gold pyramid background while retaining dark print.
    kernel_width = max(9, int(region.shape[1] * (0.025 if specification.field != "national_id" else 0.012)))
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width | 1, 5)),
    )
    _, mask_a = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    mask = cv2.bitwise_or(mask_a, adaptive)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    selected: list[tuple[int, int, int, int, int]] = []
    region_area = region.shape[0] * region.shape[1]
    min_h = max(5, int(region.shape[0] * (0.08 if specification.field != "national_id" else 0.14)))
    max_h = int(region.shape[0] * 0.72)
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < max(10, region_area * 0.00012):
            continue
        if not (min_h <= height <= max_h):
            continue
        if width > region.shape[1] * 0.48 or width / max(height, 1) > 8:
            continue
        selected.append((x, y, width, height, area))

    if len(selected) < (5 if specification.field == "national_id" else 3):
        return search, min(0.16, len(selected) / 20)

    # Percentiles prevent one scratch or border from expanding the crop.
    xs1 = np.array([item[0] for item in selected])
    ys1 = np.array([item[1] for item in selected])
    xs2 = np.array([item[0] + item[2] for item in selected])
    ys2 = np.array([item[1] + item[3] for item in selected])
    x1 = int(np.percentile(xs1, 3))
    y1 = int(np.percentile(ys1, 3))
    x2 = int(np.percentile(xs2, 97))
    y2 = int(np.percentile(ys2, 97))

    pad_left, pad_top, pad_right, pad_bottom = specification.padding
    card_h, card_w = image.shape[:2]
    x1 = search.x + x1 - int(pad_left * card_w)
    y1 = search.y + y1 - int(pad_top * card_h)
    x2 = search.x + x2 + int(pad_right * card_w)
    y2 = search.y + y2 + int(pad_bottom * card_h)

    # Do not let content refinement wander outside the semantic search zone by much.
    x1 = max(search.x - int(0.01 * card_w), x1)
    y1 = max(search.y - int(0.01 * card_h), y1)
    x2 = min(search.x2 + int(0.01 * card_w), x2)
    y2 = min(search.y2 + int(0.01 * card_h), y2)
    if x2 - x1 < search.width * 0.35 or y2 - y1 < search.height * 0.18:
        return search, 0.15
    box = BBox(x1, y1, x2 - x1, y2 - y1, CoordinateSpace.CANONICAL_CARD)
    density = sum(item[4] for item in selected) / max(1, (x2 - x1) * (y2 - y1))
    component_score = min(1.0, len(selected) / (14 if specification.field == "national_id" else 22))
    confidence = float(np.clip(0.65 * component_score + 0.35 * density / 0.12, 0, 1))
    return box, confidence


def crop_localized_field(canonical_rgb: np.ndarray, field: LocalizedField) -> np.ndarray:
    box = field.bbox
    return canonical_rgb[box.y : box.y2, box.x : box.x2].copy()
