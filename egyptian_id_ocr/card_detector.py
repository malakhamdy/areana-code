"""Multi-strategy physical card detector operating in processing-canvas coordinates."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .config import CANONICAL_ASPECT_RATIO
from .geometry import order_corners, quadrilateral_size
from .schemas import CardDetection, TransformationMetadata


@dataclass
class _Candidate:
    corners: np.ndarray
    score: float
    method: str


def detect_card(
    processing_rgb: np.ndarray, transform: TransformationMetadata
) -> CardDetection:
    """Find an ID-1-like quadrilateral without assuming card scale or margins."""
    px, py = transform.padding_x, transform.padding_y
    cw, ch = transform.content_width, transform.content_height
    content = processing_rgb[py : py + ch, px : px + cw]
    gray = cv2.cvtColor(content, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    med = float(np.median(blurred))
    lower = int(max(20, 0.55 * med))
    upper = int(min(245, max(lower + 30, 1.35 * med)))
    canny = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edges = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=2)

    candidates: list[_Candidate] = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    content_area = float(cw * ch)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        area = float(cv2.contourArea(contour))
        if area < 0.06 * content_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        for epsilon in (0.018, 0.028, 0.045):
            approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = approx.reshape(4, 2).astype(np.float32)
                score = _score_candidate(corners, area, content_area)
                candidates.append(_Candidate(corners, score, "edge_quadrilateral"))
                break
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        rect_area = max(1.0, float(rect[1][0] * rect[1][1]))
        rectangularity = area / rect_area
        if rectangularity > 0.72:
            score = _score_candidate(box, area, content_area) * 0.91
            candidates.append(_Candidate(box, score, "min_area_rectangle"))

    # Color/brightness segmentation helps when the card edge itself is weak.
    hsv = cv2.cvtColor(content, cv2.COLOR_RGB2HSV)
    value_mask = cv2.inRange(hsv[:, :, 2], 90, 255)
    value_mask = cv2.morphologyEx(value_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    bright_contours, _ = cv2.findContours(
        value_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in sorted(bright_contours, key=cv2.contourArea, reverse=True)[:10]:
        area = float(cv2.contourArea(contour))
        if area < 0.08 * content_area:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        score = _score_candidate(box, area, content_area) * 0.93
        candidates.append(_Candidate(box, score, "light_card_region"))

    # A tightly photographed card legitimately coincides with the image boundary.
    content_box = np.array(
        [[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float32
    )
    ratio = max(cw, ch) / max(1, min(cw, ch))
    ratio_score = math.exp(-abs(math.log(max(0.01, ratio / CANONICAL_ASPECT_RATIO))) * 3.5)
    boundary_score = 0.50 + 0.32 * ratio_score
    candidates.append(_Candidate(content_box, boundary_score, "image_boundary_fallback"))

    best = max(candidates, key=lambda candidate: candidate.score)
    corners = order_corners(best.corners)
    corners[:, 0] += px
    corners[:, 1] += py
    confidence = float(np.clip(best.score, 0.05, 0.99))
    warnings: list[str] = []
    if best.method == "image_boundary_fallback":
        warnings.append(
            "No stronger physical edge was found; rectification uses the image content boundary."
        )
    if confidence < 0.58:
        warnings.append("Card localization is uncertain; verify the corner overlay.")
    return CardDetection(
        corners=corners.tolist(),
        confidence=round(confidence, 4),
        method=best.method,
        warnings=warnings,
    )


def _score_candidate(corners: np.ndarray, contour_area: float, content_area: float) -> float:
    ordered = order_corners(corners)
    width, height = quadrilateral_size(ordered)
    if min(width, height) < 40:
        return 0.0
    ratio = max(width, height) / min(width, height)
    ratio_score = math.exp(-abs(math.log(ratio / CANONICAL_ASPECT_RATIO)) * 3.2)
    polygon_area = abs(float(cv2.contourArea(ordered)))
    area_fraction = np.clip(polygon_area / content_area, 0, 1)
    rectangularity = np.clip(contour_area / max(1.0, polygon_area), 0, 1)

    # Opposite sides should be similar even under moderate perspective.
    tl, tr, br, bl = ordered
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    opposite_similarity = min(top, bottom) / max(top, bottom, 1) * min(left, right) / max(
        left, right, 1
    )
    score = (
        0.40 * ratio_score
        + 0.31 * math.sqrt(float(area_fraction))
        + 0.16 * float(rectangularity)
        + 0.13 * float(opposite_similarity)
    )
    return float(np.clip(score, 0, 1))
