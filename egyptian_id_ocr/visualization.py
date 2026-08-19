"""Visual debugging overlays for every geometry/localization stage."""
from __future__ import annotations

import cv2
import numpy as np

from .config import FIELD_COLORS
from .schemas import CardDetection, LocalizedField


def draw_card_detection(
    processing_rgb: np.ndarray, detection: CardDetection
) -> np.ndarray:
    output = processing_rgb.copy()
    corners = np.asarray(detection.corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(output, [corners], True, (22, 220, 130), 6, cv2.LINE_AA)
    for index, point in enumerate(corners.reshape(-1, 2)):
        cv2.circle(output, tuple(point), 11, (255, 76, 76), -1, cv2.LINE_AA)
        cv2.putText(
            output,
            str(index + 1),
            tuple(point + np.array([12, -12])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def draw_field_boxes(
    canonical_rgb: np.ndarray, fields: dict[str, LocalizedField]
) -> np.ndarray:
    output = canonical_rgb.copy()
    for name, field in fields.items():
        box = field.bbox
        color = FIELD_COLORS.get(name, (250, 204, 21))
        cv2.rectangle(output, (box.x, box.y), (box.x2, box.y2), color, 4)
        label_y = max(28, box.y - 8)
        cv2.rectangle(
            output,
            (box.x, label_y - 26),
            (box.x + max(120, 14 * len(name)), label_y + 3),
            color,
            -1,
        )
        cv2.putText(
            output,
            name,
            (box.x + 5, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (15, 25, 21),
            2,
            cv2.LINE_AA,
        )
    return output


def draw_ocr_spans(image_rgb: np.ndarray, spans: list) -> np.ndarray:
    output = image_rgb.copy()
    scale_x = 1.0
    scale_y = 1.0
    for span in spans:
        if not span.bbox:
            continue
        points = np.asarray(span.bbox, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [points], True, (34, 197, 94), 2, cv2.LINE_AA)
    return output
