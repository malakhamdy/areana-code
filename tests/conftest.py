from __future__ import annotations

import cv2
import numpy as np
import pytest

from egyptian_id_ocr.config import CANONICAL_CARD_HEIGHT, CANONICAL_CARD_WIDTH


@pytest.fixture
def synthetic_front_card() -> np.ndarray:
    width, height = CANONICAL_CARD_WIDTH, CANONICAL_CARD_HEIGHT
    card = np.full((height, width, 3), (236, 226, 185), np.uint8)
    cv2.rectangle(card, (4, 4), (width - 5, height - 5), (25, 40, 35), 8)
    # Portrait-like dark region.
    cv2.rectangle(card, (25, 55), (340, 510), (180, 180, 170), -1)
    cv2.circle(card, (180, 200), 80, (70, 70, 70), -1)
    cv2.rectangle(card, (105, 280), (255, 500), (65, 65, 65), -1)
    # Arabic field ink is represented by separate connected strokes; content is synthetic.
    for y, x1, x2 in [(205, 560, 1215), (278, 500, 1190), (405, 610, 1210), (475, 555, 1170)]:
        cv2.rectangle(card, (x1, y), (x2, y + 22), (20, 25, 22), -1)
        for x in range(x1 + 20, x2, 62):
            cv2.circle(card, (x, y - 6), 4, (20, 25, 22), -1)
    # Fourteen synthetic digit components in the NID zone.
    for index in range(14):
        x = 535 + index * 48
        cv2.rectangle(card, (x, 665), (x + 22, 724), (18, 22, 20), 5)
    return cv2.cvtColor(card, cv2.COLOR_BGR2RGB)


@pytest.fixture
def synthetic_back_card() -> np.ndarray:
    width, height = CANONICAL_CARD_WIDTH, CANONICAL_CARD_HEIGHT
    card = np.full((height, width, 3), (232, 224, 190), np.uint8)
    cv2.rectangle(card, (4, 4), (width - 5, height - 5), (30, 35, 32), 8)
    for index, x in enumerate(range(80, 1200, 11)):
        thickness = 3 if index % 3 else 7
        cv2.rectangle(card, (x, 570), (x + thickness, 755), (10, 10, 10), -1)
    return cv2.cvtColor(card, cv2.COLOR_BGR2RGB)


def render_scene(card_rgb: np.ndarray, size=(1500, 1050), perspective=True) -> np.ndarray:
    canvas_width, canvas_height = size
    scene = np.full((canvas_height, canvas_width, 3), (42, 48, 45), np.uint8)
    card_h, card_w = card_rgb.shape[:2]
    fit_scale = min(0.80 * canvas_width / card_w, 0.76 * canvas_height / card_h)
    target_w, target_h = card_w * fit_scale, card_h * fit_scale
    cx, cy = canvas_width / 2, canvas_height / 2
    x1, x2 = cx - target_w / 2, cx + target_w / 2
    y1, y2 = cy - target_h / 2, cy + target_h / 2
    if perspective:
        dx, dy = target_w * 0.035, target_h * 0.045
        dst = np.float32(
            [
                [x1 + dx, y1 + dy],
                [x2, y1],
                [x2 - dx, y2 - dy],
                [x1, y2],
            ]
        )
    else:
        dst = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    h, w = card_rgb.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(card_rgb, matrix, (canvas_width, canvas_height))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (canvas_width, canvas_height))
    scene[mask > 0] = warped[mask > 0]
    return scene
