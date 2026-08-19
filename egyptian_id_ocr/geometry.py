"""Coordinate-safe geometric helpers and perspective rectification."""
from __future__ import annotations

import cv2
import numpy as np

from .config import CANONICAL_CARD_HEIGHT, CANONICAL_CARD_WIDTH
from .schemas import BBox, CoordinateSpace


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return corners in TL, TR, BR, BL order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]  # usually TL, TR, BR, BL from -pi
    # Rotate so top-left (minimum x+y) is first.
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    # If second corner is below fourth, orientation is reversed.
    if ordered[1, 1] > ordered[3, 1]:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


def quadrilateral_size(corners: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = order_corners(corners)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return float(width), float(height)


def rectify_card(
    processing_rgb: np.ndarray,
    corners: np.ndarray,
    output_size: tuple[int, int] = (CANONICAL_CARD_WIDTH, CANONICAL_CARD_HEIGHT),
) -> tuple[np.ndarray, np.ndarray]:
    src = order_corners(corners)
    width, height = quadrilateral_size(src)
    # Portrait ordering from an uncertain detector is rotated to landscape first.
    if height > width:
        src = src[[3, 0, 1, 2]]
    out_width, out_height = output_size
    dst = np.array(
        [[0, 0], [out_width - 1, 0], [out_width - 1, out_height - 1], [0, out_height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src, dst)
    canonical = cv2.warpPerspective(
        processing_rgb,
        homography,
        (out_width, out_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return canonical, homography


def normalized_bbox_to_pixels(
    box: tuple[float, float, float, float], image_width: int, image_height: int
) -> BBox:
    x, y, width, height = box
    x1 = int(round(np.clip(x, 0, 1) * image_width))
    y1 = int(round(np.clip(y, 0, 1) * image_height))
    x2 = int(round(np.clip(x + width, 0, 1) * image_width))
    y2 = int(round(np.clip(y + height, 0, 1) * image_height))
    return BBox(
        x=x1,
        y=y1,
        width=max(1, x2 - x1),
        height=max(1, y2 - y1),
        coordinate_space=CoordinateSpace.CANONICAL_CARD,
    )


def clip_bbox(box: BBox, image_width: int, image_height: int) -> BBox:
    x1 = int(np.clip(box.x, 0, image_width - 1))
    y1 = int(np.clip(box.y, 0, image_height - 1))
    x2 = int(np.clip(box.x2, x1 + 1, image_width))
    y2 = int(np.clip(box.y2, y1 + 1, image_height))
    return BBox(x1, y1, x2 - x1, y2 - y1, box.coordinate_space)


def bbox_iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.width * a.height + b.width * b.height - intersection
    return float(intersection / union) if union else 0.0
