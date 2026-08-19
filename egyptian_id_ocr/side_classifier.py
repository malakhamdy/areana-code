"""Front/back classification from canonical visual evidence, never filenames/order."""
from __future__ import annotations

import cv2
import numpy as np

from .schemas import DocumentSide, SideClassification


def classify_side(canonical_rgb: np.ndarray) -> SideClassification:
    height, width = canonical_rgb.shape[:2]
    gray = cv2.cvtColor(canonical_rgb, cv2.COLOR_RGB2GRAY)
    left = gray[int(0.06 * height) : int(0.68 * height), : int(0.34 * width)]
    bottom = gray[int(0.66 * height) : int(0.97 * height), :]

    faces = 0
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if not detector.empty():
            detected = detector.detectMultiScale(
                left, scaleFactor=1.08, minNeighbors=3, minSize=(45, 45)
            )
            faces = len(detected)
    except Exception:
        faces = 0

    portrait_dark = float(np.mean(left < 105))
    portrait_variance = float(np.std(left) / 70.0)
    sx = cv2.Sobel(bottom, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(bottom, cv2.CV_32F, 0, 1, ksize=3)
    vertical_edges = float(np.mean(np.abs(sx) > 110))
    horizontal_edges = float(np.mean(np.abs(sy) > 110))
    barcode_cue = float(np.clip(vertical_edges * 5.5 - horizontal_edges * 1.2, 0, 1))

    face_cue = 1.0 if faces else 0.0
    portrait_cue = float(
        np.clip(0.55 * portrait_dark / 0.22 + 0.45 * portrait_variance, 0, 1)
    )
    front_score = 0.62 * face_cue + 0.38 * portrait_cue
    back_score = 0.78 * barcode_cue + 0.22 * (1 - portrait_cue)

    # Portrait evidence is more reliable than a barcode-like digit line.
    if faces or front_score >= back_score + 0.08:
        side = DocumentSide.FRONT
        confidence = 0.55 + 0.42 * max(front_score - 0.25 * back_score, 0)
    elif back_score >= 0.36:
        side = DocumentSide.BACK
        confidence = 0.52 + 0.40 * back_score
    else:
        side = DocumentSide.UNKNOWN
        confidence = 0.35
    return SideClassification(
        side=side,
        confidence=round(float(np.clip(confidence, 0, 0.98)), 4),
        cues={
            "face_count": faces,
            "portrait_dark_fraction": round(portrait_dark, 4),
            "portrait_cue": round(portrait_cue, 4),
            "barcode_cue": round(barcode_cue, 4),
            "front_score": round(front_score, 4),
            "back_score": round(back_score, 4),
        },
    )
