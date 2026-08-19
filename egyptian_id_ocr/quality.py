"""Image and crop quality metrics. Confidence is evidence, never validation."""
from __future__ import annotations

import cv2
import numpy as np

from .schemas import ImageQuality


def assess_image_quality(rgb: np.ndarray, *, is_crop: bool = False) -> ImageQuality:
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if max(width, height) > 1200:
        scale = 1200 / max(width, height)
        gray_eval = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        gray_eval = gray

    laplacian = cv2.Laplacian(gray_eval, cv2.CV_64F)
    sharpness = float(laplacian.var())
    blur_score = float(np.clip(sharpness / (180.0 if is_crop else 140.0), 0, 1))
    contrast_raw = float(gray_eval.std())
    contrast_score = float(np.clip(contrast_raw / 55.0, 0, 1))
    brightness = float(gray_eval.mean() / 255.0)

    hsv = cv2.cvtColor(
        cv2.resize(rgb, (gray_eval.shape[1], gray_eval.shape[0])), cv2.COLOR_RGB2HSV
    )
    glare = float(np.mean((gray_eval > 246) & (hsv[:, :, 1] < 38)))
    shadow = float(np.mean(gray_eval < 35))

    denoised = cv2.GaussianBlur(gray_eval, (3, 3), 0)
    noise = float(np.mean(np.abs(gray_eval.astype(np.float32) - denoised)) / 20.0)
    noise = float(np.clip(noise, 0, 1))
    min_dimension = min(width, height)
    resolution_score = float(
        np.clip(min_dimension / (90.0 if is_crop else 480.0), 0, 1)
    )
    exposure_score = float(np.clip(1 - abs(brightness - 0.56) / 0.56, 0, 1))
    glare_score = float(np.clip(1 - glare / 0.16, 0, 1))
    shadow_score = float(np.clip(1 - shadow / 0.28, 0, 1))
    overall = float(
        0.25 * blur_score
        + 0.18 * contrast_score
        + 0.20 * resolution_score
        + 0.14 * exposure_score
        + 0.13 * glare_score
        + 0.10 * shadow_score
    )

    issues: list[str] = []
    if blur_score < 0.30:
        issues.append("The image is blurred; small Arabic dots or digits may be lost.")
    if resolution_score < 0.45:
        issues.append("The card/field resolution is too low for reliable OCR.")
    if brightness < 0.20:
        issues.append("The image is underexposed.")
    elif brightness > 0.91:
        issues.append("The image is overexposed.")
    if glare > 0.12:
        issues.append("Strong glare may hide printed characters.")
    if shadow > 0.24:
        issues.append("Large dark/shadow regions were detected.")
    if contrast_score < 0.22:
        issues.append("Text contrast is very low.")

    threshold = 0.42 if is_crop else 0.48
    # A weighted average must not let good resolution/exposure hide a completely
    # blurred or blank image; these are hard quality gates for OCR evidence.
    sufficient = (
        overall >= threshold
        and blur_score >= 0.18
        and contrast_score >= 0.15
        and resolution_score >= 0.30
    )
    return ImageQuality(
        width=width,
        height=height,
        blur_score=round(blur_score, 4),
        sharpness_score=round(sharpness, 3),
        contrast=round(contrast_score, 4),
        brightness=round(brightness, 4),
        glare_fraction=round(glare, 4),
        shadow_fraction=round(shadow, 4),
        noise_score=round(noise, 4),
        overall_score=round(overall, 4),
        sufficient=sufficient,
        issues=issues,
    )
