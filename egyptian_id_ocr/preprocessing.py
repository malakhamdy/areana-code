"""Field-specific, Arabic-preserving preprocessing variants."""
from __future__ import annotations

import cv2
import numpy as np


def preprocess_nid(crop_rgb: np.ndarray) -> dict[str, np.ndarray]:
    base = _upscale(crop_rgb, minimum_height=150, maximum_scale=3.0)
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    corrected = _illumination_correct(gray)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 4)).apply(corrected)
    otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 12
    )
    return {
        "nid_rgb_minimal": base,
        "nid_clahe": cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB),
        "nid_otsu": cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB),
        "nid_adaptive": cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB),
    }


def preprocess_name(crop_rgb: np.ndarray) -> dict[str, np.ndarray]:
    base = _upscale(crop_rgb, minimum_height=220, maximum_scale=2.7)
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    denoised = cv2.bilateralFilter(gray, 5, 28, 28)
    mild_clahe = cv2.createCLAHE(clipLimit=1.65, tileGridSize=(8, 6)).apply(denoised)
    # Unsharp mask is deliberately mild to preserve dots and connected strokes.
    blurred = cv2.GaussianBlur(mild_clahe, (0, 0), 1.0)
    sharpened = cv2.addWeighted(mild_clahe, 1.28, blurred, -0.28, 0)
    return {
        "name_rgb_minimal": base,
        "name_mild_clahe": cv2.cvtColor(mild_clahe, cv2.COLOR_GRAY2RGB),
        "name_mild_sharpen": cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB),
    }


def preprocess_address(crop_rgb: np.ndarray) -> dict[str, np.ndarray]:
    base = _upscale(crop_rgb, minimum_height=260, maximum_scale=2.8)
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    illumination = _illumination_correct(gray)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(10, 6)).apply(illumination)
    denoised = cv2.bilateralFilter(clahe, 5, 32, 32)
    return {
        "address_rgb_minimal": base,
        "address_illumination": cv2.cvtColor(illumination, cv2.COLOR_GRAY2RGB),
        "address_clahe_denoise": cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB),
    }


def preprocess_dob(crop_rgb: np.ndarray) -> dict[str, np.ndarray]:
    return preprocess_nid(crop_rgb)


def preprocess_gender(crop_rgb: np.ndarray) -> dict[str, np.ndarray]:
    return preprocess_name(crop_rgb)


def preprocess_barcode(crop_rgb: np.ndarray) -> dict[str, np.ndarray]:
    base = _upscale(crop_rgb, minimum_height=300, maximum_scale=2.5)
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    return {
        "barcode_rgb": base,
        "barcode_gray": cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        "barcode_contrast": cv2.cvtColor(
            cv2.createCLAHE(2.0, (8, 8)).apply(gray), cv2.COLOR_GRAY2RGB
        ),
    }


def preprocess_field(field: str, crop_rgb: np.ndarray) -> dict[str, np.ndarray]:
    mapping = {
        "national_id": preprocess_nid,
        "name": preprocess_name,
        "address": preprocess_address,
        "date_of_birth": preprocess_dob,
        "gender": preprocess_gender,
        "barcode": preprocess_barcode,
    }
    return mapping.get(field, preprocess_name)(crop_rgb)


def _upscale(
    rgb: np.ndarray, minimum_height: int, maximum_scale: float = 3.0
) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(maximum_scale, max(1.0, minimum_height / max(1, height)))
    if scale <= 1.01:
        return rgb.copy()
    return cv2.resize(
        rgb,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


def _illumination_correct(gray: np.ndarray) -> np.ndarray:
    sigma = max(7, int(min(gray.shape[:2]) * 0.12))
    if sigma % 2 == 0:
        sigma += 1
    background = cv2.GaussianBlur(gray, (sigma, sigma), 0)
    corrected = cv2.divide(gray, np.maximum(background, 1), scale=210)
    return np.clip(corrected, 0, 255).astype(np.uint8)
