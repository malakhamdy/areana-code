"""Independent barcode/PDF417 detection and decoding (not OCR)."""
from __future__ import annotations

from typing import Any
import numpy as np


def decode_barcode(variants: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "barcode_not_decoded",
        "detected": False,
        "decoded": False,
        "format": None,
        "text": None,
        "attempts": [],
    }
    try:
        import zxingcpp
    except ImportError:
        result["status"] = "decoder_unavailable"
        result["warning"] = "Install zxing-cpp to enable local PDF417 decoding."
        return result

    for name, image in variants.items():
        try:
            barcodes = zxingcpp.read_barcodes(image)
        except Exception as exc:
            result["attempts"].append({"variant": name, "error": type(exc).__name__})
            continue
        result["attempts"].append({"variant": name, "count": len(barcodes)})
        if not barcodes:
            continue
        result["detected"] = True
        barcode = barcodes[0]
        text = getattr(barcode, "text", "") or ""
        fmt = str(getattr(barcode, "format", "unknown"))
        result.update(
            {
                "status": "barcode_decoded" if text else "barcode_detected",
                "decoded": bool(text),
                "format": fmt,
                "text": text or None,
                "variant": name,
            }
        )
        if text:
            break
    if not result["detected"]:
        result["status"] = "barcode_not_detected"
    return result
