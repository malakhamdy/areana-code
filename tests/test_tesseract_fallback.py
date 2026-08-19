from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from egyptian_id_ocr.ocr.tesseract_js_engine import TesseractJSEngine


ROOT = Path(__file__).resolve().parents[1]
FALLBACK_AVAILABLE = bool(
    shutil.which("node")
    and (ROOT / "node_modules" / "tesseract.js").exists()
    and (ROOT / "node_modules" / "@tesseract.js-data" / "ara").exists()
    and (ROOT / "node_modules" / "@tesseract.js-data" / "eng").exists()
)


@pytest.mark.skipif(not FALLBACK_AVAILABLE, reason="npm fallback dependencies are optional")
def test_offline_fallback_reads_numeric_field_without_temp_image_files():
    image = np.full((140, 900, 3), 255, np.uint8)
    cv2.putText(
        image,
        "30001010100136",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.5,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    engine = TesseractJSEngine()
    try:
        spans = engine.recognize(image)
    finally:
        engine.close()
    assert spans
    assert spans[0].text.replace(" ", "") == "30001010100136"
    assert spans[0].confidence > 0.70
