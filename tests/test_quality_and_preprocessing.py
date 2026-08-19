import cv2
import numpy as np

from egyptian_id_ocr.preprocessing import (
    preprocess_address,
    preprocess_name,
    preprocess_nid,
)
from egyptian_id_ocr.quality import assess_image_quality


def test_low_quality_gate_reports_blur_and_low_contrast():
    image = np.full((480, 800, 3), 128, np.uint8)
    image = cv2.GaussianBlur(image, (41, 41), 15)
    quality = assess_image_quality(image)
    assert quality.sufficient is False
    assert quality.blur_score < 0.30
    assert any("blurred" in issue for issue in quality.issues)


def test_field_specific_preprocessing_keeps_independent_variants():
    crop = np.full((90, 500, 3), 220, np.uint8)
    cv2.putText(crop, "12345678901234", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)
    nid = preprocess_nid(crop)
    name = preprocess_name(crop)
    address = preprocess_address(crop)
    assert "nid_otsu" in nid
    assert "name_mild_sharpen" in name
    assert "address_illumination" in address
    assert set(nid).isdisjoint(name)
    assert set(name).isdisjoint(address)
    for variants in (nid, name, address):
        assert all(image.ndim == 3 and image.shape[2] == 3 for image in variants.values())
