from __future__ import annotations

import cv2
import numpy as np

from egyptian_id_ocr.card_detector import detect_card
from egyptian_id_ocr.geometry import bbox_iou, rectify_card
from egyptian_id_ocr.image_io import letterbox_image
from egyptian_id_ocr.localization import crop_localized_field, localize_fields
from egyptian_id_ocr.templates import FRONT_V1
from conftest import render_scene


def _run_geometry(scene):
    processing, transform = letterbox_image(scene)
    detection = detect_card(processing, transform)
    canonical, _ = rectify_card(processing, np.asarray(detection.corners, np.float32))
    fields = localize_fields(canonical, FRONT_V1)
    return detection, canonical, fields


def test_card_detection_and_crop_consistency(synthetic_front_card):
    scene = render_scene(synthetic_front_card)
    detection, canonical, fields = _run_geometry(scene)
    assert detection.confidence > 0.55
    assert canonical.shape[:2] == (808, 1280)
    for name in ("name", "address", "national_id"):
        crop = crop_localized_field(canonical, fields[name])
        assert crop.size > 0
        assert crop.shape[1] > 0.30 * canonical.shape[1]
        assert fields[name].bbox.coordinate_space.value == "canonical_card"


def test_localization_scale_invariance(synthetic_front_card):
    """Same logical card at 25/50/100/150/200% retains canonical field boxes."""
    base_scene = render_scene(synthetic_front_card, size=(1200, 850))
    baseline = None
    for scale in (0.25, 0.5, 1.0, 1.5, 2.0):
        resized = cv2.resize(
            base_scene,
            (int(base_scene.shape[1] * scale), int(base_scene.shape[0] * scale)),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        _, _, fields = _run_geometry(resized)
        current = {name: field.bbox for name, field in fields.items()}
        if baseline is None:
            baseline = current
            continue
        for name in ("name", "address", "national_id"):
            assert bbox_iou(baseline[name], current[name]) > 0.72


def test_localization_stable_with_rotation_margins_and_perspective(synthetic_front_card):
    scenes = [
        render_scene(synthetic_front_card, size=(1600, 1000), perspective=False),
        render_scene(synthetic_front_card, size=(1000, 1300), perspective=True),
    ]
    scene = render_scene(synthetic_front_card, size=(1400, 1000), perspective=False)
    matrix = cv2.getRotationMatrix2D((700, 500), 5.0, 0.92)
    scenes.append(cv2.warpAffine(scene, matrix, (1400, 1000), borderValue=(42, 48, 45)))
    boxes = []
    for item in scenes:
        detection, _, fields = _run_geometry(item)
        assert detection.confidence > 0.48
        boxes.append(fields)
    for name in ("name", "address", "national_id"):
        assert min(bbox_iou(boxes[0][name].bbox, item[name].bbox) for item in boxes[1:]) > 0.58
