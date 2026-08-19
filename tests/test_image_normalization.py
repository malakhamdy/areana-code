import numpy as np

from egyptian_id_ocr.image_io import letterbox_image, processing_to_original_points


def test_image_normalization_preserves_aspect_ratio_and_metadata():
    image = np.zeros((4032, 3024, 3), dtype=np.uint8)
    canvas, meta = letterbox_image(image, (1600, 1200))
    assert canvas.shape == (1200, 1600, 3)
    assert meta.scale == min(1600 / 3024, 1200 / 4032)
    assert round(meta.content_width / meta.content_height, 4) == round(3024 / 4032, 4)
    assert meta.padding_x > 0
    assert meta.padding_y == 0


def test_coordinate_round_trip_to_original():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, meta = letterbox_image(image, (1600, 1200))
    processing = np.array(
        [
            [meta.padding_x, meta.padding_y],
            [meta.padding_x + meta.content_width - 1, meta.padding_y + meta.content_height - 1],
        ],
        dtype=np.float32,
    )
    original = processing_to_original_points(processing, meta)
    assert np.allclose(original[0], [0, 0], atol=1)
    assert np.allclose(original[1], [639, 479], atol=1)


def test_arbitrary_aspect_ratios_never_stretch():
    for width, height in [(640, 480), (1280, 720), (1920, 1080), (3024, 4032), (777, 333)]:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        _, meta = letterbox_image(image)
        assert abs(meta.content_width / meta.content_height - width / height) < 0.01
