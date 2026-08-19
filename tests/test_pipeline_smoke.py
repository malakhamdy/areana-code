from egyptian_id_ocr.ocr.base import UnavailableOCREngine
from egyptian_id_ocr.pipeline import EgyptianIDPipeline
from egyptian_id_ocr.schemas import DocumentSide
from conftest import render_scene


def test_pipeline_returns_geometry_even_if_ocr_is_unavailable(synthetic_front_card):
    scene = render_scene(synthetic_front_card)
    pipeline = EgyptianIDPipeline(UnavailableOCREngine("offline test"))
    output = pipeline.process(scene)
    assert output.result.side == DocumentSide.FRONT
    assert output.result.fields["national_id"].status.value == "LOW_CONFIDENCE"
    assert output.result.fields["national_id"].normalized is None
    assert output.artifacts.canonical_card.shape[:2] == (808, 1280)
    assert "offline test" in output.result.warnings
