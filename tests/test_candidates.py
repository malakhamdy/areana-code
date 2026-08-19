import numpy as np

from egyptian_id_ocr.ocr.candidates import generate_candidates, rank_candidates
from egyptian_id_ocr.schemas import OCRSpan


class SequenceEngine:
    name = "test-arabic-engine"

    def __init__(self, texts):
        self.texts = iter(texts)

    def recognize(self, image):
        return [OCRSpan(next(self.texts), 0.91)]


def test_two_pass_nid_agreement_and_arabic_digits():
    crop = np.full((80, 600, 3), 255, np.uint8)
    engine = SequenceEngine(["٣٠٠٠١٠١٠١٠٠١٣٦"] * 3)
    candidates, _ = generate_candidates("national_id", crop, engine, max_variants=3)
    decision = rank_candidates("national_id", candidates)
    assert decision.selected is not None
    assert decision.selected.normalized == "30001010100136"
    assert decision.selected.validation["is_valid"] is True
    assert decision.ambiguous is False


def test_disagreeing_invalid_nids_are_not_forced():
    crop = np.full((80, 600, 3), 255, np.uint8)
    engine = SequenceEngine(
        ["30001010100131", "30001010100132", "30001010100133"]
    )
    candidates, _ = generate_candidates("national_id", crop, engine, max_variants=3)
    decision = rank_candidates("national_id", candidates)
    assert decision.selected is None
    assert decision.ambiguous is True
