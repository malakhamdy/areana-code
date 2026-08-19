from egyptian_id_ocr.normalization import (
    arabic_character_ratio,
    digits_only,
    mask_national_id,
    normalize_arabic,
)


def test_arabic_normalization_preserves_identity_letters():
    raw = "  أحمــد\u200f   عبدُالعزيز • محمد  "
    normalized = normalize_arabic(raw)
    assert normalized == "أحمد عبدالعزيز محمد"
    assert normalized.startswith("أ")  # do not silently turn أ into ا


def test_digit_normalization_supports_both_arabic_sets():
    assert digits_only("١٢٣-۴۵۶ 78") == "12345678"


def test_privacy_mask_and_arabic_ratio():
    assert mask_national_id("30001010100136") == "3000********36"
    assert arabic_character_ratio("محمد 123") == 1.0
