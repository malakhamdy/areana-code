from datetime import date

import pytest

from egyptian_id_ocr.nid_validator import calculate_check_digit, validate_national_id


def make_id(prefix13: str) -> str:
    return prefix13 + str(calculate_check_digit(prefix13))


def test_valid_synthetic_nid_and_derived_arabic_values():
    nid = make_id("3000101010013")
    result = validate_national_id(nid, today=date(2026, 8, 19))
    assert result["status"] == "VALID"
    assert result["is_valid"] is True
    assert result["derived"]["date_of_birth"] == "2000-01-01"
    assert result["derived"]["birth_governorate_ar"] == "القاهرة"
    assert result["derived"]["gender_ar"] == "ذكر"


def test_arabic_and_mixed_digits_are_normalized():
    result = validate_national_id("٣٠٠٠١٠١٠١٠٠١3٦")
    assert result["normalized"] == "30001010100136"
    assert result["is_valid"] is True


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("3001301010013", "INVALID_DATE"),
        ("3000230010013", "INVALID_DATE"),
        ("3000101990013", "INVALID_GOVERNORATE"),
        ("3000101010000", "INVALID_STRUCTURE"),
    ],
)
def test_detailed_structural_failures(prefix, expected):
    result = validate_national_id(make_id(prefix))
    assert result["status"] == expected
    assert result["is_valid"] is False


def test_wrong_length_and_checksum_are_distinct():
    assert validate_national_id("30001")["status"] == "INVALID_FORMAT"
    valid = make_id("3000101010013")
    wrong = valid[:-1] + str((int(valid[-1]) + 1) % 10)
    assert validate_national_id(wrong)["status"] == "INVALID_CHECKSUM"
