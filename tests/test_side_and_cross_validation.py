from egyptian_id_ocr.cross_field_validator import validate_cross_fields
from egyptian_id_ocr.schemas import FieldResult, FieldStatus
from egyptian_id_ocr.side_classifier import classify_side


def _field(name, value, source="printed_front_field"):
    return FieldResult(
        field=name,
        raw=value,
        normalized=value,
        source=source,
        bbox=None,
        ocr_engine=None,
        preprocessing_variant=None,
        ocr_confidence=0.9,
        localization_confidence=0.9,
        validation_confidence=0.0,
        cross_source_confidence=0.0,
        final_confidence=0.8,
        status=FieldStatus.EXTRACTED,
    )


def test_front_and_back_classification(synthetic_front_card, synthetic_back_card):
    assert classify_side(synthetic_front_card).side.value == "front"
    assert classify_side(synthetic_back_card).side.value == "back"


def test_dob_match_and_mismatch_are_explicit():
    printed = {"date_of_birth": _field("date_of_birth", "2000-01-01")}
    derived = {"date_of_birth": _field("date_of_birth", "2000-01-01", "national_id_structure")}
    matched = validate_cross_fields(printed, derived)
    assert "DOB_CROSS_VALIDATED" in matched["matches"]

    derived["date_of_birth"].normalized = "2000-01-02"
    conflict = validate_cross_fields(printed, derived)
    assert "DOB_MISMATCH" in conflict["mismatches"]
    assert conflict["overall_consistency"] == "CONFLICT"


def test_residence_address_is_not_equated_with_birth_governorate():
    result = validate_cross_fields(
        {"address": _field("address", "شارع تجريبي - الجيزة")},
        {"birth_governorate": _field("birth_governorate", "القاهرة", "national_id_structure")},
    )
    assert any("residence address" in warning for warning in result["warnings"])
