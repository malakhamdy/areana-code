"""Cross-field consistency checks. Mismatches are exposed, never auto-resolved."""
from __future__ import annotations

from typing import Any

from .normalization import normalize_arabic
from .schemas import FieldResult


def validate_cross_fields(
    fields: dict[str, FieldResult], derived: dict[str, FieldResult]
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    matches: list[str] = []
    mismatches: list[str] = []
    warnings: list[str] = []

    printed_dob = fields.get("date_of_birth")
    derived_dob = derived.get("date_of_birth")
    if printed_dob and printed_dob.normalized and derived_dob and derived_dob.normalized:
        matched = printed_dob.normalized == derived_dob.normalized
        checks.append(
            {
                "check": "printed_dob_vs_nid_dob",
                "left": printed_dob.normalized,
                "right": derived_dob.normalized,
                "matched": matched,
            }
        )
        (matches if matched else mismatches).append(
            "DOB_CROSS_VALIDATED" if matched else "DOB_MISMATCH"
        )
    elif derived_dob:
        warnings.append("DOB is derived from NID only; no independent printed DOB was available.")

    printed_gender = fields.get("gender")
    derived_gender = derived.get("gender")
    if printed_gender and printed_gender.normalized and derived_gender and derived_gender.normalized:
        matched = normalize_arabic(printed_gender.normalized) == normalize_arabic(
            derived_gender.normalized
        )
        checks.append(
            {
                "check": "printed_gender_vs_nid_gender",
                "left": printed_gender.normalized,
                "right": derived_gender.normalized,
                "matched": matched,
            }
        )
        (matches if matched else mismatches).append(
            "GENDER_CROSS_VALIDATED" if matched else "GENDER_MISMATCH"
        )
    elif derived_gender:
        warnings.append("Gender is derived from NID only; no independent printed field was available.")

    printed_governorate = fields.get("birth_governorate")
    derived_governorate = derived.get("birth_governorate")
    if printed_governorate and printed_governorate.normalized and derived_governorate:
        matched = normalize_arabic(printed_governorate.normalized) == normalize_arabic(
            derived_governorate.normalized
        )
        checks.append(
            {
                "check": "printed_birth_governorate_vs_nid_code",
                "left": printed_governorate.normalized,
                "right": derived_governorate.normalized,
                "matched": matched,
            }
        )
        (matches if matched else mismatches).append(
            "BIRTH_GOVERNORATE_MATCH" if matched else "BIRTH_GOVERNORATE_MISMATCH"
        )
    if fields.get("address") and derived_governorate:
        warnings.append(
            "NID governorate is birth/registration evidence and was not compared to the residence address."
        )

    if mismatches:
        overall = "CONFLICT"
    elif matches:
        overall = "CONSISTENT"
    else:
        overall = "INSUFFICIENT_INDEPENDENT_EVIDENCE"
    return {
        "checks": checks,
        "matches": matches,
        "mismatches": mismatches,
        "warnings": warnings,
        "overall_consistency": overall,
    }
