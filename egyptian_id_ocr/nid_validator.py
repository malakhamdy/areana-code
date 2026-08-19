"""Detailed Egyptian National ID structure parser and validator.

The checksum weights are the public Mod-11 implementation used by multiple local
validator libraries. Egypt does not publish a citable public checksum specification;
the method name is therefore exposed in every result instead of being presented as
proof that an identity exists.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from .normalization import digits_only
from .schemas import NIDValidationCode

GOVERNORATES_AR = {
    "01": "القاهرة",
    "02": "الإسكندرية",
    "03": "بور سعيد",
    "04": "السويس",
    "11": "دمياط",
    "12": "الدقهلية",
    "13": "الشرقية",
    "14": "القليوبية",
    "15": "كفر الشيخ",
    "16": "الغربية",
    "17": "المنوفية",
    "18": "البحيرة",
    "19": "الإسماعيلية",
    "21": "الجيزة",
    "22": "بني سويف",
    "23": "الفيوم",
    "24": "المنيا",
    "25": "أسيوط",
    "26": "سوهاج",
    "27": "قنا",
    "28": "أسوان",
    "29": "الأقصر",
    "31": "البحر الأحمر",
    "32": "الوادي الجديد",
    "33": "مطروح",
    "34": "شمال سيناء",
    "35": "جنوب سيناء",
    "88": "خارج الجمهورية",
}
CHECKSUM_WEIGHTS = (2, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)


def calculate_check_digit(first_thirteen_digits: str) -> int:
    if len(first_thirteen_digits) != 13 or not first_thirteen_digits.isdigit():
        raise ValueError("Checksum input must contain exactly 13 digits.")
    total = sum(int(digit) * weight for digit, weight in zip(first_thirteen_digits, CHECKSUM_WEIGHTS))
    value = 11 - (total % 11)
    if value == 10:
        return 0
    if value == 11:
        return 1
    return value


def validate_national_id(value: str | None, *, today: date | None = None) -> dict[str, Any]:
    normalized = digits_only(value)
    result: dict[str, Any] = {
        "normalized": normalized,
        "status": NIDValidationCode.VALID.value,
        "is_valid": False,
        "errors": [],
        "warnings": [],
        "checks": {
            "format": False,
            "century": False,
            "date": False,
            "governorate": False,
            "serial": False,
            "checksum": False,
        },
        "derived": {},
        "checksum_method": "public_mod11_weights_2765432765432",
        "checksum_scope": "mathematical consistency only; not proof of identity",
    }
    if len(normalized) != 14 or not normalized.isdigit():
        return _fail(result, NIDValidationCode.INVALID_FORMAT, "National ID must contain exactly 14 digits.")
    result["checks"]["format"] = True

    century_digit = normalized[0]
    if century_digit not in {"2", "3"}:
        return _fail(result, NIDValidationCode.INVALID_STRUCTURE, "Century digit must be 2 or 3.")
    result["checks"]["century"] = True

    year = (1900 if century_digit == "2" else 2000) + int(normalized[1:3])
    month, day = int(normalized[3:5]), int(normalized[5:7])
    try:
        birth_date = date(year, month, day)
        if birth_date > (today or date.today()):
            raise ValueError("future")
    except ValueError:
        return _fail(result, NIDValidationCode.INVALID_DATE, "Encoded date of birth is not a valid past date.")
    result["checks"]["date"] = True
    result["derived"]["date_of_birth"] = birth_date.isoformat()

    governorate_code = normalized[7:9]
    if governorate_code not in GOVERNORATES_AR:
        return _fail(
            result,
            NIDValidationCode.INVALID_GOVERNORATE,
            f"Unsupported birth-registration governorate code: {governorate_code}.",
        )
    result["checks"]["governorate"] = True
    result["derived"]["birth_governorate_code"] = governorate_code
    result["derived"]["birth_governorate_ar"] = GOVERNORATES_AR[governorate_code]

    serial = normalized[9:13]
    if serial == "0000":
        return _fail(result, NIDValidationCode.INVALID_STRUCTURE, "Serial component cannot be 0000.")
    result["checks"]["serial"] = True
    gender_digit = int(normalized[12])
    result["derived"]["gender_ar"] = "ذكر" if gender_digit % 2 else "أنثى"
    result["derived"]["gender_code_digit"] = str(gender_digit)

    expected = calculate_check_digit(normalized[:13])
    result["expected_check_digit"] = expected
    result["observed_check_digit"] = int(normalized[-1])
    if expected != int(normalized[-1]):
        return _fail(
            result,
            NIDValidationCode.INVALID_CHECKSUM,
            "Check digit does not match the configured public Mod-11 method.",
        )
    result["checks"]["checksum"] = True
    result["is_valid"] = True
    return result


def _fail(result: dict[str, Any], code: NIDValidationCode, message: str) -> dict[str, Any]:
    result["status"] = code.value
    result["errors"].append(message)
    result["is_valid"] = False
    return result
