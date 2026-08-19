"""Arabic-first text and digit normalization; raw OCR is never overwritten."""
from __future__ import annotations

import re
import unicodedata

ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
EASTERN_ARABIC = "۰۱۲۳۴۵۶۷۸۹"
DIGIT_TRANSLATION = str.maketrans(
    ARABIC_INDIC + EASTERN_ARABIC,
    "0123456789" * 2,
)
ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
BIDI_CONTROLS = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")


def normalize_digits(text: str | None) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKC", str(text)).translate(DIGIT_TRANSLATION)


def digits_only(text: str | None) -> str:
    return re.sub(r"\D", "", normalize_digits(text))


def normalize_arabic(text: str | None, *, remove_diacritics: bool = True) -> str:
    if not text:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = BIDI_CONTROLS.sub("", value)
    value = value.replace("ـ", "")
    value = value.translate(DIGIT_TRANSLATION)
    if remove_diacritics:
        value = ARABIC_DIACRITICS.sub("", value)
    # Normalize OCR punctuation, but do not alter identity letters (ا/أ/إ etc.).
    value = re.sub(r"[|¦•·]+", " ", value)
    value = re.sub(r"\s*([،,:؛;\-/])\s*", r" \1 ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def arabic_character_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    arabic = sum("\u0600" <= char <= "\u06ff" for char in letters)
    return arabic / len(letters)


def mask_national_id(value: str | None) -> str:
    digits = digits_only(value)
    if len(digits) <= 6:
        return "*" * len(digits)
    return f"{digits[:4]}{'*' * (len(digits) - 6)}{digits[-2:]}"
