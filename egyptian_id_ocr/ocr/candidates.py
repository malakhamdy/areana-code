"""Multi-pass OCR candidate generation and conservative ranking."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

import numpy as np

from ..nid_validator import validate_national_id
from ..normalization import (
    arabic_character_ratio,
    digits_only,
    normalize_arabic,
)
from ..preprocessing import preprocess_field
from ..schemas import OCRCandidate, OCRSpan
from .base import OCREngine


@dataclass
class CandidateDecision:
    selected: OCRCandidate | None
    agreement: float
    ambiguous: bool
    reason: str


def generate_candidates(
    field: str,
    crop_rgb: np.ndarray,
    engine: OCREngine,
    max_variants: int = 3,
) -> tuple[list[OCRCandidate], dict[str, np.ndarray]]:
    variants = preprocess_field(field, crop_rgb)
    candidates: list[OCRCandidate] = []
    for variant_name, image in list(variants.items())[:max_variants]:
        spans = engine.recognize(image)
        raw = _join_spans(spans, rtl=field not in {"national_id", "date_of_birth"})
        if field == "national_id":
            normalized = _best_nid_sequence(raw)
        elif field == "date_of_birth":
            normalized = digits_only(raw)
        else:
            normalized = normalize_arabic(raw)
        confidence = _weighted_span_confidence(spans)
        validation = validate_national_id(normalized) if field == "national_id" and normalized else {}
        score = _candidate_score(field, normalized, confidence, validation)
        candidates.append(
            OCRCandidate(
                raw=raw,
                normalized=normalized,
                confidence=round(confidence, 4),
                preprocessing_variant=variant_name,
                engine=engine.name,
                spans=spans,
                score=round(score, 4),
                validation=validation,
            )
        )
    return candidates, variants


def rank_candidates(field: str, candidates: list[OCRCandidate]) -> CandidateDecision:
    nonempty = [candidate for candidate in candidates if candidate.normalized]
    if not nonempty:
        return CandidateDecision(None, 0.0, True, "No preprocessing pass produced text.")
    ranked = sorted(nonempty, key=lambda item: item.score, reverse=True)
    best = ranked[0]
    similarities = [
        SequenceMatcher(None, best.normalized, other.normalized).ratio()
        for other in ranked[1:]
    ]
    agreement = max(similarities, default=0.0) if len(ranked) > 1 else 0.0

    distinct_close = [
        other
        for other in ranked[1:]
        if other.normalized != best.normalized and abs(best.score - other.score) < 0.08
    ]
    if field == "national_id":
        valid = [item for item in ranked if item.validation.get("is_valid")]
        valid_values = {item.normalized for item in valid}
        if len(valid_values) > 1:
            return CandidateDecision(
                None,
                agreement,
                True,
                "Multiple different structurally valid NID candidates disagree.",
            )
        if len(valid_values) == 1:
            chosen_value = next(iter(valid_values))
            chosen = max(
                (item for item in valid if item.normalized == chosen_value),
                key=lambda item: item.score,
            )
            valid_agreement = sum(
                item.normalized == chosen_value for item in nonempty
            ) / len(nonempty)
            return CandidateDecision(
                chosen,
                valid_agreement,
                False,
                "A unique structurally valid candidate outranked invalid alternatives.",
            )
        if distinct_close:
            return CandidateDecision(
                None,
                agreement,
                True,
                "NID passes disagree and no candidate validates; value was not forced.",
            )
    elif distinct_close and agreement < 0.72:
        return CandidateDecision(
            None,
            agreement,
            True,
            "Meaningfully different OCR passes have similar evidence; value was not forced.",
        )
    return CandidateDecision(
        best,
        agreement,
        False,
        "Highest-ranked candidate selected with candidate disagreement retained.",
    )


def _candidate_score(
    field: str, normalized: str, confidence: float, validation: dict
) -> float:
    if not normalized:
        return 0.0
    if field == "national_id":
        length_score = 1.0 if len(normalized) == 14 else max(0.0, 1 - abs(len(normalized) - 14) / 8)
        checks = validation.get("checks", {})
        check_score = sum(bool(value) for value in checks.values()) / max(1, len(checks))
        valid_bonus = 0.25 if validation.get("is_valid") else 0.0
        return float(np.clip(0.40 * confidence + 0.22 * length_score + 0.38 * check_score + valid_bonus, 0, 1))
    arabic_score = arabic_character_ratio(normalized)
    length_score = min(1.0, len(normalized) / (8 if field == "name" else 14))
    return float(np.clip(0.62 * confidence + 0.25 * arabic_score + 0.13 * length_score, 0, 1))


def _weighted_span_confidence(spans: list[OCRSpan]) -> float:
    if not spans:
        return 0.0
    weights = [max(1, len(span.text.strip())) for span in spans]
    return float(np.average([span.confidence for span in spans], weights=weights))


def _join_spans(spans: list[OCRSpan], *, rtl: bool) -> str:
    if not spans:
        return ""
    if any(span.bbox for span in spans):
        def center(span: OCRSpan) -> tuple[float, float]:
            if not span.bbox:
                return 0.0, 0.0
            points = np.asarray(span.bbox, dtype=float)
            return float(points[:, 0].mean()), float(points[:, 1].mean())
        # Group approximately by line, then read Arabic fragments right-to-left.
        ordered_y = sorted(spans, key=lambda span: center(span)[1])
        lines: list[list[OCRSpan]] = []
        for span in ordered_y:
            cy = center(span)[1]
            if not lines:
                lines.append([span])
                continue
            previous_y = np.mean([center(item)[1] for item in lines[-1]])
            heights = []
            for item in lines[-1] + [span]:
                if item.bbox:
                    pts = np.asarray(item.bbox, dtype=float)
                    heights.append(float(pts[:, 1].max() - pts[:, 1].min()))
            tolerance = max(10.0, np.mean(heights) * 0.55 if heights else 12.0)
            if abs(cy - previous_y) <= tolerance:
                lines[-1].append(span)
            else:
                lines.append([span])
        text_lines = []
        for line in lines:
            line.sort(key=lambda span: center(span)[0], reverse=rtl)
            text_lines.append(" ".join(span.text.strip() for span in line if span.text.strip()))
        return "\n".join(line for line in text_lines if line).strip()
    return "\n".join(span.text.strip() for span in spans if span.text.strip()).strip()


def _best_nid_sequence(raw: str) -> str:
    digits = digits_only(raw)
    if len(digits) <= 14:
        return digits
    windows = [digits[index : index + 14] for index in range(len(digits) - 13)]
    plausible = [window for window in windows if window[0] in "23"]
    pool = plausible or windows
    return max(
        pool,
        key=lambda value: (
            bool(validate_national_id(value).get("is_valid")),
            sum(validate_national_id(value).get("checks", {}).values()),
        ),
    )
