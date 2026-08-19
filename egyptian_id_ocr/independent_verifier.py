"""Separate extraction, mathematical validation, and independent verification."""
from __future__ import annotations

from typing import Any

from .ocr.candidates import CandidateDecision
from .schemas import FieldStatus, OCRCandidate


def verify_ocr_field(
    field: str,
    decision: CandidateDecision,
    candidates: list[OCRCandidate],
    *,
    localization_confidence: float,
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    if decision.selected:
        same = sum(
            candidate.normalized == decision.selected.normalized
            for candidate in candidates
            if candidate.normalized
        )
        nonempty = sum(bool(candidate.normalized) for candidate in candidates)
        evidence.append(
            {
                "type": "multi_pass_ocr_agreement",
                "supporting_passes": same,
                "available_passes": nonempty,
                "agreement": round(same / max(1, nonempty), 4),
            }
        )
    evidence.append(
        {
            "type": "localization",
            "confidence": localization_confidence,
            "independent_of_ocr": True,
        }
    )

    if decision.ambiguous or not decision.selected:
        return {
            "status": FieldStatus.LOW_CONFIDENCE.value,
            "verified": False,
            "reason": decision.reason,
            "evidence": evidence,
        }

    selected = decision.selected
    if field == "national_id":
        valid = bool(selected.validation.get("is_valid"))
        repeated = sum(
            candidate.normalized == selected.normalized for candidate in candidates
        ) >= 2
        if valid and repeated:
            status = FieldStatus.VERIFIED
            reason = "Independent OCR passes agree and mathematical/structural checks pass."
        elif valid:
            status = FieldStatus.VALIDATED
            reason = "Mathematical/structural checks pass, but a second OCR pass did not agree."
        else:
            status = FieldStatus.LOW_CONFIDENCE
            reason = "The OCR value did not pass complete NID validation."
        evidence.append(
            {
                "type": "nid_structure_and_checksum",
                "passed": valid,
                "details": selected.validation,
            }
        )
        return {
            "status": status.value,
            "verified": status == FieldStatus.VERIFIED,
            "reason": reason,
            "evidence": evidence,
        }

    # Agreement verifies repeatability of extraction, not the person's real-world identity.
    if decision.agreement >= 0.82 and selected.confidence >= 0.70:
        status = FieldStatus.EXTRACTED
        reason = "Multiple OCR passes support the extraction; identity truth is not independently verified."
    elif selected.confidence >= 0.55:
        status = FieldStatus.EXTRACTED
        reason = "Extracted from the printed field with limited independent evidence."
    else:
        status = FieldStatus.LOW_CONFIDENCE
        reason = "OCR or candidate agreement is too weak."
    return {
        "status": status.value,
        "verified": False,
        "reason": reason,
        "evidence": evidence,
    }
