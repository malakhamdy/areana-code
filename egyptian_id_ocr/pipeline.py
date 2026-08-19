"""End-to-end Arabic-first Egyptian ID document understanding pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .barcode import decode_barcode
from .card_detector import detect_card
from .config import OCRConfig
from .cross_field_validator import validate_cross_fields
from .geometry import rectify_card
from .image_io import decode_image, letterbox_image, processing_to_original_points
from .independent_verifier import verify_ocr_field
from .localization import crop_localized_field, localize_fields
from .nid_validator import validate_national_id
from .ocr.base import OCREngine, UnavailableOCREngine
from .ocr.candidates import generate_candidates, rank_candidates
from .quality import assess_image_quality
from .schemas import (
    BBox,
    CoordinateSpace,
    DocumentResult,
    DocumentSide,
    FieldResult,
    FieldStatus,
    ImageQuality,
    LocalizedField,
    OCRCandidate,
)
from .side_classifier import classify_side
from .templates import template_for_side
from .visualization import draw_card_detection, draw_field_boxes


@dataclass
class PipelineArtifacts:
    original_image: np.ndarray
    processing_image: np.ndarray
    card_detection_overlay: np.ndarray
    canonical_card: np.ndarray
    field_localization_overlay: np.ndarray
    crops: dict[str, np.ndarray] = field(default_factory=dict)
    preprocessed_crops: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


@dataclass
class PipelineOutput:
    result: DocumentResult
    artifacts: PipelineArtifacts


class EgyptianIDPipeline:
    def __init__(
        self,
        ocr_engine: OCREngine | None = None,
        config: OCRConfig | None = None,
    ) -> None:
        self.config = config or OCRConfig.from_env()
        self.ocr_engine = ocr_engine or UnavailableOCREngine()

    def process(self, image: bytes | bytearray | np.ndarray) -> PipelineOutput:
        original = decode_image(image)
        image_quality = assess_image_quality(original)
        processing, transform = letterbox_image(original)
        detection = detect_card(processing, transform)
        canonical, homography = rectify_card(
            processing, np.asarray(detection.corners, dtype=np.float32)
        )
        canonical_quality = assess_image_quality(canonical)
        side = classify_side(canonical)
        template, template_confidence = template_for_side(side.side)
        localized = localize_fields(canonical, template)

        crops: dict[str, np.ndarray] = {}
        preprocessed: dict[str, dict[str, np.ndarray]] = {}
        fields: dict[str, FieldResult] = {}
        derived: dict[str, FieldResult] = {}
        ocr_debug: dict[str, Any] = {}
        crop_quality_debug: dict[str, Any] = {}

        if side.side == DocumentSide.FRONT:
            for field_name in ("national_id", "name", "address"):
                localization = localized.get(field_name)
                if not localization:
                    continue
                crop = crop_localized_field(canonical, localization)
                crops[field_name] = crop
                crop_quality = assess_image_quality(crop, is_crop=True)
                crop_quality_debug[field_name] = asdict(crop_quality)
                try:
                    candidates, variants = generate_candidates(
                        field_name,
                        crop,
                        self.ocr_engine,
                        max_variants=self.config.max_variants_per_field,
                    )
                except Exception as exc:
                    candidates, variants = [], {}
                    localization.warnings.append(
                        f"OCR failed for this field ({type(exc).__name__}); geometry is still available."
                    )
                preprocessed[field_name] = variants
                decision = rank_candidates(field_name, candidates)
                verification = verify_ocr_field(
                    field_name,
                    decision,
                    candidates,
                    localization_confidence=localization.localization_confidence,
                )
                result = _build_ocr_field_result(
                    field_name,
                    localization,
                    crop_quality,
                    candidates,
                    decision.selected,
                    verification,
                )
                fields[field_name] = result
                ocr_debug[field_name] = [
                    {
                        "raw": candidate.raw,
                        "normalized": candidate.normalized,
                        "confidence": candidate.confidence,
                        "score": candidate.score,
                        "variant": candidate.preprocessing_variant,
                        "validation_status": candidate.validation.get("status"),
                    }
                    for candidate in candidates
                ]

            national_id = fields.get("national_id")
            if national_id and national_id.normalized:
                nid_validation = validate_national_id(national_id.normalized)
                national_id.validation = nid_validation
                derived.update(_derive_from_nid(national_id, nid_validation))
        elif side.side == DocumentSide.BACK:
            localization = localized.get("barcode")
            if localization:
                crop = crop_localized_field(canonical, localization)
                crops["barcode"] = crop
                from .preprocessing import preprocess_barcode

                variants = preprocess_barcode(crop)
                preprocessed["barcode"] = variants
                barcode_result = decode_barcode(variants)
                fields["barcode"] = _build_barcode_field(localization, barcode_result)
        else:
            # Unknown side intentionally receives no front fields.
            pass

        cross_validation = validate_cross_fields(fields, derived)
        warnings = [*detection.warnings]
        warnings.extend(image_quality.issues)
        if not canonical_quality.sufficient:
            warnings.append("The rectified card remains low quality; inspect field crops.")
        if side.side == DocumentSide.UNKNOWN:
            warnings.append("Front/back side is uncertain; semantic extraction was not forced.")
        if isinstance(self.ocr_engine, UnavailableOCREngine) and side.side == DocumentSide.FRONT:
            warnings.append(self.ocr_engine.reason)

        original_corners = processing_to_original_points(
            np.asarray(detection.corners, dtype=np.float32), transform
        )
        document_result = DocumentResult(
            is_egyptian_id=(
                detection.confidence >= 0.48
                and side.side != DocumentSide.UNKNOWN
                and template_confidence >= 0.5
            ),
            side=side.side,
            side_confidence=side.confidence,
            card_detection_confidence=detection.confidence,
            template_id=template.template_id,
            template_confidence=template_confidence,
            image_quality=image_quality,
            fields=fields,
            derived=derived,
            cross_field_validation=cross_validation,
            warnings=warnings,
            debug={
                "transformations": {
                    "input_letterbox": asdict(transform),
                    "processing_to_canonical_homography": homography.tolist(),
                    "card_corners_processing": detection.corners,
                    "card_corners_original": original_corners.tolist(),
                    "card_corner_coordinate_space": CoordinateSpace.PROCESSING_CANVAS.value,
                },
                "card_detection": {
                    "method": detection.method,
                    "confidence": detection.confidence,
                },
                "side_classification": {
                    "side": side.side.value,
                    "confidence": side.confidence,
                    "cues": side.cues,
                },
                "localization": {
                    name: {
                        "bbox": field.bbox.as_xyxy(),
                        "coordinate_space": field.bbox.coordinate_space.value,
                        "normalized_bbox": field.bbox.normalized(
                            canonical.shape[1], canonical.shape[0]
                        ),
                        "method": field.method,
                        "confidence": field.localization_confidence,
                        "warnings": field.warnings,
                    }
                    for name, field in localized.items()
                },
                "crop_quality": crop_quality_debug,
                "ocr_candidates": ocr_debug,
            },
        )
        artifacts = PipelineArtifacts(
            original_image=original,
            processing_image=processing,
            card_detection_overlay=draw_card_detection(processing, detection),
            canonical_card=canonical,
            field_localization_overlay=draw_field_boxes(canonical, localized),
            crops=crops,
            preprocessed_crops=preprocessed,
        )
        return PipelineOutput(document_result, artifacts)


def _build_ocr_field_result(
    field_name: str,
    localization: LocalizedField,
    crop_quality: ImageQuality,
    candidates: list[OCRCandidate],
    selected: OCRCandidate | None,
    verification: dict[str, Any],
) -> FieldResult:
    status = FieldStatus(verification["status"])
    ocr_confidence = selected.confidence if selected else 0.0
    quality_factor = crop_quality.overall_score
    final_confidence = (
        0.46 * ocr_confidence
        + 0.30 * localization.localization_confidence
        + 0.24 * quality_factor
    )
    if selected and field_name == "national_id" and selected.validation.get("is_valid"):
        validation_confidence = 1.0
    elif selected and field_name == "national_id":
        checks = selected.validation.get("checks", {})
        validation_confidence = sum(bool(value) for value in checks.values()) / max(1, len(checks))
    else:
        validation_confidence = 0.0
    warnings = [*localization.warnings, *crop_quality.issues]
    if selected is None and candidates:
        warnings.append("OCR candidates conflict; no final value was selected.")
    return FieldResult(
        field=field_name,
        raw=selected.raw if selected else None,
        normalized=selected.normalized if selected else None,
        source="printed_front_field",
        bbox=localization.bbox,
        ocr_engine=selected.engine if selected else getattr(candidates[0], "engine", None) if candidates else None,
        preprocessing_variant=selected.preprocessing_variant if selected else None,
        ocr_confidence=round(ocr_confidence, 4),
        localization_confidence=localization.localization_confidence,
        validation_confidence=round(validation_confidence, 4),
        cross_source_confidence=round(
            verification.get("evidence", [{}])[0].get("agreement", 0.0), 4
        ),
        final_confidence=round(float(np.clip(final_confidence, 0, 1)), 4),
        status=status,
        candidates=candidates,
        validation=selected.validation if selected else {},
        verification=verification,
        warnings=warnings,
    )


def _derive_from_nid(
    national_id: FieldResult, validation: dict[str, Any]
) -> dict[str, FieldResult]:
    derived_values = validation.get("derived", {})
    complete = bool(validation.get("is_valid"))
    source_confidence = national_id.final_confidence * (1.0 if complete else 0.55)
    status = FieldStatus.VALIDATED if complete else FieldStatus.LOW_CONFIDENCE
    warning = [] if complete else [
        "Derived from an NID candidate that did not pass every validation check."
    ]
    output: dict[str, FieldResult] = {}
    mapping = {
        "date_of_birth": derived_values.get("date_of_birth"),
        "gender": derived_values.get("gender_ar"),
        "birth_governorate": derived_values.get("birth_governorate_ar"),
    }
    for field_name, value in mapping.items():
        if value is None:
            continue
        output[field_name] = FieldResult(
            field=field_name,
            raw=value,
            normalized=value,
            source="national_id_structure",
            bbox=national_id.bbox,
            ocr_engine=None,
            preprocessing_variant=None,
            ocr_confidence=national_id.ocr_confidence,
            localization_confidence=national_id.localization_confidence,
            validation_confidence=1.0 if complete else 0.55,
            cross_source_confidence=0.0,
            final_confidence=round(source_confidence, 4),
            status=status,
            validation={
                "source_nid_status": validation.get("status"),
                "source_nid_valid": complete,
            },
            verification={
                "verified": False,
                "reason": "Derived from NID structure; no independent printed source was available.",
            },
            warnings=warning.copy(),
        )
    return output


def _build_barcode_field(
    localization: LocalizedField, barcode_result: dict[str, Any]
) -> FieldResult:
    decoded = bool(barcode_result.get("decoded"))
    return FieldResult(
        field="barcode",
        raw=barcode_result.get("text"),
        normalized=barcode_result.get("text"),
        source="back_barcode_decoder",
        bbox=localization.bbox,
        ocr_engine="zxing-cpp" if barcode_result.get("status") != "decoder_unavailable" else None,
        preprocessing_variant=barcode_result.get("variant"),
        ocr_confidence=0.0,
        localization_confidence=localization.localization_confidence,
        validation_confidence=0.0,
        cross_source_confidence=0.0,
        final_confidence=0.78 if decoded else 0.20,
        status=FieldStatus.EXTRACTED if decoded else FieldStatus.NOT_EXTRACTED,
        validation={},
        verification={
            "verified": False,
            "reason": "Decoded barcode content has not been cross-checked against front fields.",
            "decoder": barcode_result,
        },
        warnings=[] if decoded else [barcode_result.get("warning", barcode_result["status"])],
    )
