"""Streamlit interface for the Arabic-first Egyptian ID pipeline."""
from __future__ import annotations

import html
import json
from typing import Any

import streamlit as st

from egyptian_id_ocr.config import OCRConfig
from egyptian_id_ocr.ocr.base import UnavailableOCREngine
from egyptian_id_ocr.ocr.paddle_engine import get_paddle_engine
from egyptian_id_ocr.pipeline import EgyptianIDPipeline, PipelineOutput
from egyptian_id_ocr.privacy import redacted_result

st.set_page_config(
    page_title="Egyptian ID • Arabic OCR",
    page_icon="🇪🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; max-width: 1500px;}
.hero {padding: 1.3rem 1.5rem; border-radius: 18px; color: white;
  background: linear-gradient(125deg, #0d4935 0%, #157a55 60%, #d6a72b 140%);
  margin-bottom: 1rem; box-shadow: 0 12px 30px rgba(13,73,53,.16);}
.hero h1 {font-size: 2rem; margin: 0 0 .35rem 0;}
.hero p {margin: 0; opacity: .92;}
.rtl-value {direction: rtl; text-align: right; font-size: 1.15rem; font-family: Tahoma, Arial, sans-serif;
  border: 1px solid #cfe3d8; background: white; padding: .7rem .85rem; border-radius: 10px;}
.status {display:inline-block; padding:.22rem .58rem; border-radius:999px; font-size:.76rem; font-weight:700;}
.good {background:#d8f4e6; color:#075c3a}.warn {background:#fff0cc;color:#815400}.bad {background:#fde2e2;color:#912626}
.small-note {color:#547065; font-size:.84rem;}
[data-testid="stMetric"] {background:white; border:1px solid #dce9e2; padding:.65rem; border-radius:12px;}
</style>
<div class="hero">
  <h1>Arabic-first Egyptian National ID understanding</h1>
  <p>كشف البطاقة ← تصحيح المنظور ← تحديد الحقول ← OCR عربي ← تحقق مستقل</p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Arabic PP-OCRv5 models for the first time…")
def load_ocr(device: str, detection_model: str, recognition_model: str):
    return get_paddle_engine(device, detection_model, recognition_model)


with st.sidebar:
    st.header("Processing settings")
    device = st.selectbox("Compute device", ["auto", "cpu", "gpu:0"], index=0)
    detector_choice = st.selectbox(
        "Text detector",
        ["PP-OCRv5_server_det", "PP-OCRv5_mobile_det"],
        help="Server detection is the stronger default; mobile uses less memory.",
    )
    variants = st.slider("OCR passes per field", 1, 4, 3)
    include_sensitive_export = st.toggle(
        "Include complete NID in JSON export",
        value=False,
        help="Off by default. On-screen extraction remains visible to you.",
    )
    st.divider()
    st.markdown("**Canonical recognizer**")
    st.code("arabic_PP-OCRv5_mobile_rec", language=None)
    st.caption("Arabic is retained. No Arabic→English translation is used for extraction.")
    st.warning(
        "Identity images are processed in memory. This app does not save uploads or log full NID values."
    )

uploaded = st.file_uploader(
    "Upload one or two card photographs (front/back order does not matter)",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True,
    help="For best results: fill the frame, avoid glare, and keep all four corners visible.",
)
if len(uploaded) > 2:
    st.warning("Only the first two images will be processed in this run.")

run = st.button("Analyze card", type="primary", disabled=not uploaded, use_container_width=True)

if not uploaded:
    col1, col2, col3 = st.columns(3)
    col1.info("**1 · Geometry**\n\nAspect-preserving normalization, physical card detection, corners, and homography.")
    col2.info("**2 · Arabic OCR**\n\nField-specific crops and preprocessing with Arabic PP-OCRv5.")
    col3.info("**3 · Evidence**\n\nNID structure, candidate agreement, provenance, and explicit conflicts.")
    st.caption(
        "DOB and gender are derived from a validated NID when the selected card layout does not print them. "
        "They are not described as independently cross-validated unless a second source exists."
    )

if run:
    engine: Any
    try:
        engine = load_ocr(device, detector_choice, "arabic_PP-OCRv5_mobile_rec")
        st.success(f"OCR ready: {engine.name}")
    except Exception as exc:
        st.error(f"Arabic OCR could not load: {exc}")
        st.info("Geometry and localization will still run, but fields cannot be extracted.")
        engine = UnavailableOCREngine(str(exc))

    config = OCRConfig(
        device=device,
        detection_model=detector_choice,
        recognition_model="arabic_PP-OCRv5_mobile_rec",
        max_variants_per_field=variants,
    )
    pipeline = EgyptianIDPipeline(engine, config)
    outputs: list[tuple[str, PipelineOutput]] = []
    progress = st.progress(0, text="Starting document pipeline…")
    for index, file in enumerate(uploaded[:2]):
        try:
            output = pipeline.process(file.getvalue())
            outputs.append((file.name, output))
        except Exception as exc:
            st.error(f"Could not process image {index + 1}: {type(exc).__name__}: {exc}")
        progress.progress(
            (index + 1) / min(2, len(uploaded)), text=f"Processed image {index + 1}"
        )
    progress.empty()
    st.session_state["pipeline_outputs"] = outputs

for image_index, (filename, output) in enumerate(st.session_state.get("pipeline_outputs", []), start=1):
    result = output.result
    artifacts = output.artifacts
    st.markdown(f"## Document {image_index}")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Card", "Detected" if result.is_egyptian_id else "Uncertain")
    metric_cols[1].metric("Side", result.side.value.upper())
    metric_cols[2].metric("Card confidence", f"{result.card_detection_confidence:.0%}")
    metric_cols[3].metric("Side confidence", f"{result.side_confidence:.0%}")
    metric_cols[4].metric("Image quality", f"{result.image_quality.overall_score:.0%}")

    if result.warnings:
        with st.expander(f"⚠ {len(result.warnings)} processing warning(s)", expanded=True):
            for warning in result.warnings:
                st.write(f"• {warning}")

    result_tab, geometry_tab, crops_tab, evidence_tab, json_tab = st.tabs(
        ["Structured result", "Geometry", "Field crops", "OCR & evidence", "JSON"]
    )

    with result_tab:
        all_fields = {**result.fields, **result.derived}
        if not all_fields:
            st.warning("No semantic fields were forced for this side/classification.")
        rows = []
        for name, field in all_fields.items():
            value = field.normalized or "—"
            rows.append(
                {
                    "Field": name,
                    "Value": value,
                    "Source": field.source,
                    "OCR": f"{field.ocr_confidence:.0%}" if field.ocr_confidence else "—",
                    "Localization": f"{field.localization_confidence:.0%}",
                    "Validation": f"{field.validation_confidence:.0%}" if field.validation_confidence else "—",
                    "Status": field.status.value,
                }
            )
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.markdown("### Arabic canonical values")
            for name, field in all_fields.items():
                if not field.normalized:
                    continue
                css_class = "good" if field.status.value in {"VERIFIED", "VALIDATED", "CROSS_VALIDATED"} else "warn" if field.status.value in {"EXTRACTED", "LOW_CONFIDENCE"} else "bad"
                safe_value = html.escape(str(field.normalized)).replace("\n", "<br>")
                st.markdown(
                    f"**{html.escape(name)}** &nbsp; <span class='status {css_class}'>{field.status.value}</span>"
                    f"<div class='rtl-value' lang='ar'>{safe_value}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("### Cross-field consistency")
        cross = result.cross_field_validation
        st.write(f"**Overall:** {cross.get('overall_consistency')}")
        for item in cross.get("matches", []):
            st.success(item)
        for item in cross.get("mismatches", []):
            st.error(item)
        for item in cross.get("warnings", []):
            st.info(item)

    with geometry_tab:
        st.markdown("#### 1. Original image (preserved)")
        st.image(artifacts.original_image, use_container_width=True)
        st.markdown("#### 2. Aspect-preserving processing canvas")
        st.image(artifacts.processing_image, use_container_width=True)
        st.markdown("#### 3. Physical card + ordered corners")
        st.image(artifacts.card_detection_overlay, use_container_width=True)
        st.markdown("#### 4. Perspective-corrected canonical card")
        st.image(artifacts.canonical_card, use_container_width=True)
        st.markdown("#### 5. Dynamic field localization")
        st.image(artifacts.field_localization_overlay, use_container_width=True)
        with st.expander("Transformations and normalized boxes"):
            st.json(result.debug.get("transformations", {}), expanded=False)
            st.json(result.debug.get("localization", {}), expanded=False)

    with crops_tab:
        if not artifacts.crops:
            st.info("No field crops are available for this side.")
        for name, crop in artifacts.crops.items():
            st.markdown(f"### {name}")
            st.image(crop, caption="Canonical-card crop", use_container_width=True)
            quality = result.debug.get("crop_quality", {}).get(name)
            if quality:
                st.json(quality, expanded=False)
            variants_map = artifacts.preprocessed_crops.get(name, {})
            if variants_map:
                columns = st.columns(min(3, len(variants_map)))
                for variant_index, (variant_name, variant_image) in enumerate(variants_map.items()):
                    columns[variant_index % len(columns)].image(
                        variant_image, caption=variant_name, use_container_width=True
                    )

    with evidence_tab:
        all_fields = {**result.fields, **result.derived}
        for name, field in all_fields.items():
            with st.expander(f"{name} · {field.status.value}", expanded=name == "national_id"):
                st.write(field.verification.get("reason", "No verification evidence."))
                if field.warnings:
                    for warning in field.warnings:
                        st.warning(warning)
                if field.candidates:
                    st.dataframe(
                        [
                            {
                                "Variant": candidate.preprocessing_variant,
                                "Raw": candidate.raw,
                                "Normalized": candidate.normalized,
                                "OCR confidence": candidate.confidence,
                                "Rank score": candidate.score,
                                "Validation": candidate.validation.get("status", "—"),
                            }
                            for candidate in field.candidates
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                if field.validation:
                    st.markdown("**Validation details**")
                    st.json(field.validation, expanded=False)
                if field.verification:
                    st.markdown("**Verification details**")
                    st.json(field.verification, expanded=False)

    with json_tab:
        payload = result.to_dict()
        if not include_sensitive_export:
            payload = redacted_result(payload)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        st.code(serialized, language="json")
        st.download_button(
            "Download result JSON",
            serialized,
            file_name=f"egyptian_id_result_{image_index}.json",
            mime="application/json",
            key=f"download-{image_index}",
        )
    st.divider()
