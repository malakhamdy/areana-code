# Egyptian National ID — Arabic-first document understanding

A local, inspectable Streamlit pipeline for **document-level** processing of Egyptian National ID photographs. It preserves Arabic identity text, detects and rectifies the physical card, localizes semantic fields in canonical card coordinates, runs field-specific OCR passes, validates the 14-digit NID, and reports extraction/validation/verification as different states.

> This is not an identity-authenticity service. A structurally valid number or repeated OCR result does not prove that a card or person is genuine.

## Repository assessment

The repository initially contained only `LICENSE`; there was no application, OCR code, configuration, tests, or sample fixture to preserve. The implementation therefore introduces a modular Python package rather than replacing existing working components.

| Assessment | Finding |
|---|---|
| Current architecture | None (license-only repository) |
| Current problems | No runnable code, dependencies, OCR, geometry, validation, UI, or tests |
| Required changes | Create the complete canonical-card pipeline and local Streamlit interface |
| Main files | `app.py`, `egyptian_id_ocr/*`, `tests/*`, `requirements.txt` |
| New modules | image I/O, card detection, geometry, side/layout, localization, preprocessing, OCR adapter/candidates, NID validation, cross-validation, independent verification, privacy, visualization |

## Pipeline

```text
INPUT (original preserved in memory)
  → image validation + quality assessment
  → aspect-preserving resize + letterbox metadata
  → multi-strategy physical card detection
  → ordered corners + perspective correction
  → 1280×808 canonical card
  → visual front/back classification
  → versioned normalized template search zones
  → ink-projection field refinement
  → crop quality assessment
  → field-specific preprocessing variants
  → Arabic PP-OCRv5 candidate generation
  → Arabic/digit normalization (raw retained)
  → conservative candidate ranking
  → NID format/date/governorate/serial/checksum checks
  → DOB/governorate/gender derivation from NID
  → cross-field checks when a genuinely independent source exists
  → structured result + full visual debugging
```

Coordinate systems are explicit:

1. `original_image`
2. `processing_canvas`
3. `canonical_card`
4. `field_crop`

Field localization uses the canonical card. It never treats arbitrary uploaded-image pixels as the source of truth.

## Supported extraction

### Front

- **National ID** — printed numeric field, multiple OCR passes, detailed structural validation
- **Name** — Arabic canonical output, multiline/RTL ordering, no English translation
- **Address** — Arabic canonical output, multiline crop and preprocessing
- **Date of birth** — derived from NID structure when available
- **Gender** — Arabic `ذكر` / `أنثى`, derived from the 13th NID digit when available
- **Birth/registration governorate** — Arabic value derived from digits 8–9

The sampled front layout does not print a separate DOB or gender field. Consequently, those values are marked as **NID-derived**, not falsely described as independently cross-validated. A current residence in the printed address is never equated with the birth/registration governorate code.

### Back

- front-only fields are **not forced**
- a separate PDF417/barcode region is localized and sent to `zxing-cpp`
- decoded barcode content is not automatically trusted or assumed to contain all front fields

## OCR model choice

Default:

- detector: `PP-OCRv5_server_det` (stronger local detection baseline)
- recognizer: `arabic_PP-OCRv5_mobile_rec`
- runtime: PaddlePaddle/PaddleOCR, automatic GPU detection with CPU fallback

`arabic_PP-OCRv5_mobile_rec` is Paddle's Arabic-script PP-OCRv5 model and supports Arabic letters and numbers. Arabic remains canonical throughout the extraction path. The UI allows the lighter mobile detector where memory/latency matters.

No second OCR model is enabled just for architectural complexity. A secondary model should only be added after a fixture-level benchmark demonstrates a practical gain. Model instances are cached and not recreated on every Streamlit rerun.

## Install and run

Python 3.10–3.12 is supported. Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0
```

On a headless Linux host without `libGL.so.1`, use the provided installer instead:

```bash
bash scripts/install_headless.sh
```

Paddle downloads model weights on the first analysis. Later analyses reuse the cached models. Force CPU with:

```bash
EGYID_DEVICE=cpu streamlit run app.py
```

Other environment overrides:

```bash
EGYID_DETECTION_MODEL=PP-OCRv5_mobile_det
EGYID_RECOGNITION_MODEL=arabic_PP-OCRv5_mobile_rec
EGYID_MAX_OCR_VARIANTS=3
```

## Streamlit debugging

Open the Streamlit URL, upload one or two images, and press **Analyze card**. The **Structured result**, **Geometry**, **Field crops**, **OCR & evidence**, and **JSON** tabs expose the complete visual sequence:

1. original image
2. processing/letterbox canvas
3. detected quadrilateral and ordered corners
4. perspective-corrected canonical card
5. localized field boxes
6. individual canonical crops
7. every preprocessing variant
8. OCR candidates and raw text
9. normalized values
10. validation and verification evidence
11. transformation/homography metadata

This makes it possible to distinguish a card-detection problem from a crop, preprocessing, OCR, normalization, or validation problem.

## Result semantics

- `EXTRACTED`: text was read from a printed region; not independently proven true
- `VALIDATED`: a field passed applicable structural/mathematical rules
- `VERIFIED`: NID OCR passes agree and the configured structural/checksum rules pass
- `CROSS_VALIDATED`: two genuinely independent sources match
- `LOW_CONFIDENCE`: evidence is weak or competing candidates cannot be resolved safely
- `CONFLICT`: independent sources disagree; neither is silently chosen

Every OCR field preserves:

- raw and normalized value
- source and OCR engine
- preprocessing variant
- canonical bbox and coordinate space
- OCR/localization/validation/cross-source/final confidence
- all candidates, validation details, warnings, and verification evidence

## NID validation

`nid_validator.py` distinguishes:

- `VALID`
- `INVALID_FORMAT`
- `INVALID_DATE`
- `INVALID_GOVERNORATE`
- `INVALID_CHECKSUM`
- `INVALID_STRUCTURE`

It checks length, digit form, century, a real non-future date, a supported governorate code, serial structure, and the public Mod-11 weight implementation `2,7,6,5,4,3,2,7,6,5,4,3,2`.

The Egyptian government does not provide a citable public checksum specification. For that reason the checksum method and scope are included in every validation result. A checksum match is treated only as mathematical consistency—not proof of identity or card authenticity.

## Tests

```bash
pytest
```

The suite includes:

- arbitrary input dimensions/aspect ratios and coordinate mapping
- scale-invariant localization at 25%, 50%, 100%, 150%, and 200%
- margins, rotation, and perspective distortion
- crop stability
- synthetic front/back side classification
- Arabic/Persian/mixed digit normalization
- valid/invalid date, governorate, structure, checksum, and length
- competing OCR candidates that must not be forced
- low-quality blur/contrast gates and field-specific preprocessing variants
- DOB match/mismatch and residence-vs-birth-governorate semantics

Synthetic fixtures contain no real identity data. Correct OCR output alone is not used as proof that a crop is geometrically correct.

## Optional Kaggle fixtures

The referenced dataset is **not committed** because identity-card images are sensitive. After accepting its terms and configuring the Kaggle CLI outside this repository:

```bash
pip install kaggle
python scripts/download_kaggle_data.py
```

Files go to the ignored path `data/private/kaggle_egyptian_ids/`. Review consent, licensing, and redaction before use. Public CI should use synthetic or fully redacted fixtures.

## Privacy

- uploads are processed in memory and are not saved by the app
- full NID values are not written to logs
- JSON download masks NID by default; the user must explicitly enable complete export
- model/cache directories, raw data, private data, outputs, and archives are ignored by Git
- never commit real front/back identity photographs

## Current limitations and honest acceptance status

The code establishes and tests the full architecture, but production acceptance requires a labeled, consented multi-layout benchmark. In particular:

- normalized templates cover the supplied legacy/current front arrangement and a generic back; additional card generations need registered templates and labeled fixtures
- ink refinement is dynamic inside semantic search regions, but a trained field detector may outperform it after sufficient labeled data exists
- face/barcode side cues are heuristic and return `unknown` rather than forcing weak classifications
- glare, severe clipping, tiny cards, and motion blur can remain unrecoverable
- checksum validation follows a disclosed community implementation, not a public government specification
- Paddle's published general Arabic benchmark is not an Egyptian-ID field benchmark; project-specific accuracy must be measured before claiming production success

Run `pytest`, inspect all visual overlays, and benchmark on multiple resolution, scale, rotation, perspective, blur, compression, margin, and lighting variants before deployment.
