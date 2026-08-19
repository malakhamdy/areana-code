# Architecture decisions

## Existing repository

At implementation start the repository contained only an Apache 2.0 license. There were no components to migrate or preserve.

## Boundaries

- `image_io.py`: input safety, EXIF, original/canvas separation, letterbox transform
- `card_detector.py` + `geometry.py`: physical-card evidence and homography
- `side_classifier.py` + `templates.py`: side/layout evidence and versioned normalized search zones
- `localization.py`: canonical semantic zones refined by image ink
- `preprocessing.py`: independent transformations per field
- `ocr/`: engine protocol, cached Paddle adapter, candidate generation/ranking
- `normalization.py`: Arabic Unicode and digits; never overwrites raw
- `nid_validator.py`: structure/math and Arabic derivations
- `cross_field_validator.py`: relationships between genuinely different sources
- `independent_verifier.py`: extraction vs validation vs verification policy
- `pipeline.py`: orchestration and partial-failure isolation
- `visualization.py`: debuggable geometry/localization overlays
- `privacy.py`: explicit safe export behavior
- `app.py`: presentation only; model cached as a resource

## Confidence policy

OCR confidence, localization confidence, validation confidence, cross-source confidence, and final confidence are independent values. A high OCR probability cannot create `VERIFIED` status. Competing plausible values are retained as candidates and may produce no selected value.

## Geometry policy

The original image remains untouched. Detection uses a letterboxed processing canvas. All semantic field boxes live in a fixed 1280×808 canonical card and include `coordinate_space=canonical_card`. Upload size and aspect ratio therefore cannot directly move a logical field.

## Model policy

PaddleOCR is primary. `PP-OCRv5_server_det` plus `arabic_PP-OCRv5_mobile_rec` is the default local baseline. The recognizer is Arabic-script-specific and Arabic output is not routed through translation. Alternative engines must earn inclusion through a labeled project benchmark.

## Privacy policy

No uploaded image is persisted. Complete NIDs are not logged. Real ID fixtures and model caches are ignored by Git. Exports are masked by default. Synthetic tests are used in public source.
