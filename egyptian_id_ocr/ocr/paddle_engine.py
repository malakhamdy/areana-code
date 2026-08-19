"""PaddleOCR 3.x adapter using the Arabic PP-OCRv5 recognizer."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ..config import OCRConfig
from ..schemas import CoordinateSpace, OCRSpan


class PaddleOCREngine:
    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig.from_env()
        self.device = resolve_device(self.config.device)
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Run: pip install -r requirements.txt"
            ) from exc
        try:
            self._pipeline = PaddleOCR(
                text_detection_model_name=self.config.detection_model,
                text_recognition_model_name=self.config.recognition_model,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize PaddleOCR. Check model downloads, Paddle/PaddleOCR "
                f"compatibility, and device settings. Original error: {exc}"
            ) from exc
        self.name = (
            f"PaddleOCR[{self.config.detection_model}+"
            f"{self.config.recognition_model};{self.device}]"
        )

    def recognize(self, image_rgb: np.ndarray) -> list[OCRSpan]:
        predictions = self._pipeline.predict(image_rgb)
        spans: list[OCRSpan] = []
        for prediction in predictions:
            spans.extend(_parse_prediction(prediction, self.config.score_threshold))
        return spans


@lru_cache(maxsize=2)
def get_paddle_engine(
    device: str = "auto",
    detection_model: str = "PP-OCRv5_server_det",
    recognition_model: str = "arabic_PP-OCRv5_mobile_rec",
) -> PaddleOCREngine:
    """Process singleton; Streamlit wraps this in st.cache_resource as well."""
    return PaddleOCREngine(
        OCRConfig(
            device=device,
            detection_model=detection_model,
            recognition_model=recognition_model,
        )
    )


def resolve_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import paddle

        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu:0"
    except Exception:
        pass
    return "cpu"


def _parse_prediction(prediction: Any, threshold: float) -> list[OCRSpan]:
    # Paddle 3.x exposes a JSON-compatible payload; exact wrapper changed between 3.x minors.
    payload: Any = None
    for attribute in ("json", "to_dict", "res"):
        if hasattr(prediction, attribute):
            value = getattr(prediction, attribute)
            try:
                payload = value() if callable(value) else value
            except Exception:
                continue
            if payload is not None:
                break
    if payload is None and isinstance(prediction, dict):
        payload = prediction
    if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
        payload = payload["res"]
    if isinstance(payload, dict):
        texts = payload.get("rec_texts")
        if texts is None:
            texts = payload.get("texts")
        if texts is None:
            texts = []
        scores = payload.get("rec_scores")
        if scores is None:
            scores = payload.get("scores")
        if scores is None:
            scores = []
        polygons = []
        for key in ("rec_polys", "dt_polys", "text_boxes"):
            value = payload.get(key)
            if value is not None and len(value) > 0:
                polygons = value
                break
        spans: list[OCRSpan] = []
        for index, text in enumerate(texts):
            score = float(scores[index]) if index < len(scores) else 0.0
            if not str(text).strip() or score < threshold:
                continue
            polygon = _to_list(polygons[index]) if index < len(polygons) else None
            spans.append(
                OCRSpan(
                    text=str(text),
                    confidence=score,
                    bbox=polygon,
                    coordinate_space=CoordinateSpace.FIELD_CROP,
                )
            )
        return spans

    # PaddleOCR 2.x compatibility: [[box, (text, confidence)], ...].
    spans = []
    nested = prediction
    if isinstance(nested, list) and len(nested) == 1 and isinstance(nested[0], list):
        nested = nested[0]
    if isinstance(nested, list):
        for item in nested:
            try:
                box, recognized = item
                text, score = recognized
                score = float(score)
                if str(text).strip() and score >= threshold:
                    spans.append(OCRSpan(str(text), score, _to_list(box)))
            except (TypeError, ValueError):
                continue
    return spans


def _to_list(value: Any) -> list[list[float]] | None:
    try:
        array = np.asarray(value, dtype=float).reshape(-1, 2)
        return array.tolist()
    except Exception:
        return None
