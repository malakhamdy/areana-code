"""FastAPI website for Arabic-first Egyptian National ID extraction.

Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import base64
import threading
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from egyptian_id_ocr.config import OCRConfig
from egyptian_id_ocr.image_io import ImageValidationError
from egyptian_id_ocr.ocr.base import OCREngine
from egyptian_id_ocr.ocr.paddle_engine import get_paddle_engine
from egyptian_id_ocr.ocr.tesseract_js_engine import TesseractJSEngine
from egyptian_id_ocr.pipeline import EgyptianIDPipeline, PipelineOutput
from egyptian_id_ocr.privacy import redacted_result

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/x-ms-bmp",
}
ALLOWED_DETECTORS = {"PP-OCRv5_server_det", "PP-OCRv5_mobile_det"}


@dataclass(frozen=True)
class EngineBundle:
    engine: OCREngine
    provider: str
    primary_model: str
    detail: str


class ModelManager:
    """Load OCR once and cache failures/fallbacks across HTTP requests."""

    def __init__(self) -> None:
        self._bundles: dict[tuple[str, str], EngineBundle] = {}
        self._fallback: TesseractJSEngine | None = None
        self._lock = threading.Lock()

    def get(self, detector: str, device: str = "auto") -> EngineBundle:
        key = (detector, device)
        with self._lock:
            cached = self._bundles.get(key)
            if cached:
                return cached
            model_name = "arabic_PP-OCRv5_mobile_rec"
            try:
                engine = get_paddle_engine(device, detector, model_name)
                bundle = EngineBundle(
                    engine=engine,
                    provider="paddleocr",
                    primary_model=f"{detector} + {model_name}",
                    detail="Arabic PaddleOCR primary model is active.",
                )
            except Exception as exc:
                if self._fallback is None:
                    self._fallback = TesseractJSEngine()
                bundle = EngineBundle(
                    engine=self._fallback,
                    provider="tesseract_js_fallback",
                    primary_model=f"{detector} + {model_name}",
                    detail=(
                        "PaddleOCR weights were unavailable in this runtime; the local Arabic "
                        "availability fallback is active. Primary error: " + _short_error(exc)
                    ),
                )
            self._bundles[key] = bundle
            return bundle

    def close(self) -> None:
        with self._lock:
            if self._fallback:
                self._fallback.close()
                self._fallback = None
            self._bundles.clear()


model_manager = ModelManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    model_manager.close()


app = FastAPI(
    title="Egyptian ID Arabic OCR",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=STATIC), name="assets")


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path.startswith("/api/docs"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: https://fastapi.tiangolo.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "script-src 'self' https://cdn.jsdelivr.net; frame-ancestors 'self'"
        )
    else:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; font-src 'self'; frame-ancestors 'self'"
        )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", include_in_schema=False)
def homepage() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "egyptian-id-arabic-ocr",
        "version": app.version,
        "privacy": "uploads are processed in memory and are not persisted",
    }


@app.post("/api/analyze")
async def analyze(
    files: list[UploadFile] = File(...),
    detector: str = Form("PP-OCRv5_server_det"),
    variants: int = Form(3),
    device: str = Form("auto"),
) -> JSONResponse:
    if not 1 <= len(files) <= 2:
        raise HTTPException(400, "Upload one or two card images.")
    if detector not in ALLOWED_DETECTORS:
        raise HTTPException(400, "Unsupported text detector.")
    if variants not in range(1, 5):
        raise HTTPException(400, "OCR passes must be between 1 and 4.")
    if device not in {"auto", "cpu", "gpu:0"}:
        raise HTTPException(400, "Unsupported compute device.")

    uploads: list[bytes] = []
    for uploaded in files:
        if uploaded.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(415, "Only JPEG, PNG, WebP, and BMP images are supported.")
        data = await uploaded.read(MAX_UPLOAD_BYTES + 1)
        await uploaded.close()
        if not data:
            raise HTTPException(400, "An uploaded image is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Each image must be 20 MB or smaller.")
        uploads.append(data)

    try:
        bundle = await run_in_threadpool(model_manager.get, detector, device)
        config = OCRConfig(
            device=device,
            detection_model=detector,
            recognition_model="arabic_PP-OCRv5_mobile_rec",
            max_variants_per_field=variants,
        )
        pipeline = EgyptianIDPipeline(bundle.engine, config)
        documents = []
        for index, data in enumerate(uploads, start=1):
            output = await run_in_threadpool(pipeline.process, data)
            documents.append(_serialize_output(index, output))
    except ImageValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(503, _short_error(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            500,
            f"Document processing failed safely ({type(exc).__name__}). No upload was saved.",
        ) from exc

    return JSONResponse(
        {
            "ok": True,
            "engine": {
                "provider": bundle.provider,
                "name": bundle.engine.name,
                "primary_model": bundle.primary_model,
                "detail": bundle.detail,
            },
            "document_count": len(documents),
            "documents": documents,
        },
        headers={"Cache-Control": "no-store"},
    )


def _serialize_output(index: int, output: PipelineOutput) -> dict[str, Any]:
    result = output.result.to_dict()
    artifacts = output.artifacts
    return {
        "index": index,
        "result": result,
        "safe_result": redacted_result(result),
        "artifacts": {
            "original": _image_data_url(artifacts.original_image, quality=88),
            "processing_canvas": _image_data_url(artifacts.processing_image, quality=82),
            "card_detection": _image_data_url(artifacts.card_detection_overlay, quality=86),
            "canonical_card": _image_data_url(artifacts.canonical_card, quality=90),
            "field_localization": _image_data_url(
                artifacts.field_localization_overlay, quality=90
            ),
            "crops": {
                name: _image_data_url(image, quality=94)
                for name, image in artifacts.crops.items()
            },
            "preprocessed": {
                field: {
                    variant: _image_data_url(image, quality=92)
                    for variant, image in variants.items()
                }
                for field, variants in artifacts.preprocessed_crops.items()
            },
        },
    }


def _image_data_url(rgb_image, *, quality: int) -> str:
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not success:
        return ""
    value = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{value}"


def _short_error(exc: Exception, limit: int = 280) -> str:
    message = " ".join(str(exc).split())
    return (message[: limit - 1] + "…") if len(message) > limit else message
