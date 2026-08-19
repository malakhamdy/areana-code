"""Offline Arabic Tesseract.js fallback for restricted model-host networks.

PaddleOCR remains the primary engine. This adapter exists so geometry/crop testing can
still return local OCR when Paddle model weights cannot be fetched. It keeps one Node
worker alive, sends PNGs through stdin, and never writes identity crops to disk.
"""
from __future__ import annotations

import atexit
import base64
import json
from pathlib import Path
import select
import shutil
import subprocess
import threading
from typing import Any

import cv2
import numpy as np

from ..schemas import OCRSpan


class TesseractJSEngine:
    name = "Tesseract.js[ara+numeric;CPU;availability_fallback]"

    def __init__(self, *, timeout_seconds: int = 90) -> None:
        node = shutil.which("node")
        root = Path(__file__).resolve().parents[2]
        worker_script = root / "scripts" / "tesseract_worker.cjs"
        language_data = root / "node_modules" / "@tesseract.js-data" / "ara"
        if not node:
            raise RuntimeError("Node.js is required for the offline OCR fallback.")
        if not worker_script.exists() or not language_data.exists():
            raise RuntimeError(
                "Tesseract.js fallback dependencies are missing. Run: npm ci"
            )
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._request_id = 0
        self._process = subprocess.Popen(
            [node, str(worker_script)],
            cwd=str(root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        atexit.register(self.close)

    def recognize(self, image_rgb: np.ndarray) -> list[OCRSpan]:
        success, encoded = cv2.imencode(
            ".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        )
        if not success:
            return []
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("The local Tesseract.js worker stopped unexpectedly.")
            self._request_id += 1
            request = {
                "id": self._request_id,
                "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            }
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
            ready, _, _ = select.select(
                [self._process.stdout], [], [], self.timeout_seconds
            )
            if not ready:
                raise TimeoutError("The local Arabic fallback OCR timed out.")
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("The local Arabic fallback OCR returned no response.")
            response: dict[str, Any] = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Tesseract.js OCR failed."))
        text = str(response.get("text", "")).strip()
        if not text:
            return []
        return [
            OCRSpan(
                text=text,
                confidence=float(np.clip(response.get("confidence", 0.0), 0, 1)),
                bbox=None,
            )
        ]

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin:
                process.stdin.write('{"command":"terminate"}\n')
                process.stdin.flush()
            process.wait(timeout=3)
        except Exception:
            process.kill()
