#!/usr/bin/env bash
# Install for a Linux server without libGL. PaddleX declares desktop OpenCV by
# distribution name, so preserve its metadata but place headless cv2 binaries last.
set -euo pipefail
python -m pip install -r requirements.txt
python -m pip install --no-deps --force-reinstall \
  opencv-contrib-python-headless==4.10.0.84
if command -v npm >/dev/null 2>&1; then
  npm ci
fi
python - <<'PY'
import cv2
print(f"Headless OpenCV ready: {cv2.__version__}")
PY
