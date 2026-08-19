"""Download the optional Kaggle fixtures into ignored private storage.

Requires Kaggle credentials configured outside this repository. Real identity images
must never be committed, logged, or used in public CI.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

DATASET = "nagwaahmed/egyptian-national-ids"
TARGET = Path("data/private/kaggle_egyptian_ids")


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(TARGET),
        "--unzip",
    ]
    print(f"Downloading {DATASET} into ignored path {TARGET} …")
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("Install the Kaggle CLI and configure credentials first.") from exc
    print("Done. Review license/consent and redact fixtures before any use in tests.")


if __name__ == "__main__":
    main()
