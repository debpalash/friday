#!/usr/bin/env python3
"""Atomically install Friday's pinned Silero voice-activity ONNX model."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.vad import (VAD_MODEL_DIRECTORY, VAD_MODEL_FILE,  # noqa: E402
                             VAD_MODEL_REVISION, VAD_MODEL_SHA256,
                             VAD_MODEL_SIZE, VAD_MODEL_URL)
from friday_host import fs  # noqa: E402

TARGET = REPO / "models" / VAD_MODEL_DIRECTORY


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _valid(path: Path) -> bool:
    try:
        return (path.is_file() and not path.is_symlink()
                and path.stat().st_size == VAD_MODEL_SIZE
                and _digest(path) == VAD_MODEL_SHA256)
    except OSError:
        return False


def main() -> int:
    if TARGET.exists():
        if (TARGET.is_dir() and not TARGET.is_symlink()
                and _valid(TARGET / VAD_MODEL_FILE)):
            print(f"verified existing VAD model: {TARGET}")
            return 0
        raise RuntimeError(
            f"refusing to overwrite an invalid existing model directory: {TARGET}")
    TARGET.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{TARGET.name}-install-", dir=TARGET.parent))
    try:
        timeout = httpx.Timeout(connect=20, read=120, write=20, pool=20)
        path = staging / VAD_MODEL_FILE
        digest = hashlib.sha256()
        observed = 0
        with httpx.Client(
                timeout=timeout, follow_redirects=True, trust_env=False,
                headers={"User-Agent": "Friday-pinned-model-installer/1"}) as client:
            with client.stream("GET", VAD_MODEL_URL) as response:
                response.raise_for_status()
                if response.url.scheme != "https":
                    raise RuntimeError("VAD download left HTTPS")
                with path.open("xb") as output:
                    for chunk in response.iter_bytes(1024 * 1024):
                        observed += len(chunk)
                        if observed > VAD_MODEL_SIZE:
                            raise RuntimeError("VAD asset exceeded its pin")
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
        if observed != VAD_MODEL_SIZE or digest.hexdigest() != VAD_MODEL_SHA256:
            raise RuntimeError("VAD asset pin mismatch")
        fs.chmod_private(path, 0o644)
        manifest = staging / "FRIDAY_MODEL_PIN"
        manifest.write_text(
            f"model=snakers4/silero-vad\nrevision={VAD_MODEL_REVISION}\n",
            encoding="utf-8")
        fs.chmod_private(manifest, 0o644)
        os.replace(staging, TARGET)
        fs.fsync_directory(TARGET.parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"installed VAD model: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
