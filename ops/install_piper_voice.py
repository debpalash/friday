#!/usr/bin/env python3
"""Atomically install Friday's exact pinned low-latency Piper voice."""

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

from friday_core.speech import (
    PIPER_ASSETS,
    PIPER_VOICE_DIRECTORY,
    PIPER_VOICE_REVISION,
)


MODEL_ID = "rhasspy/piper-voices"
VOICE_PATH = "en/en_US/kristin/medium"
TARGET = REPO / "models" / PIPER_VOICE_DIRECTORY


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _valid(path: Path, size: int, digest: str) -> bool:
    try:
        return (path.is_file() and not path.is_symlink()
                and path.stat().st_size == size and _digest(path) == digest)
    except OSError:
        return False


def main() -> int:
    if TARGET.exists():
        if (TARGET.is_dir() and not TARGET.is_symlink()
                and all(_valid(TARGET / name, *proof)
                        for name, proof in PIPER_ASSETS.items())):
            print(f"verified existing Piper voice: {TARGET}")
            return 0
        raise RuntimeError(
            f"refusing to overwrite an invalid existing voice directory: {TARGET}")
    TARGET.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{TARGET.name}-install-", dir=TARGET.parent))
    try:
        timeout = httpx.Timeout(connect=20, read=120, write=20, pool=20)
        with httpx.Client(
                timeout=timeout, follow_redirects=True, trust_env=False,
                headers={"User-Agent": "Friday-pinned-voice-installer/1"}) as client:
            for name, (expected_size, expected_digest) in PIPER_ASSETS.items():
                url = (f"https://huggingface.co/{MODEL_ID}/resolve/"
                       f"{PIPER_VOICE_REVISION}/{VOICE_PATH}/{name}")
                path = staging / name
                digest = hashlib.sha256()
                observed = 0
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    if response.url.scheme != "https":
                        raise RuntimeError("Piper voice download left HTTPS")
                    with path.open("xb") as output:
                        for chunk in response.iter_bytes(1024 * 1024):
                            observed += len(chunk)
                            if observed > expected_size:
                                raise RuntimeError(
                                    f"Piper asset exceeded its pin: {name}")
                            digest.update(chunk)
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                if (observed != expected_size
                        or digest.hexdigest() != expected_digest):
                    raise RuntimeError(f"Piper asset pin mismatch: {name}")
                os.chmod(path, 0o644)
                print(f"verified {name} ({observed} bytes)")
        manifest = staging / "FRIDAY_MODEL_PIN"
        manifest.write_text(
            f"model={MODEL_ID}\nrevision={PIPER_VOICE_REVISION}\n"
            f"voice={VOICE_PATH}\n", encoding="utf-8")
        os.chmod(manifest, 0o644)
        os.replace(staging, TARGET)
        directory = os.open(TARGET.parent, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"installed Piper voice: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
