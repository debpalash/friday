#!/usr/bin/env python3
"""Atomically install Friday's pinned local multilingual embedding checkpoint."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import httpx


REPO = Path(__file__).resolve().parents[1]
MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
TARGET = REPO / "models" / f"multilingual-e5-small-{REVISION[:8]}"
ASSETS = {
    "config.json": (
        655, "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959"),
    "model.safetensors": (
        470_641_600,
        "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"),
    "sentencepiece.bpe.model": (
        5_069_051,
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"),
    "special_tokens_map.json": (
        167, "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7"),
    "tokenizer.json": (
        17_082_730,
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
    "tokenizer_config.json": (
        443, "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b"),
}


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
                        for name, proof in ASSETS.items())):
            print(f"verified existing embedding model: {TARGET}")
            return 0
        raise RuntimeError(
            f"refusing to overwrite an invalid existing model directory: {TARGET}")
    TARGET.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{TARGET.name}-install-", dir=TARGET.parent))
    try:
        timeout = httpx.Timeout(connect=20, read=120, write=20, pool=20)
        with httpx.Client(
                timeout=timeout, follow_redirects=True, trust_env=False,
                headers={"User-Agent": "Friday-pinned-model-installer/1"}) as client:
            for name, (expected_size, expected_digest) in ASSETS.items():
                url = (f"https://huggingface.co/{MODEL_ID}/resolve/"
                       f"{REVISION}/{name}")
                path = staging / name
                digest = hashlib.sha256()
                observed = 0
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    if response.url.scheme != "https":
                        raise RuntimeError("embedding download left HTTPS")
                    with path.open("xb") as output:
                        for chunk in response.iter_bytes(1024 * 1024):
                            observed += len(chunk)
                            if observed > expected_size:
                                raise RuntimeError(
                                    f"embedding asset exceeded its pin: {name}")
                            digest.update(chunk)
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                if (observed != expected_size
                        or digest.hexdigest() != expected_digest):
                    raise RuntimeError(f"embedding asset pin mismatch: {name}")
                os.chmod(path, 0o644)
                print(f"verified {name} ({observed} bytes)")
        manifest = staging / "FRIDAY_MODEL_PIN"
        manifest.write_text(
            f"model={MODEL_ID}\nrevision={REVISION}\n", encoding="utf-8")
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
    print(f"installed embedding model: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
