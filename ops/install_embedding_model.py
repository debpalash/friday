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
    "onnx/model.onnx": (
        470_268_510,
        "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665"),
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


def _download(client: "httpx.Client", name: str, expected_size: int,
              expected_digest: str, path: Path) -> None:
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if observed != expected_size or digest.hexdigest() != expected_digest:
        raise RuntimeError(f"embedding asset pin mismatch: {name}")
    os.chmod(path, 0o644)
    print(f"verified {name} ({observed} bytes)")


def _client() -> "httpx.Client":
    timeout = httpx.Timeout(connect=20, read=120, write=20, pool=20)
    return httpx.Client(
        timeout=timeout, follow_redirects=True, trust_env=False,
        headers={"User-Agent": "Friday-pinned-model-installer/1"})


def _fsync_parent(path: Path) -> None:
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> int:
    if TARGET.exists():
        if TARGET.is_symlink() or not TARGET.is_dir():
            raise RuntimeError(
                f"refusing to overwrite an invalid existing model directory: {TARGET}")
        missing = []
        for name, (size, digest) in ASSETS.items():
            path = TARGET / name
            if path.exists():
                if not _valid(path, size, digest):
                    raise RuntimeError(
                        f"refusing to overwrite an invalid existing model asset: {path}")
            else:
                missing.append(name)
        if not missing:
            print(f"verified existing embedding model: {TARGET}")
            return 0
        # Add assets introduced by a newer release next to a verified install.
        with _client() as client:
            for name in missing:
                size, digest = ASSETS[name]
                staged = TARGET / Path(name).parent / f".{Path(name).name}.part"
                if staged.exists():
                    staged.unlink()
                _download(client, name, size, digest, staged)
                os.replace(staged, TARGET / name)
                _fsync_parent(TARGET / name)
        print(f"completed embedding model: {TARGET}")
        return 0
    TARGET.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{TARGET.name}-install-", dir=TARGET.parent))
    try:
        with _client() as client:
            for name, (expected_size, expected_digest) in ASSETS.items():
                _download(client, name, expected_size, expected_digest,
                          staging / name)
        manifest = staging / "FRIDAY_MODEL_PIN"
        manifest.write_text(
            f"model={MODEL_ID}\nrevision={REVISION}\n", encoding="utf-8")
        os.chmod(manifest, 0o644)
        os.replace(staging, TARGET)
        _fsync_parent(TARGET)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"installed embedding model: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
