#!/usr/bin/env python3
"""Atomically install one pinned Qwen3 checkpoint for a portable engine.

Every file is downloaded from the exact Hugging Face revision recorded in
``friday_core/engine_assets.json`` and verified by size and SHA-256 before
the directory is published under the shared model root.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.engine_assets import ModelAsset, model_asset, model_assets  # noqa: E402
from friday_host import fs  # noqa: E402


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


def _download(client: httpx.Client, asset: ModelAsset, name: str,
              expected_size: int, expected_digest: str, path: Path) -> None:
    url = f"https://huggingface.co/{asset.repo}/resolve/{asset.revision}/{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    observed = 0
    with client.stream("GET", url) as response:
        response.raise_for_status()
        if response.url.scheme != "https":
            raise RuntimeError("model download left HTTPS")
        with path.open("xb") as output:
            for chunk in response.iter_bytes(4 * 1024 * 1024):
                observed += len(chunk)
                if observed > expected_size:
                    raise RuntimeError(f"model file exceeded its pin: {name}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    if observed != expected_size or digest.hexdigest() != expected_digest:
        raise RuntimeError(f"model file pin mismatch: {name}")
    fs.chmod_private(path, 0o600)
    print(f"verified {name} ({observed} bytes)", flush=True)


def install(asset: ModelAsset, model_root: Path, *,
            client: httpx.Client | None = None) -> Path:
    target = model_root / asset.directory
    if target.exists():
        if (target.is_dir() and not target.is_symlink()
                and all(_valid(target / name, size, digest)
                        for name, size, digest in asset.files)):
            print(f"verified existing model: {target}")
            return target
        raise RuntimeError(
            f"refusing to overwrite an invalid existing model directory: {target}")
    model_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-install-", dir=model_root))
    try:
        timeout = httpx.Timeout(connect=20, read=300, write=20, pool=20)
        owned_client = client or httpx.Client(
            timeout=timeout, follow_redirects=True, trust_env=False,
            headers={"User-Agent": "Friday-pinned-model-installer/1"})
        with owned_client as client:
            for name, size, digest in asset.files:
                _download(client, asset, name, size, digest, staging / name)
        manifest = staging / "FRIDAY_MODEL_PIN"
        manifest.write_text(
            f"model={asset.repo}\nrevision={asset.revision}\nasset={asset.key}\n",
            encoding="utf-8")
        fs.chmod_private(manifest, 0o600)
        os.replace(staging, target)
        fs.fsync_directory(model_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"installed model: {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--asset", required=True,
                        help="asset key from friday_core/engine_assets.json")
    parser.add_argument("--model-root", type=Path, default=REPO / "models")
    parser.add_argument("--list", action="store_true", help="list asset keys and exit")
    args = parser.parse_args()
    if args.list:
        for item in model_assets():
            print(f"{item.key}\t{item.engine}\t{item.repo}@{item.revision[:8]}\t"
                  f"{item.weights_bytes / 1024 ** 3:.1f} GiB")
        return 0
    try:
        asset = model_asset(args.asset)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    install(asset, args.model_root.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
