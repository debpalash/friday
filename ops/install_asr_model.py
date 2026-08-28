#!/usr/bin/env python3
"""Download and install Friday's exact pinned Parakeet ASR checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


REPO = Path(__file__).resolve().parents[1]
MODEL_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
ARCHIVE_NAME = MODEL_NAME + ".tar.bz2"
ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    + ARCHIVE_NAME
)
ARCHIVE_SIZE = 487_170_055
ARCHIVE_SHA256 = "5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf"
ASSETS = {
    "encoder.int8.onnx": (
        652_184_281,
        "acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247",
    ),
    "decoder.int8.onnx": (
        11_845_275,
        "179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e",
    ),
    "joiner.int8.onnx": (
        6_355_277,
        "3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3",
    ),
    "tokens.txt": (
        93_939,
        "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
    ),
}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def valid_model(target: Path) -> bool:
    try:
        return (
            target.is_dir()
            and not target.is_symlink()
            and all(
                (target / name).is_file()
                and not (target / name).is_symlink()
                and (target / name).stat().st_size == size
                and _digest(target / name) == digest
                for name, (size, digest) in ASSETS.items()
            )
        )
    except OSError:
        return False


def _safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    total = 0
    for member in members:
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != MODEL_NAME
            or member.issym()
            or member.islnk()
            or member.isdev()
        ):
            raise RuntimeError(f"unsafe ASR archive member: {member.name}")
        if member.isfile():
            total += member.size
            if total > 800_000_000:
                raise RuntimeError("ASR archive expands beyond its bound")
    return members


def _download(archive: Path) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required to download the ASR model")
    archive.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    subprocess.run(
        [
            curl,
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--retry",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "20",
            "--output",
            str(partial),
            ARCHIVE_URL,
        ],
        check=True,
    )
    if partial.stat().st_size != ARCHIVE_SIZE or _digest(partial) != ARCHIVE_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError("ASR archive size or SHA-256 did not match its pin")
    os.replace(partial, archive)
    os.chmod(archive, 0o600)


def install(model_root: Path, cache_root: Path) -> Path:
    target = model_root / MODEL_NAME
    if target.exists() or target.is_symlink():
        if valid_model(target):
            print(f"verified existing ASR model: {target}")
            return target
        raise RuntimeError(f"refusing to overwrite an invalid ASR model: {target}")

    archive_path = cache_root / ARCHIVE_NAME
    if not (
        archive_path.is_file()
        and archive_path.stat().st_size == ARCHIVE_SIZE
        and _digest(archive_path) == ARCHIVE_SHA256
    ):
        archive_path.unlink(missing_ok=True)
        _download(archive_path)

    model_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{MODEL_NAME}-install-", dir=model_root)
    )
    try:
        with tarfile.open(archive_path, "r:bz2") as archive:
            members = _safe_members(archive)
            archive.extractall(staging_parent, members=members, filter="data")
        extracted = staging_parent / MODEL_NAME
        if not valid_model(extracted):
            raise RuntimeError("extracted ASR model failed its file pins")
        os.replace(extracted, target)
        directory = os.open(model_root, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise
    shutil.rmtree(staging_parent, ignore_errors=True)
    print(f"installed ASR model: {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=REPO / "models")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    target = args.model_root.expanduser().resolve() / MODEL_NAME
    if args.verify_only:
        if not valid_model(target):
            raise RuntimeError(f"ASR model is missing or invalid: {target}")
        print(f"verified ASR model: {target}")
        return 0
    cache = args.cache_root or Path(
        os.environ.get(
            "XDG_CACHE_HOME", str(Path.home() / ".cache")
        )
    ) / "friday" / "downloads"
    install(args.model_root.expanduser().resolve(), cache.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
