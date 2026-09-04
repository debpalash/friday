#!/usr/bin/env python3
"""Install the pinned llama-server release binary for this host.

The archive for the host's operating system, architecture, and compute
backend is downloaded from the exact llama.cpp release recorded in
``friday_core/engine_assets.json``, verified by size and SHA-256, extracted
with path validation into a staging directory, and published atomically
under ``<runtime root>/llama-server/<tag>-<backend>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.engine_assets import (EngineBinary, engine_backends,  # noqa: E402
                                       llama_server_binary)
from friday_core.runtime_engines import runtime_root  # noqa: E402
from friday_host import fs  # noqa: E402
from friday_host.host import current_host  # noqa: E402

MAX_MEMBERS = 4_000
MAX_MEMBER_BYTES = 2 * 1024 ** 3
MAX_TOTAL_BYTES = 6 * 1024 ** 3


def _download(client: httpx.Client, url: str, size: int, digest: str,
              path: Path) -> None:
    observed = 0
    hasher = hashlib.sha256()
    with client.stream("GET", url) as response:
        response.raise_for_status()
        if response.url.scheme != "https":
            raise RuntimeError("engine download left HTTPS")
        with path.open("xb") as output:
            for chunk in response.iter_bytes(4 * 1024 * 1024):
                observed += len(chunk)
                if observed > size:
                    raise RuntimeError(f"engine archive exceeded its pin: {path.name}")
                hasher.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    if observed != size or hasher.hexdigest() != digest:
        raise RuntimeError(f"engine archive pin mismatch: {path.name}")
    print(f"verified {path.name} ({observed} bytes)", flush=True)


def _safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    parts = path.parts
    if (not parts or path.is_absolute() or ".." in parts
            or any(part in {"", "."} for part in parts)):
        raise RuntimeError(f"unsafe archive member path: {name}")
    return path


def _extract(archive: Path, destination: Path) -> None:
    """Extract regular files only, flattening one leading directory."""
    members = 0
    total = 0
    entries: list[tuple[PurePosixPath, int, bool, object]] = []
    if archive.suffix == ".zip":
        source = zipfile.ZipFile(archive)
        for info in source.infolist():
            if info.is_dir():
                continue
            mode = (info.external_attr >> 16) & 0o777
            entries.append((_safe_relative(info.filename), info.file_size,
                            bool(mode & 0o111), info))
    else:
        source = tarfile.open(archive, "r:gz")
        for info in source:
            if info.isdir():
                continue
            if not info.isreg():
                raise RuntimeError("engine archive contains a link or special member")
            entries.append((_safe_relative(info.name), info.size,
                            bool(info.mode & 0o111), info))
    with source:
        roots = {entry[0].parts[0] for entry in entries}
        strip = 1 if len(roots) == 1 and all(len(e[0].parts) > 1 for e in entries) else 0
        for relative, size, executable, info in entries:
            members += 1
            total += size
            if members > MAX_MEMBERS or size > MAX_MEMBER_BYTES or total > MAX_TOTAL_BYTES:
                raise RuntimeError("engine archive exceeds the extraction limits")
            target = destination.joinpath(*relative.parts[strip:])
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(source, zipfile.ZipFile):
                stream = source.open(info)
            else:
                stream = source.extractfile(info)
                if stream is None:
                    raise RuntimeError("engine archive member is unreadable")
            with stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output, 4 * 1024 * 1024)
            if executable and os.name == "posix":
                target.chmod(0o755)
            elif os.name == "posix":
                target.chmod(0o644)


def install(binary: EngineBinary, root: Path, *, cache: Path,
            client: httpx.Client | None = None) -> Path:
    target = root / "llama-server" / binary.directory
    executable = target / binary.executable
    if target.exists():
        pin = target / "FRIDAY_ENGINE_PIN"
        if executable.is_file() and pin.is_file() and binary.sha256 in pin.read_text():
            print(f"verified existing llama-server: {target}")
            return executable
        raise RuntimeError(
            f"refusing to overwrite an invalid existing engine directory: {target}")
    (root / "llama-server").mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{binary.directory}-install-",
                                    dir=root / "llama-server"))
    try:
        timeout = httpx.Timeout(connect=20, read=300, write=20, pool=20)
        owned_client = client or httpx.Client(
            timeout=timeout, follow_redirects=True, trust_env=False,
            headers={"User-Agent": "Friday-engine-installer/1"})
        with owned_client as client:
            archives = [(binary.name, binary.url, binary.size, binary.sha256)]
            archives += [(name, url, size, digest) for name, url, size, digest in binary.extra]
            for name, url, size, digest in archives:
                archive = cache / name
                if archive.exists():
                    archive.unlink()
                _download(client, url, size, digest, archive)
                _extract(archive, staging)
        if not (staging / binary.executable).is_file():
            found = sorted(str(p.relative_to(staging)) for p in staging.rglob(binary.executable))
            if not found:
                raise RuntimeError("engine archive does not contain llama-server")
            nested = staging / found[0]
            for item in nested.parent.iterdir():
                shutil.move(str(item), staging / item.name)
        pin = staging / "FRIDAY_ENGINE_PIN"
        pin.write_text(f"engine=llama-server\ntag={binary.tag}\nbackend={binary.backend}\n"
                       f"archive={binary.name}\nsha256={binary.sha256}\n", encoding="utf-8")
        fs.chmod_private(pin, 0o600)
        os.replace(staging, target)
        fs.fsync_directory(target.parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if os.name == "posix":
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    try:
        version = subprocess.run([str(executable), "--version"], capture_output=True,
                                 text=True, timeout=60)
        banner = (version.stdout or version.stderr).strip().splitlines()
        print(f"installed llama-server: {target} ({banner[0] if banner else 'no banner'})")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"installed llama-server does not run: {exc}") from exc
    return executable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="auto",
                        help="cpu, cuda, vulkan, rocm, or metal (default: from the runtime profile)")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    args = parser.parse_args()
    host = current_host()
    backend = args.backend
    if backend == "auto":
        from friday_core.hardware import detect_hardware, select_runtime_profile  # noqa: PLC0415

        profile = select_runtime_profile(
            detect_hardware(), environment={**os.environ, "FRIDAY_LLM_ENGINE": "llama_server"})
        backend = profile.engine_backend
    available = engine_backends("llama_server", host.os, host.arch)
    if backend not in available:
        raise SystemExit(f"no llama-server build for {host.os}/{host.arch}/{backend}; "
                         f"available: {', '.join(sorted(available)) or 'none'}")
    binary = llama_server_binary(host.os, host.arch, backend)
    root = (args.runtime_root or runtime_root()).expanduser().resolve()
    cache = (args.cache_root or (root.parent / "cache" / "downloads")).expanduser()
    install(binary, root, cache=cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
