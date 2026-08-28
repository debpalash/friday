"""Bounded loading for Friday's static controller interface."""

from __future__ import annotations

import os
import stat
from pathlib import Path


MAX_FRONTEND_BYTES = 512 * 1024


class FrontendAssetError(RuntimeError):
    """Raised when the local frontend asset is missing or unsafe."""


def load_frontend(
    path: Path, *, max_bytes: int = MAX_FRONTEND_BYTES,
) -> str:
    """Read one regular, non-symlinked UTF-8 HTML asset within a size bound."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, TypeError) as exc:
        raise FrontendAssetError(f"frontend asset is unavailable: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FrontendAssetError("frontend asset must be a regular file")
        if metadata.st_size > max_bytes:
            raise FrontendAssetError("frontend asset exceeds the size limit")
        raw = os.read(descriptor, max_bytes + 1)
        if len(raw) > max_bytes:
            raise FrontendAssetError("frontend asset exceeds the size limit")
    finally:
        os.close(descriptor)
    try:
        rendered = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FrontendAssetError("frontend asset is not valid UTF-8") from exc
    if not rendered.lstrip().lower().startswith("<!doctype html>"):
        raise FrontendAssetError("frontend asset is not an HTML document")
    return rendered
