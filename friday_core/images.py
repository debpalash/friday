"""Bounded OCR for exact-granted local raster images.

This module deliberately exposes text recognition, not general visual
understanding. Untrusted raster decoders run in a networkless resource-limited
Bubblewrap boundary; the broker itself parses only the small PNG/JPEG headers
needed to enforce dimensions before dispatch.
"""

from __future__ import annotations

import hashlib
import os
import re
import signal
import stat
import subprocess
import tempfile
from pathlib import PurePosixPath
from typing import Any


MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_DIMENSION = 20_000
MAX_IMAGE_PIXELS = 40_000_000
MAX_OCR_BYTES = 1_000_000
MAX_OCR_CHARS = 250_000
MAX_NATIVE_VISION_SOURCE_BYTES = 16 * 1024 * 1024
MAX_NATIVE_VISION_OUTPUT_BYTES = 16 * 1024 * 1024
SUPPORTED_IMAGE_FORMATS = frozenset({"png", "jpg", "jpeg"})
_JPEG_SOF = frozenset({
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
})
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def image_source_sha256(fd: int, size: int) -> str:
    if (isinstance(size, bool) or not isinstance(size, int)
            or not 0 <= size <= MAX_IMAGE_BYTES):
        raise ValueError("image exceeds the source-size limit")
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(65_536, size - offset), offset)
        if not chunk:
            raise RuntimeError("image became shorter while being hashed")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(fd, 1, size):
        raise RuntimeError("image grew while being hashed")
    return digest.hexdigest()


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    if (len(payload) < 33 or payload[:8] != b"\x89PNG\r\n\x1a\n"
            or payload[8:12] != b"\x00\x00\x00\r"
            or payload[12:16] != b"IHDR"):
        raise ValueError("file extension and PNG header disagree")
    return (int.from_bytes(payload[16:20], "big"),
            int.from_bytes(payload[20:24], "big"))


def _jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    if len(payload) < 4 or payload[:2] != b"\xff\xd8":
        raise ValueError("file extension and JPEG header disagree")
    offset = 2
    while offset < len(payload):
        if payload[offset] != 0xFF:
            raise ValueError("JPEG marker inventory is malformed")
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            break
        marker = payload[offset]
        offset += 1
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(payload):
            break
        length = int.from_bytes(payload[offset:offset + 2], "big")
        if length < 2 or offset + length > len(payload):
            raise ValueError("JPEG segment inventory is malformed")
        if marker in _JPEG_SOF:
            if length < 7:
                raise ValueError("JPEG frame header is malformed")
            height = int.from_bytes(payload[offset + 3:offset + 5], "big")
            width = int.from_bytes(payload[offset + 5:offset + 7], "big")
            return width, height
        if marker == 0xDA:
            break
        offset += length
    raise ValueError("JPEG dimensions are unavailable")


def _image_metadata(fd: int, filename: str, size: int) -> tuple[str, int, int]:
    format_name = PurePosixPath(filename).suffix.casefold().lstrip(".")
    if format_name not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError("unsupported image format")
    payload = os.pread(fd, size, 0)
    if len(payload) != size:
        raise RuntimeError("image became shorter while reading its header")
    if format_name == "png":
        width, height = _png_dimensions(payload)
        canonical_format = "png"
    else:
        width, height = _jpeg_dimensions(payload)
        canonical_format = "jpeg"
    pixels = width * height
    if (not 1 <= width <= MAX_IMAGE_DIMENSION
            or not 1 <= height <= MAX_IMAGE_DIMENSION
            or not 1 <= pixels <= MAX_IMAGE_PIXELS):
        raise ValueError("image dimensions exceed the OCR safety limit")
    return canonical_format, width, height


def _normalize_ocr(value: str, max_chars: int) -> tuple[str, bool]:
    value = _CONTROL.sub(" ", value.replace("\r\n", "\n").replace("\r", "\n"))
    lines: list[str] = []
    blank = False
    for raw in value.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if line:
            lines.append(line)
            blank = False
        elif lines and not blank:
            lines.append("")
            blank = True
    normalized = "\n".join(lines).strip()
    truncated = len(normalized) > max_chars
    return normalized[:max_chars], truncated


def _sandboxed_tesseract(fd: int, suffix: str) -> str:
    if not os.path.isfile("/usr/bin/tesseract"):
        raise RuntimeError("the pinned OCR extractor is unavailable")
    if not os.path.isfile("/usr/bin/bwrap"):
        raise RuntimeError("the OCR sandbox is unavailable")
    if not os.path.isdir("/dev/shm"):
        raise RuntimeError("the private in-memory OCR output filesystem is unavailable")
    output_file = tempfile.TemporaryFile(dir="/dev/shm")
    error_file = tempfile.TemporaryFile(dir="/dev/shm")
    command = [
        "/usr/bin/bwrap", "--die-with-parent", "--new-session", "--unshare-all",
        "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib", "/lib64", "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/tmp", "--clearenv", "--setenv", "HOME", "/tmp",
        "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "OMP_THREAD_LIMIT",
        "4", "--ro-bind-fd", str(fd), f"/input.{suffix}", "--chdir", "/tmp",
        "/usr/bin/prlimit", f"--fsize={MAX_OCR_BYTES}", "--cpu=20",
        "--as=1073741824", "--nofile=64", "--nproc=32", "--",
        "/usr/bin/tesseract", f"/input.{suffix}", "stdout", "-l", "eng",
        "--psm", "3",
    ]
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=output_file,
            stderr=error_file, pass_fds=(fd,), start_new_session=True,
            env={"PATH": "/usr/bin:/bin"})
        try:
            returncode = process.wait(timeout=25)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            finally:
                process.wait()
            raise RuntimeError("image OCR exceeded its time limit") from exc
        size = os.fstat(output_file.fileno()).st_size
        if returncode or size > MAX_OCR_BYTES:
            error_file.seek(0)
            detail = error_file.read(4096).decode("utf-8", errors="replace")
            raise ValueError((detail.strip() or "image OCR failed")[-500:])
        payload = os.pread(output_file.fileno(), size, 0)
    finally:
        error_file.close()
        output_file.close()
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("OCR extractor returned invalid UTF-8") from exc


def _sandboxed_canonical_png(fd: int, suffix: str, max_side: int) -> bytes:
    """Decode and canonicalize an untrusted raster outside Friday's process."""
    if not os.path.isfile("/usr/bin/magick"):
        raise RuntimeError("the pinned image sanitizer is unavailable")
    if not os.path.isfile("/usr/bin/bwrap"):
        raise RuntimeError("the image sanitizer sandbox is unavailable")
    if not os.path.isdir("/dev/shm"):
        raise RuntimeError(
            "the private in-memory image output filesystem is unavailable")
    output_file = tempfile.TemporaryFile(dir="/dev/shm")
    error_file = tempfile.TemporaryFile(dir="/dev/shm")
    sandbox = [
        "/usr/bin/bwrap", "--die-with-parent", "--new-session", "--unshare-all",
        "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib", "/lib64", "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/tmp", "--clearenv", "--setenv", "HOME", "/tmp",
        "--setenv", "PATH", "/usr/bin:/bin", "--setenv",
        "MAGICK_TEMPORARY_PATH", "/tmp",
    ]
    if os.path.isdir("/etc/ImageMagick-7"):
        sandbox.extend([
            "--ro-bind", "/etc/ImageMagick-7", "/etc/ImageMagick-7"])
    command = sandbox + [
        "--ro-bind-fd", str(fd), f"/input.{suffix}", "--chdir", "/tmp",
        "/usr/bin/prlimit", f"--fsize={MAX_NATIVE_VISION_OUTPUT_BYTES}",
        "--cpu=20", "--as=1073741824", "--nofile=64", "--nproc=32", "--",
        "/usr/bin/magick", "-limit", "memory", "256MiB", "-limit", "map",
        "512MiB", "-limit", "disk", "0", "-limit", "thread", "2",
        f"/input.{suffix}[0]", "-auto-orient", "-strip", "-resize",
        f"{max_side}x{max_side}>", "-colorspace", "sRGB", "-depth", "8",
        "PNG:-",
    ]
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=output_file,
            stderr=error_file, pass_fds=(fd,), start_new_session=True,
            env={"PATH": "/usr/bin:/bin"})
        try:
            returncode = process.wait(timeout=25)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            finally:
                process.wait()
            raise RuntimeError("image sanitization exceeded its time limit") from exc
        size = os.fstat(output_file.fileno()).st_size
        if returncode or not 1 <= size <= MAX_NATIVE_VISION_OUTPUT_BYTES:
            error_file.seek(0)
            detail = error_file.read(4096).decode("utf-8", errors="replace")
            raise ValueError((detail.strip() or "image sanitization failed")[-500:])
        payload = os.pread(output_file.fileno(), size, 0)
    finally:
        error_file.close()
        output_file.close()
    if len(payload) != size:
        raise RuntimeError("sanitized image changed while being read")
    return payload


def prepare_native_vision_image(
        fd: int, filename: str, *, max_side: int) -> tuple[bytes, dict[str, Any]]:
    """Return an ephemeral canonical PNG and hash-only source provenance."""
    if (isinstance(max_side, bool) or not isinstance(max_side, int)
            or not 256 <= max_side <= 4096):
        raise ValueError("native-vision maximum side must be between 256 and 4096")
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("only regular image files can be understood")
    if not 1 <= info.st_size <= MAX_NATIVE_VISION_SOURCE_BYTES:
        raise ValueError("image source size is outside the native-vision limit")
    source_hash = image_source_sha256(fd, int(info.st_size))
    source_format, source_width, source_height = _image_metadata(
        fd, filename, int(info.st_size))
    encoded = _sandboxed_canonical_png(fd, source_format, max_side)
    width, height = _png_dimensions(encoded)
    if (not 1 <= width <= max_side or not 1 <= height <= max_side
            or width * height > max_side ** 2):
        raise RuntimeError("image sanitizer exceeded the native-vision dimensions")
    return encoded, {
        "format": "png",
        "sanitizer": "sandboxed-imagemagick",
        "width": width,
        "height": height,
        "pixels": width * height,
        "image_bytes": len(encoded),
        "image_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_format": source_format,
        "source_width": source_width,
        "source_height": source_height,
        "source_pixels": source_width * source_height,
        "source_bytes": int(info.st_size),
        "source_sha256": source_hash,
        "limitations": "single_image_question_answering",
    }


def extract_image_text(fd: int, filename: str, *, max_chars: int = 80_000
                       ) -> dict[str, Any]:
    if (isinstance(max_chars, bool) or not isinstance(max_chars, int)
            or not 1 <= max_chars <= MAX_OCR_CHARS):
        raise ValueError(
            f"OCR character limit must be between 1 and {MAX_OCR_CHARS}")
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("only regular image files can be read")
    if not 1 <= info.st_size <= MAX_IMAGE_BYTES:
        raise ValueError("image source size is outside the safety limit")
    source_hash = image_source_sha256(fd, int(info.st_size))
    format_name, width, height = _image_metadata(
        fd, filename, int(info.st_size))
    raw_text = _sandboxed_tesseract(fd, format_name)
    text, truncated = _normalize_ocr(raw_text, max_chars)
    return {
        "format": format_name,
        "extractor": "sandboxed-tesseract",
        "language": "eng",
        "width": width,
        "height": height,
        "pixels": width * height,
        "text": text,
        "characters": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_detected": bool(text),
        "source_bytes": int(info.st_size),
        "source_sha256": source_hash,
        "truncated": truncated,
        "limitations": "ocr_only",
    }
