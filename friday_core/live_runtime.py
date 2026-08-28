"""Fail-closed discovery for installed live-evaluation commands."""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .runtime_paths import (
    default_config_root,
    default_install_root,
    default_qwen_runtime,
    default_state_root,
)


_CONFIG_KEYS = frozenset({
    "FRIDAY_CONFIG_ROOT",
    "FRIDAY_EMBEDDING_BATCH_SIZE",
    "FRIDAY_EMBEDDING_MODEL",
    "FRIDAY_INSTALL_ROOT",
    "FRIDAY_LLM_REPO",
    "FRIDAY_LOCAL_API_KEY",
    "FRIDAY_LOCAL_API_KEY_FILE",
    "FRIDAY_QWEN_ROOT",
    "FRIDAY_STATE_DIR",
})
_MODEL_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,160}")
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class LiveRuntimeIdentity:
    state_dir: Path
    base_url: str
    model: str
    fingerprint: str
    native_vision_max_side: int | None


def _read_private_regular(path: Path, *, minimum: int, maximum: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if (not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or metadata.st_mode & 0o077
                    or not minimum <= metadata.st_size <= maximum):
                raise RuntimeError(f"private runtime file is invalid: {path}")
            encoded = stream.read(maximum + 1)
    except OSError as exc:
        raise RuntimeError(f"private runtime file is unavailable: {path}") from exc
    if len(encoded) != metadata.st_size:
        raise RuntimeError(f"private runtime file changed while being read: {path}")
    return encoded


def _config_path(environment: Mapping[str, str]) -> Path:
    configured = str(environment.get("FRIDAY_CONFIG_ROOT", "")).strip()
    root = Path(configured).expanduser() if configured else default_config_root()
    return root / "friday.env"


def _parse_config(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    encoded = _read_private_regular(path, minimum=1, maximum=65_536)
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Friday runtime configuration is not UTF-8") from exc
    values: dict[str, str] = {}
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=(.*)", line)
        if match is None:
            raise RuntimeError(
                f"Friday runtime configuration line {number} is malformed")
        key, raw_value = match.groups()
        try:
            parsed = shlex.split(raw_value, posix=True)
        except ValueError as exc:
            raise RuntimeError(
                f"Friday runtime configuration line {number} is malformed") from exc
        if len(parsed) != 1:
            raise RuntimeError(
                f"Friday runtime configuration line {number} is malformed")
        if key in _CONFIG_KEYS:
            values[key] = parsed[0]
    return values


def runtime_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    supplied = dict(os.environ if environment is None else environment)
    configured = _parse_config(_config_path(supplied))
    configured.update({
        key: value for key, value in supplied.items()
        if key in _CONFIG_KEYS
    })
    return configured


def resolve_state_dir(
    repo: str | Path,
    environment: Mapping[str, str] | None = None,
) -> Path:
    values = runtime_environment(environment)
    configured = str(values.get("FRIDAY_STATE_DIR", "")).strip()
    if configured:
        return Path(configured).expanduser()
    development = Path(repo) / "state"
    if development.is_dir():
        return development
    return default_state_root()


def resolve_application_root(
    repo: str | Path,
    environment: Mapping[str, str] | None = None,
) -> Path:
    values = runtime_environment(environment)
    configured = str(values.get("FRIDAY_INSTALL_ROOT", "")).strip()
    install_root = (
        Path(configured).expanduser() if configured else default_install_root())
    current = install_root / "current"
    return current if current.is_dir() else Path(repo)


def read_live_runtime(
    repo: str | Path,
    *,
    require_native_vision: bool = False,
    environment: Mapping[str, str] | None = None,
) -> LiveRuntimeIdentity:
    state_dir = resolve_state_dir(repo, environment)
    encoded = _read_private_regular(
        state_dir / "runtime-resolved.json", minimum=2, maximum=128_000)
    try:
        value = json.loads(
            encoded,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {constant}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("resolved runtime manifest is invalid") from exc
    base_url = value.get("local_base_url") if isinstance(value, dict) else None
    model = value.get("served_model") if isinstance(value, dict) else None
    fingerprint = value.get("fingerprint") if isinstance(value, dict) else None
    if (not isinstance(base_url, str) or not isinstance(model, str)
            or not isinstance(fingerprint, str)
            or _FINGERPRINT_PATTERN.fullmatch(fingerprint) is None
            or _MODEL_PATTERN.fullmatch(model) is None):
        raise RuntimeError("resolved local model identity is incomplete")
    parsed = urllib.parse.urlsplit(base_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("resolved local model endpoint is invalid") from exc
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.username is not None or parsed.password is not None
            or parsed.path.rstrip("/") != "/v1" or parsed.query or parsed.fragment
            or port is None or not 1 <= port <= 65_535):
        raise RuntimeError(
            "live evaluation requires exact loopback model authority")

    vision = value.get("native_vision") if isinstance(value, dict) else None
    max_side: int | None = None
    if isinstance(vision, dict) and vision.get("enabled") is True:
        candidate = vision.get("max_side")
        if (isinstance(candidate, bool) or not isinstance(candidate, int)
                or not 256 <= candidate <= 4_096):
            raise RuntimeError("native-vision runtime identity is invalid")
        max_side = candidate
    if require_native_vision and max_side is None:
        raise RuntimeError(
            "native-vision scorecard requires an active verified vision profile")
    return LiveRuntimeIdentity(
        state_dir=state_dir,
        base_url=base_url.rstrip("/"),
        model=model,
        fingerprint=fingerprint,
        native_vision_max_side=max_side,
    )


def read_local_model_credential(
    repo: str | Path,
    environment: Mapping[str, str] | None = None,
) -> str:
    values = runtime_environment(environment)
    direct = str(values.get("FRIDAY_LOCAL_API_KEY", "")).strip()
    if direct:
        if any(character.isspace() for character in direct):
            raise RuntimeError("local model credential is malformed")
        return direct
    configured = str(values.get("FRIDAY_LOCAL_API_KEY_FILE", "")).strip()
    qwen_root_value = str(
        values.get("FRIDAY_LLM_REPO")
        or values.get("FRIDAY_QWEN_ROOT")
        or default_qwen_runtime())
    path = (
        Path(configured).expanduser()
        if configured else Path(qwen_root_value).expanduser() / "api_key.txt")
    encoded = _read_private_regular(path, minimum=16, maximum=4_096)
    try:
        key = encoded.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("local model credential is not UTF-8") from exc
    if not key or any(character.isspace() for character in key):
        raise RuntimeError("local model credential is empty or malformed")
    return key
