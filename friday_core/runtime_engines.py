"""Launch and probe contracts for the portable model engines.

The supervisor keeps its original vLLM code path untouched and consults this
module only for ``llama_server`` and ``mlx_lm`` profiles: where the binary or
environment lives, how the process is started, which command-line marker
identifies it, and how the boot gate proves the served context.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from friday_host import fs
from friday_host.host import HostPlatform, current_host
from friday_host.paths import venv_python

from .engine_assets import (llama_server_binary, llama_server_tag,
                            model_asset)
from .hardware import RuntimeProfile

CANARY = "Friday authenticated boot calibration canary"
_SERVED_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")


class EnginePreflightError(RuntimeError):
    """The engine binary, environment, or model is not installed."""


@dataclass(frozen=True)
class EngineLaunch:
    command: tuple[str, ...]
    environment: dict[str, str]
    cwd: Path
    log_name: str


def runtime_root(environment: Mapping[str, str] | None = None,
                 *, qwen_root: Path | None = None) -> Path:
    """Directory holding per-engine runtimes (``runtime/`` in the install)."""
    env = os.environ if environment is None else environment
    explicit = env.get("FRIDAY_RUNTIME_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if qwen_root is not None:
        return qwen_root.parent
    from friday_host.paths import default_install_root  # noqa: PLC0415

    return default_install_root(env) / "runtime"


def ensure_local_api_key(path: Path) -> str:
    """Read or mint the credential shared by a portable engine and Friday."""
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    if existing:
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_hex(24)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | fs.PRIVATE_OPEN_FLAGS, 0o600)
    try:
        os.write(descriptor, (value + "\n").encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return value


def _validated_served_model(profile: RuntimeProfile) -> str:
    if not _SERVED_MODEL.fullmatch(profile.served_model):
        raise EnginePreflightError("served model name is not safe")
    return profile.served_model


def _model_directory(profile: RuntimeProfile, repo: Path) -> Path:
    if not profile.model_asset:
        raise EnginePreflightError("profile has no pinned model asset")
    asset = model_asset(profile.model_asset)
    directory = (repo / "models" / asset.directory).resolve()
    if not directory.is_dir():
        raise EnginePreflightError(
            f"pinned model {asset.key} is not installed; run "
            f"ops/install_local_model.py --asset {asset.key}")
    for name, size, _digest in asset.files:
        path = directory / name
        try:
            metadata = path.stat()
        except OSError as exc:
            raise EnginePreflightError(
                f"pinned model file is missing: {asset.key}/{name}") from exc
        if metadata.st_size != size:
            raise EnginePreflightError(
                f"pinned model file size changed: {asset.key}/{name}")
    return directory


class LlamaServerEngine:
    name = "llama_server"
    owner_marker = "llama-server"
    stop_marker = "llama-server"
    health_timeout = 300
    supports_process_vram = False
    credential_probe_path = "/tokenize"

    def __init__(self, host: HostPlatform | None = None) -> None:
        self.host = host or current_host()

    def binary_path(self, profile: RuntimeProfile, root: Path) -> Path:
        binary = llama_server_binary(
            self.host.os, self.host.arch, profile.engine_backend)
        return root / "llama-server" / binary.directory / binary.executable

    def preflight(self, profile: RuntimeProfile, *, repo: Path, root: Path) -> None:
        binary = self.binary_path(profile, root)
        if not binary.is_file():
            raise EnginePreflightError(
                f"llama-server {llama_server_tag()} for {profile.engine_backend} "
                f"is not installed at {binary}; run ops/install_llama_server.py")
        _model_directory(profile, repo)
        _validated_served_model(profile)

    def prepare_launch(self, profile: RuntimeProfile, *, repo: Path, root: Path,
                       key_file: Path, environment: Mapping[str, str]
                       ) -> EngineLaunch:
        self.preflight(profile, repo=repo, root=root)
        asset = model_asset(profile.model_asset)
        model = _model_directory(profile, repo) / asset.model_file
        env = {key: value for key, value in environment.items()
               if not key.startswith("LLAMA_ARG_")}
        env.pop("LLAMA_LOG_COLORS", None)
        devices = profile.effective_llm_cuda_devices
        for variable in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                         "GGML_VK_VISIBLE_DEVICES"):
            env.pop(variable, None)
        if devices and profile.engine_backend == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in devices)
        launch = profile.engine_launch()
        arguments = list(launch["args"])
        # llama-server splits --ctx-size across slots; keep the profile's
        # per-slot context exact by scaling the total.
        index = arguments.index("--ctx-size")
        arguments[index + 1] = str(profile.context_tokens * profile.max_sequences)
        command = (str(self.binary_path(profile, root)), "--model", str(model),
                   "--api-key-file", str(key_file), *arguments)
        return EngineLaunch(command=command, environment=env, cwd=repo,
                            log_name="llama-server.log")

    def context_probe(self, profile: RuntimeProfile, credential: str,
                      urlopen: Callable, *, timeout: float) -> tuple[int, int]:
        base = profile.local_base_url.rstrip("/")[:-3]
        headers = {"Authorization": f"Bearer {credential}"}
        with urlopen(urllib.request.Request(base + "/props", headers=headers),
                     timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError("engine properties probe was rejected")
            props = json.loads(response.read())
        settings = props.get("default_generation_settings") if isinstance(
            props, dict) else None
        observed = settings.get("n_ctx") if isinstance(settings, dict) else None
        request = urllib.request.Request(
            base + "/tokenize",
            data=json.dumps({"content": CANARY, "add_special": True}).encode(),
            method="POST",
            headers={**headers, "Content-Type": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError("tokenization probe was rejected")
            tokens = json.loads(response.read())
        count = len(tokens.get("tokens", [])) if isinstance(tokens, dict) else 0
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise RuntimeError("engine did not report its context size")
        return observed, count


class MlxEngine:
    name = "mlx_lm"
    owner_marker = "friday_core.mlx_server"
    stop_marker = "friday_core.mlx_server"
    health_timeout = 300
    supports_process_vram = False
    credential_probe_path = "/v1/models"

    def __init__(self, host: HostPlatform | None = None) -> None:
        self.host = host or current_host()

    def python_path(self, root: Path) -> Path:
        return venv_python(root / "mlx", self.host)

    def preflight(self, profile: RuntimeProfile, *, repo: Path, root: Path) -> None:
        if not self.python_path(root).is_file():
            raise EnginePreflightError(
                "the pinned MLX runtime is not installed; run "
                "ops/install_mlx_runtime.py")
        _model_directory(profile, repo)
        _validated_served_model(profile)

    def prepare_launch(self, profile: RuntimeProfile, *, repo: Path, root: Path,
                       key_file: Path, environment: Mapping[str, str]
                       ) -> EngineLaunch:
        self.preflight(profile, repo=repo, root=root)
        model = _model_directory(profile, repo)
        env = dict(environment)
        env["PYTHONPATH"] = str(repo)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env.pop("PYTHONHOME", None)
        command = (str(self.python_path(root)), "-m", "friday_core.mlx_server",
                   "--model", str(model), "--api-key-file", str(key_file),
                   *profile.engine_launch()["args"])
        return EngineLaunch(command=command, environment=env, cwd=repo,
                            log_name="mlx-server.log")

    def context_probe(self, profile: RuntimeProfile, credential: str,
                      urlopen: Callable, *, timeout: float) -> tuple[int, int]:
        base = profile.local_base_url.rstrip("/")[:-3]
        request = urllib.request.Request(
            base + "/tokenize",
            data=json.dumps({"model": profile.served_model,
                             "prompt": CANARY}).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {credential}",
                     "Content-Type": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError("tokenization probe was rejected")
            result = json.loads(response.read())
        if not isinstance(result, dict):
            raise RuntimeError("tokenization probe returned an invalid response")
        observed = result.get("max_model_len")
        count = result.get("count")
        if (isinstance(observed, bool) or not isinstance(observed, int)
                or isinstance(count, bool) or not isinstance(count, int)):
            raise RuntimeError("tokenization probe returned an invalid response")
        return observed, count


def engine_for(profile: RuntimeProfile, host: HostPlatform | None = None):
    if profile.engine == "llama_server":
        return LlamaServerEngine(host)
    if profile.engine == "mlx_lm":
        return MlxEngine(host)
    raise ValueError(f"no portable engine named {profile.engine}")


__all__ = [
    "CANARY", "EngineLaunch", "EnginePreflightError", "LlamaServerEngine",
    "MlxEngine", "engine_for", "ensure_local_api_key", "runtime_root",
]
