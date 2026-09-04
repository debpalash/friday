"""Hardware discovery and conservative local-runtime profile selection.

The profile is deliberately a deterministic policy object rather than startup
logic.  The supervisor owns process launch, while the server consumes the same
resolved model, context, and device settings through environment variables.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from friday_host import procs
from friday_host.host import normalize_arch, normalize_os

from .engine_assets import (ENGINES, ModelAsset, engine_backends,
                            select_model_tier)


GIB = 1024 ** 3
PORTABLE_ENGINES = frozenset({"llama_server", "mlx_lm"})
DEFAULT_QWEN_MODEL = "models/Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound"
DEFAULT_SERVED_MODEL = "qwen3.8-27b"
DEFAULT_LLM_HOST = "127.0.0.1"
DEFAULT_LLM_PORT = 18021
DEFAULT_TTS_RESERVE_GIB = 6.625
FALLBACK_CUDA_MEMORY_GIB = 24.0
NATIVE_VISION_MIN_MEMORY_GIB = 30.0


@dataclass(frozen=True)
class Accelerator:
    backend: str
    index: int
    name: str
    total_memory_bytes: int
    free_memory_bytes: int | None = None
    identity: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Stable device identities bind private calibration records, but are
        # not diagnostics and must not enter the UI-visible runtime manifest.
        value.pop("identity", None)
        value["total_memory_gib"] = round(self.total_memory_bytes / GIB, 2)
        value["free_memory_gib"] = (
            round(self.free_memory_bytes / GIB, 2)
            if self.free_memory_bytes is not None else None)
        return value


@dataclass(frozen=True)
class HardwareSnapshot:
    cpu_count: int
    system_memory_bytes: int
    accelerators: tuple[Accelerator, ...] = ()
    detection_errors: tuple[str, ...] = ()
    cuda_probe: str = "unknown"
    # Additive host facts. Linux x86_64 without unified memory keeps the
    # pre-port fingerprint byte for byte.
    platform: str = "linux"
    arch: str = "x86_64"
    unified_memory: bool = False

    @property
    def primary_cuda(self) -> Accelerator | None:
        cuda = [item for item in self.accelerators if item.backend == "cuda"]
        return max(
            cuda, key=lambda item: (item.total_memory_bytes, -item.index),
            default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_count": self.cpu_count,
            "system_memory_bytes": self.system_memory_bytes,
            "system_memory_gib": round(self.system_memory_bytes / GIB, 2),
            "accelerators": [item.to_dict() for item in self.accelerators],
            "detection_errors": list(self.detection_errors),
            "cuda_probe": self.cuda_probe,
            "platform": self.platform,
            "arch": self.arch,
            "unified_memory": self.unified_memory,
        }

    @property
    def fingerprint(self) -> str:
        """Stable capacity/topology identity for persisted runtime evidence.

        Free-memory readings and probe diagnostics are intentionally excluded:
        they describe transient load, not the machine on which a calibration
        was proven.  Accelerator names remain in the identity so a different
        board with coincidentally equal VRAM cannot inherit old evidence.
        """
        topology = {
            "cpu_count": self.cpu_count,
            "system_memory_bytes": self.system_memory_bytes,
            "accelerators": [
                {
                    "backend": item.backend,
                    "index": item.index,
                    "name": item.name,
                    "total_memory_bytes": item.total_memory_bytes,
                    "identity": item.identity,
                }
                for item in sorted(
                    self.accelerators,
                    key=lambda item: (item.backend, item.index, item.name),
                )
            ],
        }
        if (self.platform, self.arch, self.unified_memory) != (
                "linux", "x86_64", False):
            topology["platform"] = self.platform
            topology["arch"] = self.arch
            topology["unified_memory"] = self.unified_memory
        encoded = json.dumps(
            topology, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    source: str
    hardware: HardwareSnapshot
    qwen_model: str
    served_model: str
    llm_host: str
    llm_port: int
    context_tokens: int
    max_sequences: int
    gpu_memory_utilization: float
    kv_mode: str
    cuda_graph_capture_size: int
    tts_device: str
    asr_threads: int
    llm_cuda_device: int | None
    tts_cuda_device: int | None
    llm_memory_budget_gib: float
    tts_reserve_gib: float
    unallocated_gpu_gib: float
    local_runtime_available: bool
    launch_override_fingerprint: str = ""
    warnings: tuple[str, ...] = ()
    overrides: tuple[str, ...] = ()
    # Empty preserves the original single-device profile encoding. Multi-GPU
    # profiles carry the complete physical CUDA set in deterministic order.
    llm_cuda_devices: tuple[int, ...] = ()
    native_vision_enabled: bool = False
    native_vision_max_images: int = 0
    native_vision_max_side: int = 0
    native_vision_gpu_reserve_gib: float = 0.0
    native_vision_host_reserve_mib: int = 0
    # Engine seam. "vllm" is the original Linux/NVIDIA runtime; every field
    # below is omitted from fingerprints while the engine is vLLM.
    engine: str = "vllm"
    engine_backend: str = "cuda"
    model_asset: str = ""
    model_path: str = ""
    gpu_layers: int = -1
    engine_threads: int = 0

    @property
    def portable_engine(self) -> bool:
        return self.engine != "vllm"

    def engine_launch(self) -> dict[str, Any]:
        """Deterministic, secret-free launch description for the engine."""
        if self.engine == "llama_server":
            arguments = [
                "--ctx-size", str(self.context_tokens),
                "--parallel", str(self.max_sequences),
                "--n-gpu-layers", str(self.gpu_layers),
                "--reasoning-budget", "0", "--jinja", "--no-webui",
                "--flash-attn", "auto",
                "--alias", self.served_model,
                "--host", self.llm_host, "--port", str(self.llm_port),
            ]
            if self.engine_threads:
                arguments += ["--threads", str(self.engine_threads)]
            return {"engine": self.engine, "backend": self.engine_backend,
                    "model_asset": self.model_asset, "args": arguments}
        if self.engine == "mlx_lm":
            return {"engine": self.engine, "backend": "metal",
                    "model_asset": self.model_asset, "args": [
                        "--context-tokens", str(self.context_tokens),
                        "--served-model", self.served_model,
                        "--max-sequences", str(self.max_sequences),
                        "--host", self.llm_host, "--port", str(self.llm_port),
                    ]}
        return {"engine": "vllm", "qwen_environment": self.qwen_environment()}

    @property
    def effective_llm_cuda_devices(self) -> tuple[int, ...]:
        if self.llm_cuda_devices:
            return self.llm_cuda_devices
        return (() if self.llm_cuda_device is None
                else (self.llm_cuda_device,))

    @property
    def tensor_parallel_size(self) -> int:
        return max(1, len(self.effective_llm_cuda_devices))

    @property
    def local_base_url(self) -> str:
        return f"http://{self.llm_host}:{self.llm_port}/v1"

    @property
    def hardware_fingerprint(self) -> str:
        return self.hardware.fingerprint

    @property
    def family_fingerprint(self) -> str:
        """Identity fields that a last-known-good tuning may not replace."""
        family = {
            "qwen_model": self.qwen_model,
            "served_model": self.served_model,
            "llm_host": self.llm_host,
            "llm_port": self.llm_port,
            "llm_cuda_device": self.llm_cuda_device,
            "tts_device": self.tts_device,
            "tts_cuda_device": self.tts_cuda_device,
            "local_runtime_available": self.local_runtime_available,
            "launch_override_fingerprint": self.launch_override_fingerprint,
        }
        if self.tensor_parallel_size > 1:
            family["llm_cuda_devices"] = list(
                self.effective_llm_cuda_devices)
            family["tensor_parallel_size"] = self.tensor_parallel_size
        if self.native_vision_enabled:
            family["native_vision"] = self.native_vision_config
        if self.portable_engine:
            family["engine"] = self.engine
            family["engine_backend"] = self.engine_backend
            family["model_asset"] = self.model_asset
        encoded = json.dumps(
            family, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def health_url(self) -> str:
        return f"http://{self.llm_host}:{self.llm_port}/health"

    def qwen_environment(self) -> dict[str, str]:
        utilization = f"{self.gpu_memory_utilization:.3f}".rstrip("0").rstrip(".")
        result = {
            "MODEL": self.qwen_model,
            "PORT": str(self.llm_port),
            "GPU_UTIL": utilization,
            "MAX_LEN": str(self.context_tokens),
            "MAX_SEQS": str(self.max_sequences),
            "CTX": self.kv_mode,
            "CG": str(self.cuda_graph_capture_size),
        }
        if self.tensor_parallel_size > 1:
            result["TENSOR_PARALLEL_SIZE"] = str(self.tensor_parallel_size)
        return result

    @property
    def native_vision_config(self) -> dict[str, Any]:
        return {
            "enabled": self.native_vision_enabled,
            "max_images": self.native_vision_max_images,
            "max_side": self.native_vision_max_side,
            "max_pixels_per_image": self.native_vision_max_side ** 2,
            "gpu_reserve_gib_per_rank": self.native_vision_gpu_reserve_gib,
            "host_reserve_mib": self.native_vision_host_reserve_mib,
        }

    def operational_config(self) -> dict[str, Any]:
        config = {
            "qwen_environment": self.qwen_environment(),
            "assistant_environment": self.assistant_environment(),
            "llm_cuda_device": self.llm_cuda_device,
            "tts_cuda_device": self.tts_cuda_device,
            "local_runtime_available": self.local_runtime_available,
            "admission_budget": self.admission_budget,
        }
        # Preserve byte-for-byte fingerprints for every already-proven
        # single-GPU profile; the new authority exists only when it matters.
        if self.tensor_parallel_size > 1:
            config["llm_cuda_devices"] = list(
                self.effective_llm_cuda_devices)
            config["tensor_parallel_size"] = self.tensor_parallel_size
        if self.native_vision_enabled:
            config["native_vision"] = self.native_vision_config
        if self.overrides:
            config["explicit_overrides"] = sorted(set(self.overrides))
        if self.launch_override_fingerprint:
            config["launch_override_fingerprint"] = (
                self.launch_override_fingerprint)
        if self.portable_engine:
            config["engine_launch"] = self.engine_launch()
        return config

    @property
    def admission_budget(self) -> dict[str, Any]:
        """Stable action capacity after interactive/runtime reservations.

        Transient free-memory readings are deliberately excluded; they are a
        live admission signal and must never churn the runtime fingerprint.
        """
        cpu_total = max(1, int(self.hardware.cpu_count))
        cpu_reserve = max(2, math.ceil(cpu_total * 0.10))
        memory_total_mib = max(
            0, int(self.hardware.system_memory_bytes // (1024 ** 2)))
        memory_reserve_mib = (
            max(2048, math.ceil(memory_total_mib * 0.10))
            + self.native_vision_host_reserve_mib)
        accelerator_vram_mib: dict[str, int] = {}
        for accelerator in self.hardware.accelerators:
            if accelerator.backend != "cuda":
                continue
            total_mib = int(accelerator.total_memory_bytes // (1024 ** 2))
            reserved_mib = 0
            if accelerator.index in self.effective_llm_cuda_devices:
                reserved_mib += math.ceil(self.llm_memory_budget_gib * 1024)
            if accelerator.index == self.tts_cuda_device:
                reserved_mib += math.ceil(self.tts_reserve_gib * 1024)
            guard_mib = max(512, math.ceil(total_mib * 0.03))
            accelerator_vram_mib[f"cuda:{accelerator.index}"] = max(
                0, total_mib - reserved_mib - guard_mib)
        return {
            "cpu_cores": float(max(1, cpu_total - cpu_reserve)),
            "ram_mib": max(128, memory_total_mib - memory_reserve_mib),
            "vram_mib_by_accelerator": accelerator_vram_mib,
            "concurrency_slots": max(2, min(8, cpu_total // 4)),
            "network_slots": max(2, min(8, cpu_total // 8)),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.operational_config(), sort_keys=True,
            separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def assistant_environment(self) -> dict[str, str]:
        result = {
            "FRIDAY_LOCAL_BASE_URL": self.local_base_url,
            "FRIDAY_LOCAL_MODEL": self.served_model,
            "FRIDAY_MODEL_CONTEXT_TOKENS": str(self.context_tokens),
            "FRIDAY_TTS_DEVICE": self.tts_device,
            "FRIDAY_ASR_THREADS": str(self.asr_threads),
        }
        if self.portable_engine:
            result["FRIDAY_LLM_ENGINE"] = self.engine
        if self.native_vision_enabled:
            result.update({
                "FRIDAY_NATIVE_VISION": "1",
                "FRIDAY_NATIVE_VISION_MAX_IMAGES": str(
                    self.native_vision_max_images),
                "FRIDAY_NATIVE_VISION_MAX_SIDE": str(
                    self.native_vision_max_side),
            })
        return result

    def to_dict(self) -> dict[str, Any]:
        value = {
            "name": self.name,
            "source": self.source,
            "qwen_model": self.qwen_model,
            "served_model": self.served_model,
            "local_base_url": self.local_base_url,
            "health_url": self.health_url,
            "context_tokens": self.context_tokens,
            "max_sequences": self.max_sequences,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "kv_mode": self.kv_mode,
            "cuda_graph_capture_size": self.cuda_graph_capture_size,
            "tts_device": self.tts_device,
            "asr_threads": self.asr_threads,
            "llm_cuda_device": self.llm_cuda_device,
            "tts_cuda_device": self.tts_cuda_device,
            "llm_memory_budget_gib": self.llm_memory_budget_gib,
            "tts_reserve_gib": self.tts_reserve_gib,
            "unallocated_gpu_gib": self.unallocated_gpu_gib,
            "local_runtime_available": self.local_runtime_available,
            "native_vision": self.native_vision_config,
            "admission_budget": self.admission_budget,
            "fingerprint": self.fingerprint,
            "overrides": list(self.overrides),
            "warnings": list(self.warnings),
            "hardware": self.hardware.to_dict(),
        }
        value["llm_cuda_devices"] = list(self.effective_llm_cuda_devices)
        value["tensor_parallel_size"] = self.tensor_parallel_size
        value["llm_total_memory_budget_gib"] = round(
            self.llm_memory_budget_gib * self.tensor_parallel_size, 3)
        value["engine"] = self.engine
        value["engine_backend"] = self.engine_backend
        value["model_asset"] = self.model_asset
        value["model_path"] = self.model_path
        value["gpu_layers"] = self.gpu_layers
        value["engine_launch"] = self.engine_launch()
        return value


Probe = Callable[[list[str]], str]


def _default_probe(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=3)
    if result.returncode:
        detail = (result.stderr or result.stdout or "probe failed").strip()[-300:]
        raise RuntimeError(detail)
    return result.stdout


def parse_nvidia_smi(output: str) -> tuple[Accelerator, ...]:
    """Parse the stable CSV query used by :func:`detect_hardware`."""
    accelerators = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            index, name, total_mib, free_mib = fields
            identity = ""
        elif len(fields) == 5:
            index, identity, name, total_mib, free_mib = fields
        else:
            raise ValueError(f"unexpected nvidia-smi row: {line[:120]}")
        accelerators.append(Accelerator(
            backend="cuda", index=int(index), name=name,
            total_memory_bytes=int(total_mib) * 1024 ** 2,
            free_memory_bytes=int(free_mib) * 1024 ** 2,
            identity=identity,
        ))
    return tuple(accelerators)


def _system_memory_bytes(meminfo_path: Path) -> int:
    try:
        for line in meminfo_path.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return procs.physical_memory_bytes()


def _drm_accelerators(root: Path) -> tuple[Accelerator, ...]:
    """Best-effort inventory for non-NVIDIA Linux accelerators."""
    vendors = {"0x1002": ("rocm", "AMD GPU"), "0x8086": ("oneapi", "Intel GPU")}
    found = []
    for card in sorted(root.glob("card[0-9]*")):
        device = card / "device"
        try:
            backend, fallback_name = vendors[(device / "vendor").read_text().strip()]
            total = int((device / "mem_info_vram_total").read_text().strip())
            free_path = device / "mem_info_vram_used"
            used = int(free_path.read_text().strip()) if free_path.is_file() else None
            index = int(card.name.removeprefix("card"))
        except (KeyError, OSError, ValueError):
            continue
        found.append(Accelerator(
            backend=backend, index=index, name=fallback_name,
            total_memory_bytes=total,
            free_memory_bytes=max(total - used, 0) if used is not None else None,
            identity=device.resolve().name,
        ))
    return tuple(found)


def parse_sysctl_darwin(memsize: str, brand: str, arch: str) -> tuple[
        int, tuple[Accelerator, ...], bool]:
    """Interpret ``sysctl -n hw.memsize`` and the CPU brand on macOS."""
    total = int(memsize.strip())
    brand = brand.strip() or "Apple CPU"
    if arch == "aarch64":
        return total, (Accelerator(
            backend="metal", index=0, name=brand, total_memory_bytes=total,
            free_memory_bytes=None, identity=brand),), True
    return total, (), False


def parse_win32_video_controllers(text: str) -> tuple[Accelerator, ...]:
    """Interpret ``Get-CimInstance Win32_VideoController`` JSON output."""
    try:
        loaded = json.loads(text) if text.strip() else []
    except json.JSONDecodeError as exc:
        raise ValueError("Win32_VideoController output is not JSON") from exc
    rows = loaded if isinstance(loaded, list) else [loaded]
    found = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        name = str(row.get("Name") or "").strip()
        ram = row.get("AdapterRAM")
        if not name or "nvidia" in name.lower():
            continue
        try:
            total = int(ram) if ram is not None else 0
        except (TypeError, ValueError):
            total = 0
        found.append(Accelerator(
            backend="vulkan", index=index, name=name,
            total_memory_bytes=max(0, total), free_memory_bytes=None,
            identity=name))
    return tuple(found)


_WIN32_VIDEO_QUERY = (
    "Get-CimInstance Win32_VideoController | "
    "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress")


def detect_hardware(*, probe: Probe | None = None,
                    meminfo_path: str | Path = "/proc/meminfo",
                    drm_root: str | Path = "/sys/class/drm",
                    device_root: str | Path = "/dev",
                    platform_name: str | None = None,
                    machine: str | None = None) -> HardwareSnapshot:
    """Collect a lightweight inventory without importing a GPU framework."""
    run = probe or _default_probe
    host_os = normalize_os(platform_name or sys.platform)
    import platform as _platform  # noqa: PLC0415

    arch = normalize_arch(machine or _platform.machine())
    errors: list[str] = []
    accelerators: tuple[Accelerator, ...] = ()
    unified = False
    memory = 0
    if host_os == "macos":
        try:
            memory, accelerators, unified = parse_sysctl_darwin(
                run(["sysctl", "-n", "hw.memsize"]),
                run(["sysctl", "-n", "machdep.cpu.brand_string"]), arch)
        except (OSError, RuntimeError, ValueError,
                subprocess.SubprocessError) as exc:
            errors.append(f"sysctl: {str(exc)[:240]}")
        cuda_probe = "absent"
    else:
        try:
            output = run([
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ])
            accelerators = parse_nvidia_smi(output)
        except (OSError, RuntimeError, ValueError,
                subprocess.SubprocessError) as exc:
            errors.append(f"nvidia-smi: {str(exc)[:240]}")
        if not accelerators and host_os == "linux":
            accelerators = _drm_accelerators(Path(drm_root))
        elif not accelerators and host_os == "windows":
            try:
                accelerators = parse_win32_video_controllers(run([
                    "powershell", "-NoProfile", "-NonInteractive",
                    "-Command", _WIN32_VIDEO_QUERY]))
            except (OSError, RuntimeError, ValueError,
                    subprocess.SubprocessError) as exc:
                errors.append(f"Win32_VideoController: {str(exc)[:240]}")
        if any(item.backend == "cuda" for item in accelerators):
            cuda_probe = "available"
        elif host_os == "linux":
            cuda_probe = ("probe_failed"
                          if (Path(device_root) / "nvidia0").exists()
                          else "absent")
        else:
            cuda_probe = ("probe_failed" if shutil.which("nvidia-smi")
                          else "absent")
    if not memory:
        memory = _system_memory_bytes(Path(meminfo_path))
    return HardwareSnapshot(
        cpu_count=max(os.cpu_count() or 1, 1),
        system_memory_bytes=memory,
        accelerators=accelerators,
        detection_errors=tuple(errors),
        cuda_probe=cuda_probe,
        platform=host_os,
        arch=arch,
        unified_memory=unified,
    )


def _env_int(environment: Mapping[str, str], name: str, default: int,
             *, minimum: int, maximum: int) -> tuple[int, bool]:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default, False
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value, True


def _env_float(environment: Mapping[str, str], name: str, default: float,
               *, minimum: float, maximum: float) -> tuple[float, bool]:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default, False
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value, True


def _env_bool(environment: Mapping[str, str], name: str,
              default: bool = False) -> tuple[bool, bool]:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default, False
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True, True
    if normalized in {"0", "false", "no", "off"}:
        return False, True
    raise ValueError(f"{name} must be true or false")


def _env_auto_bool(environment: Mapping[str, str], name: str
                   ) -> tuple[str, bool]:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return "auto", False
    normalized = raw.strip().lower()
    if normalized == "auto":
        return "auto", True
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return "enabled", True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return "disabled", True
    raise ValueError(f"{name} must be auto, enabled, or disabled")


def _cuda_device_override(
        environment: Mapping[str, str], name: str,
        *, detected_indices: set[int], allow_unprobed: bool
) -> tuple[int | None, bool]:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return None, False
    value = raw.strip()
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(
            f"{name} must contain exactly one non-negative CUDA device index")
    index = int(value)
    if detected_indices and index not in detected_indices:
        available = ", ".join(str(item) for item in sorted(detected_indices))
        raise ValueError(
            f"{name} selects CUDA device {index}; detected devices: {available}")
    if not detected_indices and not allow_unprobed:
        raise ValueError(f"{name} was set but no NVIDIA CUDA device was detected")
    return index, True


def _cuda_devices_override(
        environment: Mapping[str, str], name: str,
        *, detected_indices: set[int], allow_unprobed: bool
) -> tuple[tuple[int, ...] | None, bool]:
    """Parse a canonical physical CUDA set for tensor-parallel execution."""
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return None, False
    parts = [item.strip() for item in raw.split(",")]
    if (not parts or len(parts) > 16
            or any(not re.fullmatch(r"[0-9]+", item) for item in parts)):
        raise ValueError(
            f"{name} must be a comma-separated list of 1-16 non-negative "
            "CUDA device indices")
    values = tuple(int(item) for item in parts)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate CUDA devices")
    canonical = tuple(sorted(values))
    unknown = set(canonical) - detected_indices
    if detected_indices and unknown:
        available = ", ".join(str(item) for item in sorted(detected_indices))
        selected = ", ".join(str(item) for item in sorted(unknown))
        raise ValueError(
            f"{name} selects undetected CUDA devices {selected}; detected "
            f"devices: {available}")
    if not detected_indices and not allow_unprobed:
        raise ValueError(f"{name} was set but no NVIDIA CUDA device was detected")
    return canonical, True


def _validate_served_model(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", value):
        raise ValueError(
            "FRIDAY_LOCAL_MODEL must be a safe model identifier containing only "
            "letters, digits, '.', '_', '-', or '/'")
    return value


def select_runtime_profile(snapshot: HardwareSnapshot, *,
                           environment: Mapping[str, str] | None = None
                           ) -> RuntimeProfile:
    """Choose a shared Qwen/TTS profile for the detected CUDA memory tier.

    The assistant is reasoning-first: a single 24 GiB card runs the proven 200K
    KVarN profile and moves speech to CPU unless the user explicitly requests
    CUDA TTS.  A second suitable GPU isolates speech automatically; larger
    shared cards can hold both the long-context model and CUDA speech.
    """
    env = environment if environment is not None else os.environ
    overrides: list[str] = []
    warnings: list[str] = []
    external_launch_values = {
        name: env.get(name, "").strip()
        for name in (
            "FRIDAY_LLM_REPO", "FRIDAY_QWEN_ROOT", "FRIDAY_LLM_EXTRA_ARGS")
        if env.get(name, "").strip()
    }
    launch_override_fingerprint = ""
    if external_launch_values:
        launch_override_fingerprint = hashlib.sha256(json.dumps(
            external_launch_values, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()

    qwen_model = env.get("FRIDAY_QWEN_MODEL", "").strip() or DEFAULT_QWEN_MODEL
    served_model = _validate_served_model(
        env.get("FRIDAY_LOCAL_MODEL", "").strip() or DEFAULT_SERVED_MODEL)
    llm_host = env.get("FRIDAY_LLM_HOST", "").strip() or DEFAULT_LLM_HOST
    if llm_host.lower() == "localhost":
        llm_host = DEFAULT_LLM_HOST
    explicit_tts_device = bool(env.get("FRIDAY_TTS_DEVICE", "").strip())
    tts_device = env.get("FRIDAY_TTS_DEVICE", "").strip() or "cuda"
    for variable in (
            "FRIDAY_QWEN_MODEL", "FRIDAY_LOCAL_MODEL", "FRIDAY_LLM_HOST",
            "FRIDAY_TTS_DEVICE", "FRIDAY_LLM_EXTRA_ARGS"):
        if env.get(variable, "").strip():
            overrides.append(variable)
    if llm_host != "127.0.0.1":
        raise ValueError("FRIDAY_LLM_HOST must use IPv4 loopback or localhost")
    if tts_device.lower() not in {"cuda", "cuda:0", "cpu"}:
        raise ValueError(
            "FRIDAY_TTS_DEVICE must be cuda, cuda:0, or cpu; physical CUDA "
            "placement belongs in FRIDAY_TTS_CUDA_DEVICES")

    allow_unprobed, allow_changed = _env_bool(
        env, "FRIDAY_ALLOW_UNPROBED_CUDA")
    if allow_changed:
        overrides.append("FRIDAY_ALLOW_UNPROBED_CUDA")
    cuda_devices = sorted(
        (item for item in snapshot.accelerators if item.backend == "cuda"),
        key=lambda item: item.index)
    detected_indices = {item.index for item in cuda_devices}
    can_assume_cuda = bool(cuda_devices) or allow_unprobed or (
        snapshot.cuda_probe in {"probe_failed", "unknown"})
    local_runtime_available = bool(cuda_devices) or can_assume_cuda

    engine_request = env.get("FRIDAY_LLM_ENGINE", "").strip().lower() or "auto"
    if engine_request not in {"auto", "vllm", *PORTABLE_ENGINES}:
        raise ValueError(
            "FRIDAY_LLM_ENGINE must be auto, vllm, llama_server, or mlx_lm")
    if engine_request != "auto":
        overrides.append("FRIDAY_LLM_ENGINE")
    vllm_possible = snapshot.platform == "linux" and local_runtime_available
    if engine_request == "vllm" and not vllm_possible:
        raise ValueError(
            "FRIDAY_LLM_ENGINE=vllm requires Linux with an NVIDIA CUDA GPU")
    if engine_request in PORTABLE_ENGINES or (
            engine_request == "auto" and snapshot.platform != "linux"):
        return _select_portable_profile(
            snapshot, env, engine_request, overrides=overrides,
            warnings=warnings,
            launch_override_fingerprint=launch_override_fingerprint,
            served_model=served_model, llm_host=llm_host,
            cuda_devices=cuda_devices, explicit_tts_device=explicit_tts_device,
            tts_device=tts_device)
    _reject_vllm_only_overrides(env, engine="vllm", enforce=False)

    llm_override, llm_changed = _cuda_devices_override(
        env, "FRIDAY_LLM_CUDA_DEVICES",
        detected_indices=detected_indices, allow_unprobed=can_assume_cuda)
    if llm_changed:
        overrides.append("FRIDAY_LLM_CUDA_DEVICES")
    if cuda_devices:
        cuda_by_index = {item.index: item for item in cuda_devices}
        if llm_override is None:
            selected_cuda = (max(cuda_devices, key=lambda item: (
                item.total_memory_bytes, -item.index)),)
        else:
            selected_cuda = tuple(cuda_by_index[index]
                                  for index in llm_override)
        if len(selected_cuda) > 1 and len({
                item.total_memory_bytes for item in selected_cuda}) != 1:
            raise ValueError(
                "FRIDAY_LLM_CUDA_DEVICES tensor-parallel devices must have "
                "identical total VRAM")
        # vLLM applies gpu-memory-utilization independently to each tensor-
        # parallel rank. Equal-capacity ranks make this representative exact.
        cuda = min(selected_cuda, key=lambda item: (
            item.total_memory_bytes, item.index))
        llm_cuda_devices = tuple(item.index for item in selected_cuda)
        llm_cuda_device = llm_cuda_devices[0]
    else:
        cuda = None
        llm_cuda_devices = (llm_override if llm_override is not None
                            else (0,) if local_runtime_available else ())
        llm_cuda_device = (llm_cuda_devices[0]
                           if llm_cuda_devices else None)

    tts_override, tts_changed = _cuda_device_override(
        env, "FRIDAY_TTS_CUDA_DEVICES",
        detected_indices=detected_indices, allow_unprobed=can_assume_cuda)
    if tts_changed:
        overrides.append("FRIDAY_TTS_CUDA_DEVICES")
    uses_cuda_tts = tts_device.lower().startswith("cuda")
    if tts_changed and not uses_cuda_tts:
        raise ValueError(
            "FRIDAY_TTS_CUDA_DEVICES cannot be used with a non-CUDA "
            "FRIDAY_TTS_DEVICE")
    if not uses_cuda_tts or not local_runtime_available:
        tts_cuda_device = None
    elif tts_override is not None:
        tts_cuda_device = tts_override
    elif cuda_devices:
        # Placement must be stable across restarts, so total capacity—not the
        # workload-dependent memory.free counter—decides the speech GPU.
        alternate = min(
            (item for item in cuda_devices
             if item.index not in llm_cuda_devices
             and item.total_memory_bytes >= 8 * GIB),
            key=lambda item: (item.total_memory_bytes, item.index), default=None)
        if alternate is not None:
            tts_cuda_device = alternate.index
        elif len(llm_cuda_devices) > 1 and not explicit_tts_device:
            # Do not silently make a model-parallel rank share its fixed
            # memory envelope with speech. An explicit CUDA TTS request may.
            tts_device = "cpu"
            tts_cuda_device = None
        else:
            tts_cuda_device = llm_cuda_device
    else:
        tts_cuda_device = llm_cuda_device

    dedicated_tts = (
        tts_cuda_device is not None and llm_cuda_device is not None
        and tts_cuda_device not in llm_cuda_devices)
    reserve_gib = DEFAULT_TTS_RESERVE_GIB
    source = "hardware"
    if cuda is None:
        vram_gib = FALLBACK_CUDA_MEMORY_GIB
        context, sequences, utilization, kv_mode, capture = (
            8192, 4, 0.724, "huge", 8)
        llm_budget_gib = min(
            round(vram_gib * utilization, 3), vram_gib - reserve_gib)
        if local_runtime_available:
            name, source = "fallback-24gb", "fallback"
            warnings.append(
                "CUDA memory could not be measured; retaining the known-safe "
                "24 GB profile")
            if snapshot.cuda_probe == "absent" and allow_unprobed:
                warnings.append(
                    "CUDA availability is being assumed because "
                    "FRIDAY_ALLOW_UNPROBED_CUDA is enabled")
        else:
            name, source = "unsupported-local-runtime", "unsupported"
            warnings.append(
                "No NVIDIA CUDA GPU was detected; the installed local "
                "Qwen/OmniVoice runtime cannot be started")
            if snapshot.accelerators:
                warnings.append(
                    "the installed Qwen/OmniVoice stack currently requires an "
                    "NVIDIA CUDA GPU")
            warnings.append(
                "set FRIDAY_LLM_ENGINE=llama_server to run a smaller Qwen3 "
                "checkpoint through llama.cpp on this machine")
    else:
        vram_gib = cuda.total_memory_bytes / GIB
        if dedicated_tts:
            utilization = 0.93
            llm_budget_gib = round(vram_gib * utilization, 3)
            if vram_gib < 22:
                name, context, sequences, kv_mode, capture = (
                    "dedicated-constrained-cuda", 4096, 1, "huge", 4)
                warnings.append(
                    "the installed 27B model may not fit on the selected LLM GPU")
            elif vram_gib < 40:
                name, context, sequences, kv_mode, capture = (
                    "dedicated-llm-24gb", 200000, 8, "huge", 32)
            else:
                name, context, sequences, kv_mode, capture = (
                    "dedicated-llm-48gb-plus", 200000, 8, "huge", 32)
                llm_budget_gib = min(22.5, llm_budget_gib)
                utilization = round(llm_budget_gib / vram_gib, 3)
        else:
            if vram_gib < 22:
                name, context, sequences, kv_mode, capture = (
                    "constrained-cuda", 4096, 1, "huge", 4)
                llm_budget_gib = max(
                    vram_gib - reserve_gib, vram_gib * 0.55)
                warnings.append(
                    "the installed 27B model may not fit; configure "
                    "FRIDAY_QWEN_MODEL to a smaller checkpoint")
            elif vram_gib < 28:
                if ((explicit_tts_device or tts_changed)
                        and tts_device.startswith("cuda")):
                    name, context, sequences, kv_mode, capture = (
                        "shared-24gb-voice", 8192, 4, "huge", 8)
                    llm_budget_gib = min(
                        round(vram_gib * 0.724, 3),
                        vram_gib - reserve_gib)
                else:
                    # The model's measured 200K KVarN mode needs essentially the
                    # whole 24 GiB card.  CPU speech is slower, but an 8K hard cap
                    # is the wrong default for a general reasoning assistant.
                    name, context, sequences, kv_mode, capture = (
                        "reasoning-24gb", 200000, 8, "huge", 32)
                    tts_device = "cpu"
                    tts_cuda_device = None
                    reserve_gib = 0.0
                    llm_budget_gib = round(vram_gib * 0.93, 3)
            elif vram_gib < 40:
                name, context, sequences, kv_mode, capture = (
                    "shared-32gb", 200000, 8, "huge", 32)
                llm_budget_gib = min(22.5, vram_gib - reserve_gib)
            elif vram_gib < 64:
                name, context, sequences, kv_mode, capture = (
                    "shared-48gb", 200000, 8, "huge", 32)
                llm_budget_gib = min(22.5, vram_gib - reserve_gib)
            else:
                name, context, sequences, kv_mode, capture = (
                    "shared-64gb-plus", 200000, 8, "huge", 32)
                llm_budget_gib = min(22.5, vram_gib - reserve_gib)
            utilization_ceiling = (
                0.93 if not tts_device.lower().startswith("cuda") else 0.90)
            utilization = round(
                min(utilization_ceiling,
                    max(0.30, llm_budget_gib / vram_gib)), 3)

    if len(llm_cuda_devices) > 1:
        name = f"tensor-parallel-{len(llm_cuda_devices)}x-{name}"

    profile = RuntimeProfile(
        name=name, source=source, hardware=snapshot,
        qwen_model=qwen_model, served_model=served_model,
        llm_host=llm_host, llm_port=DEFAULT_LLM_PORT,
        context_tokens=context, max_sequences=sequences,
        gpu_memory_utilization=utilization, kv_mode=kv_mode,
        cuda_graph_capture_size=capture, tts_device=tts_device,
        asr_threads=min(8, max(2, snapshot.cpu_count // 8)),
        llm_cuda_device=llm_cuda_device,
        tts_cuda_device=tts_cuda_device,
        llm_memory_budget_gib=round(llm_budget_gib, 3),
        tts_reserve_gib=reserve_gib,
        unallocated_gpu_gib=round(max(vram_gib - llm_budget_gib, 0.0), 3),
        local_runtime_available=local_runtime_available,
        launch_override_fingerprint=launch_override_fingerprint,
        warnings=tuple(warnings),
        llm_cuda_devices=llm_cuda_devices,
    )

    integer_overrides = (
        ("FRIDAY_LLM_PORT", "llm_port", 1, 65535),
        ("FRIDAY_MODEL_CONTEXT_TOKENS", "context_tokens", 2048, 1_000_000),
        ("FRIDAY_MAX_SEQS", "max_sequences", 1, 256),
        ("FRIDAY_CUDAGRAPH_CAPTURE", "cuda_graph_capture_size", 1, 4096),
        ("FRIDAY_ASR_THREADS", "asr_threads", 1, 256),
    )
    for variable, field, minimum, maximum in integer_overrides:
        value, changed = _env_int(
            env, variable, getattr(profile, field), minimum=minimum, maximum=maximum)
        if changed:
            profile = replace(profile, **{field: value})
            overrides.append(variable)

    reserve, changed = _env_float(
        env, "FRIDAY_TTS_RESERVE_GIB", profile.tts_reserve_gib,
        minimum=0.5, maximum=64.0)
    if changed:
        overrides.append("FRIDAY_TTS_RESERVE_GIB")
        if reserve >= vram_gib:
            raise ValueError(
                "FRIDAY_TTS_RESERVE_GIB must be smaller than LLM GPU memory")
        # A reserve is a lower bound on headroom, not a request to expand the
        # model to consume every other byte.  It may only reduce the tier cap.
        budget = min(profile.llm_memory_budget_gib, vram_gib - reserve)
        profile = replace(
            profile, tts_reserve_gib=reserve,
            gpu_memory_utilization=round(budget / vram_gib, 3),
            llm_memory_budget_gib=round(budget, 3),
            unallocated_gpu_gib=round(vram_gib - budget, 3))

    utilization, changed = _env_float(
        env, "FRIDAY_GPU_UTIL", profile.gpu_memory_utilization,
        minimum=0.30, maximum=0.98)
    if changed:
        budget = vram_gib * utilization
        unallocated = max(vram_gib - budget, 0.0)
        shared_cuda_tts = (
            profile.local_runtime_available
            and profile.tts_device.lower().startswith("cuda")
            and profile.tts_cuda_device in profile.effective_llm_cuda_devices)
        if shared_cuda_tts and unallocated + 0.001 < profile.tts_reserve_gib:
            raise ValueError(
                "FRIDAY_GPU_UTIL leaves less shared GPU memory than "
                "FRIDAY_TTS_RESERVE_GIB requires")
        profile = replace(
            profile, gpu_memory_utilization=round(utilization, 3),
            llm_memory_budget_gib=round(budget, 3),
            unallocated_gpu_gib=round(unallocated, 3))
        overrides.append("FRIDAY_GPU_UTIL")

    mode = env.get("FRIDAY_KV_MODE", "").strip().lower()
    if mode:
        if mode not in {"fast", "long", "huge"}:
            raise ValueError("FRIDAY_KV_MODE must be fast, long, or huge")
        profile = replace(profile, kv_mode=mode)
        overrides.append("FRIDAY_KV_MODE")
    vision_mode, vision_changed = _env_auto_bool(
        env, "FRIDAY_NATIVE_VISION")
    if vision_changed:
        overrides.append("FRIDAY_NATIVE_VISION")
    vision_eligible = bool(
        cuda is not None and vram_gib >= NATIVE_VISION_MIN_MEMORY_GIB)
    known_vision_checkpoint = qwen_model == DEFAULT_QWEN_MODEL
    native_vision = (
        vision_mode == "enabled"
        or (vision_mode == "auto" and vision_eligible
            and known_vision_checkpoint))
    if native_vision and not vision_eligible:
        raise ValueError(
            "native vision requires at least 30 GiB on every selected model "
            "rank; retain OCR or select a larger homogeneous LLM device set")
    if native_vision:
        if vram_gib >= 60:
            max_images, max_side, gpu_reserve = 4, 2048, 8.0
        elif vram_gib >= 46:
            max_images, max_side, gpu_reserve = 2, 1536, 4.0
        else:
            # The installed launcher's language-only mode saves roughly
            # 2.7 GiB. Round upward so the vision tower is not admitted against
            # a nominal reserve smaller than its observed class of footprint.
            max_images, max_side, gpu_reserve = 1, 1024, 3.0
        required_budget = 22.5 + gpu_reserve
        shared_cuda_tts = (
            profile.tts_device.lower().startswith("cuda")
            and profile.tts_cuda_device in profile.effective_llm_cuda_devices)
        shared_speech_is_explicit = bool(
            explicit_tts_device or tts_changed
            or "FRIDAY_TTS_RESERVE_GIB" in overrides)
        maximum_budget = vram_gib - (
            profile.tts_reserve_gib if shared_cuda_tts else 0.0)
        if (required_budget > maximum_budget + 0.001
                and shared_cuda_tts and not shared_speech_is_explicit):
            # On the smallest vision-capable tier, retain speech through its
            # CPU path rather than silently dropping native vision or
            # overcommitting the only GPU. Explicit speech placement still has
            # priority below and never gets rewritten.
            profile = replace(
                profile, tts_device="cpu", tts_cuda_device=None,
                tts_reserve_gib=0.0)
            shared_cuda_tts = False
            maximum_budget = vram_gib
            warnings.append(
                "speech synthesis moved to CPU to preserve the bounded native-"
                "vision GPU envelope")
        if required_budget > maximum_budget + 0.001:
            if vision_mode == "enabled":
                raise ValueError(
                    "native vision and shared speech reserves exceed selected "
                    "model-rank memory")
            native_vision = False
            warnings.append(
                "native vision was disabled because explicit shared speech "
                "headroom takes priority")
        if ("FRIDAY_GPU_UTIL" in overrides
                and profile.llm_memory_budget_gib + 0.001 < required_budget):
            if vision_mode == "enabled":
                raise ValueError(
                    "FRIDAY_GPU_UTIL is below the native-vision memory envelope")
            native_vision = False
            warnings.append(
                "native vision was disabled because the explicit model memory "
                "limit takes priority")
        if native_vision:
            budget = max(profile.llm_memory_budget_gib, required_budget)
            profile = replace(
                profile,
                name=f"{profile.name}-native-vision",
                gpu_memory_utilization=round(budget / vram_gib, 3),
                llm_memory_budget_gib=round(budget, 3),
                unallocated_gpu_gib=round(vram_gib - budget, 3),
                native_vision_enabled=True,
                native_vision_max_images=max_images,
                native_vision_max_side=max_side,
                native_vision_gpu_reserve_gib=gpu_reserve,
                native_vision_host_reserve_mib=2048,
            )
    # Some automatic capability decisions happen after the base profile is
    # constructed.  Carry their diagnostics into the immutable result instead
    # of silently dropping warnings appended during those decisions.
    return replace(
        profile, overrides=tuple(overrides), warnings=tuple(warnings))


_VLLM_ONLY_VARIABLES = (
    "FRIDAY_GPU_UTIL", "FRIDAY_KV_MODE", "FRIDAY_CUDAGRAPH_CAPTURE",
    "FRIDAY_LLM_CUDA_DEVICES", "FRIDAY_TTS_CUDA_DEVICES",
    "FRIDAY_TTS_RESERVE_GIB", "FRIDAY_LLM_EXTRA_ARGS",
)


def _reject_vllm_only_overrides(environment: Mapping[str, str], *,
                                engine: str, enforce: bool) -> None:
    if not enforce:
        return
    for name in _VLLM_ONLY_VARIABLES:
        if environment.get(name, "").strip():
            raise ValueError(f"{name} requires the vLLM engine, not {engine}")
    vision = environment.get("FRIDAY_NATIVE_VISION", "").strip().lower()
    if vision in {"1", "true", "yes", "on", "enabled"}:
        raise ValueError(
            f"FRIDAY_NATIVE_VISION=enabled requires the vLLM engine, not {engine}")
    tts = environment.get("FRIDAY_TTS_DEVICE", "").strip().lower()
    if tts.startswith("cuda"):
        raise ValueError(
            f"FRIDAY_TTS_DEVICE=cuda requires the vLLM engine, not {engine}")


def _portable_backend(snapshot: HardwareSnapshot, engine: str,
                      requested: str, cuda_devices: list[Accelerator]
                      ) -> tuple[str, Accelerator | None]:
    """Choose the compute backend and the accelerator that funds the budget."""
    backends = engine_backends(engine, snapshot.platform, snapshot.arch)
    if engine == "mlx_lm":
        metal = next((item for item in snapshot.accelerators
                      if item.backend == "metal"), None)
        return "metal", metal
    if requested:
        if requested not in backends:
            available = ", ".join(sorted(backends)) or "none"
            raise ValueError(
                f"FRIDAY_LLM_BACKEND={requested} is not available for "
                f"{snapshot.platform}/{snapshot.arch}; available: {available}")
        backend = requested
    elif cuda_devices and "cuda" in backends:
        backend = "cuda"
    elif any(item.backend == "rocm" for item in snapshot.accelerators) \
            and "rocm" in backends:
        backend = "rocm"
    elif any(item.backend in {"vulkan", "rocm", "oneapi"}
             for item in snapshot.accelerators) and "vulkan" in backends:
        backend = "vulkan"
    elif "metal" in backends:
        backend = "metal"
    else:
        backend = "cpu"
    if backend == "cuda":
        device = max(cuda_devices, key=lambda item: (
            item.total_memory_bytes, -item.index), default=None)
    elif backend in {"rocm", "vulkan"}:
        candidates = [item for item in snapshot.accelerators
                      if item.backend in {"rocm", "vulkan", "oneapi"}]
        device = max(candidates, key=lambda item: (
            item.total_memory_bytes, -item.index), default=None)
    elif backend == "metal":
        device = next((item for item in snapshot.accelerators
                       if item.backend == "metal"), None)
    else:
        device = None
    return backend, device


def _select_portable_profile(
        snapshot: HardwareSnapshot, env: Mapping[str, str], engine_request: str,
        *, overrides: list[str], warnings: list[str],
        launch_override_fingerprint: str, served_model: str, llm_host: str,
        cuda_devices: list[Accelerator], explicit_tts_device: bool,
        tts_device: str) -> RuntimeProfile:
    apple_silicon = snapshot.platform == "macos" and snapshot.arch == "aarch64"
    if engine_request == "mlx_lm" and not apple_silicon:
        raise ValueError("FRIDAY_LLM_ENGINE=mlx_lm requires Apple Silicon macOS")
    engine = ("mlx_lm" if engine_request == "mlx_lm"
              or (engine_request == "auto" and apple_silicon)
              else "llama_server")
    _reject_vllm_only_overrides(env, engine=engine, enforce=True)
    requested_backend = env.get("FRIDAY_LLM_BACKEND", "").strip().lower()
    if requested_backend:
        overrides.append("FRIDAY_LLM_BACKEND")
    backend, device = _portable_backend(
        snapshot, engine, requested_backend, cuda_devices)
    if device is not None and device.total_memory_bytes > 0 and (
            backend in {"cuda", "rocm", "vulkan"}) and not snapshot.unified_memory:
        total_bytes = device.total_memory_bytes
        budget_bytes = int(total_bytes * (0.90 if backend == "cuda" else 0.85))
        budget_source = f"{backend} device memory"
    elif backend == "metal":
        total_bytes = snapshot.system_memory_bytes
        budget_bytes = int(total_bytes * 0.70)
        budget_source = "unified memory"
    else:
        total_bytes = snapshot.system_memory_bytes
        budget_bytes = int(total_bytes * 0.60)
        budget_source = "system memory"
    tier = select_model_tier(engine, budget_bytes, cpu_only=backend == "cpu")
    if explicit_tts_device and tts_device.lower() != "cpu":
        raise ValueError(
            f"FRIDAY_TTS_DEVICE must be cpu with the {engine} engine")
    cuda_index = device.index if backend == "cuda" and device else None
    asr_threads = min(8, max(2, snapshot.cpu_count // 8))
    if tier is None:
        warnings.append(
            f"no Qwen3 tier fits the {budget_source} budget of "
            f"{budget_bytes / GIB:.1f} GiB; the local model runtime is unavailable")
        profile = RuntimeProfile(
            name="unsupported-local-runtime", source="unsupported",
            hardware=snapshot, qwen_model="", served_model=served_model,
            llm_host=llm_host, llm_port=DEFAULT_LLM_PORT,
            context_tokens=8192, max_sequences=1, gpu_memory_utilization=0.30,
            kv_mode="huge", cuda_graph_capture_size=1, tts_device="cpu",
            asr_threads=asr_threads, llm_cuda_device=None, tts_cuda_device=None,
            llm_memory_budget_gib=0.0, tts_reserve_gib=0.0,
            unallocated_gpu_gib=round(total_bytes / GIB, 3),
            local_runtime_available=False,
            launch_override_fingerprint=launch_override_fingerprint,
            warnings=tuple(warnings), overrides=tuple(overrides),
            engine=engine, engine_backend=backend)
        return profile
    asset, context, sequences = tier
    utilization = round(min(0.98, max(0.30, budget_bytes / total_bytes)), 3)
    profile = RuntimeProfile(
        name=f"{engine.replace('_', '-')}-{backend}-{asset.key}",
        source="hardware", hardware=snapshot,
        qwen_model=f"models/{asset.directory}",
        served_model=(served_model if "FRIDAY_LOCAL_MODEL" in overrides
                      else f"qwen3-{asset.size_label}"),
        llm_host=llm_host, llm_port=DEFAULT_LLM_PORT,
        context_tokens=context, max_sequences=sequences,
        gpu_memory_utilization=utilization, kv_mode="huge",
        cuda_graph_capture_size=1, tts_device="cpu", asr_threads=asr_threads,
        llm_cuda_device=cuda_index, tts_cuda_device=None,
        llm_memory_budget_gib=round(budget_bytes / GIB, 3),
        tts_reserve_gib=0.0,
        unallocated_gpu_gib=round(max(total_bytes - budget_bytes, 0) / GIB, 3),
        local_runtime_available=True,
        launch_override_fingerprint=launch_override_fingerprint,
        llm_cuda_devices=(cuda_index,) if cuda_index is not None else (),
        engine=engine, engine_backend=backend, model_asset=asset.key,
        model_path=f"models/{asset.directory}",
        gpu_layers=-1 if backend != "cpu" else 0)
    integer_overrides = (
        ("FRIDAY_LLM_PORT", "llm_port", 1, 65535),
        ("FRIDAY_MODEL_CONTEXT_TOKENS", "context_tokens", 2048, 1_000_000),
        ("FRIDAY_MAX_SEQS", "max_sequences", 1, 256),
        ("FRIDAY_ASR_THREADS", "asr_threads", 1, 256),
        ("FRIDAY_LLM_GPU_LAYERS", "gpu_layers", -1, 1024),
        ("FRIDAY_LLM_THREADS", "engine_threads", 0, 1024),
    )
    for variable, field, minimum, maximum in integer_overrides:
        value, changed = _env_int(
            env, variable, getattr(profile, field), minimum=minimum,
            maximum=maximum)
        if changed:
            profile = replace(profile, **{field: value})
            overrides.append(variable)
    warnings.append(
        f"{engine} selected {asset.repo} ({asset.size_label}) for a "
        f"{budget_bytes / GIB:.1f} GiB {budget_source} budget")
    return replace(profile, overrides=tuple(overrides), warnings=tuple(warnings))


def _write_runtime_manifest(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(
            target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_runtime_profile(path: str | Path, profile: RuntimeProfile) -> None:
    """Atomically publish the proven profile for inspection and diagnostics."""
    _write_runtime_manifest(path, profile.to_dict())


def write_pending_runtime_profile(path: str | Path) -> None:
    """Durably invalidate a stale active profile before a replacement spawn."""
    _write_runtime_manifest(path, {
        "schema_version": 1,
        "state": "starting",
    })
