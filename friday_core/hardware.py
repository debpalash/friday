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
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


GIB = 1024 ** 3
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
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    pages = int(os.sysconf("SC_PHYS_PAGES"))
    return page_size * pages


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


def detect_hardware(*, probe: Probe | None = None,
                    meminfo_path: str | Path = "/proc/meminfo",
                    drm_root: str | Path = "/sys/class/drm",
                    device_root: str | Path = "/dev") -> HardwareSnapshot:
    """Collect a lightweight inventory without importing a GPU framework."""
    run = probe or _default_probe
    errors: list[str] = []
    accelerators: tuple[Accelerator, ...] = ()
    try:
        output = run([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        accelerators = parse_nvidia_smi(output)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        errors.append(f"nvidia-smi: {str(exc)[:240]}")
    if not accelerators:
        accelerators = _drm_accelerators(Path(drm_root))
    cuda_probe = "available" if any(
        item.backend == "cuda" for item in accelerators
    ) else ("probe_failed" if (Path(device_root) / "nvidia0").exists()
            else "absent")
    return HardwareSnapshot(
        cpu_count=max(os.cpu_count() or 1, 1),
        system_memory_bytes=_system_memory_bytes(Path(meminfo_path)),
        accelerators=accelerators,
        detection_errors=tuple(errors),
        cuda_probe=cuda_probe,
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
