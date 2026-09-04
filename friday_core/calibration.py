"""Bounded boot calibration and durable last-known-good runtime policy.

This module contains no process-launch code.  It validates measured evidence,
persists only private calibration state, and produces a small deterministic
candidate ladder for the external supervisor.
"""

from __future__ import annotations

import json
import hashlib
import os
import stat
import statistics
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from friday_host import fs

from .hardware import RuntimeProfile


CALIBRATION_SCHEMA_VERSION = 1
RECOVERY_SCHEMA_VERSION = 1
PERFORMANCE_SCHEMA_VERSION = 1
PERFORMANCE_PORTFOLIO_SCHEMA_VERSION = 1
MAX_LAST_KNOWN_GOOD_AGE_SECONDS = 90 * 24 * 60 * 60
MAX_PERFORMANCE_AGE_SECONDS = 30 * 24 * 60 * 60
MAX_PERFORMANCE_PROFILES = 12
MAX_BOOT_CANDIDATES = 3
BOOT_BACKOFF_BASE_SECONDS = 15
BOOT_BACKOFF_MAX_SECONDS = 15 * 60
BOOT_STABILITY_SECONDS = 120


def _boot_session_id(
        path: Path = Path("/proc/sys/kernel/random/boot_id")) -> str:
    try:
        value = path.read_text().strip()
    except OSError:
        value = "unavailable"
    return hashlib.sha256(value.encode()).hexdigest()


def _recovery_now() -> float:
    return time.monotonic()


@dataclass(frozen=True)
class BootCalibrationEvidence:
    """Privacy-safe measurements proving one model process became usable."""

    startup_ms: int
    identity_probe_ms: int
    tokenization_probe_ms: int
    observed_context_tokens: int
    authenticated: bool = True
    model_identity_verified: bool = True
    tokenization_verified: bool = True
    credential_rejection_verified: bool = True
    startup_measured: bool = True
    native_vision_required: bool = False
    native_vision_verified: bool = True
    native_vision_score_verified: bool = False
    native_vision_probe_ms: int = 0
    native_vision_vram_mib: int = 0
    native_vision_vram_verified: bool = True

    def to_record(self) -> dict[str, Any]:
        return {
            "startup_ms": max(0, int(self.startup_ms)),
            "identity_probe_ms": max(0, int(self.identity_probe_ms)),
            "tokenization_probe_ms": max(0, int(self.tokenization_probe_ms)),
            "observed_context_tokens": max(
                0, int(self.observed_context_tokens)),
            "authenticated": bool(self.authenticated),
            "model_identity_verified": bool(self.model_identity_verified),
            "tokenization_verified": bool(self.tokenization_verified),
            "credential_rejection_verified": bool(
                self.credential_rejection_verified),
            "startup_measured": bool(self.startup_measured),
            "native_vision_required": bool(self.native_vision_required),
            "native_vision_verified": bool(self.native_vision_verified),
            "native_vision_score_verified": bool(
                self.native_vision_score_verified),
            "native_vision_probe_ms": max(
                0, int(self.native_vision_probe_ms)),
            "native_vision_vram_mib": max(
                0, int(self.native_vision_vram_mib)),
            "native_vision_vram_verified": bool(
                self.native_vision_vram_verified),
        }


class PerformanceCalibrationStore:
    """Private, profile-bound empirical latency/throughput evidence."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @staticmethod
    def _sample(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != {
                "first_token_ms", "completion_tokens",
                "decode_tokens_per_second", "total_ms"}:
            raise ValueError("performance sample shape is invalid")
        return {
            "first_token_ms": _plain_float(
                value["first_token_ms"], minimum=0, maximum=600_000),
            "completion_tokens": _plain_int(
                value["completion_tokens"], minimum=16, maximum=4096),
            "decode_tokens_per_second": _plain_float(
                value["decode_tokens_per_second"],
                minimum=0.01, maximum=100_000),
            "total_ms": _plain_float(
                value["total_ms"], minimum=0.01, maximum=3_600_000),
        }

    def record(
        self, profile: RuntimeProfile, *, runtime_identity: str,
        samples: list[dict[str, Any]], qwen_vram_mib: int,
        now: float | None = None,
    ) -> dict[str, Any]:
        if (not isinstance(runtime_identity, str)
                or len(runtime_identity) != 64
                or any(char not in "0123456789abcdef"
                       for char in runtime_identity)):
            raise ValueError("runtime identity is invalid")
        if not isinstance(samples, list) or not 1 <= len(samples) <= 10:
            raise ValueError("performance sample count is invalid")
        normalized = [self._sample(item) for item in samples]
        vram = _plain_int(qwen_vram_mib, minimum=0, maximum=1_048_576)
        median_first = float(statistics.median(
            item["first_token_ms"] for item in normalized))
        median_decode = float(statistics.median(
            item["decode_tokens_per_second"] for item in normalized))
        record = {
            "schema_version": PERFORMANCE_SCHEMA_VERSION,
            "recorded_at": float(time.time() if now is None else now),
            "hardware_fingerprint": profile.hardware_fingerprint,
            "family_fingerprint": profile.family_fingerprint,
            "profile_fingerprint": profile.fingerprint,
            "runtime_identity_sha256": runtime_identity,
            "context_tokens": profile.context_tokens,
            "max_sequences": profile.max_sequences,
            "gpu_memory_utilization": profile.gpu_memory_utilization,
            "kv_mode": profile.kv_mode,
            "qwen_vram_mib": vram,
            "samples": normalized,
            "median_first_token_ms": median_first,
            "median_decode_tokens_per_second": median_decode,
        }
        _atomic_write_private_json(self.path, record)
        return self.public_status(profile, now=record["recorded_at"])

    def public_status(
        self, profile: RuntimeProfile, *, now: float | None = None,
    ) -> dict[str, Any]:
        read_status, record = _read_private_json(self.path)
        if read_status == "missing":
            return {"state": "missing"}
        if read_status != "ok" or record is None:
            return {"state": "invalid"}
        if (record.get("schema_version") != PERFORMANCE_SCHEMA_VERSION
                or record.get("hardware_fingerprint")
                    != profile.hardware_fingerprint
                or record.get("family_fingerprint")
                    != profile.family_fingerprint
                or record.get("profile_fingerprint") != profile.fingerprint):
            return {"state": "new_profile"}
        try:
            recorded_at = _plain_float(
                record.get("recorded_at"), minimum=0, maximum=10 ** 12)
            runtime_identity = record.get("runtime_identity_sha256")
            if (not isinstance(runtime_identity, str)
                    or len(runtime_identity) != 64
                    or any(char not in "0123456789abcdef"
                           for char in runtime_identity)):
                raise ValueError("invalid runtime identity")
            if (_plain_int(record.get("context_tokens"), minimum=2048,
                           maximum=1_000_000) != profile.context_tokens
                    or _plain_int(record.get("max_sequences"), minimum=1,
                                  maximum=256) != profile.max_sequences
                    or abs(_plain_float(
                        record.get("gpu_memory_utilization"), minimum=0.30,
                        maximum=0.98) - profile.gpu_memory_utilization) > 0.0005
                    or record.get("kv_mode") != profile.kv_mode):
                raise ValueError("performance tuning identity mismatch")
            raw_samples = record.get("samples")
            if not isinstance(raw_samples, list) or not 1 <= len(raw_samples) <= 10:
                raise ValueError("invalid sample count")
            samples = [self._sample(item) for item in raw_samples]
            median_first = float(statistics.median(
                item["first_token_ms"] for item in samples))
            median_decode = float(statistics.median(
                item["decode_tokens_per_second"] for item in samples))
            if (abs(_plain_float(
                    record.get("median_first_token_ms"), minimum=0,
                    maximum=600_000) - median_first) > 0.001
                    or abs(_plain_float(
                        record.get("median_decode_tokens_per_second"),
                        minimum=0.01, maximum=100_000) - median_decode) > 0.001):
                raise ValueError("performance aggregate mismatch")
            vram = _plain_int(
                record.get("qwen_vram_mib"), minimum=0, maximum=1_048_576)
        except (TypeError, ValueError):
            return {"state": "invalid"}
        current = float(time.time() if now is None else now)
        if recorded_at > current + 300:
            return {"state": "invalid"}
        age = max(0, int(current - recorded_at))
        if age > MAX_PERFORMANCE_AGE_SECONDS:
            return {"state": "expired"}
        return {
            "state": "measured",
            "age_seconds": age,
            "sample_count": len(samples),
            "median_first_token_ms": round(median_first, 1),
            "median_decode_tokens_per_second": round(median_decode, 1),
            "qwen_vram_mib": vram,
        }


@dataclass(frozen=True)
class PerformanceRecommendation:
    status: str
    profile: RuntimeProfile | None = None
    median_first_token_ms: float | None = None
    median_decode_tokens_per_second: float | None = None
    qwen_vram_mib: int | None = None


class PerformancePortfolioStore:
    """Comparable fixed-canary evidence for multiple exact runtime profiles."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @staticmethod
    def _entry(
        profile: RuntimeProfile, *, runtime_identity: str,
        samples: list[dict[str, Any]], qwen_vram_mib: int,
        now: float,
    ) -> dict[str, Any]:
        if (not isinstance(runtime_identity, str)
                or len(runtime_identity) != 64
                or any(char not in "0123456789abcdef"
                       for char in runtime_identity)):
            raise ValueError("runtime identity is invalid")
        if not isinstance(samples, list) or not 1 <= len(samples) <= 10:
            raise ValueError("performance sample count is invalid")
        normalized = [PerformanceCalibrationStore._sample(item)
                      for item in samples]
        median_first = float(statistics.median(
            item["first_token_ms"] for item in normalized))
        median_decode = float(statistics.median(
            item["decode_tokens_per_second"] for item in normalized))
        return {
            "profile_fingerprint": profile.fingerprint,
            "profile_name": profile.name[:96],
            "tuning": _profile_tuning(profile),
            "recorded_at": float(now),
            "runtime_identity_sha256": runtime_identity,
            "qwen_vram_mib": _plain_int(
                qwen_vram_mib, minimum=0, maximum=1_048_576),
            "samples": normalized,
            "median_first_token_ms": median_first,
            "median_decode_tokens_per_second": median_decode,
        }

    def record(
        self, profile: RuntimeProfile, *, runtime_identity: str,
        samples: list[dict[str, Any]], qwen_vram_mib: int,
        now: float | None = None,
    ) -> dict[str, Any]:
        if profile.overrides:
            raise ValueError(
                "explicit override profiles cannot enter automatic evidence")
        recorded_at = float(time.time() if now is None else now)
        new_entry = self._entry(
            profile, runtime_identity=runtime_identity, samples=samples,
            qwen_vram_mib=qwen_vram_mib, now=recorded_at)
        status, current = _read_private_json(self.path)
        entries: list[dict[str, Any]] = []
        if (status == "ok" and current is not None
                and current.get("schema_version")
                    == PERFORMANCE_PORTFOLIO_SCHEMA_VERSION
                and current.get("hardware_fingerprint")
                    == profile.hardware_fingerprint
                and current.get("family_fingerprint")
                    == profile.family_fingerprint
                and isinstance(current.get("entries"), list)):
            # Validate existing evidence before carrying it forward. A damaged
            # portfolio is repaired from this new authenticated measurement,
            # never partially trusted.
            resolution, validated = self._validated_entries(
                profile, (profile,), now=recorded_at, require_candidate=False)
            if resolution == "measured":
                entries = [dict(item) for item in validated]
        entries = [item for item in entries
                   if item.get("profile_fingerprint") != profile.fingerprint]
        entries.append(new_entry)
        entries.sort(key=lambda item: float(item["recorded_at"]), reverse=True)
        record = {
            "schema_version": PERFORMANCE_PORTFOLIO_SCHEMA_VERSION,
            "hardware_fingerprint": profile.hardware_fingerprint,
            "family_fingerprint": profile.family_fingerprint,
            "entries": entries[:MAX_PERFORMANCE_PROFILES],
        }
        _atomic_write_private_json(self.path, record)
        return self.public_status(profile, (profile,), now=recorded_at)

    def _validated_entries(
        self, proposed: RuntimeProfile,
        candidates: tuple[RuntimeProfile, ...], *,
        now: float | None = None, require_candidate: bool = True,
    ) -> tuple[str, list[dict[str, Any]]]:
        status, record = _read_private_json(self.path)
        if status == "missing":
            return "missing", []
        if status != "ok" or record is None:
            return "invalid", []
        if (record.get("schema_version")
                != PERFORMANCE_PORTFOLIO_SCHEMA_VERSION):
            return "invalid", []
        if (record.get("hardware_fingerprint")
                != proposed.hardware_fingerprint
                or record.get("family_fingerprint")
                    != proposed.family_fingerprint):
            return "new_machine", []
        raw_entries = record.get("entries")
        if (not isinstance(raw_entries, list)
                or not 1 <= len(raw_entries) <= MAX_PERFORMANCE_PROFILES):
            return "invalid", []
        candidate_map = {item.fingerprint: item for item in candidates}
        current = float(time.time() if now is None else now)
        seen: set[str] = set()
        valid: list[dict[str, Any]] = []
        try:
            for raw in raw_entries:
                if not isinstance(raw, dict) or set(raw) != {
                        "profile_fingerprint", "profile_name", "tuning",
                        "recorded_at", "runtime_identity_sha256",
                        "qwen_vram_mib", "samples",
                        "median_first_token_ms",
                        "median_decode_tokens_per_second"}:
                    raise ValueError("portfolio entry shape is invalid")
                fingerprint = raw["profile_fingerprint"]
                if (not isinstance(fingerprint, str)
                        or len(fingerprint) != 64
                        or any(char not in "0123456789abcdef"
                               for char in fingerprint)
                        or fingerprint in seen):
                    raise ValueError("portfolio profile identity is invalid")
                seen.add(fingerprint)
                if (not isinstance(raw["profile_name"], str)
                        or not 1 <= len(raw["profile_name"]) <= 96):
                    raise ValueError("portfolio profile name is invalid")
                runtime_identity = raw["runtime_identity_sha256"]
                if (not isinstance(runtime_identity, str)
                        or len(runtime_identity) != 64
                        or any(char not in "0123456789abcdef"
                               for char in runtime_identity)):
                    raise ValueError("portfolio runtime identity is invalid")
                recorded_at = _plain_float(
                    raw["recorded_at"], minimum=0, maximum=10 ** 12)
                if recorded_at > current + 300:
                    raise ValueError("portfolio entry is from the future")
                samples = raw["samples"]
                if not isinstance(samples, list) or not 1 <= len(samples) <= 10:
                    raise ValueError("portfolio sample count is invalid")
                normalized = [PerformanceCalibrationStore._sample(item)
                              for item in samples]
                median_first = float(statistics.median(
                    item["first_token_ms"] for item in normalized))
                median_decode = float(statistics.median(
                    item["decode_tokens_per_second"] for item in normalized))
                if (abs(_plain_float(
                        raw["median_first_token_ms"], minimum=0,
                        maximum=600_000) - median_first) > 0.001
                        or abs(_plain_float(
                            raw["median_decode_tokens_per_second"],
                            minimum=0.01, maximum=100_000)
                            - median_decode) > 0.001):
                    raise ValueError("portfolio aggregate is invalid")
                vram = _plain_int(
                    raw["qwen_vram_mib"], minimum=0, maximum=1_048_576)
                tuning = _validated_profile_tuning(raw["tuning"])
                candidate = candidate_map.get(fingerprint)
                if require_candidate and candidate is None:
                    continue
                if candidate is not None and tuning != _profile_tuning(candidate):
                    raise ValueError("portfolio tuning identity mismatch")
                if current - recorded_at > MAX_PERFORMANCE_AGE_SECONDS:
                    continue
                valid.append(dict(raw) | {
                    "tuning": tuning,
                    "recorded_at": recorded_at,
                    "qwen_vram_mib": vram,
                    "median_first_token_ms": median_first,
                    "median_decode_tokens_per_second": median_decode,
                })
        except (TypeError, ValueError):
            return "invalid", []
        return ("measured" if valid else "expired"), valid

    @staticmethod
    def _recommendation(
        proposed: RuntimeProfile, candidates: tuple[RuntimeProfile, ...],
        entries: list[dict[str, Any]], *, preference: str,
    ) -> PerformanceRecommendation:
        if preference not in {"reasoning", "throughput"}:
            raise ValueError("unknown performance preference")
        candidate_map = {item.fingerprint: item for item in candidates}
        if preference == "reasoning":
            key = lambda item: (
                candidate_map[item["profile_fingerprint"]].context_tokens,
                candidate_map[item["profile_fingerprint"]].max_sequences,
                item["median_decode_tokens_per_second"],
                -item["median_first_token_ms"],
                -item["qwen_vram_mib"],
            )
        else:
            key = lambda item: (
                item["median_decode_tokens_per_second"],
                -item["median_first_token_ms"],
                candidate_map[item["profile_fingerprint"]].context_tokens,
                candidate_map[item["profile_fingerprint"]].max_sequences,
                -item["qwen_vram_mib"],
            )
        selected = max(entries, key=key)
        return PerformanceRecommendation(
            "measured", candidate_map[selected["profile_fingerprint"]],
            median_first_token_ms=selected["median_first_token_ms"],
            median_decode_tokens_per_second=
                selected["median_decode_tokens_per_second"],
            qwen_vram_mib=selected["qwen_vram_mib"])

    def recommend(
        self, proposed: RuntimeProfile,
        candidates: tuple[RuntimeProfile, ...], *,
        preference: str = "reasoning", now: float | None = None,
    ) -> PerformanceRecommendation:
        if preference not in {"reasoning", "throughput"}:
            raise ValueError("unknown performance preference")
        status, entries = self._validated_entries(
            proposed, candidates, now=now)
        if status != "measured":
            return PerformanceRecommendation(status)
        return self._recommendation(
            proposed, candidates, entries, preference=preference)

    def public_status(
        self, proposed: RuntimeProfile,
        candidates: tuple[RuntimeProfile, ...], *,
        now: float | None = None,
    ) -> dict[str, Any]:
        status, entries = self._validated_entries(
            proposed, candidates, now=now)
        if status != "measured":
            return {"state": status}
        # Select both views from one validated snapshot. Atomic replacement
        # keeps each read coherent; avoiding another read prevents a concurrent
        # benchmark write from mixing two portfolio generations.
        reasoning = self._recommendation(
            proposed, candidates, entries, preference="reasoning")
        throughput = self._recommendation(
            proposed, candidates, entries, preference="throughput")

        def summary(value: PerformanceRecommendation) -> dict[str, Any]:
            profile = value.profile
            assert profile is not None
            return {
                "matches_proposed": profile.fingerprint == proposed.fingerprint,
                "context_tokens": profile.context_tokens,
                "max_sequences": profile.max_sequences,
                "kv_mode": profile.kv_mode,
                "median_first_token_ms": round(
                    float(value.median_first_token_ms), 1),
                "median_decode_tokens_per_second": round(
                    float(value.median_decode_tokens_per_second), 1),
                "qwen_vram_mib": int(value.qwen_vram_mib),
            }

        return {
            "state": "measured",
            "profile_count": len(entries),
            "reasoning": summary(reasoning),
            "throughput": summary(throughput),
        }


@dataclass(frozen=True)
class LastKnownGoodResolution:
    status: str
    profile: RuntimeProfile | None = None


def _read_private_json(path: Path) -> tuple[str, dict[str, Any] | None]:
    """Read an owner-private regular file without following a final symlink."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return "missing", None
    except OSError:
        return "invalid", None
    if (not stat.S_ISREG(metadata.st_mode)
            or not fs.owned_by_caller(metadata)
            or not fs.private_mode_ok(metadata)):
        return "insecure", None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "invalid", None
    return ("ok", value) if isinstance(value, dict) else ("invalid", None)


def _atomic_write_private_json(path: Path, value: dict[str, Any]) -> None:
    """Durably replace a JSON record with mode 0600 and an fsynced directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        fs.chmod_private(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fs.fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _plain_int(value: Any, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected integer")
    if not minimum <= value <= maximum:
        raise ValueError("integer outside safe range")
    return value


def _plain_float(value: Any, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected number")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError("number outside safe range")
    return result


def _profile_tuning(profile: RuntimeProfile) -> dict[str, Any]:
    return {
        "context_tokens": profile.context_tokens,
        "max_sequences": profile.max_sequences,
        "gpu_memory_utilization": profile.gpu_memory_utilization,
        "kv_mode": profile.kv_mode,
        "cuda_graph_capture_size": profile.cuda_graph_capture_size,
        "asr_threads": profile.asr_threads,
        "llm_memory_budget_gib": profile.llm_memory_budget_gib,
        "tts_reserve_gib": profile.tts_reserve_gib,
        "unallocated_gpu_gib": profile.unallocated_gpu_gib,
    }


def _native_vision_vram_ceiling_mib(profile: RuntimeProfile) -> int:
    """Bound measured allocation by the exact per-rank launch envelope."""
    return (int(float(profile.llm_memory_budget_gib)
                * max(1, int(profile.tensor_parallel_size)) * 1024) + 512)


def _validated_profile_tuning(value: Any) -> dict[str, Any]:
    """Validate carried evidence even when its candidate is not in this view."""
    expected = {
        "context_tokens", "max_sequences", "gpu_memory_utilization",
        "kv_mode", "cuda_graph_capture_size", "asr_threads",
        "llm_memory_budget_gib", "tts_reserve_gib",
        "unallocated_gpu_gib",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("portfolio tuning shape is invalid")
    kv_mode = value["kv_mode"]
    if kv_mode not in {"fast", "long", "huge"}:
        raise ValueError("portfolio KV mode is invalid")
    return {
        "context_tokens": _plain_int(
            value["context_tokens"], minimum=2048, maximum=1_000_000),
        "max_sequences": _plain_int(
            value["max_sequences"], minimum=1, maximum=256),
        "gpu_memory_utilization": _plain_float(
            value["gpu_memory_utilization"], minimum=0.30, maximum=0.98),
        "kv_mode": kv_mode,
        "cuda_graph_capture_size": _plain_int(
            value["cuda_graph_capture_size"], minimum=1, maximum=4096),
        "asr_threads": _plain_int(
            value["asr_threads"], minimum=1, maximum=256),
        "llm_memory_budget_gib": _plain_float(
            value["llm_memory_budget_gib"], minimum=0, maximum=1024),
        "tts_reserve_gib": _plain_float(
            value["tts_reserve_gib"], minimum=0, maximum=64),
        "unallocated_gpu_gib": _plain_float(
            value["unallocated_gpu_gib"], minimum=0, maximum=1024),
    }


def _calibration_record(
        profile: RuntimeProfile, evidence: BootCalibrationEvidence, *,
        stability_state: str, now: float | None) -> dict[str, Any]:
    proof = evidence.to_record()
    native_vram_valid = bool(
        not profile.native_vision_enabled
        or (0 < proof["native_vision_vram_mib"]
            <= _native_vision_vram_ceiling_mib(profile)))
    if (not proof["authenticated"]
            or not proof["model_identity_verified"]
            or not proof["tokenization_verified"]
            or not proof["credential_rejection_verified"]
            or not proof["startup_measured"]
            or (profile.native_vision_enabled
                and (not proof["native_vision_required"]
                     or not proof["native_vision_verified"]
                     or not proof["native_vision_score_verified"]
                     or not proof["native_vision_vram_verified"]
                     or not native_vram_valid))
            or proof["observed_context_tokens"] < profile.context_tokens):
        raise ValueError("last-known-good evidence is incomplete")
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "recorded_at": float(time.time() if now is None else now),
        "hardware_fingerprint": profile.hardware_fingerprint,
        "family_fingerprint": profile.family_fingerprint,
        "profile_fingerprint": profile.fingerprint,
        "profile_name": profile.name[:96],
        "tuning": _profile_tuning(profile),
        "evidence": proof,
        "stability_state": stability_state,
    }


class LastKnownGoodStore:
    """Private evidence store scoped to one exact machine/profile family."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, profile: RuntimeProfile,
               evidence: BootCalibrationEvidence, *,
               now: float | None = None) -> bool:
        # An explicit operator choice may become active, but must not overwrite
        # the automatic baseline or later be silently reused without overrides.
        if profile.overrides:
            return False
        record = _calibration_record(
            profile, evidence, stability_state="stable", now=now)
        _atomic_write_private_json(self.path, record)
        return True

    def resolve(self, proposed: RuntimeProfile, *,
                now: float | None = None) -> LastKnownGoodResolution:
        if proposed.overrides:
            return LastKnownGoodResolution("overrides_active")
        read_status, record = _read_private_json(self.path)
        if read_status != "ok" or record is None:
            return LastKnownGoodResolution(read_status)
        try:
            if record.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
                return LastKnownGoodResolution("unsupported_schema")
            stability_state = record.get("stability_state")
            if stability_state != "stable":
                return LastKnownGoodResolution("invalid")
            if record.get("hardware_fingerprint") != proposed.hardware_fingerprint:
                return LastKnownGoodResolution("hardware_mismatch")
            if record.get("family_fingerprint") != proposed.family_fingerprint:
                return LastKnownGoodResolution("profile_family_mismatch")
            recorded_at = _plain_float(
                record.get("recorded_at"), minimum=0, maximum=10 ** 12)
            current_time = time.time() if now is None else now
            if recorded_at > current_time + 300:
                return LastKnownGoodResolution("future_record")
            if current_time - recorded_at > MAX_LAST_KNOWN_GOOD_AGE_SECONDS:
                return LastKnownGoodResolution("expired")
            evidence = record.get("evidence")
            tuning = record.get("tuning")
            if not isinstance(evidence, dict) or not isinstance(tuning, dict):
                raise ValueError("missing record section")
            if not all(evidence.get(field) is True for field in (
                    "authenticated", "model_identity_verified",
                    "tokenization_verified",
                    "credential_rejection_verified", "startup_measured")):
                return LastKnownGoodResolution("unverified")
            if (proposed.native_vision_enabled
                    and (evidence.get("native_vision_required") is not True
                         or evidence.get("native_vision_verified") is not True
                         or evidence.get("native_vision_score_verified")
                         is not True
                         or evidence.get("native_vision_vram_verified")
                         is not True)):
                return LastKnownGoodResolution("unverified")
            if proposed.native_vision_enabled:
                vision_probe_ms = _plain_int(
                    evidence.get("native_vision_probe_ms"),
                    minimum=0, maximum=600_000)
                vision_vram_mib = _plain_int(
                    evidence.get("native_vision_vram_mib"),
                    minimum=1, maximum=1_048_576)
                if (vision_probe_ms > 600_000
                        or vision_vram_mib
                        > _native_vision_vram_ceiling_mib(proposed)):
                    return LastKnownGoodResolution("unverified")
            context = _plain_int(
                tuning.get("context_tokens"), minimum=2048, maximum=1_000_000)
            sequences = _plain_int(
                tuning.get("max_sequences"), minimum=1, maximum=256)
            capture = _plain_int(
                tuning.get("cuda_graph_capture_size"), minimum=1, maximum=4096)
            asr_threads = _plain_int(
                tuning.get("asr_threads"), minimum=1, maximum=256)
            observed_context = _plain_int(
                evidence.get("observed_context_tokens"),
                minimum=1, maximum=10_000_000)
            if (context > proposed.context_tokens
                    or sequences > proposed.max_sequences
                    or capture > proposed.cuda_graph_capture_size
                    or asr_threads > proposed.asr_threads
                    or observed_context < context):
                return LastKnownGoodResolution("not_a_safe_degradation")
            kv_mode = tuning.get("kv_mode")
            if kv_mode not in {"fast", "long", "huge"}:
                raise ValueError("invalid KV mode")
            utilization = _plain_float(
                tuning.get("gpu_memory_utilization"), minimum=0.30, maximum=0.98)
            budget = _plain_float(
                tuning.get("llm_memory_budget_gib"), minimum=0, maximum=1024)
            reserve = _plain_float(
                tuning.get("tts_reserve_gib"), minimum=0, maximum=1024)
            unallocated = _plain_float(
                tuning.get("unallocated_gpu_gib"), minimum=0, maximum=1024)
            if (kv_mode != proposed.kv_mode
                    or utilization > proposed.gpu_memory_utilization + 0.0005
                    or budget > proposed.llm_memory_budget_gib + 0.001
                    or reserve + 0.001 < proposed.tts_reserve_gib
                    or unallocated + 0.001 < proposed.unallocated_gpu_gib):
                return LastKnownGoodResolution("not_a_safe_degradation")
            cuda = next((item for item in proposed.hardware.accelerators
                         if item.backend == "cuda"
                         and item.index == proposed.llm_cuda_device), None)
            if cuda is not None:
                total_gib = cuda.total_memory_bytes / (1024 ** 3)
                if (budget > total_gib + 0.01 or reserve > total_gib + 0.01
                        or unallocated > total_gib + 0.01
                        or abs((budget + unallocated) - total_gib) > 0.02):
                    raise ValueError("invalid memory tuning")
            candidate = replace(
                proposed,
                name=str(record.get("profile_name") or proposed.name)[:96],
                source="last-known-good",
                context_tokens=context,
                max_sequences=sequences,
                gpu_memory_utilization=utilization,
                kv_mode=kv_mode,
                cuda_graph_capture_size=capture,
                asr_threads=asr_threads,
                llm_memory_budget_gib=budget,
                tts_reserve_gib=reserve,
                unallocated_gpu_gib=unallocated,
                warnings=proposed.warnings + (
                    "using authenticated last-known-good boot calibration",),
            )
            if candidate.fingerprint != record.get("profile_fingerprint"):
                return LastKnownGoodResolution("profile_fingerprint_mismatch")
            return LastKnownGoodResolution("usable", candidate)
        except (TypeError, ValueError):
            return LastKnownGoodResolution("invalid")

    def public_status(self, proposed: RuntimeProfile, *,
                      now: float | None = None) -> dict[str, Any]:
        resolution = self.resolve(proposed, now=now)
        return {
            "state": resolution.status,
            "usable": resolution.profile is not None,
            "matches_proposed": bool(
                resolution.profile is not None
                and resolution.profile.fingerprint == proposed.fingerprint),
        }


class PendingCalibrationStore:
    """Separate probation record that cannot overwrite a stable baseline."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def stage(self, profile: RuntimeProfile,
              evidence: BootCalibrationEvidence, *,
              runtime_identity: str,
              now: float | None = None) -> bool:
        if profile.overrides:
            return False
        if not runtime_identity or len(runtime_identity) > 128:
            raise ValueError("invalid pending runtime identity")
        record = _calibration_record(
            profile, evidence, stability_state="probation", now=now)
        record["runtime_identity"] = runtime_identity
        _atomic_write_private_json(self.path, record)
        return True

    def promote(self, profile: RuntimeProfile,
                stable_store: LastKnownGoodStore, *,
                runtime_identity: str) -> bool:
        status, record = _read_private_json(self.path)
        if status != "ok" or record is None:
            return False
        evidence = record.get("evidence")
        if (record.get("schema_version") != CALIBRATION_SCHEMA_VERSION
                or record.get("stability_state") != "probation"
                or record.get("hardware_fingerprint")
                != profile.hardware_fingerprint
                or record.get("family_fingerprint") != profile.family_fingerprint
                or record.get("profile_fingerprint") != profile.fingerprint
                or record.get("runtime_identity") != runtime_identity
                or record.get("tuning") != _profile_tuning(profile)
                or not isinstance(evidence, dict)
                or not all(evidence.get(field) is True for field in (
                    "authenticated", "model_identity_verified",
                    "tokenization_verified", "credential_rejection_verified",
                    "startup_measured"))
                or (profile.native_vision_enabled
                    and (evidence.get("native_vision_required") is not True
                         or evidence.get("native_vision_verified") is not True
                         or evidence.get("native_vision_score_verified")
                         is not True
                         or evidence.get("native_vision_vram_verified")
                         is not True
                         or not isinstance(
                             evidence.get("native_vision_vram_mib"), int)
                         or isinstance(
                             evidence.get("native_vision_vram_mib"), bool)
                         or not 0 < evidence["native_vision_vram_mib"]
                         <= _native_vision_vram_ceiling_mib(profile)))):
            return False
        promoted = dict(record)
        promoted["stability_state"] = "stable"
        promoted["stabilized_at"] = time.time()
        _atomic_write_private_json(stable_store.path, promoted)
        self.discard(
            profile.fingerprint, runtime_identity=runtime_identity)
        return True

    def discard(self, profile_fingerprint: str | None = None, *,
                runtime_identity: str | None = None) -> bool:
        """Remove only the exact probation record the caller inspected.

        A same-profile replacement process may stage a new probation record
        between observation and cleanup.  Binding deletion to its private
        runtime identity prevents the older observer from deleting the newer
        process's evidence.
        """
        if profile_fingerprint is not None or runtime_identity is not None:
            status, record = _read_private_json(self.path)
            if (status != "ok" or record is None
                    or (profile_fingerprint is not None
                        and record.get("profile_fingerprint")
                        != profile_fingerprint)
                    or (runtime_identity is not None
                        and record.get("runtime_identity")
                        != runtime_identity)):
                return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            return False
        fs.fsync_directory(self.path.parent)
        return True

    def public_status(self, profile: RuntimeProfile) -> dict[str, Any]:
        status, record = _read_private_json(self.path)
        exact = bool(
            status == "ok" and record is not None
            and record.get("profile_fingerprint") == profile.fingerprint
            and record.get("hardware_fingerprint")
            == profile.hardware_fingerprint)
        return {
            "state": "probation" if exact else status,
            "matches_active": exact,
        }


def runtime_boot_candidates(
        proposed: RuntimeProfile,
        last_known_good: RuntimeProfile | None = None,
        *, maximum: int = MAX_BOOT_CANDIDATES) -> tuple[RuntimeProfile, ...]:
    """Return a bounded, monotonic capacity ladder headed by the proposal."""
    if maximum < 1:
        raise ValueError("maximum candidates must be positive")
    candidates: list[RuntimeProfile] = [proposed]
    fingerprints = {proposed.fingerprint}
    if proposed.overrides:
        return tuple(candidates)

    if (len(candidates) < maximum
            and last_known_good is not None
            and last_known_good.hardware_fingerprint == proposed.hardware_fingerprint
            and last_known_good.family_fingerprint == proposed.family_fingerprint
            and last_known_good.context_tokens <= proposed.context_tokens
            and last_known_good.max_sequences <= proposed.max_sequences
            and last_known_good.cuda_graph_capture_size
            <= proposed.cuda_graph_capture_size
            and last_known_good.fingerprint not in fingerprints):
        candidates.append(last_known_good)
        fingerprints.add(last_known_good.fingerprint)

    for context, sequences, capture in (
            (131_072, 4, 16),
            (65_536, 2, 8),
            (32_768, 1, 4)):
        if len(candidates) >= maximum:
            break
        prior = candidates[-1]
        candidate = replace(
            prior,
            name=f"{proposed.name}-boot-safe-{min(context, proposed.context_tokens)}",
            source="bounded-degradation",
            context_tokens=min(context, prior.context_tokens),
            max_sequences=min(sequences, prior.max_sequences),
            cuda_graph_capture_size=min(
                capture, prior.cuda_graph_capture_size),
            warnings=prior.warnings + (
                "bounded automatic boot fallback reduced context/concurrency",),
        )
        if (candidate.context_tokens > prior.context_tokens
                or candidate.max_sequences > prior.max_sequences
                or candidate.cuda_graph_capture_size
                > prior.cuda_graph_capture_size
                or candidate.gpu_memory_utilization
                > prior.gpu_memory_utilization + 0.0005
                or candidate.llm_memory_budget_gib
                > prior.llm_memory_budget_gib + 0.001
                or candidate.tts_reserve_gib + 0.001
                < prior.tts_reserve_gib
                or candidate.unallocated_gpu_gib + 0.001
                < prior.unallocated_gpu_gib):
            continue
        if candidate.fingerprint not in fingerprints:
            candidates.append(candidate)
            fingerprints.add(candidate.fingerprint)
    return tuple(candidates)


def runtime_benchmark_candidates(
        proposed: RuntimeProfile, *, maximum: int = 3,
        ) -> tuple[RuntimeProfile, ...]:
    """Return exact launcher-supported KV/context modes for measurement.

    This function never selects or promotes a profile. It only defines a
    bounded same-family experiment set whose members can be restored by exact
    fingerprint after benchmarking.
    """
    if not 1 <= maximum <= 3:
        raise ValueError("benchmark candidate maximum must be between 1 and 3")
    if proposed.overrides or not proposed.local_runtime_available:
        return (proposed,)
    modes = (
        ("huge", 200_000),
        ("long", 150_000),
        ("fast", 65_536),
    )
    candidates: list[RuntimeProfile] = []
    fingerprints: set[str] = set()
    for mode, ceiling in modes:
        candidate = replace(
            proposed,
            name=f"{proposed.name}-benchmark-{mode}",
            source="benchmark-candidate",
            context_tokens=min(proposed.context_tokens, ceiling),
            kv_mode=mode,
            warnings=proposed.warnings + (
                "temporary benchmark candidate; never auto-promoted",),
        )
        if candidate.fingerprint in fingerprints:
            continue
        candidates.append(candidate)
        fingerprints.add(candidate.fingerprint)
        if len(candidates) >= maximum:
            break
    if proposed.fingerprint not in fingerprints:
        # A nonstandard automatic proposal remains measurable and restorable.
        candidates.insert(0, proposed)
        candidates = candidates[:maximum]
    return tuple(candidates)


def match_active_candidate(
        candidates: tuple[RuntimeProfile, ...],
        active_manifest: dict[str, Any] | None) -> RuntimeProfile | None:
    fingerprint = (active_manifest or {}).get("fingerprint")
    return next((item for item in candidates
                 if item.fingerprint == fingerprint), None)


class BootRecoveryStore:
    """Durable exponential backoff and post-launch crash-loop memory."""

    def __init__(self, path: str | Path, *,
                 stability_seconds: int = BOOT_STABILITY_SECONDS):
        self.path = Path(path)
        self.stability_seconds = max(1, int(stability_seconds))

    def _load(self, proposed: RuntimeProfile) -> dict[str, Any] | None:
        status, value = _read_private_json(self.path)
        if status != "ok" or value is None:
            return None
        if (value.get("schema_version") != RECOVERY_SCHEMA_VERSION
                or value.get("hardware_fingerprint")
                != proposed.hardware_fingerprint
                or value.get("proposed_profile_fingerprint")
                != proposed.fingerprint
                or value.get("clock_boot_id") != _boot_session_id()):
            return None
        try:
            failures = _plain_int(
                value.get("consecutive_failures"), minimum=0, maximum=32)
            if not isinstance(value.get("expected_running"), bool):
                raise ValueError("invalid expected-running flag")
            success_at = _plain_float(
                value.get("last_success_at"), minimum=0, maximum=10 ** 12)
            retry_at = _plain_float(
                value.get("next_retry_at"), minimum=0, maximum=10 ** 12)
            state = value.get("state")
            if state not in {"ready", "backoff", "probation", "stable"}:
                raise ValueError("invalid recovery state")
            active_fingerprint = value.get("active_profile_fingerprint")
            if (active_fingerprint is not None
                    and (not isinstance(active_fingerprint, str)
                         or not 1 <= len(active_fingerprint) <= 128)):
                raise ValueError("invalid active fingerprint")
            runtime_identity = value.get("runtime_identity")
            if (runtime_identity is not None
                    and (not isinstance(runtime_identity, str)
                         or not 1 <= len(runtime_identity) <= 128)):
                raise ValueError("invalid runtime identity")
        except (TypeError, ValueError):
            return None
        return value | {
            "consecutive_failures": failures,
            "last_success_at": success_at,
            "next_retry_at": retry_at,
        }

    @staticmethod
    def _backoff(failures: int) -> int:
        exponent = min(max(failures - 1, 0), 16)
        return min(BOOT_BACKOFF_MAX_SECONDS,
                   BOOT_BACKOFF_BASE_SECONDS * (2 ** exponent))

    def _base(self, proposed: RuntimeProfile) -> dict[str, Any]:
        return {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "hardware_fingerprint": proposed.hardware_fingerprint,
            "proposed_profile_fingerprint": proposed.fingerprint,
            "clock_boot_id": _boot_session_id(),
            "consecutive_failures": 0,
            "expected_running": False,
            "last_success_at": 0.0,
            "next_retry_at": 0.0,
            "state": "ready",
        }

    def record_launch_failure(self, proposed: RuntimeProfile, *,
                              now: float | None = None) -> int:
        current_time = float(_recovery_now() if now is None else now)
        value = self._load(proposed) or self._base(proposed)
        failures = min(int(value.get("consecutive_failures", 0)) + 1, 32)
        delay = self._backoff(failures)
        value.update({
            "consecutive_failures": failures,
            "expected_running": False,
            "next_retry_at": current_time + delay,
            "state": "backoff",
        })
        _atomic_write_private_json(self.path, value)
        return delay

    def record_launch_success(self, proposed: RuntimeProfile,
                              active: RuntimeProfile, *,
                              runtime_identity: str | None = None,
                              now: float | None = None) -> None:
        current_time = float(_recovery_now() if now is None else now)
        value = self._load(proposed) or self._base(proposed)
        value.update({
            "active_profile_fingerprint": active.fingerprint,
            "runtime_identity": runtime_identity,
            "expected_running": True,
            "last_success_at": current_time,
            "next_retry_at": 0.0,
            "state": "probation",
        })
        _atomic_write_private_json(self.path, value)

    def record_planned_stop(
        self, proposed: RuntimeProfile, active: RuntimeProfile, *,
        runtime_identity: str,
    ) -> bool:
        """Clear crash-loop accounting only for the exact active runtime."""
        value = self._load(proposed)
        if (value is None or not value.get("expected_running")
                or value.get("active_profile_fingerprint")
                    != active.fingerprint
                or value.get("runtime_identity") != runtime_identity):
            return False
        value.update({
            "consecutive_failures": 0,
            "expected_running": False,
            "next_retry_at": 0.0,
            "state": "ready",
        })
        _atomic_write_private_json(self.path, value)
        return True

    def record_planned_stop_identity(
        self, *, active_profile_fingerprint: str,
        runtime_identity: str,
    ) -> bool:
        """Record SIGTERM shutdown without probing hardware in the handler."""
        status, value = _read_private_json(self.path)
        if (status != "ok" or value is None
                or value.get("schema_version") != RECOVERY_SCHEMA_VERSION
                or value.get("clock_boot_id") != _boot_session_id()
                or not value.get("expected_running")
                or value.get("active_profile_fingerprint")
                    != active_profile_fingerprint
                or value.get("runtime_identity") != runtime_identity):
            return False
        try:
            _plain_int(
                value.get("consecutive_failures"), minimum=0, maximum=32)
            _plain_float(
                value.get("next_retry_at"), minimum=0, maximum=10 ** 12)
        except (TypeError, ValueError):
            return False
        value.update({
            "consecutive_failures": 0,
            "expected_running": False,
            "next_retry_at": 0.0,
            "state": "ready",
        })
        _atomic_write_private_json(self.path, value)
        return True

    def observe(self, proposed: RuntimeProfile, *, running: bool,
                active: RuntimeProfile | None = None,
                runtime_identity: str | None = None,
                now: float | None = None) -> int:
        """Record one edge-triggered runtime observation and return retry delay."""
        current_time = float(_recovery_now() if now is None else now)
        value = self._load(proposed)
        if value is None:
            if running and active is not None:
                self.record_launch_success(
                    proposed, active, runtime_identity=runtime_identity,
                    now=current_time)
            return 0
        if running:
            if active is None:
                return 0
            if (not value.get("expected_running")
                    or value.get("active_profile_fingerprint")
                    != active.fingerprint
                    or value.get("runtime_identity") != runtime_identity):
                self.record_launch_success(
                    proposed, active, runtime_identity=runtime_identity,
                    now=current_time)
                return 0
            success_at = float(value.get("last_success_at", current_time))
            if success_at > current_time:
                value.update({
                    "last_success_at": current_time,
                    "state": "probation",
                })
                _atomic_write_private_json(self.path, value)
                return 0
            if current_time - success_at >= self.stability_seconds:
                if (value.get("state") != "stable"
                        or value.get("consecutive_failures") != 0
                        or value.get("next_retry_at") != 0.0):
                    value.update({
                        "consecutive_failures": 0,
                        "next_retry_at": 0.0,
                        "state": "stable",
                    })
                    _atomic_write_private_json(self.path, value)
            return 0

        if value.get("expected_running"):
            return self.record_launch_failure(proposed, now=current_time)
        return min(
            BOOT_BACKOFF_MAX_SECONDS,
            max(0, int(float(value.get("next_retry_at", 0)) - current_time)))

    def public_status(self, proposed: RuntimeProfile, *,
                      now: float | None = None) -> dict[str, Any]:
        current_time = float(_recovery_now() if now is None else now)
        read_status, raw = _read_private_json(self.path)
        if read_status == "missing":
            return {"state": "ready", "consecutive_failures": 0,
                    "retry_after_seconds": 0}
        if read_status != "ok" or raw is None:
            return {"state": "invalid", "consecutive_failures": 0,
                    "retry_after_seconds": 0}
        value = self._load(proposed)
        if value is None:
            same_epoch = (
                raw.get("schema_version") == RECOVERY_SCHEMA_VERSION
                and raw.get("hardware_fingerprint")
                == proposed.hardware_fingerprint
                and raw.get("proposed_profile_fingerprint")
                == proposed.fingerprint
                and raw.get("clock_boot_id") == _boot_session_id())
            return {"state": "invalid" if same_epoch else "new_profile",
                    "consecutive_failures": 0, "retry_after_seconds": 0}
        failures = max(0, min(int(value.get("consecutive_failures", 0)), 32))
        retry = min(
            BOOT_BACKOFF_MAX_SECONDS,
            max(0, int(float(value.get("next_retry_at", 0)) - current_time)))
        state = str(value.get("state") or "ready")
        if state not in {"ready", "backoff", "probation", "stable"}:
            state = "invalid"
        return {"state": state, "consecutive_failures": failures,
                "retry_after_seconds": retry}
