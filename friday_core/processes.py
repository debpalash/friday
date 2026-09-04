"""Permissioned, durable lifecycle management for curated local processes.

This module deliberately does not expose a command runner.  Callers select an
immutable :class:`ProcessSpec` and supply only values for its typed parameters;
the registry, rather than model output, constructs the exact argument vector.
The broker journals an intent before asking a backend to start a process and
uses an opaque instance identifier for every later operation.

The production backend uses transient user-systemd services.  There is no
``Popen`` fallback: losing the cgroup and unit identity boundary is a hard
failure, not a reason to launch with weaker fencing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)
from friday_host import fs

from .cognition import ResourceClaim
from .graph import GraphStore, canonical_json, new_id, sha256_text, utc_now
from .step_payloads import StepPayloadCipher


_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{8,200}\Z")
_SPEC_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{2,127}\Z")
_SPEC_NAME = re.compile(r"[a-z][a-z0-9_.-]{1,63}\Z")
_PARAMETER_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_FLAG = re.compile(r"(?:-[A-Za-z0-9]|--[a-z0-9][a-z0-9-]{0,62})\Z")
_UNIT_NAME = re.compile(r"friday-proc-[0-9a-f]{32}\.service\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_CODE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,79}\Z")
_WAYLAND_DISPLAY = re.compile(r"wayland-[0-9]{1,5}\Z")
_APPLICATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")

_MAX_ARGUMENTS = 128
_MAX_ARGUMENT_BYTES = 32_768
_MAX_SINGLE_ARGUMENT_BYTES = 4_096
_MAX_ENV_VALUE_BYTES = 1_024
_MAX_SYSTEMD_OUTPUT_BYTES = 32_768
_CLEANUP_CLAIM_SECONDS = 60

_SAFE_ENVIRONMENT_KEYS = frozenset({
    "LANG", "LC_ALL", "LC_CTYPE", "NO_COLOR", "TERM", "TZ",
})
_SECRET_ENV_FRAGMENTS = (
    "AUTH", "CREDENTIAL", "KEY", "PASSWORD", "SECRET", "TOKEN",
)
_FORBIDDEN_EXECUTABLE_NAMES = frozenset({
    "ash", "bash", "busybox", "bwrap", "csh", "dash", "deno", "env", "fish",
    "java", "ksh", "node", "perl", "php", "pypy", "ruby", "sh",
    "systemd-run", "tcsh", "zsh",
})

_INSTANCE_STATES = frozenset({
    "prepared", "starting", "running", "stop_requested", "stopping",
    "terminated", "exited", "launch_failed", "identity_mismatch",
    "reconcile_required",
})
_TERMINAL_INSTANCE_STATES = frozenset({
    "terminated", "exited", "launch_failed",
})
_ACTIVE_INSTANCE_STATES = frozenset({
    "prepared", "starting", "running", "stop_requested", "stopping",
    "reconcile_required", "identity_mismatch",
})


class ProcessBrokerError(RuntimeError):
    """A stable, non-sensitive process broker failure."""

    code = "process_broker_error"

    def __init__(self, message: str | None = None):
        super().__init__(message or self.code)


class ProcessSpecError(ValueError):
    """A curated spec or typed value is invalid."""


class ProcessBackendError(ProcessBrokerError):
    """The backend failed without exposing command output or model arguments."""

    code = "process_backend_failed"

    def __init__(self, code: str = "process_backend_failed", *,
                 outcome_unknown: bool = True):
        self.code = code if _SAFE_CODE.fullmatch(code) else "process_backend_failed"
        self.outcome_unknown = bool(outcome_unknown)
        super().__init__(self.code)


class ProcessIdentityError(ProcessBrokerError):
    code = "process_identity_mismatch"


class ProcessBindingError(ProcessBrokerError):
    """An approved opaque instance binding no longer names the same state."""

    code = "process_instance_binding_changed"


class ProcessAdmissionError(ProcessBrokerError):
    code = "process_workload_admission_failed"


class ProcessCleanupBlocked(ProcessBrokerError):
    """A retained unit cannot be retired without weakening its fence."""

    code = "process_unit_cleanup_blocked"

    def __init__(self, code: str = "process_unit_cleanup_blocked"):
        self.code = (
            code if _SAFE_CODE.fullmatch(code)
            else "process_unit_cleanup_blocked")
        super().__init__(self.code)


def _validate_opaque_identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded opaque identifier")
    return value


def _stable_code(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower()
    return text if _SAFE_CODE.fullmatch(text) else fallback


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


class ExecutableIdentity(BaseModel):
    """Pinned identity of one exact executable file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device: int = Field(ge=0)
    inode: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    mode: int = Field(ge=0, le=0o7777)


def _executable_identity(path: str | Path, *, reject_alias: bool = True,
                         reject_interpreter: bool = True) -> tuple[str,
                                                                    ExecutableIdentity]:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise ProcessSpecError("process executables must use an exact absolute path")
    lexical = Path(os.path.abspath(raw))
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ProcessSpecError("process executable is unavailable") from exc
    if reject_alias and lexical != resolved:
        raise ProcessSpecError("symbolic-link executable paths are not accepted")
    try:
        before_lstat = os.lstat(lexical)
    except OSError as exc:
        raise ProcessSpecError("process executable is unavailable") from exc
    if stat.S_ISLNK(before_lstat.st_mode):
        raise ProcessSpecError("symbolic-link executable paths are not accepted")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise ProcessSpecError("process executable cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProcessSpecError("process executable must be a regular file")
        mode = stat.S_IMODE(before.st_mode)
        if not mode & 0o111:
            raise ProcessSpecError("process executable is not executable")
        if mode & 0o022:
            raise ProcessSpecError(
                "group/world-writable process executables are not accepted")
        if mode & 0o6000:
            raise ProcessSpecError("setuid/setgid process executables are not accepted")
        if os.pread(descriptor, 2, 0) == b"#!":
            raise ProcessSpecError(
                "script interpreters are not accepted as v1 executables")
        executable_hash = _sha256_fd(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
            before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
            after.st_ctime_ns):
        raise ProcessSpecError("process executable changed while it was verified")
    name = lexical.name.casefold()
    interpreter = (name in _FORBIDDEN_EXECUTABLE_NAMES
                   or re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", name)
                   or re.fullmatch(r"pypy(?:[0-9]+)?", name)
                   or re.fullmatch(r"perl(?:[0-9.]+)?", name)
                   or re.fullmatch(r"ruby(?:[0-9.]+)?", name)
                   or re.fullmatch(r"node(?:js)?", name))
    if reject_interpreter and interpreter:
        raise ProcessSpecError("shells and interpreters are not accepted in v1 specs")
    return str(lexical), ExecutableIdentity(
        device=int(before.st_dev), inode=int(before.st_ino),
        sha256=executable_hash, size=int(before.st_size), mode=mode)


def _canonical_directory(path: str | Path) -> str:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise ProcessSpecError("process working directories must be absolute")
    lexical = Path(os.path.abspath(raw))
    try:
        resolved = raw.resolve(strict=True)
        observed = os.lstat(lexical)
    except OSError as exc:
        raise ProcessSpecError("process working directory is unavailable") from exc
    if lexical != resolved or stat.S_ISLNK(observed.st_mode):
        raise ProcessSpecError("symbolic-link working directories are not accepted")
    if not stat.S_ISDIR(observed.st_mode):
        raise ProcessSpecError("process working directory must be a directory")
    return str(lexical)


def _validate_argument(argument: str) -> str:
    if not isinstance(argument, str):
        raise ProcessSpecError("process arguments must be strings")
    encoded = argument.encode("utf-8")
    if b"\0" in encoded or len(encoded) > _MAX_SINGLE_ARGUMENT_BYTES:
        raise ProcessSpecError("process argument violates the bounded argv contract")
    return argument


class PrevalidatedFile(BaseModel):
    """A file parameter authorized by an external exact-path broker hook."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    authorization_id: str = Field(min_length=8, max_length=200,
                                  pattern=r"^[A-Za-z0-9_.:-]+$")

    @model_validator(mode="after")
    def validate_exact_regular_file(self) -> "PrevalidatedFile":
        raw = Path(self.path).expanduser()
        if not raw.is_absolute():
            raise ValueError("prevalidated file path must be absolute")
        lexical = Path(os.path.abspath(raw))
        try:
            resolved = raw.resolve(strict=True)
            observed = os.lstat(lexical)
        except OSError as exc:
            raise ValueError("prevalidated file is unavailable") from exc
        if lexical != resolved or stat.S_ISLNK(observed.st_mode):
            raise ValueError("prevalidated file path cannot be a symbolic link")
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError("prevalidated file must be regular")
        object.__setattr__(self, "path", str(lexical))
        return self


FileParameterValidator = Callable[[str, "ProcessParameter"], PrevalidatedFile]


class ProcessParameter(BaseModel):
    """One typed value and its curated mapping into an argument vector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    kind: Literal["string", "integer", "boolean", "enum", "file"]
    flag: str | None = None
    required: bool = True
    default: str | int | bool | None = None
    min_length: int = Field(default=0, ge=0, le=4_096)
    max_length: int = Field(default=256, ge=1, le=4_096)
    minimum: int | None = None
    maximum: int | None = None
    choices: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_contract(self) -> "ProcessParameter":
        if self.flag is not None and _FLAG.fullmatch(self.flag) is None:
            raise ValueError("parameter flag is not a bounded curated flag")
        if self.min_length > self.max_length:
            raise ValueError("parameter min_length exceeds max_length")
        if (self.minimum is not None and self.maximum is not None
                and self.minimum > self.maximum):
            raise ValueError("parameter minimum exceeds maximum")
        if self.kind == "enum":
            if not self.choices or len(set(self.choices)) != len(self.choices):
                raise ValueError("enum parameters need unique choices")
            for choice in self.choices:
                _validate_argument(choice)
        elif self.choices:
            raise ValueError("choices are accepted only for enum parameters")
        if self.kind == "boolean" and self.flag is None:
            raise ValueError("boolean parameters need a curated flag")
        if self.kind != "integer" and (
                self.minimum is not None or self.maximum is not None):
            raise ValueError("numeric bounds are accepted only for integer parameters")
        if not self.required and self.default is not None:
            self.validate_value(self.default, file_validator=None,
                                validating_default=True)
        if self.required and self.default is not None:
            raise ValueError("required parameters cannot also have a default")
        return self

    def validate_value(
        self,
        value: Any,
        *,
        file_validator: FileParameterValidator | None,
        validating_default: bool = False,
    ) -> tuple[str | int | bool, PrevalidatedFile | None]:
        if self.kind in {"string", "enum", "file"}:
            if not isinstance(value, str):
                raise ProcessSpecError(f"parameter {self.name} must be a string")
            length = len(value)
            if not self.min_length <= length <= self.max_length:
                raise ProcessSpecError(
                    f"parameter {self.name} violates its length contract")
            _validate_argument(value)
            if self.kind == "enum" and value not in self.choices:
                raise ProcessSpecError(
                    f"parameter {self.name} is not an allowed choice")
            if self.kind == "file":
                if validating_default:
                    raise ValueError("file parameters cannot have defaults")
                if file_validator is None:
                    raise PermissionError(
                        f"parameter {self.name} requires exact path authorization")
                try:
                    authorized = file_validator(value, self)
                except PermissionError:
                    raise
                except Exception as exc:
                    raise PermissionError(
                        f"parameter {self.name} path authorization failed") from exc
                if not isinstance(authorized, PrevalidatedFile):
                    raise PermissionError(
                        "file path hook must return PrevalidatedFile authority")
                return authorized.path, authorized
            return value, None
        if self.kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ProcessSpecError(f"parameter {self.name} must be an integer")
            if self.minimum is not None and value < self.minimum:
                raise ProcessSpecError(f"parameter {self.name} is below its minimum")
            if self.maximum is not None and value > self.maximum:
                raise ProcessSpecError(f"parameter {self.name} exceeds its maximum")
            return value, None
        if not isinstance(value, bool):
            raise ProcessSpecError(f"parameter {self.name} must be a boolean")
        return value, None


class ProcessResources(BaseModel):
    """Immutable resource reservation embedded in a curated process spec."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    cpu_cores: float = Field(default=1.0, gt=0.0, le=256.0)
    ram_mib: int = Field(default=512, gt=0, le=1_048_576)
    vram_mib: int = Field(default=0, ge=0, le=1_048_576)
    accelerator: str = Field(default="none",
                             pattern=r"^(?:none|cuda(?::[0-9]+)?)$")
    network: bool = False
    concurrency_slots: int = Field(default=1, ge=1, le=64)
    latency_class: Literal["interactive", "background", "batch"] = (
        "background")

    @model_validator(mode="after")
    def validate_accelerator(self) -> "ProcessResources":
        if self.vram_mib and self.accelerator == "none":
            raise ValueError("nonzero process VRAM requires an accelerator")
        return self

    def as_claim(self) -> ResourceClaim:
        return ResourceClaim.model_validate(self.model_dump(mode="json"))


class ProcessLimits(BaseModel):
    """Hard limits applied by the backend, all chosen by the curated spec."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    cpu_quota_percent: float = Field(default=100.0, gt=0.0, le=25_600.0)
    memory_high_mib: int = Field(default=384, gt=0, le=1_048_576)
    memory_max_mib: int = Field(default=512, gt=0, le=1_048_576)
    memory_swap_max_mib: int = Field(default=0, ge=0, le=1_048_576)
    tasks_max: int = Field(default=64, ge=1, le=65_536)
    runtime_max_seconds: int = Field(default=3_600, ge=1, le=31_536_000)
    stop_grace_seconds: float = Field(default=10.0, ge=0.1, le=300.0)

    @model_validator(mode="after")
    def validate_memory(self) -> "ProcessLimits":
        if self.memory_high_mib > self.memory_max_mib:
            raise ValueError("MemoryHigh cannot exceed MemoryMax")
        return self


class BubblewrapProfile(BaseModel):
    """Small fail-closed namespace profile for non-desktop workloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    share_network: bool = False
    writable_cwd: bool = False

    @property
    def fingerprint(self) -> str:
        return sha256_text(canonical_json(self.model_dump(mode="json")))


class ProcessSessionAccess(BaseModel):
    """Pinned local user-session transports required by a curated GUI app.

    Session transports are backend-derived capabilities, never environment
    values supplied by the model. V1 supports only the local Wayland socket
    and the already pinned user D-Bus socket.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    wayland: bool = False
    session_bus: bool = False

    @model_validator(mode="after")
    def validate_access(self) -> "ProcessSessionAccess":
        if not self.wayland and not self.session_bus:
            raise ValueError("session access must name at least one transport")
        return self


class ProcessPresentation(BaseModel):
    """Expected visible surface for one exact curated application."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["wayland_window"] = "wayland_window"
    application_id: str = Field(min_length=1, max_length=128)
    application: str = Field(min_length=1, max_length=80)
    startup_timeout_seconds: float = Field(default=8.0, ge=0.5, le=30.0)
    window_owner: Literal["leader", "managed_cgroup"] = "leader"

    @field_validator("application_id")
    @classmethod
    def validate_application_id(cls, value: str) -> str:
        if _APPLICATION_ID.fullmatch(value) is None:
            raise ValueError("presentation application ID is invalid")
        return value

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        # Preserve the fingerprint of every existing leader-only presentation.
        # Managed-cgroup attribution is a new, explicitly hashed authority.
        if payload.get("window_owner") == "leader":
            payload.pop("window_owner", None)
        return sha256_text(canonical_json(payload))

    def safe_display(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "application": self.application,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "window_owner": self.window_owner,
        }


class ProcessSpec(BaseModel):
    """An immutable, versioned, exact executable contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    spec_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{2,127}$")
    name: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,63}$")
    version: int = Field(default=1, ge=1)
    display_name: str = Field(min_length=1, max_length=120)
    executable: str
    cwd: str
    fixed_args: tuple[str, ...] = ()
    parameters: tuple[ProcessParameter, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    resources: ProcessResources = Field(default_factory=ProcessResources)
    limits: ProcessLimits = Field(default_factory=ProcessLimits)
    sandbox: BubblewrapProfile = Field(default_factory=BubblewrapProfile)
    session_access: ProcessSessionAccess | None = None
    presentation: ProcessPresentation | None = None
    instance_policy: Literal["multiple", "singleton"] = "multiple"
    persistent: bool = False
    executable_identity: ExecutableIdentity | None = None

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: Any) -> tuple[tuple[str, str], ...]:
        if value is None:
            return ()
        if isinstance(value, Mapping):
            return tuple(sorted((str(key), item) for key, item in value.items()))
        try:
            return tuple((str(key), item) for key, item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError("process environment must be a mapping") from exc

    @model_validator(mode="after")
    def validate_spec(self) -> "ProcessSpec":
        canonical_executable, observed = _executable_identity(self.executable)
        canonical_cwd = _canonical_directory(self.cwd)
        if self.executable_identity is not None and self.executable_identity != observed:
            raise ValueError("pinned executable identity does not match the exact file")
        object.__setattr__(self, "executable", canonical_executable)
        object.__setattr__(self, "cwd", canonical_cwd)
        object.__setattr__(self, "executable_identity", observed)

        if len(self.fixed_args) + len(self.parameters) + 1 > _MAX_ARGUMENTS:
            raise ValueError("process spec exceeds the argv item limit")
        for argument in self.fixed_args:
            _validate_argument(argument)
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("process parameter names must be unique")

        environment = dict(self.environment)
        if len(environment) != len(self.environment):
            raise ValueError("process environment keys must be unique")
        for raw_name, raw_value in environment.items():
            name = str(raw_name)
            if (name not in _SAFE_ENVIRONMENT_KEYS or name == "PATH"
                    or name.startswith(("LD_", "PYTHON"))
                    or any(fragment in name.upper()
                           for fragment in _SECRET_ENV_FRAGMENTS)):
                raise ValueError("process environment contains a forbidden variable")
            if not isinstance(raw_value, str):
                raise ValueError("process environment values must be strings")
            encoded = raw_value.encode("utf-8")
            if (b"\0" in encoded or b"\n" in encoded
                    or len(encoded) > _MAX_ENV_VALUE_BYTES):
                raise ValueError("process environment value violates its bound")
        object.__setattr__(self, "environment", tuple(sorted(environment.items())))

        if self.limits.cpu_quota_percent / 100.0 > self.resources.cpu_cores:
            raise ValueError("CPU hard limit exceeds the reserved CPU claim")
        if self.limits.memory_max_mib > self.resources.ram_mib:
            raise ValueError("memory hard limit exceeds the reserved RAM claim")
        if self.sandbox.share_network and not self.resources.network:
            raise ValueError("sandbox network sharing requires a network claim")
        if not self.sandbox.enabled and not self.resources.network:
            raise ValueError(
                "an unsandboxed process cannot claim that network is isolated")
        if self.sandbox.enabled and any(
                parameter.kind == "file" for parameter in self.parameters):
            raise ValueError(
                "sandboxed v1 process specs cannot expose host file parameters")
        if self.session_access is not None and self.sandbox.enabled:
            raise ValueError(
                "v1 user-session access requires an unsandboxed curated app")
        if (self.presentation is not None
                and (self.session_access is None
                     or not self.session_access.wayland)):
            raise ValueError(
                "a Wayland presentation requires explicit Wayland session access")
        return self

    @property
    def sandbox_fingerprint(self) -> str:
        return self.sandbox.fingerprint

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        # Preserve durable fingerprints for specs created before session access
        # existed. An actual session capability remains immutable and hashed.
        if payload.get("session_access") is None:
            payload.pop("session_access", None)
        if payload.get("presentation") is None:
            payload.pop("presentation", None)
        elif payload["presentation"].get("window_owner") == "leader":
            # Preserve durable fingerprints for the original leader-only
            # presentation schema. Managed-cgroup ownership remains hashed.
            payload["presentation"].pop("window_owner", None)
        # Preserve every pre-singleton durable spec fingerprint. Singleton is
        # a new launch authority and therefore remains explicitly hashed.
        if payload.get("instance_policy") == "multiple":
            payload.pop("instance_policy", None)
        return sha256_text(canonical_json(payload))

    def verify_current(self) -> None:
        executable, identity = _executable_identity(self.executable)
        cwd = _canonical_directory(self.cwd)
        if (executable != self.executable or cwd != self.cwd
                or identity != self.executable_identity):
            raise ProcessSpecError("process spec executable identity changed")

    def safe_display(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "display_name": self.display_name,
            "parameters": [{
                "name": parameter.name,
                "kind": parameter.kind,
                "required": parameter.required,
                "flag": parameter.flag,
                "default": parameter.default,
                "min_length": parameter.min_length,
                "max_length": parameter.max_length,
                "minimum": parameter.minimum,
                "maximum": parameter.maximum,
                "choices": list(parameter.choices),
            } for parameter in self.parameters],
            "persistent": self.persistent,
            "instance_policy": self.instance_policy,
            "resources": self.resources.model_dump(mode="json"),
            "limits": self.limits.model_dump(mode="json"),
            "sandboxed": self.sandbox.enabled,
            "network": self.resources.network,
            "session_access": (
                self.session_access.model_dump(mode="json")
                if self.session_access is not None else None),
            "presentation": (
                self.presentation.safe_display()
                if self.presentation is not None else None),
        }


class RenderedProcess(BaseModel):
    """Exact ephemeral launch payload; repr deliberately omits private values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_id: str
    spec_fingerprint: str
    argv: tuple[str, ...] = Field(repr=False)
    environment: tuple[tuple[str, str], ...] = Field(repr=False)
    values: dict[str, Any] = Field(repr=False)
    authorizations: dict[str, str] = Field(default_factory=dict, repr=False)
    args_sha256: str


class ProcessLaunchBinding(BaseModel):
    """Persistence-safe immutable executor binding for one exact launch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["process_spec"] = "process_spec"
    spec_id: str
    name: str
    version: int = Field(ge=1)
    spec_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    sandbox_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    args_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    executable_identity: ExecutableIdentity
    resource_claim: ProcessResources
    persistent: bool


class ProcessInstanceBinding(BaseModel):
    """Privacy-safe, immutable approval binding for one lifecycle operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["process_instance"] = "process_instance"
    operation: Literal["inspect", "terminate"]
    instance_id: str
    spec_id: str
    spec_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    sandbox_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    args_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: Literal[
        "prepared", "starting", "running", "stop_requested", "stopping",
        "terminated", "exited", "launch_failed", "identity_mismatch",
        "reconcile_required",
    ]
    runtime_identity_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$")
    state_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    persistent: bool


class ProcessOperationContext(BaseModel):
    """Exact durable task claim allowed to cross a process-effect boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{8,200}$")
    step_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{8,200}$")
    action_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{8,200}$")
    idempotency_key: str = Field(pattern=r"^[A-Za-z0-9_.:-]{8,200}$")
    attempt_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{8,200}$")
    attempt_number: int = Field(ge=1)
    lease_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{8,200}$")
    worker_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{8,200}$")


class ProcessApprovalPreview(BaseModel):
    """Ephemeral, trusted consent preview; exact values must not be persisted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_id: str
    display_name: str
    version: int = Field(ge=1)
    parameter_values: dict[str, Any] = Field(repr=False)
    resources: ProcessResources
    limits: ProcessLimits
    sandboxed: bool
    network: bool
    session_access: ProcessSessionAccess | None = None
    presentation: dict[str, Any] | None = None
    instance_policy: Literal["multiple", "singleton"] = "multiple"
    persistent: bool
    args_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProcessSpecRegistry:
    """In-memory curated spec inventory with deterministic typed rendering."""

    def __init__(self, specs: Sequence[ProcessSpec] = ()):
        self._by_id: dict[str, ProcessSpec] = {}
        self._by_version: dict[tuple[str, int], str] = {}
        self._lock = threading.RLock()
        for spec in specs:
            self.register(spec)

    def register(self, spec: ProcessSpec) -> ProcessSpec:
        if not isinstance(spec, ProcessSpec):
            spec = ProcessSpec.model_validate(spec)
        with self._lock:
            prior = self._by_id.get(spec.spec_id)
            if prior is not None:
                if prior.fingerprint != spec.fingerprint:
                    raise ProcessSpecError("process spec ID is already pinned")
                return prior
            version_key = (spec.name, spec.version)
            other_id = self._by_version.get(version_key)
            if other_id is not None and other_id != spec.spec_id:
                raise ProcessSpecError("process spec name/version is already registered")
            self._by_id[spec.spec_id] = spec
            self._by_version[version_key] = spec.spec_id
            return spec

    def get(self, spec_id: str) -> ProcessSpec:
        with self._lock:
            try:
                return self._by_id[spec_id]
            except KeyError as exc:
                raise ProcessSpecError("process spec is not curated") from exc

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            specs = sorted(self._by_id.values(), key=lambda item: (item.name,
                                                                   item.version))
        return [{"spec_id": item.spec_id, **item.safe_display()} for item in specs]

    def specs(self) -> tuple[ProcessSpec, ...]:
        with self._lock:
            return tuple(self._by_id.values())

    def render(
        self,
        spec_id: str,
        parameter_values: Mapping[str, Any],
        *,
        file_validator: FileParameterValidator | None = None,
    ) -> RenderedProcess:
        spec = self.get(spec_id)
        if not isinstance(parameter_values, Mapping):
            raise ProcessSpecError("process parameter_values must be an object")
        supplied = {str(key): value for key, value in parameter_values.items()}
        parameters = {parameter.name: parameter for parameter in spec.parameters}
        unknown = sorted(set(supplied) - set(parameters))
        if unknown:
            raise ProcessSpecError("process parameter_values contain unknown fields")
        argv = [spec.executable, *spec.fixed_args]
        normalized: dict[str, Any] = {}
        authorizations: dict[str, str] = {}
        for parameter in spec.parameters:
            if parameter.name in supplied:
                raw_value = supplied[parameter.name]
            elif parameter.default is not None:
                raw_value = parameter.default
            elif parameter.required:
                raise ProcessSpecError(
                    f"required process parameter {parameter.name} is missing")
            else:
                continue
            value, authorization = parameter.validate_value(
                raw_value, file_validator=file_validator)
            normalized[parameter.name] = value
            if authorization is not None:
                authorizations[parameter.name] = authorization.authorization_id
            if parameter.kind == "boolean":
                if value:
                    argv.append(str(parameter.flag))
                continue
            if parameter.flag is not None:
                argv.append(parameter.flag)
            argv.append(str(value))

        for argument in argv:
            _validate_argument(argument)
        encoded_size = sum(len(item.encode("utf-8")) + 1 for item in argv)
        if len(argv) > _MAX_ARGUMENTS or encoded_size > _MAX_ARGUMENT_BYTES:
            raise ProcessSpecError("rendered process argv exceeds its bound")
        exact = {
            "parameter_values": normalized,
            "path_authorizations": authorizations,
        }
        args_sha256 = sha256_text(canonical_json(exact))
        return RenderedProcess(
            spec_id=spec.spec_id, spec_fingerprint=spec.fingerprint,
            argv=tuple(argv),
            environment=spec.environment,
            values=normalized, authorizations=authorizations,
            args_sha256=args_sha256)


class BackendObservation(BaseModel):
    """Private backend identity and lifecycle observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_name: str = Field(pattern=r"^friday-proc-[0-9a-f]{32}\.service$")
    identity_token: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)
    state: Literal["starting", "running", "stopping", "exited"]
    boot_id: str = Field(min_length=1, max_length=80, repr=False)
    invocation_id: str = Field(min_length=1, max_length=128, repr=False)
    control_group: str = Field(min_length=1, max_length=500, repr=False)
    leader_pid: int | None = Field(default=None, gt=0, repr=False)
    start_ticks: int | None = Field(default=None, ge=0, repr=False)
    exe_device: int | None = Field(default=None, ge=0, repr=False)
    exe_inode: int | None = Field(default=None, gt=0, repr=False)
    exe_sha256: str | None = Field(default=None,
                                   pattern=r"^[0-9a-f]{64}$", repr=False)
    cgroup_empty: bool = False
    exit_code: int | None = Field(default=None, ge=0, le=255)
    exit_signal: int | None = Field(default=None, ge=1, le=255)
    result_code: str = Field(default="unknown",
                             pattern=r"^[a-z0-9][a-z0-9_.:-]{0,79}$")
    unit_active_state: str = Field(
        default="unknown", pattern=r"^[a-z0-9][a-z0-9-]{0,39}$",
        repr=False)
    unit_sub_state: str = Field(
        default="unknown", pattern=r"^[a-z0-9][a-z0-9-]{0,39}$",
        repr=False)
    unit_job_pending: bool = Field(default=False, repr=False)

    @model_validator(mode="after")
    def validate_live_identity(self) -> "BackendObservation":
        identity = (self.leader_pid, self.start_ticks, self.exe_device,
                    self.exe_inode, self.exe_sha256)
        if self.state in {"running", "stopping"} and any(
                value is None for value in identity):
            raise ValueError("live backend observations require complete identity")
        if any(value is None for value in identity) and not all(
                value is None for value in identity):
            raise ValueError("backend executable identity must be all-or-none")
        return self

    def same_live_execution(self, other: "BackendObservation") -> bool:
        return bool(
            self.unit_name == other.unit_name
            and self.identity_token == other.identity_token
            and self.boot_id == other.boot_id
            and self.invocation_id == other.invocation_id
            and self.control_group == other.control_group
            and self.leader_pid is not None
            and self.leader_pid == other.leader_pid
            and self.start_ticks is not None
            and self.start_ticks == other.start_ticks
            and self.exe_device is not None
            and self.exe_device == other.exe_device
            and self.exe_inode is not None
            and self.exe_inode == other.exe_inode
            and self.exe_sha256 is not None
            and self.exe_sha256 == other.exe_sha256
        )


class BackendLaunchRequest(BaseModel):
    """Private exact request passed only from the broker to its backend."""

    model_config = ConfigDict(extra="forbid", frozen=True,
                              arbitrary_types_allowed=True)

    instance_id: str
    unit_name: str = Field(pattern=r"^friday-proc-[0-9a-f]{32}\.service$")
    identity_token: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)
    argv: tuple[str, ...] = Field(repr=False)
    cwd: str = Field(repr=False)
    environment: tuple[tuple[str, str], ...] = Field(repr=False)
    executable_identity: ExecutableIdentity = Field(repr=False)
    limits: ProcessLimits
    sandbox: BubblewrapProfile
    session_access: ProcessSessionAccess | None = None


class BackendTerminalFence(BaseModel):
    """Private durable identity required to retire one terminal unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_name: str = Field(pattern=r"^friday-proc-[0-9a-f]{32}\.service$")
    identity_token: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)
    boot_id: str | None = Field(default=None, min_length=1, max_length=80,
                                repr=False)
    invocation_id: str | None = Field(
        default=None, min_length=1, max_length=128, repr=False)
    control_group: str | None = Field(
        default=None, min_length=1, max_length=500, repr=False)


class ProcessBackend(Protocol):
    """Backend boundary used by :class:`ProcessBroker` and deterministic fakes."""

    def enforcement(self, spec: ProcessSpec) -> Mapping[str, Any]:
        ...

    def supports_persistence(self) -> bool:
        ...

    def launch(self, request: BackendLaunchRequest) -> BackendObservation:
        ...

    def inspect(self, unit_name: str) -> BackendObservation | None:
        ...

    def member_identity(
        self, expected: BackendObservation, pid: int,
    ) -> tuple[int, int, int, int, str] | None:
        """Return an exact identity only while PID belongs to this execution."""
        ...

    def owns_loopback_listener(
        self, expected: BackendObservation, port: int,
    ) -> bool:
        """Prove one exact loopback listener belongs to this execution."""
        ...

    def terminate(self, expected: BackendObservation, *,
                  force: bool = False) -> BackendObservation:
        ...

    def retire_terminal(self, fence: BackendTerminalFence) -> None:
        ...


class WorkloadAdmission(Protocol):
    """Subset of resource admission used for long-lived process ownership."""

    def transfer_step_to_workload_in_transaction(
        self,
        conn: Any,
        claim: ResourceClaim,
        instance_id: str,
        source_step_lease_id: str,
        source_attempt_id: str,
        source_worker_id: str,
        enforcement_json: str,
        now: datetime | str | None = None,
    ) -> Any:
        ...

    def heartbeat_workload_in_transaction(
        self, conn: Any, lease_id: str, instance_id: str, *,
        now: datetime | str | None = None,
    ) -> bool:
        ...

    def mark_workload_reconciling_in_transaction(
        self, conn: Any, lease_id: str, instance_id: str, *,
        reason: str = "monitor_lost", now: datetime | str | None = None,
    ) -> bool:
        ...

    def release_workload_in_transaction(
        self, conn: Any, lease_id: str, instance_id: str, *,
        cgroup_empty: bool, previous_runtime_id: str | None = None,
        reason: str = "completed",
        now: datetime | str | None = None,
    ) -> bool:
        ...

    def adopt_workload_in_transaction(
        self, conn: Any, lease_id: str, instance_id: str, *,
        previous_runtime_id: str, now: datetime | str | None = None,
    ) -> bool:
        ...

    def workloads_needing_reconciliation(
        self, *, now: datetime | str | None = None,
    ) -> list[dict[str, str]]:
        ...

    def mark_stale_workload_runtime_reconciling_in_transaction(
        self, conn: Any, lease_id: str, instance_id: str,
        previous_runtime_id: str, *, reason: str = "runtime_restarted",
        now: datetime | str | None = None,
    ) -> bool:
        ...


Runner = Callable[..., subprocess.CompletedProcess[str]]


class SystemdUserProcessBackend:
    """Exact-argv transient user-service backend with cgroup-owned termination."""

    def __init__(self, *, runner: Runner = subprocess.run,
                 systemd_run: str = "/usr/bin/systemd-run",
                 systemctl: str = "/usr/bin/systemctl",
                 env_executable: str = "/usr/bin/env",
                 bwrap_executable: str = "/usr/bin/bwrap",
                 session_runtime_dir: str | None = None,
                 session_bus_address: str | None = None,
                 wayland_display: str | None = None,
                 user_manager_control_group: str | None = None,
                 command_timeout_seconds: float = 15.0,
                 termination_poll_seconds: float = 0.05,
                 persistent_session: bool = False):
        for value in (systemd_run, systemctl, env_executable, bwrap_executable):
            if not Path(value).is_absolute():
                raise ValueError("backend executables must use absolute paths")
        if command_timeout_seconds <= 0 or termination_poll_seconds <= 0:
            raise ValueError("backend timeouts must be positive")
        self.runner = runner
        self.systemd_run = systemd_run
        self.systemctl = systemctl
        self.env_executable = env_executable
        self.bwrap_executable = bwrap_executable
        self.command_timeout_seconds = float(command_timeout_seconds)
        self.termination_poll_seconds = float(termination_poll_seconds)
        self._persistent_session = bool(persistent_session)
        self._session_environment = self._validated_session_environment(
            session_runtime_dir, session_bus_address)
        self._wayland_display = (
            wayland_display if wayland_display is not None
            else os.environ.get("WAYLAND_DISPLAY"))
        self._backend_identities: dict[str, ExecutableIdentity] = {}
        for path in (self.systemd_run, self.systemctl, self.env_executable):
            canonical, identity = _executable_identity(
                path, reject_interpreter=False)
            if canonical != path:
                raise ValueError("backend executable aliases are not accepted")
            self._backend_identities[path] = identity
        if Path(self.bwrap_executable).exists():
            canonical, identity = _executable_identity(
                self.bwrap_executable, reject_interpreter=False)
            if canonical != self.bwrap_executable:
                raise ValueError("backend executable aliases are not accepted")
            self._backend_identities[self.bwrap_executable] = identity
        self._user_manager_control_group = self._pin_user_manager_control_group(
            user_manager_control_group)

    def _pin_user_manager_control_group(self, supplied: str | None) -> str:
        if supplied is None:
            result = self._run([
                self.systemctl, "--user", "show",
                "--property=ControlGroup", "--value"])
            if result.returncode:
                raise ProcessBackendError(
                    "user_manager_identity_unavailable", outcome_unknown=False)
            supplied = (result.stdout or "").strip()
        expected = (f"/user.slice/user-{os.getuid()}.slice/"
                    f"user@{os.getuid()}.service")
        if supplied != expected:
            raise ProcessBackendError(
                "user_manager_identity_invalid", outcome_unknown=False)
        directory = self._cgroup_directory(supplied)
        try:
            observed = os.lstat(directory)
        except OSError as exc:
            raise ProcessBackendError(
                "user_manager_identity_unavailable",
                outcome_unknown=False) from exc
        if (not stat.S_ISDIR(observed.st_mode)
                or stat.S_ISLNK(observed.st_mode)
                or observed.st_uid != os.getuid()):
            raise ProcessBackendError(
                "user_manager_identity_invalid", outcome_unknown=False)
        return supplied

    def _unit_control_group(self, unit_name: str) -> str:
        self._ensure_unit(unit_name)
        return (f"{self._user_manager_control_group}/"
                f"friday.slice/friday-processes.slice/{unit_name}")

    @staticmethod
    def _validated_session_environment(
        runtime_dir: str | None,
        bus_address: str | None,
    ) -> dict[str, str]:
        """Pin the one local user-bus transport without inheriting host env."""
        uid = os.getuid()
        expected_runtime = f"/run/user/{uid}"
        selected_runtime = runtime_dir or os.environ.get(
            "XDG_RUNTIME_DIR") or expected_runtime
        if selected_runtime != expected_runtime:
            raise ProcessBackendError(
                "user_session_transport_invalid", outcome_unknown=False)
        try:
            runtime_lstat = os.lstat(selected_runtime)
            runtime_resolved = str(Path(selected_runtime).resolve(strict=True))
        except OSError as exc:
            raise ProcessBackendError(
                "user_session_transport_unavailable",
                outcome_unknown=False) from exc
        if (runtime_resolved != expected_runtime
                or stat.S_ISLNK(runtime_lstat.st_mode)
                or not stat.S_ISDIR(runtime_lstat.st_mode)
                or runtime_lstat.st_uid != uid
                or stat.S_IMODE(runtime_lstat.st_mode) & 0o077):
            raise ProcessBackendError(
                "user_session_transport_invalid", outcome_unknown=False)

        bus_path = f"{expected_runtime}/bus"
        expected_bus = f"unix:path={bus_path}"
        selected_bus = bus_address or os.environ.get(
            "DBUS_SESSION_BUS_ADDRESS") or expected_bus
        if selected_bus != expected_bus:
            raise ProcessBackendError(
                "user_session_transport_invalid", outcome_unknown=False)
        try:
            bus_lstat = os.lstat(bus_path)
        except OSError as exc:
            raise ProcessBackendError(
                "user_session_transport_unavailable",
                outcome_unknown=False) from exc
        if (stat.S_ISLNK(bus_lstat.st_mode)
                or not stat.S_ISSOCK(bus_lstat.st_mode)
                or bus_lstat.st_uid != uid):
            raise ProcessBackendError(
                "user_session_transport_invalid", outcome_unknown=False)
        return {
            "PATH": "/usr/bin:/bin",
            "XDG_RUNTIME_DIR": expected_runtime,
            "DBUS_SESSION_BUS_ADDRESS": expected_bus,
        }

    def supports_persistence(self) -> bool:
        return self._persistent_session

    @staticmethod
    def enforcement(spec: ProcessSpec) -> dict[str, Any]:
        limits = spec.limits
        return {
            "backend": "systemd_user_service",
            "cpu": "hard",
            "memory": "hard",
            "pids": "hard",
            "runtime": "hard",
            "network": ("namespace" if spec.sandbox.enabled
                        and not spec.sandbox.share_network else "allowed"),
            "gpu_vram": "admission_only" if spec.resources.vram_mib else "unused",
            "cpu_quota_percent": limits.cpu_quota_percent,
            "memory_high_mib": limits.memory_high_mib,
            "memory_max_mib": limits.memory_max_mib,
            "memory_swap_max_mib": limits.memory_swap_max_mib,
            "tasks_max": limits.tasks_max,
            "runtime_max_seconds": limits.runtime_max_seconds,
            "kill_mode": "control-group",
            "sandbox_fingerprint": spec.sandbox_fingerprint,
            "session_access": (
                spec.session_access.model_dump(mode="json")
                if spec.session_access is not None else None),
        }

    def _session_access_environment(
        self, access: ProcessSessionAccess,
    ) -> dict[str, str]:
        environment: dict[str, str] = {}
        if access.wayland:
            display = self._wayland_display
            if (not isinstance(display, str)
                    or _WAYLAND_DISPLAY.fullmatch(display) is None):
                raise ProcessBackendError(
                    "wayland_session_unavailable", outcome_unknown=False)
            socket_path = Path(
                self._session_environment["XDG_RUNTIME_DIR"]) / display
            try:
                observed = os.lstat(socket_path)
            except OSError as exc:
                raise ProcessBackendError(
                    "wayland_session_unavailable", outcome_unknown=False) from exc
            if (not stat.S_ISSOCK(observed.st_mode)
                    or stat.S_ISLNK(observed.st_mode)
                    or observed.st_uid != os.getuid()):
                raise ProcessBackendError(
                    "wayland_session_invalid", outcome_unknown=False)
            environment.update({
                "XDG_RUNTIME_DIR": self._session_environment["XDG_RUNTIME_DIR"],
                "WAYLAND_DISPLAY": display,
            })
        if access.session_bus:
            environment.update({
                "XDG_RUNTIME_DIR": self._session_environment["XDG_RUNTIME_DIR"],
                "DBUS_SESSION_BUS_ADDRESS":
                    self._session_environment["DBUS_SESSION_BUS_ADDRESS"],
            })
        return environment

    def _verify_backend_executable(self, path: str) -> None:
        expected = self._backend_identities.get(path)
        if expected is None:
            raise ProcessBackendError(
                "backend_executable_unavailable", outcome_unknown=False)
        canonical, observed = _executable_identity(
            path, reject_interpreter=False)
        if canonical != path or observed != expected:
            raise ProcessBackendError(
                "backend_executable_identity_changed", outcome_unknown=False)

    @staticmethod
    def _ensure_unit(unit_name: str) -> str:
        if _UNIT_NAME.fullmatch(unit_name) is None:
            raise ValueError("invalid broker-derived systemd unit")
        return unit_name

    def _run(self, command: list[str], *, timeout: float | None = None
             ) -> subprocess.CompletedProcess[str]:
        self._verify_backend_executable(command[0])
        try:
            result = self.runner(
                command, text=True, capture_output=True,
                timeout=timeout or self.command_timeout_seconds,
                check=False, env=dict(self._session_environment))
        except (OSError, subprocess.SubprocessError) as exc:
            raise ProcessBackendError("systemd_command_unavailable") from exc
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if len(stdout.encode("utf-8", errors="replace")) > _MAX_SYSTEMD_OUTPUT_BYTES \
                or len(stderr.encode("utf-8", errors="replace")) > \
                _MAX_SYSTEMD_OUTPUT_BYTES:
            raise ProcessBackendError("systemd_response_too_large")
        return result

    def _execution_argv(self, request: BackendLaunchRequest) -> list[str]:
        environment = list(request.environment)
        if request.session_access is not None:
            derived = self._session_access_environment(request.session_access)
            if set(dict(environment)) & set(derived):
                raise ProcessBackendError(
                    "process_session_environment_conflict",
                    outcome_unknown=False)
            environment.extend(sorted(derived.items()))
        if request.sandbox.enabled:
            self._verify_backend_executable(self.bwrap_executable)
            command = [
                self.bwrap_executable, "--die-with-parent", "--new-session",
                "--unshare-all", "--clearenv", "--ro-bind", "/usr", "/usr",
                "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib",
                "/lib64", "--proc", "/proc", "--dev", "/dev", "--tmpfs",
                "/tmp", "--dir", "/app", "--ro-bind", request.argv[0],
                "/app/executable",
            ]
            if request.sandbox.share_network:
                command.append("--share-net")
            command.extend([
                "--dir", "/work",
                "--bind" if request.sandbox.writable_cwd else "--ro-bind",
                request.cwd, "/work", "--chdir", "/work",
            ])
            for key, value in environment:
                command.extend(["--setenv", key, value])
            command.extend(["/app/executable", *request.argv[1:]])
            return command
        self._verify_backend_executable(self.env_executable)
        return [self.env_executable, "-i",
                *(f"{key}={value}" for key, value in environment),
                *request.argv]

    def launch(self, request: BackendLaunchRequest) -> BackendObservation:
        self._ensure_unit(request.unit_name)
        bwrap_identity = self._backend_identities.get(self.bwrap_executable)
        if bwrap_identity is not None and all((
                request.executable_identity.device == bwrap_identity.device,
                request.executable_identity.inode == bwrap_identity.inode,
                request.executable_identity.sha256 == bwrap_identity.sha256)):
            raise ProcessBackendError(
                "sandbox_wrapper_cannot_be_target", outcome_unknown=False)
        command = [
            self.systemd_run, "--user", f"--unit={request.unit_name}",
            "--slice=friday-processes.slice",
            "--service-type=exec", "--quiet",
            f"--property=Description=friday-managed:{request.identity_token}",
            "--property=Type=exec", "--property=ExitType=cgroup",
            "--property=RemainAfterExit=yes",
            "--property=KillMode=control-group", "--property=OOMPolicy=kill",
            "--property=Restart=no", "--property=SendSIGKILL=yes",
            "--property=StandardInput=null", "--property=StandardOutput=null",
            "--property=StandardError=null", "--property=LimitCORE=0",
            f"--property=CPUQuota={request.limits.cpu_quota_percent:g}%",
            f"--property=MemoryHigh={request.limits.memory_high_mib}M",
            f"--property=MemoryMax={request.limits.memory_max_mib}M",
            f"--property=MemorySwapMax={request.limits.memory_swap_max_mib}M",
            f"--property=TasksMax={request.limits.tasks_max}",
            f"--property=RuntimeMaxSec={request.limits.runtime_max_seconds}s",
            f"--property=TimeoutStopSec={request.limits.stop_grace_seconds:g}s",
            f"--working-directory={request.cwd}", "--",
            *self._execution_argv(request),
        ]
        result = self._run(command)
        deadline = time.monotonic() + min(5.0, self.command_timeout_seconds)
        retryable_topology = {
            "process_identity_unavailable", "sandbox_topology_unavailable",
            "sandbox_topology_invalid", "sandbox_topology_not_single_process",
            "sandbox_topology_changed", "target_exec_pending",
        }
        while True:
            try:
                observation = self.inspect(request.unit_name)
            except ProcessBackendError as exc:
                if (exc.code not in retryable_topology
                        or time.monotonic() >= deadline):
                    raise
                time.sleep(self.termination_poll_seconds)
                continue
            if observation is not None:
                if observation.identity_token != request.identity_token:
                    raise ProcessIdentityError()
                if (observation.state in {"running", "stopping"}
                        and (observation.exe_device
                             != request.executable_identity.device
                             or observation.exe_inode
                             != request.executable_identity.inode
                             or observation.exe_sha256
                             != request.executable_identity.sha256)):
                    raise ProcessIdentityError()
                return observation
            break
        if result.returncode:
            raise ProcessBackendError("systemd_launch_failed")
        raise ProcessBackendError("systemd_launch_unobservable")

    @staticmethod
    def _boot_id() -> str:
        try:
            value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError as exc:
            raise ProcessBackendError("boot_identity_unavailable") from exc
        if not value or len(value) > 80:
            raise ProcessBackendError("boot_identity_invalid")
        return value

    @staticmethod
    def _process_identity(pid: int) -> tuple[int, int, int, str]:
        try:
            stat_text = Path(f"/proc/{pid}/stat").read_text()
            remainder = stat_text.rsplit(")", 1)[1].split()
            start_ticks = int(remainder[19])
            descriptor = os.open(
                f"/proc/{pid}/exe", os.O_RDONLY | os.O_CLOEXEC)
        except (OSError, ValueError, IndexError) as exc:
            raise ProcessBackendError("process_identity_unavailable") from exc
        try:
            observed = os.fstat(descriptor)
            if not stat.S_ISREG(observed.st_mode):
                raise ProcessBackendError("process_identity_invalid")
            digest = _sha256_fd(descriptor)
        finally:
            os.close(descriptor)
        return start_ticks, int(observed.st_dev), int(observed.st_ino), digest

    @staticmethod
    def _cgroup_directory(control_group: str) -> Path:
        if (not control_group.startswith("/") or ".." in Path(control_group).parts
                or len(control_group) > 500):
            raise ProcessBackendError("control_group_identity_invalid")
        root = Path("/sys/fs/cgroup")
        directory = root / control_group.lstrip("/")
        try:
            resolved = directory.resolve(strict=True)
        except FileNotFoundError:
            return directory
        except OSError as exc:
            raise ProcessBackendError(
                "control_group_status_unavailable") from exc
        if resolved != directory or root not in resolved.parents:
            raise ProcessBackendError("control_group_identity_invalid")
        return directory

    @classmethod
    def _cgroup_empty(cls, control_group: str) -> bool:
        events = cls._cgroup_directory(control_group) / "cgroup.events"
        try:
            values = dict(line.split(None, 1) for line in events.read_text().splitlines()
                          if len(line.split(None, 1)) == 2)
        except FileNotFoundError:
            return True
        except OSError as exc:
            raise ProcessBackendError("control_group_status_unavailable") from exc
        return values.get("populated") == "0"

    def _sandbox_target_identity(
        self,
        control_group: str,
        wrapper_pid: int,
    ) -> tuple[int, int, int, int, str]:
        """Resolve the sole non-wrapper target in a v1 process cgroup."""
        procs = self._cgroup_directory(control_group) / "cgroup.procs"
        try:
            raw = procs.read_bytes()
        except OSError as exc:
            raise ProcessBackendError(
                "sandbox_topology_unavailable") from exc
        if len(raw) > 16_384:
            raise ProcessBackendError("sandbox_topology_invalid")
        try:
            members = [int(item) for item in raw.split()]
        except ValueError as exc:
            raise ProcessBackendError("sandbox_topology_invalid") from exc
        if (len(members) != len(set(members)) or wrapper_pid not in members
                or any(pid <= 0 for pid in members) or len(members) > 8):
            raise ProcessBackendError("sandbox_topology_invalid")
        wrapper_identity = self._backend_identities.get(self.bwrap_executable)
        if wrapper_identity is None:
            raise ProcessBackendError("backend_executable_unavailable")
        wrappers: list[int] = []
        targets: list[tuple[int, int, int, int, str]] = []
        for pid in members:
            start_ticks, device, inode, executable_hash = self._process_identity(pid)
            if (device == wrapper_identity.device
                    and inode == wrapper_identity.inode
                    and executable_hash == wrapper_identity.sha256):
                wrappers.append(pid)
            else:
                targets.append(
                    (pid, start_ticks, device, inode, executable_hash))
        if (wrapper_pid not in wrappers or not 1 <= len(wrappers) <= 4
                or len(targets) != 1):
            raise ProcessBackendError("sandbox_topology_not_single_process")
        try:
            after = {int(item) for item in procs.read_bytes().split()}
        except (OSError, ValueError) as exc:
            raise ProcessBackendError(
                "sandbox_topology_unavailable") from exc
        if after != set(members):
            raise ProcessBackendError("sandbox_topology_changed")
        return targets[0]

    @classmethod
    def _pid_in_cgroup(cls, control_group: str, pid: int) -> bool:
        procs = cls._cgroup_directory(control_group) / "cgroup.procs"
        try:
            raw = procs.read_bytes()
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise ProcessBackendError(
                "control_group_status_unavailable") from exc
        if len(raw) > 64 * 1024:
            raise ProcessBackendError("control_group_status_invalid")
        try:
            return pid in {int(item) for item in raw.split()}
        except ValueError as exc:
            raise ProcessBackendError("control_group_status_invalid") from exc

    def member_identity(
        self, expected: BackendObservation, pid: int,
    ) -> tuple[int, int, int, int, str] | None:
        """Fence one member by execution, cgroup, PID reuse, and executable.

        Membership is checked on both sides of the process identity read and
        the unit's complete live execution is re-observed afterwards. This is
        private evidence used by compound desktop receipts; no PID, cgroup, or
        executable identity is projected to the model.
        """
        if (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                or expected.state not in {"running", "stopping"}
                or expected.cgroup_empty):
            raise ProcessBackendError("process_member_request_invalid")
        before = self.inspect(expected.unit_name)
        if before is None or not expected.same_live_execution(before):
            raise ProcessIdentityError()
        if not self._pid_in_cgroup(expected.control_group, pid):
            return None
        start_ticks, device, inode, executable_hash = self._process_identity(pid)
        if not self._pid_in_cgroup(expected.control_group, pid):
            return None
        after = self.inspect(expected.unit_name)
        if after is None or not expected.same_live_execution(after):
            raise ProcessIdentityError()
        return pid, start_ticks, device, inode, executable_hash

    @staticmethod
    def _loopback_listener_inode(port: int) -> str | None:
        if (isinstance(port, bool) or not isinstance(port, int)
                or not 1024 <= port <= 65535):
            raise ProcessBackendError("loopback_listener_request_invalid")
        encoded_port = f"{port:04X}"
        records: list[str] = []
        for path, address in (
                (Path("/proc/net/tcp"), "0100007F"),
                (Path("/proc/net/tcp6"),
                 "00000000000000000000000001000000")):
            try:
                lines = path.read_text().splitlines()
            except FileNotFoundError:
                if path.name == "tcp6":
                    continue
                raise ProcessBackendError("loopback_listener_unavailable")
            except OSError as exc:
                raise ProcessBackendError(
                    "loopback_listener_unavailable") from exc
            if len(lines) > 65_536:
                raise ProcessBackendError("loopback_listener_invalid")
            for line in lines[1:]:
                fields = line.split()
                if len(fields) < 10:
                    raise ProcessBackendError("loopback_listener_invalid")
                try:
                    observed_address, observed_port = fields[1].split(":", 1)
                    socket_uid = int(fields[7])
                    inode = fields[9]
                except (TypeError, ValueError) as exc:
                    raise ProcessBackendError(
                        "loopback_listener_invalid") from exc
                if (fields[3] == "0A"
                        and observed_address.upper() == address
                        and observed_port.upper() == encoded_port):
                    if socket_uid != os.geteuid() or not inode.isdigit() \
                            or inode == "0":
                        raise ProcessIdentityError()
                    records.append(inode)
        return records[0] if len(records) == 1 else None

    @classmethod
    def _cgroup_members(
        cls, control_group: str, *, maximum: int = 4096,
    ) -> tuple[int, ...]:
        procs = cls._cgroup_directory(control_group) / "cgroup.procs"
        try:
            raw = procs.read_bytes()
        except OSError as exc:
            raise ProcessBackendError(
                "control_group_status_unavailable") from exc
        if len(raw) > 64 * 1024:
            raise ProcessBackendError("control_group_status_invalid")
        try:
            members = tuple(int(item) for item in raw.split())
        except ValueError as exc:
            raise ProcessBackendError("control_group_status_invalid") from exc
        if (not members or len(members) > maximum
                or len(set(members)) != len(members)
                or any(pid <= 0 for pid in members)):
            raise ProcessBackendError("control_group_status_invalid")
        return members

    def owns_loopback_listener(
        self, expected: BackendObservation, port: int,
    ) -> bool:
        """Double-fence a listener inode inside the exact managed cgroup."""
        if expected.state not in {"running", "stopping"} or expected.cgroup_empty:
            raise ProcessBackendError("loopback_listener_request_invalid")
        before = self.inspect(expected.unit_name)
        if before is None or not expected.same_live_execution(before):
            raise ProcessIdentityError()
        inode = self._loopback_listener_inode(port)
        if inode is None:
            return False
        owners: set[int] = set()
        for pid in self._cgroup_members(expected.control_group):
            descriptors = Path(f"/proc/{pid}/fd")
            try:
                entries = list(descriptors.iterdir())
            except FileNotFoundError:
                continue
            except PermissionError:
                # Sandboxed Chromium children may deliberately make their fd
                # tables unreadable. They cannot become positive evidence;
                # continue looking for the exact accessible listener owner.
                continue
            except OSError as exc:
                raise ProcessBackendError(
                    "loopback_listener_owner_unavailable") from exc
            if len(entries) > 16_384:
                raise ProcessBackendError("loopback_listener_owner_invalid")
            for descriptor in entries:
                try:
                    target = os.readlink(descriptor)
                except FileNotFoundError:
                    continue
                except PermissionError:
                    continue
                except OSError as exc:
                    raise ProcessBackendError(
                        "loopback_listener_owner_unavailable") from exc
                if target == f"socket:[{inode}]":
                    owners.add(pid)
                    break
        if not owners:
            return False
        if any(not self._pid_in_cgroup(expected.control_group, pid)
               for pid in owners):
            return False
        after = self.inspect(expected.unit_name)
        if after is None or not expected.same_live_execution(after):
            raise ProcessIdentityError()
        return True

    def inspect(self, unit_name: str) -> BackendObservation | None:
        self._ensure_unit(unit_name)
        properties = (
            "LoadState", "ActiveState", "SubState", "Description", "InvocationID",
            "ControlGroup", "MainPID", "ExecMainCode", "ExecMainStatus", "Result",
            "Job",
        )
        command = [self.systemctl, "--user", "show", unit_name,
                   *(f"--property={name}" for name in properties)]
        result = self._run(command)
        if result.returncode:
            raise ProcessBackendError("systemd_inspection_failed")
        values: dict[str, str] = {}
        for line in (result.stdout or "").splitlines():
            key, separator, value = line.partition("=")
            if (not separator or key not in properties or key in values):
                raise ProcessBackendError("systemd_inspection_invalid")
            values[key] = value.strip()
        if not values:
            raise ProcessBackendError("systemd_inspection_invalid")
        if values.get("LoadState") == "not-found":
            return None
        if set(values) != set(properties) or values.get("LoadState") != "loaded":
            raise ProcessBackendError("systemd_inspection_invalid")
        description = values.get("Description", "")
        prefix = "friday-managed:"
        identity_token = description.removeprefix(prefix) if description.startswith(
            prefix) else ""
        if _HASH.fullmatch(identity_token) is None:
            raise ProcessIdentityError()
        invocation_id = values.get("InvocationID", "")
        reported_control_group = values.get("ControlGroup", "")
        expected_control_group = self._unit_control_group(unit_name)
        if (reported_control_group
                and reported_control_group != expected_control_group):
            raise ProcessIdentityError()
        control_group = expected_control_group
        if not invocation_id:
            raise ProcessBackendError("systemd_identity_incomplete")
        try:
            pid = int(values.get("MainPID") or 0)
        except ValueError as exc:
            raise ProcessBackendError("systemd_identity_invalid") from exc
        start_ticks = exe_device = exe_inode = None
        exe_hash = None
        if pid > 0:
            if not self._pid_in_cgroup(control_group, pid):
                raise ProcessIdentityError()
            start_ticks, exe_device, exe_inode, exe_hash = self._process_identity(pid)
            env_identity = self._backend_identities[self.env_executable]
            if (exe_device == env_identity.device
                    and exe_inode == env_identity.inode
                    and exe_hash == env_identity.sha256):
                self._verify_backend_executable(self.env_executable)
                raise ProcessBackendError("target_exec_pending")
            bwrap_identity = self._backend_identities.get(self.bwrap_executable)
            if (bwrap_identity is not None
                    and exe_device == bwrap_identity.device
                    and exe_inode == bwrap_identity.inode
                    and exe_hash == bwrap_identity.sha256):
                self._verify_backend_executable(self.bwrap_executable)
                (pid, start_ticks, exe_device, exe_inode,
                 exe_hash) = self._sandbox_target_identity(
                    control_group, pid)
        cgroup_empty = self._cgroup_empty(control_group)
        active = values.get("ActiveState", "")
        if pid <= 0 and cgroup_empty:
            # RemainAfterExit keeps the trusted InvocationID/control-group/token
            # inspectable until the broker durably records and releases exit.
            state = "exited"
        elif active in {"active", "reloading"}:
            state = "running"
        elif active == "activating":
            state = "starting"
        elif active == "deactivating":
            state = "stopping"
        else:
            state = "exited"
        try:
            main_code = int(values.get("ExecMainCode") or 0)
            main_status = int(values.get("ExecMainStatus") or 0)
        except ValueError:
            main_code = main_status = 0
        exit_code = main_status if main_code == 1 and main_status >= 0 else None
        exit_signal = main_status if main_code in {2, 3} and main_status > 0 else None
        return BackendObservation(
            unit_name=unit_name, identity_token=identity_token, state=state,
            boot_id=self._boot_id(), invocation_id=invocation_id,
            control_group=control_group,
            leader_pid=pid if pid > 0 else None, start_ticks=start_ticks,
            exe_device=exe_device, exe_inode=exe_inode, exe_sha256=exe_hash,
            cgroup_empty=cgroup_empty, exit_code=exit_code,
            exit_signal=exit_signal,
            result_code=_stable_code(values.get("Result"), "unknown"),
            unit_active_state=_stable_code(active, "unknown"),
            unit_sub_state=_stable_code(values.get("SubState"), "unknown"),
            unit_job_pending=bool(values.get("Job")))

    def terminate(self, expected: BackendObservation, *,
                  force: bool = False) -> BackendObservation:
        current = self.inspect(expected.unit_name)
        if current is None or not expected.same_live_execution(current):
            raise ProcessIdentityError()
        if force:
            command = [
                self.systemctl, "--user", "kill", "--kill-whom=all",
                "--signal=SIGKILL", expected.unit_name,
            ]
            result = self._run(command)
        else:
            # The unit's trusted TimeoutStopSec and SendSIGKILL properties make
            # this a bounded TERM -> KILL transition for the complete cgroup.
            command = [self.systemctl, "--user", "stop", expected.unit_name]
            result = self._run(command, timeout=310.0)
        # A coincident natural exit is not proof that a failed systemd request
        # was accepted.  Check the dispatch result before observing any terminal
        # postcondition so callers cannot attribute that exit to this action.
        if result.returncode:
            raise ProcessBackendError("systemd_termination_failed")
        deadline = time.monotonic() + (
            5.0 if force else min(5.0, self.command_timeout_seconds))
        last = current
        while time.monotonic() < deadline:
            observed = self.inspect(expected.unit_name)
            if observed is None:
                # Unit absence is not enough: a transiently missing/unloaded
                # unit can still have processes in the exact retained cgroup.
                # Only an absent or authoritatively unpopulated cgroup proves
                # that termination completed.
                if self._cgroup_empty(expected.control_group):
                    return last.model_copy(update={
                        "state": "exited", "cgroup_empty": True,
                        "result_code": "terminated"})
                time.sleep(self.termination_poll_seconds)
                continue
            if (observed.identity_token != expected.identity_token
                    or observed.boot_id != expected.boot_id
                    or observed.invocation_id != expected.invocation_id
                    or observed.control_group != expected.control_group):
                raise ProcessIdentityError()
            last = observed
            if observed.cgroup_empty:
                return observed.model_copy(update={"state": "exited"})
            time.sleep(self.termination_poll_seconds)
        return last.model_copy(update={"state": "stopping"})

    def retire_terminal(self, fence: BackendTerminalFence) -> None:
        """Unload one retained terminal unit behind its durable identity.

        Unit absence is accepted only together with an empty deterministic
        cgroup.  A loaded unit is stopped only when its token, invocation and
        cgroup still match the terminal projection and it has no members.
        """
        unit_name = self._ensure_unit(fence.unit_name)
        expected_control_group = self._unit_control_group(unit_name)
        if (fence.control_group is not None
                and fence.control_group != expected_control_group):
            raise ProcessCleanupBlocked(
                "process_unit_cleanup_control_group_mismatch")
        try:
            current = self.inspect(unit_name)
        except ProcessIdentityError as exc:
            raise ProcessCleanupBlocked(
                "process_unit_cleanup_identity_mismatch") from exc
        if current is None:
            if not self._cgroup_empty(expected_control_group):
                raise ProcessCleanupBlocked(
                    "process_unit_cleanup_cgroup_not_empty")
            return
        if fence.invocation_id is None or fence.control_group is None:
            raise ProcessCleanupBlocked(
                "process_unit_cleanup_fence_incomplete")
        if (current.identity_token != fence.identity_token
                or current.invocation_id != fence.invocation_id
                or current.control_group != fence.control_group
                or (fence.boot_id is not None
                    and current.boot_id != fence.boot_id)):
            raise ProcessCleanupBlocked(
                "process_unit_cleanup_identity_mismatch")
        if not current.cgroup_empty:
            raise ProcessCleanupBlocked(
                "process_unit_cleanup_cgroup_not_empty")
        if current.unit_job_pending:
            raise ProcessBackendError(
                "process_unit_cleanup_job_pending", outcome_unknown=False)
        if (current.unit_active_state, current.unit_sub_state) not in {
                ("active", "exited"), ("failed", "failed"),
                ("inactive", "dead")}:
            raise ProcessBackendError(
                "process_unit_cleanup_state_not_terminal",
                outcome_unknown=False)

        # systemd has no compare-and-stop API keyed by InvocationID.  Friday
        # therefore treats its same-UID user-manager namespace as trusted (that
        # UID can signal these processes directly anyway).  job-mode=fail keeps
        # this stop from replacing an already queued start/restart job.
        result = self._run([
            self.systemctl, "--user", "stop", "--job-mode=fail", unit_name,
        ])
        try:
            observed = self.inspect(unit_name)
        except ProcessIdentityError as exc:
            raise ProcessCleanupBlocked(
                "process_unit_cleanup_identity_mismatch") from exc
        if observed is None:
            if self._cgroup_empty(expected_control_group):
                return
            raise ProcessCleanupBlocked(
                "process_unit_cleanup_cgroup_not_empty")
        if (observed.identity_token != fence.identity_token
                or observed.invocation_id != fence.invocation_id
                or observed.control_group != fence.control_group
                or (fence.boot_id is not None
                    and observed.boot_id != fence.boot_id)):
            raise ProcessCleanupBlocked(
                "process_unit_cleanup_identity_mismatch")
        if not observed.cgroup_empty:
            raise ProcessCleanupBlocked(
                "process_unit_cleanup_cgroup_not_empty")
        if result.returncode:
            raise ProcessBackendError("systemd_cleanup_failed")
        # A still-loaded, empty, exact unit is harmless but not retired.  Retry
        # it later instead of claiming cleanup completed.
        raise ProcessBackendError(
            "systemd_cleanup_incomplete", outcome_unknown=False)


def _redacted_values(values: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for name, value in values.items():
        encoded = canonical_json(value)
        item: dict[str, Any] = {
            "type": ("boolean" if isinstance(value, bool) else
                     "integer" if isinstance(value, int) else "string"),
            "sha256": sha256_text(encoded),
        }
        if isinstance(value, str):
            item["characters"] = len(value)
        redacted[str(name)] = item
    return redacted


class ProcessBroker:
    """Durable prepare-before-effect broker for curated local processes."""

    def __init__(
        self,
        graph: GraphStore,
        registry: ProcessSpecRegistry,
        backend: ProcessBackend,
        admission: WorkloadAdmission,
        *,
        state_root: str | Path | None = None,
        cipher: StepPayloadCipher | None = None,
    ):
        self.graph = graph
        self.registry = registry
        self.backend = backend
        self.admission = admission
        self.state_root = Path(state_root or graph.path.parent).expanduser().resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.state_root, 0o700)
        self.lock_root = self.state_root / "process-locks"
        self.lock_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.lock_root, 0o700)
        self._cipher = cipher or StepPayloadCipher(
            self.state_root / "process-payloads.key")
        self.degraded_specs: dict[str, str] = {}
        for spec in registry.specs():
            self._persist_spec(spec, degrade_on_mismatch=True)
        self._revoke_absent_specs()

    @staticmethod
    def _spec_context(spec: ProcessSpec) -> str:
        return f"process-spec\0{spec.spec_id}\0{spec.fingerprint}"

    @staticmethod
    def _args_context(instance_id: str, spec: ProcessSpec,
                      args_sha256: str) -> str:
        return (f"process-args\0{instance_id}\0{spec.spec_id}\0"
                f"{spec.fingerprint}\0{args_sha256}")

    @staticmethod
    def _unit_name(instance_id: str) -> str:
        suffix = instance_id.removeprefix("process_")
        if re.fullmatch(r"[0-9a-f]{32}", suffix) is None:
            raise ValueError("process instance ID cannot form a safe unit")
        return f"friday-proc-{suffix}.service"

    @staticmethod
    def _identity_token(instance_id: str, spec_fingerprint: str,
                        args_sha256: str) -> str:
        return sha256_text("\0".join((instance_id, spec_fingerprint,
                                      args_sha256)))

    @contextmanager
    def _operation_lock(self, identity: str) -> Iterator[None]:
        lock_name = hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".lock"
        path = self.lock_root / lock_name
        flags = os.O_RDWR | os.O_CREAT | fs.PRIVATE_OPEN_FLAGS
        descriptor = os.open(path, flags, 0o600)
        fs.chmod_private(descriptor, 0o600)
        try:
            fs.lock_exclusive(descriptor)
            yield
        finally:
            fs.unlock(descriptor)
            os.close(descriptor)

    def _persist_spec(self, spec: ProcessSpec, *,
                      source_task_id: str | None = None,
                      degrade_on_mismatch: bool = False) -> None:
        safe = {"spec_id": spec.spec_id, **spec.safe_display(),
                "spec_fingerprint": spec.fingerprint,
                "sandbox_fingerprint": spec.sandbox_fingerprint,
                "status": "active"}
        ciphertext = self._cipher.seal(
            spec.model_dump(mode="json"), context=self._spec_context(spec))
        now = utc_now()
        with self.graph.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM process_specs WHERE spec_id=?", (spec.spec_id,)
            ).fetchone()
            if row is not None:
                if (row["spec_sha256"] != spec.fingerprint
                        or row["sandbox_fingerprint"] != spec.sandbox_fingerprint
                        or row["name"] != spec.name
                        or int(row["version"]) != spec.version):
                    if degrade_on_mismatch:
                        # An OS/package update must not silently repin an
                        # approved executable, but one unavailable capability
                        # must not crash-loop the entire assistant either.
                        self.degraded_specs[spec.spec_id] = (
                            "durable_process_spec_binding_changed")
                        return
                    raise ProcessSpecError("durable process spec binding differs")
                return
            event_id, seq = self.graph.append_event(
                conn, "process_spec.registered", safe,
                actor="process_registry", task_id=source_task_id)
            self.graph.append_node(
                conn, "process_spec", safe, event_id=event_id,
                node_id=spec.spec_id)
            conn.execute(
                """INSERT INTO process_specs
                   (spec_id,name,version,spec_ciphertext,display_json,spec_sha256,
                    sandbox_fingerprint,status,source_task_id,created_at,updated_at,
                    last_event_seq)
                   VALUES (?,?,?,?,?,?,?,'active',?,?,?,?)""",
                (spec.spec_id, spec.name, spec.version, ciphertext,
                 canonical_json(spec.safe_display()), spec.fingerprint,
                 spec.sandbox_fingerprint, source_task_id, now, now, seq))

    def register_spec(self, spec: ProcessSpec, *,
                      source_task_id: str | None = None) -> dict[str, Any]:
        selected = self.registry.register(spec)
        self._persist_spec(selected, source_task_id=source_task_id)
        return {"spec_id": selected.spec_id, **selected.safe_display(),
                "status": "active"}

    def _revoke_absent_specs(self) -> int:
        """Journal unused durable specs removed from the curated registry.

        An absent spec with a live instance is retained and surfaced as
        degraded because reconciliation still needs its exact contract. Once
        no nonterminal instance remains, the obsolete authority is revoked.
        """
        current_ids = {spec.spec_id for spec in self.registry.specs()}
        revoked = 0
        with self.graph.transaction() as conn:
            rows = conn.execute(
                "SELECT spec_id,name,version FROM process_specs "
                "WHERE status='active' ORDER BY spec_id").fetchall()
            for row in rows:
                spec_id = str(row["spec_id"])
                if spec_id in current_ids:
                    continue
                placeholders = ",".join("?" for _ in _ACTIVE_INSTANCE_STATES)
                live = conn.execute(
                    f"SELECT 1 FROM process_instances WHERE spec_id=? "
                    f"AND state IN ({placeholders}) LIMIT 1",
                    (spec_id, *sorted(_ACTIVE_INSTANCE_STATES))).fetchone()
                if live is not None:
                    self.degraded_specs[spec_id] = (
                        "durable_process_spec_absent_with_active_instance")
                    continue
                body = {
                    "spec_id": spec_id,
                    "name": str(row["name"]),
                    "version": int(row["version"]),
                    "status": "revoked",
                    "reason": "absent_from_curated_registry",
                }
                _, seq = self.graph.append_event(
                    conn, "process_spec.revoked", body,
                    actor="process_registry")
                changed = conn.execute(
                    "UPDATE process_specs SET status='revoked',updated_at=?,"
                    "last_event_seq=? WHERE spec_id=? AND status='active'",
                    (utc_now(), seq, spec_id)).rowcount
                if changed != 1:
                    raise ProcessBackendError(
                        "process_spec_revocation_fence_lost",
                        outcome_unknown=False)
                revoked += 1
        return revoked

    def list_specs(self) -> list[dict[str, Any]]:
        """Return only curated specs whose durable active binding still matches."""
        with self.graph._connect() as conn:
            rows = {
                str(row["spec_id"]): row for row in conn.execute(
                    "SELECT * FROM process_specs WHERE status='active'"
                ).fetchall()
            }
        values: list[dict[str, Any]] = []
        for item in self.registry.list():
            spec = self.registry.get(str(item["spec_id"]))
            row = rows.get(spec.spec_id)
            try:
                spec.verify_current()
            except ProcessSpecError:
                continue
            if (row is not None and row["spec_sha256"] == spec.fingerprint
                    and row["sandbox_fingerprint"] == spec.sandbox_fingerprint):
                values.append(item | {"status": "active"})
        return values

    def binding_for_launch(
        self,
        spec_id: str,
        parameter_values: Mapping[str, Any],
        *,
        file_validator: FileParameterValidator | None = None,
    ) -> ProcessLaunchBinding:
        spec = self.registry.get(spec_id)
        self._require_durable_active_spec(spec)
        spec.verify_current()
        rendered = self.registry.render(
            spec_id, parameter_values, file_validator=file_validator)
        return ProcessLaunchBinding(
            spec_id=spec.spec_id, name=spec.name, version=spec.version,
            spec_fingerprint=spec.fingerprint,
            sandbox_fingerprint=spec.sandbox_fingerprint,
            args_sha256=rendered.args_sha256,
            executable_identity=spec.executable_identity,
            resource_claim=spec.resources, persistent=spec.persistent)

    @staticmethod
    def _binding_for_row(
        row: Any,
        operation: Literal["inspect", "terminate"],
    ) -> ProcessInstanceBinding:
        runtime_values = {
            "boot_id": row["boot_id"],
            "invocation_id": row["invocation_id"],
            "control_group": row["control_group"],
            "leader_pid": row["leader_pid"],
            "start_ticks": row["start_ticks"],
            "exe_device": row["exe_device"],
            "exe_inode": row["exe_inode"],
            "exe_sha256": row["exe_sha256"],
        }
        runtime_identity_sha256 = (
            sha256_text("process-runtime-identity\0" + canonical_json(
                runtime_values))
            if any(value is not None for value in runtime_values.values())
            else None)
        state_values = {
            "instance_id": row["instance_id"],
            "state": row["state"],
            "result_code": row["result_code"],
            "started_at": row["started_at"],
            "stop_requested_at": row["stop_requested_at"],
            "finished_at": row["finished_at"],
            "runtime_identity_sha256": runtime_identity_sha256,
        }
        return ProcessInstanceBinding(
            operation=operation,
            instance_id=str(row["instance_id"]),
            spec_id=str(row["spec_id"]),
            spec_fingerprint=str(row["spec_fingerprint"]),
            sandbox_fingerprint=str(row["sandbox_fingerprint"]),
            args_sha256=str(row["args_sha256"]),
            state=str(row["state"]),
            runtime_identity_sha256=runtime_identity_sha256,
            state_fingerprint=sha256_text(
                "process-instance-state\0" + canonical_json(state_values)),
            persistent=bool(row["persistent"]),
        )

    def binding_for_instance(
        self,
        instance_id: str,
        operation: Literal["inspect", "terminate"],
    ) -> ProcessInstanceBinding:
        if operation not in {"inspect", "terminate"}:
            raise ValueError("invalid process instance operation")
        with self._operation_lock(instance_id):
            row, _ = self._load_row(instance_id)
            return self._binding_for_row(row, operation)

    @classmethod
    def _verify_expected_binding(
        cls,
        row: Any,
        operation: Literal["inspect", "terminate"],
        expected_binding: ProcessInstanceBinding | Mapping[str, Any] | None,
    ) -> None:
        if expected_binding is None:
            return
        try:
            expected = ProcessInstanceBinding.model_validate(expected_binding)
        except (TypeError, ValueError) as exc:
            raise ProcessBindingError() from exc
        if expected.operation != operation or expected != cls._binding_for_row(
                row, operation):
            raise ProcessBindingError()

    def verify_instance_binding(
        self,
        binding: ProcessInstanceBinding | Mapping[str, Any],
    ) -> bool:
        try:
            expected = ProcessInstanceBinding.model_validate(binding)
            with self._operation_lock(expected.instance_id):
                row, _ = self._load_row(expected.instance_id)
                self._verify_expected_binding(
                    row, expected.operation, expected)
        except (TypeError, ValueError, ProcessBrokerError):
            return False
        return True

    @staticmethod
    def _operation_context(
        value: ProcessOperationContext | Mapping[str, Any] | None,
    ) -> ProcessOperationContext:
        try:
            return ProcessOperationContext.model_validate(value)
        except (TypeError, ValueError) as exc:
            raise ProcessBindingError() from exc

    @classmethod
    def _target_boundary_sha256_from_row(cls, row: Mapping[str, Any]) -> str:
        values = {
            "unit_name": row["unit_name"],
            "identity_token": cls._identity_token(
                str(row["instance_id"]), str(row["spec_fingerprint"]),
                str(row["args_sha256"])),
            "boot_id": row["boot_id"],
            "invocation_id": row["invocation_id"],
            "control_group": row["control_group"],
            "leader_pid": row["leader_pid"],
            "start_ticks": row["start_ticks"],
            "exe_device": row["exe_device"],
            "exe_inode": row["exe_inode"],
            "exe_sha256": row["exe_sha256"],
        }
        return sha256_text(
            "process-termination-boundary\0" + canonical_json(values))

    @staticmethod
    def _target_boundary_sha256_from_observation(
        observation: BackendObservation,
    ) -> str:
        values = {
            "unit_name": observation.unit_name,
            "identity_token": observation.identity_token,
            "boot_id": observation.boot_id,
            "invocation_id": observation.invocation_id,
            "control_group": observation.control_group,
            "leader_pid": observation.leader_pid,
            "start_ticks": observation.start_ticks,
            "exe_device": observation.exe_device,
            "exe_inode": observation.exe_inode,
            "exe_sha256": observation.exe_sha256,
        }
        return sha256_text(
            "process-termination-boundary\0" + canonical_json(values))

    @staticmethod
    def _operation_event_body(
        operation: Mapping[str, Any],
        status: str,
        *,
        error_code: str | None = None,
        postcondition_state: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "operation_id": str(operation["operation_id"]),
            "instance_id": str(operation["instance_id"]),
            "task_id": str(operation["task_id"]),
            "step_id": str(operation["step_id"]),
            "action_id": str(operation["action_id"]),
            "attempt_id": str(operation["attempt_id"]),
            "status": status,
            "args_sha256": str(operation["args_sha256"]),
            "executor_binding_sha256": str(
                operation["executor_binding_sha256"]),
            "target_boundary_sha256": str(
                operation["target_boundary_sha256"]),
            "forced": bool(operation["force"]),
        }
        if error_code is not None:
            body["error_code"] = _stable_code(
                error_code, "process_termination_outcome_unknown")
        if postcondition_state is not None:
            body["postcondition_state"] = postcondition_state
        return body

    @staticmethod
    def _exact_termination_args_sha256(instance_id: str) -> str:
        return sha256_text(canonical_json({"instance_id": instance_id}))

    @staticmethod
    def _claim_lease_is_live(expires_at: Any, now: str) -> bool:
        return isinstance(expires_at, str) and expires_at > now

    def _validate_exact_operation_claim(
        self,
        conn: Any,
        context: ProcessOperationContext,
        *,
        instance_id: str,
        binding_json: str,
        binding_sha256: str,
        args_sha256: str,
    ) -> None:
        step = conn.execute(
            "SELECT * FROM task_steps WHERE step_id=?", (context.step_id,)
        ).fetchone()
        receipt = conn.execute(
            "SELECT * FROM action_receipts WHERE idempotency_key=?",
            (context.idempotency_key,),
        ).fetchone()
        attempt = conn.execute(
            "SELECT * FROM action_attempts WHERE attempt_id=?",
            (context.attempt_id,),
        ).fetchone()
        approval = (conn.execute(
            "SELECT * FROM approval_state WHERE approval_id=?",
            (step["approval_id"],),
        ).fetchone() if step is not None and step["approval_id"] else None)
        approval_args_sha256 = None
        if approval is not None:
            try:
                approval_args = json.loads(str(approval["args_json"]))
            except (TypeError, json.JSONDecodeError):
                approval_args = None
            if isinstance(approval_args, dict):
                approval_args_sha256 = approval_args.get("_args_sha256")
        task = conn.execute(
            "SELECT status,cancellation_requested FROM task_state WHERE task_id=?",
            (context.task_id,),
        ).fetchone()
        batch = (conn.execute(
            "SELECT status,task_id FROM task_step_batches WHERE batch_id=?",
            (step["batch_id"],),
        ).fetchone() if step is not None else None)
        now = utc_now()
        if (step is None or receipt is None or attempt is None
                or task is None or batch is None
                or step["status"] != "running"
                or step["task_id"] != context.task_id
                or step["action_id"] != context.action_id
                or step["idempotency_key"] != context.idempotency_key
                or step["tool_name"] != "machine_terminate_process"
                or step["args_sha256"] != args_sha256
                or step["executor_binding_json"] != binding_json
                or sha256_text(str(step["executor_binding_json"]))
                    != binding_sha256
                or step["lease_id"] != context.lease_id
                or step["worker_id"] != context.worker_id
                or int(step["attempt_count"]) != context.attempt_number
                or not self._claim_lease_is_live(step["lease_expires_at"], now)
                or step["approval_status"] != "approved"
                or approval is None
                or approval["approval_id"] != step["approval_id"]
                or approval["task_id"] != context.task_id
                or approval["step_id"] != context.step_id
                or approval["tool_name"] != "machine_terminate_process"
                or approval["status"] != "approved"
                or not approval["decided_at"]
                or approval_args_sha256 != args_sha256
                or receipt["status"] != "running"
                or receipt["task_id"] != context.task_id
                or receipt["step_id"] != context.step_id
                or receipt["action_id"] != context.action_id
                or receipt["tool_name"] != "machine_terminate_process"
                or receipt["args_sha256"] != args_sha256
                or attempt["status"] != "running"
                or attempt["step_id"] != context.step_id
                or attempt["idempotency_key"] != context.idempotency_key
                or attempt["lease_id"] != context.lease_id
                or attempt["worker_id"] != context.worker_id
                or int(attempt["attempt_number"]) != context.attempt_number
                or task["status"] in {"completed", "failed", "cancelled"}
                or bool(task["cancellation_requested"])
                or batch["status"] != "running"
                or batch["task_id"] != context.task_id):
            raise ProcessBindingError()

    @staticmethod
    def _operation_matches_context(
        operation: Mapping[str, Any],
        context: ProcessOperationContext,
        *,
        instance_id: str,
        binding_json: str,
        binding_sha256: str,
        args_sha256: str,
        target_boundary_sha256: str,
        force: bool,
    ) -> bool:
        return bool(
            operation["idempotency_key"] == context.idempotency_key
            and operation["task_id"] == context.task_id
            and operation["step_id"] == context.step_id
            and operation["action_id"] == context.action_id
            and operation["attempt_id"] == context.attempt_id
            and int(operation["attempt_number"]) == context.attempt_number
            and operation["step_lease_id"] == context.lease_id
            and operation["worker_id"] == context.worker_id
            and operation["tool_name"] == "machine_terminate_process"
            and operation["args_sha256"] == args_sha256
            and operation["instance_id"] == instance_id
            and operation["executor_binding_json"] == binding_json
            and operation["executor_binding_sha256"] == binding_sha256
            and operation["target_boundary_sha256"]
                == target_boundary_sha256
            and bool(operation["force"]) is force)

    def _prepare_termination_operation(
        self,
        row: Mapping[str, Any],
        observed: BackendObservation,
        binding: ProcessInstanceBinding,
        context: ProcessOperationContext,
        *,
        force: bool,
    ) -> dict[str, Any]:
        binding_json = canonical_json(binding.model_dump(mode="json"))
        binding_sha256 = sha256_text(binding_json)
        args_sha256 = self._exact_termination_args_sha256(
            str(row["instance_id"]))
        target_sha256 = self._target_boundary_sha256_from_observation(observed)
        if (target_sha256 != self._target_boundary_sha256_from_row(row)
                or not self._row_matches_observation(
                    row, observed,
                    self._identity_token(
                        str(row["instance_id"]), str(row["spec_fingerprint"]),
                        str(row["args_sha256"])),
                    allow_terminal=False)):
            raise ProcessIdentityError()
        with self.graph.transaction() as conn:
            current = conn.execute(
                "SELECT * FROM process_instances WHERE instance_id=?",
                (row["instance_id"],),
            ).fetchone()
            if current is None:
                raise ValueError("process instance does not exist")
            self._verify_expected_binding(current, "terminate", binding)
            if (self._target_boundary_sha256_from_row(current) != target_sha256
                    or str(current["state"]) in _TERMINAL_INSTANCE_STATES):
                raise ProcessBindingError()
            collisions = conn.execute(
                """SELECT * FROM process_operations
                   WHERE idempotency_key=? OR step_id=? OR action_id=?
                      OR attempt_id=?""",
                (context.idempotency_key, context.step_id, context.action_id,
                 context.attempt_id),
            ).fetchall()
            if collisions:
                if len(collisions) != 1 or not self._operation_matches_context(
                        collisions[0], context, instance_id=str(row["instance_id"]),
                        binding_json=binding_json,
                        binding_sha256=binding_sha256,
                        args_sha256=args_sha256,
                        target_boundary_sha256=target_sha256,
                        force=force):
                    raise ProcessBrokerError("process_operation_idempotency_conflict")
                return dict(collisions[0])
            self._validate_exact_operation_claim(
                conn, context, instance_id=str(row["instance_id"]),
                binding_json=binding_json, binding_sha256=binding_sha256,
                args_sha256=args_sha256)
            operation_id = f"process_operation_{uuid.uuid4().hex}"
            operation = {
                "operation_id": operation_id,
                "idempotency_key": context.idempotency_key,
                "task_id": context.task_id,
                "step_id": context.step_id,
                "action_id": context.action_id,
                "attempt_id": context.attempt_id,
                "attempt_number": context.attempt_number,
                "step_lease_id": context.lease_id,
                "worker_id": context.worker_id,
                "tool_name": "machine_terminate_process",
                "args_sha256": args_sha256,
                "instance_id": str(row["instance_id"]),
                "executor_binding_sha256": binding_sha256,
                "executor_binding_json": binding_json,
                "target_boundary_sha256": target_sha256,
                "force": int(force),
            }
            body = self._operation_event_body(operation, "prepared")
            _, seq = self.graph.append_event(
                conn, "process_operation.prepared", body,
                actor="process_broker", task_id=context.task_id)
            now = utc_now()
            conn.execute(
                """INSERT INTO process_operations
                   (operation_id,idempotency_key,task_id,step_id,action_id,
                    attempt_id,attempt_number,step_lease_id,worker_id,tool_name,
                    args_sha256,instance_id,executor_binding_sha256,
                    executor_binding_json,target_boundary_sha256,force,status,
                    prepared_event_seq,created_at,updated_at,last_event_seq)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                           'prepared',?,?,?,?)""",
                (operation_id, context.idempotency_key, context.task_id,
                 context.step_id, context.action_id, context.attempt_id,
                 context.attempt_number, context.lease_id, context.worker_id,
                 "machine_terminate_process", args_sha256,
                 str(row["instance_id"]), binding_sha256, binding_json,
                 target_sha256, int(force), seq, now, now, seq),
            )
            created = conn.execute(
                "SELECT * FROM process_operations WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if created is None:
                raise RuntimeError("prepared process operation disappeared")
            return dict(created)

    def _mark_termination_dispatching(
        self,
        operation: Mapping[str, Any],
        binding: ProcessInstanceBinding,
        context: ProcessOperationContext,
    ) -> dict[str, Any]:
        binding_json = canonical_json(binding.model_dump(mode="json"))
        binding_sha256 = sha256_text(binding_json)
        args_sha256 = self._exact_termination_args_sha256(
            str(operation["instance_id"]))
        with self.graph.transaction() as conn:
            current = conn.execute(
                "SELECT * FROM process_operations WHERE operation_id=?",
                (operation["operation_id"],),
            ).fetchone()
            row = conn.execute(
                "SELECT * FROM process_instances WHERE instance_id=?",
                (operation["instance_id"],),
            ).fetchone()
            if current is None or row is None:
                raise ProcessBindingError()
            if current["status"] != "prepared":
                return dict(current)
            self._verify_expected_binding(row, "terminate", binding)
            target_sha256 = self._target_boundary_sha256_from_row(row)
            if not self._operation_matches_context(
                    current, context, instance_id=str(row["instance_id"]),
                    binding_json=binding_json,
                    binding_sha256=binding_sha256, args_sha256=args_sha256,
                    target_boundary_sha256=target_sha256,
                    force=bool(current["force"])):
                raise ProcessBindingError()
            self._validate_exact_operation_claim(
                conn, context, instance_id=str(row["instance_id"]),
                binding_json=binding_json, binding_sha256=binding_sha256,
                args_sha256=args_sha256)
            body = self._operation_event_body(current, "dispatching")
            _, seq = self.graph.append_event(
                conn, "process_operation.dispatching", body,
                actor="process_broker", task_id=context.task_id)
            changed = conn.execute(
                """UPDATE process_operations
                      SET status='dispatching',dispatch_event_seq=?,updated_at=?,
                          last_event_seq=?
                    WHERE operation_id=? AND status='prepared'""",
                (seq, utc_now(), seq, operation["operation_id"]),
            ).rowcount
            if changed != 1:
                raise ProcessBackendError("process_operation_dispatch_fence_lost")
            return dict(conn.execute(
                "SELECT * FROM process_operations WHERE operation_id=?",
                (operation["operation_id"],),
            ).fetchone())

    def _record_termination_outcome(
        self,
        operation_id: str,
        status: Literal[
            "effect_acknowledged", "known_failed", "outcome_unknown"],
        *,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        if status == "effect_acknowledged":
            error_code = None
        else:
            error_code = _stable_code(
                error_code, "process_termination_outcome_unknown")
        with self.graph.transaction() as conn:
            operation = conn.execute(
                "SELECT * FROM process_operations WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise ProcessBackendError("process_operation_missing")
            if operation["status"] == status:
                return dict(operation)
            if operation["status"] != "dispatching":
                raise ProcessBackendError("process_operation_outcome_fence_lost")
            body = self._operation_event_body(
                operation, status, error_code=error_code)
            _, seq = self.graph.append_event(
                conn, f"process_operation.{status}", body,
                actor="process_broker", task_id=operation["task_id"])
            changed = conn.execute(
                """UPDATE process_operations
                      SET status=?,outcome_event_seq=?,error_code=?,updated_at=?,
                          last_event_seq=?
                    WHERE operation_id=? AND status='dispatching'""",
                (status, seq, error_code, utc_now(), seq, operation_id),
            ).rowcount
            if changed != 1:
                raise ProcessBackendError("process_operation_outcome_fence_lost")
            return dict(conn.execute(
                "SELECT * FROM process_operations WHERE operation_id=?",
                (operation_id,),
            ).fetchone())

    def _acknowledged_termination_for_row(
        self, row: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        target_sha256 = self._target_boundary_sha256_from_row(row)
        with self.graph._connect() as conn:
            matches = conn.execute(
                """SELECT * FROM process_operations
                   WHERE instance_id=?
                     AND status IN ('effect_acknowledged','completed')""",
                (row["instance_id"],),
            ).fetchall()
        if len(matches) > 1:
            raise ProcessBackendError("process_operation_projection_invalid")
        if not matches:
            return None
        operation = dict(matches[0])
        if operation["target_boundary_sha256"] != target_sha256:
            raise ProcessIdentityError()
        return operation

    def _complete_acknowledged_termination_in_transaction(
        self, conn: Any, row: Mapping[str, Any], *, state: str,
    ) -> None:
        if state != "terminated":
            return
        matches = conn.execute(
            """SELECT * FROM process_operations
               WHERE instance_id=? AND status='effect_acknowledged'""",
            (row["instance_id"],),
        ).fetchall()
        if not matches:
            return
        if len(matches) != 1:
            raise ProcessBackendError("process_operation_projection_invalid")
        operation = matches[0]
        if (operation["target_boundary_sha256"]
                != self._target_boundary_sha256_from_row(row)):
            raise ProcessIdentityError()
        body = self._operation_event_body(
            operation, "completed", postcondition_state="terminated")
        _, seq = self.graph.append_event(
            conn, "process_operation.completed", body,
            actor="process_broker", task_id=operation["task_id"])
        now = utc_now()
        changed = conn.execute(
            """UPDATE process_operations
                  SET status='completed',postcondition_event_seq=?,completed_at=?,
                      updated_at=?,last_event_seq=?
                WHERE operation_id=? AND status='effect_acknowledged'""",
            (seq, now, now, seq, operation["operation_id"]),
        ).rowcount
        if changed != 1:
            raise ProcessBackendError("process_operation_completion_fence_lost")

    def approval_preview(
        self,
        spec_id: str,
        parameter_values: Mapping[str, Any],
        *,
        file_validator: FileParameterValidator | None = None,
    ) -> ProcessApprovalPreview:
        spec = self.registry.get(spec_id)
        self._require_durable_active_spec(spec)
        spec.verify_current()
        rendered = self.registry.render(
            spec_id, parameter_values, file_validator=file_validator)
        return ProcessApprovalPreview(
            spec_id=spec.spec_id, display_name=spec.display_name,
            version=spec.version, parameter_values=dict(rendered.values),
            resources=spec.resources, limits=spec.limits,
            sandboxed=spec.sandbox.enabled, network=spec.resources.network,
            session_access=spec.session_access,
            presentation=(spec.presentation.safe_display()
                          if spec.presentation is not None else None),
            instance_policy=spec.instance_policy,
            persistent=spec.persistent, args_sha256=rendered.args_sha256)

    def _require_durable_active_spec(self, spec: ProcessSpec) -> None:
        with self.graph._connect() as conn:
            row = conn.execute(
                "SELECT status,spec_sha256,sandbox_fingerprint "
                "FROM process_specs WHERE spec_id=?", (spec.spec_id,)
            ).fetchone()
        if (row is None or row["status"] != "active"
                or row["spec_sha256"] != spec.fingerprint
                or row["sandbox_fingerprint"] != spec.sandbox_fingerprint):
            raise PermissionError("process spec is not active")

    def _enforcement(self, spec: ProcessSpec) -> dict[str, Any]:
        describe = getattr(self.backend, "enforcement", None)
        if not callable(describe):
            raise ProcessBackendError(
                "backend_enforcement_unavailable", outcome_unknown=False)
        try:
            value = describe(spec)
            encoded = canonical_json(value)
            normalized = json.loads(encoded)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProcessBackendError(
                "backend_enforcement_invalid", outcome_unknown=False) from exc
        if not isinstance(normalized, dict) or len(encoded) > 16_384:
            raise ProcessBackendError(
                "backend_enforcement_invalid", outcome_unknown=False)
        return normalized

    @staticmethod
    def _decision_lease_id(decision: Any) -> str:
        admitted = bool(getattr(decision, "admitted", False))
        lease_id = getattr(decision, "lease_id", None)
        if not admitted or not isinstance(lease_id, str):
            raise ProcessAdmissionError()
        _validate_opaque_identifier("workload lease ID", lease_id)
        return lease_id

    @staticmethod
    def _workload_row(conn: Any, instance_id: str) -> Any | None:
        return conn.execute(
            """SELECT * FROM workload_resource_leases
               WHERE instance_id=? ORDER BY acquired_at DESC LIMIT 1""",
            (instance_id,)).fetchone()

    def _prepare(
        self,
        spec: ProcessSpec,
        rendered: RenderedProcess,
        *,
        launch_idempotency_key: str,
        task_id: str | None,
        step_id: str | None,
        action_id: str | None,
        source_step_lease_id: str,
        source_attempt_id: str,
        source_worker_id: str,
    ) -> tuple[Any, bool, str | None]:
        now = utc_now()
        with self.graph.transaction() as conn:
            existing = conn.execute(
                """SELECT * FROM process_instances
                   WHERE launch_idempotency_key=?""",
                (launch_idempotency_key,)).fetchone()
            if existing is not None:
                if (existing["spec_id"] != spec.spec_id
                        or existing["spec_fingerprint"] != spec.fingerprint
                        or existing["sandbox_fingerprint"] !=
                        spec.sandbox_fingerprint
                        or existing["args_sha256"] != rendered.args_sha256):
                    raise ProcessBrokerError("process_idempotency_conflict")
                workload = self._workload_row(conn, str(existing["instance_id"]))
                return existing, False, (
                    str(workload["lease_id"]) if workload is not None else None)

            durable_spec = conn.execute(
                "SELECT * FROM process_specs WHERE spec_id=?", (spec.spec_id,)
            ).fetchone()
            if (durable_spec is None or durable_spec["status"] != "active"
                    or durable_spec["spec_sha256"] != spec.fingerprint
                    or durable_spec["sandbox_fingerprint"] !=
                    spec.sandbox_fingerprint):
                raise PermissionError("process spec is not active at dispatch")

            if spec.instance_policy == "singleton":
                placeholders = ",".join("?" for _ in _ACTIVE_INSTANCE_STATES)
                active = conn.execute(
                    f"SELECT instance_id FROM process_instances "
                    f"WHERE spec_id=? AND state IN ({placeholders}) LIMIT 1",
                    (spec.spec_id, *sorted(_ACTIVE_INSTANCE_STATES)),
                ).fetchone()
                if active is not None:
                    raise ProcessBackendError(
                        "process_singleton_active", outcome_unknown=False)

            instance_id = f"process_{uuid.uuid4().hex}"
            unit_name = self._unit_name(instance_id)
            redacted = _redacted_values(rendered.values)
            body = {
                "instance_id": instance_id, "spec_id": spec.spec_id,
                "task_id": task_id, "step_id": step_id,
                "state": "prepared", "args": redacted,
                "args_sha256": rendered.args_sha256,
                "spec_fingerprint": spec.fingerprint,
                "sandbox_fingerprint": spec.sandbox_fingerprint,
                "persistent": spec.persistent,
            }
            event_id, seq = self.graph.append_event(
                conn, "process.prepared", body, actor="process_broker",
                task_id=task_id,
                idempotency_key=f"process-prepare:{launch_idempotency_key}")
            self.graph.append_node(
                conn, "process_instance", body, event_id=event_id,
                node_id=instance_id)
            self.graph.append_edge(
                conn, spec.spec_id, "instantiates", instance_id,
                event_id=event_id)
            if task_id:
                self.graph.append_edge(
                    conn, task_id, "owns_process", instance_id,
                    event_id=event_id)
            encrypted_args = self._cipher.seal(
                {"parameter_values": rendered.values,
                 "path_authorizations": rendered.authorizations},
                context=self._args_context(instance_id, spec,
                                           rendered.args_sha256))
            conn.execute(
                """INSERT INTO process_instances
                   (instance_id,spec_id,task_id,step_id,action_id,
                    launch_idempotency_key,args_ciphertext,args_redacted_json,
                    args_sha256,spec_fingerprint,sandbox_fingerprint,state,
                    unit_name,persistent,stdout_bytes,stdout_truncated,
                    stderr_bytes,stderr_truncated,prepared_at,created_at,updated_at,
                    last_event_seq)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,'prepared',?,?,0,0,0,0,?,?,?,?)""",
                (instance_id, spec.spec_id, task_id, step_id, action_id,
                 launch_idempotency_key, encrypted_args, canonical_json(redacted),
                 rendered.args_sha256, spec.fingerprint,
                 spec.sandbox_fingerprint, unit_name, int(spec.persistent), now,
                 now, now, seq))
            decision = self.admission.transfer_step_to_workload_in_transaction(
                conn, spec.resources.as_claim(), instance_id,
                source_step_lease_id, source_attempt_id, source_worker_id,
                canonical_json(self._enforcement(spec)))
            workload_lease_id = self._decision_lease_id(decision)
            row = conn.execute(
                "SELECT * FROM process_instances WHERE instance_id=?",
                (instance_id,)).fetchone()
            if row is None:
                raise RuntimeError("prepared process intent disappeared")
            return row, True, workload_lease_id

    def _transition(
        self,
        instance_id: str,
        state: str,
        *,
        observation: BackendObservation | None = None,
        result_code: str | None = None,
        finished: bool = False,
        release_reason: str | None = None,
        reconcile_reason: str | None = None,
    ) -> Any:
        if state not in _INSTANCE_STATES:
            raise ValueError("invalid process state")
        now = utc_now()
        with self.graph.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM process_instances WHERE instance_id=?",
                (instance_id,)).fetchone()
            if row is None:
                raise ValueError("process instance does not exist")
            body = {
                "instance_id": instance_id, "spec_id": row["spec_id"],
                "from": row["state"], "to": state,
                "result_code": _stable_code(
                    result_code or (observation.result_code
                                    if observation else None), "unknown"),
            }
            _, seq = self.graph.append_event(
                conn, f"process.{state}", body, actor="process_broker",
                task_id=row["task_id"])
            if state in {"terminated", "exited", "launch_failed"}:
                finished = True
            started_at = (now if observation is not None
                          and observation.state in {"running", "stopping"}
                          and row["started_at"] is None else row["started_at"])
            finished_at = now if finished else row["finished_at"]
            stop_requested_at = (now if state == "stop_requested"
                                 and row["stop_requested_at"] is None
                                 else row["stop_requested_at"])
            complete_observation_identity = bool(
                observation is not None
                and observation.leader_pid is not None
                and observation.start_ticks is not None
                and observation.exe_device is not None
                and observation.exe_inode is not None
                and observation.exe_sha256 is not None)
            if complete_observation_identity:
                boot_id = observation.boot_id
                leader_pid = observation.leader_pid
                start_ticks = observation.start_ticks
                exe_device = observation.exe_device
                exe_inode = observation.exe_inode
                exe_sha256 = observation.exe_sha256
            else:
                # A terminal observation commonly reports MainPID=0. Preserve the
                # last complete live identity; if none was ever captured, keep the
                # boot/PID/start and executable tuples wholly NULL for v8 fencing.
                boot_id = row["boot_id"]
                leader_pid = row["leader_pid"]
                start_ticks = row["start_ticks"]
                exe_device = row["exe_device"]
                exe_inode = row["exe_inode"]
                exe_sha256 = row["exe_sha256"]
            conn.execute(
                """UPDATE process_instances SET
                       state=?,boot_id=?,invocation_id=?,control_group=?,
                       leader_pid=?,start_ticks=?,exe_device=?,exe_inode=?,
                       exe_sha256=?,exit_code=?,exit_signal=?,result_code=?,
                       started_at=?,heartbeat_at=?,stop_requested_at=?,finished_at=?,
                       updated_at=?,last_event_seq=?
                   WHERE instance_id=?""",
                (state,
                 boot_id,
                 observation.invocation_id if observation else row["invocation_id"],
                 observation.control_group if observation else row["control_group"],
                 leader_pid, start_ticks, exe_device, exe_inode, exe_sha256,
                 observation.exit_code if observation else row["exit_code"],
                 observation.exit_signal if observation else row["exit_signal"],
                 body["result_code"], started_at, now, stop_requested_at,
                 finished_at, now, seq, instance_id))
            workload = self._workload_row(conn, instance_id)
            if state in {"running", "stopping", "stop_requested"} and (
                    workload is None or workload["status"] not in {
                        "active", "reconciling"}):
                raise ProcessAdmissionError(
                    "process_workload_reservation_missing")
            if workload is not None:
                lease_id = str(workload["lease_id"])
                if release_reason is not None:
                    recorded_runtime = str(workload["runtime_id"])
                    current_runtime = str(getattr(
                        self.admission, "runtime_id", ""))
                    previous_runtime_id = (
                        recorded_runtime
                        if workload["status"] == "reconciling"
                        and recorded_runtime != current_runtime else None)
                    released = self.admission.release_workload_in_transaction(
                        conn, lease_id, instance_id, cgroup_empty=True,
                        previous_runtime_id=previous_runtime_id,
                        reason=release_reason)
                    if not released and workload["status"] not in {
                            "released", "fenced"}:
                        raise ProcessAdmissionError(
                            "process_workload_release_failed")
                elif reconcile_reason is not None:
                    changed = self.admission.mark_workload_reconciling_in_transaction(
                        conn, lease_id, instance_id, reason=reconcile_reason)
                    if not changed and workload["status"] not in {
                            "reconciling", "released", "fenced"}:
                        raise ProcessAdmissionError(
                            "process_workload_reconcile_failed")
                elif state in {"running", "stopping", "stop_requested"}:
                    if workload["status"] == "reconciling":
                        adopted = self.admission.adopt_workload_in_transaction(
                            conn, lease_id, instance_id,
                            previous_runtime_id=str(workload["runtime_id"]))
                        if not adopted:
                            raise ProcessAdmissionError(
                                "process_workload_adoption_failed")
                    alive = self.admission.heartbeat_workload_in_transaction(
                        conn, lease_id, instance_id)
                    if not alive and workload["status"] == "active":
                        raise ProcessAdmissionError(
                            "process_workload_heartbeat_failed")
            projected = conn.execute(
                "SELECT * FROM process_instances WHERE instance_id=?",
                (instance_id,)).fetchone()
            if projected is None:
                raise RuntimeError("process projection disappeared")
            self._complete_acknowledged_termination_in_transaction(
                conn, projected, state=state)
            return projected

    @staticmethod
    def _row_matches_observation(row: Any, observation: BackendObservation,
                                 expected_token: str, *,
                                 allow_terminal: bool = True) -> bool:
        if (observation.unit_name != row["unit_name"]
                or observation.identity_token != expected_token):
            return False
        for column, value in (
            ("boot_id", observation.boot_id),
            ("invocation_id", observation.invocation_id),
            ("control_group", observation.control_group),
        ):
            if row[column] is not None and str(row[column]) != str(value):
                return False
        stored_leader = row["leader_pid"]
        if stored_leader is None:
            return True
        if observation.leader_pid is None:
            return bool(allow_terminal and observation.cgroup_empty
                        and observation.state == "exited")
        return all((
            int(stored_leader) == observation.leader_pid,
            int(row["start_ticks"]) == observation.start_ticks,
            int(row["exe_device"]) == observation.exe_device,
            int(row["exe_inode"]) == observation.exe_inode,
            str(row["exe_sha256"]) == observation.exe_sha256,
        ))

    def _request(self, row: Any, spec: ProcessSpec,
                 rendered: RenderedProcess) -> BackendLaunchRequest:
        return BackendLaunchRequest(
            instance_id=str(row["instance_id"]), unit_name=str(row["unit_name"]),
            identity_token=self._identity_token(
                str(row["instance_id"]), spec.fingerprint,
                rendered.args_sha256),
            argv=rendered.argv, cwd=spec.cwd,
            environment=rendered.environment, limits=spec.limits,
            executable_identity=spec.executable_identity,
            sandbox=spec.sandbox, session_access=spec.session_access)

    def _settle_observation(self, row: Any, observation: BackendObservation,
                            *, expected_token: str, spec: ProcessSpec) -> Any:
        first_live_identity_matches = True
        if row["leader_pid"] is None and observation.state in {
                "running", "stopping"}:
            expected_executable = spec.executable_identity
            first_live_identity_matches = bool(
                expected_executable is not None
                and observation.exe_device == expected_executable.device
                and observation.exe_inode == expected_executable.inode
                and observation.exe_sha256 == expected_executable.sha256)
        if (not first_live_identity_matches
                or not self._row_matches_observation(
                    row, observation, expected_token)):
            return self._transition(
                str(row["instance_id"]), "identity_mismatch",
                result_code="identity_mismatch",
                reconcile_reason="identity_mismatch")
        if observation.cgroup_empty and observation.state == "exited":
            # A stop-request projection is only intent.  Causal termination
            # requires the exact backend effect acknowledgement for this runtime
            # boundary; otherwise this is an ordinary natural exit.
            requested_stop = self._acknowledged_termination_for_row(row) is not None
            return self._transition(
                str(row["instance_id"]),
                "terminated" if requested_stop else "exited",
                observation=observation, finished=True,
                release_reason=("terminated" if requested_stop else "exited"))
        if observation.state == "running":
            return self._transition(
                str(row["instance_id"]), "running", observation=observation)
        if observation.state == "stopping":
            return self._transition(
                str(row["instance_id"]), "stopping", observation=observation)
        return self._transition(
            str(row["instance_id"]), "reconcile_required",
            observation=observation, result_code="backend_state_uncertain",
            reconcile_reason="backend_state_uncertain")

    def launch(
        self,
        spec_id: str,
        parameter_values: Mapping[str, Any],
        *,
        launch_idempotency_key: str,
        source_step_lease_id: str,
        source_attempt_id: str,
        source_worker_id: str,
        task_id: str | None = None,
        step_id: str | None = None,
        action_id: str | None = None,
        file_validator: FileParameterValidator | None = None,
    ) -> dict[str, Any]:
        _validate_opaque_identifier(
            "launch idempotency key", launch_idempotency_key)
        _validate_opaque_identifier("source step lease ID", source_step_lease_id)
        _validate_opaque_identifier("source attempt ID", source_attempt_id)
        _validate_opaque_identifier("source worker ID", source_worker_id)
        for label, value in (("task ID", task_id), ("step ID", step_id),
                             ("action ID", action_id)):
            if value is not None:
                _validate_opaque_identifier(label, value)
        spec = self.registry.get(spec_id)
        rendered = self.registry.render(
            spec_id, parameter_values, file_validator=file_validator)
        # This is intentionally immediately before the durable intent/launch
        # sequence; a changed inode/hash can never reuse an approved spec.
        spec.verify_current()
        persistence = getattr(self.backend, "supports_persistence", None)
        if spec.persistent and (not callable(persistence) or not persistence()):
            raise ProcessBackendError(
                "persistent_session_unavailable", outcome_unknown=False)
        with self._operation_lock(launch_idempotency_key):
            row, created, workload_lease_id = self._prepare(
                spec, rendered,
                launch_idempotency_key=launch_idempotency_key,
                task_id=task_id, step_id=step_id, action_id=action_id,
                source_step_lease_id=source_step_lease_id,
                source_attempt_id=source_attempt_id,
                source_worker_id=source_worker_id)
            request = self._request(row, spec, rendered)
            if not created:
                if str(row["state"]) in _TERMINAL_INSTANCE_STATES:
                    return self._public_receipt(
                        row, workload_lease_id=workload_lease_id,
                        idempotent_replay=True)
                try:
                    try:
                        observed = self.backend.inspect(str(row["unit_name"]))
                    except ProcessIdentityError:
                        observed = None
                        row = self._transition(
                            str(row["instance_id"]), "identity_mismatch",
                            result_code="identity_mismatch",
                            reconcile_reason="identity_mismatch")
                    except ProcessBackendError:
                        observed = None
                    if observed is None:
                        if str(row["state"]) != "identity_mismatch":
                            row = self._transition(
                                str(row["instance_id"]), "reconcile_required",
                                result_code="backend_identity_unavailable",
                                reconcile_reason="backend_identity_unavailable")
                        return self._public_receipt(
                            row, workload_lease_id=workload_lease_id,
                            idempotent_replay=True)
                    row = self._settle_observation(
                        row, observed, expected_token=request.identity_token,
                        spec=spec)
                    return self._public_receipt(
                        row, workload_lease_id=workload_lease_id,
                        idempotent_replay=True)
                except Exception as exc:
                    # A replay names a launch whose effect boundary was crossed
                    # by the original attempt.  Projection failure must not
                    # rewrite that durable action as a definite failure.
                    if (isinstance(exc, ProcessBackendError)
                            and exc.outcome_unknown
                            and exc.code == "process_launch_outcome_unknown"):
                        raise
                    raise ProcessBackendError(
                        "process_launch_outcome_unknown",
                        outcome_unknown=True) from exc

            row = self._transition(
                str(row["instance_id"]), "starting",
                result_code="launch_requested")
            known_launch_failure: ProcessBackendError | None = None
            try:
                try:
                    observed = self.backend.launch(request)
                except ProcessIdentityError:
                    row = self._transition(
                        str(row["instance_id"]), "identity_mismatch",
                        result_code="identity_mismatch",
                        reconcile_reason="identity_mismatch")
                except ProcessBackendError as exc:
                    if not exc.outcome_unknown:
                        known_launch_failure = exc
                    try:
                        observed = self.backend.inspect(str(row["unit_name"]))
                    except (ProcessBackendError, ProcessIdentityError):
                        observed = None
                    if observed is not None:
                        # Exact positive observation disproves a backend claim
                        # that no launch effect occurred.  Any later projection
                        # failure is therefore genuine outcome ambiguity.
                        known_launch_failure = None
                        row = self._settle_observation(
                            row, observed, expected_token=request.identity_token,
                            spec=spec)
                    elif exc.outcome_unknown:
                        row = self._transition(
                            str(row["instance_id"]), "reconcile_required",
                            result_code=exc.code,
                            reconcile_reason="launch_outcome_unknown")
                    else:
                        row = self._transition(
                            str(row["instance_id"]), "launch_failed",
                            result_code=exc.code, finished=True,
                            release_reason="launch_failed")
                else:
                    row = self._settle_observation(
                        row, observed, expected_token=request.identity_token,
                        spec=spec)
                return self._public_receipt(
                    row, workload_lease_id=workload_lease_id,
                    idempotent_replay=False)
            except Exception as exc:
                # Once backend.launch has been invoked, a projection/admission
                # failure cannot safely turn the external action into a normal
                # failed receipt.  Leave the prepared/starting row as a durable
                # reconciliation anchor and quarantine the exact step.
                if known_launch_failure is not None:
                    raise known_launch_failure from exc
                if (isinstance(exc, ProcessBackendError)
                        and exc.outcome_unknown
                        and exc.code == "process_launch_outcome_unknown"):
                    raise
                raise ProcessBackendError(
                    "process_launch_outcome_unknown",
                    outcome_unknown=True) from exc

    def _load_row(self, instance_id: str) -> tuple[Any, str | None]:
        _validate_opaque_identifier("process instance ID", instance_id)
        with self.graph._connect() as conn:
            row = conn.execute(
                "SELECT * FROM process_instances WHERE instance_id=?",
                (instance_id,)).fetchone()
            workload = self._workload_row(conn, instance_id) if row else None
        if row is None:
            raise ValueError("process instance does not exist")
        return row, (str(workload["lease_id"]) if workload is not None else None)

    def _runtime_observation_locked(
        self, instance_id: str,
    ) -> BackendObservation:
        row, _ = self._load_row(instance_id)
        if str(row["state"]) in _TERMINAL_INSTANCE_STATES:
            raise ProcessBindingError()
        expected_token = self._identity_token(
            instance_id, str(row["spec_fingerprint"]),
            str(row["args_sha256"]))
        observation = self.backend.inspect(str(row["unit_name"]))
        if (observation is None
                or observation.state not in {"running", "stopping"}
                or observation.leader_pid is None
                or observation.cgroup_empty
                or not self._row_matches_observation(
                    row, observation, expected_token,
                    allow_terminal=False)):
            raise ProcessIdentityError()
        spec = self.registry.get(str(row["spec_id"]))
        spec.verify_current()
        expected = spec.executable_identity
        if (expected is None
                or observation.exe_device != expected.device
                or observation.exe_inode != expected.inode
                or observation.exe_sha256 != expected.sha256):
            raise ProcessIdentityError()
        return observation

    def runtime_observation(self, instance_id: str) -> BackendObservation:
        """Return a private exact live boundary for a compound verifier.

        This never projects raw process identity into a public receipt. It is
        intentionally read-only so a desktop verifier cannot turn absence or
        ambiguity into a process lifecycle transition.
        """
        with self._operation_lock(instance_id):
            return self._runtime_observation_locked(instance_id)

    def singleton_runtime_observation(
        self, spec_id: str,
    ) -> tuple[str, BackendObservation] | None:
        """Resolve the sole live runtime for an explicitly singleton spec.

        This is a private prerequisite hook for brokers such as the managed
        browser. It never launches, adopts, transitions, or returns a public
        receipt, and it refuses contradictory duplicate active projections.
        """
        spec = self.registry.get(spec_id)
        if spec.instance_policy != "singleton":
            raise ProcessBindingError()
        self._require_durable_active_spec(spec)
        placeholders = ",".join("?" for _ in _ACTIVE_INSTANCE_STATES)
        with self.graph._connect() as conn:
            rows = conn.execute(
                f"SELECT instance_id FROM process_instances "
                f"WHERE spec_id=? AND state IN ({placeholders}) "
                f"ORDER BY created_at,instance_id LIMIT 2",
                (spec_id, *sorted(_ACTIVE_INSTANCE_STATES)),
            ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise ProcessIdentityError()
        instance_id = str(rows[0]["instance_id"])
        with self._operation_lock(instance_id):
            return instance_id, self._runtime_observation_locked(instance_id)

    def singleton_loopback_listener_matches(
        self, spec_id: str, port: int,
    ) -> bool:
        """Prove a singleton's exact managed execution owns one listener."""
        runtime = self.singleton_runtime_observation(spec_id)
        if runtime is None:
            return False
        instance_id, expected = runtime
        with self._operation_lock(instance_id):
            current = self._runtime_observation_locked(instance_id)
            if not expected.same_live_execution(current):
                raise ProcessIdentityError()
            verify = getattr(self.backend, "owns_loopback_listener", None)
            if not callable(verify):
                raise ProcessBackendError(
                    "loopback_listener_verifier_unavailable",
                    outcome_unknown=False)
            return bool(verify(current, port))

    def runtime_process_member_matches(
        self,
        instance_id: str,
        expected_execution: BackendObservation,
        *,
        pid: int,
        start_ticks: int,
        executable_identity: ExecutableIdentity,
    ) -> bool:
        """Prove a compositor PID is in the exact managed live cgroup.

        The backend fences membership on both sides of the identity read. The
        broker then binds that private observation to the durable instance and
        the compositor's PID/start/executable tuple. Absence is ordinary false;
        disagreement in the managed execution fails closed as an identity
        error. No raw identity enters a receipt.
        """
        if not isinstance(expected_execution, BackendObservation):
            raise ProcessBindingError()
        if not isinstance(executable_identity, ExecutableIdentity):
            executable_identity = ExecutableIdentity.model_validate(
                executable_identity)
        with self._operation_lock(instance_id):
            current = self._runtime_observation_locked(instance_id)
            if not expected_execution.same_live_execution(current):
                raise ProcessIdentityError()
            observed = self.backend.member_identity(current, pid)
            if observed is None:
                return False
            (observed_pid, observed_start, observed_device,
             observed_inode, observed_hash) = observed
            return bool(
                observed_pid == pid
                and observed_start == start_ticks
                and observed_device == executable_identity.device
                and observed_inode == executable_identity.inode
                and observed_hash == executable_identity.sha256)

    def inspect(
        self,
        instance_id: str,
        *,
        expected_binding: ProcessInstanceBinding | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._operation_lock(instance_id):
            row, workload_lease_id = self._load_row(instance_id)
            self._verify_expected_binding(row, "inspect", expected_binding)
            if str(row["state"]) in _TERMINAL_INSTANCE_STATES:
                return self._public_receipt(
                    row, workload_lease_id=workload_lease_id)
            expected_token = self._identity_token(
                instance_id, str(row["spec_fingerprint"]),
                str(row["args_sha256"]))
            try:
                observed = self.backend.inspect(str(row["unit_name"]))
            except ProcessIdentityError:
                observed = None
                row = self._transition(
                    instance_id, "identity_mismatch",
                    result_code="identity_mismatch",
                    reconcile_reason="identity_mismatch")
            except ProcessBackendError:
                observed = None
            if observed is None and str(row["state"]) != "identity_mismatch":
                row = self._transition(
                    instance_id, "reconcile_required",
                    result_code="backend_identity_unavailable",
                    reconcile_reason="backend_identity_unavailable")
            elif observed is not None:
                spec = self.registry.get(str(row["spec_id"]))
                spec.verify_current()
                row = self._settle_observation(
                    row, observed, expected_token=expected_token, spec=spec)
            return self._public_receipt(
                row, workload_lease_id=workload_lease_id)

    def terminate(
        self,
        instance_id: str,
        *,
        force: bool = False,
        expected_binding: ProcessInstanceBinding | Mapping[str, Any] | None = None,
        operation_context: ProcessOperationContext | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(force, bool):
            raise ValueError("force must be boolean")
        if expected_binding is None and operation_context is None:
            return self._terminate_untracked(instance_id, force=force)
        if expected_binding is None or operation_context is None or force:
            # The public v1 task action approves ordinary TERM only.  A force
            # escalation must be a separately modelled and approved action.
            raise ProcessBindingError()
        try:
            binding = ProcessInstanceBinding.model_validate(expected_binding)
        except (TypeError, ValueError) as exc:
            raise ProcessBindingError() from exc
        context = self._operation_context(operation_context)
        if binding.operation != "terminate" or binding.instance_id != instance_id:
            raise ProcessBindingError()

        with self._operation_lock(instance_id):
            row, workload_lease_id = self._load_row(instance_id)
            binding_json = canonical_json(binding.model_dump(mode="json"))
            binding_sha256 = sha256_text(binding_json)
            args_sha256 = self._exact_termination_args_sha256(instance_id)
            target_sha256 = self._target_boundary_sha256_from_row(row)
            with self.graph._connect() as conn:
                existing_rows = conn.execute(
                    """SELECT * FROM process_operations
                       WHERE idempotency_key=? OR step_id=? OR action_id=?
                          OR attempt_id=?""",
                    (context.idempotency_key, context.step_id,
                     context.action_id, context.attempt_id),
                ).fetchall()
            if existing_rows:
                if len(existing_rows) != 1 or not self._operation_matches_context(
                        existing_rows[0], context, instance_id=instance_id,
                        binding_json=binding_json,
                        binding_sha256=binding_sha256,
                        args_sha256=args_sha256,
                        target_boundary_sha256=target_sha256, force=False):
                    raise ProcessBrokerError(
                        "process_operation_idempotency_conflict")
                operation = dict(existing_rows[0])
                status = str(operation["status"])
                if status == "completed":
                    if str(row["state"]) != "terminated":
                        raise ProcessBackendError(
                            "process_operation_projection_invalid")
                    return self._public_receipt(
                        row, workload_lease_id=workload_lease_id,
                        idempotent_replay=True)
                if status == "known_failed":
                    raise ProcessBackendError(
                        _stable_code(operation["error_code"],
                                     "process_termination_failed"),
                        outcome_unknown=False)
                if status in {"dispatching", "outcome_unknown"}:
                    raise ProcessBackendError(
                        "process_termination_outcome_unknown",
                        outcome_unknown=True)
                if status == "effect_acknowledged":
                    return self._observe_acknowledged_termination(
                        row, operation, workload_lease_id=workload_lease_id)
                if status != "prepared":
                    raise ProcessBackendError(
                        "process_operation_projection_invalid")
                self._verify_expected_binding(row, "terminate", binding)
                try:
                    observed = self.backend.inspect(str(row["unit_name"]))
                except (ProcessBackendError, ProcessIdentityError):
                    observed = None
                if (observed is None
                        or not self._row_matches_observation(
                            row, observed,
                            self._identity_token(
                                instance_id, str(row["spec_fingerprint"]),
                                str(row["args_sha256"])),
                            allow_terminal=False)
                        or observed.leader_pid is None or observed.cgroup_empty
                        or self._target_boundary_sha256_from_observation(observed)
                            != operation["target_boundary_sha256"]):
                    raise ProcessIdentityError()
            else:
                self._verify_expected_binding(row, "terminate", binding)
                if str(row["state"]) in _TERMINAL_INSTANCE_STATES:
                    raise ProcessBindingError()
                expected_token = self._identity_token(
                    instance_id, str(row["spec_fingerprint"]),
                    str(row["args_sha256"]))
                try:
                    observed = self.backend.inspect(str(row["unit_name"]))
                except (ProcessBackendError, ProcessIdentityError):
                    observed = None
                if (observed is None
                        or not self._row_matches_observation(
                            row, observed, expected_token, allow_terminal=False)
                        or observed.leader_pid is None or observed.cgroup_empty):
                    self._transition(
                        instance_id, "identity_mismatch",
                        result_code="identity_mismatch",
                        reconcile_reason="identity_mismatch")
                    raise ProcessIdentityError()
                operation = self._prepare_termination_operation(
                    row, observed, binding, context, force=False)

            operation = self._mark_termination_dispatching(
                operation, binding, context)
            if operation["status"] != "dispatching":
                raise ProcessBackendError(
                    "process_operation_dispatch_fence_lost")
            try:
                row = self._transition(
                    instance_id, "stop_requested", observation=observed,
                    result_code="term_requested")
            except Exception as exc:
                try:
                    self._record_termination_outcome(
                        str(operation["operation_id"]), "known_failed",
                        error_code="process_termination_projection_failed")
                except Exception as journal_exc:
                    raise ProcessBackendError(
                        "process_termination_outcome_unknown",
                        outcome_unknown=True) from journal_exc
                raise ProcessBackendError(
                    "process_termination_projection_failed",
                    outcome_unknown=False) from exc

            expected_token = self._identity_token(
                instance_id, str(row["spec_fingerprint"]),
                str(row["args_sha256"]))
            # The backend repeats the full identity check immediately before the
            # signal.  The committed dispatching row is the crash boundary: an
            # exact replay will never enter the backend a second time.
            try:
                result = self.backend.terminate(observed, force=False)
            except ProcessBackendError as exc:
                outcome = ("outcome_unknown" if exc.outcome_unknown
                           else "known_failed")
                try:
                    self._record_termination_outcome(
                        str(operation["operation_id"]), outcome,
                        error_code=exc.code)
                except Exception as journal_exc:
                    raise ProcessBackendError(
                        "process_termination_outcome_unknown",
                        outcome_unknown=True) from journal_exc
                if exc.outcome_unknown:
                    raise ProcessBackendError(
                        "process_termination_outcome_unknown",
                        outcome_unknown=True) from exc
                raise
            except ProcessIdentityError as exc:
                try:
                    self._record_termination_outcome(
                        str(operation["operation_id"]), "outcome_unknown",
                        error_code="process_termination_identity_lost")
                except Exception as journal_exc:
                    raise ProcessBackendError(
                        "process_termination_outcome_unknown",
                        outcome_unknown=True) from journal_exc
                raise ProcessBackendError(
                    "process_termination_outcome_unknown",
                    outcome_unknown=True) from exc
            except Exception as exc:
                try:
                    self._record_termination_outcome(
                        str(operation["operation_id"]), "outcome_unknown",
                        error_code="process_termination_backend_invalid")
                except Exception as journal_exc:
                    raise ProcessBackendError(
                        "process_termination_outcome_unknown",
                        outcome_unknown=True) from journal_exc
                raise ProcessBackendError(
                    "process_termination_outcome_unknown",
                    outcome_unknown=True) from exc

            if (result.identity_token != expected_token
                    or (not observed.same_live_execution(result)
                        and not (result.cgroup_empty
                                 and result.boot_id == observed.boot_id
                                 and result.invocation_id == observed.invocation_id
                                 and result.control_group
                                     == observed.control_group))):
                if (result.identity_token != expected_token
                        or result.boot_id != observed.boot_id):
                    code = "process_termination_identity_lost"
                else:
                    code = "process_termination_boundary_changed"
                try:
                    self._record_termination_outcome(
                        str(operation["operation_id"]), "outcome_unknown",
                        error_code=code)
                except Exception as journal_exc:
                    raise ProcessBackendError(
                        "process_termination_outcome_unknown",
                        outcome_unknown=True) from journal_exc
                raise ProcessBackendError(
                    "process_termination_outcome_unknown",
                    outcome_unknown=True)

            try:
                self._record_termination_outcome(
                    str(operation["operation_id"]), "effect_acknowledged")
            except Exception as exc:
                raise ProcessBackendError(
                    "process_termination_outcome_unknown",
                    outcome_unknown=True) from exc
            try:
                if result.cgroup_empty:
                    row = self._transition(
                        instance_id, "terminated", observation=result,
                        result_code="terminated",
                        finished=True, release_reason="terminated")
                else:
                    row = self._transition(
                        instance_id, "stopping", observation=result,
                        result_code="term_pending")
                return self._public_receipt(
                    row, workload_lease_id=workload_lease_id,
                    forced=False)
            except Exception as exc:
                raise ProcessBackendError(
                    "process_termination_postcondition_unknown",
                    outcome_unknown=True) from exc

    def _observe_acknowledged_termination(
        self,
        row: Mapping[str, Any],
        operation: Mapping[str, Any],
        *,
        workload_lease_id: str | None,
    ) -> dict[str, Any]:
        if str(row["state"]) == "terminated":
            raise ProcessBackendError("process_operation_projection_invalid")
        try:
            observed = self.backend.inspect(str(row["unit_name"]))
        except (ProcessBackendError, ProcessIdentityError) as exc:
            raise ProcessBackendError(
                "process_termination_postcondition_unknown",
                outcome_unknown=True) from exc
        expected_token = self._identity_token(
            str(row["instance_id"]), str(row["spec_fingerprint"]),
            str(row["args_sha256"]))
        if (observed is None
                or not self._row_matches_observation(
                    row, observed, expected_token, allow_terminal=True)
                or operation["target_boundary_sha256"]
                    != self._target_boundary_sha256_from_row(row)):
            raise ProcessBackendError(
                "process_termination_postcondition_unknown",
                outcome_unknown=True)
        if observed.cgroup_empty and observed.state == "exited":
            projected = self._transition(
                str(row["instance_id"]), "terminated", observation=observed,
                result_code="terminated", finished=True,
                release_reason="terminated")
        elif observed.state in {"running", "stopping"}:
            projected = self._transition(
                str(row["instance_id"]), "stopping", observation=observed,
                result_code="term_pending")
        else:
            raise ProcessBackendError(
                "process_termination_postcondition_unknown",
                outcome_unknown=True)
        return self._public_receipt(
            projected, workload_lease_id=workload_lease_id,
            idempotent_replay=True, forced=False)

    def _terminate_untracked(
        self, instance_id: str, *, force: bool = False,
    ) -> dict[str, Any]:
        """Trusted administrative stop; never valid as a durable task receipt."""
        with self._operation_lock(instance_id):
            row, workload_lease_id = self._load_row(instance_id)
            if str(row["state"]) in _TERMINAL_INSTANCE_STATES:
                return self._public_receipt(
                    row, workload_lease_id=workload_lease_id,
                    idempotent_replay=True)
            expected_token = self._identity_token(
                instance_id, str(row["spec_fingerprint"]),
                str(row["args_sha256"]))
            try:
                observed = self.backend.inspect(str(row["unit_name"]))
            except (ProcessBackendError, ProcessIdentityError):
                observed = None
            if (observed is None
                    or not self._row_matches_observation(
                        row, observed, expected_token, allow_terminal=False)
                    or observed.leader_pid is None or observed.cgroup_empty):
                self._transition(
                    instance_id, "identity_mismatch",
                    result_code="identity_mismatch",
                    reconcile_reason="identity_mismatch")
                raise ProcessIdentityError()
            row = self._transition(
                instance_id, "stop_requested", observation=observed,
                result_code="force_requested" if force else "term_requested")
            try:
                result = self.backend.terminate(observed, force=force)
            except ProcessIdentityError as exc:
                raise ProcessBackendError(
                    "process_termination_outcome_unknown",
                    outcome_unknown=True) from exc
            if (result.identity_token != expected_token
                    or (not observed.same_live_execution(result)
                        and not (result.cgroup_empty
                                 and result.boot_id == observed.boot_id
                                 and result.invocation_id == observed.invocation_id
                                 and result.control_group
                                     == observed.control_group))):
                raise ProcessBackendError(
                    "process_termination_outcome_unknown",
                    outcome_unknown=True)
            if result.cgroup_empty:
                row = self._transition(
                    instance_id, "terminated", observation=result,
                    result_code="force_terminated" if force else "terminated",
                    finished=True, release_reason="terminated")
            else:
                row = self._transition(
                    instance_id, "stopping", observation=result,
                    result_code="force_pending" if force else "term_pending")
            return self._public_receipt(
                row, workload_lease_id=workload_lease_id, forced=force)

    @staticmethod
    def _cleanup_fence(row: Mapping[str, Any]) -> BackendTerminalFence:
        return BackendTerminalFence(
            unit_name=str(row["unit_name"]),
            identity_token=ProcessBroker._identity_token(
                str(row["instance_id"]), str(row["spec_fingerprint"]),
                str(row["args_sha256"])),
            boot_id=(str(row["boot_id"])
                     if row["boot_id"] is not None else None),
            invocation_id=(str(row["invocation_id"])
                           if row["invocation_id"] is not None else None),
            control_group=(str(row["control_group"])
                           if row["control_group"] is not None else None),
        )

    def _claim_terminal_cleanup(
        self, instance_id: str,
    ) -> tuple[dict[str, Any], str, int] | None:
        now_value = datetime.now(UTC)
        now = now_value.isoformat(
            timespec="microseconds").replace("+00:00", "Z")
        claim_expires_at = (now_value + timedelta(
            seconds=_CLEANUP_CLAIM_SECONDS)).isoformat(
                timespec="microseconds").replace("+00:00", "Z")
        with self.graph.transaction() as conn:
            row = conn.execute(
                """SELECT p.*,
                          c.state AS cleanup_state,
                          c.attempt_count AS cleanup_attempt_count,
                          c.next_attempt_at AS cleanup_next_attempt_at,
                          c.claim_expires_at AS cleanup_claim_expires_at,
                          w.status AS cleanup_workload_status,
                          w.released_at AS cleanup_workload_released_at
                     FROM process_unit_cleanups c
                     JOIN process_instances p USING(instance_id)
                LEFT JOIN workload_resource_leases w USING(instance_id)
                    WHERE c.instance_id=?""",
                (instance_id,),
            ).fetchone()
            if row is None:
                return None
            cleanup_state = str(row["cleanup_state"])
            due = bool(
                cleanup_state == "pending"
                and (row["cleanup_next_attempt_at"] is None
                     or str(row["cleanup_next_attempt_at"]) <= now)
                or cleanup_state == "cleaning"
                and str(row["cleanup_claim_expires_at"]) <= now)
            if not due:
                return None
            if str(row["state"]) not in _TERMINAL_INSTANCE_STATES:
                code = "process_unit_cleanup_projection_mismatch"
                _, seq = self.graph.append_event(
                    conn, "process.cleanup_blocked", {
                        "instance_id": instance_id,
                        "outcome": "blocked", "error_code": code,
                    }, actor="process_cleanup", task_id=row["task_id"])
                conn.execute(
                    """UPDATE process_unit_cleanups
                          SET state='blocked',claim_token=NULL,
                              claim_expires_at=NULL,next_attempt_at=NULL,
                              last_error_code=?,last_event_seq=?
                        WHERE instance_id=?""",
                    (code, seq, instance_id),
                )
                return None
            if (row["cleanup_workload_status"] not in {"released", "fenced"}
                    or row["cleanup_workload_released_at"] is None):
                code = "process_unit_cleanup_workload_not_released"
                _, seq = self.graph.append_event(
                    conn, "process.cleanup_blocked", {
                        "instance_id": instance_id,
                        "outcome": "blocked", "error_code": code,
                    }, actor="process_cleanup", task_id=row["task_id"])
                conn.execute(
                    """UPDATE process_unit_cleanups
                          SET state='blocked',claim_token=NULL,
                              claim_expires_at=NULL,next_attempt_at=NULL,
                              last_error_code=?,last_event_seq=?
                        WHERE instance_id=?""",
                    (code, seq, instance_id),
                )
                return None
            claim_token = uuid.uuid4().hex
            attempt = int(row["cleanup_attempt_count"]) + 1
            _, seq = self.graph.append_event(
                conn, "process.cleanup_claimed", {
                    "instance_id": instance_id, "attempt": attempt,
                    "reclaimed": cleanup_state == "cleaning",
                }, actor="process_cleanup", task_id=row["task_id"])
            changed = conn.execute(
                """UPDATE process_unit_cleanups
                      SET state='cleaning',attempt_count=?,last_attempt_at=?,
                          next_attempt_at=NULL,claim_token=?,claim_expires_at=?,
                          last_error_code=NULL,last_event_seq=?
                    WHERE instance_id=? AND state=?""",
                (attempt, now, claim_token, claim_expires_at, seq, instance_id,
                 cleanup_state),
            )
            if changed.rowcount != 1:
                raise ProcessBackendError(
                    "process_unit_cleanup_claim_failed",
                    outcome_unknown=False)
            return dict(row), claim_token, attempt

    def _settle_terminal_cleanup(
        self,
        instance_id: str,
        claim_token: str,
        attempt: int,
        *,
        outcome: Literal["complete", "retry", "blocked"],
        error_code: str | None = None,
    ) -> Literal["complete", "pending", "blocked"]:
        if outcome not in {"complete", "retry", "blocked"}:
            raise ValueError("invalid cleanup outcome")
        now_value = datetime.now(UTC)
        now = now_value.isoformat(
            timespec="microseconds").replace("+00:00", "Z")
        code = (_stable_code(error_code, "process_unit_cleanup_failed")
                if error_code is not None else None)
        state: Literal["complete", "pending", "blocked"] = (
            "complete" if outcome == "complete"
            else "blocked" if outcome == "blocked" else "pending")
        next_attempt_at: str | None = None
        completed_at: str | None = None
        if outcome == "retry":
            state = "pending"
            delay = min(300, 2 ** min(max(attempt - 1, 0), 8))
            next_attempt_at = (now_value + timedelta(
                seconds=delay)).isoformat(
                    timespec="microseconds").replace("+00:00", "Z")
        elif outcome == "complete":
            completed_at = now
            code = None
        event_outcome = "blocked" if state == "blocked" else outcome
        with self.graph.transaction() as conn:
            row = conn.execute(
                """SELECT c.state,c.claim_token,p.task_id,
                          w.status AS workload_status,
                          w.released_at AS workload_released_at
                     FROM process_unit_cleanups c
                     JOIN process_instances p USING(instance_id)
                LEFT JOIN workload_resource_leases w USING(instance_id)
                    WHERE c.instance_id=?""",
                (instance_id,),
            ).fetchone()
            if (row is None or row["state"] != "cleaning"
                    or row["claim_token"] != claim_token):
                raise ProcessBackendError(
                    "process_unit_cleanup_claim_lost")
            if (outcome == "complete"
                    and (row["workload_status"] not in {"released", "fenced"}
                         or row["workload_released_at"] is None)):
                state = "blocked"
                event_outcome = "blocked"
                code = "process_unit_cleanup_workload_not_released"
                completed_at = None
            body: dict[str, Any] = {
                "instance_id": instance_id, "attempt": attempt,
                "outcome": event_outcome,
            }
            if code is not None:
                body["error_code"] = code
            _, seq = self.graph.append_event(
                conn, f"process.cleanup_{event_outcome}", body,
                actor="process_cleanup", task_id=row["task_id"])
            changed = conn.execute(
                """UPDATE process_unit_cleanups
                      SET state=?,next_attempt_at=?,claim_token=NULL,
                          claim_expires_at=NULL,completed_at=?,
                          last_error_code=?,last_event_seq=?
                    WHERE instance_id=? AND state='cleaning'
                      AND claim_token=?""",
                (state, next_attempt_at, completed_at, code, seq,
                 instance_id, claim_token),
            )
            if changed.rowcount != 1:
                raise ProcessBackendError(
                    "process_unit_cleanup_claim_lost")
        return state

    def cleanup_status(self) -> dict[str, int]:
        counts = {name: 0 for name in (
            "pending", "cleaning", "blocked", "retrying")}
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT state,COUNT(*) AS count,
                          SUM(CASE WHEN last_error_code IS NOT NULL
                                   THEN 1 ELSE 0 END) AS retrying
                     FROM process_unit_cleanups
                    WHERE state <> 'complete'
                    GROUP BY state"""
            ).fetchall()
        for row in rows:
            state = str(row["state"])
            if state not in {"pending", "cleaning", "blocked"}:
                raise ProcessBackendError(
                    "process_unit_cleanup_projection_invalid",
                    outcome_unknown=False)
            counts[state] = int(row["count"])
            if state in {"pending", "cleaning"}:
                counts["retrying"] += int(row["retrying"] or 0)
        return counts

    def cleanup_retained(self, *, limit: int = 32) -> dict[str, int]:
        """Retire due terminal units without weakening their identity fence."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 256:
            raise ValueError("cleanup limit must be between 1 and 256")
        now = utc_now()
        with self.graph._connect() as conn:
            instance_ids = [str(row[0]) for row in conn.execute(
                """SELECT instance_id FROM process_unit_cleanups
                    WHERE (state='pending' AND
                           (next_attempt_at IS NULL OR next_attempt_at <= ?))
                       OR (state='cleaning' AND claim_expires_at <= ?)
                    ORDER BY requested_at,instance_id LIMIT ?""",
                (now, now, limit),
            ).fetchall()]
        attempted = completed = blocked = 0
        for instance_id in instance_ids:
            with self._operation_lock(instance_id):
                claimed = self._claim_terminal_cleanup(instance_id)
                if claimed is None:
                    continue
                row, claim_token, attempt = claimed
                attempted += 1
                retire = getattr(self.backend, "retire_terminal", None)
                if not callable(retire):
                    self._settle_terminal_cleanup(
                        instance_id, claim_token, attempt, outcome="blocked",
                        error_code="process_unit_cleanup_backend_unavailable")
                    blocked += 1
                    continue
                try:
                    retire(self._cleanup_fence(row))
                except ProcessCleanupBlocked as exc:
                    self._settle_terminal_cleanup(
                        instance_id, claim_token, attempt, outcome="blocked",
                        error_code=exc.code)
                    blocked += 1
                except ProcessIdentityError:
                    self._settle_terminal_cleanup(
                        instance_id, claim_token, attempt, outcome="blocked",
                        error_code="process_unit_cleanup_identity_mismatch")
                    blocked += 1
                except ProcessBackendError as exc:
                    self._settle_terminal_cleanup(
                        instance_id, claim_token, attempt, outcome="retry",
                        error_code=exc.code)
                    # A user-bus outage usually affects every unit.  Persist
                    # one retry and end this pass instead of multiplying a
                    # bounded command timeout across the whole backlog.
                    break
                except Exception as exc:
                    self._settle_terminal_cleanup(
                        instance_id, claim_token, attempt, outcome="retry",
                        error_code="process_unit_cleanup_backend_invalid")
                    raise ProcessBackendError(
                        "process_unit_cleanup_backend_invalid") from exc
                else:
                    settled = self._settle_terminal_cleanup(
                        instance_id, claim_token, attempt, outcome="complete")
                    if settled == "complete":
                        completed += 1
                    else:
                        blocked += 1
        return {
            "attempted": attempted,
            "completed_last_pass": completed,
            "blocked_last_pass": blocked,
            **self.cleanup_status(),
        }

    def list_instances(self) -> list[dict[str, Any]]:
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM process_instances
                   ORDER BY created_at DESC""").fetchall()
            workloads = {
                str(item["instance_id"]): str(item["lease_id"])
                for item in conn.execute(
                    "SELECT instance_id,lease_id FROM workload_resource_leases"
                ).fetchall()
            }
        return [self._public_receipt(
            row, workload_lease_id=workloads.get(str(row["instance_id"])))
            for row in rows]

    def reconcile_active(self) -> list[dict[str, Any]]:
        """Adopt exact live units after restart without freeing ambiguity.

        Old/expired workload owners are first moved to ``reconciling``.  A new
        runtime adopts a lease only after the backend identity has been proven;
        absence or disagreement leaves the reservation in place.
        """
        needing = getattr(self.admission, "workloads_needing_reconciliation", None)
        current_runtime = str(getattr(self.admission, "runtime_id", ""))
        if callable(needing):
            for item in needing():
                lease_id = str(item["lease_id"])
                instance_id = str(item["instance_id"])
                previous_runtime = str(item["runtime_id"])
                if item.get("status") != "active":
                    continue
                with self.graph.transaction() as conn:
                    if previous_runtime != current_runtime:
                        changed = (
                            self.admission
                            .mark_stale_workload_runtime_reconciling_in_transaction(
                                conn, lease_id, instance_id, previous_runtime,
                                reason="runtime_restarted"))
                    else:
                        changed = (
                            self.admission
                            .mark_workload_reconciling_in_transaction(
                                conn, lease_id, instance_id,
                                reason="monitor_lost"))
                    if not changed:
                        row = self._workload_row(conn, instance_id)
                        if row is None or row["status"] != "reconciling":
                            raise ProcessAdmissionError(
                                "process_workload_reconcile_failed")
        with self.graph._connect() as conn:
            instance_ids = [str(row[0]) for row in conn.execute(
                """SELECT instance_id FROM process_instances
                   WHERE state IN ('prepared','starting','running',
                                   'stop_requested','stopping',
                                   'reconcile_required')
                   ORDER BY created_at""").fetchall()]
        receipts: list[dict[str, Any]] = []
        for instance_id in instance_ids:
            try:
                receipts.append(self.inspect(instance_id))
            except (ProcessBrokerError, ProcessSpecError, PermissionError):
                row, workload_lease_id = self._load_row(instance_id)
                receipts.append(self._public_receipt(
                    row, workload_lease_id=workload_lease_id))
        return receipts

    @staticmethod
    def _contains_private_process_fields(value: Any) -> bool:
        forbidden = {
            "argv", "command", "control_group", "cwd", "environment", "env",
            "invocation_id", "leader_pid", "pgid", "pid", "raw_output",
            "stderr", "stdout", "unit", "unit_name",
        }
        if isinstance(value, dict):
            return any(
                str(key).casefold() in forbidden
                or ProcessBroker._contains_private_process_fields(item)
                for key, item in value.items())
        if isinstance(value, list):
            return any(ProcessBroker._contains_private_process_fields(item)
                       for item in value)
        return False

    def _verified_historical_launch_receipt(
        self, row: Any, value: Mapping[str, Any],
    ) -> bool:
        """Verify a launch observation that lifecycle monitoring advanced.

        Short-lived processes can exit between ``launch()`` returning and the
        independent verifier's backend sample.  The append-only lifecycle
        event remains authoritative proof that the exact idempotency-bound
        instance reached the returned state, while immutable receipt fields
        remain bound to the current durable row.
        """
        state = value.get("state")
        if state not in {
                "running", "stop_requested", "stopping",
                "terminated", "exited"}:
            return False
        replay = value.get("idempotent_replay")
        if (not isinstance(replay, bool)
                or value.get("forced") is not False
                or value.get("verified") is not True):
            return False
        event_type = f"process.{state}"
        with self.graph._connect() as conn:
            events = conn.execute(
                """SELECT payload_json FROM graph_events
                   WHERE event_type=? AND task_id IS ? ORDER BY seq""",
                (event_type, row["task_id"])).fetchall()
        result_code = _stable_code(value.get("result_code"), "unknown")
        observed = False
        for event in events:
            try:
                payload = json.loads(event["payload_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if (payload.get("instance_id") == row["instance_id"]
                    and payload.get("to") == state
                    and payload.get("result_code") == result_code):
                observed = True
                break
        if not observed:
            return False
        expected = self._public_receipt(
            row, workload_lease_id=None,
            idempotent_replay=replay, forced=False)
        expected["state"] = state
        expected["verified"] = True
        expected["result_code"] = result_code
        if state not in _TERMINAL_INSTANCE_STATES:
            expected["finished_at"] = None
        return dict(value) == expected

    def reconciliation_receipt(
        self,
        tool_name: str,
        expected_binding: ProcessLaunchBinding | ProcessInstanceBinding
                          | Mapping[str, Any],
        args: Mapping[str, Any],
        idempotency_key: str,
        *,
        task_id: str,
        step_id: str,
        action_id: str,
        attempt_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Observe an uncertain process effect without launching/signalling.

        ``inspect`` may advance Friday's durable projection from authoritative
        systemd evidence, but this path never invokes the launch or termination
        backend operations.
        """
        _validate_opaque_identifier("idempotency key", idempotency_key)
        _validate_opaque_identifier("task ID", task_id)
        _validate_opaque_identifier("step ID", step_id)
        _validate_opaque_identifier("action ID", action_id)
        with self.graph._connect() as conn:
            durable_step = conn.execute(
                """SELECT task_id,action_id,tool_name,args_sha256,
                          idempotency_key,executor_binding_json
                   FROM task_steps WHERE step_id=?""", (step_id,)).fetchone()
        if (durable_step is None
                or durable_step["task_id"] != task_id
                or durable_step["action_id"] != action_id
                or durable_step["tool_name"] != tool_name
                or durable_step["idempotency_key"] != idempotency_key
                or durable_step["args_sha256"]
                    != sha256_text(canonical_json(dict(args)))):
            return None
        try:
            durable_binding = json.loads(
                str(durable_step["executor_binding_json"]))
        except (TypeError, json.JSONDecodeError):
            return None
        supplied_binding = (
            expected_binding.model_dump(mode="json")
            if isinstance(expected_binding,
                          (ProcessLaunchBinding, ProcessInstanceBinding))
            else dict(expected_binding))
        if durable_binding != supplied_binding:
            return None
        if tool_name == "machine_launch_process":
            process_binding: Any = expected_binding
            if (isinstance(supplied_binding, dict)
                    and supplied_binding.get("kind")
                        == "desktop_application_launch"):
                process_binding = supplied_binding.get("process")
            binding = ProcessLaunchBinding.model_validate(process_binding)
            if (set(args) - {"spec_id", "parameter_values"}
                    or args.get("spec_id") != binding.spec_id):
                return None
            with self.graph._connect() as conn:
                row = conn.execute(
                    """SELECT * FROM process_instances
                       WHERE launch_idempotency_key=?""",
                    (idempotency_key,)).fetchone()
            if (row is None or row["spec_id"] != binding.spec_id
                    or row["task_id"] != task_id
                    or row["step_id"] != step_id
                    or row["action_id"] != action_id
                    or row["spec_fingerprint"] != binding.spec_fingerprint
                    or row["sandbox_fingerprint"]
                        != binding.sandbox_fingerprint
                    or row["args_sha256"] != binding.args_sha256):
                return None
            receipt = self.inspect(str(row["instance_id"]))
            state = receipt.get("state")
            if state == "launch_failed":
                return dict(receipt) | {"idempotent_replay": True,
                                        "forced": False}
            if (receipt.get("verified") is not True
                    or state not in {
                        "running", "stop_requested", "stopping",
                        "terminated", "exited"}):
                return None
            return dict(receipt) | {"idempotent_replay": True,
                                    "forced": False}
        if tool_name == "machine_terminate_process":
            try:
                binding = ProcessInstanceBinding.model_validate(expected_binding)
                _validate_opaque_identifier("attempt ID", str(attempt_id or ""))
            except (TypeError, ValueError):
                return None
            if (binding.operation != "terminate"
                    or set(args) != {"instance_id"}
                    or args.get("instance_id") != binding.instance_id):
                return None
            binding_json = canonical_json(binding.model_dump(mode="json"))
            binding_sha256 = sha256_text(binding_json)
            with self.graph._connect() as conn:
                operation = conn.execute(
                    "SELECT * FROM process_operations WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                row = conn.execute(
                    "SELECT * FROM process_instances WHERE instance_id=?",
                    (binding.instance_id,),
                ).fetchone()
            if (operation is None or row is None
                    or operation["task_id"] != task_id
                    or operation["step_id"] != step_id
                    or operation["action_id"] != action_id
                    or operation["attempt_id"] != attempt_id
                    or operation["tool_name"] != tool_name
                    or operation["args_sha256"]
                        != sha256_text(canonical_json(dict(args)))
                    or operation["instance_id"] != binding.instance_id
                    or operation["executor_binding_json"] != binding_json
                    or operation["executor_binding_sha256"] != binding_sha256
                    or operation["target_boundary_sha256"]
                        != self._target_boundary_sha256_from_row(row)
                    or bool(operation["force"])):
                return None
            status = str(operation["status"])
            if status == "known_failed":
                return {
                    "status": "failed", "verified": True,
                    "instance_id": binding.instance_id,
                    "operation": "terminate",
                    "result_code": _stable_code(
                        operation["error_code"], "process_termination_failed"),
                    "idempotent_replay": True, "forced": False,
                }
            if status == "effect_acknowledged":
                try:
                    self.inspect(binding.instance_id)
                except (ProcessBrokerError, ProcessSpecError, PermissionError):
                    return None
            elif status != "completed":
                # A passive lifecycle sample may still release resources, but it
                # can never promote prepared/dispatching/unknown causality.
                try:
                    self.inspect(binding.instance_id)
                except (ProcessBrokerError, ProcessSpecError, PermissionError):
                    pass
                return None
            with self.graph._connect() as conn:
                operation = conn.execute(
                    "SELECT * FROM process_operations WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                row = conn.execute(
                    "SELECT * FROM process_instances WHERE instance_id=?",
                    (binding.instance_id,),
                ).fetchone()
            if (operation is None or row is None
                    or operation["status"] != "completed"
                    or row["state"] != "terminated"):
                return None
            receipt = self._public_receipt(
                row, workload_lease_id=None,
                idempotent_replay=True, forced=False)
            return dict(receipt) | {"idempotent_replay": True,
                                    "forced": False}
        return None

    def _operation_event_matches(
        self,
        conn: Any,
        operation: Mapping[str, Any],
        sequence: Any,
        status: str,
        *,
        error_code: str | None = None,
        postcondition_state: str | None = None,
    ) -> bool:
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            return False
        event = conn.execute(
            "SELECT * FROM graph_events WHERE seq=?", (sequence,)
        ).fetchone()
        expected = canonical_json(self._operation_event_body(
            operation, status, error_code=error_code,
            postcondition_state=postcondition_state))
        return bool(
            event is not None
            and event["event_type"] == f"process_operation.{status}"
            and event["actor"] == "process_broker"
            and event["task_id"] == operation["task_id"]
            and event["payload_json"] == expected
            and event["payload_sha256"] == sha256_text(expected))

    def _verified_completed_termination_operation(
        self,
        conn: Any,
        operation: Mapping[str, Any],
        row: Mapping[str, Any],
        *,
        args_sha256: str,
    ) -> bool:
        step = conn.execute(
            "SELECT * FROM task_steps WHERE step_id=?",
            (operation["step_id"],),
        ).fetchone()
        receipt = conn.execute(
            "SELECT * FROM action_receipts WHERE idempotency_key=?",
            (operation["idempotency_key"],),
        ).fetchone()
        attempt = conn.execute(
            "SELECT * FROM action_attempts WHERE attempt_id=?",
            (operation["attempt_id"],),
        ).fetchone()
        if (step is None or receipt is None or attempt is None
                or operation["status"] != "completed"
                or operation["tool_name"] != "machine_terminate_process"
                or operation["args_sha256"] != args_sha256
                or operation["instance_id"] != row["instance_id"]
                or row["state"] != "terminated"
                or bool(operation["force"])
                or operation["error_code"] is not None
                or operation["completed_at"] is None
                or operation["executor_binding_json"]
                    != step["executor_binding_json"]
                or operation["executor_binding_sha256"]
                    != sha256_text(str(step["executor_binding_json"]))
                or operation["target_boundary_sha256"]
                    != self._target_boundary_sha256_from_row(row)
                or step["task_id"] != operation["task_id"]
                or step["action_id"] != operation["action_id"]
                or step["idempotency_key"] != operation["idempotency_key"]
                or step["tool_name"] != operation["tool_name"]
                or step["args_sha256"] != operation["args_sha256"]
                or receipt["task_id"] != operation["task_id"]
                or receipt["step_id"] != operation["step_id"]
                or receipt["action_id"] != operation["action_id"]
                or receipt["tool_name"] != operation["tool_name"]
                or receipt["args_sha256"] != operation["args_sha256"]
                or attempt["step_id"] != operation["step_id"]
                or attempt["idempotency_key"] != operation["idempotency_key"]
                or int(attempt["attempt_number"])
                    != int(operation["attempt_number"])
                or attempt["lease_id"] != operation["step_lease_id"]
                or attempt["worker_id"] != operation["worker_id"]):
            return False
        sequences = (
            operation["prepared_event_seq"], operation["dispatch_event_seq"],
            operation["outcome_event_seq"],
            operation["postcondition_event_seq"],
        )
        if (any(isinstance(value, bool) or not isinstance(value, int)
                for value in sequences)
                or not (sequences[0] < sequences[1] < sequences[2]
                        < sequences[3])
                or operation["last_event_seq"] != sequences[3]):
            return False
        return bool(
            self._operation_event_matches(
                conn, operation, sequences[0], "prepared")
            and self._operation_event_matches(
                conn, operation, sequences[1], "dispatching")
            and self._operation_event_matches(
                conn, operation, sequences[2], "effect_acknowledged")
            and self._operation_event_matches(
                conn, operation, sequences[3], "completed",
                postcondition_state="terminated"))

    def verify_receipt(
        self,
        tool_name: str,
        result: Any,
        args: Mapping[str, Any] | None,
        idempotency_key: str | None,
    ) -> bool:
        """Independently bind a public receipt to durable/backend state."""
        if isinstance(result, str):
            try:
                value = json.loads(result)
            except json.JSONDecodeError:
                return False
        else:
            value = result
        if not isinstance(args, Mapping) or self._contains_private_process_fields(
                value):
            return False
        if tool_name == "machine_list_process_specs":
            if args or idempotency_key not in {None, ""}:
                return False
            current = self.list_specs()
            if isinstance(value, list):
                return value == current
            return bool(
                isinstance(value, dict)
                and set(value) == {"status", "verified", "specs"}
                and value.get("status") == "ok"
                and value.get("verified") is True
                and value.get("specs") == current)
        if not isinstance(value, dict):
            return False
        instance_id = value.get("instance_id")
        if not isinstance(instance_id, str):
            return False
        try:
            _validate_opaque_identifier("process instance ID", instance_id)
        except ValueError:
            return False

        linked_launch_row = None
        linked_termination_row = None
        if tool_name == "machine_launch_process":
            if not isinstance(idempotency_key, str):
                return False
            try:
                _validate_opaque_identifier("idempotency key", idempotency_key)
                if not set(args).issubset({"spec_id", "parameter_values"}):
                    return False
                spec_id = str(args.get("spec_id") or "")
                parameter_values = args.get("parameter_values") or {}
                rendered = self.registry.render(spec_id, parameter_values)
            except (ValueError, ProcessSpecError, PermissionError):
                return False
            with self.graph._connect() as conn:
                row = conn.execute(
                    """SELECT * FROM process_instances
                       WHERE launch_idempotency_key=?""",
                    (idempotency_key,)).fetchone()
            if (row is None or row["instance_id"] != instance_id
                    or row["spec_id"] != spec_id
                    or row["args_sha256"] != rendered.args_sha256):
                return False
            linked_launch_row = row
        elif tool_name in {"machine_inspect_process",
                           "machine_terminate_process"}:
            if (set(args) != {"instance_id"}
                    or args.get("instance_id") != instance_id):
                return False
            if tool_name == "machine_terminate_process":
                if not isinstance(idempotency_key, str):
                    return False
                with self.graph._connect() as conn:
                    step = conn.execute(
                        "SELECT * FROM task_steps WHERE idempotency_key=?",
                        (idempotency_key,)).fetchone()
                    operation = conn.execute(
                        """SELECT * FROM process_operations
                           WHERE idempotency_key=?""",
                        (idempotency_key,)).fetchone()
                    termination_row = conn.execute(
                        "SELECT * FROM process_instances WHERE instance_id=?",
                        (instance_id,),
                    ).fetchone()
                if (step is None
                        or step["tool_name"] != "machine_terminate_process"
                        or step["args_sha256"]
                            != sha256_text(canonical_json(dict(args)))
                        or operation is None or termination_row is None):
                    return False
                try:
                    binding = ProcessInstanceBinding.model_validate_json(
                        str(step["executor_binding_json"]))
                except (TypeError, ValueError):
                    return False
                if (binding.operation != "terminate"
                        or binding.instance_id != instance_id):
                    return False
                with self.graph._connect() as conn:
                    if not self._verified_completed_termination_operation(
                            conn, operation, termination_row,
                            args_sha256=sha256_text(
                                canonical_json(dict(args)))):
                        return False
                linked_termination_row = termination_row
        else:
            return False
        if (tool_name == "machine_launch_process"
                and linked_launch_row is not None
                and self._verified_historical_launch_receipt(
                    linked_launch_row, value)):
            return True
        if (tool_name == "machine_terminate_process"
                and linked_termination_row is not None):
            actual = self._public_receipt(
                linked_termination_row, workload_lease_id=None)
        else:
            try:
                actual = self.inspect(instance_id)
            except (ValueError, ProcessBrokerError, ProcessSpecError):
                return False
        if (actual.get("verified") is not True
                or value.get("verified") is not True
                or set(value) != set(actual)):
            return False
        expected = dict(actual)
        if tool_name == "machine_launch_process":
            # An exact replay legitimately differs only in this presentation
            # bit; the durable identity/result/output fields remain identical.
            replay = value.get("idempotent_replay")
            if not isinstance(replay, bool) or value.get("forced") is not False:
                return False
            expected["idempotent_replay"] = replay
        elif tool_name == "machine_terminate_process":
            # A stop request is not a termination receipt.  Only an exact
            # terminal postcondition may satisfy the durable task, and the
            # public v1 tool never authorizes force escalation.
            if actual.get("state") != "terminated":
                return False
            replay = value.get("idempotent_replay")
            if not isinstance(replay, bool) or value.get("forced") is not False:
                return False
            expected["idempotent_replay"] = replay
        return value == expected

    @staticmethod
    def _public_receipt(
        row: Any,
        *,
        workload_lease_id: str | None,
        idempotent_replay: bool = False,
        forced: bool = False,
    ) -> dict[str, Any]:
        state = str(row["state"])
        receipt: dict[str, Any] = {
            "status": "ok",
            "verified": state in {
                "running", "stop_requested", "stopping", "terminated", "exited"},
            "instance_id": str(row["instance_id"]),
            "spec_id": str(row["spec_id"]),
            "state": state,
            "persistent": bool(row["persistent"]),
            "idempotent_replay": bool(idempotent_replay),
            "forced": bool(forced),
            "prepared_at": row["prepared_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "result_code": _stable_code(row["result_code"], "unknown"),
            "output": {
                "stdout_bytes": int(row["stdout_bytes"] or 0),
                "stdout_sha256": row["stdout_sha256"],
                "stdout_truncated": bool(row["stdout_truncated"]),
                "stderr_bytes": int(row["stderr_bytes"] or 0),
                "stderr_sha256": row["stderr_sha256"],
                "stderr_truncated": bool(row["stderr_truncated"]),
            },
        }
        return receipt


__all__ = [
    "BackendLaunchRequest", "BackendObservation", "BubblewrapProfile",
    "ExecutableIdentity", "PrevalidatedFile", "ProcessAdmissionError",
    "ProcessApprovalPreview", "ProcessBindingError", "ProcessInstanceBinding",
    "ProcessLaunchBinding", "ProcessOperationContext",
    "ProcessBackend", "ProcessBackendError", "ProcessBroker",
    "ProcessBrokerError", "ProcessIdentityError", "ProcessLimits",
    "ProcessParameter", "ProcessPresentation", "ProcessResources",
    "ProcessSessionAccess", "ProcessSpec", "ProcessSpecError",
    "ProcessSpecRegistry", "RenderedProcess",
    "SystemdUserProcessBackend",
    "WorkloadAdmission",
]
