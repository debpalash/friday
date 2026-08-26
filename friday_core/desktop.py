"""State-bound, privacy-preserving control of the local Hyprland desktop.

The model never receives compositor addresses, PIDs, process paths, window
titles, Lua, or dispatcher arguments.  It selects an opaque window ID from a
fresh observation.  Consequential actions are then bound to the exact desktop
session and process identity before the durable task worker can dispatch them.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .graph import GraphStore, canonical_json, sha256_text
from .processes import (BackendObservation, ExecutableIdentity,
                        ProcessLaunchBinding, ProcessPresentation)


_WINDOW_ID = re.compile(r"win_[0-9a-f]{40}\Z")
_ADDRESS = re.compile(r"0x[0-9a-fA-F]{1,32}\Z")
_INSTANCE = re.compile(r"[A-Za-z0-9_.:-]{8,240}\Z")
_WAYLAND_SOCKET = re.compile(r"wayland-[0-9]{1,5}\Z")
_APPLICATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_STABLE_ID = re.compile(r"[0-9a-fA-F]{8,64}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RESPONSE_BYTES = 2_000_000
_SUPPORTED_HYPRLAND_COMMITS = frozenset({
    "efb50993780079460b0cbed1363e2166a2de1d9f",  # Hyprland 0.56.2
})
_HYPRLAND_USER_UNIT = "wayland-wm@hyprland.desktop.service"


class DesktopBrokerError(RuntimeError):
    """A desktop failure with a stable, non-sensitive public code."""

    code = "desktop_broker_error"
    outcome_unknown = False

    def __init__(self, code: str | None = None):
        selected = str(code or self.code)
        if re.fullmatch(r"[a-z0-9_.:-]{1,80}", selected) is None:
            selected = self.code
        self.code = selected
        super().__init__(selected)


class DesktopUnavailableError(DesktopBrokerError):
    code = "desktop_unavailable"


class DesktopBindingError(DesktopBrokerError):
    code = "desktop_window_binding_changed"


class DesktopActionError(DesktopBrokerError):
    code = "desktop_action_not_confirmed"
    # This exception is emitted only after an action may have crossed the
    # compositor dispatch boundary.  The durable worker must quarantine the
    # attempt for reconciliation rather than recording an ordinary failure.
    outcome_unknown = True


class DesktopWindowObservation(BaseModel):
    """Private compositor/process observation; never returned to the model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)
    session_signature: str = Field(min_length=8, max_length=240, repr=False)
    address: str = Field(pattern=r"^0x[0-9a-fA-F]{1,32}$", repr=False)
    stable_id: str = Field(pattern=r"^[0-9a-fA-F]{8,64}$", repr=False)
    application_id: str = Field(min_length=1, max_length=128)
    workspace_id: int = Field(ge=-100_000, le=100_000)
    active: bool = False
    floating: bool = False
    fullscreen: bool = False
    pid: int = Field(gt=0, repr=False)
    start_ticks: int = Field(ge=0, repr=False)
    executable_identity: ExecutableIdentity = Field(repr=False)

    @property
    def runtime_identity_sha256(self) -> str:
        return sha256_text(canonical_json({
            "session_fingerprint": self.session_fingerprint,
            "address": self.address.casefold(),
            "stable_id": self.stable_id.casefold(),
            "application_id_sha256": sha256_text(self.application_id),
            "pid": self.pid,
            "start_ticks": self.start_ticks,
            "executable_identity": self.executable_identity.model_dump(mode="json"),
        }))


class DesktopSnapshot(BaseModel):
    """One coherent-enough compositor snapshot used for postcondition checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)
    session_signature: str = Field(min_length=8, max_length=240, repr=False)
    windows: tuple[DesktopWindowObservation, ...]
    # Exact clients that still exist but may currently be unmapped.  Public
    # listing uses ``windows`` only; close absence checks use this private set.
    present_windows: tuple[DesktopWindowObservation, ...] | None = Field(
        default=None, repr=False)
    # False means at least one mapped compositor client could not be given an
    # exact same-user process identity.  Positive target observations remain
    # usable, but absence is not evidence that a bound window closed.
    inventory_complete: bool = True


class DesktopWindowBinding(BaseModel):
    """Persistence-safe exact binding for one approved window operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["desktop_window"] = "desktop_window"
    operation: Literal["focus", "close"]
    window_id: str = Field(pattern=r"^win_[0-9a-f]{40}$")
    session_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    application_id_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    application: str = Field(min_length=1, max_length=80)
    workspace_id: int = Field(ge=-100_000, le=100_000)
    args_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DesktopApplicationLaunchBinding(BaseModel):
    """Exact process + compositor session bound before GUI launch approval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["desktop_application_launch"] = "desktop_application_launch"
    process: ProcessLaunchBinding
    session_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    presentation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    application_id_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    application: str = Field(min_length=1, max_length=80)
    window_owner: Literal["leader", "managed_cgroup"] = "leader"


class DesktopBackend(Protocol):
    def snapshot(self) -> DesktopSnapshot:
        ...

    def focus_window(self, session_signature: str, address: str) -> None:
        ...

    def close_window(self, session_signature: str, address: str) -> None:
        ...


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _hash_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: str) -> tuple[str, ExecutableIdentity]:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise DesktopUnavailableError("desktop_executable_invalid")
    try:
        canonical = str(candidate.resolve(strict=True))
        fd = os.open(canonical, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise DesktopUnavailableError("desktop_executable_unavailable") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise DesktopUnavailableError("desktop_executable_invalid")
        digest = _hash_fd(fd)
        after = os.fstat(fd)
        if ((before.st_dev, before.st_ino, before.st_size,
             before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns)):
            raise DesktopUnavailableError("desktop_executable_changed")
        return canonical, ExecutableIdentity(
            device=int(after.st_dev), inode=int(after.st_ino), sha256=digest,
            size=int(after.st_size), mode=stat.S_IMODE(after.st_mode))
    finally:
        os.close(fd)


def _proc_start_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        tail = raw[raw.rfind(")") + 2:].split()
        # The tail starts at field 3; process start time is field 22.
        return int(tail[19])
    except (OSError, ValueError, IndexError) as exc:
        raise DesktopUnavailableError("desktop_process_identity_unavailable") from exc


def _proc_identity(pid: int) -> tuple[int, ExecutableIdentity, str]:
    if pid <= 0:
        raise DesktopUnavailableError("desktop_process_identity_invalid")
    start_before = _proc_start_ticks(pid)
    try:
        status_lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        uid_line = next(line for line in status_lines if line.startswith("Uid:"))
        real_uid = int(uid_line.split()[1])
        path = os.readlink(f"/proc/{pid}/exe")
        fd = os.open(f"/proc/{pid}/exe", os.O_RDONLY | os.O_CLOEXEC)
    except (OSError, StopIteration, ValueError, IndexError) as exc:
        raise DesktopUnavailableError("desktop_process_identity_unavailable") from exc
    try:
        observed = os.fstat(fd)
        if real_uid != os.getuid() or not stat.S_ISREG(observed.st_mode):
            raise DesktopUnavailableError("desktop_process_identity_invalid")
        digest = _hash_fd(fd)
        identity = ExecutableIdentity(
            device=int(observed.st_dev), inode=int(observed.st_ino), sha256=digest,
            size=int(observed.st_size), mode=stat.S_IMODE(observed.st_mode))
    finally:
        os.close(fd)
    if _proc_start_ticks(pid) != start_before:
        raise DesktopUnavailableError("desktop_process_identity_changed")
    return start_before, identity, path


class HyprlandDesktopBackend:
    """Pinned `hyprctl` backend for the one live user-owned compositor."""

    def __init__(
        self,
        *,
        runner: Runner = subprocess.run,
        hyprctl: str = "/usr/bin/hyprctl",
        loginctl: str = "/usr/bin/loginctl",
        systemctl: str = "/usr/bin/systemctl",
        runtime_dir: str | None = None,
        command_timeout_seconds: float = 5.0,
        supported_commits: Sequence[str] = tuple(_SUPPORTED_HYPRLAND_COMMITS),
    ):
        if command_timeout_seconds <= 0:
            raise ValueError("desktop command timeout must be positive")
        self.runner = runner
        self.command_timeout_seconds = float(command_timeout_seconds)
        self.hyprctl, self._hyprctl_identity = _file_identity(hyprctl)
        self.loginctl, self._loginctl_identity = _file_identity(loginctl)
        self.systemctl, self._systemctl_identity = _file_identity(systemctl)
        self._command_identities = {
            self.hyprctl: self._hyprctl_identity,
            self.loginctl: self._loginctl_identity,
            self.systemctl: self._systemctl_identity,
        }
        self.supported_commits = frozenset(str(item) for item in supported_commits)
        if (not self.supported_commits
                or any(re.fullmatch(r"[0-9a-f]{40}", item) is None
                       for item in self.supported_commits)):
            raise ValueError("desktop adapter commits must be exact SHA-1 values")
        expected_runtime = f"/run/user/{os.getuid()}"
        self.runtime_dir = runtime_dir or expected_runtime
        if self.runtime_dir != expected_runtime:
            raise DesktopUnavailableError("desktop_runtime_directory_invalid")
        try:
            observed = os.lstat(self.runtime_dir)
        except OSError as exc:
            raise DesktopUnavailableError(
                "desktop_runtime_directory_unavailable") from exc
        if (not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode)
                or observed.st_uid != os.getuid()
                or stat.S_IMODE(observed.st_mode) & 0o077):
            raise DesktopUnavailableError("desktop_runtime_directory_invalid")
        self._environment = {
            "PATH": "/usr/bin:/bin",
            "XDG_RUNTIME_DIR": self.runtime_dir,
        }

    def _verify_command(self, path: str) -> None:
        expected = self._command_identities.get(path)
        if expected is None:
            raise DesktopUnavailableError("desktop_control_identity_invalid")
        canonical, identity = _file_identity(path)
        if canonical != path or identity != expected:
            raise DesktopUnavailableError("desktop_control_identity_changed")

    def _run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if not command:
            raise DesktopUnavailableError("desktop_control_identity_invalid")
        self._verify_command(str(command[0]))
        try:
            result = self.runner(
                list(command), capture_output=True, text=True,
                timeout=self.command_timeout_seconds, check=False,
                env=self._environment)
        except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
            raise DesktopUnavailableError("desktop_control_unavailable") from exc
        stdout = getattr(result, "stdout", None)
        stderr = getattr(result, "stderr", None)
        returncode = getattr(result, "returncode", None)
        if (not isinstance(stdout, str) or not isinstance(stderr, str)
                or not isinstance(returncode, int)):
            raise DesktopUnavailableError("desktop_response_invalid")
        try:
            output_size = len(stdout.encode("utf-8")) + len(
                stderr.encode("utf-8"))
        except UnicodeError as exc:
            raise DesktopUnavailableError("desktop_response_invalid") from exc
        if output_size > _MAX_RESPONSE_BYTES:
            raise DesktopUnavailableError("desktop_response_too_large")
        if returncode:
            raise DesktopUnavailableError("desktop_control_failed")
        return result

    def _json(self, *arguments: str) -> Any:
        result = self._run((self.hyprctl, "-j", *arguments))
        try:
            return json.loads(result.stdout or "null")
        except (json.JSONDecodeError, RecursionError) as exc:
            raise DesktopUnavailableError("desktop_response_invalid") from exc

    @staticmethod
    def _properties(output: str) -> dict[str, str]:
        values: dict[str, str] = {}
        for line in output.splitlines():
            if not line or "=" not in line:
                raise DesktopUnavailableError("desktop_response_invalid")
            key, value = line.split("=", 1)
            if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", key)
                    or key in values or len(value.encode("utf-8")) > 1024):
                raise DesktopUnavailableError("desktop_response_invalid")
            values[key] = value
        return values

    def _local_wayland_session(self) -> tuple[str, str]:
        result = self._run((
            self.loginctl, "list-sessions", "--json=short"))
        try:
            sessions = json.loads(result.stdout or "null")
        except json.JSONDecodeError as exc:
            raise DesktopUnavailableError("desktop_session_invalid") from exc
        if not isinstance(sessions, list) or len(sessions) > 64:
            raise DesktopUnavailableError("desktop_session_invalid")
        valid: list[tuple[str, str]] = []
        for item in sessions:
            if (not isinstance(item, Mapping)
                    or item.get("uid") != os.getuid()
                    or item.get("class") != "user"
                    or not isinstance(item.get("seat"), str)
                    or not item.get("seat")):
                continue
            session_id = str(item.get("session") or "")
            if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", session_id) is None:
                continue
            details = self._run((
                self.loginctl, "show-session", session_id,
                "--property=User", "--property=Seat", "--property=Remote",
                "--property=Active", "--property=State", "--property=Type",
                "--property=Class", "--property=LockedHint",
                "--property=Service"))
            values = self._properties(details.stdout or "")
            required = {
                "User", "Seat", "Remote", "Active", "State", "Type",
                "Class", "LockedHint", "Service",
            }
            if set(values) != required:
                continue
            if not (
                values["User"] == str(os.getuid())
                and values["Seat"] == item.get("seat")
                and values["Remote"] == "no"
                and values["Active"] == "yes"
                and values["State"] == "active"
                and values["Type"] == "wayland"
                and values["Class"] == "user"
                and values["LockedHint"] == "no"
                and re.fullmatch(r"[A-Za-z0-9_.@-]{1,128}",
                                 values["Service"]) is not None
            ):
                continue
            valid.append((session_id, sha256_text(canonical_json(values))))
        if len(valid) != 1:
            raise DesktopUnavailableError(
                "desktop_session_unavailable" if not valid
                else "desktop_session_ambiguous")
        return valid[0]

    def _compositor_authority(self, compositor_pid: int) -> dict[str, Any]:
        result = self._run((
            self.systemctl, "--user", "show", _HYPRLAND_USER_UNIT,
            "--property=ActiveState", "--property=SubState",
            "--property=InvocationID", "--property=MainPID",
            "--property=ControlGroup"))
        values = self._properties(result.stdout or "")
        if set(values) != {
                "ActiveState", "SubState", "InvocationID", "MainPID",
                "ControlGroup"}:
            raise DesktopUnavailableError("desktop_compositor_identity_invalid")
        try:
            main_pid = int(values["MainPID"])
        except ValueError as exc:
            raise DesktopUnavailableError(
                "desktop_compositor_identity_invalid") from exc
        expected_control_group = (
            f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
            f"session.slice/{_HYPRLAND_USER_UNIT}")
        control_group = values["ControlGroup"]
        if (values["ActiveState"] != "active"
                or values["SubState"] != "running"
                or re.fullmatch(r"[0-9a-f]{32}", values["InvocationID"]) is None
                or main_pid <= 0
                or control_group != expected_control_group):
            raise DesktopUnavailableError("desktop_compositor_identity_invalid")
        try:
            compositor_cgroup = Path(
                f"/proc/{compositor_pid}/cgroup").read_text().splitlines()
            main_cgroup = Path(f"/proc/{main_pid}/cgroup").read_text().splitlines()
            main_start, main_identity, _ = _proc_identity(main_pid)
        except (OSError, DesktopBrokerError) as exc:
            raise DesktopUnavailableError(
                "desktop_compositor_identity_unavailable") from exc
        expected_line = "0::" + control_group
        if expected_line not in compositor_cgroup or expected_line not in main_cgroup:
            raise DesktopUnavailableError("desktop_compositor_identity_invalid")
        return {
            "invocation_id": values["InvocationID"],
            "control_group_sha256": sha256_text(control_group),
            "main_pid_start_ticks": main_start,
            "main_executable": main_identity.model_dump(mode="json"),
        }

    def _adapter_version(self, signature: str) -> dict[str, Any]:
        value = self._json("-i", signature, "version")
        if not isinstance(value, Mapping):
            raise DesktopUnavailableError("desktop_adapter_unsupported")
        commit = str(value.get("commit") or "")
        if (commit not in self.supported_commits
                or value.get("dirty") is not False
                or not isinstance(value.get("flags"), list)):
            raise DesktopUnavailableError("desktop_adapter_unsupported")
        return {
            "commit": commit,
            "tag": str(value.get("tag") or "")[:64],
            "branch": str(value.get("branch") or "")[:64],
            "build_aquamarine": str(value.get("buildAquamarine") or "")[:64],
            "build_hyprlang": str(value.get("buildHyprlang") or "")[:64],
            "build_hyprutils": str(value.get("buildHyprutils") or "")[:64],
            "build_hyprgraphics": str(value.get("buildHyprgraphics") or "")[:64],
        }

    def _session(self) -> tuple[str, str]:
        raw = self._json("instances")
        if not isinstance(raw, list) or len(raw) > 64:
            raise DesktopUnavailableError("desktop_session_invalid")
        hypr_root = Path(self.runtime_dir) / "hypr"
        try:
            hypr_root_info = os.lstat(hypr_root)
        except OSError as exc:
            raise DesktopUnavailableError("desktop_session_unavailable") from exc
        if (not stat.S_ISDIR(hypr_root_info.st_mode)
                or stat.S_ISLNK(hypr_root_info.st_mode)
                or hypr_root_info.st_uid != os.getuid()
                or stat.S_IMODE(hypr_root_info.st_mode) & 0o077):
            raise DesktopUnavailableError("desktop_session_invalid")
        valid: list[tuple[str, str]] = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            signature = str(item.get("instance") or "")
            socket_name = str(item.get("wl_socket") or "")
            try:
                pid = int(item.get("pid") or 0)
            except (TypeError, ValueError):
                continue
            if (_INSTANCE.fullmatch(signature) is None
                    or _WAYLAND_SOCKET.fullmatch(socket_name) is None):
                continue
            session_dir = hypr_root / signature
            socket_path = session_dir / ".socket.sock"
            wayland_path = Path(self.runtime_dir) / socket_name
            try:
                directory = os.lstat(session_dir)
                socket_info = os.lstat(socket_path)
                wayland_info = os.lstat(wayland_path)
                start_ticks, executable, executable_path = _proc_identity(pid)
            except (OSError, DesktopBrokerError):
                continue
            if (not stat.S_ISDIR(directory.st_mode)
                    or stat.S_ISLNK(directory.st_mode)
                    or directory.st_uid != os.getuid()
                    or stat.S_IMODE(directory.st_mode) & 0o077
                    or not stat.S_ISSOCK(socket_info.st_mode)
                    or socket_info.st_uid != os.getuid()
                    or not stat.S_ISSOCK(wayland_info.st_mode)
                    or wayland_info.st_uid != os.getuid()
                    or Path(executable_path).name.casefold() != "hyprland"):
                continue
            login_session, login_fingerprint = self._local_wayland_session()
            compositor_authority = self._compositor_authority(pid)
            adapter_version = self._adapter_version(signature)
            fingerprint = sha256_text(canonical_json({
                "signature": signature,
                "pid": pid,
                "start_ticks": start_ticks,
                "executable": executable.model_dump(mode="json"),
                "socket": socket_name,
                "socket_device": int(socket_info.st_dev),
                "socket_inode": int(socket_info.st_ino),
                "wayland_socket_device": int(wayland_info.st_dev),
                "wayland_socket_inode": int(wayland_info.st_ino),
                "login_session_sha256": sha256_text(login_session),
                "login_fingerprint": login_fingerprint,
                "compositor_authority": compositor_authority,
                "adapter_version": adapter_version,
            }))
            valid.append((signature, fingerprint))
        if len(valid) != 1:
            raise DesktopUnavailableError(
                "desktop_session_unavailable" if not valid
                else "desktop_session_ambiguous")
        return valid[0]

    @staticmethod
    def _application_id(item: Mapping[str, Any]) -> str | None:
        value = str(item.get("initialClass") or item.get("class") or "").strip()
        return value if _APPLICATION_ID.fullmatch(value) else None

    def snapshot(self) -> DesktopSnapshot:
        signature, fingerprint = self._session()
        clients = self._json("-i", signature, "clients")
        active = self._json("-i", signature, "activewindow")
        if (not isinstance(clients, list) or len(clients) > 1024
                or not isinstance(active, Mapping)):
            raise DesktopUnavailableError("desktop_response_invalid")
        active_address = str(active.get("address") or "").casefold()
        if active_address and _ADDRESS.fullmatch(active_address) is None:
            raise DesktopUnavailableError("desktop_response_invalid")
        windows: list[DesktopWindowObservation] = []
        present_windows: list[DesktopWindowObservation] = []
        seen_addresses: set[str] = set()
        seen_stable_ids: set[str] = set()
        inventory_complete = True
        for item in clients:
            if not isinstance(item, Mapping):
                inventory_complete = False
                continue
            mapped = item.get("mapped")
            if type(mapped) is not bool:
                inventory_complete = False
                continue
            address = str(item.get("address") or "")
            stable_id = str(item.get("stableId") or "")
            application_id = self._application_id(item)
            workspace = item.get("workspace")
            if (_ADDRESS.fullmatch(address) is None
                    or _STABLE_ID.fullmatch(stable_id) is None
                    or application_id is None
                    or not isinstance(workspace, Mapping)):
                inventory_complete = False
                continue
            normalized_address = address.casefold()
            normalized_stable_id = stable_id.casefold()
            if (normalized_address in seen_addresses
                    or normalized_stable_id in seen_stable_ids):
                raise DesktopUnavailableError("desktop_response_ambiguous")
            seen_addresses.add(normalized_address)
            seen_stable_ids.add(normalized_stable_id)
            try:
                pid = int(item.get("pid") or 0)
                workspace_id = int(workspace.get("id"))
                start_ticks, executable, _ = _proc_identity(pid)
            except (TypeError, ValueError, DesktopBrokerError):
                # A window without a stable same-user process identity is not a
                # controllable target and is intentionally absent from receipts.
                # Keep the incompleteness bit so its absence can never prove a
                # close postcondition.
                inventory_complete = False
                continue
            fullscreen = item.get("fullscreen", 0)
            observed = DesktopWindowObservation(
                session_fingerprint=fingerprint,
                session_signature=signature,
                address=normalized_address,
                stable_id=normalized_stable_id,
                application_id=application_id,
                workspace_id=workspace_id,
                active=bool(mapped and address.casefold() == active_address),
                floating=bool(item.get("floating", False)),
                fullscreen=bool(fullscreen),
                pid=pid, start_ticks=start_ticks,
                executable_identity=executable,
            )
            present_windows.append(observed)
            if mapped:
                windows.append(observed)
        windows.sort(key=lambda item: (
            not item.active, item.workspace_id,
            item.application_id.casefold(), item.address))
        present_windows.sort(key=lambda item: (
            not item.active, item.workspace_id,
            item.application_id.casefold(), item.address))
        return DesktopSnapshot(
            session_fingerprint=fingerprint,
            session_signature=signature,
            windows=tuple(windows),
            present_windows=tuple(present_windows),
            inventory_complete=inventory_complete)

    def _dispatch(self, signature: str, address: str, code: str) -> None:
        current_signature, _ = self._session()
        if signature != current_signature or _ADDRESS.fullmatch(address) is None:
            raise DesktopBindingError()
        # `code` is assembled exclusively from the address validated above; no
        # model/user text can enter the Lua evaluator.
        self._run((self.hyprctl, "-i", signature, "eval", code))

    def focus_window(self, session_signature: str, address: str) -> None:
        selector = f"address:{address.casefold()}"
        self._dispatch(
            session_signature, address,
            "hl.dispatch(hl.dsp.focus({ window = '" + selector + "' }))")

    def close_window(self, session_signature: str, address: str) -> None:
        selector = f"address:{address.casefold()}"
        self._dispatch(
            session_signature, address,
            "hl.dispatch(hl.dsp.window.close({ window = '" + selector + "' }))")


class DesktopBroker:
    """Opaque target selection, exact bindings, and verified postconditions."""

    _APPLICATION_LABELS = {
        "chromium": "Chromium",
        "com.friday.managedterminal": "Friday Terminal",
        "com.mitchellh.ghostty": "Terminal",
        "ghostty": "Terminal",
        "librewolf": "LibreWolf",
        "org.gnome.nautilus": "Files",
        "thunar": "Files",
    }

    def __init__(
        self,
        graph: GraphStore,
        backend: DesktopBackend,
        *,
        state_root: str | Path,
        action_timeout_seconds: float = 5.0,
        poll_seconds: float = 0.05,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if action_timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("desktop action timeouts must be positive")
        self.graph = graph
        self.backend = backend
        self.action_timeout_seconds = float(action_timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.sleeper = sleeper
        self.state_root = Path(state_root)
        self.state_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.state_root, 0o700)
        self._key = self._load_or_create_key(self.state_root / "window-id-key")

    @staticmethod
    def _load_or_create_key(path: Path) -> bytes:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            fd = -1
        if fd >= 0:
            try:
                key = os.urandom(32)
                os.write(fd, key)
                os.fsync(fd)
            finally:
                os.close(fd)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        try:
            info = os.lstat(path)
            if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                    or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600):
                raise DesktopUnavailableError("desktop_identity_key_invalid")
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                key = os.read(fd, 33)
            finally:
                os.close(fd)
        except OSError as exc:
            raise DesktopUnavailableError("desktop_identity_key_unavailable") from exc
        if len(key) != 32:
            raise DesktopUnavailableError("desktop_identity_key_invalid")
        return key

    def _window_id(self, window: DesktopWindowObservation) -> str:
        payload = canonical_json({
            "session": window.session_fingerprint,
            "address": window.address,
            "runtime": window.runtime_identity_sha256,
        }).encode("utf-8")
        digest = hmac.new(self._key, payload, hashlib.sha256).hexdigest()[:40]
        return f"win_{digest}"

    @classmethod
    def _application_label(cls, application_id: str) -> str:
        return cls._APPLICATION_LABELS.get(
            application_id.casefold(), "Application")

    def _public_window(self, window: DesktopWindowObservation) -> dict[str, Any]:
        return {
            "window_id": self._window_id(window),
            "application": self._application_label(window.application_id),
            "workspace_id": window.workspace_id,
            "active": window.active,
            "floating": window.floating,
            "fullscreen": window.fullscreen,
        }

    def list_windows(self) -> dict[str, Any]:
        snapshot = self.backend.snapshot()
        return {
            "status": "ok", "verified": True,
            "windows": [self._public_window(item) for item in snapshot.windows],
        }

    def binding_for_application_launch(
        self,
        process_binding: ProcessLaunchBinding | Mapping[str, Any],
        presentation: ProcessPresentation,
    ) -> DesktopApplicationLaunchBinding:
        """Bind a curated process launch to the one current desktop session."""
        process = ProcessLaunchBinding.model_validate(process_binding)
        if not isinstance(presentation, ProcessPresentation):
            presentation = ProcessPresentation.model_validate(presentation)
        snapshot = self.backend.snapshot()
        return DesktopApplicationLaunchBinding(
            process=process,
            session_fingerprint=snapshot.session_fingerprint,
            presentation_fingerprint=presentation.fingerprint,
            application_id_sha256=sha256_text(presentation.application_id),
            application=presentation.application,
            window_owner=presentation.window_owner,
        )

    @staticmethod
    def _validate_application_runtime(
        binding: DesktopApplicationLaunchBinding,
        observation: BackendObservation,
    ) -> None:
        expected = binding.process.executable_identity
        if (observation.state not in {"running", "stopping"}
                or observation.leader_pid is None
                or observation.start_ticks is None
                or observation.cgroup_empty
                or observation.exe_device != expected.device
                or observation.exe_inode != expected.inode
                or observation.exe_sha256 != expected.sha256):
            raise DesktopBindingError("desktop_application_process_changed")

    def _resolve_application_window(
        self,
        binding: DesktopApplicationLaunchBinding,
        observation: BackendObservation,
        snapshot: DesktopSnapshot,
        runtime_owner: Callable[[
            BackendObservation, DesktopWindowObservation], bool] | None = None,
    ) -> DesktopWindowObservation | None:
        if snapshot.session_fingerprint != binding.session_fingerprint:
            raise DesktopBindingError("desktop_session_binding_changed")
        self._validate_application_runtime(binding, observation)
        candidates = [window for window in snapshot.windows if
                      sha256_text(window.application_id)
                      == binding.application_id_sha256]
        if binding.window_owner == "leader":
            expected = binding.process.executable_identity
            matches = [window for window in candidates if (
                window.pid == observation.leader_pid
                and window.start_ticks == observation.start_ticks
                and window.executable_identity == expected
            )]
        else:
            if runtime_owner is None:
                raise DesktopBindingError(
                    "desktop_application_cgroup_verifier_unavailable")
            # A true result means the broker revalidated this exact PID,
            # start time, and executable inside the same managed execution.
            matches = [window for window in candidates
                       if runtime_owner(observation, window) is True]
        if len(matches) > 1:
            raise DesktopBindingError("desktop_application_window_ambiguous")
        return matches[0] if matches else None

    def _application_launch_receipt(
        self,
        binding: DesktopApplicationLaunchBinding,
        process_receipt: Mapping[str, Any],
        window: DesktopWindowObservation,
    ) -> dict[str, Any]:
        return dict(process_receipt) | {
            "presentation": {
                "status": "ok",
                "verified": True,
                "window_id": self._window_id(window),
                "application": binding.application,
            },
        }

    def confirm_application_launch(
        self,
        expected_binding: DesktopApplicationLaunchBinding | Mapping[str, Any],
        process_receipt: Mapping[str, Any],
        observation: BackendObservation,
        presentation: ProcessPresentation,
        runtime_owner: Callable[[
            BackendObservation, DesktopWindowObservation], bool] | None = None,
    ) -> dict[str, Any]:
        """Require the exact launched process to own its intended window."""
        try:
            binding = DesktopApplicationLaunchBinding.model_validate(
                expected_binding)
            if (binding.process.spec_id != process_receipt.get("spec_id")
                    or process_receipt.get("verified") is not True
                    or process_receipt.get("state") not in {"running", "stopping"}
                    or binding.presentation_fingerprint
                        != presentation.fingerprint
                    or binding.application_id_sha256
                        != sha256_text(presentation.application_id)
                    or binding.application != presentation.application
                    or binding.window_owner != presentation.window_owner):
                raise DesktopActionError(
                    "desktop_application_launch_binding_changed")
            deadline = time.monotonic() + presentation.startup_timeout_seconds
            while True:
                snapshot = self.backend.snapshot()
                window = self._resolve_application_window(
                    binding, observation, snapshot, runtime_owner)
                if window is not None:
                    return self._application_launch_receipt(
                        binding, process_receipt, window)
                if time.monotonic() >= deadline:
                    raise DesktopActionError(
                        "desktop_application_launch_not_confirmed")
                self.sleeper(self.poll_seconds)
        except DesktopActionError:
            raise
        except Exception as exc:
            # The process effect already exists. Any compositor/session failure
            # from this point is an unknown compound outcome, never a retryable
            # ordinary launch failure.
            raise DesktopActionError(
                "desktop_application_launch_outcome_unknown") from exc

    def reconciliation_application_receipt(
        self,
        expected_binding: DesktopApplicationLaunchBinding | Mapping[str, Any],
        process_receipt: Mapping[str, Any],
        observation: BackendObservation,
        presentation: ProcessPresentation,
        runtime_owner: Callable[[
            BackendObservation, DesktopWindowObservation], bool] | None = None,
    ) -> dict[str, Any] | None:
        """Observe a GUI launch postcondition without dispatching any effect."""
        binding = DesktopApplicationLaunchBinding.model_validate(
            expected_binding)
        if (binding.process.spec_id != process_receipt.get("spec_id")
                or process_receipt.get("verified") is not True
                or process_receipt.get("state") not in {"running", "stopping"}
                or binding.presentation_fingerprint != presentation.fingerprint
                or binding.application_id_sha256
                    != sha256_text(presentation.application_id)
                or binding.application != presentation.application
                or binding.window_owner != presentation.window_owner):
            return None
        snapshot = self.backend.snapshot()
        window = self._resolve_application_window(
            binding, observation, snapshot, runtime_owner)
        return (self._application_launch_receipt(
            binding, process_receipt, window) if window is not None else None)

    def verify_application_launch_receipt(
        self,
        expected_binding: DesktopApplicationLaunchBinding | Mapping[str, Any],
        result: Mapping[str, Any],
        process_receipt: Mapping[str, Any],
        observation: BackendObservation,
        presentation: ProcessPresentation,
        runtime_owner: Callable[[
            BackendObservation, DesktopWindowObservation], bool] | None = None,
    ) -> bool:
        """Independently re-observe the exact compound process/window receipt."""
        try:
            binding = DesktopApplicationLaunchBinding.model_validate(
                expected_binding)
            if (binding.presentation_fingerprint != presentation.fingerprint
                    or binding.application_id_sha256
                        != sha256_text(presentation.application_id)
                    or binding.application != presentation.application
                    or binding.window_owner != presentation.window_owner):
                return False
            snapshot = self.backend.snapshot()
            window = self._resolve_application_window(
                binding, observation, snapshot, runtime_owner)
            if window is None:
                return False
            expected = self._application_launch_receipt(
                binding, process_receipt, window)
        except (DesktopBrokerError, RuntimeError, TypeError, ValueError):
            return False
        return dict(result) == expected

    def binding_for_action(
        self,
        window_id: str,
        operation: Literal["focus", "close"],
    ) -> DesktopWindowBinding:
        if _WINDOW_ID.fullmatch(window_id) is None:
            raise ValueError("desktop window ID is invalid")
        snapshot = self.backend.snapshot()
        window = next((item for item in snapshot.windows
                       if self._window_id(item) == window_id), None)
        if window is None:
            raise DesktopBindingError("desktop_window_unavailable")
        args = {"window_id": window_id}
        return DesktopWindowBinding(
            operation=operation, window_id=window_id,
            session_fingerprint=snapshot.session_fingerprint,
            runtime_identity_sha256=window.runtime_identity_sha256,
            application_id_sha256=sha256_text(window.application_id),
            application=self._application_label(window.application_id),
            workspace_id=window.workspace_id,
            args_sha256=sha256_text(canonical_json(args)),
        )

    @staticmethod
    def approval_preview(binding: DesktopWindowBinding) -> dict[str, Any]:
        return {
            "window_id": binding.window_id,
            "application": binding.application,
            "workspace_id": binding.workspace_id,
            "operation": binding.operation,
        }

    def _resolve(
        self,
        binding: DesktopWindowBinding,
        snapshot: DesktopSnapshot,
        *,
        include_unmapped: bool = False,
    ) -> DesktopWindowObservation | None:
        if snapshot.session_fingerprint != binding.session_fingerprint:
            return None
        candidates = (
            snapshot.present_windows
            if include_unmapped and snapshot.present_windows is not None
            else snapshot.windows)
        for window in candidates:
            if self._window_id(window) != binding.window_id:
                continue
            if (window.runtime_identity_sha256
                    != binding.runtime_identity_sha256
                    or sha256_text(window.application_id)
                        != binding.application_id_sha256
                    or window.workspace_id != binding.workspace_id):
                raise DesktopBindingError()
            return window
        return None

    @staticmethod
    def _receipt(
        binding: DesktopWindowBinding,
        state: Literal["focused", "closed"],
        *,
        idempotent_replay: bool,
    ) -> dict[str, Any]:
        return {
            "status": "ok", "verified": True,
            "operation": binding.operation,
            "window_id": binding.window_id,
            "application": binding.application,
            "workspace_id": binding.workspace_id,
            "state": state,
            "idempotent_replay": bool(idempotent_replay),
        }

    def focus_window(
        self,
        window_id: str,
        *,
        expected_binding: DesktopWindowBinding | Mapping[str, Any],
    ) -> dict[str, Any]:
        binding = DesktopWindowBinding.model_validate(expected_binding)
        if (binding.operation != "focus" or binding.window_id != window_id
                or binding.args_sha256 != sha256_text(canonical_json(
                    {"window_id": window_id}))):
            raise DesktopBindingError()
        snapshot = self.backend.snapshot()
        if snapshot.session_fingerprint != binding.session_fingerprint:
            raise DesktopBindingError("desktop_session_binding_changed")
        window = self._resolve(binding, snapshot)
        if window is None:
            raise DesktopBindingError("desktop_window_unavailable")
        if window.active:
            return self._receipt(binding, "focused", idempotent_replay=True)
        try:
            self.backend.focus_window(window.session_signature, window.address)
            deadline = time.monotonic() + self.action_timeout_seconds
            while True:
                observed = self.backend.snapshot()
                if (observed.session_fingerprint
                        != binding.session_fingerprint):
                    raise DesktopBindingError(
                        "desktop_session_binding_changed")
                current = self._resolve(binding, observed)
                if current is not None and current.active:
                    return self._receipt(
                        binding, "focused", idempotent_replay=False)
                if current is None or time.monotonic() >= deadline:
                    raise DesktopActionError("desktop_focus_not_confirmed")
                self.sleeper(self.poll_seconds)
        except DesktopActionError:
            raise
        except Exception as exc:
            # Every exception from compositor dispatch onward is ambiguous,
            # including raw OS/process-identity failures during verification.
            raise DesktopActionError(
                "desktop_focus_outcome_unknown") from exc

    def close_window(
        self,
        window_id: str,
        *,
        expected_binding: DesktopWindowBinding | Mapping[str, Any],
    ) -> dict[str, Any]:
        binding = DesktopWindowBinding.model_validate(expected_binding)
        if (binding.operation != "close" or binding.window_id != window_id
                or binding.args_sha256 != sha256_text(canonical_json(
                    {"window_id": window_id}))):
            raise DesktopBindingError()
        snapshot = self.backend.snapshot()
        if snapshot.session_fingerprint != binding.session_fingerprint:
            raise DesktopBindingError("desktop_session_binding_changed")
        window = self._resolve(binding, snapshot, include_unmapped=True)
        if window is None:
            if not snapshot.inventory_complete:
                raise DesktopUnavailableError(
                    "desktop_inventory_incomplete")
            return self._receipt(binding, "closed", idempotent_replay=True)
        try:
            self.backend.close_window(window.session_signature, window.address)
            deadline = time.monotonic() + self.action_timeout_seconds
            while True:
                observed = self.backend.snapshot()
                if (observed.session_fingerprint
                        != binding.session_fingerprint):
                    raise DesktopBindingError(
                        "desktop_session_binding_changed")
                current = self._resolve(
                    binding, observed, include_unmapped=True)
                if current is None:
                    if not observed.inventory_complete:
                        raise DesktopActionError(
                            "desktop_close_outcome_unknown")
                    return self._receipt(
                        binding, "closed", idempotent_replay=False)
                if time.monotonic() >= deadline:
                    raise DesktopActionError("desktop_close_not_confirmed")
                self.sleeper(self.poll_seconds)
        except DesktopActionError:
            raise
        except Exception as exc:
            raise DesktopActionError(
                "desktop_close_outcome_unknown") from exc

    def reconciliation_receipt(
        self,
        expected_binding: DesktopWindowBinding | Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Observe an interrupted action without ever dispatching it again.

        A receipt is returned only when the exact bound session still exists
        and the intended postcondition is currently authoritative.  A changed
        session, an inactive focus target, or a still-present close target is
        deliberately left unknown for explicit operator handling.
        """
        binding = DesktopWindowBinding.model_validate(expected_binding)
        snapshot = self.backend.snapshot()
        if snapshot.session_fingerprint != binding.session_fingerprint:
            return None
        current = self._resolve(
            binding, snapshot, include_unmapped=(binding.operation == "close"))
        if binding.operation == "focus":
            if current is None or not current.active:
                return None
            return self._receipt(
                binding, "focused", idempotent_replay=True)
        if current is not None:
            return None
        if not snapshot.inventory_complete:
            return None
        return self._receipt(binding, "closed", idempotent_replay=True)

    @staticmethod
    def _contains_private_fields(value: Any) -> bool:
        forbidden = {
            "address", "argv", "command", "environment", "exe", "executable",
            "application_id", "class",
            "hyprland_instance_signature", "pid", "raw_output", "signature",
            "start_ticks", "stderr", "stdout", "title",
        }
        if isinstance(value, dict):
            return any(
                str(key).casefold() in forbidden
                or DesktopBroker._contains_private_fields(item)
                for key, item in value.items())
        if isinstance(value, list):
            return any(DesktopBroker._contains_private_fields(item)
                       for item in value)
        return False

    def _durable_binding(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        idempotency_key: str | None,
    ) -> DesktopWindowBinding | None:
        if not isinstance(idempotency_key, str) or not idempotency_key:
            return None
        with self.graph._connect() as conn:
            row = conn.execute(
                """SELECT tool_name,args_sha256,executor_binding_json
                   FROM task_steps WHERE idempotency_key=?""",
                (idempotency_key,)).fetchone()
        if (row is None or row["tool_name"] != tool_name
                or row["args_sha256"] != sha256_text(canonical_json(dict(args)))):
            return None
        try:
            return DesktopWindowBinding.model_validate_json(
                str(row["executor_binding_json"]))
        except (ValueError, TypeError):
            return None

    def verify_receipt(
        self,
        tool_name: str,
        result: Any,
        args: Mapping[str, Any] | None,
        idempotency_key: str | None,
    ) -> bool:
        if isinstance(result, str):
            try:
                value = json.loads(result)
            except json.JSONDecodeError:
                return False
        else:
            value = result
        if (not isinstance(value, dict) or not isinstance(args, Mapping)
                or self._contains_private_fields(value)):
            return False
        if tool_name == "machine_list_windows":
            if args or idempotency_key not in {None, ""}:
                return False
            try:
                return value == self.list_windows()
            except DesktopBrokerError:
                return False
        operation = {
            "machine_focus_window": "focus",
            "machine_close_window": "close",
        }.get(tool_name)
        if operation is None or set(args) != {"window_id"}:
            return False
        window_id = args.get("window_id")
        if not isinstance(window_id, str) or _WINDOW_ID.fullmatch(window_id) is None:
            return False
        binding = self._durable_binding(tool_name, args, idempotency_key)
        if (binding is None or binding.operation != operation
                or binding.window_id != window_id):
            return False
        replay = value.get("idempotent_replay")
        if not isinstance(replay, bool):
            return False
        try:
            snapshot = self.backend.snapshot()
            if snapshot.session_fingerprint != binding.session_fingerprint:
                return False
            current = self._resolve(
                binding, snapshot, include_unmapped=(operation == "close"))
        except DesktopBrokerError:
            return False
        if operation == "focus":
            if current is None or not current.active:
                return False
            expected = self._receipt(
                binding, "focused", idempotent_replay=replay)
        else:
            if current is not None or not snapshot.inventory_complete:
                return False
            expected = self._receipt(
                binding, "closed", idempotent_replay=replay)
        return value == expected


__all__ = [
    "DesktopActionError", "DesktopApplicationLaunchBinding", "DesktopBackend", "DesktopBindingError",
    "DesktopBroker", "DesktopBrokerError", "DesktopSnapshot",
    "DesktopUnavailableError", "DesktopWindowBinding",
    "DesktopWindowObservation", "HyprlandDesktopBackend",
]
