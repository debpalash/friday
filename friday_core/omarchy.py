"""Typed, receipt-backed control of an installed Omarchy desktop."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .graph import canonical_json, sha256_text
from .processes import ExecutableIdentity


OMARCHY_STATUS_TOOL = "machine_omarchy_status"
OMARCHY_ACTION_TOOLS = frozenset({
    "machine_omarchy_set_theme",
    "machine_omarchy_set_font",
    "machine_omarchy_set_nightlight",
    "machine_omarchy_set_idle",
    "machine_omarchy_set_brightness",
    "machine_omarchy_take_screenshot",
    "machine_omarchy_lock",
    "machine_omarchy_install_browser",
})
OMARCHY_TOOL_NAMES = frozenset({OMARCHY_STATUS_TOOL, *OMARCHY_ACTION_TOOLS})

_SAFE_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9]+)?$")
_SAFE_BINARY = re.compile(r"^omarchy-[a-z0-9-]+$")
_SAFE_LABEL = re.compile(r"^[^\x00-\x1f\x7f/\\]{1,100}$")
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_TOOL_OPERATION = {
    "machine_omarchy_set_theme": "set_theme",
    "machine_omarchy_set_font": "set_font",
    "machine_omarchy_set_nightlight": "set_nightlight",
    "machine_omarchy_set_idle": "set_idle",
    "machine_omarchy_set_brightness": "set_brightness",
    "machine_omarchy_take_screenshot": "take_screenshot",
    "machine_omarchy_lock": "lock",
    "machine_omarchy_install_browser": "start_browser_installer",
}

_TOOL_ROUTES = {
    "machine_omarchy_set_theme": (
        "omarchy theme set", "omarchy theme current", "omarchy theme list"),
    "machine_omarchy_set_font": (
        "omarchy font set", "omarchy font current", "omarchy font list"),
    "machine_omarchy_set_nightlight": ("omarchy toggle nightlight",),
    "machine_omarchy_set_idle": ("omarchy toggle idle",),
    "machine_omarchy_set_brightness": ("omarchy brightness display",),
    "machine_omarchy_take_screenshot": (
        "omarchy capture screenshot",),
    "machine_omarchy_lock": (
        "omarchy system lock", "omarchy shell"),
    "machine_omarchy_install_browser": (
        "omarchy launch terminal", "omarchy install browser"),
}

_REQUIRED_ROUTES = frozenset({
    "omarchy version",
    "omarchy theme current", "omarchy theme list", "omarchy theme set",
    "omarchy font current", "omarchy font list", "omarchy font set",
    "omarchy toggle nightlight", "omarchy toggle idle",
    "omarchy brightness display", "omarchy capture screenshot",
    "omarchy system lock", "omarchy shell",
    "omarchy launch terminal", "omarchy install browser",
})


class OmarchyBrokerError(RuntimeError):
    """An Omarchy failure with a stable, non-sensitive public code."""

    code = "omarchy_broker_error"

    def __init__(self, message: str | None = None):
        super().__init__(message or self.code)


class OmarchyUnavailableError(OmarchyBrokerError):
    code = "omarchy_unavailable"


class OmarchyBindingError(OmarchyBrokerError):
    code = "omarchy_binding_changed"


class OmarchyActionError(OmarchyBrokerError):
    code = "omarchy_action_outcome_unknown"


class OmarchyActionBinding(BaseModel):
    """Exact Omarchy action bound before approval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["omarchy_action"] = "omarchy_action"
    tool_name: str = Field(pattern=r"^machine_omarchy_[a-z_]+$")
    operation: Literal[
        "set_theme", "set_font", "set_nightlight", "set_idle",
        "set_brightness", "take_screenshot", "lock",
        "start_browser_installer",
    ]
    target: str = Field(min_length=1, max_length=120)
    args_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    command_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    omarchy_version: str = Field(min_length=1, max_length=40)
    bound_at_ns: int = Field(ge=1)


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _hash_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: str | Path) -> tuple[str, ExecutableIdentity]:
    lexical = Path(os.path.abspath(Path(path).expanduser()))
    if not lexical.is_absolute():
        raise OmarchyUnavailableError("omarchy_command_path_invalid")
    try:
        canonical = lexical.resolve(strict=True)
        descriptor = os.open(
            canonical,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise OmarchyUnavailableError("omarchy_command_unavailable") from exc
    try:
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        if (not stat.S_ISREG(before.st_mode) or not mode & 0o111
                or mode & 0o022 or mode & 0o6000):
            raise OmarchyUnavailableError("omarchy_command_identity_invalid")
        digest = _hash_fd(descriptor)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns):
            raise OmarchyUnavailableError("omarchy_command_identity_changed")
        return str(canonical), ExecutableIdentity(
            device=int(after.st_dev), inode=int(after.st_ino),
            sha256=digest, size=int(after.st_size), mode=mode)
    finally:
        os.close(descriptor)


def _bounded_label(value: str, *, field: str) -> str:
    text = value.strip()
    if _SAFE_LABEL.fullmatch(text) is None:
        raise OmarchyUnavailableError(f"omarchy_{field}_invalid")
    return text


class OmarchyDesktopBackend:
    """Identity-pinned adapter for the packaged Omarchy command router."""

    def __init__(
        self,
        executable: str = "/usr/share/omarchy/bin/omarchy",
        *,
        capture_root: str | Path | None = None,
        runner: Runner = subprocess.run,
    ):
        self.executable, self._router_identity = _file_identity(executable)
        self.bin_dir = Path(self.executable).parent
        self.capture_root = Path(
            capture_root or (Path.home() / "Pictures" / "Friday"))
        self.runner = runner
        catalog = self._router_command(("commands", "--json"), timeout=5.0,
                                       output_limit=512_000)
        try:
            payload = json.loads(catalog)
            commands = payload["commands"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise OmarchyUnavailableError("omarchy_catalog_invalid") from exc
        if not isinstance(commands, list):
            raise OmarchyUnavailableError("omarchy_catalog_invalid")
        self._route_identities: dict[str, ExecutableIdentity] = {}
        self._route_paths: dict[str, str] = {}
        for item in commands:
            if not isinstance(item, dict):
                continue
            route = str(item.get("route") or "")
            if route not in _REQUIRED_ROUTES:
                continue
            binary = str(item.get("binary") or "")
            if _SAFE_BINARY.fullmatch(binary) is None:
                raise OmarchyUnavailableError("omarchy_catalog_invalid")
            binary_path, identity = _file_identity(self.bin_dir / binary)
            if Path(binary_path).parent != self.bin_dir:
                raise OmarchyUnavailableError("omarchy_catalog_invalid")
            self._route_identities[route] = identity
            self._route_paths[route] = binary_path
        missing = _REQUIRED_ROUTES - set(self._route_identities)
        if missing:
            raise OmarchyUnavailableError("omarchy_routes_unavailable")
        version = self.version()
        if _SAFE_VERSION.fullmatch(version) is None:
            raise OmarchyUnavailableError("omarchy_version_invalid")

    @staticmethod
    def _environment() -> dict[str, str]:
        environment = dict(os.environ)
        for name in (
            "BASH_ENV", "CDPATH", "ENV", "GLOBIGNORE", "LD_PRELOAD",
            "PYTHONPATH", "PROMPT_COMMAND", "SHELLOPTS",
        ):
            environment.pop(name, None)
        environment["HOME"] = str(Path.home())
        environment["OMARCHY_PATH"] = "/usr/share/omarchy"
        environment["PATH"] = (
            "/usr/share/omarchy/bin:/usr/local/bin:/usr/bin:/bin")
        return environment

    def _assert_identity(self, route: str | None = None) -> None:
        _path, router = _file_identity(self.executable)
        if router != self._router_identity:
            raise OmarchyBindingError("omarchy_router_identity_changed")
        if route is not None:
            expected = self._route_identities.get(route)
            if expected is None:
                raise OmarchyUnavailableError("omarchy_route_unavailable")
            _path, observed = _file_identity(self._route_paths[route])
            if observed != expected:
                raise OmarchyBindingError("omarchy_route_identity_changed")

    def _execute(
        self,
        arguments: tuple[str, ...],
        *,
        route: str | None,
        timeout: float,
        output_limit: int = 64_000,
        environment: Mapping[str, str] | None = None,
    ) -> str:
        self._assert_identity(route)
        selected_environment = dict(environment or self._environment())
        try:
            completed = self.runner(
                [self.executable, *arguments], text=True,
                capture_output=True, timeout=timeout, check=False,
                env=selected_environment)
        except (OSError, subprocess.SubprocessError) as exc:
            raise OmarchyUnavailableError("omarchy_command_failed") from exc
        self._assert_identity(route)
        if completed.returncode != 0:
            raise OmarchyUnavailableError("omarchy_command_failed")
        output = str(completed.stdout or "")
        if len(output.encode("utf-8")) > output_limit:
            raise OmarchyUnavailableError("omarchy_response_too_large")
        return output.strip()

    def _router_command(
        self, arguments: tuple[str, ...], *, timeout: float,
        output_limit: int,
    ) -> str:
        return self._execute(
            arguments, route=None, timeout=timeout, output_limit=output_limit)

    def run(
        self, route: str, *arguments: str, timeout: float = 8.0,
        environment: Mapping[str, str] | None = None,
    ) -> str:
        if route not in self._route_identities:
            raise OmarchyUnavailableError("omarchy_route_unavailable")
        route_arguments = tuple(route.split()[1:])
        return self._execute(
            (*route_arguments, *arguments), route=route, timeout=timeout,
            environment=environment)

    def command_fingerprint(self, tool_name: str) -> str:
        routes = _TOOL_ROUTES.get(tool_name)
        if routes is None:
            raise ValueError("unsupported Omarchy tool")
        self._assert_identity()
        payload = {
            "router": self._router_identity.model_dump(mode="json"),
            "routes": {
                route: self._route_identities[route].model_dump(mode="json")
                for route in sorted(routes)
            },
        }
        return sha256_text(canonical_json(payload))

    def version(self) -> str:
        return _bounded_label(
            self.run("omarchy version", timeout=5.0), field="version")

    @staticmethod
    def _labels(output: str, *, field: str) -> list[str]:
        values = [_bounded_label(line, field=field)
                  for line in output.splitlines() if line.strip()]
        if len(values) > 256 or len(values) != len(set(values)):
            raise OmarchyUnavailableError(f"omarchy_{field}_list_invalid")
        return values

    def themes(self) -> list[str]:
        return self._labels(
            self.run("omarchy theme list"), field="theme")

    def current_theme(self) -> str:
        return _bounded_label(
            self.run("omarchy theme current"), field="theme")

    def set_theme(self, theme: str) -> None:
        self.run("omarchy theme set", theme, timeout=60.0)

    def fonts(self) -> list[str]:
        return self._labels(self.run("omarchy font list"), field="font")

    def current_font(self) -> str:
        return _bounded_label(
            self.run("omarchy font current"), field="font")

    def set_font(self, font: str) -> None:
        self.run("omarchy font set", font, timeout=20.0)

    def nightlight_enabled(self) -> bool:
        try:
            value = json.loads(self.run(
                "omarchy toggle nightlight", "--status"))
        except json.JSONDecodeError as exc:
            raise OmarchyUnavailableError("omarchy_nightlight_status_invalid") from exc
        if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
            raise OmarchyUnavailableError("omarchy_nightlight_status_invalid")
        return value["enabled"]

    def toggle_nightlight(self) -> None:
        self.run("omarchy toggle nightlight", timeout=8.0)

    def idle_mode(self) -> str:
        try:
            value = json.loads(self.run("omarchy toggle idle", "status"))
        except json.JSONDecodeError as exc:
            raise OmarchyUnavailableError("omarchy_idle_status_invalid") from exc
        if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
            raise OmarchyUnavailableError("omarchy_idle_status_invalid")
        return "stay_awake" if value["enabled"] else "allow_idle"

    def set_idle(self, mode: str) -> None:
        argument = {
            "stay_awake": "stay-awake", "allow_idle": "allow-idle",
        }.get(mode)
        if argument is None:
            raise ValueError("invalid Omarchy idle mode")
        self.run("omarchy toggle idle", argument)

    def brightness(self) -> int:
        output = self.run("omarchy brightness display", "--no-osd",
                          timeout=12.0)
        try:
            value = int(output)
        except ValueError as exc:
            raise OmarchyUnavailableError("omarchy_brightness_invalid") from exc
        if not 0 <= value <= 100:
            raise OmarchyUnavailableError("omarchy_brightness_invalid")
        return value

    def set_brightness(self, percent: int) -> None:
        self.run("omarchy brightness display", "--no-osd", f"{percent}%",
                 timeout=12.0)

    def locked(self) -> bool:
        output = self.run("omarchy shell", "lock", "isLocked")
        if output not in {"true", "false"}:
            raise OmarchyUnavailableError("omarchy_lock_status_invalid")
        return output == "true"

    def lock(self) -> None:
        self.run("omarchy system lock", timeout=8.0)

    def start_browser_installer(self, browser: str) -> None:
        if browser != "firefox":
            raise ValueError("unsupported Omarchy browser installer")
        self._assert_identity("omarchy install browser")
        self.run(
            "omarchy launch terminal",
            self._route_paths["omarchy install browser"], browser,
            timeout=8.0)
        self._assert_identity("omarchy install browser")

    def _capture_directory(self) -> Path:
        root = Path(os.path.abspath(self.capture_root.expanduser()))
        try:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if root.resolve(strict=True) != root:
                raise OmarchyUnavailableError("omarchy_capture_root_invalid")
            info = os.lstat(root)
            if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                    or info.st_uid != os.getuid()):
                raise OmarchyUnavailableError("omarchy_capture_root_invalid")
            os.chmod(root, 0o700)
        except OSError as exc:
            raise OmarchyUnavailableError("omarchy_capture_root_unavailable") from exc
        return root

    def screenshot(self) -> dict[str, Any]:
        root = self._capture_directory()
        environment = self._environment()
        environment["OMARCHY_SCREENSHOT_DIR"] = str(root)
        output = self.run(
            "omarchy capture screenshot", "fullscreen", "save",
            timeout=20.0, environment=environment)
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            raise OmarchyUnavailableError("omarchy_capture_not_created")
        candidate = Path(os.path.abspath(Path(lines[-1]).expanduser()))
        try:
            if candidate.parent != root or candidate.resolve(strict=True) != candidate:
                raise OmarchyUnavailableError("omarchy_capture_path_invalid")
            info = os.lstat(candidate)
            if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                    or info.st_uid != os.getuid()
                    or not 8 <= info.st_size <= 64 * 1024 * 1024):
                raise OmarchyUnavailableError("omarchy_capture_invalid")
            with candidate.open("rb") as handle:
                if handle.read(8) != _PNG_MAGIC:
                    raise OmarchyUnavailableError("omarchy_capture_invalid")
                handle.seek(0)
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        except OSError as exc:
            raise OmarchyUnavailableError("omarchy_capture_unavailable") from exc
        return {
            "path": str(candidate), "bytes": int(info.st_size),
            "sha256": digest,
        }


class OmarchyDesktopBroker:
    """Bind, execute, and re-observe a curated Omarchy control surface."""

    def __init__(
        self,
        backend: OmarchyDesktopBackend,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        poll_seconds: float = 0.05,
        action_timeout_seconds: float = 3.0,
    ):
        if poll_seconds <= 0 or action_timeout_seconds <= 0:
            raise ValueError("Omarchy action timeouts must be positive")
        self.backend = backend
        self.sleeper = sleeper
        self.poll_seconds = float(poll_seconds)
        self.action_timeout_seconds = float(action_timeout_seconds)

    def status(self) -> dict[str, Any]:
        return {
            "status": "ok", "verified": True, "platform": "omarchy",
            "version": self.backend.version(),
            "theme": {
                "current": self.backend.current_theme(),
                "available": self.backend.themes(),
            },
            "font": {
                "current": self.backend.current_font(),
                "available": self.backend.fonts(),
            },
            "nightlight_enabled": self.backend.nightlight_enabled(),
            "idle_mode": self.backend.idle_mode(),
            "brightness_percent": self.backend.brightness(),
            "locked": self.backend.locked(),
        }

    @staticmethod
    def _exact_keys(args: Mapping[str, Any], expected: set[str]) -> None:
        if set(args) != expected:
            raise ValueError("Omarchy arguments do not match the typed action")

    def _target(self, tool_name: str, args: Mapping[str, Any]) -> str:
        if tool_name == "machine_omarchy_set_theme":
            self._exact_keys(args, {"theme"})
            value = args.get("theme")
            if not isinstance(value, str):
                raise ValueError("theme must be an exact installed label")
            target = _bounded_label(value, field="theme")
            if target != value:
                raise ValueError("theme must be an exact installed label")
            if target not in self.backend.themes():
                raise ValueError("Omarchy theme is not installed")
            return target
        if tool_name == "machine_omarchy_set_font":
            self._exact_keys(args, {"font"})
            value = args.get("font")
            if not isinstance(value, str):
                raise ValueError("font must be an exact installed label")
            target = _bounded_label(value, field="font")
            if target != value:
                raise ValueError("font must be an exact installed label")
            if target not in self.backend.fonts():
                raise ValueError("Omarchy font is not installed")
            return target
        if tool_name == "machine_omarchy_set_nightlight":
            self._exact_keys(args, {"enabled"})
            if type(args.get("enabled")) is not bool:
                raise ValueError("enabled must be a boolean")
            return "enabled" if args["enabled"] else "disabled"
        if tool_name == "machine_omarchy_set_idle":
            self._exact_keys(args, {"mode"})
            target = str(args.get("mode") or "")
            if target not in {"stay_awake", "allow_idle"}:
                raise ValueError("invalid Omarchy idle mode")
            return target
        if tool_name == "machine_omarchy_set_brightness":
            self._exact_keys(args, {"percent"})
            value = args.get("percent")
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 100:
                raise ValueError("brightness percent must be between 1 and 100")
            return str(value)
        if tool_name == "machine_omarchy_take_screenshot":
            self._exact_keys(args, set())
            return "fullscreen_save"
        if tool_name == "machine_omarchy_lock":
            self._exact_keys(args, set())
            return "locked"
        if tool_name == "machine_omarchy_install_browser":
            self._exact_keys(args, {"browser"})
            if args.get("browser") != "firefox":
                raise ValueError("only the Firefox installer is supported")
            return "firefox"
        raise ValueError("unsupported Omarchy action")

    def binding_for_action(
        self, tool_name: str, args: Mapping[str, Any],
    ) -> OmarchyActionBinding:
        if tool_name not in OMARCHY_ACTION_TOOLS:
            raise ValueError("unsupported Omarchy action")
        target = self._target(tool_name, args)
        return OmarchyActionBinding(
            tool_name=tool_name, operation=_TOOL_OPERATION[tool_name],
            target=target,
            args_sha256=sha256_text(canonical_json(dict(args))),
            command_fingerprint=self.backend.command_fingerprint(tool_name),
            omarchy_version=self.backend.version(),
            bound_at_ns=time.time_ns(),
        )

    @staticmethod
    def approval_preview(binding: OmarchyActionBinding) -> dict[str, Any]:
        return {
            "platform": "Omarchy", "operation": binding.operation,
            "target": binding.target,
        }

    def _validate_binding(
        self, tool_name: str, args: Mapping[str, Any],
        expected_binding: OmarchyActionBinding | Mapping[str, Any],
    ) -> OmarchyActionBinding:
        binding = OmarchyActionBinding.model_validate(expected_binding)
        target = self._target(tool_name, args)
        if (binding.tool_name != tool_name
                or binding.operation != _TOOL_OPERATION.get(tool_name)
                or binding.target != target
                or binding.args_sha256 != sha256_text(canonical_json(dict(args)))
                or binding.command_fingerprint
                    != self.backend.command_fingerprint(tool_name)
                or binding.omarchy_version != self.backend.version()):
            raise OmarchyBindingError()
        return binding

    @staticmethod
    def _receipt(
        binding: OmarchyActionBinding, state: str, *, replay: bool,
    ) -> dict[str, Any]:
        return {
            "status": "ok", "verified": True,
            "operation": binding.operation, "target": binding.target,
            "state": state, "idempotent_replay": bool(replay),
        }

    def _current_state(self, binding: OmarchyActionBinding) -> str | None:
        if binding.operation == "set_theme":
            return self.backend.current_theme()
        if binding.operation == "set_font":
            return self.backend.current_font()
        if binding.operation == "set_nightlight":
            return "enabled" if self.backend.nightlight_enabled() else "disabled"
        if binding.operation == "set_idle":
            return self.backend.idle_mode()
        if binding.operation == "set_brightness":
            return str(self.backend.brightness())
        if binding.operation == "lock":
            return "locked" if self.backend.locked() else "unlocked"
        return None

    def execute(
        self, tool_name: str, args: Mapping[str, Any], *,
        expected_binding: OmarchyActionBinding | Mapping[str, Any],
    ) -> dict[str, Any]:
        binding = self._validate_binding(tool_name, args, expected_binding)
        if binding.operation == "take_screenshot":
            try:
                capture = self.backend.screenshot()
            except Exception as exc:
                raise OmarchyActionError() from exc
            return self._receipt(binding, "captured", replay=False) | {
                "capture": capture,
            }
        if binding.operation == "start_browser_installer":
            try:
                self.backend.start_browser_installer(binding.target)
            except Exception as exc:
                raise OmarchyActionError() from exc
            return self._receipt(binding, "installer_started", replay=False)
        current = self._current_state(binding)
        if current == binding.target:
            return self._receipt(binding, current, replay=True)
        try:
            if binding.operation == "set_theme":
                self.backend.set_theme(binding.target)
            elif binding.operation == "set_font":
                self.backend.set_font(binding.target)
            elif binding.operation == "set_nightlight":
                self.backend.toggle_nightlight()
            elif binding.operation == "set_idle":
                self.backend.set_idle(binding.target)
            elif binding.operation == "set_brightness":
                self.backend.set_brightness(int(binding.target))
            elif binding.operation == "lock":
                self.backend.lock()
            else:
                raise OmarchyBindingError()
            deadline = time.monotonic() + self.action_timeout_seconds
            while True:
                observed = self._current_state(binding)
                if observed == binding.target:
                    return self._receipt(binding, observed, replay=False)
                if time.monotonic() >= deadline:
                    raise OmarchyActionError()
                self.sleeper(self.poll_seconds)
        except OmarchyActionError:
            raise
        except Exception as exc:
            raise OmarchyActionError() from exc

    def reconciliation_receipt(
        self, expected_binding: OmarchyActionBinding | Mapping[str, Any],
    ) -> dict[str, Any] | None:
        binding = OmarchyActionBinding.model_validate(expected_binding)
        if binding.operation in {"take_screenshot", "start_browser_installer"}:
            return None
        if (binding.command_fingerprint
                != self.backend.command_fingerprint(binding.tool_name)
                or binding.omarchy_version != self.backend.version()):
            return None
        current = self._current_state(binding)
        return (self._receipt(binding, current, replay=True)
                if current == binding.target else None)

    def verify_receipt(
        self, tool_name: str, result: Any, args: Mapping[str, Any],
        idempotency_key: str | None,
        expected_binding: OmarchyActionBinding | Mapping[str, Any] | None = None,
    ) -> bool:
        try:
            value = json.loads(result) if isinstance(result, str) else result
            if tool_name == OMARCHY_STATUS_TOOL:
                return (idempotency_key is None and dict(args) == {}
                        and value == self.status())
            if (tool_name not in OMARCHY_ACTION_TOOLS
                    or not isinstance(idempotency_key, str)
                    or expected_binding is None or not isinstance(value, dict)):
                return False
            binding = self._validate_binding(
                tool_name, args, expected_binding)
            if binding.operation == "take_screenshot":
                capture = value.get("capture")
                if not isinstance(capture, dict):
                    return False
                path = Path(str(capture.get("path") or ""))
                if (not path.is_absolute() or not path.is_file()
                        or path.is_symlink()
                        or path.parent != Path(os.path.abspath(
                            self.backend.capture_root.expanduser()))):
                    return False
                info = os.lstat(path)
                if (info.st_uid != os.getuid()
                        or int(capture.get("bytes") or -1) != info.st_size):
                    return False
                with path.open("rb") as handle:
                    if handle.read(8) != _PNG_MAGIC:
                        return False
                    handle.seek(0)
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                expected = self._receipt(
                    binding, "captured", replay=False) | {
                        "capture": {
                            "path": str(path), "bytes": int(info.st_size),
                            "sha256": digest,
                        },
                    }
                return value == expected
            if binding.operation == "start_browser_installer":
                return value == self._receipt(
                    binding, "installer_started", replay=False)
            current = self._current_state(binding)
            replay = value.get("idempotent_replay")
            if type(replay) is not bool:
                return False
            return (current == binding.target
                    and value == self._receipt(
                        binding, current, replay=replay))
        except (OSError, OmarchyBrokerError, TypeError, ValueError,
                json.JSONDecodeError):
            return False
