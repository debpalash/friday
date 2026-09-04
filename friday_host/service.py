"""Service lifecycle backends: systemd user units and launchd agents.

The Linux backend issues exactly the ``systemctl --user`` and ``journalctl``
commands the bash tooling used before the port. The macOS backend manages a
LaunchAgent that runs the supervisor through ``ops/friday_launch.py``.
Windows Task Scheduler support arrives with the Windows phase.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .host import HostPlatform, current_host

Runner = Callable[..., subprocess.CompletedProcess]
LABEL = "dev.palash.friday"
UNIT = "friday.service"
LEGACY_UNIT = "friday-supervisor.service"


@dataclass(frozen=True)
class ServiceSpec:
    current_dir: Path
    env_file: Path
    python: Path
    supervisor: Path
    launcher: Path
    log_dir: Path
    label: str = LABEL
    unit: str = UNIT


class ServiceBackend(Protocol):
    kind: str

    def install(self, spec: ServiceSpec, template: Path) -> None: ...
    def uninstall(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def restart(self) -> None: ...
    def is_active(self) -> bool: ...
    def is_enabled(self) -> bool: ...
    def enable(self) -> None: ...
    def disable(self) -> None: ...
    def status_text(self) -> str: ...
    def log_command(self, *, follow: bool, lines: int = 40) -> list[str]: ...
    def unit_paths(self) -> tuple[Path, ...]: ...


def _write_private(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)


class SystemdUserBackend:
    kind = "systemd"

    def __init__(self, *, home: Path | None = None, runner: Runner = subprocess.run,
                 unit: str = UNIT) -> None:
        self.home = home or Path.home()
        self._run = runner
        self.unit = unit

    @property
    def unit_path(self) -> Path:
        return self.home / ".config" / "systemd" / "user" / self.unit

    def unit_paths(self) -> tuple[Path, ...]:
        return (self.unit_path,)

    def _systemctl(self, *arguments: str, check: bool = False, quiet: bool = True,
                   capture: bool = False) -> subprocess.CompletedProcess:
        options: dict[str, object] = {"check": check, "timeout": 120}
        if capture:
            options.update(text=True, capture_output=True)
        elif quiet:
            options.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self._run(["systemctl", "--user", *arguments], **options)

    def install(self, spec: ServiceSpec, template: Path) -> None:
        body = (template.read_text(encoding="utf-8")
                .replace("@CURRENT@", str(spec.current_dir))
                .replace("@ENV_FILE@", str(spec.env_file)))
        _write_private(self.unit_path, body)
        self._systemctl("daemon-reload", check=True)
        self.enable()

    def uninstall(self) -> None:
        self._systemctl("stop", self.unit)
        self._systemctl("disable", self.unit)
        try:
            self.unit_path.unlink()
        except FileNotFoundError:
            pass
        self._systemctl("daemon-reload")

    def start(self) -> None:
        self._systemctl("start", self.unit, check=True, quiet=False)

    def stop(self) -> None:
        self._systemctl("stop", self.unit, check=True, quiet=False)

    def restart(self) -> None:
        self._systemctl("restart", self.unit, check=True, quiet=False)

    def is_active(self) -> bool:
        return self._systemctl("is-active", "--quiet", self.unit).returncode == 0

    def is_enabled(self) -> bool:
        return self._systemctl("is-enabled", "--quiet", self.unit).returncode == 0

    def enable(self) -> None:
        self._systemctl("enable", self.unit, check=True)

    def disable(self) -> None:
        self._systemctl("disable", self.unit)

    def status_text(self) -> str:
        result = self._systemctl("status", self.unit, "--no-pager", capture=True)
        return str(result.stdout or result.stderr or "")

    def log_command(self, *, follow: bool, lines: int = 40) -> list[str]:
        command = ["journalctl", "--user", "-u", self.unit]
        return command + (["-f"] if follow else ["-n", str(lines), "--no-pager"])


class LaunchdBackend:
    kind = "launchd"

    def __init__(self, *, home: Path | None = None, runner: Runner = subprocess.run,
                 uid: int | None = None, label: str = LABEL) -> None:
        self.home = home or Path.home()
        self._run = runner
        self.uid = os.getuid() if uid is None else uid
        self.label = label
        self._log_dir: Path | None = None

    @property
    def plist_path(self) -> Path:
        return self.home / "Library" / "LaunchAgents" / f"{self.label}.plist"

    @property
    def target(self) -> str:
        return f"gui/{self.uid}/{self.label}"

    def unit_paths(self) -> tuple[Path, ...]:
        return (self.plist_path,)

    def _launchctl(self, *arguments: str, check: bool = False,
                   capture: bool = False) -> subprocess.CompletedProcess:
        options: dict[str, object] = {"check": check, "timeout": 120}
        if capture:
            options.update(text=True, capture_output=True)
        else:
            options.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self._run(["launchctl", *arguments], **options)

    def install(self, spec: ServiceSpec, template: Path) -> None:
        body = (template.read_text(encoding="utf-8")
                .replace("@LABEL@", spec.label)
                .replace("@PYTHON@", str(spec.python))
                .replace("@LAUNCHER@", str(spec.launcher))
                .replace("@SUPERVISOR@", str(spec.supervisor))
                .replace("@ENV_FILE@", str(spec.env_file))
                .replace("@CURRENT@", str(spec.current_dir))
                .replace("@LOG_DIR@", str(spec.log_dir)))
        spec.log_dir.mkdir(parents=True, exist_ok=True)
        _write_private(self.plist_path, body)
        self._log_dir = spec.log_dir
        self._launchctl("bootout", self.target)
        self._launchctl("bootstrap", f"gui/{self.uid}", str(self.plist_path), check=True)
        self.enable()

    def uninstall(self) -> None:
        self._launchctl("bootout", self.target)
        try:
            self.plist_path.unlink()
        except FileNotFoundError:
            pass

    def _loaded(self) -> bool:
        return self._launchctl("print", self.target).returncode == 0

    def start(self) -> None:
        if not self._loaded():
            if not self.plist_path.is_file():
                raise RuntimeError("Friday's launch agent is not installed")
            self._launchctl("bootstrap", f"gui/{self.uid}", str(self.plist_path), check=True)
        self._launchctl("kickstart", self.target, check=True)

    def stop(self) -> None:
        self._launchctl("bootout", self.target)

    def restart(self) -> None:
        if self._loaded():
            self._launchctl("kickstart", "-k", self.target, check=True)
        else:
            self.start()

    def is_active(self) -> bool:
        result = self._launchctl("print", self.target, capture=True)
        return result.returncode == 0 and "state = running" in str(result.stdout)

    def is_enabled(self) -> bool:
        return self.plist_path.is_file()

    def enable(self) -> None:
        self._launchctl("enable", self.target)

    def disable(self) -> None:
        self._launchctl("disable", self.target)

    def status_text(self) -> str:
        result = self._launchctl("print", self.target, capture=True)
        if result.returncode != 0:
            return "launch agent is not loaded"
        return "\n".join(str(result.stdout).splitlines()[:25])

    def log_command(self, *, follow: bool, lines: int = 40) -> list[str]:
        log_dir = self._log_dir or (self.home / "Library" / "Logs" / "Friday")
        command = ["tail", "-n", str(lines)]
        return command + (["-f"] if follow else []) + [str(log_dir / "supervisor.log")]


def backend_for(host: HostPlatform | None = None, *, runner: Runner = subprocess.run,
                home: Path | None = None) -> ServiceBackend:
    host = host or current_host()
    if host.is_linux:
        return SystemdUserBackend(home=home, runner=runner)
    if host.is_macos:
        return LaunchdBackend(home=home, runner=runner)
    raise NotImplementedError(
        "the Windows service lifecycle arrives with the Windows phase of the port")


__all__ = ["LABEL", "LEGACY_UNIT", "LaunchdBackend", "ServiceBackend",
           "ServiceSpec", "SystemdUserBackend", "UNIT", "backend_for"]
