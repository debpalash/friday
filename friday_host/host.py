"""Host platform detection.

``detect_host`` takes every probe as an argument so tests can describe a
macOS or Windows machine from Linux. ``current_host`` caches one detection per
process; production code never overrides it through the environment.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Mapping


_UV_PLATFORM_TAGS = {
    ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
    ("macos", "x86_64"): "x86_64-apple-darwin",
    ("macos", "aarch64"): "aarch64-apple-darwin",
    ("windows", "x86_64"): "x86_64-pc-windows-msvc",
    ("windows", "aarch64"): "aarch64-pc-windows-msvc",
}

_LOCK_ARCH = {"x86_64": "x86_64", "aarch64": "arm64"}


@dataclass(frozen=True)
class HostPlatform:
    """Facts about the machine Friday is running on."""

    os: str
    arch: str
    wsl: bool = False
    session: str = "headless"
    has_systemd_user: bool = False
    has_launchd: bool = False
    has_task_scheduler: bool = False
    has_hyprland: bool = False
    has_omarchy: bool = False
    has_bwrap: bool = False
    has_systemd_run: bool = False
    has_managed_chromium: bool = False
    has_dev_shm: bool = False
    has_nvidia_smi: bool = False
    has_pdftotext: bool = False
    has_tesseract: bool = False
    has_magick: bool = False

    def __post_init__(self) -> None:
        if self.os not in {"linux", "macos", "windows"}:
            raise ValueError(f"unsupported host os: {self.os}")
        if self.arch not in {"x86_64", "aarch64"}:
            raise ValueError(f"unsupported host arch: {self.arch}")

    @property
    def is_linux(self) -> bool:
        return self.os == "linux"

    @property
    def is_macos(self) -> bool:
        return self.os == "macos"

    @property
    def is_windows(self) -> bool:
        return self.os == "windows"

    @property
    def is_posix(self) -> bool:
        return self.os != "windows"

    @property
    def lock_id(self) -> str:
        """Identifier of the hash lock for this host, e.g. ``linux-x86_64``."""
        return f"{self.os}-{_LOCK_ARCH[self.arch]}"

    @property
    def python_platform_tag(self) -> str:
        """The ``uv --python-platform`` target for this host."""
        return _UV_PLATFORM_TAGS[(self.os, self.arch)]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["lock_id"] = self.lock_id
        value["python_platform_tag"] = self.python_platform_tag
        return value


def normalize_os(platform_name: str) -> str:
    name = platform_name.lower()
    if name.startswith("linux"):
        return "linux"
    if name.startswith("darwin"):
        return "macos"
    if name.startswith("win") or name == "cygwin" or name == "msys":
        return "windows"
    raise ValueError(f"unsupported operating system: {platform_name}")


def normalize_arch(machine: str) -> str:
    value = machine.lower()
    if value in {"x86_64", "amd64", "x64"}:
        return "x86_64"
    if value in {"aarch64", "arm64"}:
        return "aarch64"
    raise ValueError(f"unsupported architecture: {machine}")


def _wsl(read_text: Callable[[Path], str]) -> bool:
    try:
        return "microsoft" in read_text(Path("/proc/version")).lower()
    except OSError:
        return False


def _systemd_user_available(run: Callable[..., subprocess.CompletedProcess]) -> bool:
    try:
        result = run(
            ["systemctl", "--user", "show-environment"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _session(host_os: str, environment: Mapping[str, str]) -> str:
    if host_os == "linux":
        if environment.get("WAYLAND_DISPLAY"):
            return "wayland"
        if environment.get("DISPLAY"):
            return "x11"
        return "headless"
    if host_os == "macos":
        return "aqua"
    return "win32"


def detect_host(
    environment: Mapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
    machine: str | None = None,
    uid: int | None = None,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[Path], bool] = Path.exists,
    is_dir: Callable[[Path], bool] = Path.is_dir,
    is_file: Callable[[Path], bool] = Path.is_file,
    read_text: Callable[[Path], str] = Path.read_text,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> HostPlatform:
    """Describe the host from injectable probes."""
    environment = os.environ if environment is None else environment
    host_os = normalize_os(platform_name or sys.platform)
    arch = normalize_arch(machine or platform.machine())
    if uid is None and host_os != "windows":
        uid = os.getuid() if hasattr(os, "getuid") else 0
    runtime_dir = Path(f"/run/user/{uid}") if host_os == "linux" else None
    has_systemd_user = host_os == "linux" and _systemd_user_available(run)
    return HostPlatform(
        os=host_os,
        arch=arch,
        wsl=host_os == "linux" and _wsl(read_text),
        session=_session(host_os, environment),
        has_systemd_user=has_systemd_user,
        has_launchd=host_os == "macos" and which("launchctl") is not None,
        has_task_scheduler=(
            host_os == "windows" and which("schtasks") is not None),
        has_hyprland=(
            runtime_dir is not None and is_dir(runtime_dir / "hypr")),
        has_omarchy=(
            host_os == "linux"
            and exists(Path("/usr/share/omarchy/bin/omarchy"))),
        has_bwrap=host_os == "linux" and is_file(Path("/usr/bin/bwrap")),
        has_systemd_run=(
            host_os == "linux" and which("systemd-run") is not None),
        has_managed_chromium=(
            host_os == "linux"
            and is_file(Path("/usr/lib/chromium/chromium"))),
        has_dev_shm=host_os == "linux" and is_dir(Path("/dev/shm")),
        has_nvidia_smi=which("nvidia-smi") is not None,
        has_pdftotext=host_os == "linux" and is_file(Path("/usr/bin/pdftotext")),
        has_tesseract=host_os == "linux" and is_file(Path("/usr/bin/tesseract")),
        has_magick=host_os == "linux" and is_file(Path("/usr/bin/magick")),
    )


_CURRENT: HostPlatform | None = None


def current_host() -> HostPlatform:
    """Return the cached description of this process's host."""
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = detect_host()
    return _CURRENT


def _reset_host_cache() -> None:
    """Test hook: forget the cached host description."""
    global _CURRENT
    _CURRENT = None


__all__ = [
    "HostPlatform", "current_host", "detect_host", "normalize_arch",
    "normalize_os",
]
