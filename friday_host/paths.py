"""Per-operating-system default locations for Friday's installed files.

Precedence on every platform: an explicit ``FRIDAY_*`` variable, then the XDG
variable (honoured everywhere so one test harness can drive every layout),
then the platform default. The Linux defaults are exactly the values Friday
used before the port.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from .host import HostPlatform, current_host


def _home(environment: Mapping[str, str]) -> Path:
    for name in ("HOME", "USERPROFILE"):
        value = environment.get(name)
        if value:
            return Path(value).expanduser()
    return Path.home()


def _resolve(
    environment: Mapping[str, str] | None,
    host: HostPlatform | None,
    *,
    friday_variable: str,
    xdg_variable: str,
    xdg_default: tuple[str, ...],
    macos: tuple[str, ...],
    windows_variable: str,
    windows: tuple[str, ...],
    macos_suffix: str = "",
    windows_suffix: str = "",
) -> Path:
    environment = os.environ if environment is None else environment
    host = host or current_host()
    explicit = environment.get(friday_variable)
    if explicit:
        return Path(explicit).expanduser()
    home = _home(environment)
    if host.is_linux:
        base = Path(environment.get(
            xdg_variable, str(home.joinpath(*xdg_default))))
        return base.expanduser() / "friday"
    xdg = environment.get(xdg_variable)
    if xdg:
        return Path(xdg).expanduser() / "friday"
    if host.is_macos:
        root = home.joinpath(*macos)
        return root / macos_suffix if macos_suffix else root
    local = environment.get(windows_variable)
    base = Path(local).expanduser() if local else home / "AppData" / "Local"
    root = base.joinpath(*windows)
    return root / windows_suffix if windows_suffix else root


def default_install_root(
    environment: Mapping[str, str] | None = None,
    host: HostPlatform | None = None,
) -> Path:
    return _resolve(
        environment, host,
        friday_variable="FRIDAY_INSTALL_ROOT",
        xdg_variable="XDG_DATA_HOME", xdg_default=(".local", "share"),
        macos=("Library", "Application Support", "Friday"),
        windows_variable="LOCALAPPDATA", windows=("Friday",),
        macos_suffix="app", windows_suffix="app")


def default_state_root(
    environment: Mapping[str, str] | None = None,
    host: HostPlatform | None = None,
) -> Path:
    return _resolve(
        environment, host,
        friday_variable="FRIDAY_STATE_ROOT",
        xdg_variable="XDG_STATE_HOME", xdg_default=(".local", "state"),
        macos=("Library", "Application Support", "Friday"),
        windows_variable="LOCALAPPDATA", windows=("Friday",),
        macos_suffix="state", windows_suffix="state")


def default_config_root(
    environment: Mapping[str, str] | None = None,
    host: HostPlatform | None = None,
) -> Path:
    return _resolve(
        environment, host,
        friday_variable="FRIDAY_CONFIG_ROOT",
        xdg_variable="XDG_CONFIG_HOME", xdg_default=(".config",),
        macos=("Library", "Application Support", "Friday"),
        windows_variable="LOCALAPPDATA", windows=("Friday",),
        macos_suffix="config", windows_suffix="config")


def default_cache_root(
    environment: Mapping[str, str] | None = None,
    host: HostPlatform | None = None,
) -> Path:
    return _resolve(
        environment, host,
        friday_variable="FRIDAY_CACHE_ROOT",
        xdg_variable="XDG_CACHE_HOME", xdg_default=(".cache",),
        macos=("Library", "Caches", "Friday"),
        windows_variable="LOCALAPPDATA", windows=("Friday",),
        windows_suffix="cache")


def default_log_root(
    environment: Mapping[str, str] | None = None,
    host: HostPlatform | None = None,
) -> Path:
    environment = os.environ if environment is None else environment
    host = host or current_host()
    explicit = environment.get("FRIDAY_LOG_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    if host.is_macos and not environment.get("XDG_STATE_HOME"):
        return _home(environment) / "Library" / "Logs" / "Friday"
    return default_state_root(environment, host) / "logs"


def default_bin_root(
    environment: Mapping[str, str] | None = None,
    host: HostPlatform | None = None,
) -> Path:
    environment = os.environ if environment is None else environment
    host = host or current_host()
    explicit = environment.get("FRIDAY_BIN_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    xdg = environment.get("XDG_BIN_HOME")
    if xdg:
        return Path(xdg).expanduser()
    if host.is_windows:
        return default_install_root(environment, host).parent / "bin"
    return _home(environment) / ".local" / "bin"


def default_qwen_runtime(
    environment: Mapping[str, str] | None = None,
    host: HostPlatform | None = None,
) -> Path:
    return default_install_root(environment, host) / "runtime" / "qwen"


def venv_python(release_dir: Path, host: HostPlatform | None = None) -> Path:
    """Interpreter path inside a release's virtual environment."""
    host = host or current_host()
    if host.is_windows:
        return release_dir / "venv" / "Scripts" / "python.exe"
    return release_dir / "venv" / "bin" / "python"


def venv_pythonw(release_dir: Path, host: HostPlatform | None = None) -> Path:
    """Console-less interpreter on Windows; the normal one elsewhere."""
    host = host or current_host()
    if host.is_windows:
        return release_dir / "venv" / "Scripts" / "pythonw.exe"
    return venv_python(release_dir, host)


__all__ = [
    "default_bin_root", "default_cache_root", "default_config_root",
    "default_install_root", "default_log_root", "default_qwen_runtime",
    "default_state_root", "venv_python", "venv_pythonw",
]
