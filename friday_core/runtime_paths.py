"""Portable defaults for Friday's installed runtime paths."""

from __future__ import annotations

import os
from pathlib import Path


def default_install_root() -> Path:
    data_home = Path(os.environ.get(
        "XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return data_home.expanduser() / "friday"


def default_config_root() -> Path:
    config_home = Path(os.environ.get(
        "XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return config_home.expanduser() / "friday"


def default_state_root() -> Path:
    state_home = Path(os.environ.get(
        "XDG_STATE_HOME", str(Path.home() / ".local" / "state")))
    return state_home.expanduser() / "friday"


def default_qwen_runtime() -> Path:
    return default_install_root() / "runtime" / "qwen"
