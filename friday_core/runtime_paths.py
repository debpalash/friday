"""Portable defaults for Friday's installed runtime paths."""

from __future__ import annotations

import os
from pathlib import Path


def default_install_root() -> Path:
    data_home = Path(os.environ.get(
        "XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return data_home.expanduser() / "friday"


def default_qwen_runtime() -> Path:
    return default_install_root() / "runtime" / "qwen"
