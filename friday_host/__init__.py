"""Operating-system boundary for Friday.

``friday_host`` is deliberately standard-library only so the installer, the
uninstaller, ``friday doctor``, and the non-Linux command-line front end can
import it from a bare interpreter before any virtual environment exists.
``friday_core`` imports from this package; nothing here imports
``friday_core``.

Every Linux branch reproduces the expression Friday used before the port, so
behaviour on the original Linux target is unchanged.
"""

from __future__ import annotations

from .host import HostPlatform, current_host, detect_host
from . import paths

__all__ = ["HostPlatform", "current_host", "detect_host", "paths"]
