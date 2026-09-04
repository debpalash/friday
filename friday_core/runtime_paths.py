"""Portable defaults for Friday's installed runtime paths.

The implementation lives in :mod:`friday_host.paths`; these names remain for
existing callers.
"""

from __future__ import annotations

from friday_host.paths import (default_config_root, default_install_root,
                               default_qwen_runtime, default_state_root)

__all__ = [
    "default_config_root", "default_install_root", "default_qwen_runtime",
    "default_state_root",
]
