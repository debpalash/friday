#!/usr/bin/env python3
"""Remove Friday's app and runtime while preserving personal data.

This is the portable counterpart of ``scripts/uninstall.sh``. Without
``--purge`` it removes the releases, the ``current`` link, the bootstrapped
tools, the service registration, the CLI shim, and the launcher entries, and
keeps the state, configuration, and shared models. ``--purge`` removes the
exact Friday roots as well.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_host import paths  # noqa: E402
from friday_host.host import current_host  # noqa: E402
from friday_host.service import backend_for  # noqa: E402


def safe_root(value: Path, home: Path) -> Path:
    resolved = Path(os.path.abspath(value))
    if (not resolved.is_absolute() or resolved == Path(resolved.anchor)
            or resolved == home or resolved == home.parent):
        raise SystemExit(f"Refusing unsafe uninstall root: {resolved}")
    if resolved.is_symlink():
        raise SystemExit(f"Refusing symlink uninstall root: {resolved}")
    return resolved


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--purge", action="store_true",
                        help="also remove Friday's exact state, models, and configuration roots")
    args = parser.parse_args(argv)
    host = current_host()
    home = Path.home()
    install_root = safe_root(paths.default_install_root(), home)
    state_root = safe_root(Path(os.environ.get("FRIDAY_STATE_DIR")
                                or paths.default_state_root()), home)
    config_root = safe_root(paths.default_config_root(), home)
    bin_root = paths.default_bin_root()

    backend = backend_for(host)
    backend.uninstall()
    for shim in (bin_root / "friday", bin_root / "friday.cmd"):
        _remove(shim)
    if host.is_linux:
        data_home = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
        _remove(data_home / "applications" / "friday.desktop")
        _remove(data_home / "icons" / "hicolor" / "scalable" / "apps" / "friday.svg")

    if args.purge:
        for root in (install_root, state_root, config_root):
            _remove(root)
        try:
            from friday_host.host_keyring import SecretStore  # noqa: PLC0415

            SecretStore(host=host).delete("corrected-audio")
        except Exception:
            pass
        print("Friday and its personal data were permanently removed.")
        return 0
    for relative in ("releases", "current", "tools", "runtime/qwen/venv",
                     "runtime/mlx", "runtime/llama-server"):
        _remove(install_root / relative)
    print("Friday was uninstalled. Personal state, configuration, and models were preserved.")
    print(f"Preserved: {state_root}, {config_root}, and {install_root / 'shared'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
