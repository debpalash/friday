#!/usr/bin/env python3
"""Create the pinned MLX inference environment on Apple Silicon.

The environment is a separate hash-locked virtual environment under
``<runtime root>/mlx``; Friday launches ``friday_core.mlx_server`` inside it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.engine_assets import mlx_runtime_pins  # noqa: E402
from friday_core.runtime_engines import runtime_root  # noqa: E402
from friday_host import fs  # noqa: E402
from friday_host.host import current_host  # noqa: E402
from friday_host.paths import venv_python  # noqa: E402

LOCK = REPO / "requirements" / "mlx-runtime.lock"


def install(root: Path, *, uv: str, python: str = "3.12") -> Path:
    host = current_host()
    if not (host.is_macos and host.arch == "aarch64"):
        raise SystemExit("the MLX runtime is only available on Apple Silicon macOS")
    target = root / "mlx"
    interpreter = venv_python(target, host)
    pins = mlx_runtime_pins()
    pin_file = target / "FRIDAY_ENGINE_PIN"
    expected = (f"engine=mlx-lm\nmlx={pins['mlx']}\nmlx-lm={pins['mlx-lm']}\n"
                f"lock={LOCK.name}\n")
    if interpreter.is_file() and pin_file.is_file() and pin_file.read_text() == expected:
        check = subprocess.run([str(interpreter), "-c", "import mlx.core, mlx_lm"],
                               capture_output=True, timeout=120)
        if check.returncode == 0:
            print(f"verified existing MLX runtime: {target}")
            return interpreter
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([uv, "venv", "--python", python, "--relocatable", str(target / "venv")],
                   check=True, timeout=600)
    subprocess.run([uv, "pip", "sync", "--python", str(interpreter), "--require-hashes",
                    str(LOCK)], check=True, timeout=1800)
    subprocess.run([str(interpreter), "-c", "import mlx.core, mlx_lm"], check=True,
                   timeout=300)
    pin_file.write_text(expected, encoding="utf-8")
    fs.chmod_private(pin_file, 0o600)
    print(f"installed MLX runtime: {target}")
    return interpreter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--uv", default=os.environ.get("FRIDAY_UV") or shutil.which("uv") or "uv")
    args = parser.parse_args()
    root = (args.runtime_root or runtime_root()).expanduser().resolve()
    install(root, uv=args.uv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
