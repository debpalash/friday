#!/usr/bin/env python3
"""Launch the supervisor with Friday's private environment file applied.

systemd applies ``friday.env`` through ``EnvironmentFile=``; launchd and Task
Scheduler have no equivalent, so this launcher reads the file, changes to the
release directory, and replaces itself with the supervisor. It is standard
library only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_host.envfile import read_env_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--chdir", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="script and arguments to run under the environment")
    args = parser.parse_args(argv)
    command = [item for item in args.command if item != "--"] if args.command else []
    if not command:
        parser.error("a command is required after --")
    values = read_env_file(args.env_file)
    for key, value in values.items():
        os.environ.setdefault(key, value)
    if args.chdir is not None:
        os.chdir(args.chdir)
    python = sys.executable
    if os.name == "posix":
        os.execv(python, [python, *command])
    import subprocess  # noqa: PLC0415

    return subprocess.call([python, *command])


if __name__ == "__main__":
    raise SystemExit(main())
