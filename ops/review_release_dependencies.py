#!/usr/bin/env python3
"""Verify exact installed locks and emit a private license inventory.

Every lock in the compliance policy is reviewed. Pass an interpreter for each
environment that is installed on this machine with ``--environment
NAME=PYTHON``; ``--app-python`` and ``--qwen-python`` remain as shorthands for
the host application environment and the vLLM runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.dependency_review import (  # noqa: E402
    run_dependency_review,
    write_private_review,
)


def _environment(value: str) -> tuple[str, Path]:
    name, separator, python = value.partition("=")
    if not separator or not name or not python:
        raise argparse.ArgumentTypeError("expected NAME=PYTHON")
    return name, Path(python)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--app-python", type=Path)
    parser.add_argument("--qwen-python", type=Path)
    parser.add_argument("--environment", type=_environment, action="append",
                        default=[], metavar="NAME=PYTHON")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (args.app_python or args.qwen_python or args.environment):
        parser.error("provide --app-python, --qwen-python, or --environment")
    result = run_dependency_review(
        REPO, environments=dict(args.environment),
        app_python=args.app_python, qwen_python=args.qwen_python)
    if args.output:
        write_private_review(args.output, result)
    summary = {
        "passed": result["passed"],
        "installed_reviewed": result["installed_reviewed"],
        "locks": {
            name: {"packages": value["locked_packages"],
                   "review": value["review"], "passed": value["passed"]}
            for name, value in result["environments"].items()},
        "models": len(result["models_and_assets"]),
        "binaries": len(result["binary_assets"]),
        "distribution_approval": result["distribution_approval"],
        "policy_sha256": result["policy_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
