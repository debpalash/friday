#!/usr/bin/env python3
"""Verify exact installed locks and emit a private license inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.dependency_review import (
    run_dependency_review,
    write_private_review,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-python", type=Path, required=True)
    parser.add_argument("--qwen-python", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_dependency_review(
        REPO, app_python=args.app_python, qwen_python=args.qwen_python)
    if args.output:
        write_private_review(args.output, result)
    summary = {
        "passed": result["passed"],
        "application_packages": result["environments"]["application"][
            "locked_packages"],
        "qwen_runtime_packages": result["environments"]["qwen_runtime"][
            "locked_packages"],
        "models": len(result["models_and_assets"]),
        "distribution_approval": result["distribution_approval"],
        "policy_sha256": result["policy_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
