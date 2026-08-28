#!/usr/bin/env python3
"""Run all local release gates and write one private candidate report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.release_candidate import (
    ReleaseCandidateRunner,
    write_private_candidate_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--app-python", type=Path,
                        default=REPO / "venv" / "bin" / "python")
    parser.add_argument("--qwen-python", type=Path, required=True)
    args = parser.parse_args()
    report = ReleaseCandidateRunner(
        REPO, app_python=args.app_python,
        qwen_python=args.qwen_python).run()
    write_private_candidate_report(args.output, report)
    print(json.dumps({
        "local_gates_passed": report["local_gates_passed"],
        "candidate": report["candidate"],
        "report_payload_sha256": report["report_payload_sha256"],
        "report": str(args.output),
        "publication_performed": False,
    }, indent=2, sort_keys=True))
    return 0 if report["local_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
