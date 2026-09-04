#!/usr/bin/env python3
"""Owner-facing private data lifecycle commands."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.data_lifecycle import (
    delete_private_data,
    export_private_data,
    plan_private_deletion,
    verify_private_export,
)
from friday_core.live_runtime import resolve_state_dir


def main() -> int:
    parser = argparse.ArgumentParser(prog="friday data")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="override Friday's installed state directory",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export")
    export_parser.add_argument("target", type=Path)
    verify_parser = commands.add_parser("verify-export")
    verify_parser.add_argument("target", type=Path)
    delete_parser = commands.add_parser("delete")
    delete_parser.add_argument(
        "scope",
        choices=(
            "conversation", "task", "artifact", "memory_claim", "time_range",
        ),
    )
    delete_parser.add_argument("identifier", nargs="?")
    delete_parser.add_argument("--start")
    delete_parser.add_argument("--end")
    delete_parser.add_argument(
        "--confirm",
        action="store_true",
        help="apply the previewed deletion to a stopped Friday database",
    )
    args = parser.parse_args()

    try:
        if args.command == "export":
            state = args.state_dir or resolve_state_dir(REPO)
            result = export_private_data(state / "friday.db", args.target)
        elif args.command == "verify-export":
            result = verify_private_export(args.target)
        else:
            state = args.state_dir or resolve_state_dir(REPO)
            options = {
                "value": args.identifier,
                "start": args.start,
                "end": args.end,
            }
            if not args.confirm:
                result = {
                    "status": "preview",
                    **plan_private_deletion(
                        state / "friday.db", args.scope, **options),
                }
            else:
                from friday_host.service import backend_for

                if backend_for().is_active():
                    raise RuntimeError(
                        "run 'friday stop' before confirming selective deletion")
                result = delete_private_data(
                    state / "friday.db", args.scope,
                    runtime_stopped=True, **options,
                )
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"Friday data command failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
