#!/usr/bin/env python3
"""Run Friday's injected model, worker, browser, and filesystem recovery suite."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.graph import GraphStore
from friday_core.live_runtime import resolve_state_dir, runtime_environment
from friday_core.recovery_evals import RecoveryEvalRunner


async def run() -> int:
    state_dir = resolve_state_dir(REPO, runtime_environment())
    result = await RecoveryEvalRunner(
        GraphStore(state_dir / "friday.db")).run(
            REPO / "evals" / "recovery-v1.json")
    print(json.dumps({
        "evaluation_run_id": result["evaluation_run_id"],
        "suite": result["suite"],
        "version": result["version"],
        "passed": result["passed"],
        "scenarios": result["scenarios"],
        "metrics": result["metrics"],
        "checks": result["checks"],
        "privacy": result["privacy"],
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
