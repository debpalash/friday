#!/usr/bin/env python3
"""Run Friday's isolated deterministic long-term-memory scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.graph import GraphStore
from friday_core.live_runtime import resolve_state_dir
from friday_core.memory_evals import MemoryEvalRunner


def main() -> int:
    state_dir = resolve_state_dir(REPO)
    result = MemoryEvalRunner(GraphStore(state_dir / "friday.db")).run(
        REPO / "evals" / "memory-retrieval-v1.json")
    print(json.dumps({
        "evaluation_run_id": result["evaluation_run_id"],
        "suite": result["suite"], "version": result["version"],
        "passed": result["passed"], "total": result["total"],
        "pass_rate": result["pass_rate"],
    }, indent=2))
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
