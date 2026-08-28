#!/usr/bin/env python3
"""Run Friday's seven-scenario adversarial boundary scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.adversarial_evals import AdversarialEvalRunner
from friday_core.graph import GraphStore
from friday_core.live_runtime import resolve_state_dir, runtime_environment


def main() -> int:
    state = resolve_state_dir(REPO, runtime_environment())
    result = AdversarialEvalRunner(
        GraphStore(state / "friday.db"), REPO).run(
            REPO / "evals" / "adversarial-v1.json")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
