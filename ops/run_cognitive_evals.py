#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.evals import CognitiveEvalRunner
from friday_core.graph import GraphStore
from friday_core.live_runtime import resolve_state_dir


result = CognitiveEvalRunner(
    GraphStore(resolve_state_dir(REPO) / "friday.db")).run(
        REPO / "evals" / "core-v2.json")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passed"] == result["total"] else 1)
