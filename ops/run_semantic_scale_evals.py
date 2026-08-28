#!/usr/bin/env python3
"""Run Friday's held-out semantic-memory scale qualification."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.embeddings import configured_local_embedder
from friday_core.graph import GraphStore
from friday_core.live_runtime import (
    resolve_application_root,
    resolve_state_dir,
    runtime_environment,
)
from friday_core.semantic_scale_evals import SemanticScaleEvalRunner


def main() -> int:
    environment = runtime_environment()
    application_root = resolve_application_root(REPO, environment)
    state_dir = resolve_state_dir(REPO, environment)
    embedder = configured_local_embedder(application_root, environment)
    if embedder is None:
        raise RuntimeError(
            "pinned embedding model is not installed; run "
            "ops/install_embedding_model.py")
    result = SemanticScaleEvalRunner(
        GraphStore(state_dir / "friday.db"), embedder).run(
            REPO / "evals" / "semantic-scale-v1.json")
    print(json.dumps({
        "evaluation_run_id": result["evaluation_run_id"],
        "suite": result["suite"],
        "version": result["version"],
        "passed": result["passed"],
        "embedding_fingerprint": result["embedding_fingerprint"],
        "scale": result["scale"],
        "quality": result["quality"],
        "correction": result["correction"],
        "checks": result["checks"],
        "decision": result["decision"],
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
