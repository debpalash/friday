#!/usr/bin/env python3
"""Run Friday's local, artifact-backed voice qualification."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.asr import ParakeetASR
from friday_core.graph import GraphStore
from friday_core.live_runtime import (
    resolve_application_root,
    resolve_state_dir,
    runtime_environment,
)
from friday_core.speech import PiperSpeechSynthesizer
from friday_core.voice_evals import VoiceEvalRunner


def main() -> int:
    environment = runtime_environment()
    application_root = resolve_application_root(REPO, environment)
    state_dir = resolve_state_dir(REPO, environment)
    asr = ParakeetASR(
        application_root / "models"
        / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
    speech = PiperSpeechSynthesizer(application_root, output_rate=24_000)
    result = VoiceEvalRunner(
        GraphStore(state_dir / "friday.db"), asr, speech).run(
            REPO / "evals" / "voice-v1.json")
    print(json.dumps({
        "evaluation_run_id": result["evaluation_run_id"],
        "suite": result["suite"],
        "version": result["version"],
        "passed": result["passed"],
        "artifact_set_sha256": result["artifact_set_sha256"],
        "asr": result["asr"],
        "tts": result["tts"],
        "quality": result["quality"],
        "latency": result["latency"],
        "echo": result["echo"],
        "interruption": result["interruption"],
        "checks": result["checks"],
        "privacy": result["privacy"],
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
