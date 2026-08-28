from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from friday_core.graph import GraphStore
from friday_core.voice_evals import VoiceEvalRunner


class FixtureSpeech:
    backend_name = "fixture-speech"
    voice_name = "fixture"

    @staticmethod
    def synthesize(_text: str) -> np.ndarray:
        samples = np.arange(12_000, dtype=np.float32)
        return (0.2 * np.sin(samples * 2 * np.pi * 220 / 24_000)).astype(
            np.float32)


class FixtureASR:
    name = "fixture-asr"
    device = "cpu"

    def __init__(self, transcripts: list[str]):
        self.transcripts = iter(
            transcript
            for value in transcripts
            for transcript in (value, value)
        )

    def transcribe_file(self, path: Path) -> str:
        audio, rate = sf.read(path, dtype="float32")
        if rate != 24_000 or not audio.size:
            raise AssertionError("expected a real WAV artifact")
        return next(self.transcripts)


def fixture_suite() -> dict:
    utterances = [
        "alpha one", "beta two", "gamma three", "delta four", "echo five",
    ]
    return {
        "name": "friday-voice",
        "version": 1,
        "repetitions": 3,
        "gates": {
            "maximum_word_error_rate": 0,
            "minimum_exact_rate": 1,
            "maximum_asr_rtf_p95": 1,
            "maximum_tts_rtf_p95": 1,
            "maximum_first_audio_p95_ms": 5_000,
            "minimum_rms": 0.01,
            "maximum_clipped_ratio": 0,
            "minimum_barge_capture_ratio": 1,
        },
        "utterances": utterances,
        "reply": "ready now",
        "echo_tail_ms": 650,
        "barge_in_ms": 220,
    }


class VoiceEvalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.suite = self.root / "suite.json"
        self.suite.write_text(json.dumps(fixture_suite()))
        self.graph = GraphStore(self.root / "results.db")

    def tearDown(self):
        self.temporary.cleanup()

    def test_scorecard_measures_artifacts_latency_echo_and_interruption(self):
        suite = fixture_suite()
        result = VoiceEvalRunner(
            self.graph,
            FixtureASR(suite["utterances"] * suite["repetitions"]),
            FixtureSpeech()).run(
                self.suite)

        self.assertTrue(result["passed"])
        self.assertEqual(result["quality"]["word_error_rate"], 0)
        self.assertEqual(result["quality"]["utterances"], 15)
        self.assertEqual(result["echo"]["frames_forwarded_to_vad"], 0)
        self.assertTrue(result["interruption"]["captured_prefix_matches"])
        self.assertEqual(result["artifacts_retained"], 0)
        self.assertTrue(result["privacy"]["artifact_cleanup_verified"])
        self.assertEqual(self.graph.count_nodes("voice_evaluation_run"), 1)
        encoded = json.dumps(result)
        for utterance in suite["utterances"]:
            self.assertNotIn(utterance, encoded)

    def test_suite_rejects_symlinks_nonfinite_and_invalid_gates(self):
        self.suite.write_text('{"value": NaN}')
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            VoiceEvalRunner._load_suite(self.suite)

        invalid = fixture_suite()
        invalid["gates"]["minimum_exact_rate"] = 2
        self.suite.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, "quality gate"):
            VoiceEvalRunner._load_suite(self.suite)

        real = self.root / "real.json"
        real.write_text(json.dumps(fixture_suite()))
        self.suite.unlink()
        self.suite.symlink_to(real)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            VoiceEvalRunner._load_suite(self.suite)

    def test_production_suite_has_broad_voice_coverage(self):
        production = Path(__file__).parents[1] / "evals" / "voice-v1.json"
        suite, digest = VoiceEvalRunner._load_suite(production)
        self.assertGreaterEqual(len(suite["utterances"]), 8)
        self.assertEqual(len(digest), 64)
        self.assertGreaterEqual(suite["echo_tail_ms"], 650)


if __name__ == "__main__":
    unittest.main()
