from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from friday_core.live_assistant_evals import LiveAssistantEvalRunner


FINGERPRINT = "a" * 64


class LiveAssistantEvalTests(unittest.TestCase):
    def _suite(self, root: Path, cases: list[dict]) -> Path:
        path = root / "suite.json"
        path.write_text(json.dumps({
            "name": "fixture", "version": 1,
            "coverage": ["transport", "judgment"], "cases": cases,
        }))
        return path

    @staticmethod
    def _turn(prompt: str = "Question?") -> dict:
        return {
            "prompt": prompt, "min_words": 1, "max_words": 20,
            "required_events": ["you", "friday", "done"],
            "forbidden_events": ["progress"],
        }

    def test_runner_grades_output_transport_and_hidden_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            turn = self._turn()
            turn.update({
                "required_any": [["verified"]], "exact_url_count": 1,
                "progress_cursor_must_advance": True,
            })
            suite = self._suite(root, [{
                "name": "case", "mode": "text", "turns": [turn],
            }])

            result = LiveAssistantEvalRunner(
                lambda _case: [{
                    "output": "Verified: https://example.com",
                    "events": ["you", "sources", "friday", "done"],
                    "progress_cursor_advanced": True,
                }], runtime_fingerprint=FINGERPRINT).run(suite)

        self.assertEqual((result["passed"], result["total"]), (1, 1))
        checks = result["results"][0]["turns"][0]["checks"]
        self.assertTrue(checks["transport_envelope"])
        self.assertTrue(checks["progress_cursor"])
        self.assertNotIn("Verified", json.dumps(result))

    def test_failure_is_contained_and_successor_still_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = self._suite(root, [
                {"name": "broken", "mode": "text",
                 "turns": [self._turn("Broken?")]},
                {"name": "good", "mode": "text",
                 "turns": [self._turn("Good?")]},
            ])
            calls = []

            def run(case):
                calls.append(case["name"])
                if case["name"] == "broken":
                    raise RuntimeError("contained")
                return [{
                    "output": "Good.",
                    "events": ["you", "friday", "done"],
                    "progress_cursor_advanced": False,
                }]

            result = LiveAssistantEvalRunner(
                run, runtime_fingerprint=FINGERPRINT).run(suite)

        self.assertEqual(calls, ["broken", "good"])
        self.assertEqual((result["passed"], result["total"]), (1, 2))
        self.assertEqual(result["results"][0]["failure"], "RuntimeError")

    def test_suite_rejects_symlink_nonfinite_and_invalid_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._suite(root, [{
                "name": "case", "mode": "text", "turns": [self._turn()],
            }])
            linked = root / "linked.json"
            linked.symlink_to(valid)
            runner = LiveAssistantEvalRunner(
                lambda _case: [], runtime_fingerprint=FINGERPRINT)
            with self.assertRaisesRegex(ValueError, "bounded regular"):
                runner.run(linked)
            valid.write_text('{"value": NaN}')
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                runner.run(valid)
            invalid = self._suite(root, [{
                "name": "case", "mode": "voice", "turns": [self._turn()],
            }])
            with self.assertRaisesRegex(ValueError, "metadata"):
                runner.run(invalid)

    def test_production_suite_covers_real_server_boundaries(self):
        suite = LiveAssistantEvalRunner._read_suite(
            Path(__file__).parents[1] / "evals" / "live-assistant-v1.json")
        self.assertGreaterEqual(len(suite["cases"]), 7)
        coverage = set(suite["coverage"])
        self.assertIn("deployed transport", coverage)
        self.assertIn("capability truthfulness", coverage)
        self.assertIn("live news and source follow-up", coverage)

    def test_general_suite_covers_cross_domain_assistant_boundaries(self):
        suite = LiveAssistantEvalRunner._read_suite(
            Path(__file__).parents[1] / "evals" / "general-assistant-v1.json")
        self.assertGreaterEqual(len(suite["cases"]), 12)
        coverage = set(suite["coverage"])
        self.assertIn("deductive reasoning", coverage)
        self.assertIn("uncertainty calibration", coverage)
        self.assertIn("live multi-source research", coverage)
        self.assertIn("runtime voice control", coverage)


if __name__ == "__main__":
    unittest.main()
