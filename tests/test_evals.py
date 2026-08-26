import json
import tempfile
import unittest
from pathlib import Path

from friday_core.evals import CognitiveEvalRunner
from friday_core.graph import GraphStore


ROOT = Path(__file__).resolve().parents[1]


class CapabilityEvalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph = GraphStore(self.root / "results.db")
        self.runner = CognitiveEvalRunner(self.graph)

    def tearDown(self):
        self.temporary.cleanup()

    def _suite(self, value: dict) -> Path:
        path = self.root / "suite.json"
        path.write_text(json.dumps(value))
        return path

    def test_v2_suite_passes_and_scenarios_do_not_contaminate_result_store(self):
        result = self.runner.run(ROOT / "evals" / "core-v2.json")

        self.assertEqual((result["passed"], result["total"]), (13, 13))
        self.assertEqual(result["version"], 2)
        self.assertIn("restart recovery", result["coverage"])
        self.assertEqual(self.graph.count("task_state"), 0)
        self.assertEqual(self.graph.count("claim_state"), 0)
        self.assertEqual(self.graph.count_nodes("evaluation_run"), 1)

    def test_unknown_scenario_and_duplicate_case_names_reject_before_recording(self):
        unknown = self._suite({
            "name": "invalid", "version": 1, "cases": [{
                "name": "case", "kind": "scenario",
                "scenario": "shell_command", "expected": {},
            }],
        })
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            self.runner.run(unknown)

        duplicate = self._suite({
            "name": "invalid", "version": 1, "cases": [{
                "name": "same", "kind": "intent", "text": "hello",
                "expected": "conversation",
            }, {
                "name": "same", "kind": "intent", "text": "hello",
                "expected": "conversation",
            }],
        })
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.runner.run(duplicate)
        self.assertEqual(self.graph.count_nodes("evaluation_run"), 0)

    def test_scenario_exception_is_a_failed_result_not_a_false_pass(self):
        suite = self._suite({
            "name": "failure containment", "version": 1, "cases": [{
                "name": "heterogeneous ranks", "kind": "scenario",
                "scenario": "hardware_tensor_parallel",
                "input": {"cuda_gib": [24, 48], "llm_devices": "0,1"},
                "expected": {"tensor_parallel_size": 2},
            }],
        })

        result = self.runner.run(suite)

        self.assertEqual((result["passed"], result["total"]), (0, 1))
        self.assertEqual(result["results"][0]["actual"], {
            "scenario_error": "ValueError",
        })
        self.assertEqual(self.graph.count_nodes("evaluation_run"), 1)

    def test_suite_symlink_and_nonfinite_fixture_fail_closed(self):
        target = self.root / "target.json"
        target.write_text(json.dumps({
            "name": "valid", "version": 1, "cases": [{
                "name": "hello", "kind": "intent", "text": "hello",
                "expected": "conversation",
            }],
        }))
        linked = self.root / "linked.json"
        linked.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "bounded regular file"):
            self.runner.run(linked)

        nonfinite = self.root / "nonfinite.json"
        nonfinite.write_text(
            '{"name":"bad","version":1,"cases":[{"name":"x",'
            '"kind":"intent","text":"hello","expected":NaN}]}')
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.runner.run(nonfinite)
        self.assertEqual(self.graph.count_nodes("evaluation_run"), 0)


if __name__ == "__main__":
    unittest.main()
