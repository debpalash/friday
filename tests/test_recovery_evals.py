from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from friday_core.graph import GraphStore
from friday_core.recovery_evals import RecoveryEvalRunner


def fixture_suite() -> dict:
    return {
        "name": "friday-injected-recovery",
        "version": 1,
        "gates": {
            "minimum_recovery_rate": 1.0,
            "maximum_control_path_p95_ms": 5_000,
            "maximum_model_retry_seconds": 30,
        },
    }


class RecoveryEvalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph = GraphStore(self.root / "results.db")
        self.suite = self.root / "suite.json"
        self.suite.write_text(json.dumps(fixture_suite()))

    async def asyncTearDown(self):
        self.temporary.cleanup()

    async def test_all_four_injected_failures_recover(self):
        result = await RecoveryEvalRunner(self.graph).run(self.suite)

        self.assertTrue(result["passed"])
        self.assertEqual(result["metrics"]["scenarios"], 4)
        self.assertEqual(result["metrics"]["recovered"], 4)
        self.assertEqual(result["metrics"]["recovery_rate"], 1.0)
        self.assertEqual(
            {item["name"] for item in result["scenarios"]},
            {"model", "worker", "browser", "filesystem"})
        self.assertTrue(all(item["recovered"] for item in result["scenarios"]))
        self.assertTrue(result["privacy"]["cleanup_verified"])
        self.assertEqual(self.graph.count_nodes("recovery_evaluation_run"), 1)

    async def test_suite_rejects_symlink_nonfinite_and_bad_gate(self):
        self.suite.write_text('{"value": NaN}')
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            RecoveryEvalRunner._load_suite(self.suite)

        invalid = fixture_suite()
        invalid["gates"]["minimum_recovery_rate"] = 1.1
        self.suite.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, "gate"):
            RecoveryEvalRunner._load_suite(self.suite)

        real = self.root / "real.json"
        real.write_text(json.dumps(fixture_suite()))
        self.suite.unlink()
        self.suite.symlink_to(real)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            RecoveryEvalRunner._load_suite(self.suite)

    async def test_production_suite_requires_every_scenario(self):
        production = Path(__file__).parents[1] / "evals" / "recovery-v1.json"
        suite, digest = RecoveryEvalRunner._load_suite(production)
        self.assertEqual(suite["gates"]["minimum_recovery_rate"], 1.0)
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
