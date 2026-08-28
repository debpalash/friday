from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from friday_core.controller_browser_evals import ControllerBrowserEvalRunner
from friday_core.graph import GraphStore


def fixture_suite() -> dict:
    return {
        "name": "friday-controller-browser",
        "version": 1,
        "origin": "https://192.0.2.10:8500",
        "transport_binding_sha256": "a" * 64,
        "page_url": "https://93.184.216.34/form",
        "selector": "#query",
        "input": "controller-browser-unit-private-sentinel",
        "controller_label": "Evaluation controller",
    }


class ControllerBrowserEvalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph = GraphStore(self.root / "results.db")
        self.suite = self.root / "suite.json"
        self.suite.write_text(json.dumps(fixture_suite()))

    def tearDown(self):
        self.temporary.cleanup()

    def test_signed_workflow_rejects_uses_reconnects_and_revokes(self):
        result = ControllerBrowserEvalRunner(self.graph).run(self.suite)

        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["approval"]["rejected_actions_executed"], 0)
        self.assertEqual(result["approval"]["approved_effect_uses"], 1)
        self.assertEqual(result["browser"]["managed_runtime_checks"], 2)
        self.assertEqual(result["browser"]["mutations"], 1)
        self.assertEqual(result["revocation"]["active_sessions"], 0)
        self.assertEqual(
            self.graph.count_nodes("controller_browser_evaluation_run"), 1)
        with self.graph._connect() as connection:
            dump = "\n".join(connection.iterdump())
        self.assertNotIn(fixture_suite()["input"], dump)

    def test_suite_rejects_symlink_nonfinite_and_bad_binding(self):
        self.suite.write_text('{"value": NaN}')
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            ControllerBrowserEvalRunner._load_suite(self.suite)

        invalid = fixture_suite()
        invalid["transport_binding_sha256"] = "bad"
        self.suite.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, "metadata"):
            ControllerBrowserEvalRunner._load_suite(self.suite)

        real = self.root / "real.json"
        real.write_text(json.dumps(fixture_suite()))
        self.suite.unlink()
        self.suite.symlink_to(real)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            ControllerBrowserEvalRunner._load_suite(self.suite)

    def test_production_suite_has_public_fixture_and_private_input(self):
        production = (
            Path(__file__).parents[1] / "evals" / "controller-browser-v1.json")
        suite, digest = ControllerBrowserEvalRunner._load_suite(production)
        self.assertTrue(suite["page_url"].startswith("https://"))
        self.assertGreater(len(suite["input"]), 20)
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
