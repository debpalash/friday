from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from friday_core.graph import GraphStore
from friday_core.project_evals import ProjectEvalRunner


def fixture_suite() -> dict:
    return {
        "name": "friday-long-horizon-project",
        "version": 1,
        "gates": {
            "minimum_files": 3,
            "minimum_tests": 2,
            "maximum_recovery_ms": 5_000,
        },
        "files": [
            {"path": "subject.py", "content": "def value():\n    return 7\n"},
            {"path": "test_subject.py", "content": (
                "import unittest\nfrom subject import value\n\n"
                "class SubjectTests(unittest.TestCase):\n"
                "    def test_value(self):\n        self.assertEqual(value(), 7)\n"
                "    def test_stable(self):\n        self.assertEqual(value(), value())\n")},
            {"path": "README.md", "content": "project-eval-private-sentinel\n"},
        ],
    }


class ProjectEvalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph = GraphStore(self.root / "results.db")
        self.suite = self.root / "suite.json"
        self.suite.write_text(json.dumps(fixture_suite()))

    async def asyncTearDown(self):
        self.temporary.cleanup()

    async def test_project_is_recovered_and_graded_from_receipts(self):
        result = await ProjectEvalRunner(self.graph).run(self.suite)

        self.assertTrue(result["passed"])
        self.assertEqual(result["files"]["matched"], 3)
        self.assertEqual(result["tests"]["tests"], 2)
        self.assertEqual(result["recovery"]["interrupted_step_attempts"], 2)
        self.assertEqual(result["receipts"]["verified"], 5)
        self.assertEqual(result["user_visible_outcome"], {
            "status": "completed", "files_verified": 3,
            "tests_passed": 2, "recovered": True,
        })
        self.assertTrue(result["privacy"]["cleanup_verified"])
        self.assertEqual(
            self.graph.count_nodes("long_horizon_project_evaluation_run"), 1)
        with self.graph._connect() as connection:
            dump = "\n".join(connection.iterdump())
        self.assertNotIn("project-eval-private-sentinel", dump)

    async def test_suite_rejects_traversal_duplicates_nonfinite_and_symlink(self):
        for mutate, message in (
            (lambda value: value["files"][0].update(path="../escape"), "path"),
            (lambda value: value["files"][1].update(path=value["files"][0]["path"]),
             "file"),
        ):
            value = fixture_suite()
            mutate(value)
            self.suite.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, message):
                ProjectEvalRunner._load_suite(self.suite)

        self.suite.write_text('{"value": NaN}')
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            ProjectEvalRunner._load_suite(self.suite)

        real = self.root / "real.json"
        real.write_text(json.dumps(fixture_suite()))
        self.suite.unlink()
        self.suite.symlink_to(real)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            ProjectEvalRunner._load_suite(self.suite)

    async def test_production_suite_has_files_and_tests(self):
        production = (
            Path(__file__).parents[1]
            / "evals" / "long-horizon-project-v1.json")
        suite, digest = ProjectEvalRunner._load_suite(production)
        self.assertGreaterEqual(len(suite["files"]), 3)
        self.assertGreaterEqual(suite["gates"]["minimum_tests"], 3)
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
