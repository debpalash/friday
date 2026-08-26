import json
import tempfile
import unittest
from pathlib import Path

from friday_core.graph import GraphStore
from friday_core.memory_evals import MemoryEvalRunner


REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "evals" / "memory-retrieval-v1.json"


class MemoryEvalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph = GraphStore(self.root / "evaluation.db")

    def tearDown(self):
        self.temporary.cleanup()

    def database_dump(self):
        with self.graph._connect() as conn:
            return "\n".join(conn.iterdump())

    def test_full_scorecard_passes_without_contaminating_live_memory(self):
        result = MemoryEvalRunner(self.graph).run(SUITE)

        self.assertEqual((result["passed"], result["total"]), (7, 7))
        self.assertEqual(result["pass_rate"], 1.0)
        self.assertEqual(self.graph.count_nodes("memory_evaluation_run"), 1)
        self.assertEqual(self.graph.count("claim_state"), 0)
        dumped = self.database_dump()
        self.assertNotIn("visible progress", dumped)
        self.assertNotIn("Latin", dumped)
        self.assertTrue(all(len(item["observation_sha256"]) == 64
                            for item in result["results"]))

    def test_suite_tampering_symlinks_duplicates_and_nonfinite_fail_closed(self):
        alias = self.root / "alias.json"
        alias.symlink_to(SUITE)
        with self.assertRaises(ValueError):
            MemoryEvalRunner._load_suite(alias)

        suite = json.loads(SUITE.read_text())
        suite["cases"][0]["expected"]["returned"] = 2
        tampered = self.root / "tampered.json"
        tampered.write_text(json.dumps(suite))
        with self.assertRaises(ValueError):
            MemoryEvalRunner._load_suite(tampered)

        suite = json.loads(SUITE.read_text())
        suite["cases"][1]["scenario"] = suite["cases"][0]["scenario"]
        suite["cases"][1]["expected"] = suite["cases"][0]["expected"]
        duplicate = self.root / "duplicate.json"
        duplicate.write_text(json.dumps(suite))
        with self.assertRaises(ValueError):
            MemoryEvalRunner._load_suite(duplicate)

        nonfinite = self.root / "nonfinite.json"
        nonfinite.write_text(
            '{"name":"friday-memory-retrieval","version":NaN,'
            '"coverage":[],"cases":[]}')
        with self.assertRaises(ValueError):
            MemoryEvalRunner._load_suite(nonfinite)
        self.assertEqual(self.graph.count_nodes("memory_evaluation_run"), 0)

    def test_scenario_exception_is_contained_and_successors_continue(self):
        class OneFailureRunner(MemoryEvalRunner):
            calls = 0

            @classmethod
            def _scenario(cls, name, memory, graph):
                cls.calls += 1
                if cls.calls == 1:
                    raise RuntimeError("synthetic failure")
                return super()._scenario(name, memory, graph)

        result = OneFailureRunner(self.graph).run(SUITE)

        self.assertEqual((result["passed"], result["total"]), (6, 7))
        self.assertEqual(result["results"][0]["failure"], "RuntimeError")
        self.assertTrue(all(item["passed"] for item in result["results"][1:]))


if __name__ == "__main__":
    unittest.main()
