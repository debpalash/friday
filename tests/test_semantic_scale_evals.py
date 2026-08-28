from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from friday_core.graph import GraphStore
from friday_core.semantic_scale_evals import SemanticScaleEvalRunner


class FixtureEmbedder:
    fingerprint = "f" * 64
    dimension = 8
    batch_size = 16

    @staticmethod
    def _concept(text: str, *, kind: str) -> int:
        lowered = text.casefold()
        for index, marker in enumerate(("alpha", "beta", "gamma", "delta")):
            if marker in lowered:
                return index
        return 5 if kind == "query" else 4

    def encode(self, texts, *, kind):
        output = np.zeros((len(texts), self.dimension), dtype="<f4")
        for row, text in enumerate(texts):
            output[row, self._concept(text, kind=kind)] = 1.0
        return output


def fixture_suite() -> dict:
    memories = []
    for marker in ("alpha", "beta", "gamma", "delta"):
        memories.append({
            "id": f"{marker}_memory",
            "predicate": f"{marker}_preference",
            "value": f"use the {marker} response mode",
            "queries": [
                f"apply {marker} mode",
                f"remember my {marker} preference",
            ],
        })
    return {
        "name": "friday-semantic-scale",
        "version": 1,
        "gates": {
            "precision": 1.0,
            "recall": 1.0,
            "abstention": 1.0,
            "corrected_task_lift": 1.0,
            "warm_p95_ms": 5000,
            "maximum_corpus_minimum": 4_100,
        },
        "scale_sizes": [16, 4_100],
        "memories": memories,
        "negative_queries": [
            "unknown weather question",
            "unknown sports question",
            "unknown currency question",
            "unknown recipe question",
        ],
        "corrections": [
            {
                "id": "alpha_correction",
                "predicate": "correction_alpha",
                "initial": "alpha old setting",
                "corrected": "alpha corrected setting",
                "query": "use the alpha corrected setting",
            },
            {
                "id": "beta_correction",
                "predicate": "correction_beta",
                "initial": "beta old setting",
                "corrected": "beta corrected setting",
                "query": "use the beta corrected setting",
            },
        ],
    }


class SemanticScaleEvalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph = GraphStore(self.root / "results.db")
        self.suite_path = self.root / "suite.json"
        self.suite_path.write_text(json.dumps(fixture_suite()))

    def tearDown(self):
        self.temporary.cleanup()

    def test_scorecard_measures_quality_scale_latency_and_correction_lift(self):
        result = SemanticScaleEvalRunner(
            self.graph, FixtureEmbedder()).run(self.suite_path)

        self.assertTrue(result["passed"])
        self.assertEqual(result["quality"]["precision"], 1.0)
        self.assertEqual(result["quality"]["recall"], 1.0)
        self.assertEqual(result["quality"]["abstention"], 1.0)
        self.assertEqual(result["correction"]["corrected_task_lift"], 1.0)
        self.assertEqual(
            [item["corpus"] for item in result["scale"]], [16, 4_100])
        self.assertTrue(all(item["index_complete"] for item in result["scale"]))
        self.assertTrue(all(
            item["oldest_target_first"] for item in result["scale"]))
        self.assertEqual(result["decision"]["index_strategy"], "sharded_exact")
        self.assertFalse(result["decision"]["ann_required"])
        encoded = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("unknown weather question", encoded)
        self.assertEqual(
            self.graph.count_nodes("semantic_scale_evaluation_run"), 1)

    def test_suite_rejects_duplicates_nonfinite_and_symlinks(self):
        duplicate = fixture_suite()
        duplicate["memories"][1]["id"] = duplicate["memories"][0]["id"]
        self.suite_path.write_text(json.dumps(duplicate))
        with self.assertRaisesRegex(ValueError, "memory case"):
            SemanticScaleEvalRunner._load_suite(self.suite_path)

        self.suite_path.write_text('{"value": NaN}')
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            SemanticScaleEvalRunner._load_suite(self.suite_path)

        real = self.root / "real.json"
        real.write_text(json.dumps(fixture_suite()))
        self.suite_path.unlink()
        self.suite_path.symlink_to(real)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            SemanticScaleEvalRunner._load_suite(self.suite_path)

    def test_production_suite_crosses_the_legacy_scan_limit(self):
        production = (
            Path(__file__).parents[1] / "evals" / "semantic-scale-v1.json")
        suite, digest = SemanticScaleEvalRunner._load_suite(production)
        self.assertGreater(suite["scale_sizes"][-1], 4_096)
        self.assertEqual(len(digest), 64)
        self.assertGreaterEqual(
            sum(len(item["queries"]) for item in suite["memories"]), 12)


if __name__ == "__main__":
    unittest.main()
