import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server
from friday_core.graph import GraphStore
from friday_core.vision_evals import NativeVisionEvalRunner

from tests.platform_markers import require_platform

require_platform("linux")
if not all(Path(tool).is_file() for tool in (
        "/usr/bin/bwrap", "/usr/bin/magick", "/usr/bin/tesseract")):
    raise unittest.SkipTest("environment: sandboxed image tools are unavailable")


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "evals" / "native-vision-v1.json"


class NativeVisionEvalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph = GraphStore(self.root / "evaluation.db")
        self.suite = json.loads(SUITE.read_text())

    def tearDown(self):
        self.temporary.cleanup()

    def _runner(self, complete):
        return NativeVisionEvalRunner(
            self.graph, complete, model="vision-test-model",
            runtime_fingerprint="a" * 64, max_side=1024)

    def _write_suite(self, value, name="suite.json"):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def database_dump(self):
        with self.graph._connect() as conn:
            return "\n".join(conn.iterdump())

    def test_five_artifact_backed_scenes_pass_without_raw_answer_journaling(self):
        answers = {
            case["question"]: case["expected_answer"]
            for case in self.suite["cases"]}
        observed = []

        def complete(question, encoded):
            observed.append((question, encoded))
            return answers[question]

        result = self._runner(complete).run(SUITE)

        self.assertEqual((result["passed"], result["total"]), (5, 5))
        self.assertEqual(len(observed), 5)
        self.assertTrue(all(image.startswith(b"\x89PNG")
                            for _question, image in observed))
        self.assertEqual(self.graph.count_nodes(
            "native_vision_evaluation_run"), 1)
        dump = self.database_dump()
        for answer in answers.values():
            if len(answer) > 3:
                self.assertNotIn(answer, dump)
        for case in result["results"]:
            self.assertEqual(len(case["source_sha256"]), 64)
            self.assertEqual(len(case["image_sha256"]), 64)
            self.assertEqual(len(case["output_sha256"]), 64)

    def test_tool_qualification_requires_current_exact_five_scene_score(self):
        answers = {
            case["question"]: case["expected_answer"]
            for case in self.suite["cases"]}
        with patch.multiple(
                server, GRAPH=self.graph, NATIVE_VISION_ENABLED=True,
                NATIVE_VISION_MAX_SIDE=1024,
                RUNTIME_FINGERPRINT="a" * 64,
                LOCAL_MODEL="vision-test-model"):
            self.assertFalse(server._native_vision_qualified())
            self._runner(
                lambda question, _image: answers[question]).run(SUITE)
            self.assertTrue(server._native_vision_qualified())

    def test_wrong_or_oversized_answer_fails_exact_grading(self):
        one = dict(self.suite)
        one["cases"] = [self.suite["cases"][0]]
        path = self._write_suite(one)
        for raw in ("blue circle", "red square " + "x" * 300):
            graph = GraphStore(self.root / f"{len(raw)}.db")
            result = NativeVisionEvalRunner(
                graph, lambda _question, _image, value=raw: value,
                model="vision-test-model", runtime_fingerprint="b" * 64,
                max_side=512).run(path)
            self.assertEqual((result["passed"], result["total"]), (0, 1))
            self.assertNotIn(raw, "\n".join(
                conn_line for conn_line in self._dump(graph)))

    @staticmethod
    def _dump(graph):
        with graph._connect() as conn:
            return list(conn.iterdump())

    def test_case_failure_is_contained_and_successor_still_runs(self):
        calls = 0

        def complete(question, _image):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("model unavailable")
            return next(case["expected_answer"] for case in self.suite["cases"]
                        if case["question"] == question)

        result = self._runner(complete).run(SUITE)
        self.assertEqual((result["passed"], result["total"]), (4, 5))
        self.assertEqual(result["results"][0]["failure"], "RuntimeError")
        self.assertTrue(result["results"][1]["passed"])

    def test_suite_tampering_symlinks_and_nonfinite_json_fail_closed(self):
        tampered = json.loads(json.dumps(self.suite))
        tampered["cases"][0]["expected_answer"] = "blue circle"
        with self.assertRaisesRegex(ValueError, "case metadata"):
            NativeVisionEvalRunner._load_suite(self._write_suite(tampered))

        valid = self._write_suite(self.suite, "valid.json")
        alias = self.root / "alias.json"
        alias.symlink_to(valid)
        with self.assertRaisesRegex(ValueError, "bounded regular file"):
            NativeVisionEvalRunner._load_suite(alias)

        nonfinite = self.root / "nonfinite.json"
        nonfinite.write_text(
            '{"name":"bad","version":NaN,"coverage":[],"cases":[]}')
        with self.assertRaisesRegex(ValueError, "non-finite"):
            NativeVisionEvalRunner._load_suite(nonfinite)

    def test_rendered_and_sanitized_artifacts_are_deterministic(self):
        runner = self._runner(lambda _question, _image: "unused")
        for scene in ("left_color_shape", "count_green_triangles",
                      "shape_inside_frame"):
            first_image, first = runner._artifact(scene)
            second_image, second = runner._artifact(scene)
            self.assertEqual(first_image, second_image)
            self.assertEqual(first, second)
            self.assertLessEqual(first["width"], 1024)
            self.assertLessEqual(first["height"], 1024)

    def test_runtime_identity_and_hardware_limit_are_mandatory(self):
        with self.assertRaises(ValueError):
            NativeVisionEvalRunner(
                self.graph, lambda _question, _image: "x",
                model="vision-test-model", runtime_fingerprint="bad",
                max_side=1024)
        with self.assertRaises(ValueError):
            NativeVisionEvalRunner(
                self.graph, lambda _question, _image: "x",
                model="vision-test-model", runtime_fingerprint="a" * 64,
                max_side=128)


if __name__ == "__main__":
    unittest.main()
