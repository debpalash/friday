import json
import os
import tempfile
import unittest
from pathlib import Path

from friday_core.release_candidate import (
    ReleaseCandidateRunner,
    canonical_sha256,
    evaluation_passed,
    write_private_candidate_report,
)


ROOT = Path(__file__).resolve().parents[1]


class ReleaseCandidateTests(unittest.TestCase):
    def test_virtualenv_python_launcher_path_is_not_resolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base-python"
            base.write_text("#!/bin/sh\nexit 0\n")
            base.chmod(0o755)
            app = root / "app-python"
            qwen = root / "qwen-python"
            app.symlink_to(base)
            qwen.symlink_to(base)

            runner = ReleaseCandidateRunner(
                ROOT, app_python=app, qwen_python=qwen)

            self.assertEqual(runner.app_python, Path(os.path.abspath(app)))
            self.assertEqual(runner.qwen_python, Path(os.path.abspath(qwen)))
            self.assertNotEqual(runner.app_python, base.resolve())

    def test_evaluation_results_require_exact_boolean_or_complete_count(self):
        self.assertTrue(evaluation_passed({"passed": True}))
        self.assertTrue(evaluation_passed({"passed": 8, "total": 8}))
        for value in (
            {"passed": False}, {"passed": 7, "total": 8},
            {"passed": "8", "total": 8}, {"passed": 0, "total": 0},
        ):
            with self.subTest(value=value):
                self.assertFalse(evaluation_passed(value))

    def test_private_report_is_hash_bound_mode_600_and_non_overwriting(self):
        body = {"format_version": 1, "local_gates_passed": True}
        body["report_payload_sha256"] = canonical_sha256(body)
        with tempfile.TemporaryDirectory() as temporary:
            private = Path(temporary) / "private"
            private.mkdir(mode=0o700)
            path = private / "candidate.json"
            write_private_candidate_report(path, body)

            stored = json.loads(path.read_text())
            digest = stored.pop("report_payload_sha256")
            self.assertEqual(digest, canonical_sha256(stored))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_private_candidate_report(path, body)

    def test_report_rejects_non_private_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "public"
            parent.mkdir(mode=0o755)
            with self.assertRaises(PermissionError):
                write_private_candidate_report(
                    parent / "candidate.json", {"passed": True})


if __name__ == "__main__":
    unittest.main()
