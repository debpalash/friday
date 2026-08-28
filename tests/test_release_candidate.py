import json
import tempfile
import unittest
from pathlib import Path

from friday_core.release_candidate import (
    canonical_sha256,
    evaluation_passed,
    write_private_candidate_report,
)


class ReleaseCandidateTests(unittest.TestCase):
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
