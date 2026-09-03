import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from friday_core.dependency_review import parse_lock, write_private_review


REPO = Path(__file__).resolve().parents[1]


class DependencyReviewTests(unittest.TestCase):
    def test_policy_is_bound_to_exact_complete_locks_and_models(self):
        policy = json.loads((
            REPO / "compliance" / "dependency-review-v1.json").read_text())
        for name, expected in policy["locks"].items():
            with self.subTest(name=name):
                path = REPO / expected["path"]
                packages = parse_lock(path)
                self.assertEqual(len(packages), expected["packages"])
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    expected["sha256"],
                )
        self.assertTrue(policy["models_and_assets"])
        self.assertEqual(policy["engineering_review"], "complete")
        self.assertEqual(
            policy["distribution_approval"],
            "approved_apache_2_0_with_gpl_3_piper_runtime",
        )

    def test_private_report_is_mode_600_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "private" / "review.json"
            write_private_review(path, {"passed": True})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_private_review(path, {"passed": False})


if __name__ == "__main__":
    unittest.main()
