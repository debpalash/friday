import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from friday_core.dependency_review import (host_lock_name, parse_lock,
                                           run_dependency_review,
                                           write_private_review)


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

    def test_review_probes_the_host_environment_and_checks_foreign_locks(self):
        import sys

        python = Path(sys.executable)
        from friday_host.paths import default_qwen_runtime

        qwen_python = default_qwen_runtime() / "venv" / "bin" / "python"
        result = run_dependency_review(
            REPO, app_python=python,
            qwen_python=qwen_python if qwen_python.is_file() else None)
        self.assertEqual(result["review_version"], 2)
        reviewed = result["installed_reviewed"]
        self.assertIn(len(reviewed), (1, 2))
        installed = result["environments"][[n for n in reviewed if n != "qwen_runtime"][0]]
        self.assertEqual(installed["review"], "installed")
        self.assertEqual(installed["mismatches"], [], installed["mismatches"][:5])
        self.assertTrue(installed["passed"], installed["missing_license_evidence"][:5])
        for name, value in result["environments"].items():
            if name in reviewed or "vllm" in value["engines"]:
                # The CUDA and vLLM locks are reviewed against their installed
                # environments by the release-candidate run on the Linux box.
                continue
            with self.subTest(lock=name):
                self.assertEqual(value["review"], "lock_only")
                self.assertTrue(value["policy_sha256_matches"])
                self.assertEqual(value["missing_license_evidence"], [])
        self.assertTrue(result["binaries_complete"])
        self.assertTrue(result["models_complete"])
        if "qwen_runtime" in reviewed:
            self.assertTrue(result["passed"], result["environments"]["qwen_runtime"]["missing_license_evidence"][:5])
        self.assertEqual(host_lock_name("macos-arm64"), "application-macos-arm64")

    def test_unknown_environment_names_are_rejected(self):
        with self.assertRaises(ValueError):
            run_dependency_review(REPO, environments={"nope": Path("/usr/bin/python3")})

    def test_private_report_is_mode_600_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "private" / "review.json"
            write_private_review(path, {"passed": True})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_private_review(path, {"passed": False})


if __name__ == "__main__":
    unittest.main()
