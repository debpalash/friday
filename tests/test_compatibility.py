import json
import unittest
from pathlib import Path

from friday_core.db_migrations import LATEST_SCHEMA_VERSION
from scripts.check_architecture import inspect_architecture


ROOT = Path(__file__).resolve().parents[1]


class CompatibilityPolicyTests(unittest.TestCase):
    def test_machine_policy_matches_authoritative_schema_and_alpha_version(self):
        policy = json.loads(
            (ROOT / "compatibility" / "v1.json").read_text(encoding="utf-8"))
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

        self.assertEqual(policy["policy_version"], 1)
        self.assertEqual(policy["release_stage"], "alpha")
        self.assertTrue(version.startswith("0.1."))
        self.assertEqual(policy["graph"]["current_schema"],
                         LATEST_SCHEMA_VERSION)
        self.assertFalse(policy["graph"]["downgrade_supported"])
        self.assertEqual(
            policy["extension_apis"]["compatibility_window"],
            "none_during_alpha",
        )

    def test_server_composes_explicit_review_boundaries(self):
        result = inspect_architecture()

        self.assertTrue(result["passed"], result)
        self.assertFalse(result["embedded_frontend"])
        self.assertTrue(result["external_frontend"])


if __name__ == "__main__":
    unittest.main()
