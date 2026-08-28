import json
import tempfile
import unittest
from pathlib import Path

from friday_core.adversarial_evals import AdversarialEvalRunner
from friday_core.graph import GraphStore


REPO = Path(__file__).resolve().parents[1]


class AdversarialEvalTests(unittest.TestCase):
    def test_all_boundary_attacks_fail_closed_with_bounded_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = AdversarialEvalRunner(
                GraphStore(root / "journal.db"), REPO).run(
                    REPO / "evals" / "adversarial-v1.json")

        self.assertTrue(result["passed"], result)
        self.assertEqual(result["scenarios_passed"], 7)
        self.assertEqual(result["scenarios_total"], 7)
        effects = {item["scenario"]: item["effect_count"]
                   for item in result["results"]}
        self.assertEqual(effects["stale_approval_replay"], 1)
        self.assertTrue(all(count == 0 for scenario, count in effects.items()
                            if scenario != "stale_approval_replay"))
        self.assertTrue(result["privacy"]["cleanup_verified"])

    def test_suite_tampering_and_symlinks_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = json.loads(
                (REPO / "evals" / "adversarial-v1.json").read_text())
            suite["scenarios"] = list(reversed(suite["scenarios"]))
            altered = root / "altered.json"
            altered.write_text(json.dumps(suite))
            runner = AdversarialEvalRunner(
                GraphStore(root / "journal.db"), REPO)
            with self.assertRaisesRegex(ValueError, "metadata"):
                runner.run(altered)
            link = root / "linked.json"
            link.symlink_to(REPO / "evals" / "adversarial-v1.json")
            with self.assertRaisesRegex(ValueError, "unavailable"):
                runner.run(link)


if __name__ == "__main__":
    unittest.main()
