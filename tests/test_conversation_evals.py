import json
import tempfile
import unittest
from pathlib import Path

from friday_core.conversation_evals import ConversationQualityEvalRunner


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "evals" / "conversation-quality-v1.json"


class ConversationQualityEvalTests(unittest.TestCase):
    def test_held_out_suite_loads_with_both_delivery_modes(self):
        suite = ConversationQualityEvalRunner._load_suite(SUITE)

        self.assertGreaterEqual(len(suite["cases"]), 8)
        self.assertEqual({case["mode"] for case in suite["cases"]},
                         {"voice", "text"})

    def test_exact_grader_passes_compliant_output_and_hashes_raw_text(self):
        output = "Recursion solves a problem by applying the same rule to itself."
        case = {
            "min_words": 5, "max_words": 20, "max_sentences": 1,
            "forbid_markdown": True, "required_any": [["itself"]],
            "forbidden_terms": [],
        }

        checks = ConversationQualityEvalRunner._grade(case, output)

        self.assertTrue(checks["passed"])
        self.assertEqual(checks["word_count"], 11)

    def test_forbidden_terms_match_words_not_substrings(self):
        case = {
            "min_words": 1, "max_words": 20, "max_sentences": 2,
            "forbid_markdown": False, "required_any": [],
            "forbidden_terms": ["sure"],
        }

        self.assertTrue(ConversationQualityEvalRunner._grade(
            case, "A closure walks into a compiler.")["passed"])
        self.assertFalse(ConversationQualityEvalRunner._grade(
            case, "Sure, a compiler walks into a closure.")["passed"])

    def test_exact_grader_rejects_voice_markdown_ceremony_and_repetition(self):
        case = {
            "min_words": 1, "max_words": 100, "max_sentences": 8,
            "forbid_markdown": True, "required_any": [],
            "forbidden_terms": [],
        }
        output = (
            "## Result\nTask completed. The answer is 42. The answer is 42.")

        checks = ConversationQualityEvalRunner._grade(case, output)

        self.assertFalse(checks["passed"])
        self.assertFalse(checks["markdown_policy"])
        self.assertFalse(checks["no_task_ceremony"])
        self.assertFalse(checks["no_repeated_sentences"])

    def test_exact_grader_rejects_oversized_output_before_quality_can_pass(self):
        case = {
            "min_words": 1, "max_words": 2_000, "max_sentences": 2_000,
            "forbid_markdown": False, "required_any": [],
            "forbidden_terms": [],
        }

        checks = ConversationQualityEvalRunner._grade(case, "word " * 2_000)

        self.assertFalse(checks["bounded_output"])
        self.assertFalse(checks["passed"])

    def test_run_continues_after_completion_failure_without_storing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary) / "suite.json"
            suite.write_text(json.dumps({
                "name": "failure isolation", "version": 1, "coverage": [],
                "cases": [
                    {"name": "first", "mode": "voice", "prompt": "first"},
                    {"name": "second", "mode": "text", "prompt": "second"},
                ],
            }))
            calls = 0

            def complete(_system, _user):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("model unavailable")
                return "A direct answer."

            result = ConversationQualityEvalRunner(
                complete, model="test-model",
                runtime_fingerprint="a" * 64).run(suite)

        self.assertEqual((result["passed"], result["total"]), (1, 2))
        self.assertEqual(result["results"][0]["failure"], "RuntimeError")
        self.assertNotIn("output", result["results"][1])
        self.assertEqual(len(result["results"][1]["output_sha256"]), 64)

    def test_suite_rejects_symlink_duplicate_case_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.json"
            case = {"name": "one", "mode": "voice", "prompt": "hello"}
            valid.write_text(json.dumps({
                "name": "valid", "version": 1, "coverage": [],
                "cases": [case],
            }))
            alias = root / "alias.json"
            alias.symlink_to(valid)
            with self.assertRaises(ValueError):
                ConversationQualityEvalRunner._load_suite(alias)

            duplicate = root / "duplicate.json"
            duplicate.write_text(json.dumps({
                "name": "duplicate", "version": 1, "coverage": [],
                "cases": [case, case],
            }))
            with self.assertRaises(ValueError):
                ConversationQualityEvalRunner._load_suite(duplicate)

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text(
                '{"name":"bad","version":NaN,"coverage":[],"cases":[]}')
            with self.assertRaises(ValueError):
                ConversationQualityEvalRunner._load_suite(nonfinite)

    def test_runtime_fingerprint_is_required(self):
        with self.assertRaises(ValueError):
            ConversationQualityEvalRunner(
                lambda _system, _user: "answer", model="test-model",
                runtime_fingerprint="not-a-fingerprint")


if __name__ == "__main__":
    unittest.main()
