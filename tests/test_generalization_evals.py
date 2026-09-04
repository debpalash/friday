import json
import tempfile
import unittest

from tests.platform_markers import sandbox_available
from pathlib import Path

from friday_core.generalization_evals import GroundedQAEvalRunner
from friday_core.graph import GraphStore


def _case(name="grounded case", *, secret="ORBIT-PRIVATE-742"):
    return {
        "name": name,
        "modality": "docx",
        "context": f"The verified code is {secret}.",
        "question": "What is the verified code?",
        "required_answer_terms": [secret],
        "required_evidence_terms": [secret],
    }


class GroundedQAEvalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph = GraphStore(self.root / "evaluation.db")

    def tearDown(self):
        self.temporary.cleanup()

    def _suite(self, cases):
        path = self.root / "suite.json"
        path.write_text(json.dumps({
            "name": "test-grounded-suite", "version": 1,
            "coverage": ["grounding"], "cases": cases,
        }))
        return path

    def database_dump(self):
        with self.graph._connect() as conn:
            return "\n".join(conn.iterdump())

    def test_exact_answer_and_verbatim_evidence_pass_without_raw_journaling(self):
        secret = "ORBIT-PRIVATE-742"
        raw = json.dumps({
            "answer": f"The code is {secret}.",
            "evidence": [f"The verified code is {secret}."],
        })
        result = GroundedQAEvalRunner(
            self.graph, lambda _system, _user: raw,
            model="test-model", runtime_fingerprint="a" * 64).run(
                self._suite([_case(secret=secret)]))
        self.assertEqual((result["passed"], result["total"]), (1, 1))
        self.assertNotIn(secret, self.database_dump())
        self.assertNotIn(raw, self.database_dump())
        self.assertEqual(len(result["results"][0]["output_sha256"]), 64)

    def test_paraphrased_or_incomplete_evidence_fails_exact_grading(self):
        raw = json.dumps({
            "answer": "The code is ORBIT-PRIVATE-742.",
            "evidence": ["A code was verified."],
        })
        result = GroundedQAEvalRunner(
            self.graph, lambda _system, _user: raw,
            model="test-model", runtime_fingerprint="a" * 64).run(
                self._suite([_case()]))
        case = result["results"][0]
        self.assertFalse(case["passed"])
        self.assertTrue(case["answer_requirements_met"])
        self.assertFalse(case["evidence_grounded"])
        self.assertFalse(case["evidence_requirements_met"])

    @unittest.skipUnless(
        sandbox_available() and Path("/usr/bin/pdftotext").is_file(),
        "environment: sandboxed PDF extraction is unavailable")
    def test_insufficient_evidence_requires_exact_abstention_and_empty_citations(self):
        case = {
            "name": "missing fact", "modality": "pdf",
            "context": "The room is Lab 4.", "question": "What is the budget?",
            "required_answer_terms": [], "required_evidence_terms": [],
            "expect_insufficient": True,
        }
        for raw, expected in (
                (json.dumps({"answer": "INSUFFICIENT_EVIDENCE", "evidence": []}), True),
                (json.dumps({"answer": "Probably $100", "evidence": []}), False),
                (json.dumps({"answer": "INSUFFICIENT_EVIDENCE",
                             "evidence": ["The room is Lab 4."]}), False)):
            graph = GraphStore(self.root / f"{hash(raw)}.db")
            result = GroundedQAEvalRunner(
                graph, lambda _system, _user, value=raw: value,
                model="test-model", runtime_fingerprint="a" * 64).run(
                    self._suite([case]))
            self.assertIs(result["results"][0]["passed"], expected)

    def test_case_exception_is_failed_and_does_not_abort_successor(self):
        calls = 0

        def complete(_system, _user):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("model unavailable")
            return json.dumps({
                "answer": "ORBIT-PRIVATE-742",
                "evidence": ["The verified code is ORBIT-PRIVATE-742."],
            })

        result = GroundedQAEvalRunner(
            self.graph, complete, model="test-model",
            runtime_fingerprint="a" * 64).run(
                self._suite([_case("first"), _case("second")]))
        self.assertEqual((result["passed"], result["total"]), (1, 2))
        self.assertEqual(result["results"][0]["failure"], "RuntimeError")
        self.assertTrue(result["results"][1]["passed"])

    def test_suite_rejects_symlink_duplicate_case_and_nonfinite_json(self):
        valid = self._suite([_case()])
        alias = self.root / "alias.json"
        alias.symlink_to(valid)
        with self.assertRaises(ValueError):
            GroundedQAEvalRunner._load_suite(alias)

        duplicate = self.root / "duplicate.json"
        duplicate.write_text(json.dumps({
            "name": "duplicate", "version": 1, "coverage": [],
            "cases": [_case(), _case()],
        }))
        with self.assertRaises(ValueError):
            GroundedQAEvalRunner._load_suite(duplicate)

        nonfinite = self.root / "nonfinite.json"
        nonfinite.write_text(
            '{"name":"bad","version":NaN,"coverage":[],"cases":[]}')
        with self.assertRaises(ValueError):
            GroundedQAEvalRunner._load_suite(nonfinite)

    def test_generated_archive_artifacts_have_stable_source_and_context_hashes(self):
        for modality in ("docx", "xlsx"):
            case = _case()
            case["modality"] = modality
            first_context, first = GroundedQAEvalRunner._artifact_context(case)
            second_context, second = GroundedQAEvalRunner._artifact_context(case)
            self.assertEqual(first_context, second_context)
            self.assertEqual(first, second)

    def test_runtime_fingerprint_is_mandatory_evaluation_provenance(self):
        with self.assertRaises(ValueError):
            GroundedQAEvalRunner(
                self.graph, lambda _system, _user: "{}",
                model="test-model", runtime_fingerprint="not-a-fingerprint")
