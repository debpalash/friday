import tempfile
import unittest
from pathlib import Path

from friday_core import CorrectedAudioStore, FeedbackService, GraphStore, TaskService
from friday_core.feedback import ApprovalService


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.graph = GraphStore(Path(self.tmp.name) / "friday.db")
        self.feedback = FeedbackService(self.graph)

    def tearDown(self):
        self.tmp.cleanup()

    def test_wrong_feedback_disputes_task_verification(self):
        task_id, _ = TaskService(self.graph).create("x", {"evidence": "legacy"})
        result = self.feedback.record("wrong", task_id=task_id)
        self.assertTrue(result["feedback_id"].startswith("feedback_"))
        self.assertEqual(TaskService(self.graph).get(task_id)["verification_status"],
                         "failed")

    def test_actionable_feedback_is_retrieved_only_for_similar_tasks(self):
        tasks = TaskService(self.graph)
        news_task, _ = tasks.create("Summarize today's India news",
                                    {"evidence": "legacy"})
        file_task, _ = tasks.create("List project files", {"evidence": "legacy"})
        self.feedback.record(
            "problem", task_id=news_task,
            comment="Give one synthesized sentence instead of reading every headline")
        self.feedback.record(
            "problem", task_id=file_task,
            comment="Include hidden files")

        hits = self.feedback.relevant_context("What's the India news today?")

        self.assertEqual(len(hits), 1)
        self.assertIn("synthesized sentence", hits[0]["comment"])

    def test_transcript_correction_links_to_utterance(self):
        utterance_id = self.graph.record_node(
            "utterance", {"text": "wrong words", "source": "asr"}, actor="user")
        result = self.feedback.correct_transcript(utterance_id, "right words")
        self.assertEqual(result["original_text"], "wrong words")
        corrected, evidence = self.feedback.apply_transcript_corrections("Wrong words")
        self.assertEqual(corrected, "right words")
        self.assertEqual(evidence, [result["correction_id"]])
        with self.graph._connect() as conn:
            relation = conn.execute(
                "SELECT relation FROM edges WHERE from_node_id=?",
                (result["correction_id"],)).fetchone()[0]
        self.assertEqual(relation, "corrects")

    def test_corrected_audio_is_encrypted_and_deletable(self):
        root = Path(self.tmp.name) / "audio"
        store = CorrectedAudioStore(root, key_provider=lambda: b"k" * 64)
        path = Path(store.store("utterance_1", b"private voice bytes",
                               {"sample_rate": 16000}))
        self.assertNotIn(b"private voice bytes", path.read_bytes())
        self.assertTrue(store.delete(path))
        self.assertFalse(path.exists())

    def test_browser_input_approval_redacts_text_but_matches_hash(self):
        task_id, _ = TaskService(self.graph).create("fill form", {"evidence": "legacy"})
        approvals = ApprovalService(self.graph)
        args = {"selector": "#name", "text": "private value"}
        item = approvals.request(task_id, "browser_type", args, "external effect")
        self.assertEqual(item["args"]["text"], "[REDACTED]")
        approvals.decide(item["approval_id"], True)
        self.assertTrue(approvals.is_approved(task_id, "browser_type", args))

    def test_file_approval_persists_hash_metadata_not_content(self):
        task_id, _ = TaskService(self.graph).create(
            "edit file", {"evidence": "legacy"})
        approvals = ApprovalService(self.graph)
        args = {"path": "server.py", "content": "approval-content-secret"}

        item = approvals.request(task_id, "write_file", args, "filesystem change")

        self.assertEqual(item["args"]["content"], "[REDACTED]")
        with self.graph._connect() as conn:
            durable_dump = "\n".join(conn.iterdump())
        self.assertNotIn("approval-content-secret", durable_dump)


if __name__ == "__main__":
    unittest.main()
