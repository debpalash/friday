import tempfile
import unittest
from pathlib import Path

from friday_core import GraphStore, TaskService, VoiceManager


class VoiceManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.graph = GraphStore(root / "friday.db")
        self.tasks = TaskService(self.graph)
        self.voices = VoiceManager(self.graph, root / "persona" / "voices")
        self.voices.discover()
        self.task_id, _ = self.tasks.create("create calm voice", {"evidence": "audio"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_candidate_activates_only_with_synthesis_receipt(self):
        self.voices.create(
            "calm", "female, calm, lower pitch", source_node_ids=[self.task_id])
        self.assertEqual(self.voices.get("calm")["status"], "candidate")
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.voices.activate("calm", {"passed": False})
        self.assertEqual(self.voices.active()["name"], "base")

        self.voices.activate(
            "calm", {"passed": True, "samples": 24000, "sample_rate": 24000})
        self.assertEqual(self.voices.active()["name"], "calm")
        self.assertEqual(self.voices.previous()["name"], "base")

    def test_profile_requires_task_provenance(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.voices.create("orphan", "bright", source_node_ids=[])


if __name__ == "__main__":
    unittest.main()
