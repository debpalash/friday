import tempfile
import unittest
from pathlib import Path

from friday_core import GraphStore, ReflectionService, TaskService


class ReflectionTests(unittest.TestCase):
    def test_reflection_and_lessons_remain_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = GraphStore(Path(tmp) / "friday.db")
            tasks = TaskService(graph)
            task_id, _ = tasks.create("fix voice", {"evidence": "test"})
            reflection = ReflectionService(graph)

            reflection_id = reflection.record(
                task_id, "Voice initialization failed before fallback.",
                ["Load GPU services in dependency order."])

            self.assertEqual(graph.get_node(reflection_id)["body"]["lifecycle"],
                             "candidate")
            self.assertEqual(graph.count("claim_state"), 0)
            with graph._connect() as conn:
                lesson = conn.execute(
                    "SELECT body_json FROM nodes WHERE kind='lesson'").fetchone()
            self.assertIn('"knowledge_layer":"candidate"', lesson[0])


if __name__ == "__main__":
    unittest.main()
