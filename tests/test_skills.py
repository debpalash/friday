import tempfile
import unittest
from pathlib import Path

from friday_core import GraphStore, SkillManager


class SkillManagerTests(unittest.TestCase):
    def test_versions_require_tests_and_activation_requires_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = GraphStore(root / "friday.db")
            skills = SkillManager(graph, root / "skills")
            source = graph.record_node("observation", {"result": "worked"})
            version = skills.create_version(
                "Inspect Files", "Use list_files and verify the requested path.",
                {"permissions": ["read_project"]},
                [{"name": "lists root", "expected": "nonempty"}],
                source_node_ids=[source])

            with self.assertRaises(ValueError):
                skills.activate(version)
            self.assertTrue(skills.evaluate(
                version, [{"name": "lists root", "passed": True}]))
            skills.activate(version)

            active = skills.active_context()
            self.assertEqual(active[0]["name"], "inspect-files")
            self.assertTrue((root / "skills/inspect-files/v1/SKILL.md").is_file())

    def test_failed_evaluation_quarantines_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = GraphStore(root / "friday.db")
            skills = SkillManager(graph, root / "skills")
            source = graph.record_node("observation", {"result": "attempt"})
            version = skills.create_version(
                "Unsafe Guess", "Guess.", {}, [{"name": "truth"}],
                source_node_ids=[source])
            self.assertFalse(skills.evaluate(
                version, [{"name": "truth", "passed": False}]))
            self.assertEqual(skills.list()[0]["status"], "quarantined")

    def test_active_skill_is_hidden_without_its_executable_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = GraphStore(root / "friday.db")
            skills = SkillManager(graph, root / "skills")
            source = graph.record_node("observation", {"result": "verified news"})
            version = skills.create_version(
                "News Workflow", "Use fetch_news, then summarize its receipt.",
                {"permissions": ["fetch_news"]},
                [{"name": "historical fetch", "passed": True}],
                source_node_ids=[source])
            self.assertTrue(skills.evaluate(version, [{"passed": True}]))
            skills.activate(version)

            self.assertEqual(skills.active_context(available_tools=set()), [])
            self.assertEqual(len(skills.active_context(
                available_tools={"fetch_news"})), 1)

    def test_active_skills_are_retrieved_only_for_relevant_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = GraphStore(root / "friday.db")
            skills = SkillManager(graph, root / "skills")
            source = graph.record_node("observation", {"result": "verified news"})
            version = skills.create_version(
                "News Workflow", "Use fetch_news, then summarize its receipt.",
                {"permissions": ["fetch_news"],
                 "example_objectives": ["Fetch today's India news"]},
                [{"name": "historical fetch", "passed": True}],
                source_node_ids=[source])
            skills.evaluate(version, [{"passed": True}])
            skills.activate(version)

            self.assertEqual(skills.relevant_context(
                "Hello there", available_tools={"fetch_news"}), [])
            self.assertEqual(len(skills.relevant_context(
                "What's the India news today?",
                available_tools={"fetch_news"})), 1)

    def test_quarantine_removes_an_active_skill_from_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = GraphStore(root / "friday.db")
            skills = SkillManager(graph, root / "skills")
            source = graph.record_node("observation", {"result": "bad evidence"})
            version = skills.create_version(
                "Bad Workflow", "Use recall_memory for live news.",
                {"permissions": ["recall_memory"]}, [{"name": "old", "passed": True}],
                source_node_ids=[source])
            skills.evaluate(version, [{"passed": True}])
            skills.activate(version)

            skills.quarantine(version, "historical receipt contained no evidence")

            self.assertEqual(skills.active_context(), [])
            self.assertEqual(skills.list()[0]["status"], "quarantined")


if __name__ == "__main__":
    unittest.main()
