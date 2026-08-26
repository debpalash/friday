import tempfile
import unittest
from pathlib import Path

from friday_core import (EvolutionEngine, GraphStore, ReflectionService,
                         SkillManager, TaskService)


class EvolutionTests(unittest.TestCase):
    def test_repeated_verified_workflow_becomes_active_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = GraphStore(root / "friday.db")
            tasks = TaskService(graph)
            for index in range(2):
                task_id, _ = tasks.create(f"inspect folder {index}", {"evidence": "receipt"})
                tasks.transition(task_id, "interpreting")
                tasks.set_plan(task_id, ["list"])
                tasks.transition(task_id, "planned")
                tasks.transition(task_id, "running")
                handle, _ = tasks.begin_action(
                    task_id, "list_files", {"path": f"folder{index}"}, ordinal=1)
                tasks.finish_action(handle, "file.txt", succeeded=True)
                tasks.transition(task_id, "verifying")
                tasks.transition(task_id, "completed")
            skills = SkillManager(graph, root / "skills")
            engine = EvolutionEngine(tasks, ReflectionService(graph), skills)

            result = engine.run_once()
            repeated = engine.run_once()

            self.assertEqual(result["skills_activated"], 1)
            self.assertEqual(repeated["skills_activated"], 0)
            self.assertEqual(len(skills.active_context()), 1)
            self.assertEqual(graph.count_nodes("reflection"), 2)

    def test_single_success_does_not_become_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = GraphStore(root / "friday.db")
            tasks = TaskService(graph)
            task_id, _ = tasks.create("one off", {"evidence": "receipt"})
            tasks.transition(task_id, "interpreting")
            tasks.set_plan(task_id, ["read"])
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")
            handle, _ = tasks.begin_action(task_id, "read_file", {"path": "x"}, ordinal=1)
            tasks.finish_action(handle, "x", succeeded=True)
            tasks.transition(task_id, "verifying")
            tasks.transition(task_id, "completed")
            skills = SkillManager(graph, root / "skills")
            engine = EvolutionEngine(tasks, ReflectionService(graph), skills)
            self.assertEqual(engine.run_once()["skills_activated"], 0)
            self.assertEqual(skills.list(), [])

    def test_empty_receipts_never_become_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = GraphStore(root / "friday.db")
            tasks = TaskService(graph)
            for index in range(2):
                task_id, _ = tasks.create(f"fetch news {index}", {"evidence": "receipt"})
                tasks.transition(task_id, "interpreting")
                tasks.set_plan(task_id, ["recall"])
                tasks.transition(task_id, "planned")
                tasks.transition(task_id, "running")
                handle, _ = tasks.begin_action(
                    task_id, "recall_memory", {"query": "news"}, ordinal=1)
                tasks.finish_action(
                    handle, "(no verified memories found)", succeeded=True)
                tasks.transition(task_id, "verifying")
                tasks.transition(task_id, "completed")
            skills = SkillManager(graph, root / "skills")
            engine = EvolutionEngine(tasks, ReflectionService(graph), skills)

            self.assertEqual(engine.run_once()["skills_activated"], 0)
            self.assertEqual(skills.list(), [])

    def test_read_only_receipts_do_not_satisfy_change_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = GraphStore(root / "friday.db")
            tasks = TaskService(graph)
            for index in range(2):
                task_id, _ = tasks.create(
                    f"change voice to scarlet {index}", {"evidence": "activation"})
                tasks.transition(task_id, "interpreting")
                tasks.set_plan(task_id, ["inspect"])
                tasks.transition(task_id, "planned")
                tasks.transition(task_id, "running")
                handle, _ = tasks.begin_action(
                    task_id, "read_file", {"path": "server.py"}, ordinal=1)
                tasks.finish_action(handle, "source code", succeeded=True)
                tasks.transition(task_id, "verifying")
                tasks.transition(task_id, "completed")
            skills = SkillManager(graph, root / "skills")
            engine = EvolutionEngine(tasks, ReflectionService(graph), skills)

            self.assertEqual(engine.run_once()["skills_activated"], 0)
            self.assertEqual(skills.list(), [])


if __name__ == "__main__":
    unittest.main()
