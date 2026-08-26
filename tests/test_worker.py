import asyncio
import tempfile
import unittest
from pathlib import Path

from friday_core import BackgroundTaskWorker, GraphStore, TaskService


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupted_task_is_resumed_by_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = GraphStore(Path(tmp) / "friday.db")
            tasks = TaskService(graph)
            task_id, _ = tasks.create("continue me", {"evidence": "runner"})
            tasks.transition(task_id, "interpreting")
            tasks.set_plan(task_id, ["resume"])
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")
            ran = asyncio.Event()

            async def runner(got_id, state):
                self.assertEqual(got_id, task_id)
                self.assertEqual(state["status"], "running")
                tasks.transition(got_id, "verifying")
                tasks.transition(got_id, "completed")
                ran.set()

            worker = BackgroundTaskWorker(tasks, runner)
            resumed = await worker.start()
            await asyncio.wait_for(ran.wait(), 1)
            await worker.stop()

            self.assertEqual(resumed, [task_id])
            self.assertEqual(tasks.get(task_id)["status"], "completed")

    async def test_runner_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = GraphStore(Path(tmp) / "friday.db")
            tasks = TaskService(graph)
            task_id, _ = tasks.create("fail me", {"evidence": "runner"})
            tasks.transition(task_id, "interpreting")
            tasks.set_plan(task_id, ["resume"])
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")

            async def runner(_task_id, _state):
                raise RuntimeError("boom")

            worker = BackgroundTaskWorker(tasks, runner)
            await worker.start()
            for _ in range(100):
                if tasks.get(task_id)["status"] == "failed":
                    break
                await asyncio.sleep(0.01)
            await worker.stop()
            self.assertEqual(tasks.get(task_id)["status"], "failed")
            self.assertIn("boom", tasks.get(task_id)["last_error"])


if __name__ == "__main__":
    unittest.main()
