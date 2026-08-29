import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import server
from friday_core import ApprovalService, GraphStore, TaskService


class DurableApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "friday.db"
        self.graph = GraphStore(self.db_path)
        self.tasks = TaskService(self.graph)
        self.approvals = ApprovalService(self.graph)
        self.task_id, _ = self.tasks.create(
            "Execute only the exactly approved durable actions",
            {"version": 0, "evidence": "durable approval receipts"},
        )

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def call(call_id, args, *, approval_status="pending"):
        return {
            "tool_call_id": call_id,
            "tool_name": "write_file",
            "args": args,
            "risk": "high",
            "approval_status": approval_status,
            "idempotency_class": "non_repeatable",
        }

    def stage(self, *calls):
        return self.tasks.stage_step_batch(
            self.task_id,
            list(calls),
            round_index=0,
            context={"session_id": "approval-session", "turn_id": "turn-1"},
        )

    def request_for(self, step, args):
        return self.approvals.request(
            self.task_id,
            "write_file",
            args,
            "filesystem change",
            step_id=step["step_id"],
        )

    def test_request_rejects_mismatch_and_binds_the_exact_step(self):
        exact_args = {"path": "approved.txt", "content": "exact content"}
        _batch_id, steps = self.stage(self.call("call-exact", exact_args))
        step = steps[0]

        with self.assertRaisesRegex(ValueError, "exact durable step"):
            self.request_for(
                step,
                {"path": "approved.txt", "content": "different content"},
            )
        with self.assertRaisesRegex(ValueError, "exact durable step"):
            self.approvals.request(
                self.task_id,
                "browser_type",
                exact_args,
                "different tool",
                step_id=step["step_id"],
            )

        approval = self.request_for(step, exact_args)
        rebound = self.tasks.list_steps(task_id=self.task_id)[0]

        self.assertEqual(approval["step_id"], step["step_id"])
        self.assertEqual(rebound["approval_id"], approval["approval_id"])
        self.assertEqual(rebound["args_sha256"],
                         approval["args"]["_args_sha256"])

    def test_approval_survives_restart_and_batch_waits_for_every_hash(self):
        first_args = {"path": "first.txt", "content": "first exact payload"}
        second_args = {"path": "second.txt", "content": "second exact payload"}
        batch_id, steps = self.stage(
            self.call("call-first", first_args),
            self.call("call-second", second_args),
        )
        first = self.request_for(steps[0], first_args)
        second = self.request_for(steps[1], second_args)

        restarted_graph = GraphStore(self.db_path)
        restarted_tasks = TaskService(restarted_graph)
        restarted_approvals = ApprovalService(restarted_graph)

        pending_ids = {
            item["approval_id"]
            for item in restarted_approvals.list(status="pending")
        }
        self.assertEqual(pending_ids,
                         {first["approval_id"], second["approval_id"]})
        self.assertEqual(
            restarted_tasks.list_steps(batch_id=batch_id)[0]["approval_id"],
            first["approval_id"],
        )

        first_decision = restarted_approvals.decide(first["approval_id"], True)

        self.assertEqual(first_decision["batch_id"], batch_id)
        self.assertEqual(restarted_tasks.step_batch(batch_id)["status"],
                         "waiting_approval")
        self.assertTrue(restarted_approvals.is_approved(
            self.task_id, "write_file", first_args))
        self.assertFalse(restarted_approvals.is_approved(
            self.task_id,
            "write_file",
            {"path": "first.txt", "content": "unapproved payload"},
        ))
        self.assertFalse(restarted_approvals.is_approved(
            self.task_id, "write_file", second_args))

        second_decision = restarted_approvals.decide(second["approval_id"], True)

        self.assertEqual(second_decision["batch_id"], batch_id)
        self.assertEqual(restarted_tasks.step_batch(batch_id)["status"], "queued")
        self.assertEqual(
            [step["approval_status"]
             for step in restarted_tasks.list_steps(batch_id=batch_id)],
            ["approved", "approved"],
        )
        self.assertTrue(restarted_approvals.is_approved(
            self.task_id, "write_file", second_args))

    def test_denial_cancels_the_whole_suffix_and_invalidates_its_approvals(self):
        prefix_args = {"path": "prefix.txt", "content": "prefix"}
        denied_args = {"path": "denied.txt", "content": "denied"}
        suffix_args = {"path": "suffix.txt", "content": "suffix"}
        batch_id, steps = self.stage(
            self.call("call-prefix", prefix_args, approval_status="not_required"),
            self.call("call-denied", denied_args),
            self.call("call-suffix", suffix_args),
        )
        denied = self.request_for(steps[1], denied_args)
        suffix = self.request_for(steps[2], suffix_args)

        decision = self.approvals.decide(denied["approval_id"], False)

        durable_steps = self.tasks.list_steps(batch_id=batch_id)
        self.assertEqual(decision["batch_id"], batch_id)
        self.assertEqual(self.tasks.step_batch(batch_id)["status"], "cancelled")
        self.assertEqual([step["status"] for step in durable_steps[1:]],
                         ["cancelled", "cancelled"])
        self.assertNotEqual(durable_steps[2]["approval_status"], "pending")
        self.assertNotIn(
            suffix["approval_id"],
            {item["approval_id"]
             for item in self.approvals.list(status="pending")},
        )
        with self.assertRaisesRegex(ValueError, "already decided"):
            self.approvals.decide(suffix["approval_id"], True)
        self.assertEqual(self.tasks.step_batch(batch_id)["status"], "cancelled")


class ApprovalEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_approved_endpoint_enqueues_batch_id_without_replanning(self):
        batch_id = "batch_exact"
        task_id = "task_exact"
        approvals = SimpleNamespace(decide=Mock(return_value={
            "approval_id": "approval_exact",
            "task_id": task_id,
            "step_id": "step_exact",
            "batch_id": batch_id,
            "status": "approved",
        }))
        tasks = SimpleNamespace(
            get=Mock(return_value={"task_id": task_id,
                                   "status": "waiting_input"}),
            step_batch=Mock(return_value={"batch_id": batch_id,
                                          "status": "queued"}),
            transition=Mock(),
        )
        worker = SimpleNamespace(enqueue=AsyncMock())
        friday = SimpleNamespace(respond=AsyncMock())

        with (patch.object(server, "APPROVALS", approvals),
              patch.object(server, "TASKS", tasks),
              patch.object(server, "WORKER", worker),
              patch.object(server, "FRIDAY", friday)):
            decision = await server.api_decide_approval(
                "approval_exact", {"approved": True})

        self.assertEqual(decision["batch_id"], batch_id)
        tasks.step_batch.assert_called_once_with(batch_id)
        tasks.transition.assert_called_once_with(
            task_id,
            "running",
            label="Approval granted; executing exact step",
        )
        worker.enqueue.assert_awaited_once_with(batch_id)
        friday.respond.assert_not_awaited()
        approvals.decide.assert_called_once_with(
            "approval_exact", True, actor="local_user")

    async def test_endpoint_does_not_enqueue_until_all_approvals_are_done(self):
        approvals = SimpleNamespace(decide=Mock(return_value={
            "approval_id": "approval_first",
            "task_id": "task_exact",
            "step_id": "step_first",
            "batch_id": "batch_exact",
            "status": "approved",
        }))
        tasks = SimpleNamespace(
            get=Mock(return_value={"task_id": "task_exact",
                                   "status": "waiting_input"}),
            step_batch=Mock(return_value={"batch_id": "batch_exact",
                                          "status": "waiting_approval"}),
            transition=Mock(),
        )
        worker = SimpleNamespace(enqueue=AsyncMock())

        with (patch.object(server, "APPROVALS", approvals),
              patch.object(server, "TASKS", tasks),
              patch.object(server, "WORKER", worker)):
            await server.api_decide_approval(
                "approval_first", {"approved": True})

        tasks.transition.assert_not_called()
        worker.enqueue.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
