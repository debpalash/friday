import asyncio
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from friday_core import (DurableStepWorker, GraphStore, StepExecutionResult,
                         TaskService)


class _UnknownDispatch(RuntimeError):
    code = "desktop_close_outcome_unknown"
    outcome_unknown = True


class ReconciliationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.graph = GraphStore(Path(self.temporary.name) / "friday.db")
        self.tasks = TaskService(self.graph)
        self.task_id, _ = self.tasks.create(
            "Reconcile an uncertain external action",
            {"version": 0, "evidence": "authoritative postcondition"})
        self.tasks.transition(self.task_id, "interpreting")
        self.tasks.transition(self.task_id, "planned")
        self.tasks.transition(self.task_id, "running")
        self.workers = []

    async def asyncTearDown(self):
        for worker in reversed(self.workers):
            await worker.stop(timeout=0.05)
        self.temporary.cleanup()

    @staticmethod
    def _call(call_id, *, reconcile=True):
        return {
            "tool_call_id": call_id,
            "tool_name": ("machine_close_window" if reconcile
                          else "machine_list_windows"),
            "args": ({"window_id": "win_" + "a" * 40}
                     if reconcile else {}),
            "risk": "high" if reconcile else "read_only",
            "approval_status": "approved" if reconcile else "not_required",
            "idempotency_class": "reconcilable" if reconcile else "read_only",
            "recovery_policy": "reconcile" if reconcile else "retry",
            "executor_binding": ({
                "kind": "desktop_window",
                "operation": "close",
                "window_id": "win_" + "a" * 40,
                "session_fingerprint": "b" * 64,
                "runtime_identity_sha256": "c" * 64,
                "application_id_sha256": "d" * 64,
                "application": "Terminal",
                "workspace_id": 1,
                "args_sha256": "e" * 64,
            } if reconcile else {}),
            "resource_claims": {},
        }

    def _stage(self, *, with_successor=False):
        calls = [self._call("close-unknown")]
        if with_successor:
            calls.append(self._call("list-after", reconcile=False))
        return self.tasks.stage_step_batch(
            self.task_id, calls, round_index=0)

    def _quarantine(self, *, with_successor=False):
        batch_id, steps = self._stage(with_successor=with_successor)
        claim = self.tasks.claim_next_step(batch_id, "uncertain-worker")
        self.assertIsNotNone(claim)
        self.tasks.mark_step_outcome_unknown(
            claim, reason_code="desktop_close_outcome_unknown")
        return batch_id, steps, claim

    async def test_live_unknown_exception_is_quarantined_not_failed(self):
        batch_id, _ = self._stage()

        async def executor(_claim):
            raise _UnknownDispatch()

        worker = DurableStepWorker(
            self.tasks, executor, worker_id="unknown-worker")
        self.workers.append(worker)
        await worker.start(recover_interrupted=False)
        async with asyncio.timeout(1):
            outcome = await worker.submit(batch_id)

        self.assertEqual(outcome.status, "reconcile_required")
        self.assertTrue(outcome.outcomes[0].outcome_unknown)
        self.assertEqual(
            self.tasks.list_steps(batch_id=batch_id)[0]["status"],
            "reconcile_required")
        self.assertNotIn(batch_id, self.tasks.pending_step_batches())
        with self.graph._connect() as conn:
            receipt = conn.execute(
                "SELECT status FROM action_receipts WHERE step_id=?",
                (outcome.outcomes[0].claim.step_id,)).fetchone()
            attempt = conn.execute(
                "SELECT status FROM action_attempts WHERE step_id=?",
                (outcome.outcomes[0].claim.step_id,)).fetchone()
        self.assertEqual(receipt["status"], "outcome_unknown")
        self.assertEqual(attempt["status"], "abandoned")

    async def test_uncertain_verifier_quarantines_reconcilable_step_by_default(self):
        batch_id, _ = self._stage()

        async def executor(_claim):
            return StepExecutionResult(
                result={"status": "ok"}, succeeded=True,
                verification={
                    "status": "uncertain",
                    "summary": "authoritative probe unavailable",
                    "evidence": [], "missing": ["postcondition"],
                    "effects": [],
                })

        worker = DurableStepWorker(
            self.tasks, executor, worker_id="uncertain-verifier-worker")
        self.workers.append(worker)
        await worker.start(recover_interrupted=False)
        async with asyncio.timeout(1):
            outcome = await worker.submit(batch_id)

        self.assertEqual(outcome.status, "reconcile_required")
        self.assertTrue(outcome.outcomes[0].outcome_unknown)
        with self.graph._connect() as conn:
            receipt = conn.execute(
                "SELECT status FROM action_receipts WHERE step_id=?",
                (outcome.outcomes[0].claim.step_id,)).fetchone()
        self.assertEqual(receipt["status"], "outcome_unknown")

    async def test_proven_success_settles_receipt_and_queues_suffix(self):
        batch_id, steps, _claim = self._quarantine(with_successor=True)
        candidate = self.tasks.reconciliation_candidate(steps[0]["step_id"])
        self.assertNotIn("session_fingerprint", repr(candidate))
        result = json.dumps({
            "status": "ok", "verified": True, "operation": "close",
            "window_id": "win_" + "a" * 40,
            "application": "Terminal", "workspace_id": 1,
            "state": "closed", "idempotent_replay": True,
        })
        settled = self.tasks.resolve_reconciliation(
            candidate, result, succeeded=True,
            verification={
                "status": "passed", "summary": "postcondition observed",
                "evidence": ["same-session exact target absent"],
                "missing": [], "effects": [{
                    "kind": "machine_close_window", "verified": True}],
            }, reason_code="desktop_postcondition_observed")

        self.assertEqual(settled["batch_status"], "queued")
        self.assertEqual(
            [item["status"] for item in self.tasks.list_steps(
                batch_id=batch_id)], ["succeeded", "pending"])
        successor = self.tasks.claim_next_step(batch_id, "successor-worker")
        self.assertIsNotNone(successor)
        self.assertEqual(successor.step_id, steps[1]["step_id"])
        with self.assertRaises((PermissionError, ValueError)):
            self.tasks.resolve_reconciliation(
                candidate, result, succeeded=True,
                verification={"status": "passed"})
        with self.graph._connect() as conn:
            event_types = [row[0] for row in conn.execute(
                "SELECT event_type FROM graph_events ORDER BY seq")]
        self.assertEqual(event_types.count("action.reconciled"), 1)
        self.assertEqual(event_types.count("step.reconciled"), 1)

    async def test_concurrent_authoritative_resolvers_commit_exactly_once(self):
        _batch_id, steps, _claim = self._quarantine()
        candidate = self.tasks.reconciliation_candidate(steps[0]["step_id"])
        result = json.dumps({
            "status": "ok", "verified": True, "operation": "close",
            "window_id": "win_" + "a" * 40,
            "application": "Terminal", "workspace_id": 1,
            "state": "closed", "idempotent_replay": True,
        })

        def settle(_index):
            try:
                self.tasks.resolve_reconciliation(
                    candidate, result, succeeded=True,
                    verification={
                        "status": "passed", "summary": "observed",
                        "evidence": [], "missing": [], "effects": [],
                    })
                return "won"
            except (PermissionError, ValueError):
                return "stale"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(settle, range(2)))
        self.assertEqual(sorted(outcomes), ["stale", "won"])
        with self.graph._connect() as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM graph_events WHERE event_type="
                "'action.reconciled'").fetchone()[0], 1)

    async def test_authoritative_failure_skips_suffix(self):
        batch_id, steps, _claim = self._quarantine(with_successor=True)
        candidate = self.tasks.reconciliation_candidate(steps[0]["step_id"])
        settled = self.tasks.resolve_reconciliation(
            candidate, {"status": "failed", "reason": "operator_abandoned"},
            succeeded=False,
            verification={
                "status": "failed", "summary": "success not established",
                "evidence": [], "missing": ["postcondition"], "effects": [],
            }, reason_code="operator_marked_failed", actor="user")
        self.assertEqual(settled["batch_status"], "failed")
        self.assertEqual(
            [item["status"] for item in self.tasks.list_steps(
                batch_id=batch_id)], ["failed", "skipped"])
        with self.graph._connect() as conn:
            receipt = conn.execute(
                "SELECT status FROM action_receipts WHERE step_id=?",
                (steps[0]["step_id"],)).fetchone()
        self.assertEqual(receipt["status"], "failed")

    async def test_operator_abandonment_preserves_unknown_receipt(self):
        batch_id, steps, _claim = self._quarantine(with_successor=True)
        candidate = self.tasks.reconciliation_candidate(steps[0]["step_id"])
        settled = self.tasks.acknowledge_unknown_reconciliation(candidate)
        self.assertEqual(settled["status"], "abandoned_unknown")
        self.assertEqual(settled["receipt_status"], "outcome_unknown")
        self.assertEqual(settled["batch_status"], "failed")
        self.assertEqual(
            [item["status"] for item in self.tasks.list_steps(
                batch_id=batch_id)], ["abandoned_unknown", "skipped"])
        self.assertEqual(self.tasks.list_reconciliations(), [])
        with self.graph._connect() as conn:
            receipt = conn.execute(
                "SELECT status,verification_json FROM action_receipts "
                "WHERE step_id=?", (steps[0]["step_id"],)).fetchone()
            event_count = conn.execute(
                "SELECT COUNT(*) FROM graph_events WHERE event_type="
                "'action.outcome_unknown_acknowledged'").fetchone()[0]
        self.assertEqual(receipt["status"], "outcome_unknown")
        self.assertEqual(
            json.loads(receipt["verification_json"])["status"], "uncertain")
        self.assertEqual(event_count, 1)

    async def test_acknowledgement_rejects_cross_wired_candidate_atomically(self):
        batch_a, steps_a, _claim_a = self._quarantine()
        other_task, _ = self.tasks.create(
            "Second uncertain action", {"version": 0})
        self.tasks.transition(other_task, "interpreting")
        self.tasks.transition(other_task, "planned")
        self.tasks.transition(other_task, "running")
        batch_b, _steps_b = self.tasks.stage_step_batch(
            other_task, [self._call("second-close")], round_index=0)
        claim_b = self.tasks.claim_next_step(batch_b, "second-worker")
        self.tasks.mark_step_outcome_unknown(
            claim_b, reason_code="desktop_close_outcome_unknown")
        candidate_a = self.tasks.reconciliation_candidate(
            steps_a[0]["step_id"])

        def snapshot():
            with self.graph._connect() as conn:
                return {
                    table: [tuple(row) for row in conn.execute(
                        f"SELECT * FROM {table} ORDER BY rowid")]
                    for table in ("task_state", "task_step_batches",
                                  "task_steps", "action_attempts",
                                  "action_receipts", "graph_events")
                }

        before = snapshot()
        with self.assertRaises(PermissionError):
            self.tasks.acknowledge_unknown_reconciliation(
                replace(candidate_a, batch_id=batch_b))
        self.assertEqual(before, snapshot())
        self.assertEqual(self.tasks.step_batch(batch_a)["status"],
                         "reconcile_required")
        self.assertEqual(self.tasks.step_batch(batch_b)["status"],
                         "reconcile_required")
        self.assertEqual(len(self.tasks.list_reconciliations()), 2)

    async def test_acknowledgement_rejects_corrupt_receipt_tuple_atomically(self):
        batch_id, steps, _claim = self._quarantine()
        candidate = self.tasks.reconciliation_candidate(steps[0]["step_id"])
        with self.graph.transaction() as conn:
            conn.execute(
                "UPDATE action_receipts SET tool_name=? WHERE step_id=?",
                ("machine_focus_window", candidate.step_id))

        with self.graph._connect() as conn:
            before = {
                table: [tuple(row) for row in conn.execute(
                    f"SELECT * FROM {table} ORDER BY rowid")]
                for table in ("task_state", "task_step_batches", "task_steps",
                              "action_attempts", "action_receipts",
                              "graph_events")
            }
        with self.assertRaises(PermissionError):
            self.tasks.acknowledge_unknown_reconciliation(candidate)
        with self.graph._connect() as conn:
            after = {
                table: [tuple(row) for row in conn.execute(
                    f"SELECT * FROM {table} ORDER BY rowid")]
                for table in before
            }

        self.assertEqual(before, after)
        self.assertEqual(
            self.tasks.step_batch(batch_id)["status"], "reconcile_required")

    async def test_cancel_race_terminalizes_task_but_keeps_unknown_visible(self):
        batch_id, steps = self._stage()
        claim = self.tasks.claim_next_step(batch_id, "cancel-race-worker")
        cancellation = self.tasks.request_cancel(self.task_id)
        self.assertEqual(cancellation["status"], "cancelling")

        self.tasks.mark_step_outcome_unknown(
            claim, reason_code="desktop_close_outcome_unknown")

        self.assertEqual(self.tasks.get(self.task_id)["status"], "cancelled")
        self.assertEqual(self.tasks.step_batch(batch_id)["status"],
                         "reconcile_required")
        self.assertEqual(
            [item["step_id"] for item in self.tasks.list_reconciliations()],
            [steps[0]["step_id"]])
        with self.graph._connect() as conn:
            receipt = conn.execute(
                "SELECT status FROM action_receipts WHERE step_id=?",
                (steps[0]["step_id"],)).fetchone()
        self.assertEqual(receipt["status"], "outcome_unknown")

    async def test_restart_and_cancel_preserve_visible_unknown_effect(self):
        batch_id, steps, _claim = self._quarantine()
        recovered = self.tasks.recover_interrupted()
        self.assertTrue(recovered)
        self.assertEqual(self.tasks.get(self.task_id)["status"], "waiting_input")
        self.assertEqual(self.tasks.recover_interrupted(), [])
        self.assertNotIn(batch_id, self.tasks.pending_step_batches())

        self.tasks.request_cancel(self.task_id)
        self.assertEqual(self.tasks.get(self.task_id)["status"], "cancelled")
        self.assertEqual(
            self.tasks.step_batch(batch_id)["status"],
            "reconcile_required")
        queue = self.tasks.list_reconciliations()
        self.assertEqual([item["step_id"] for item in queue],
                         [steps[0]["step_id"]])
        encoded = json.dumps(queue)
        self.assertNotIn("session_fingerprint", encoded)
        self.assertNotIn("runtime_identity_sha256", encoded)


if __name__ == "__main__":
    unittest.main()
