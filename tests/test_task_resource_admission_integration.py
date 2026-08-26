from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

from friday_core import (
    AdmissionBudget,
    DurableStepWorker,
    GraphStore,
    ResourceAdmissionController,
    ResourceSnapshot,
    StepExecutionResult,
    TaskService,
)


def snapshot(*, cpu_millis: int = 8_000, ram_mib: int = 16_384,
             network_slots: int = 8) -> ResourceSnapshot:
    return ResourceSnapshot(
        available_cpu_millis=cpu_millis,
        available_ram_mib=ram_mib,
        available_network_slots=network_slots,
        available_accelerator_vram_mib={"cuda:0": 8_192},
        captured_at=datetime.now(UTC),
    )


def budget(*, cpu_millis: int = 8_000, ram_mib: int = 16_384,
           concurrency_slots: int = 8,
           network_slots: int = 8) -> AdmissionBudget:
    return AdmissionBudget(
        cpu_millis=cpu_millis,
        ram_mib=ram_mib,
        concurrency_slots=concurrency_slots,
        network_slots=network_slots,
        accelerator_vram_mib={"cuda:0": 8_192},
    )


class MutableSnapshotSource:
    def __init__(self, value: ResourceSnapshot):
        self._value = value
        self._lock = threading.Lock()
        self._calls = 0

    def __call__(self) -> ResourceSnapshot:
        with self._lock:
            self._calls += 1
            return self._value

    def set(self, value: ResourceSnapshot) -> None:
        with self._lock:
            self._value = value

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls


class TaskResourceAdmissionIntegrationTests(
        unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.graph = GraphStore(Path(self.tmp.name) / "friday.db")
        self.workers: list[DurableStepWorker] = []

    async def asyncTearDown(self):
        for worker in reversed(self.workers):
            await worker.stop(timeout=0.05)
        self.tmp.cleanup()

    @staticmethod
    def call(call_id: str, *, cpu_cores: float = 1.0,
             ram_mib: int = 512,
             idempotency_class: str = "read_only") -> dict:
        return {
            "tool_call_id": call_id,
            "tool_name": "list_files",
            "args": {"path": f"/tmp/{call_id}"},
            "idempotency_class": idempotency_class,
            "resource_claims": {
                "cpu_cores": cpu_cores,
                "ram_mib": ram_mib,
                "vram_mib": 0,
                "accelerator": "none",
                "network": False,
                "concurrency_slots": 1,
                "latency_class": "interactive",
            },
        }

    def create_task_and_batch(self, tasks: TaskService, call: dict):
        task_id, _ = tasks.create(
            "Exercise integrated resource admission",
            {"version": 0, "evidence": "admitted durable receipt"},
        )
        batch_id, steps = tasks.stage_step_batch(
            task_id, [call], round_index=0)
        return task_id, batch_id, steps[0]

    def table_rows(self, table: str) -> list[dict]:
        if table not in {"resource_leases", "action_attempts",
                         "action_receipts", "task_steps"}:
            raise ValueError("unsupported test table")
        with self.graph._connect() as conn:
            return [dict(row) for row in conn.execute(
                f"SELECT * FROM {table} ORDER BY rowid").fetchall()]

    def task_event_count(self, task_id: str, event_type: str) -> int:
        return sum(
            event["event_type"] == event_type
            for event in self.graph.events_since(task_id=task_id, limit=500)
        )

    async def wait_until(self, predicate, *, timeout: float = 3.0) -> None:
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.01)

    async def start_worker(self, tasks: TaskService, executor, *,
                           worker_id: str) -> DurableStepWorker:
        worker = DurableStepWorker(
            tasks, executor, worker_id=worker_id, lease_seconds=30)
        self.workers.append(worker)
        await worker.start(recover_interrupted=False)
        return worker

    async def test_temporary_deferral_is_deduplicated_and_worker_resumes(self):
        snapshots = MutableSnapshotSource(snapshot(cpu_millis=0))
        controller = ResourceAdmissionController(
            self.graph, budget(), snapshots,
            snapshot_ttl_seconds=0, runtime_id="runtime-deferral")
        tasks = TaskService(self.graph, admission=controller)
        task_id, batch_id, _step = self.create_task_and_batch(
            tasks, self.call("temporarily-deferred"))

        for _ in range(3):
            self.assertIsNone(tasks.claim_next_step(
                batch_id, "manual-deferral-probe", lease_seconds=30))

        self.assertEqual(self.table_rows("action_receipts"), [])
        self.assertEqual(self.table_rows("action_attempts"), [])
        self.assertEqual(self.table_rows("resource_leases"), [])
        self.assertEqual(
            self.task_event_count(task_id, "step.admission_deferred"), 1)

        executed = []

        async def executor(claim):
            executed.append(claim.step_id)
            return StepExecutionResult(result={"entries": []}, succeeded=True)

        calls_before_worker = snapshots.calls
        worker = await self.start_worker(
            tasks, executor, worker_id="automatic-deferral-worker")
        submission = asyncio.create_task(worker.submit(batch_id))
        await self.wait_until(lambda: snapshots.calls > calls_before_worker)
        self.assertFalse(submission.done())
        self.assertEqual(executed, [])
        self.assertEqual(self.table_rows("action_receipts"), [])
        self.assertEqual(self.table_rows("action_attempts"), [])
        self.assertEqual(
            self.task_event_count(task_id, "step.admission_deferred"), 1)

        snapshots.set(snapshot())
        controller.invalidate_snapshot()
        async with asyncio.timeout(5):
            outcome = await submission

        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(len(executed), 1)
        self.assertEqual(len(self.table_rows("action_receipts")), 1)
        self.assertEqual(len(self.table_rows("action_attempts")), 1)
        leases = self.table_rows("resource_leases")
        self.assertEqual(len(leases), 1)
        self.assertEqual(leases[0]["status"], "released")

    async def test_permanent_over_budget_fails_batch_without_dispatch(self):
        controller = ResourceAdmissionController(
            self.graph, budget(cpu_millis=500),
            snapshot_provider=lambda: snapshot(),
            snapshot_ttl_seconds=0, runtime_id="runtime-too-small")
        tasks = TaskService(self.graph, admission=controller)
        task_id, batch_id, _step = self.create_task_and_batch(
            tasks, self.call("permanently-too-large", cpu_cores=1.0))
        executed = []

        async def executor(claim):
            executed.append(claim.step_id)
            return StepExecutionResult(result="must not run", succeeded=True)

        worker = await self.start_worker(
            tasks, executor, worker_id="over-budget-worker")
        async with asyncio.timeout(2):
            outcome = await worker.submit(batch_id)

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(executed, [])
        self.assertEqual(tasks.step_batch(batch_id)["status"], "failed")
        self.assertEqual(
            tasks.list_steps(batch_id=batch_id)[0]["status"], "failed")
        self.assertEqual(self.table_rows("action_receipts"), [])
        self.assertEqual(self.table_rows("action_attempts"), [])
        self.assertEqual(self.table_rows("resource_leases"), [])
        self.assertEqual(
            self.task_event_count(task_id, "step.admission_rejected"), 1)

    async def test_claim_heartbeat_and_finish_share_the_exact_resource_lease(self):
        controller = ResourceAdmissionController(
            self.graph, budget(), snapshot_provider=lambda: snapshot(),
            snapshot_ttl_seconds=0, runtime_id="runtime-lifecycle")
        tasks = TaskService(self.graph, admission=controller)
        _task_id, batch_id, _step = self.create_task_and_batch(
            tasks, self.call("lease-lifecycle"))

        claim = tasks.claim_next_step(
            batch_id, "lease-lifecycle-worker", lease_seconds=30)
        self.assertIsNotNone(claim)
        self.assertTrue(claim.resource_lease_id.startswith("resource_lease_"))

        step_before = self.table_rows("task_steps")[0]
        lease_before = self.table_rows("resource_leases")[0]
        self.assertEqual(step_before["resource_lease_id"],
                         claim.resource_lease_id)
        self.assertEqual(lease_before["lease_id"], claim.resource_lease_id)
        self.assertEqual(lease_before["attempt_id"], claim.attempt_id)
        self.assertEqual(lease_before["worker_id"], claim.worker_id)
        self.assertEqual(lease_before["status"], "active")
        self.assertEqual(step_before["lease_expires_at"],
                         lease_before["expires_at"])

        def heartbeat_then_fail(conn, lease_id, _attempt_id, **_kwargs):
            conn.execute(
                "UPDATE resource_leases SET status='fenced' WHERE lease_id=?",
                (lease_id,))
            raise RuntimeError("injected heartbeat failure")

        with mock.patch.object(
                controller, "heartbeat_in_transaction",
                side_effect=heartbeat_then_fail):
            with self.assertRaisesRegex(RuntimeError, "injected heartbeat"):
                tasks.heartbeat_step(claim, lease_seconds=120)
        step_after_failed_heartbeat = self.table_rows("task_steps")[0]
        lease_after_failed_heartbeat = self.table_rows("resource_leases")[0]
        self.assertEqual(step_after_failed_heartbeat["lease_expires_at"],
                         step_before["lease_expires_at"])
        self.assertEqual(lease_after_failed_heartbeat["status"], "active")
        self.assertEqual(lease_after_failed_heartbeat["expires_at"],
                         lease_before["expires_at"])

        self.assertTrue(tasks.heartbeat_step(claim, lease_seconds=120))
        step_after_heartbeat = self.table_rows("task_steps")[0]
        lease_after_heartbeat = self.table_rows("resource_leases")[0]
        self.assertEqual(lease_after_heartbeat["lease_id"],
                         claim.resource_lease_id)
        self.assertGreater(step_after_heartbeat["lease_expires_at"],
                           step_before["lease_expires_at"])
        self.assertGreater(lease_after_heartbeat["expires_at"],
                           lease_before["expires_at"])

        def release_then_fail(conn, lease_id, _attempt_id, **_kwargs):
            conn.execute(
                "UPDATE resource_leases SET status='released' WHERE lease_id=?",
                (lease_id,))
            raise RuntimeError("injected finish failure")

        with mock.patch.object(
                controller, "release_in_transaction",
                side_effect=release_then_fail):
            with self.assertRaisesRegex(RuntimeError, "injected finish"):
                tasks.finish_step(claim, {"entries": []}, succeeded=True)
        self.assertEqual(self.table_rows("task_steps")[0]["status"], "running")
        self.assertEqual(
            self.table_rows("action_attempts")[0]["status"], "running")
        self.assertEqual(
            self.table_rows("action_receipts")[0]["status"], "running")
        self.assertEqual(
            self.table_rows("resource_leases")[0]["status"], "active")

        tasks.finish_step(claim, {"entries": []}, succeeded=True)
        step_finished = self.table_rows("task_steps")[0]
        lease_finished = self.table_rows("resource_leases")[0]
        receipt = self.table_rows("action_receipts")[0]
        self.assertEqual(step_finished["status"], "succeeded")
        self.assertEqual(receipt["status"], "succeeded")
        self.assertEqual(lease_finished["status"], "released")
        self.assertEqual(lease_finished["release_reason"], "step_succeeded")

    def interrupted_and_retried(self):
        snapshots = MutableSnapshotSource(snapshot())
        old_controller = ResourceAdmissionController(
            self.graph, budget(), snapshots, snapshot_ttl_seconds=0,
            runtime_id="runtime-before-crash")
        old_tasks = TaskService(self.graph, admission=old_controller)
        _task_id, batch_id, _step = self.create_task_and_batch(
            old_tasks, self.call("restart-fenced"))
        stale_claim = old_tasks.claim_next_step(
            batch_id, "worker-before-crash", lease_seconds=300)
        self.assertIsNotNone(stale_claim)

        new_controller = ResourceAdmissionController(
            self.graph, budget(), snapshots, snapshot_ttl_seconds=0,
            runtime_id="runtime-after-crash")
        new_tasks = TaskService(self.graph, admission=new_controller)
        recovered = new_tasks.recover_inflight_steps(
            force=True, dead_worker_id="worker-before-crash")
        self.assertEqual(recovered, {
            "retry": [stale_claim.step_id], "reconcile": []})

        old_lease = next(
            row for row in self.table_rows("resource_leases")
            if row["lease_id"] == stale_claim.resource_lease_id)
        self.assertEqual(old_lease["status"], "fenced")
        replacement = new_tasks.claim_next_step(
            batch_id, "worker-after-crash", lease_seconds=300)
        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.step_id, stale_claim.step_id)
        self.assertEqual(replacement.action_id, stale_claim.action_id)
        self.assertEqual(replacement.attempt_number, 2)
        self.assertNotEqual(replacement.resource_lease_id,
                            stale_claim.resource_lease_id)
        return old_tasks, new_tasks, stale_claim, replacement

    async def test_forced_recovery_fences_old_resource_lease_and_retries_new(self):
        _old_tasks, new_tasks, _stale_claim, replacement = (
            self.interrupted_and_retried())

        leases = self.table_rows("resource_leases")
        self.assertEqual([row["status"] for row in leases].count("fenced"), 1)
        self.assertEqual([row["status"] for row in leases].count("active"), 1)
        new_tasks.finish_step(replacement, {"entries": []}, succeeded=True)
        leases = self.table_rows("resource_leases")
        self.assertEqual([row["status"] for row in leases].count("released"), 1)

    async def test_stale_claim_cannot_renew_finish_or_release_new_resource_lease(self):
        old_tasks, new_tasks, stale_claim, replacement = (
            self.interrupted_and_retried())
        replacement_before = next(
            row for row in self.table_rows("resource_leases")
            if row["lease_id"] == replacement.resource_lease_id)

        self.assertFalse(old_tasks.heartbeat_step(
            stale_claim, lease_seconds=900))
        with self.assertRaisesRegex(PermissionError, "lease is stale"):
            old_tasks.finish_step(
                stale_claim, {"entries": ["stale"]}, succeeded=True)

        replacement_after = next(
            row for row in self.table_rows("resource_leases")
            if row["lease_id"] == replacement.resource_lease_id)
        self.assertEqual(replacement_after["status"], "active")
        self.assertEqual(replacement_after["expires_at"],
                         replacement_before["expires_at"])
        new_tasks.finish_step(replacement, {"entries": []}, succeeded=True)
        replacement_finished = next(
            row for row in self.table_rows("resource_leases")
            if row["lease_id"] == replacement.resource_lease_id)
        self.assertEqual(replacement_finished["status"], "released")

    async def test_worker_cancels_execution_and_recovers_when_heartbeat_loses_lease(self):
        controller = ResourceAdmissionController(
            self.graph, budget(), snapshot_provider=lambda: snapshot(),
            snapshot_ttl_seconds=0, runtime_id="runtime-heartbeat-loss")
        tasks = TaskService(self.graph, admission=controller)
        task_id, batch_id, step = self.create_task_and_batch(
            tasks, self.call("heartbeat-lost"))
        executor_started = asyncio.Event()
        executor_cancelled = asyncio.Event()
        executions = 0

        async def executor(_claim):
            nonlocal executions
            executions += 1
            executor_started.set()
            if executions == 1:
                try:
                    await asyncio.Future()
                finally:
                    executor_cancelled.set()
            return StepExecutionResult(result={"attempt": executions},
                                       succeeded=True)

        worker = DurableStepWorker(
            tasks, executor, worker_id="heartbeat-loss-worker",
            lease_seconds=1, executor_cancel_grace_seconds=0.1)
        self.workers.append(worker)
        await worker.start(recover_interrupted=False)

        with mock.patch.object(tasks, "heartbeat_step", return_value=False):
            submission = asyncio.create_task(worker.submit(batch_id))
            async with asyncio.timeout(1):
                await executor_started.wait()
            async with asyncio.timeout(3):
                outcome = await submission

        self.assertTrue(executor_cancelled.is_set())
        self.assertEqual(executions, 2)
        self.assertEqual(outcome.status, "succeeded")
        self.assertTrue(worker.is_running)
        current = tasks.step_batch(batch_id)
        self.assertEqual(current["status"], "succeeded")
        self.assertEqual(current["steps"][0]["status"], "succeeded")
        self.assertEqual(current["steps"][0]["step_id"], step["step_id"])
        attempts = self.table_rows("action_attempts")
        self.assertEqual(
            [item["status"] for item in attempts], ["abandoned", "succeeded"])
        self.assertEqual(
            self.table_rows("action_receipts")[0]["status"], "succeeded")
        leases = self.table_rows("resource_leases")
        self.assertEqual(
            [item["status"] for item in leases], ["fenced", "released"])
        events = self.graph.events_since(task_id=task_id, limit=500)
        event_types = {event["event_type"] for event in events}
        self.assertIn("step.retry_scheduled", event_types)
        finished = [event for event in events
                    if event["event_type"] == "action.finished"]
        self.assertEqual(len(finished), 1)
        self.assertEqual(
            finished[0]["payload"]["attempt_id"], attempts[1]["attempt_id"])

    async def test_cancellation_resistant_executor_stops_worker_without_false_success(self):
        controller = ResourceAdmissionController(
            self.graph, budget(), snapshot_provider=lambda: snapshot(),
            snapshot_ttl_seconds=0, runtime_id="runtime-resistant-executor")
        tasks = TaskService(self.graph, admission=controller)
        task_id, batch_id, _step = self.create_task_and_batch(
            tasks, self.call("resistant-heartbeat-loss"))
        executor_started = asyncio.Event()
        cancellation_seen = asyncio.Event()
        release_executor = asyncio.Event()

        async def executor(_claim):
            executor_started.set()
            while not release_executor.is_set():
                try:
                    await release_executor.wait()
                except asyncio.CancelledError:
                    cancellation_seen.set()
            # This apparent success must be ignored after lease loss.
            return StepExecutionResult(result={"late": "success"}, succeeded=True)

        worker = DurableStepWorker(
            tasks, executor, worker_id="resistant-executor-worker",
            lease_seconds=1, executor_cancel_grace_seconds=0.01)
        self.workers.append(worker)
        await worker.start(recover_interrupted=False)

        with mock.patch.object(tasks, "heartbeat_step", return_value=False):
            submission = asyncio.create_task(worker.submit(batch_id))
            async with asyncio.timeout(1):
                await executor_started.wait()
            with self.assertRaisesRegex(RuntimeError, "lease was lost"):
                async with asyncio.timeout(3):
                    await submission

        self.assertTrue(cancellation_seen.is_set())
        await self.wait_until(lambda: not worker.is_running, timeout=1)
        started_at = asyncio.get_running_loop().time()
        await worker.stop(timeout=0.01)
        self.assertLess(asyncio.get_running_loop().time() - started_at, 0.3)

        release_executor.set()
        await self.wait_until(lambda: not worker._detached_executors, timeout=1)
        current = tasks.step_batch(batch_id)
        self.assertEqual(current["status"], "queued")
        self.assertEqual(current["steps"][0]["status"], "pending")
        self.assertEqual(
            self.table_rows("action_receipts")[0]["status"], "running")
        self.assertEqual(
            self.table_rows("action_attempts")[0]["status"], "abandoned")
        self.assertEqual(
            self.table_rows("resource_leases")[0]["status"], "fenced")
        event_types = {
            event["event_type"]
            for event in self.graph.events_since(task_id=task_id, limit=500)
        }
        self.assertNotIn("action.finished", event_types)
        self.assertNotIn("step.succeeded", event_types)

    async def test_lease_loss_routes_nonrepeatable_execution_to_reconciliation(self):
        controller = ResourceAdmissionController(
            self.graph, budget(), snapshot_provider=lambda: snapshot(),
            snapshot_ttl_seconds=0, runtime_id="runtime-uncertain-effect")
        tasks = TaskService(self.graph, admission=controller)
        task_id, batch_id, _step = self.create_task_and_batch(
            tasks, self.call(
                "uncertain-heartbeat-loss",
                idempotency_class="non_repeatable"))
        executor_started = asyncio.Event()

        async def executor(_claim):
            executor_started.set()
            await asyncio.Future()

        worker = DurableStepWorker(
            tasks, executor, worker_id="uncertain-effect-worker",
            lease_seconds=1, executor_cancel_grace_seconds=0.1)
        self.workers.append(worker)
        await worker.start(recover_interrupted=False)

        with mock.patch.object(tasks, "heartbeat_step", return_value=False):
            submission = asyncio.create_task(worker.submit(batch_id))
            async with asyncio.timeout(1):
                await executor_started.wait()
            async with asyncio.timeout(3):
                outcome = await submission

        self.assertEqual(outcome.status, "reconcile_required")
        current = tasks.step_batch(batch_id)
        self.assertEqual(current["status"], "reconcile_required")
        self.assertEqual(
            current["steps"][0]["status"], "reconcile_required")
        self.assertEqual(
            self.table_rows("action_receipts")[0]["status"],
            "outcome_unknown")
        self.assertEqual(
            self.table_rows("action_attempts")[0]["status"], "abandoned")
        self.assertEqual(
            self.table_rows("resource_leases")[0]["status"], "fenced")
        event_types = {
            event["event_type"]
            for event in self.graph.events_since(task_id=task_id, limit=500)
        }
        self.assertIn("step.reconciliation_required", event_types)
        self.assertNotIn("action.finished", event_types)


if __name__ == "__main__":
    unittest.main()
