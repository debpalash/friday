import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from friday_core import (DurableStepWorker, GraphStore, StepExecutionResult,
                         TaskService)


class DurableStepWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.graph = GraphStore(Path(self.tmp.name) / "friday.db")
        self.tasks = TaskService(self.graph)
        self.task_id, _ = self.tasks.create(
            "Run a durable worker batch",
            {"version": 0, "evidence": "durable step receipts"},
        )
        self.workers = []

    async def asyncTearDown(self):
        for worker in reversed(self.workers):
            await worker.stop(timeout=0.05)
        self.tmp.cleanup()

    async def test_running_state_tracks_live_executor_loop(self):
        async def executor(_claim):
            return StepExecutionResult(result={"ok": True}, succeeded=True)

        worker = DurableStepWorker(self.tasks, executor)
        self.assertFalse(worker.is_running)
        await worker.start(recover_interrupted=False)
        self.assertTrue(worker.is_running)
        await worker.stop()
        self.assertFalse(worker.is_running)

    @staticmethod
    def call(call_id, *, tool_name="list_files", args=None,
             idempotency_class="read_only"):
        return {
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "args": args if args is not None else {"path": call_id},
            "idempotency_class": idempotency_class,
        }

    def stage(self, *calls):
        return self.tasks.stage_step_batch(
            self.task_id,
            list(calls),
            round_index=0,
            context={"session_id": "session-worker", "turn_id": "turn-worker"},
        )

    async def start_worker(self, executor, **kwargs):
        worker = DurableStepWorker(self.tasks, executor, **kwargs)
        self.workers.append(worker)
        await worker.start(recover_interrupted=False)
        return worker

    async def test_submit_executes_two_staged_calls_in_order_off_caller_task(self):
        batch_id, _ = self.stage(
            self.call("call-first"),
            self.call("call-second"),
        )
        caller_task = asyncio.current_task()
        executions = []

        async def executor(claim):
            executions.append((claim.tool_call_id, asyncio.current_task()))
            await asyncio.sleep(0)
            return StepExecutionResult(
                result={"call": claim.tool_call_id}, succeeded=True)

        worker = await self.start_worker(executor, worker_id="ordered-worker")
        async with asyncio.timeout(1):
            outcome = await worker.submit(batch_id)

        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(
            [call_id for call_id, _ in executions],
            ["call-first", "call-second"],
        )
        self.assertTrue(all(task is not caller_task for _, task in executions))
        self.assertEqual(
            [completed.result for completed in outcome.outcomes],
            [{"call": "call-first"}, {"call": "call-second"}],
        )

    async def test_submit_returns_private_raw_result_but_database_is_redacted(self):
        secret = "private-worker-result-8f790f4d"
        raw_result = {
            "status": "ok",
            "content": secret,
            "url": f"https://private.example/{secret}",
        }
        batch_id, _ = self.stage(
            self.call(
                "call-private-result",
                tool_name="read_file",
                args={"path": "/tmp/worker-private-fixture"},
            )
        )

        async def executor(_claim):
            return StepExecutionResult(result=raw_result, succeeded=True)

        worker = await self.start_worker(executor, worker_id="privacy-worker")
        async with asyncio.timeout(1):
            outcome = await worker.submit(batch_id)

        self.assertIs(outcome.outcomes[0].result, raw_result)
        with self.graph._connect() as conn:
            receipt = conn.execute(
                "SELECT result_json FROM action_receipts WHERE step_id=?",
                (outcome.outcomes[0].claim.step_id,),
            ).fetchone()
            database_dump = "\n".join(conn.iterdump())

        persisted_result = json.loads(receipt["result_json"])
        self.assertTrue(persisted_result["_redacted"])
        self.assertEqual(persisted_result["status"], "ok")
        self.assertEqual(persisted_result["content_characters"], len(secret))
        self.assertNotIn(secret, database_dump)

    async def test_executor_exception_fails_first_step_and_skips_suffix(self):
        batch_id, _ = self.stage(
            self.call("call-fails"),
            self.call("call-must-not-run"),
        )
        executed = []

        async def executor(claim):
            executed.append(claim.tool_call_id)
            raise RuntimeError("executor exploded")

        worker = await self.start_worker(executor, worker_id="failure-worker")
        async with asyncio.timeout(1):
            outcome = await worker.submit(batch_id)

        self.assertEqual(executed, ["call-fails"])
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(len(outcome.outcomes), 1)
        self.assertFalse(outcome.outcomes[0].succeeded)
        self.assertIn("executor exploded", outcome.outcomes[0].result)
        self.assertEqual(
            [step["status"] for step in self.tasks.list_steps(batch_id=batch_id)],
            ["failed", "skipped"],
        )

    async def test_start_discovers_and_retries_interrupted_read_only_batch(self):
        batch_id, _ = self.stage(self.call("call-interrupted"))
        interrupted = self.tasks.claim_next_step(
            batch_id, "worker-before-restart")
        self.assertIsNotNone(interrupted)
        completed = asyncio.Queue()
        retried_claims = []

        async def executor(claim):
            retried_claims.append(claim)
            return StepExecutionResult(result={"recovered": True}, succeeded=True)

        async def completion_hook(outcome):
            await completed.put(outcome)

        worker = DurableStepWorker(
            self.tasks,
            executor,
            worker_id="worker-after-restart",
            completion_hook=completion_hook,
        )
        self.workers.append(worker)
        resumed = await worker.start(dead_worker_id="worker-before-restart")
        async with asyncio.timeout(1):
            outcome = await completed.get()

        self.assertEqual(resumed, [batch_id])
        self.assertEqual(outcome.status, "succeeded")
        self.assertFalse(outcome.recovered_without_raw_results)
        self.assertEqual(len(retried_claims), 1)
        retry = retried_claims[0]
        self.assertEqual(retry.step_id, interrupted.step_id)
        self.assertEqual(retry.action_id, interrupted.action_id)
        self.assertEqual(retry.attempt_number, 2)
        self.assertEqual(self.tasks.step_batch(batch_id)["status"], "succeeded")

    async def test_bounded_stop_cancels_hanging_executor_and_claim_is_recoverable(self):
        batch_id, _ = self.stage(self.call("call-hangs"))
        executor_started = asyncio.Event()
        executor_cancelled = asyncio.Event()

        async def executor(_claim):
            executor_started.set()
            try:
                await asyncio.Future()
            finally:
                executor_cancelled.set()

        worker = await self.start_worker(executor, worker_id="stopping-worker")
        submission = asyncio.create_task(worker.submit(batch_id))
        async with asyncio.timeout(1):
            await executor_started.wait()

        started_at = asyncio.get_running_loop().time()
        await worker.stop(timeout=0.01)
        elapsed = asyncio.get_running_loop().time() - started_at

        self.assertLess(elapsed, 0.5)
        self.assertTrue(executor_cancelled.is_set())
        with self.assertRaises(asyncio.CancelledError):
            await submission
        running = self.tasks.list_steps(batch_id=batch_id)[0]
        self.assertEqual(running["status"], "running")
        self.assertEqual(running["recovery_policy"], "retry")

        recovered = self.tasks.recover_inflight_steps(
            force=True, dead_worker_id="stopping-worker")
        self.assertEqual(recovered, {"retry": [running["step_id"]],
                                     "reconcile": []})
        self.assertEqual(
            self.tasks.list_steps(batch_id=batch_id)[0]["status"], "pending")

    async def test_unknown_settlement_failure_recovers_to_reconciliation(self):
        batch_id, _ = self.stage(
            self.call(
                "call-unknown-settlement-fails",
                idempotency_class="reconcilable"),
            self.call("call-must-remain-undispatched"),
        )
        executed = []

        async def executor(claim):
            executed.append(claim.tool_call_id)
            return StepExecutionResult(
                result="error: injected_action_outcome_unknown",
                succeeded=False,
                verification={"status": "uncertain"},
                outcome_unknown=True)

        worker = await self.start_worker(
            executor, worker_id="unknown-settlement-worker")
        with mock.patch.object(
                self.tasks, "mark_step_outcome_unknown",
                side_effect=RuntimeError("injected receipt commit failure")):
            async with asyncio.timeout(1):
                outcome = await worker.submit(batch_id)

        self.assertEqual(outcome.status, "reconcile_required")
        self.assertTrue(worker.is_running)
        self.assertEqual(executed, ["call-unknown-settlement-fails"])
        durable = self.tasks.step_batch(batch_id)
        self.assertEqual(durable["status"], "reconcile_required")
        self.assertEqual(durable["steps"][0]["status"], "reconcile_required")
        self.assertEqual(durable["steps"][1]["status"], "pending")

    async def test_unknown_signal_overrides_stale_retry_policy_on_recovery(self):
        batch_id, _ = self.stage(self.call("call-unknown-but-marked-retry"))
        executions = 0

        async def executor(_claim):
            nonlocal executions
            executions += 1
            return StepExecutionResult(
                result="error: external_action_outcome_unknown",
                succeeded=False, outcome_unknown=True)

        worker = await self.start_worker(
            executor, worker_id="unknown-retry-policy-worker")
        with mock.patch.object(
                self.tasks, "mark_step_outcome_unknown",
                side_effect=RuntimeError("injected first quarantine failure")):
            async with asyncio.timeout(1):
                outcome = await worker.submit(batch_id)

        self.assertEqual(outcome.status, "reconcile_required")
        self.assertEqual(executions, 1)
        durable = self.tasks.step_batch(batch_id)
        self.assertEqual(durable["status"], "reconcile_required")
        self.assertEqual(durable["steps"][0]["status"], "reconcile_required")

    async def test_cancel_race_terminalizes_forced_unknown_recovery(self):
        batch_id, _ = self.stage(self.call("call-unknown-cancel-race"))

        async def executor(_claim):
            return StepExecutionResult(
                result="error: external_action_outcome_unknown",
                succeeded=False, outcome_unknown=True)

        def cancel_then_fail(*_args, **_kwargs):
            self.tasks.request_cancel(self.task_id)
            raise RuntimeError("injected quarantine failure after cancellation")

        worker = await self.start_worker(
            executor, worker_id="unknown-cancel-race-worker")
        with mock.patch.object(
                self.tasks, "mark_step_outcome_unknown",
                side_effect=cancel_then_fail):
            async with asyncio.timeout(1):
                outcome = await worker.submit(batch_id)

        self.assertEqual(outcome.status, "reconcile_required")
        self.assertEqual(self.tasks.get(self.task_id)["status"], "cancelled")
        durable = self.tasks.step_batch(batch_id)
        self.assertEqual(durable["status"], "reconcile_required")
        self.assertEqual(durable["steps"][0]["status"], "reconcile_required")

    async def test_known_settlement_failure_never_replays_reconcilable_effect(self):
        batch_id, _ = self.stage(self.call(
            "call-known-settlement-fails",
            idempotency_class="reconcilable"))
        executions = 0

        async def executor(_claim):
            nonlocal executions
            executions += 1
            return StepExecutionResult(result={"effect": "done"}, succeeded=True)

        worker = await self.start_worker(
            executor, worker_id="known-settlement-worker")
        with mock.patch.object(
                self.tasks, "finish_step",
                side_effect=RuntimeError("injected receipt commit failure")):
            async with asyncio.timeout(1):
                outcome = await worker.submit(batch_id)

        self.assertEqual(outcome.status, "reconcile_required")
        self.assertEqual(executions, 1)
        self.assertEqual(
            self.tasks.step_batch(batch_id)["steps"][0]["status"],
            "reconcile_required")

    async def test_settlement_and_recovery_failure_stops_authoritative_loop(self):
        batch_id, _ = self.stage(self.call(
            "call-poisons-worker", idempotency_class="reconcilable"))

        async def executor(_claim):
            return StepExecutionResult(
                result="error: external_action_outcome_unknown",
                succeeded=False, outcome_unknown=True)

        worker = await self.start_worker(
            executor, worker_id="poisoned-settlement-worker")
        with (mock.patch.object(
                self.tasks, "mark_step_outcome_unknown",
                side_effect=RuntimeError("injected receipt commit failure")),
              mock.patch.object(
                self.tasks, "recover_inflight_steps",
                side_effect=RuntimeError("injected recovery failure"))):
            with self.assertRaisesRegex(RuntimeError, "lease was lost"):
                async with asyncio.timeout(1):
                    await worker.submit(batch_id)

        for _ in range(100):
            if not worker.is_running:
                break
            await asyncio.sleep(0.01)
        self.assertFalse(worker.is_running)
        self.assertEqual(
            self.tasks.step_batch(batch_id)["steps"][0]["status"], "running")

    async def test_stop_drains_blocked_receipt_commit_thread(self):
        batch_id, _ = self.stage(self.call("call-blocked-finish"))
        commit_started = threading.Event()
        release_commit = threading.Event()
        original_finish = self.tasks.finish_step

        def blocked_finish(*args, **kwargs):
            commit_started.set()
            if not release_commit.wait(timeout=2):
                raise RuntimeError("blocked finish timed out")
            return original_finish(*args, **kwargs)

        async def executor(_claim):
            return StepExecutionResult(result={"ok": True}, succeeded=True)

        worker = await self.start_worker(
            executor, worker_id="blocked-finish-worker")
        with mock.patch.object(
                self.tasks, "finish_step", side_effect=blocked_finish):
            submission = asyncio.create_task(worker.submit(batch_id))
            for _ in range(100):
                if commit_started.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(commit_started.is_set())
            stop = asyncio.create_task(worker.stop(timeout=0.01))
            await asyncio.sleep(0.03)
            self.assertFalse(stop.done())
            stop.cancel()
            await asyncio.sleep(0.02)
            self.assertFalse(stop.done())
            release_commit.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(stop, 1)

        with self.assertRaises(asyncio.CancelledError):
            await submission
        before = self.tasks.step_batch(batch_id)
        await asyncio.sleep(0.03)
        self.assertEqual(before, self.tasks.step_batch(batch_id))
        self.assertEqual(before["status"], "succeeded")

    async def test_stop_drains_blocked_fallback_recovery_thread(self):
        batch_id, _ = self.stage(self.call("call-blocked-recovery"))
        recovery_started = threading.Event()
        release_recovery = threading.Event()
        original_recovery = self.tasks.recover_inflight_steps

        def blocked_recovery(*args, **kwargs):
            recovery_started.set()
            if not release_recovery.wait(timeout=2):
                raise RuntimeError("blocked recovery timed out")
            return original_recovery(*args, **kwargs)

        async def executor(_claim):
            return StepExecutionResult(
                result="error: external_action_outcome_unknown",
                succeeded=False, outcome_unknown=True)

        worker = await self.start_worker(
            executor, worker_id="blocked-recovery-worker")
        with (mock.patch.object(
                self.tasks, "mark_step_outcome_unknown",
                side_effect=RuntimeError("injected quarantine failure")),
              mock.patch.object(
                self.tasks, "recover_inflight_steps",
                side_effect=blocked_recovery)):
            submission = asyncio.create_task(worker.submit(batch_id))
            for _ in range(100):
                if recovery_started.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(recovery_started.is_set())
            stop = asyncio.create_task(worker.stop(timeout=0.01))
            await asyncio.sleep(0.03)
            self.assertFalse(stop.done())
            release_recovery.set()
            await asyncio.wait_for(stop, 1)

        with self.assertRaises(asyncio.CancelledError):
            await submission
        durable = self.tasks.step_batch(batch_id)
        self.assertEqual(durable["status"], "reconcile_required")
        self.assertEqual(durable["steps"][0]["status"], "reconcile_required")


if __name__ == "__main__":
    unittest.main()
