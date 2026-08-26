"""Lease-owning background recovery worker for interrupted tasks."""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from collections.abc import Awaitable, Callable

from .tasks import ClaimedStep, TERMINAL, TaskService

Runner = Callable[[str, dict], Awaitable[None]]
StepExecutor = Callable[[ClaimedStep], Awaitable["StepExecutionResult"]]
ProgressSink = Callable[[dict[str, Any]], Awaitable[None]]
CompletionHook = Callable[["BatchExecutionOutcome"], Awaitable[None]]


class _StepLeaseLost(RuntimeError):
    """Internal control-flow signal for an execution that lost its fence."""

    def __init__(self, *, executor_stopped: bool, recovery_succeeded: bool):
        super().__init__(
            "durable step lease was lost; execution was cancelled and "
            "durable recovery was scheduled")
        self.executor_stopped = executor_stopped
        self.recovery_succeeded = recovery_succeeded

    @property
    def requires_worker_stop(self) -> bool:
        # A task that ignored cancellation may still be consuming resources or
        # producing an external effect.  Likewise, failed durable recovery means
        # this process no longer has a trustworthy dispatch state.  Taking the
        # worker out of readiness lets the supervisor recycle the whole process.
        return not self.executor_stopped or not self.recovery_succeeded


@dataclass(frozen=True, repr=False)
class StepExecutionResult:
    """Ephemeral executor result; raw content must never be logged or persisted."""

    result: Any
    succeeded: bool
    verification: dict[str, Any] | None = None
    effects: list[dict[str, Any]] | None = None
    outcome_unknown: bool = False


@dataclass(frozen=True, repr=False)
class CompletedStep:
    claim: ClaimedStep
    result: Any
    succeeded: bool
    verification: dict[str, Any] | None
    outcome_unknown: bool = False


@dataclass(frozen=True, repr=False)
class BatchExecutionOutcome:
    batch_id: str
    status: str
    outcomes: tuple[CompletedStep, ...]
    recovered_without_raw_results: bool = False


class DurableStepWorker:
    """Execute persisted step batches; the queue is only a latency hint.

    SQLite is authoritative.  A claim and its dispatch marker commit together,
    and every finish is fenced by the per-attempt lease.  Raw results exist only
    in an ephemeral Future for the live conversation continuation.
    """

    def __init__(self, tasks: TaskService, executor: StepExecutor, *,
                 worker_id: str | None = None, lease_seconds: int = 300,
                 completion_hook: CompletionHook | None = None,
                 executor_cancel_grace_seconds: float = 1.0):
        if executor_cancel_grace_seconds < 0:
            raise ValueError("executor_cancel_grace_seconds cannot be negative")
        self.tasks = tasks
        self.executor = executor
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex}"
        self.lease_seconds = lease_seconds
        self.completion_hook = completion_hook
        self.executor_cancel_grace_seconds = float(
            executor_cancel_grace_seconds)
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._loop_task: asyncio.Task | None = None
        self._queued: set[str] = set()
        self._waiters: dict[str, asyncio.Future[BatchExecutionOutcome]] = {}
        self._progress_sinks: dict[str, ProgressSink] = {}
        self._deferred_requeues: dict[str, asyncio.Task] = {}
        self._detached_executors: set[asyncio.Task] = set()
        self._durable_io_tasks: set[asyncio.Task] = set()

    @property
    def is_running(self) -> bool:
        """Whether the authoritative executor loop is alive and accepting work."""
        return self._loop_task is not None and not self._loop_task.done()

    async def start(self, *, recover_interrupted: bool = True,
                    dead_worker_id: str | None = None) -> list[str]:
        if self.is_running:
            return []
        # A completed loop must never be mistaken for a healthy worker. Clear
        # the stale reference so start() can construct a replacement.
        self._loop_task = None
        if recover_interrupted:
            await self._durable_io(
                self.tasks.recover_inflight_steps,
                force=True, dead_worker_id=dead_worker_id)
        self._loop_task = asyncio.create_task(
            self._loop(), name=f"friday-durable-step-worker:{self.worker_id}")
        resumed = self.tasks.pending_step_batches()
        for batch_id in resumed:
            await self.enqueue(batch_id)
        return resumed

    async def stop(self, *, timeout: float = 10.0) -> None:
        loop_task = self._loop_task
        if loop_task is None:
            await self._drain_durable_io()
            return
        await self.queue.put(None)
        try:
            await asyncio.wait_for(asyncio.shield(loop_task), timeout=timeout)
        except TimeoutError:
            loop_task.cancel()
            await asyncio.gather(loop_task, return_exceptions=True)
        except asyncio.CancelledError:
            # Cancelling stop must cancel the loop, not merely the shielded
            # waiter, and must still wait for any in-flight durable mutation.
            loop_task.cancel()
            try:
                await asyncio.shield(loop_task)
            except asyncio.CancelledError:
                # A repeated cancellation cannot be allowed to abandon the
                # tracked SQLite/admission thread; the finally barrier drains it.
                pass
            except Exception:
                pass
            raise
        finally:
            self._loop_task = None
            deferred = list(self._deferred_requeues.values())
            self._deferred_requeues.clear()
            for task in deferred:
                task.cancel()
            if deferred:
                await asyncio.gather(*deferred, return_exceptions=True)
            for future in self._waiters.values():
                if not future.done():
                    future.cancel()
            self._waiters.clear()
            self._progress_sinks.clear()
            # Cancellation-resistant executors must not make stop unbounded.
            # Re-signal cancellation, but leave process-level supervision as the
            # final fence if user code continues to suppress it.
            for task in tuple(self._detached_executors):
                task.cancel()
            await self._drain_durable_io()

    async def _durable_io(self, function, /, *args, **kwargs):
        """Run one SQLite/admission mutation without abandoning its thread."""
        operation = asyncio.create_task(
            asyncio.to_thread(function, *args, **kwargs),
            name=f"durable-step-io:{self.worker_id}")
        self._durable_io_tasks.add(operation)
        operation.add_done_callback(self._durable_io_tasks.discard)
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            # Thread cancellation is not real cancellation.  Drain the exact
            # mutation before allowing worker/server shutdown to advance.  If
            # it failed, surface that failure so the caller can run fenced
            # recovery instead of losing an outcome-unknown signal.
            try:
                await asyncio.shield(operation)
            except Exception:
                raise
            raise

    async def _drain_durable_io(self) -> None:
        cancelled = False
        while True:
            pending = [task for task in tuple(self._durable_io_tasks)
                       if not task.done()]
            if not pending:
                if cancelled:
                    raise asyncio.CancelledError
                return
            aggregate = asyncio.gather(*pending, return_exceptions=True)
            while not aggregate.done():
                try:
                    await asyncio.shield(aggregate)
                except asyncio.CancelledError:
                    # Do not let cancellation propagate into gather: cancelling
                    # a to_thread Task only hides its still-running thread.
                    cancelled = True

    async def enqueue(self, batch_id: str) -> None:
        if batch_id not in self._queued:
            self._queued.add(batch_id)
            await self.queue.put(batch_id)

    def _schedule_deferred_requeue(self, batch_id: str,
                                   next_admission_at: str | None) -> None:
        prior = self._deferred_requeues.get(batch_id)
        if prior is not None and not prior.done():
            return
        delay = 1.0
        if next_admission_at:
            try:
                target = datetime.fromisoformat(
                    next_admission_at.replace("Z", "+00:00"))
                delay = max(0.05, min(30.0, (
                    target - datetime.now(UTC)).total_seconds()))
            except ValueError:
                pass

        async def requeue() -> None:
            try:
                await asyncio.sleep(delay)
                if self.is_running:
                    await self.enqueue(batch_id)
            finally:
                self._deferred_requeues.pop(batch_id, None)

        self._deferred_requeues[batch_id] = asyncio.create_task(
            requeue(), name=f"admission-requeue:{batch_id}")

    async def submit(self, batch_id: str, *,
                     progress_sink: ProgressSink | None = None
                     ) -> BatchExecutionOutcome:
        if not self.is_running:
            raise RuntimeError("durable step worker is not running")
        loop = asyncio.get_running_loop()
        future = self._waiters.get(batch_id)
        if future is None or future.done():
            future = loop.create_future()
            self._waiters[batch_id] = future
        if progress_sink is not None:
            self._progress_sinks[batch_id] = progress_sink
        await self.enqueue(batch_id)
        return await asyncio.shield(future)

    async def _heartbeat(self, claim: ClaimedStep) -> None:
        interval = max(1.0, min(30.0, self.lease_seconds / 3))
        while True:
            await asyncio.sleep(interval)
            try:
                alive = await self._durable_io(
                    self.tasks.heartbeat_step, claim,
                    lease_seconds=self.lease_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A heartbeat error cannot be distinguished safely from losing
                # the task/resource fence.  Fail closed and never let a result
                # from this execution be committed as a normal success.
                raise _StepLeaseLost(
                    executor_stopped=False,
                    recovery_succeeded=False) from exc
            if not alive:
                raise _StepLeaseLost(
                    executor_stopped=False,
                    recovery_succeeded=False)

    @staticmethod
    def _consume_detached_executor(task: asyncio.Task) -> None:
        """Retrieve a detached task's result so it cannot emit log warnings."""
        try:
            task.result()
        except BaseException:
            pass

    def _detach_executor(self, task: asyncio.Task) -> None:
        if task.done():
            self._consume_detached_executor(task)
            return
        self._detached_executors.add(task)

        def finished(completed: asyncio.Task) -> None:
            self._detached_executors.discard(completed)
            self._consume_detached_executor(completed)

        task.add_done_callback(finished)

    async def _cancel_executor_after_lease_loss(
            self, task: asyncio.Task) -> bool:
        """Cancel an executor and wait only for the configured grace period."""
        task.cancel()
        done, _ = await asyncio.wait(
            {task}, timeout=self.executor_cancel_grace_seconds)
        if task in done:
            self._consume_detached_executor(task)
            return True
        # Do not let cancellation-suppressing tool code pin the authoritative
        # worker loop or shutdown.  Its eventual result is deliberately ignored.
        self._detach_executor(task)
        return False

    async def _recover_after_lease_loss(
            self, claim: ClaimedStep, *, force_reconcile: bool = False) -> bool:
        try:
            await self._durable_io(
                self.tasks.recover_inflight_steps,
                force=True, dead_worker_id=claim.worker_id,
                force_reconcile_step_id=(claim.step_id
                                         if force_reconcile else None))
            return True
        except Exception:
            return False

    async def _invoke_executor(self, claim: ClaimedStep) -> StepExecutionResult:
        try:
            outcome = await self.executor(claim)
            if not isinstance(outcome, StepExecutionResult):
                raise TypeError("step executor must return StepExecutionResult")
            return outcome
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if bool(getattr(exc, "outcome_unknown", False)):
                raw_code = str(getattr(exc, "code", "") or "").lower()
                code = (raw_code if re.fullmatch(
                    r"[a-z0-9][a-z0-9_.:-]{0,79}", raw_code)
                        else "external_action_outcome_unknown")
                return StepExecutionResult(
                    result=f"error: {code}", succeeded=False,
                    verification={
                        "status": "uncertain",
                        "summary": "external action outcome is unknown",
                        "evidence": [],
                        "missing": ["authoritative postcondition evidence"],
                        "effects": [],
                    }, outcome_unknown=True)
            return StepExecutionResult(
                result=f"error: {exc}", succeeded=False,
                verification={"status": "failed",
                              "summary": "step executor raised an exception",
                              "evidence": [],
                              "missing": ["successful receipt"],
                              "effects": []})

    async def _execute_claim(self, claim: ClaimedStep) -> CompletedStep:
        execution = asyncio.create_task(
            self._invoke_executor(claim),
            name=f"step-executor:{claim.step_id}")
        heartbeat = asyncio.create_task(
            self._heartbeat(claim), name=f"step-heartbeat:{claim.step_id}")
        try:
            done, _ = await asyncio.wait(
                {execution, heartbeat},
                return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                # Retrieve the heartbeat exception, but expose only a stable,
                # non-sensitive lease-loss error to callers.
                try:
                    heartbeat.result()
                except BaseException:
                    pass
                executor_stopped = await self._cancel_executor_after_lease_loss(
                    execution)
                recovery_succeeded = await self._recover_after_lease_loss(claim)
                raise _StepLeaseLost(
                    executor_stopped=executor_stopped,
                    recovery_succeeded=recovery_succeeded)

            outcome = execution.result()
            verification_status = str(
                (outcome.verification or {}).get("status") or "")
            verification_unknown = bool(
                claim.recovery_policy == "reconcile"
                and verification_status in {
                    "uncertain", "user_confirmation_required"})
            outcome_unknown = bool(
                outcome.outcome_unknown or verification_unknown)
            try:
                if outcome_unknown:
                    reason = (
                        re.sub(r"^error:\s*", "", str(outcome.result)).strip()
                        if outcome.outcome_unknown
                        else "authoritative_verification_uncertain")
                    progress = await self._durable_io(
                        self.tasks.mark_step_outcome_unknown, claim,
                        reason_code=reason)
                else:
                    progress = await self._durable_io(
                        self.tasks.finish_step, claim, outcome.result,
                        succeeded=outcome.succeeded,
                        verification=outcome.verification,
                        effects=outcome.effects)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # The executor has returned, so a machine effect may already
                # exist.  A failed receipt commit must be recovered under the
                # exact claim fence; if recovery itself fails, poison the
                # authoritative loop so readiness forces process replacement.
                recovery_succeeded = await self._recover_after_lease_loss(
                    claim, force_reconcile=outcome_unknown)
                raise _StepLeaseLost(
                    executor_stopped=True,
                    recovery_succeeded=recovery_succeeded) from exc
            sink = self._progress_sinks.get(claim.batch_id)
            if sink is not None:
                await sink(progress)
            effective_success = bool(
                outcome.succeeded and
                (outcome.verification is None
                 or str(outcome.verification.get("status")) == "passed"))
            return CompletedStep(
                claim=claim, result=outcome.result,
                succeeded=effective_success,
                verification=outcome.verification,
                outcome_unknown=outcome_unknown)
        except asyncio.CancelledError:
            # Worker shutdown remains bounded even if executor code suppresses
            # cancellation.  The still-running durable claim is recovered by the
            # normal startup/dead-worker path after process fencing.
            execution.cancel()
            self._detach_executor(execution)
            raise
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _run_batch(self, batch_id: str) -> BatchExecutionOutcome | None:
        outcomes: list[CompletedStep] = []
        while True:
            claim = await self._durable_io(
                self.tasks.claim_next_step, batch_id, self.worker_id,
                lease_seconds=self.lease_seconds)
            if claim is not None:
                outcome = await self._execute_claim(claim)
                outcomes.append(outcome)
                if not outcome.succeeded:
                    break
                continue
            state = self.tasks.step_batch(batch_id)
            if state is None:
                raise ValueError("durable step batch disappeared")
            status = str(state["status"])
            if status in {"succeeded", "failed", "cancelled",
                          "reconcile_required"}:
                return BatchExecutionOutcome(
                    batch_id=batch_id, status=status,
                    outcomes=tuple(outcomes),
                    recovered_without_raw_results=(
                        not outcomes and status == "succeeded"))
            deferred = next((
                step for step in state.get("steps", [])
                if step.get("status") == "pending"
                and step.get("admission_state") == "deferred"), None)
            if deferred is not None:
                self._schedule_deferred_requeue(
                    batch_id, deferred.get("next_admission_at"))
                return None
            # Another worker owns an unexpired running step or the batch is
            # awaiting an approval.  Its durable state will be re-enqueued by
            # the owner/approval endpoint or on restart.
            return None
        state = self.tasks.step_batch(batch_id)
        return BatchExecutionOutcome(
            batch_id=batch_id,
            status=str(state["status"] if state else "failed"),
            outcomes=tuple(outcomes))

    async def _resolve(self, outcome: BatchExecutionOutcome) -> None:
        future = self._waiters.pop(outcome.batch_id, None)
        self._progress_sinks.pop(outcome.batch_id, None)
        if future is not None and not future.done():
            future.set_result(outcome)
        elif self.completion_hook is not None:
            # Never await model continuation in the single executor loop: it may
            # stage and submit the next round.  Schedule it independently.
            asyncio.create_task(
                self.completion_hook(outcome),
                name=f"step-batch-continuation:{outcome.batch_id}")

    async def _loop(self) -> None:
        while True:
            batch_id = await self.queue.get()
            if batch_id is None:
                return
            self._queued.discard(batch_id)
            try:
                outcome = await self._run_batch(batch_id)
                if outcome is not None:
                    await self._resolve(outcome)
            except asyncio.CancelledError:
                raise
            except _StepLeaseLost as exc:
                if exc.requires_worker_stop:
                    future = self._waiters.pop(batch_id, None)
                    self._progress_sinks.pop(batch_id, None)
                    if future is not None and not future.done():
                        future.set_exception(exc)
                    # Health readiness observes the completed loop and asks the
                    # supervisor to replace the process, which is the only hard
                    # fence for cancellation-resistant in-process tool code.
                    return
                state = self.tasks.step_batch(batch_id)
                status = str(state["status"] if state else "failed")
                if status == "queued":
                    # Cooperative cancellation completed and recovery selected a
                    # safe retry. Keep the original submit waiter and resume with
                    # a freshly fenced attempt instead of silently stranding it.
                    await self.enqueue(batch_id)
                elif status in {
                        "succeeded", "failed", "cancelled",
                        "reconcile_required"}:
                    await self._resolve(BatchExecutionOutcome(
                        batch_id=batch_id, status=status, outcomes=()))
                else:
                    future = self._waiters.pop(batch_id, None)
                    self._progress_sinks.pop(batch_id, None)
                    if future is not None and not future.done():
                        future.set_exception(exc)
            except Exception as exc:
                future = self._waiters.pop(batch_id, None)
                self._progress_sinks.pop(batch_id, None)
                if future is not None and not future.done():
                    future.set_exception(exc)


class BackgroundTaskWorker:
    def __init__(self, tasks: TaskService, runner: Runner):
        self.tasks = tasks
        self.runner = runner
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._loop_task: asyncio.Task | None = None
        self._queued: set[str] = set()

    async def start(self) -> list[str]:
        self.tasks.recover_interrupted()
        self._loop_task = asyncio.create_task(self._loop(), name="friday-task-worker")
        resumed = []
        for task in self.tasks.nonterminal():
            if task["status"] == "recovering":
                await self.enqueue(task["task_id"])
                resumed.append(task["task_id"])
        return resumed

    async def stop(self) -> None:
        if self._loop_task is None:
            return
        await self.queue.put(None)
        await self._loop_task
        self._loop_task = None

    async def enqueue(self, task_id: str) -> None:
        if task_id not in self._queued:
            self._queued.add(task_id)
            await self.queue.put(task_id)

    async def _loop(self) -> None:
        while True:
            task_id = await self.queue.get()
            if task_id is None:
                return
            self._queued.discard(task_id)
            state = self.tasks.get(task_id)
            if state is None or state["status"] in TERMINAL:
                continue
            try:
                if state["status"] == "recovering":
                    self.tasks.transition(
                        task_id, "running", expected_status="recovering",
                        label="Resuming interrupted task")
                self.tasks.acquire_lease(task_id)
                await self.runner(task_id, self.tasks.get(task_id))
            except Exception as exc:
                current = self.tasks.get(task_id)
                if current and current["status"] not in TERMINAL:
                    try:
                        self.tasks.transition(
                            task_id, "failed", label="Recovered task failed",
                            detail=str(exc)[:180], error=str(exc)[:1000])
                    except ValueError:
                        self.tasks.publish(task_id, "recovery", "failed",
                                           "Recovered task failed", str(exc)[:180])
