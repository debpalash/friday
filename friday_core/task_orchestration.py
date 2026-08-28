"""Task lifecycle composition above durable storage and worker execution."""

from __future__ import annotations

from .cognition import ContractBuilder, OutcomeVerifier, TaskContract
from .graph import GraphStore
from .tasks import TaskService
from .worker import BatchExecutionOutcome


class RecoveredBatchFinalizer:
    """Finalize one worker-owned batch from durable receipts after recovery."""

    def __init__(
        self, tasks: TaskService, graph: GraphStore,
        contracts: ContractBuilder, outcomes: OutcomeVerifier,
    ) -> None:
        self.tasks = tasks
        self.graph = graph
        self.contracts = contracts
        self.outcomes = outcomes

    def complete(self, outcome: BatchExecutionOutcome) -> None:
        batch = self.tasks.step_batch(outcome.batch_id)
        if batch is None:
            return
        task_id = str(batch["task_id"])
        state = self.tasks.get(task_id)
        if state is None or state["status"] in {"completed", "failed", "cancelled"}:
            return
        status = outcome.status
        step_statuses = {str(item.get("status") or "")
                         for item in batch.get("steps", [])}
        if (status == "failed" and "abandoned_unknown" in step_statuses
                and step_statuses <= {
                    "succeeded", "abandoned_unknown", "skipped"}):
            status = "abandoned_unknown"
        message: str
        if status == "succeeded":
            incomplete_steps = [
                item for item in self.tasks.list_steps(task_id=task_id)
                if item.get("status") != "succeeded"
            ]
            if incomplete_steps:
                if (any(item.get("status") == "waiting_approval"
                        for item in incomplete_steps)
                        and state["status"] != "waiting_input"):
                    self.tasks.transition(
                        task_id, "waiting_input", label="Approval required",
                        detail="A later recorded step is waiting for approval.")
                return
            if state["status"] in {"recovering", "waiting_input"}:
                self.tasks.transition(
                    task_id, "running", label="Resuming exact reconciled steps")
            state = self.tasks.get(task_id)
            if state and state["status"] == "running":
                self.tasks.transition(
                    task_id, "verifying",
                    label="Verifying recovered task outcome")
            state = self.tasks.get(task_id)
            contract = (
                TaskContract.model_validate(state["completion_contract"])
                if state and int(state.get("contract_version") or 0) >= 1
                else self.contracts.build(
                    state["objective"] if state else "Recovered task",
                    [item["tool_name"]
                     for item in self.tasks.action_history(task_id)],
                )
            )
            verification = self.outcomes.verify_task(
                contract, self.tasks.action_history(task_id))
            self.tasks.record_verification(task_id, verification)
            if verification.status.value == "passed":
                self.tasks.transition(
                    task_id, "completed", label="Recovered task completed",
                    detail=("Every recorded step and receipt passed "
                            "verification."))
                message = ("I completed the exact recorded steps after recovery "
                           "and verified their receipts.")
            else:
                self.tasks.transition(
                    task_id, "failed", label="Recovered task failed verification",
                    detail=verification.summary, error=verification.summary)
                message = ("The recovered steps ran, but their receipts did not "
                           "satisfy the task contract.")
        elif status == "reconcile_required":
            if state["status"] != "waiting_input":
                self.tasks.transition(
                    task_id, "waiting_input",
                    label="Outcome reconciliation required",
                    detail=("A consequential action was interrupted after dispatch "
                            "and was not replayed."))
            message = ("I did not repeat an interrupted consequential action "
                       "because its outcome is uncertain; it needs reconciliation "
                       "first.")
        elif status == "abandoned_unknown":
            self.tasks.transition(
                task_id, "failed",
                label="Reconciliation stopped; outcome remains unknown",
                detail=("The operator stopped waiting without asserting whether "
                        "the external action occurred."),
                error="external_action_outcome_unknown_acknowledged")
            message = ("I stopped waiting for reconciliation as requested. The task "
                       "is closed, but the external action remains outcome unknown.")
        elif status == "cancelled":
            self.tasks.transition(
                task_id, "cancelled", label="Recorded action batch cancelled")
            message = "The recorded action batch was cancelled."
        else:
            self.tasks.transition(
                task_id, "failed", label="Recorded action batch failed",
                detail=f"Batch status: {status}", error=status)
            message = ("The recorded action batch failed; no dependent step was "
                       "dispatched.")
        self.graph.record_node(
            "assistant_message", {"text": message, "delivery": "recovery_outbox"},
            actor="friday", task_id=task_id,
            event_type="assistant.recovery_message",
            links=[("derived_from", task_id)])
        self.tasks.publish(
            task_id, "recovery", "reported", "Recovered task status recorded",
            message[:300])
