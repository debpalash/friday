import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import server
from friday_core import (ApprovalService, ClaimedStep, GraphStore,
                         ReflectionService, TaskService)
from friday_core.desktop import DesktopBindingError, DesktopWindowBinding
from friday_core.graph import canonical_json, sha256_text


WINDOW_ID = "win_" + "a" * 40


class _NoCapabilities:
    def active_metadata(self, _name):
        return None

    def active_names(self):
        return set()

    def tool_schemas(self):
        return []


class _DesktopBroker:
    def __init__(self):
        self.binding = DesktopWindowBinding(
            operation="focus", window_id=WINDOW_ID,
            session_fingerprint="b" * 64,
            runtime_identity_sha256="c" * 64,
            application_id_sha256="d" * 64,
            application="Friday Terminal", workspace_id=2,
            args_sha256=sha256_text(canonical_json({"window_id": WINDOW_ID})),
        )
        self.focus_calls = []
        self.reconciliation_calls = []
        self.reconciliation_proven = True

    def list_windows(self):
        return {
            "status": "ok", "verified": True,
            "windows": [{
                "window_id": WINDOW_ID, "application": "Friday Terminal",
                "workspace_id": 2, "active": True,
                "floating": False, "fullscreen": False,
            }],
        }

    def binding_for_action(self, window_id, operation):
        if window_id != WINDOW_ID or operation != "focus":
            raise DesktopBindingError()
        return self.binding

    def focus_window(self, window_id, *, expected_binding):
        if (window_id != WINDOW_ID
                or DesktopWindowBinding.model_validate(expected_binding)
                != self.binding):
            raise DesktopBindingError()
        self.focus_calls.append((window_id, expected_binding))
        return {
            "status": "ok", "verified": True, "operation": "focus",
            "window_id": WINDOW_ID, "application": "Friday Terminal",
            "workspace_id": 2, "state": "focused",
            "idempotent_replay": False,
        }

    def reconciliation_receipt(self, expected_binding):
        binding = DesktopWindowBinding.model_validate(expected_binding)
        self.reconciliation_calls.append(binding)
        if binding.operation == "close" and self.reconciliation_proven:
            return {
                "status": "ok", "verified": True, "operation": "close",
                "window_id": WINDOW_ID, "application": "Friday Terminal",
                "workspace_id": 2, "state": "closed",
                "idempotent_replay": True,
            }
        return None

    def verify_receipt(self, tool_name, result, args, idempotency_key):
        value = json.loads(result) if isinstance(result, str) else result
        if tool_name == "machine_list_windows":
            return bool(
                args == {} and idempotency_key is None
                and value == self.list_windows())
        return bool(
            tool_name in {"machine_focus_window", "machine_close_window"}
            and args == {"window_id": WINDOW_ID}
            and isinstance(idempotency_key, str)
            and value.get("state") == (
                "focused" if tool_name == "machine_focus_window" else "closed")
            and value.get("window_id") == WINDOW_ID)


async def _collect(items, value):
    items.append(value)


def _claim(binding: dict, *, mutate: bool = False,
           tool_name: str = "machine_focus_window") -> ClaimedStep:
    selected = dict(binding)
    if mutate:
        selected["runtime_identity_sha256"] = "e" * 64
    return ClaimedStep(
        step_id="step_desktop_focus_0001",
        batch_id="batch_desktop_focus_0001",
        task_id="task_desktop_focus_0001",
        round_index=0,
        ordinal=1,
        tool_call_id="call_desktop_focus_0001",
        tool_name=tool_name,
        args=({"window_id": WINDOW_ID}
              if tool_name != "machine_list_windows" else {}),
        idempotency_key="act_" + "a" * 64,
        idempotency_class=(
            "read_only" if tool_name == "machine_list_windows"
            else "reconcilable"),
        recovery_policy=(
            "retry" if tool_name == "machine_list_windows" else "reconcile"),
        risk="read_only" if tool_name == "machine_list_windows" else "medium",
        approval_status=(
            "not_required" if tool_name == "machine_list_windows"
            else "approved"),
        action_id="action_desktop_focus_0001",
        attempt_id="attempt_desktop_focus_0001",
        attempt_number=1,
        lease_id="step_execution_lease_0001",
        worker_id="worker_desktop_focus_0001",
        verifier="desktop_focus_receipt",
        executor_binding=selected,
        resource_claims={
            "cpu_cores": 0.1, "ram_mib": 64, "vram_mib": 0,
            "accelerator": "none", "network": False,
            "concurrency_slots": 1, "latency_class": "interactive",
        },
        context={},
        resource_lease_id="resource_step_lease_0001",
    )


class DesktopServerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_authoritative_probe_settles_close_without_redispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            broker = _DesktopBroker()
            binding = broker.binding.model_copy(update={"operation": "close"})
            contract = server.CONTRACTS.build(
                "Close the managed terminal window.",
                ["machine_close_window"])
            task_id, _ = tasks.create(
                contract.objective, contract.model_dump(mode="json"))
            tasks.transition(task_id, "interpreting")
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")
            batch_id, steps = tasks.stage_step_batch(task_id, [{
                "tool_call_id": "call_desktop_reconcile",
                "tool_name": "machine_close_window",
                "args": {"window_id": WINDOW_ID},
                "risk": "high", "approval_status": "approved",
                "idempotency_class": "reconcilable",
                "recovery_policy": "reconcile",
                "executor_binding": binding.model_dump(mode="json"),
                "resource_claims": {},
            }], round_index=0)
            claim = tasks.claim_next_step(batch_id, "crashed-worker")
            tasks.mark_step_outcome_unknown(
                claim, reason_code="desktop_close_outcome_unknown")
            tasks.recover_interrupted()

            with patch.multiple(
                    server, TASKS=tasks, GRAPH=graph,
                    DESKTOP_BROKER=broker, WORKER=None):
                result = await server._probe_reconciliation(
                    steps[0]["step_id"])

            self.assertTrue(result["resolved"])
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(len(broker.reconciliation_calls), 1)
            self.assertEqual(broker.focus_calls, [])
            self.assertEqual(tasks.get(task_id)["status"], "completed")
            with graph._connect() as conn:
                receipt = conn.execute(
                    "SELECT status FROM action_receipts WHERE step_id=?",
                    (steps[0]["step_id"],)).fetchone()
            self.assertEqual(receipt["status"], "succeeded")

    async def test_cancelled_probe_finishes_cas_and_continuation(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            broker = _DesktopBroker()
            binding = broker.binding.model_copy(update={"operation": "close"})
            contract = server.CONTRACTS.build(
                "Close the managed terminal window.",
                ["machine_close_window"])
            task_id, _ = tasks.create(
                contract.objective, contract.model_dump(mode="json"))
            tasks.transition(task_id, "interpreting")
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")
            batch_id, steps = tasks.stage_step_batch(task_id, [{
                "tool_call_id": "close-cancelled-probe",
                "tool_name": "machine_close_window",
                "args": {"window_id": WINDOW_ID},
                "risk": "high", "approval_status": "approved",
                "idempotency_class": "reconcilable",
                "recovery_policy": "reconcile",
                "executor_binding": binding.model_dump(mode="json"),
                "resource_claims": {},
            }], round_index=0)
            claim = tasks.claim_next_step(batch_id, "cancelled-probe-worker")
            tasks.mark_step_outcome_unknown(
                claim, reason_code="desktop_close_outcome_unknown")
            tasks.recover_interrupted()
            settle_started = threading.Event()
            release_settle = threading.Event()
            original_resolve = tasks.resolve_reconciliation

            def blocked_resolve(*args, **kwargs):
                settle_started.set()
                if not release_settle.wait(timeout=2):
                    raise RuntimeError("blocked reconciliation timed out")
                return original_resolve(*args, **kwargs)

            server.RECONCILIATION_SHUTTING_DOWN = False
            with (patch.multiple(
                    server, TASKS=tasks, GRAPH=graph,
                    DESKTOP_BROKER=broker, WORKER=None),
                  patch.object(
                    tasks, "resolve_reconciliation",
                    side_effect=blocked_resolve)):
                probe = asyncio.create_task(
                    server._probe_reconciliation(steps[0]["step_id"]))
                for _ in range(100):
                    if settle_started.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(settle_started.is_set())
                probe.cancel()
                await asyncio.sleep(0.02)
                self.assertFalse(probe.done())
                release_settle.set()
                with self.assertRaises(asyncio.CancelledError):
                    await probe

            self.assertEqual(tasks.get(task_id)["status"], "completed")
            self.assertEqual(tasks.step_batch(batch_id)["status"], "succeeded")

    async def test_cancel_winning_after_reconcile_cas_never_enqueues_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            broker = _DesktopBroker()
            binding = broker.binding.model_copy(update={"operation": "close"})
            task_id, _ = tasks.create(
                "Close then list", {"version": 0, "evidence": "exact"})
            tasks.transition(task_id, "interpreting")
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")
            batch_id, steps = tasks.stage_step_batch(task_id, [{
                "tool_call_id": "close-before-cancel-race",
                "tool_name": "machine_close_window",
                "args": {"window_id": WINDOW_ID},
                "risk": "high", "approval_status": "approved",
                "idempotency_class": "reconcilable",
                "recovery_policy": "reconcile",
                "executor_binding": binding.model_dump(mode="json"),
                "resource_claims": {},
            }, {
                "tool_call_id": "list-after-cancel-race",
                "tool_name": "machine_list_windows", "args": {},
                "risk": "read_only", "approval_status": "not_required",
                "idempotency_class": "read_only",
                "recovery_policy": "retry", "executor_binding": {},
                "resource_claims": {},
            }], round_index=0)
            claim = tasks.claim_next_step(batch_id, "cancel-race-worker")
            tasks.mark_step_outcome_unknown(
                claim, reason_code="desktop_close_outcome_unknown")
            tasks.recover_interrupted()
            original_transition = tasks.transition
            worker = SimpleNamespace(is_running=True, enqueue=AsyncMock())

            def cancel_before_resume(selected_task, status, **kwargs):
                if status == "running":
                    tasks.request_cancel(selected_task)
                return original_transition(selected_task, status, **kwargs)

            with (patch.multiple(
                    server, TASKS=tasks, GRAPH=graph,
                    DESKTOP_BROKER=broker, WORKER=worker),
                  patch.object(
                    tasks, "transition", side_effect=cancel_before_resume)):
                result = await server._probe_reconciliation(
                    steps[0]["step_id"])

            self.assertTrue(result["resolved"])
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(tasks.get(task_id)["status"], "cancelled")
            worker.enqueue.assert_not_awaited()

    async def test_inconclusive_probe_preserves_every_durable_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            broker = _DesktopBroker()
            broker.reconciliation_proven = False
            binding = broker.binding.model_copy(update={"operation": "close"})
            task_id, _ = tasks.create(
                "Close a window", {"version": 0, "evidence": "exact"})
            batch_id, steps = tasks.stage_step_batch(task_id, [{
                "tool_call_id": "close-inconclusive",
                "tool_name": "machine_close_window",
                "args": {"window_id": WINDOW_ID},
                "risk": "high", "approval_status": "approved",
                "idempotency_class": "reconcilable",
                "recovery_policy": "reconcile",
                "executor_binding": binding.model_dump(mode="json"),
                "resource_claims": {},
            }], round_index=0)
            claim = tasks.claim_next_step(batch_id, "unknown-worker")
            tasks.mark_step_outcome_unknown(
                claim, reason_code="desktop_close_outcome_unknown")

            def snapshot():
                with graph._connect() as conn:
                    return {
                        table: [tuple(row) for row in conn.execute(
                            f"SELECT * FROM {table} ORDER BY rowid")]
                        for table in ("task_state", "task_step_batches",
                                      "task_steps", "action_attempts",
                                      "action_receipts")
                    }

            before = snapshot()
            with patch.multiple(
                    server, TASKS=tasks, GRAPH=graph,
                    DESKTOP_BROKER=broker, WORKER=None):
                result = await server._probe_reconciliation(
                    steps[0]["step_id"])
            after = snapshot()
            self.assertFalse(result["resolved"])
            self.assertEqual(result["reason"],
                             "postcondition_not_currently_proven")
            self.assertEqual(before, after)
            self.assertEqual(len(broker.reconciliation_calls), 1)

    async def test_public_decisions_reject_loose_booleans_and_evidence(self):
        with self.assertRaises(server.HTTPException) as approval:
            await server.api_decide_approval(
                "approval_untrusted", SimpleNamespace(),
                {"approved": "false"})
        self.assertEqual(approval.exception.status_code, 400)

        for body in (
            {"decision": "success", "confirm": True},
            {"decision": "abandon_unknown", "confirm": "true"},
            {"decision": "abandon_unknown", "confirm": True,
             "result": {"verified": True}},
        ):
            with self.subTest(body=body):
                with self.assertRaises(server.HTTPException) as decision:
                    await server.api_decide_reconciliation(
                        "step_untrusted", body)
                self.assertEqual(decision.exception.status_code, 400)

    async def test_operator_abandonment_reports_unknown_not_action_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            broker = _DesktopBroker()
            binding = broker.binding.model_copy(update={"operation": "close"})
            task_id, _ = tasks.create(
                "Close a window", {"version": 0, "evidence": "exact"})
            tasks.transition(task_id, "interpreting")
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")
            batch_id, steps = tasks.stage_step_batch(task_id, [{
                "tool_call_id": "close-abandon-unknown",
                "tool_name": "machine_close_window",
                "args": {"window_id": WINDOW_ID},
                "risk": "high", "approval_status": "approved",
                "idempotency_class": "reconcilable",
                "recovery_policy": "reconcile",
                "executor_binding": binding.model_dump(mode="json"),
                "resource_claims": {},
            }], round_index=0)
            claim = tasks.claim_next_step(batch_id, "abandon-worker")
            tasks.mark_step_outcome_unknown(
                claim, reason_code="desktop_close_outcome_unknown")
            tasks.recover_interrupted()

            with patch.multiple(
                    server, TASKS=tasks, GRAPH=graph,
                    DESKTOP_BROKER=broker, WORKER=None):
                result = await server.api_decide_reconciliation(
                    steps[0]["step_id"],
                    {"decision": "abandon_unknown", "confirm": True})

            self.assertEqual(result["status"], "abandoned_unknown")
            state = tasks.get(task_id)
            self.assertEqual(state["status"], "failed")
            self.assertEqual(
                state["last_error"],
                "external_action_outcome_unknown_acknowledged")
            with graph._connect() as conn:
                receipt = conn.execute(
                    "SELECT status FROM action_receipts WHERE step_id=?",
                    (steps[0]["step_id"],)).fetchone()
                message = conn.execute(
                    """SELECT body_json FROM nodes
                       WHERE kind='assistant_message' ORDER BY rowid DESC
                       LIMIT 1""").fetchone()
            self.assertEqual(receipt["status"], "outcome_unknown")
            self.assertIn("remains outcome unknown", message["body_json"])

    async def test_cancelled_ack_finishes_unknown_finalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            broker = _DesktopBroker()
            binding = broker.binding.model_copy(update={"operation": "close"})
            task_id, _ = tasks.create(
                "Close a window", {"version": 0, "evidence": "exact"})
            tasks.transition(task_id, "interpreting")
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")
            _batch_id, steps = tasks.stage_step_batch(task_id, [{
                "tool_call_id": "close-cancelled-ack",
                "tool_name": "machine_close_window",
                "args": {"window_id": WINDOW_ID},
                "risk": "high", "approval_status": "approved",
                "idempotency_class": "reconcilable",
                "recovery_policy": "reconcile",
                "executor_binding": binding.model_dump(mode="json"),
                "resource_claims": {},
            }], round_index=0)
            claim = tasks.claim_next_step(_batch_id, "cancelled-ack-worker")
            tasks.mark_step_outcome_unknown(
                claim, reason_code="desktop_close_outcome_unknown")
            tasks.recover_interrupted()
            ack_started = threading.Event()
            release_ack = threading.Event()
            original_ack = tasks.acknowledge_unknown_reconciliation

            def blocked_ack(*args, **kwargs):
                ack_started.set()
                if not release_ack.wait(timeout=2):
                    raise RuntimeError("blocked acknowledgement timed out")
                return original_ack(*args, **kwargs)

            server.RECONCILIATION_SHUTTING_DOWN = False
            with (patch.multiple(
                    server, TASKS=tasks, GRAPH=graph,
                    DESKTOP_BROKER=broker, WORKER=None),
                  patch.object(
                    tasks, "acknowledge_unknown_reconciliation",
                    side_effect=blocked_ack)):
                decision = asyncio.create_task(
                    server.api_decide_reconciliation(
                        steps[0]["step_id"],
                        {"decision": "abandon_unknown", "confirm": True}))
                for _ in range(100):
                    if ack_started.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(ack_started.is_set())
                decision.cancel()
                await asyncio.sleep(0.02)
                self.assertFalse(decision.done())
                release_ack.set()
                with self.assertRaises(asyncio.CancelledError):
                    await decision

            state = tasks.get(task_id)
            self.assertEqual(state["status"], "failed")
            self.assertEqual(
                state["last_error"],
                "external_action_outcome_unknown_acknowledged")

    async def test_restart_after_unknown_ack_preserves_explicit_disposition(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            broker = _DesktopBroker()
            binding = broker.binding.model_copy(update={"operation": "close"})
            task_id, _ = tasks.create(
                "Close a window", {"version": 0, "evidence": "exact"})
            tasks.transition(task_id, "interpreting")
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")
            batch_id, _steps = tasks.stage_step_batch(task_id, [{
                "tool_call_id": "close-abandon-before-restart",
                "tool_name": "machine_close_window",
                "args": {"window_id": WINDOW_ID},
                "risk": "high", "approval_status": "approved",
                "idempotency_class": "reconcilable",
                "recovery_policy": "reconcile",
                "executor_binding": binding.model_dump(mode="json"),
                "resource_claims": {},
            }], round_index=0)
            claim = tasks.claim_next_step(batch_id, "crashed-after-ack-worker")
            tasks.mark_step_outcome_unknown(
                claim, reason_code="desktop_close_outcome_unknown")
            tasks.recover_interrupted()
            candidate = tasks.reconciliation_candidate(claim.step_id)
            tasks.acknowledge_unknown_reconciliation(candidate)

            # Simulate restart: only the durable batch enum survives, and the
            # worker reconstructs the old generic ``failed`` outcome.
            self.assertIn(batch_id, tasks.pending_step_batches())
            durable_status = tasks.step_batch(batch_id)["status"]
            self.assertEqual(durable_status, "failed")
            with patch.multiple(server, TASKS=tasks, GRAPH=graph):
                await server._complete_recovered_batch(
                    server.BatchExecutionOutcome(
                        batch_id=batch_id, status=durable_status, outcomes=(),
                        recovered_without_raw_results=False))

            state = tasks.get(task_id)
            self.assertEqual(state["status"], "failed")
            self.assertEqual(
                state["last_error"],
                "external_action_outcome_unknown_acknowledged")
            with graph._connect() as conn:
                message = conn.execute(
                    """SELECT body_json FROM nodes
                       WHERE kind='assistant_message' ORDER BY rowid DESC
                       LIMIT 1""").fetchone()
            self.assertIn("remains outcome unknown", message["body_json"])

    async def test_recovered_older_batch_does_not_complete_later_approval(self):
        transitions = []
        tasks = SimpleNamespace(
            step_batch=lambda _batch_id: {
                "task_id": "task_1", "steps": [{"status": "succeeded"}]},
            get=lambda _task_id: {
                "status": "verifying", "objective": "test",
                "contract_version": 1, "completion_contract": {},
            },
            list_steps=lambda **_kwargs: [
                {"status": "succeeded"},
                {"status": "waiting_approval"},
            ],
            transition=lambda task_id, status, **kwargs: transitions.append(
                (task_id, status, kwargs)),
        )

        with patch.object(server, "TASKS", tasks):
            await server._complete_recovered_batch(
                server.BatchExecutionOutcome(
                    batch_id="older_batch", status="succeeded", outcomes=(),
                    recovered_without_raw_results=False))

        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0][1], "waiting_input")
        self.assertEqual(transitions[0][2]["label"], "Approval required")

    async def test_interrupted_close_is_never_replayed_automatically(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            broker = _DesktopBroker()
            binding = broker.binding.model_copy(update={"operation": "close"})
            contract = server.CONTRACTS.build(
                "Close the managed terminal window.",
                ["machine_close_window"])
            task_id, _ = tasks.create(
                contract.objective, contract.model_dump(mode="json"))
            batch_id, _ = tasks.stage_step_batch(task_id, [{
                "tool_call_id": "call_desktop_close",
                "tool_name": "machine_close_window",
                "args": {"window_id": WINDOW_ID},
                "risk": "high", "approval_status": "approved",
                "idempotency_class": "reconcilable",
                "recovery_policy": "reconcile",
                "executor_binding": binding.model_dump(mode="json"),
                "resource_claims": {},
            }], round_index=0)
            claim = tasks.claim_next_step(batch_id, "recovery-worker")
            self.assertIsNotNone(claim)

            recovered = tasks.recover_inflight_steps(
                force=True, dead_worker_id="recovery-worker")

            self.assertEqual(recovered["retry"], [])
            self.assertEqual(recovered["reconcile"], [claim.step_id])
            batch = tasks.step_batch(batch_id)
            self.assertEqual(batch["status"], "reconcile_required")
            self.assertEqual(batch["steps"][0]["status"],
                             "reconcile_required")
            with graph._connect() as conn:
                receipt = conn.execute(
                    "SELECT status FROM action_receipts WHERE step_id=?",
                    (claim.step_id,)).fetchone()
            self.assertEqual(receipt["status"], "outcome_unknown")

    async def test_focus_is_bound_before_ephemeral_exact_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            approvals = ApprovalService(graph)
            broker = _DesktopBroker()
            contract = server.CONTRACTS.build(
                "Focus the managed terminal window.",
                ["machine_focus_window"])
            task_id, _ = tasks.create(
                contract.objective, contract.model_dump(mode="json"))
            tasks.transition(task_id, "interpreting")
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")

            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            friday.save_session = lambda: None

            async def fake_stream(_messages, _speak_q, use_tools=True,
                                  required_tool=None):
                return "", [{
                    "id": "call_desktop_focus",
                    "name": "machine_focus_window",
                    "args": json.dumps({"window_id": WINDOW_ID}),
                }]

            friday._stream_once = fake_stream
            queue = asyncio.Queue()
            progress = []
            empty_context = SimpleNamespace(retrieve=lambda *_args, **_kwargs: [])
            no_feedback = SimpleNamespace(
                relevant_context=lambda *_args, **_kwargs: [])
            no_skills = SimpleNamespace(
                relevant_context=lambda *_args, **_kwargs: [])

            with patch.multiple(
                    server, TASKS=tasks, APPROVALS=approvals,
                    DESKTOP_BROKER=broker, CAPABILITIES=_NoCapabilities(),
                    REFLECTION=ReflectionService(graph), MEMORY=empty_context,
                    FEEDBACK=no_feedback, SKILLS=no_skills, WORKER=None):
                await friday.respond(
                    "Focus the Friday terminal window.", queue,
                    existing_task_id=task_id,
                    progress_sink=lambda event: _collect(progress, event))

            step = tasks.list_steps(task_id=task_id)[0]
            self.assertEqual(step["executor_binding"],
                             broker.binding.model_dump(mode="json"))
            self.assertEqual(step["idempotency_class"], "reconcilable")
            self.assertEqual(step["recovery_policy"], "reconcile")
            self.assertEqual(step["resource_claims"]["cpu_cores"], 0.1)
            self.assertEqual(step["resource_claims"]["ram_mib"], 64)
            approval = next(item for item in progress
                            if item.get("type") == "approval_required")
            self.assertEqual(approval["args"]["window_id"], WINDOW_ID)
            self.assertEqual(approval["args"]["application"], "Friday Terminal")
            self.assertEqual(approval["args"]["workspace_id"], 2)
            self.assertNotIn("session_fingerprint", approval["args"])
            self.assertNotIn("runtime_identity_sha256", approval["args"])
            with graph._connect() as conn:
                persisted = json.loads(conn.execute(
                    "SELECT args_json FROM approval_state WHERE task_id=?",
                    (task_id,)).fetchone()[0])
            self.assertNotIn("session_fingerprint", persisted)
            self.assertNotIn("runtime_identity_sha256", persisted)

    async def test_execution_revalidates_exact_desktop_binding(self):
        broker = _DesktopBroker()
        friday = server.Friday.__new__(server.Friday)
        with patch.object(server, "DESKTOP_BROKER", broker):
            outcome = await friday.execute_claimed_step(
                _claim(broker.binding.model_dump(mode="json")))
            self.assertTrue(outcome.succeeded)
            self.assertEqual(outcome.verification["status"], "passed")
            self.assertEqual(len(broker.focus_calls), 1)
            with self.assertRaises(DesktopBindingError):
                await friday.execute_claimed_step(
                    _claim(broker.binding.model_dump(mode="json"), mutate=True))

    async def test_window_listing_uses_fresh_authoritative_verification(self):
        broker = _DesktopBroker()
        friday = server.Friday.__new__(server.Friday)
        with patch.object(server, "DESKTOP_BROKER", broker):
            outcome = await friday.execute_claimed_step(
                _claim({}, tool_name="machine_list_windows"))
        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.verification["status"], "passed")


if __name__ == "__main__":
    unittest.main()
