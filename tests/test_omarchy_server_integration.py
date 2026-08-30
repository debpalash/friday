from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server
from friday_core import ContractBuilder, GraphStore, TaskService
from friday_core.omarchy import OmarchyDesktopBroker


class _Backend:
    def __init__(self, root: Path):
        self.capture_root = root / "Pictures" / "Friday"
        self.theme = "Gta6"
        self.font = "JetBrainsMono Nerd Font"
        self.nightlight = False
        self.idle = "allow_idle"
        self.brightness_value = 50
        self.locked_value = False

    def version(self):
        return "4.0.1-1"

    def command_fingerprint(self, _tool_name):
        return "a" * 64

    def themes(self):
        return ["Gta6", "Tokyo Night"]

    def current_theme(self):
        return self.theme

    def set_theme(self, value):
        self.theme = value

    def fonts(self):
        return ["JetBrainsMono Nerd Font"]

    def current_font(self):
        return self.font

    def set_font(self, value):
        self.font = value

    def nightlight_enabled(self):
        return self.nightlight

    def toggle_nightlight(self):
        self.nightlight = not self.nightlight

    def idle_mode(self):
        return self.idle

    def set_idle(self, value):
        self.idle = value

    def brightness(self):
        return self.brightness_value

    def set_brightness(self, value):
        self.brightness_value = value

    def locked(self):
        return self.locked_value

    def lock(self):
        self.locked_value = True


def _stage(
    graph: GraphStore,
    tasks: TaskService,
    broker: OmarchyDesktopBroker,
    tool_name: str,
    args: dict,
):
    contract = ContractBuilder().build("Omarchy control", [tool_name])
    task_id, _ = tasks.create(
        contract.objective, contract.model_dump(mode="json"))
    tasks.transition(task_id, "interpreting")
    tasks.transition(task_id, "planned")
    tasks.transition(task_id, "running")
    binding = (
        broker.binding_for_action(tool_name, args).model_dump(mode="json")
        if tool_name in server.OMARCHY_ACTION_TOOLS else {})
    batch_id, steps = tasks.stage_step_batch(task_id, [{
        "tool_call_id": "call_omarchy_1", "tool_name": tool_name,
        "args": args,
        "risk": ("read_only" if tool_name == server.OMARCHY_STATUS_TOOL
                 else "medium"),
        "approval_status": ("not_required"
                            if tool_name == server.OMARCHY_STATUS_TOOL
                            else "approved"),
        "idempotency_class": ("read_only"
                              if tool_name == server.OMARCHY_STATUS_TOOL
                              else "reconcilable"),
        "recovery_policy": ("retry"
                            if tool_name == server.OMARCHY_STATUS_TOOL
                            else "reconcile"),
        "executor_binding": binding, "resource_claims": {},
    }], round_index=0)
    return task_id, batch_id, steps[0], tasks.claim_next_step(
        batch_id, "omarchy-test-worker")


class OmarchyServerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_durable_execution_uses_bound_omarchy_broker_and_verifier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = GraphStore(root / "friday.db")
            tasks = TaskService(graph)
            backend = _Backend(root)
            broker = OmarchyDesktopBroker(
                backend, sleeper=lambda _seconds: None,
                action_timeout_seconds=0.1)
            _task_id, _batch_id, _step, claim = _stage(
                graph, tasks, broker, "machine_omarchy_set_theme",
                {"theme": "Tokyo Night"})
            friday = server.Friday.__new__(server.Friday)

            with patch.multiple(
                    server, GRAPH=graph, TASKS=tasks,
                    OMARCHY_BROKER=broker):
                outcome = await friday.execute_claimed_step(claim)

            self.assertTrue(outcome.succeeded)
            self.assertEqual(outcome.verification["status"], "passed")
            self.assertEqual(backend.theme, "Tokyo Night")

    async def test_status_tool_is_read_only_and_freshly_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = GraphStore(root / "friday.db")
            tasks = TaskService(graph)
            broker = OmarchyDesktopBroker(_Backend(root))
            _task_id, _batch_id, _step, claim = _stage(
                graph, tasks, broker, server.OMARCHY_STATUS_TOOL, {})
            friday = server.Friday.__new__(server.Friday)

            with patch.multiple(
                    server, GRAPH=graph, TASKS=tasks,
                    OMARCHY_BROKER=broker):
                outcome = await friday.execute_claimed_step(claim)

            self.assertTrue(outcome.succeeded)
            self.assertEqual(outcome.verification["status"], "passed")

    async def test_unknown_theme_change_reconciles_without_redispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = GraphStore(root / "friday.db")
            tasks = TaskService(graph)
            backend = _Backend(root)
            broker = OmarchyDesktopBroker(backend)
            task_id, _batch_id, step, claim = _stage(
                graph, tasks, broker, "machine_omarchy_set_theme",
                {"theme": "Tokyo Night"})
            backend.theme = "Tokyo Night"
            tasks.mark_step_outcome_unknown(
                claim, reason_code="omarchy_action_outcome_unknown")
            tasks.recover_interrupted()

            with patch.multiple(
                    server, GRAPH=graph, TASKS=tasks,
                    OMARCHY_BROKER=broker, WORKER=None):
                result = await server._probe_reconciliation(step["step_id"])

            self.assertTrue(result["resolved"])
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(tasks.get(task_id)["status"], "completed")


if __name__ == "__main__":
    unittest.main()
