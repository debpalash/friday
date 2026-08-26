import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import server
from friday_core import (ApprovalService, CapabilityManager, ClaimedStep,
                         GraphStore, ReflectionService, TaskService)


class _SelectedCapabilityRegistry:
    """Minimal registry fake for exercising Friday's selection boundary."""

    def __init__(self, metadata):
        self.metadata = dict(metadata)
        self.executions = []

    def active_metadata(self, name):
        return dict(self.metadata) if name == self.metadata["name"] else None

    def active_names(self):
        return {self.metadata["name"]}

    def tool_schemas(self):
        return []

    def execute_version(self, version_id, args, *, expected_name,
                        expected_version, expected_code_sha256,
                        expected_permissions):
        self.executions.append((version_id, args, expected_name,
                                expected_version, expected_code_sha256,
                                expected_permissions))
        return "should not execute before approval"


def _claim(tool_name, binding):
    return ClaimedStep(
        step_id="step_test", batch_id="batch_test", task_id="task_test",
        round_index=0, ordinal=1, tool_call_id="call_test",
        tool_name=tool_name, args={"value": 4},
        idempotency_key="act_test", idempotency_class="non_repeatable",
        recovery_policy="reconcile", risk="high",
        approval_status="approved", action_id="action_test",
        attempt_id="attempt_test", attempt_number=1,
        lease_id="lease_test", worker_id="worker_test",
        verifier="successful_receipt", executor_binding=dict(binding),
        resource_claims={}, context={},
    )


class DynamicCapabilityServerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_dynamic_call_ignores_prior_name_args_approval_and_persists_exact_selection(self):
        metadata = {
            "kind": "capability",
            "name": "dynamic_sync",
            "version_id": "capv_selected",
            "version": 7,
            "code_sha256": "a" * 64,
            "permissions": ["network", "process"],
        }
        args = {"value": 4}
        registry = _SelectedCapabilityRegistry(metadata)

        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            approvals = ApprovalService(graph)
            contract = server.CONTRACTS.build(
                "Run the dynamic synchronization capability.",
                [metadata["name"]],
                dynamic_permissions={metadata["name"]: metadata["permissions"]},
            )
            task_id, _ = tasks.create(
                "Run the dynamic synchronization capability.",
                contract.model_dump(mode="json"),
            )
            tasks.transition(task_id, "interpreting")
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")

            # A legacy approval for the same name and arguments is deliberately
            # not version-bound. It must never authorize a newly selected
            # dynamic executor version.
            old_approval = approvals.request(
                task_id, metadata["name"], args, "legacy name/args approval")
            approvals.decide(old_approval["approval_id"], True)

            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            friday.save_session = lambda: None

            async def fake_stream(_messages, _speak_q, use_tools=True,
                                  required_tool=None):
                return "", [{
                    "id": "call_dynamic",
                    "name": metadata["name"],
                    "args": json.dumps(args),
                }]

            friday._stream_once = fake_stream
            queue = asyncio.Queue()
            progress = []
            no_memory = SimpleNamespace(retrieve=lambda *_args, **_kwargs: [])
            no_feedback = SimpleNamespace(
                relevant_context=lambda *_args, **_kwargs: [])
            no_skills = SimpleNamespace(
                relevant_context=lambda *_args, **_kwargs: [])

            with patch.multiple(
                server,
                TASKS=tasks,
                APPROVALS=approvals,
                CAPABILITIES=registry,
                REFLECTION=ReflectionService(graph),
                MEMORY=no_memory,
                FEEDBACK=no_feedback,
                SKILLS=no_skills,
                WORKER=None,
            ):
                await friday.respond(
                    "Run dynamic_sync with value 4.", queue,
                    existing_task_id=task_id,
                    progress_sink=lambda event: _collect(progress, event),
                )

            steps = tasks.list_steps(task_id=task_id)
            pending = approvals.list(status="pending")
            with graph._connect() as conn:
                stored = conn.execute(
                    """SELECT executor_binding_json,resource_claims_json,
                              args_sha256,approval_status,status
                       FROM task_steps WHERE step_id=?""",
                    (steps[0]["step_id"],),
                ).fetchone()
                batch = conn.execute(
                    "SELECT status FROM task_step_batches WHERE batch_id=?",
                    (steps[0]["batch_id"],),
                ).fetchone()

            expected_resources = {
                "cpu_cores": 1.0,
                "ram_mib": 512,
                "vram_mib": 0,
                "accelerator": "none",
                "network": True,
                "concurrency_slots": 1,
                "latency_class": "interactive",
            }
            self.assertEqual(tasks.get(task_id)["status"], "waiting_input")
            self.assertEqual(batch["status"], "waiting_approval")
            self.assertEqual(stored["status"], "waiting_approval")
            self.assertEqual(stored["approval_status"], "pending")
            self.assertEqual(json.loads(stored["executor_binding_json"]), metadata)
            self.assertEqual(
                json.loads(stored["resource_claims_json"]), expected_resources)
            self.assertEqual(steps[0]["executor_binding"]["version_id"],
                             "capv_selected")
            self.assertEqual(steps[0]["executor_binding"]["code_sha256"],
                             "a" * 64)
            self.assertEqual(steps[0]["resource_claims"], expected_resources)

            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["step_id"], steps[0]["step_id"])
            self.assertEqual(pending[0]["args"]["_args_sha256"],
                             stored["args_sha256"])
            self.assertIn("dynamic_sync@v7", pending[0]["reason"])
            self.assertIn(("a" * 64)[:12], pending[0]["reason"])
            self.assertEqual(registry.executions, [])
            self.assertEqual(await queue.get(),
                             "I need your approval before I can do that.")
            self.assertIsNone(await queue.get())
            self.assertTrue(any(item["status"] == "approved"
                                and item["step_id"] is None
                                for item in approvals.list(status=None)))

    async def test_execution_refuses_missing_name_or_hash_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            manager = CapabilityManager(
                graph, Path(temporary) / "capabilities",
                reserved_names=server.BUILTIN_TOOL_NAMES,
            )
            source_task, _ = tasks.create("create bound tool", {})
            version_id = manager.create_version(
                "bound_math", "Return twice a number.",
                {"type": "object",
                 "properties": {"value": {"type": "integer"}},
                 "required": ["value"]},
                "def run(args):\n    return args['value'] * 2\n",
                [],
                [{"args": {"value": 2}, "expected": 4},
                 {"args": {"value": -3}, "expected": -6}],
                source_node_ids=[source_task],
            )
            with patch.object(manager, "sandbox_status",
                              return_value=(True, None)), \
                 patch.object(manager, "_run",
                              side_effect=lambda _version, call_args,
                              **_kwargs: call_args["value"] * 2):
                self.assertTrue(manager.evaluate_and_activate(version_id))
            metadata = manager.active_metadata("bound_math")
            friday = server.Friday.__new__(server.Friday)

            with patch.object(server, "CAPABILITIES", manager):
                with self.assertRaisesRegex(RuntimeError,
                                            "lacks an immutable executor binding"):
                    await friday.execute_claimed_step(_claim("bound_math", {}))
                with self.assertRaisesRegex(RuntimeError, "binding name mismatch"):
                    await friday.execute_claimed_step(_claim(
                        "bound_math", metadata | {"name": "other_tool"}))
                with self.assertRaisesRegex(RuntimeError, "binding hash"):
                    await friday.execute_claimed_step(_claim(
                        "bound_math", metadata | {"code_sha256": "0" * 64}))
                with self.assertRaisesRegex(RuntimeError, "binding version"):
                    await friday.execute_claimed_step(_claim(
                        "bound_math", metadata | {
                            "version": metadata["version"] + 1}))
                with self.assertRaisesRegex(RuntimeError,
                                            "binding permissions"):
                    await friday.execute_claimed_step(_claim(
                        "bound_math", metadata | {
                            "permissions": ["network"]}))
                with patch.object(manager, "_run", return_value=8):
                    outcome = await friday.execute_claimed_step(
                        _claim("bound_math", metadata))
                self.assertTrue(outcome.succeeded)
                self.assertEqual(outcome.result, "8")

    async def test_execution_refuses_version_from_a_different_capability(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            manager = CapabilityManager(
                graph, Path(temporary) / "capabilities",
                reserved_names=server.BUILTIN_TOOL_NAMES,
            )
            source_task, _ = tasks.create("create two tools", {})

            def create(name):
                return manager.create_version(
                    name, f"Return {name}.",
                    {"type": "object", "properties": {
                        "value": {"type": "integer"}}},
                    f"def run(args):\n    return '{name}'\n",
                    [],
                    [{"args": {}, "expected": name},
                     {"args": {"value": 1}, "expected": name}],
                    source_node_ids=[source_task],
                )

            first_version = create("first_dynamic")
            second_version = create("second_dynamic")
            with patch.object(manager, "sandbox_status",
                              return_value=(True, None)), \
                 patch.object(manager, "_run",
                              side_effect=lambda version, _args,
                              **_kwargs: version["name"]):
                self.assertTrue(manager.evaluate_and_activate(first_version))
                self.assertTrue(manager.evaluate_and_activate(second_version))
                first = manager.active_metadata("first_dynamic")
                second = manager.active_metadata("second_dynamic")
                forged = first | {
                    "version_id": second["version_id"],
                    "version": second["version"],
                    "code_sha256": second["code_sha256"],
                    "permissions": second["permissions"],
                }
                friday = server.Friday.__new__(server.Friday)
                with patch.object(server, "CAPABILITIES", manager):
                    with self.assertRaises((RuntimeError, ValueError)):
                        await friday.execute_claimed_step(
                            _claim("first_dynamic", forged))


async def _collect(target, event):
    target.append(event)


if __name__ == "__main__":
    unittest.main()
