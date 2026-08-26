import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import server
from friday_core import (ApprovalService, ClaimedStep, GraphStore,
                         ReflectionService, TaskService)
from friday_core.desktop import DesktopApplicationLaunchBinding
from friday_core.graph import sha256_text
from friday_core.processes import (ProcessBindingError, ProcessBroker,
                                   ProcessInstanceBinding)


class _NoCapabilities:
    def active_metadata(self, _name):
        return None

    def active_names(self):
        return set()

    def tool_schemas(self):
        return []


def _broker(root: Path, graph: GraphStore) -> ProcessBroker:
    # Binding and approval do not invoke the backend or admission object.
    return ProcessBroker(
        graph, server._curated_process_registry(), object(), object(),
        state_root=root / "process-runtime")


async def _collect(items, value):
    items.append(value)


def _claim(binding: dict, *, mutate: bool = False,
           terminate: bool = False) -> ClaimedStep:
    selected = dict(binding)
    if mutate:
        selected["args_sha256"] = "f" * 64
    instance_id = str(binding.get("instance_id") or "")
    return ClaimedStep(
        step_id=("step_process_terminate_0001" if terminate
                 else "step_process_launch_0001"),
        batch_id=("batch_process_terminate_0001" if terminate
                  else "batch_process_launch_0001"),
        task_id=("task_process_terminate_0001" if terminate
                 else "task_process_launch_0001"),
        round_index=0,
        ordinal=1,
        tool_call_id=("call_process_terminate_0001" if terminate
                      else "call_process_launch_0001"),
        tool_name=("machine_terminate_process" if terminate
                   else "machine_launch_process"),
        args=({"instance_id": instance_id} if terminate else
              {"spec_id": "proc.managed_wait.v1",
               "parameter_values": {"seconds": 30}}),
        idempotency_key="act_" + "a" * 64,
        idempotency_class="reconcilable",
        recovery_policy="reconcile",
        risk="high",
        approval_status="approved",
        action_id=("action_process_terminate_0001" if terminate
                   else "action_process_launch_0001"),
        attempt_id=("attempt_process_terminate_0001" if terminate
                    else "attempt_process_launch_0001"),
        attempt_number=1,
        lease_id="step_execution_lease_0001",
        worker_id=("worker_process_terminate_0001" if terminate
                   else "worker_process_launch_0001"),
        verifier=("process_terminate_receipt" if terminate
                  else "process_launch_receipt"),
        executor_binding=selected,
        resource_claims=(binding.get("resource_claim") or {}),
        context={},
        resource_lease_id="resource_step_lease_0001",
    )


class ProcessServerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_curated_browser_is_persistent_singleton_and_cgroup_presented(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            broker = _broker(root, GraphStore(root / "friday.db"))
            specs = {item["spec_id"]: item for item in broker.list_specs()}
            if not Path(server.MANAGED_BROWSER_EXECUTABLE).is_file():
                self.skipTest("curated Chromium profile is unavailable")
            browser = specs[server.MANAGED_BROWSER_SPEC_ID]
            internal = broker.registry.get(server.MANAGED_BROWSER_SPEC_ID)
            self.assertEqual(
                internal.executable, server.MANAGED_BROWSER_EXECUTABLE)
            self.assertEqual(internal.version, 2)

            self.assertTrue(browser["persistent"])
            self.assertEqual(browser["instance_policy"], "singleton")
            self.assertEqual(
                browser["session_access"],
                {"wayland": True, "session_bus": False})
            self.assertEqual(
                browser["presentation"]["window_owner"],
                "managed_cgroup")
            self.assertEqual(
                browser["presentation"]["application"], "Managed Browser")
            self.assertIn(
                f"--remote-debugging-port={server.MANAGED_BROWSER_DEBUG_PORT}",
                internal.fixed_args)
            self.assertIn(
                f"--user-data-dir={server.MANAGED_BROWSER_PROFILE_DIR}",
                internal.fixed_args)
            for argument in (
                    f"--proxy-server=socks5://127.0.0.1:"
                    f"{server.MANAGED_BROWSER_PROXY_PORT}",
                    "--proxy-bypass-list=<-loopback>",
                    "--host-resolver-rules=MAP * ~NOTFOUND, "
                    "EXCLUDE 127.0.0.1",
                    "--webrtc-ip-handling-policy=disable_non_proxied_udp",
                    "--disable-quic", "--dns-prefetch-disable",
                    "--disable-background-networking"):
                self.assertIn(argument, internal.fixed_args)
            encoded = json.dumps(browser)
            self.assertNotIn(str(server.MANAGED_BROWSER_PROFILE_DIR), encoded)
            self.assertNotIn(str(server.MANAGED_BROWSER_DEBUG_PORT), encoded)
            self.assertNotIn(str(server.MANAGED_BROWSER_PROXY_PORT), encoded)

    def test_server_browser_runtime_verifier_uses_exact_singleton_listener(self):
        calls = []

        class Process:
            @staticmethod
            def singleton_loopback_listener_matches(spec_id, port):
                calls.append((spec_id, port))
                return True

        proxy = Mock(healthy=Mock(return_value=True))
        with (patch.object(server, "PROCESS_BROKER", Process()),
              patch.object(server, "WEB_PROXY", proxy)):
            self.assertTrue(server._managed_browser_runtime_verified())
        self.assertEqual(calls, [(
            server.MANAGED_BROWSER_SPEC_ID,
            server.MANAGED_BROWSER_DEBUG_PORT,
        )])

        calls.clear()
        proxy.healthy.return_value = False
        with (patch.object(server, "PROCESS_BROKER", Process()),
              patch.object(server, "WEB_PROXY", proxy)):
            self.assertFalse(server._managed_browser_runtime_verified())
        self.assertEqual(calls, [])

    def test_curated_terminal_is_exact_versioned_and_session_scoped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            broker = _broker(root, GraphStore(root / "friday.db"))
            specs = {item["spec_id"]: item for item in broker.list_specs()}
            terminal_id = "app.friday_terminal.foot_1_27_0.v2"
            if Path("/usr/bin/foot").is_file():
                self.assertIn(terminal_id, specs)
                terminal = specs[terminal_id]
                self.assertEqual(
                    terminal["session_access"],
                    {"wayland": True, "session_bus": False})
                self.assertFalse(terminal["sandboxed"])
                self.assertTrue(terminal["network"])
                self.assertNotIn("executable", terminal)
                self.assertNotIn("environment", terminal)
                self.assertEqual(
                    terminal["presentation"]["application"],
                    "Friday Terminal")
                self.assertNotIn(
                    "application_id", terminal["presentation"])

    def test_terminal_launch_binding_combines_process_and_desktop_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process_broker = _broker(root, GraphStore(root / "friday.db"))
            spec_id = "app.friday_terminal.foot_1_27_0.v2"
            if not Path("/usr/bin/foot").is_file():
                self.skipTest("curated Foot profile is unavailable")
            process = process_broker.binding_for_launch(spec_id, {})
            spec = process_broker.registry.get(spec_id)
            expected = DesktopApplicationLaunchBinding(
                process=process, session_fingerprint="a" * 64,
                presentation_fingerprint=spec.presentation.fingerprint,
                application_id_sha256=sha256_text(
                    spec.presentation.application_id),
                application=spec.presentation.application)

            class Desktop:
                @staticmethod
                def binding_for_application_launch(binding, presentation):
                    self.assertEqual(binding, process)
                    self.assertEqual(presentation, spec.presentation)
                    return expected

            with patch.multiple(
                    server, PROCESS_BROKER=process_broker,
                    DESKTOP_BROKER=Desktop(), PROCESS_CLEANUP_LAST_ERROR=None):
                binding, claim, preview = server._bind_process_step(
                    "machine_launch_process",
                    {"spec_id": spec_id, "parameter_values": {}})

            self.assertEqual(binding, expected.model_dump(mode="json"))
            self.assertEqual(
                claim.model_dump(mode="json"),
                process.resource_claim.as_claim().model_dump(mode="json"))
            self.assertEqual(preview["presentation"]["application"],
                             "Friday Terminal")
            self.assertNotIn("application_id", json.dumps(preview))

    async def test_compound_launch_executes_process_then_confirms_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_broker = _broker(root, GraphStore(root / "friday.db"))
            spec_id = "app.friday_terminal.foot_1_27_0.v2"
            if not Path("/usr/bin/foot").is_file():
                self.skipTest("curated Foot profile is unavailable")
            process = registry_broker.binding_for_launch(spec_id, {})
            spec = registry_broker.registry.get(spec_id)
            managed_presentation = spec.presentation.model_copy(update={
                "window_owner": "managed_cgroup"})
            managed_spec = spec.model_copy(update={
                "presentation": managed_presentation})
            binding = DesktopApplicationLaunchBinding(
                process=process, session_fingerprint="b" * 64,
                presentation_fingerprint=managed_presentation.fingerprint,
                application_id_sha256=sha256_text(
                    managed_presentation.application_id),
                application=managed_presentation.application,
                window_owner="managed_cgroup")
            process_receipt = {
                "status": "ok", "verified": True,
                "instance_id": "process_" + "c" * 32,
                "spec_id": spec_id, "state": "running",
            }
            observation = object()
            member_checks = []

            class Registry:
                @staticmethod
                def get(selected):
                    self.assertEqual(selected, spec_id)
                    return managed_spec

            class Process:
                registry = Registry()

                @staticmethod
                def binding_for_launch(selected, values):
                    self.assertEqual((selected, values), (spec_id, {}))
                    return process

                @staticmethod
                def launch(*args, **kwargs):
                    self.assertEqual(args, (spec_id, {}))
                    self.assertEqual(kwargs["source_step_lease_id"],
                                     "resource_step_lease_0001")
                    return process_receipt

                @staticmethod
                def runtime_observation(instance_id):
                    self.assertEqual(instance_id,
                                     process_receipt["instance_id"])
                    return observation

                @staticmethod
                def runtime_process_member_matches(
                        instance_id, expected, **identity):
                    member_checks.append((instance_id, expected, identity))
                    return True

            class Desktop:
                @staticmethod
                def binding_for_application_launch(selected, presentation):
                    self.assertEqual(selected, process)
                    self.assertEqual(presentation, managed_presentation)
                    return binding

                @staticmethod
                def confirm_application_launch(
                        selected, receipt, observed, presentation,
                        runtime_owner):
                    self.assertEqual(selected, binding)
                    self.assertEqual(receipt, process_receipt)
                    self.assertIs(observed, observation)
                    self.assertEqual(presentation, managed_presentation)
                    owned_window = SimpleNamespace(
                        pid=2222, start_ticks=77_777,
                        executable_identity=process.executable_identity)
                    self.assertTrue(runtime_owner(observed, owned_window))
                    return dict(receipt) | {"presentation": {
                        "status": "ok", "verified": True,
                        "window_id": "win_" + "d" * 40,
                        "application": "Friday Terminal",
                    }}

            verified = SimpleNamespace(model_dump=lambda mode="json": {
                "status": "passed", "summary": "verified",
                "evidence": [], "missing": [], "effects": [],
            })
            outcome_verifier = SimpleNamespace(
                verify_action=lambda *_args, **_kwargs: verified)
            claim = ClaimedStep(
                step_id="step_application_launch_0001",
                batch_id="batch_application_launch_0001",
                task_id="task_application_launch_0001",
                round_index=0, ordinal=1,
                tool_call_id="call_application_launch_0001",
                tool_name="machine_launch_process",
                args={"spec_id": spec_id, "parameter_values": {}},
                idempotency_key="act_" + "e" * 64,
                idempotency_class="reconcilable", recovery_policy="reconcile",
                risk="high", approval_status="approved",
                action_id="action_application_launch_0001",
                attempt_id="attempt_application_launch_0001",
                attempt_number=1, lease_id="step_execution_lease_0001",
                worker_id="worker_application_launch_0001",
                verifier="process_launch_receipt",
                executor_binding=binding.model_dump(mode="json"),
                resource_claims=process.resource_claim.as_claim().model_dump(
                    mode="json"), context={},
                resource_lease_id="resource_step_lease_0001")
            friday = server.Friday.__new__(server.Friday)
            with patch.multiple(
                    server, PROCESS_BROKER=Process(), DESKTOP_BROKER=Desktop(),
                    OUTCOMES=outcome_verifier,
                    PROCESS_CLEANUP_LAST_ERROR=None):
                result = await friday.execute_claimed_step(claim)

            value = json.loads(result.result)
            self.assertTrue(result.succeeded)
            self.assertTrue(value["presentation"]["verified"])
            self.assertEqual(value["presentation"]["application"],
                             "Friday Terminal")
            self.assertEqual(len(member_checks), 1)
            self.assertEqual(member_checks[0][0], process_receipt["instance_id"])
            self.assertIs(member_checks[0][1], observation)
            self.assertEqual(member_checks[0][2]["pid"], 2222)

    async def test_probe_settles_exact_known_launch_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = GraphStore(root / "friday.db")
            tasks = TaskService(graph)
            registry_broker = _broker(root, graph)
            binding = registry_broker.binding_for_launch(
                "proc.managed_wait.v1", {"seconds": 30})
            contract = server.CONTRACTS.build(
                "Launch the managed process.", ["machine_launch_process"])
            task_id, _ = tasks.create(
                contract.objective, contract.model_dump(mode="json"))
            tasks.transition(task_id, "interpreting")
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")
            batch_id, steps = tasks.stage_step_batch(task_id, [{
                "tool_call_id": "launch-known-failure",
                "tool_name": "machine_launch_process",
                "args": {"spec_id": "proc.managed_wait.v1",
                         "parameter_values": {"seconds": 30}},
                "risk": "high", "approval_status": "approved",
                "idempotency_class": "reconcilable",
                "recovery_policy": "reconcile",
                "executor_binding": binding.model_dump(mode="json"),
                "resource_claims": {},
            }], round_index=0)
            claim = tasks.claim_next_step(batch_id, "failed-launch-worker")
            tasks.mark_step_outcome_unknown(
                claim, reason_code="worker_crashed_before_finish")
            tasks.recover_interrupted()
            candidate = tasks.reconciliation_candidate(steps[0]["step_id"])
            receipt = {
                "status": "ok", "verified": False,
                "instance_id": "process_" + "c" * 32,
                "spec_id": "proc.managed_wait.v1",
                "state": "launch_failed", "persistent": False,
                "idempotent_replay": True, "forced": False,
                "prepared_at": "2026-08-24T00:00:00Z",
                "started_at": None,
                "finished_at": "2026-08-24T00:00:01Z",
                "result_code": "backend_executable_identity_changed",
                "output": {"stdout_bytes": 0, "stdout_sha256": None,
                           "stdout_truncated": False, "stderr_bytes": 0,
                           "stderr_sha256": None,
                           "stderr_truncated": False},
            }
            reconciliation_calls = []

            def reconciliation_receipt(*call_args, **call_kwargs):
                reconciliation_calls.append((call_args, call_kwargs))
                return receipt

            process_broker = SimpleNamespace(
                reconciliation_receipt=reconciliation_receipt)

            with patch.multiple(
                    server, TASKS=tasks, GRAPH=graph,
                    PROCESS_BROKER=process_broker, WORKER=None):
                result = await server._probe_reconciliation(
                    steps[0]["step_id"])

            self.assertTrue(result["resolved"])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(tasks.get(task_id)["status"], "failed")
            self.assertEqual(reconciliation_calls, [(tuple([
                candidate.tool_name, candidate.executor_binding,
                candidate.args, candidate.idempotency_key,
            ]), {
                "task_id": candidate.task_id,
                "step_id": candidate.step_id,
                "action_id": candidate.action_id,
                "attempt_id": claim.attempt_id,
            })])
            with graph._connect() as conn:
                durable = conn.execute(
                    "SELECT status FROM action_receipts WHERE step_id=?",
                    (steps[0]["step_id"],)).fetchone()
            self.assertEqual(durable["status"], "failed")

    async def test_probe_requires_compound_window_receipt_for_gui_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = GraphStore(root / "friday.db")
            tasks = TaskService(graph)
            registry_broker = _broker(root, graph)
            spec_id = "app.friday_terminal.foot_1_27_0.v2"
            if not Path("/usr/bin/foot").is_file():
                self.skipTest("curated Foot profile is unavailable")
            process_binding = registry_broker.binding_for_launch(spec_id, {})
            spec = registry_broker.registry.get(spec_id)
            binding = DesktopApplicationLaunchBinding(
                process=process_binding, session_fingerprint="1" * 64,
                presentation_fingerprint=spec.presentation.fingerprint,
                application_id_sha256=sha256_text(
                    spec.presentation.application_id),
                application=spec.presentation.application)
            contract = server.CONTRACTS.build(
                "Open Friday Terminal.", ["machine_launch_process"])
            task_id, _ = tasks.create(
                contract.objective, contract.model_dump(mode="json"))
            tasks.transition(task_id, "interpreting")
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")
            batch_id, steps = tasks.stage_step_batch(task_id, [{
                "tool_call_id": "launch-gui-reconcile",
                "tool_name": "machine_launch_process",
                "args": {"spec_id": spec_id, "parameter_values": {}},
                "risk": "high", "approval_status": "approved",
                "idempotency_class": "reconcilable",
                "recovery_policy": "reconcile",
                "executor_binding": binding.model_dump(mode="json"),
                "resource_claims": {},
            }], round_index=0)
            claim = tasks.claim_next_step(batch_id, "gui-launch-worker")
            tasks.mark_step_outcome_unknown(
                claim, reason_code="desktop_application_launch_outcome_unknown")
            tasks.recover_interrupted()
            base = {
                "status": "ok", "verified": True,
                "instance_id": "process_" + "2" * 32,
                "spec_id": spec_id, "state": "running",
            }
            compound = dict(base) | {"presentation": {
                "status": "ok", "verified": True,
                "window_id": "win_" + "3" * 40,
                "application": "Friday Terminal",
            }}
            observation = object()
            calls = []

            class Process:
                registry = registry_broker.registry

                @staticmethod
                def reconciliation_receipt(*args, **kwargs):
                    calls.append("process_reconcile")
                    return base

                @staticmethod
                def runtime_observation(instance_id):
                    self.assertEqual(instance_id, base["instance_id"])
                    return observation

                @staticmethod
                def verify_receipt(tool_name, result, args, key):
                    calls.append("process_verify")
                    self.assertEqual(tool_name, "machine_launch_process")
                    self.assertNotIn("presentation", result)
                    return result == base

            class Desktop:
                @staticmethod
                def reconciliation_application_receipt(
                        selected, receipt, observed, presentation):
                    calls.append("desktop_reconcile")
                    self.assertEqual(selected, binding)
                    self.assertEqual(receipt, base)
                    self.assertIs(observed, observation)
                    self.assertEqual(presentation, spec.presentation)
                    return compound

                @staticmethod
                def verify_application_launch_receipt(
                        selected, result, receipt, observed, presentation):
                    calls.append("desktop_verify")
                    return bool(
                        selected == binding and result == compound
                        and receipt == base and observed is observation
                        and presentation == spec.presentation)

            with patch.multiple(
                    server, TASKS=tasks, GRAPH=graph,
                    PROCESS_BROKER=Process(), DESKTOP_BROKER=Desktop(),
                    WORKER=None):
                result = await server._probe_reconciliation(
                    steps[0]["step_id"])

            self.assertTrue(result["resolved"])
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(tasks.get(task_id)["status"], "completed")
            self.assertEqual(calls, [
                "process_reconcile", "desktop_reconcile",
                "process_verify", "desktop_verify",
            ])
            self.assertFalse(hasattr(Process, "launch"))

    async def test_probe_settles_exact_known_termination_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = GraphStore(root / "friday.db")
            tasks = TaskService(graph)
            registry_broker = _broker(root, graph)
            spec = registry_broker.registry.get("proc.managed_wait.v1")
            instance_id = "process_" + "d" * 32
            binding = ProcessInstanceBinding(
                operation="terminate", instance_id=instance_id,
                spec_id=spec.spec_id, spec_fingerprint=spec.fingerprint,
                sandbox_fingerprint=spec.sandbox_fingerprint,
                args_sha256="e" * 64, state="running",
                runtime_identity_sha256="f" * 64,
                state_fingerprint="1" * 64,
                persistent=False).model_dump(mode="json")
            contract = server.CONTRACTS.build(
                "Terminate the managed process.",
                ["machine_terminate_process"])
            task_id, _ = tasks.create(
                contract.objective, contract.model_dump(mode="json"))
            tasks.transition(task_id, "interpreting")
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")
            batch_id, steps = tasks.stage_step_batch(task_id, [{
                "tool_call_id": "termination-known-rejection",
                "tool_name": "machine_terminate_process",
                "args": {"instance_id": instance_id},
                "risk": "high", "approval_status": "approved",
                "idempotency_class": "reconcilable",
                "recovery_policy": "reconcile",
                "executor_binding": binding,
                "resource_claims": {},
            }], round_index=0)
            claim = tasks.claim_next_step(
                batch_id, "termination-reconciliation-worker")
            tasks.mark_step_outcome_unknown(
                claim, reason_code="worker_crashed_before_recording_rejection")
            tasks.recover_interrupted()
            candidate = tasks.reconciliation_candidate(steps[0]["step_id"])
            receipt = {
                "status": "failed", "verified": True,
                "instance_id": instance_id, "operation": "terminate",
                "result_code": "backend_stop_rejected",
                "idempotent_replay": True, "forced": False,
            }
            reconciliation_calls = []

            def reconciliation_receipt(*call_args, **call_kwargs):
                reconciliation_calls.append((call_args, call_kwargs))
                return receipt

            process_broker = SimpleNamespace(
                reconciliation_receipt=reconciliation_receipt)
            with patch.multiple(
                    server, TASKS=tasks, GRAPH=graph,
                    PROCESS_BROKER=process_broker, WORKER=None):
                result = await server._probe_reconciliation(
                    steps[0]["step_id"])

            self.assertTrue(result["resolved"])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(tasks.get(task_id)["status"], "failed")
            self.assertEqual(reconciliation_calls, [(tuple([
                candidate.tool_name, candidate.executor_binding,
                candidate.args, candidate.idempotency_key,
            ]), {
                "task_id": candidate.task_id,
                "step_id": candidate.step_id,
                "action_id": candidate.action_id,
                "attempt_id": candidate.attempt_id,
            })])
            self.assertFalse(hasattr(process_broker, "terminate"))
            with graph._connect() as conn:
                durable = conn.execute(
                    "SELECT status FROM action_receipts WHERE step_id=?",
                    (steps[0]["step_id"],)).fetchone()
            self.assertEqual(durable["status"], "failed")

    async def test_terminate_is_staged_as_reconcilable_never_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = GraphStore(root / "friday.db")
            tasks = TaskService(graph)
            approvals = ApprovalService(graph)
            registry_broker = _broker(root, graph)
            spec = registry_broker.registry.get("proc.managed_wait.v1")
            instance_id = "process_" + "b" * 32
            binding = ProcessInstanceBinding(
                operation="terminate", instance_id=instance_id,
                spec_id=spec.spec_id, spec_fingerprint=spec.fingerprint,
                sandbox_fingerprint=spec.sandbox_fingerprint,
                args_sha256=sha256_text("private-process-args"),
                state="running", runtime_identity_sha256="c" * 64,
                state_fingerprint="d" * 64, persistent=False)

            def bind_instance(selected, operation):
                self.assertEqual((selected, operation),
                                 (instance_id, "terminate"))
                return binding

            broker = SimpleNamespace(
                registry=registry_broker.registry,
                binding_for_instance=bind_instance)
            contract = server.CONTRACTS.build(
                "Terminate the managed process.",
                ["machine_terminate_process"])
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
                    "id": "call_process_terminate",
                    "name": "machine_terminate_process",
                    "args": json.dumps({"instance_id": instance_id}),
                }]

            friday._stream_once = fake_stream
            queue = asyncio.Queue()
            progress = []
            empty_context = SimpleNamespace(
                retrieve=lambda *_args, **_kwargs: [])
            no_context = SimpleNamespace(
                relevant_context=lambda *_args, **_kwargs: [])
            with patch.multiple(
                    server, TASKS=tasks, APPROVALS=approvals,
                    PROCESS_BROKER=broker, CAPABILITIES=_NoCapabilities(),
                    REFLECTION=ReflectionService(graph), MEMORY=empty_context,
                    FEEDBACK=no_context, SKILLS=no_context, WORKER=None):
                await friday.respond(
                    "Terminate that exact managed process.", queue,
                    existing_task_id=task_id,
                    progress_sink=lambda event: _collect(progress, event))

            step = tasks.list_steps(task_id=task_id)[0]
            self.assertEqual(step["idempotency_class"], "reconcilable")
            self.assertEqual(step["recovery_policy"], "reconcile")
            self.assertEqual(step["max_attempts"], 1)
            self.assertEqual(step["executor_binding"],
                             binding.model_dump(mode="json"))
            self.assertTrue(any(item.get("type") == "approval_required"
                                for item in progress))

    async def test_launch_is_bound_before_exact_ephemeral_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = GraphStore(root / "friday.db")
            tasks = TaskService(graph)
            approvals = ApprovalService(graph)
            broker = _broker(root, graph)
            contract = server.CONTRACTS.build(
                "Run the managed process canary.",
                ["machine_launch_process"])
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
                    "id": "call_process_launch",
                    "name": "machine_launch_process",
                    "args": json.dumps({
                        "spec_id": "proc.managed_wait.v1",
                        "parameter_values": {"seconds": 30},
                    }),
                }]

            friday._stream_once = fake_stream
            queue = asyncio.Queue()
            progress = []
            empty_context = SimpleNamespace(
                retrieve=lambda *_args, **_kwargs: [])
            no_feedback = SimpleNamespace(
                relevant_context=lambda *_args, **_kwargs: [])
            no_skills = SimpleNamespace(
                relevant_context=lambda *_args, **_kwargs: [])

            with patch.multiple(
                    server, TASKS=tasks, APPROVALS=approvals,
                    PROCESS_BROKER=broker, CAPABILITIES=_NoCapabilities(),
                    REFLECTION=ReflectionService(graph), MEMORY=empty_context,
                    FEEDBACK=no_feedback, SKILLS=no_skills, WORKER=None):
                await friday.respond(
                    "Run the managed process canary for 30 seconds.", queue,
                    existing_task_id=task_id,
                    progress_sink=lambda event: _collect(progress, event))

            step = tasks.list_steps(task_id=task_id)[0]
            pending = approvals.list(status="pending")[0]
            approval_event = next(
                item for item in progress
                if item.get("type") == "approval_required")
            expected_binding = broker.binding_for_launch(
                "proc.managed_wait.v1", {"seconds": 30})

            self.assertEqual(
                step["executor_binding"],
                expected_binding.model_dump(mode="json"))
            self.assertEqual(
                step["resource_claims"],
                expected_binding.resource_claim.as_claim().model_dump(mode="json"))
            self.assertEqual(step["idempotency_class"], "reconcilable")
            self.assertEqual(step["recovery_policy"], "reconcile")
            self.assertEqual(approval_event["args"]["parameter_values"],
                             {"seconds": 30})
            self.assertEqual(approval_event["args"]["display_name"],
                             "Managed wait (process-control canary)")
            self.assertEqual(approval_event["args"]["_args_sha256"],
                             step["args_sha256"])
            self.assertEqual(pending["args"]["parameter_values"], "[REDACTED]")
            self.assertEqual(pending["step_id"], step["step_id"])
            self.assertEqual(await queue.get(),
                             "I need your approval before I can do that.")
            self.assertIsNone(await queue.get())

    async def test_execution_revalidates_binding_and_passes_step_lease_fence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = GraphStore(root / "friday.db")
            broker = _broker(root, graph)
            binding = broker.binding_for_launch(
                "proc.managed_wait.v1", {"seconds": 30}).model_dump(mode="json")
            receipt = {
                "status": "ok", "verified": True,
                "instance_id": "process_" + "b" * 32,
                "spec_id": "proc.managed_wait.v1", "state": "running",
                "persistent": False, "idempotent_replay": False,
                "forced": False, "prepared_at": "2026-08-24T00:00:00Z",
                "started_at": "2026-08-24T00:00:00Z",
                "finished_at": None, "result_code": "success",
                "output": {"stdout_bytes": 0, "stdout_sha256": None,
                           "stdout_truncated": False, "stderr_bytes": 0,
                           "stderr_sha256": None,
                           "stderr_truncated": False},
            }
            friday = server.Friday.__new__(server.Friday)

            with (patch.object(broker, "launch", return_value=receipt) as launch,
                  patch.object(broker, "verify_receipt", return_value=True),
                  patch.object(server, "PROCESS_BROKER", broker)):
                outcome = await friday.execute_claimed_step(_claim(binding))

            self.assertTrue(outcome.succeeded)
            self.assertEqual(outcome.verification["status"], "passed")
            self.assertEqual(
                launch.call_args.kwargs["source_step_lease_id"],
                "resource_step_lease_0001")
            self.assertEqual(
                launch.call_args.kwargs["source_attempt_id"],
                "attempt_process_launch_0001")
            self.assertEqual(
                launch.call_args.kwargs["source_worker_id"],
                "worker_process_launch_0001")

            with patch.object(server, "PROCESS_BROKER", broker):
                with self.assertRaises(ProcessBindingError):
                    await friday.execute_claimed_step(
                        _claim(binding, mutate=True))

    async def test_termination_execution_forwards_exact_operation_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = GraphStore(root / "friday.db")
            broker = _broker(root, graph)
            spec = broker.registry.get("proc.managed_wait.v1")
            instance_id = "process_" + "7" * 32
            binding = ProcessInstanceBinding(
                operation="terminate", instance_id=instance_id,
                spec_id=spec.spec_id, spec_fingerprint=spec.fingerprint,
                sandbox_fingerprint=spec.sandbox_fingerprint,
                args_sha256="8" * 64, state="running",
                runtime_identity_sha256="9" * 64,
                state_fingerprint="a" * 64,
                persistent=False).model_dump(mode="json")
            receipt = {
                "status": "ok", "verified": True,
                "instance_id": instance_id, "spec_id": spec.spec_id,
                "state": "terminated", "persistent": False,
                "idempotent_replay": False, "forced": False,
                "prepared_at": "2026-08-24T00:00:00Z",
                "started_at": "2026-08-24T00:00:00Z",
                "finished_at": "2026-08-24T00:00:01Z",
                "result_code": "terminated",
                "output": {"stdout_bytes": 0, "stdout_sha256": None,
                           "stdout_truncated": False, "stderr_bytes": 0,
                           "stderr_sha256": None,
                           "stderr_truncated": False},
            }
            claim = _claim(binding, terminate=True)
            friday = server.Friday.__new__(server.Friday)

            with (patch.object(
                    broker, "terminate", return_value=receipt) as terminate,
                  patch.object(broker, "verify_receipt", return_value=True),
                  patch.object(server, "PROCESS_BROKER", broker),
                  patch.object(server, "PROCESS_CLEANUP_LAST_ERROR", None)):
                outcome = await friday.execute_claimed_step(claim)

            self.assertTrue(outcome.succeeded)
            self.assertEqual(terminate.call_args.args, (instance_id,))
            self.assertEqual(terminate.call_args.kwargs["expected_binding"],
                             binding)
            self.assertEqual(terminate.call_args.kwargs["operation_context"], {
                "task_id": claim.task_id, "step_id": claim.step_id,
                "action_id": claim.action_id,
                "idempotency_key": claim.idempotency_key,
                "attempt_id": claim.attempt_id,
                "attempt_number": claim.attempt_number,
                "lease_id": claim.lease_id, "worker_id": claim.worker_id,
            })

    async def test_uncertain_process_receipt_uses_unknown_disposition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = GraphStore(root / "friday.db")
            broker = _broker(root, graph)
            binding = broker.binding_for_launch(
                "proc.managed_wait.v1", {"seconds": 30}).model_dump(mode="json")
            receipt = {
                "status": "ok", "verified": False,
                "instance_id": "process_" + "b" * 32,
                "spec_id": "proc.managed_wait.v1",
                "state": "reconcile_required", "persistent": False,
                "idempotent_replay": False, "forced": False,
                "prepared_at": "2026-08-24T00:00:00Z",
                "started_at": None, "finished_at": None,
                "result_code": "backend_identity_unavailable",
                "output": {"stdout_bytes": 0, "stdout_sha256": None,
                           "stdout_truncated": False, "stderr_bytes": 0,
                           "stderr_sha256": None,
                           "stderr_truncated": False},
            }
            friday = server.Friday.__new__(server.Friday)
            with (patch.object(broker, "launch", return_value=receipt),
                  patch.object(server, "PROCESS_BROKER", broker)):
                outcome = await friday.execute_claimed_step(_claim(binding))
            self.assertFalse(outcome.succeeded)
            self.assertTrue(outcome.outcome_unknown)
            self.assertEqual(outcome.verification["status"], "uncertain")

    async def test_authoritative_launch_failure_is_not_quarantined(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = GraphStore(root / "friday.db")
            broker = _broker(root, graph)
            binding = broker.binding_for_launch(
                "proc.managed_wait.v1", {"seconds": 30}).model_dump(mode="json")
            receipt = {
                "status": "ok", "verified": False,
                "instance_id": "process_" + "b" * 32,
                "spec_id": "proc.managed_wait.v1",
                "state": "launch_failed", "persistent": False,
                "idempotent_replay": False, "forced": False,
                "prepared_at": "2026-08-24T00:00:00Z",
                "started_at": None,
                "finished_at": "2026-08-24T00:00:01Z",
                "result_code": "backend_executable_identity_changed",
                "output": {"stdout_bytes": 0, "stdout_sha256": None,
                           "stdout_truncated": False, "stderr_bytes": 0,
                           "stderr_sha256": None,
                           "stderr_truncated": False},
            }
            friday = server.Friday.__new__(server.Friday)
            with (patch.object(broker, "launch", return_value=receipt),
                  patch.object(broker, "verify_receipt", return_value=False),
                  patch.object(server, "PROCESS_BROKER", broker)):
                outcome = await friday.execute_claimed_step(_claim(binding))

            self.assertFalse(outcome.outcome_unknown)
            self.assertEqual(outcome.verification["status"], "failed")


if __name__ == "__main__":
    unittest.main()
