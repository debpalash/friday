from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from friday_core.admission import (
    AdmissionBudget,
    ResourceAdmissionController,
    ResourceSnapshot,
)
from friday_core.graph import GraphStore
from friday_core.feedback import ApprovalService
from friday_core.processes import (
    BackendLaunchRequest,
    BackendObservation,
    BackendTerminalFence,
    BubblewrapProfile,
    ExecutableIdentity,
    ProcessBackendError,
    ProcessBindingError,
    ProcessBroker,
    ProcessBrokerError,
    ProcessCleanupBlocked,
    ProcessIdentityError,
    ProcessLimits,
    ProcessOperationContext,
    ProcessParameter,
    ProcessPresentation,
    ProcessResources,
    ProcessSessionAccess,
    ProcessSpec,
    ProcessSpecError,
    ProcessSpecRegistry,
    SystemdUserProcessBackend,
)
from friday_core.tasks import ClaimedStep, TaskService

from tests.platform_markers import require_platform

require_platform('linux')


USER_MANAGER_CGROUP = (
    f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service")


def unit_cgroup(unit_name: str) -> str:
    return (f"{USER_MANAGER_CGROUP}/friday.slice/"
            f"friday-processes.slice/{unit_name}")


class FakeProcessBackend:
    """Exact-identity backend fake; it never executes the rendered argv."""

    def __init__(self, *, fast_exit: bool = False,
                 resist_term: bool = False,
                 wrong_target: bool = False):
        self.fast_exit = fast_exit
        self.resist_term = resist_term
        self.wrong_target = wrong_target
        self.launch_count = 0
        self.terminate_calls: list[bool] = []
        self.retire_calls: list[str] = []
        self.requests: list[BackendLaunchRequest] = []
        self.observations: dict[str, BackendObservation] = {}
        self.member_identities: dict[
            tuple[str, int], tuple[int, int, int, int, str]] = {}
        self.loopback_listeners: set[tuple[str, int]] = set()
        self._lock = threading.Lock()

    @staticmethod
    def enforcement(_spec: ProcessSpec) -> dict[str, object]:
        return {"backend": "fake_owned_cgroup", "cpu": "hard",
                "memory": "hard", "pids": "hard"}

    @staticmethod
    def supports_persistence() -> bool:
        return False

    def _running(self, request: BackendLaunchRequest,
                 sequence: int) -> BackendObservation:
        identity = request.executable_identity
        return BackendObservation(
            unit_name=request.unit_name,
            identity_token=request.identity_token,
            state="running",
            boot_id="test-boot-id",
            invocation_id=f"invocation-{sequence}",
            control_group=f"/test.slice/process-{sequence}.scope",
            leader_pid=10_000 + sequence,
            start_ticks=50_000 + sequence,
            exe_device=identity.device,
            exe_inode=identity.inode,
            exe_sha256=("f" * 64 if self.wrong_target else identity.sha256),
            cgroup_empty=False,
            result_code="success",
        )

    @staticmethod
    def _exited(observation: BackendObservation, *, preserve_identity: bool
                ) -> BackendObservation:
        values: dict[str, object] = {
            "unit_name": observation.unit_name,
            "identity_token": observation.identity_token,
            "state": "exited",
            "boot_id": observation.boot_id,
            "invocation_id": observation.invocation_id,
            "control_group": observation.control_group,
            "cgroup_empty": True,
            "exit_code": 0,
            "result_code": "success",
        }
        if preserve_identity:
            values.update({
                "leader_pid": observation.leader_pid,
                "start_ticks": observation.start_ticks,
                "exe_device": observation.exe_device,
                "exe_inode": observation.exe_inode,
                "exe_sha256": observation.exe_sha256,
            })
        return BackendObservation(**values)

    def launch(self, request: BackendLaunchRequest) -> BackendObservation:
        with self._lock:
            self.launch_count += 1
            sequence = self.launch_count
            self.requests.append(request)
            running = self._running(request, sequence)
            observed = (self._exited(running, preserve_identity=False)
                        if self.fast_exit else running)
            self.observations[request.unit_name] = observed
        # Enlarge the idempotency race without holding the fake's lock.
        time.sleep(0.01)
        return observed

    def adopt_preexisting(self, request: BackendLaunchRequest) -> None:
        with self._lock:
            self.requests.append(request)
            self.observations[request.unit_name] = self._running(request, 77)

    def inspect(self, unit_name: str) -> BackendObservation | None:
        with self._lock:
            return self.observations.get(unit_name)

    def member_identity(
        self, expected: BackendObservation, pid: int,
    ) -> tuple[int, int, int, int, str] | None:
        with self._lock:
            current = self.observations.get(expected.unit_name)
            if current is None or not expected.same_live_execution(current):
                raise ProcessIdentityError()
            custom = self.member_identities.get((expected.unit_name, pid))
            if custom is not None:
                return custom
            if pid == current.leader_pid:
                return (pid, int(current.start_ticks), int(current.exe_device),
                        int(current.exe_inode), str(current.exe_sha256))
            return None

    def owns_loopback_listener(
        self, expected: BackendObservation, port: int,
    ) -> bool:
        with self._lock:
            current = self.observations.get(expected.unit_name)
            if current is None or not expected.same_live_execution(current):
                raise ProcessIdentityError()
            return (expected.unit_name, port) in self.loopback_listeners

    def terminate(self, expected: BackendObservation, *,
                  force: bool = False) -> BackendObservation:
        with self._lock:
            current = self.observations.get(expected.unit_name)
            if current is None or not expected.same_live_execution(current):
                raise ProcessIdentityError()
            self.terminate_calls.append(force)
            if self.resist_term and not force:
                result = current.model_copy(update={
                    "state": "stopping", "result_code": "term_pending"})
            else:
                result = self._exited(current, preserve_identity=True)
            self.observations[expected.unit_name] = result
            return result

    def retire_terminal(self, fence: BackendTerminalFence) -> None:
        with self._lock:
            current = self.observations.get(fence.unit_name)
            if current is None:
                return
            if (current.identity_token != fence.identity_token
                    or fence.invocation_id is None
                    or current.invocation_id != fence.invocation_id
                    or fence.control_group is None
                    or current.control_group != fence.control_group
                    or (fence.boot_id is not None
                        and current.boot_id != fence.boot_id)):
                raise ProcessCleanupBlocked(
                    "process_unit_cleanup_identity_mismatch")
            if not current.cgroup_empty or current.state != "exited":
                raise ProcessCleanupBlocked(
                    "process_unit_cleanup_cgroup_not_empty")
            self.retire_calls.append(fence.unit_name)
            del self.observations[fence.unit_name]

    def natural_exit(self, instance_id: str) -> None:
        unit = ProcessBroker._unit_name(instance_id)
        with self._lock:
            current = self.observations[unit]
            self.observations[unit] = self._exited(
                current, preserve_identity=False)

    def replace_identity(self, instance_id: str) -> None:
        unit = ProcessBroker._unit_name(instance_id)
        with self._lock:
            current = self.observations[unit]
            self.observations[unit] = current.model_copy(update={
                "invocation_id": "foreign-invocation"})


class ProcessBrokerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.executable = self.root / "curated-app"
        shutil.copyfile("/usr/bin/true", self.executable)
        self.executable.chmod(0o755)
        self.graph = GraphStore(self.root / "friday.db")
        self.admission = ResourceAdmissionController(
            self.graph,
            AdmissionBudget(
                cpu_millis=8_000,
                ram_mib=16_384,
                concurrency_slots=16,
                network_slots=16,
                accelerator_vram_mib={}),
            snapshot_provider=lambda: ResourceSnapshot(
                available_cpu_millis=8_000,
                available_ram_mib=16_384,
                available_network_slots=16,
                available_accelerator_vram_mib={},
                captured_at=datetime.now(UTC)),
            snapshot_ttl_seconds=0,
            runtime_id="runtime-process-tests",
            profile_fingerprint="a" * 64,
        )
        self.spec = self.make_spec()
        self.registry = ProcessSpecRegistry([self.spec])
        self.backend = FakeProcessBackend()
        self.broker = ProcessBroker(
            self.graph, self.registry, self.backend, self.admission,
            state_root=self.root / "state")
        self.tasks = TaskService(self.graph, admission=self.admission)
        self.approvals = ApprovalService(self.graph)
        self.claim_counter = 0

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_spec(self, **updates: object) -> ProcessSpec:
        values: dict[str, object] = {
            "spec_id": "ProcessSpec_test_app_v1",
            "name": "test-app",
            "version": 1,
            "display_name": "Test application",
            "executable": str(self.executable),
            "cwd": str(self.root),
            "parameters": (
                ProcessParameter(
                    name="label", kind="string", flag="--label",
                    min_length=0, max_length=256),
                ProcessParameter(
                    name="count", kind="integer", flag="--count",
                    required=False, default=1, minimum=1, maximum=5),
            ),
            "resources": ProcessResources(
                cpu_cores=1.0, ram_mib=512, network=True,
                latency_class="background"),
            "limits": ProcessLimits(
                cpu_quota_percent=100.0,
                memory_high_mib=384,
                memory_max_mib=512,
                runtime_max_seconds=60,
                stop_grace_seconds=0.2),
            "sandbox": BubblewrapProfile(enabled=False),
        }
        values.update(updates)
        return ProcessSpec(**values)

    def claim(self, values: dict[str, object]) -> ClaimedStep:
        self.claim_counter += 1
        suffix = self.claim_counter
        task_id, _ = self.tasks.create(
            f"Launch curated process {suffix}", {"launched": True})
        call = {
            "tool_call_id": f"process-call-{suffix}",
            "tool_name": "machine_launch_process",
            "args": {"spec_id": self.spec.spec_id,
                     "parameter_values": values},
            "idempotency_class": "reconcilable",
            "recovery_policy": "reconcile",
            "executor_binding": self.broker.binding_for_launch(
                self.spec.spec_id, values).model_dump(mode="json"),
            "resource_claims": self.spec.resources.as_claim().model_dump(
                mode="json"),
        }
        batch_id, _ = self.tasks.stage_step_batch(
            task_id, [call], round_index=0)
        claim = self.tasks.claim_next_step(
            batch_id, f"process-worker-{suffix}", lease_seconds=60)
        self.assertIsNotNone(claim)
        self.assertIsNotNone(claim.resource_lease_id)
        return claim

    def launch(self, values: dict[str, object] | None = None,
               *, claim: ClaimedStep | None = None) -> dict[str, object]:
        values = values or {"label": "ordinary"}
        claim = claim or self.claim(values)
        return self.broker.launch(
            self.spec.spec_id,
            values,
            launch_idempotency_key=claim.idempotency_key,
            source_step_lease_id=str(claim.resource_lease_id),
            source_attempt_id=claim.attempt_id,
            source_worker_id=claim.worker_id,
            task_id=claim.task_id,
            step_id=claim.step_id,
            action_id=claim.action_id,
        )

    def terminate_claim(
        self,
        instance_id: str,
        *,
        broker: ProcessBroker | None = None,
        binding=None,
        label: str = "terminate",
    ) -> tuple[ClaimedStep, object]:
        broker = broker or self.broker
        binding = binding or broker.binding_for_instance(
            instance_id, "terminate")
        self.claim_counter += 1
        suffix = self.claim_counter
        task_id, _ = self.tasks.create(
            f"Terminate exact process {suffix}", {"terminated": True})
        exact_args = {"instance_id": instance_id}
        batch_id, steps = self.tasks.stage_step_batch(task_id, [{
            "tool_call_id": f"{label}-{suffix}",
            "tool_name": "machine_terminate_process",
            "args": exact_args,
            "risk": "high", "approval_status": "pending",
            "idempotency_class": "reconcilable",
            "recovery_policy": "reconcile",
            "executor_binding": binding.model_dump(mode="json"),
            "resource_claims": {},
        }], round_index=0)
        approval = self.approvals.request(
            task_id, "machine_terminate_process", exact_args,
            "Exact process termination requires approval.",
            step_id=steps[0]["step_id"])
        self.approvals.decide(approval["approval_id"], True)
        claim = self.tasks.claim_next_step(
            batch_id, f"terminate-worker-{suffix}", lease_seconds=60)
        self.assertIsNotNone(claim)
        return claim, binding

    @staticmethod
    def operation_context(claim: ClaimedStep) -> ProcessOperationContext:
        return ProcessOperationContext(
            task_id=claim.task_id, step_id=claim.step_id,
            action_id=claim.action_id,
            idempotency_key=claim.idempotency_key,
            attempt_id=claim.attempt_id,
            attempt_number=claim.attempt_number,
            lease_id=claim.lease_id, worker_id=claim.worker_id)

    def terminate_exact(
        self,
        broker: ProcessBroker,
        instance_id: str,
        claim: ClaimedStep,
        binding,
    ) -> dict[str, object]:
        return broker.terminate(
            instance_id, expected_binding=binding,
            operation_context=self.operation_context(claim))

    def instance_row(self, instance_id: str) -> dict[str, object]:
        with self.graph._connect() as conn:
            row = conn.execute(
                "SELECT * FROM process_instances WHERE instance_id=?",
                (instance_id,)).fetchone()
        self.assertIsNotNone(row)
        return dict(row)

    def workload_row(self, instance_id: str) -> dict[str, object]:
        with self.graph._connect() as conn:
            row = conn.execute(
                "SELECT * FROM workload_resource_leases WHERE instance_id=?",
                (instance_id,)).fetchone()
        self.assertIsNotNone(row)
        return dict(row)

    def cleanup_row(self, instance_id: str) -> dict[str, object]:
        with self.graph._connect() as conn:
            row = conn.execute(
                "SELECT * FROM process_unit_cleanups WHERE instance_id=?",
                (instance_id,)).fetchone()
        self.assertIsNotNone(row)
        return dict(row)

    def operation_row(self, idempotency_key: str) -> dict[str, object]:
        with self.graph._connect() as conn:
            row = conn.execute(
                "SELECT * FROM process_operations WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
        self.assertIsNotNone(row)
        return dict(row)

    def naturally_exit(self, instance_id: str) -> dict[str, object]:
        self.backend.natural_exit(instance_id)
        receipt = self.broker.inspect(instance_id)
        self.assertEqual(receipt["state"], "exited")
        return receipt

    def test_shell_metacharacters_are_one_literal_argument(self):
        marker = self.root / "must-not-exist"
        literal = f"$(touch {marker}); `id` && echo $HOME"

        receipt = self.launch({"label": literal})

        self.assertEqual(receipt["state"], "running")
        request = self.backend.requests[0]
        self.assertIn(literal, request.argv)
        self.assertEqual(request.argv.count(literal), 1)
        self.assertFalse(marker.exists())

    def test_changed_pinned_spec_degrades_without_crashing_or_repinning(self):
        changed = self.make_spec(fixed_args=("--changed-package-binding",))

        broker = ProcessBroker(
            self.graph, ProcessSpecRegistry([changed]), self.backend,
            self.admission, state_root=self.root / "state")

        self.assertEqual(broker.list_specs(), [])
        self.assertEqual(
            broker.degraded_specs,
            {changed.spec_id: "durable_process_spec_binding_changed"})
        with self.assertRaisesRegex(PermissionError, "process spec is not active"):
            broker.binding_for_launch(
                changed.spec_id, {"label": "must-not-launch"})
        with self.graph._connect() as conn:
            durable = conn.execute(
                "SELECT spec_sha256 FROM process_specs WHERE spec_id=?",
                (changed.spec_id,)).fetchone()[0]
        self.assertEqual(durable, self.spec.fingerprint)
        self.assertNotEqual(durable, changed.fingerprint)

    def test_absent_curated_spec_is_journal_revoked_only_when_unused(self):
        receipt = self.launch({"label": "still-running"})
        absent = ProcessBroker(
            self.graph, ProcessSpecRegistry(), self.backend,
            self.admission, state_root=self.root / "state")
        self.assertEqual(
            absent.degraded_specs,
            {self.spec.spec_id:
             "durable_process_spec_absent_with_active_instance"})
        with self.graph._connect() as conn:
            self.assertEqual(conn.execute(
                "SELECT status FROM process_specs WHERE spec_id=?",
                (self.spec.spec_id,)).fetchone()[0], "active")

        self.backend.natural_exit(str(receipt["instance_id"]))
        self.broker.inspect(str(receipt["instance_id"]))
        retired = ProcessBroker(
            self.graph, ProcessSpecRegistry(), self.backend,
            self.admission, state_root=self.root / "state")
        self.assertEqual(retired.degraded_specs, {})
        with self.graph._connect() as conn:
            row = conn.execute(
                "SELECT status,last_event_seq FROM process_specs WHERE spec_id=?",
                (self.spec.spec_id,)).fetchone()
            event = conn.execute(
                "SELECT event_type,payload_json FROM graph_events WHERE seq=?",
                (row["last_event_seq"],)).fetchone()
        self.assertEqual(row["status"], "revoked")
        self.assertEqual(event["event_type"], "process_spec.revoked")
        self.assertIn("absent_from_curated_registry", event["payload_json"])

    def test_interpreter_symlink_mutable_and_forbidden_environment_denied(self):
        with self.assertRaises((ProcessSpecError, ValidationError)):
            self.make_spec(executable="/usr/bin/env")

        alias = self.root / "app-alias"
        alias.symlink_to(self.executable)
        with self.assertRaises((ProcessSpecError, ValidationError)):
            self.make_spec(executable=str(alias))

        mutable = self.root / "mutable-app"
        shutil.copyfile("/usr/bin/true", mutable)
        mutable.chmod(0o775)
        with self.assertRaises((ProcessSpecError, ValidationError)):
            self.make_spec(executable=str(mutable))

        for environment in (
                {"PATH": "/tmp"}, {"LD_PRELOAD": "x"},
                {"PYTHONPATH": "x"}, {"API_TOKEN": "secret"}):
            with self.subTest(environment=environment), self.assertRaises(
                    (ValueError, ValidationError)):
                self.make_spec(environment=environment)

        # Identity is re-pinned immediately before intent/launch.
        with self.executable.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            byte = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([byte[0] ^ 1]))
        with self.assertRaises(ProcessSpecError):
            self.broker.binding_for_launch(
                self.spec.spec_id, {"label": "changed"})
        self.assertEqual(self.broker.list_specs(), [])
        self.assertEqual(self.backend.launch_count, 0)

    def test_session_access_is_explicit_fingerprinted_and_unsandboxed_only(self):
        with self.assertRaises((ValueError, ValidationError)):
            self.make_spec(
                sandbox=BubblewrapProfile(enabled=True),
                session_access=ProcessSessionAccess(wayland=True))
        with self.assertRaises((ValueError, ValidationError)):
            self.make_spec(presentation=ProcessPresentation(
                application_id="com.example.Test",
                application="Test"))

        ordinary_payload = self.spec.model_dump(mode="json")
        ordinary_payload.pop("session_access", None)
        ordinary_payload.pop("presentation", None)
        ordinary_payload.pop("instance_policy", None)
        from friday_core.graph import canonical_json, sha256_text
        self.assertEqual(
            self.spec.fingerprint, sha256_text(canonical_json(ordinary_payload)))

        desktop = self.make_spec(
            spec_id="ProcessSpec_desktop_app_v1",
            name="desktop-app",
            session_access=ProcessSessionAccess(wayland=True),
            presentation=ProcessPresentation(
                application_id="com.example.Test",
                application="Test"))
        self.assertNotEqual(desktop.fingerprint, self.spec.fingerprint)
        self.assertEqual(
            desktop.safe_display()["session_access"],
            {"wayland": True, "session_bus": False})
        self.assertEqual(desktop.safe_display()["presentation"]["application"],
                         "Test")
        self.assertNotIn(
            "application_id", desktop.safe_display()["presentation"])
        self.assertEqual(
            desktop.safe_display()["presentation"]["window_owner"],
            "leader")

        leader = ProcessPresentation(
            application_id="com.example.Test", application="Test")
        legacy_payload = leader.model_dump(mode="json")
        legacy_payload.pop("window_owner")
        from friday_core.graph import canonical_json, sha256_text
        self.assertEqual(
            leader.fingerprint,
            sha256_text(canonical_json(legacy_payload)))
        legacy_spec_payload = desktop.model_dump(mode="json")
        legacy_spec_payload["presentation"].pop("window_owner")
        legacy_spec_payload.pop("instance_policy")
        self.assertEqual(
            desktop.fingerprint,
            sha256_text(canonical_json(legacy_spec_payload)))
        managed = leader.model_copy(update={"window_owner": "managed_cgroup"})
        self.assertNotEqual(managed.fingerprint, leader.fingerprint)

    def test_singleton_policy_is_atomic_replay_safe_and_reusable_after_exit(self):
        singleton = self.make_spec(
            spec_id="ProcessSpec_singleton_app_v1",
            name="singleton-app", instance_policy="singleton")
        broker = ProcessBroker(
            self.graph, ProcessSpecRegistry([self.spec, singleton]),
            self.backend, self.admission,
            state_root=self.root / "singleton-state")
        tasks = TaskService(self.graph, admission=self.admission)

        def claim_for(suffix: str) -> ClaimedStep:
            values = {"label": "singleton"}
            binding = broker.binding_for_launch(singleton.spec_id, values)
            task_id, _ = tasks.create(
                f"Singleton launch {suffix}", {"launched": True})
            batch_id, _ = tasks.stage_step_batch(task_id, [{
                "tool_call_id": f"singleton-{suffix}",
                "tool_name": "machine_launch_process",
                "args": {"spec_id": singleton.spec_id,
                         "parameter_values": values},
                "idempotency_class": "reconcilable",
                "recovery_policy": "reconcile",
                "executor_binding": binding.model_dump(mode="json"),
                "resource_claims": singleton.resources.as_claim().model_dump(
                    mode="json"),
            }], round_index=0)
            claim = tasks.claim_next_step(
                batch_id, f"singleton-worker-{suffix}", lease_seconds=60)
            self.assertIsNotNone(claim)
            return claim

        first_claim = claim_for("first")
        second_claim = claim_for("second")

        def launch(claim: ClaimedStep):
            return broker.launch(
                singleton.spec_id, {"label": "singleton"},
                launch_idempotency_key=claim.idempotency_key,
                source_step_lease_id=str(claim.resource_lease_id),
                source_attempt_id=claim.attempt_id,
                source_worker_id=claim.worker_id,
                task_id=claim.task_id, step_id=claim.step_id,
                action_id=claim.action_id)

        receipts = []
        errors = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [(claim, pool.submit(launch, claim))
                       for claim in (first_claim, second_claim)]
            for claim, future in futures:
                try:
                    receipts.append((claim, future.result()))
                except ProcessBackendError as exc:
                    errors.append(exc)

        self.assertEqual(self.backend.launch_count, 1)
        self.assertEqual(len(receipts), 1)
        self.assertEqual([error.code for error in errors],
                         ["process_singleton_active"])
        winning_claim, winning_receipt = receipts[0]
        replay = launch(winning_claim)
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(replay["instance_id"], winning_receipt["instance_id"])

        runtime = broker.singleton_runtime_observation(singleton.spec_id)
        self.assertEqual(runtime[0], winning_receipt["instance_id"])
        self.assertEqual(runtime[1].state, "running")
        self.assertFalse(broker.singleton_loopback_listener_matches(
            singleton.spec_id, 9223))
        self.backend.loopback_listeners.add((runtime[1].unit_name, 9223))
        self.assertTrue(broker.singleton_loopback_listener_matches(
            singleton.spec_id, 9223))

        self.backend.natural_exit(str(winning_receipt["instance_id"]))
        broker.inspect(str(winning_receipt["instance_id"]))
        third = launch(claim_for("third"))
        self.assertEqual(third["state"], "running")
        self.assertEqual(self.backend.launch_count, 2)

    def test_singleton_runtime_refuses_multiple_policy_and_identity_drift(self):
        with self.assertRaises(ProcessBindingError):
            self.broker.singleton_runtime_observation(self.spec.spec_id)

        singleton = self.make_spec(
            spec_id="ProcessSpec_singleton_probe_v1",
            name="singleton-probe", instance_policy="singleton")
        broker = ProcessBroker(
            self.graph, ProcessSpecRegistry([self.spec, singleton]),
            self.backend, self.admission,
            state_root=self.root / "singleton-probe-state")
        self.assertNotEqual(singleton.fingerprint, self.spec.fingerprint)
        self.assertEqual(
            singleton.safe_display()["instance_policy"], "singleton")
        multiple = singleton.model_copy(update={"instance_policy": "multiple"})
        multiple_payload = multiple.model_dump(mode="json")
        for legacy_default in (
                "session_access", "presentation", "instance_policy"):
            multiple_payload.pop(legacy_default)
        from friday_core.graph import canonical_json, sha256_text
        self.assertEqual(
            multiple.fingerprint,
            sha256_text(canonical_json(multiple_payload)))
        self.assertNotEqual(singleton.fingerprint, multiple.fingerprint)
        self.assertIsNone(
            broker.singleton_runtime_observation(singleton.spec_id))

    def test_runtime_member_proof_binds_exact_instance_and_process_identity(self):
        receipt = self.launch({"label": "member-proof"})
        instance_id = str(receipt["instance_id"])
        execution = self.broker.runtime_observation(instance_id)
        child = ExecutableIdentity(
            device=9, inode=98765, sha256="e" * 64,
            size=8192, mode=0o755)
        self.backend.member_identities[(execution.unit_name, 2222)] = (
            2222, 77_777, child.device, child.inode, child.sha256)

        self.assertTrue(self.broker.runtime_process_member_matches(
            instance_id, execution, pid=2222, start_ticks=77_777,
            executable_identity=child))
        self.assertFalse(self.broker.runtime_process_member_matches(
            instance_id, execution, pid=2222, start_ticks=77_778,
            executable_identity=child))
        self.assertFalse(self.broker.runtime_process_member_matches(
            instance_id, execution, pid=3333, start_ticks=1,
            executable_identity=child))

        self.backend.replace_identity(instance_id)
        with self.assertRaises(ProcessIdentityError):
            self.broker.runtime_process_member_matches(
                instance_id, execution, pid=2222, start_ticks=77_777,
                executable_identity=child)

    def test_concurrent_idempotent_launch_spawns_exactly_once(self):
        values = {"label": "same"}
        claim = self.claim(values)
        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(
                lambda _index: self.launch(values, claim=claim), range(2)))

        self.assertEqual(self.backend.launch_count, 1)
        self.assertEqual(
            {receipt["instance_id"] for receipt in receipts},
            {receipts[0]["instance_id"]})
        self.assertEqual(
            sorted(receipt["idempotent_replay"] for receipt in receipts),
            [False, True])
        with self.graph._connect() as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM workload_resource_leases").fetchone()[0], 1)

    def test_same_idempotency_key_with_changed_args_is_rejected(self):
        first_values = {"label": "first"}
        claim = self.claim(first_values)
        self.launch(first_values, claim=claim)

        with self.assertRaises(ProcessBrokerError):
            self.launch({"label": "different"}, claim=claim)
        self.assertEqual(self.backend.launch_count, 1)

    def test_prepared_crash_window_adopts_existing_backend_effect(self):
        values = {"label": "adopt"}
        claim = self.claim(values)
        rendered = self.registry.render(self.spec.spec_id, values)
        row, created, _lease_id = self.broker._prepare(
            self.spec, rendered,
            launch_idempotency_key=claim.idempotency_key,
            task_id=claim.task_id,
            step_id=claim.step_id,
            action_id=claim.action_id,
            source_step_lease_id=str(claim.resource_lease_id),
            source_attempt_id=claim.attempt_id,
            source_worker_id=claim.worker_id,
        )
        self.assertTrue(created)
        self.backend.adopt_preexisting(
            self.broker._request(row, self.spec, rendered))

        restarted = ProcessBroker(
            self.graph, self.registry, self.backend, self.admission,
            state_root=self.root / "state")
        receipt = restarted.launch(
            self.spec.spec_id, values,
            launch_idempotency_key=claim.idempotency_key,
            source_step_lease_id=str(claim.resource_lease_id),
            source_attempt_id=claim.attempt_id,
            source_worker_id=claim.worker_id,
            task_id=claim.task_id,
            step_id=claim.step_id,
            action_id=claim.action_id)

        self.assertEqual(receipt["state"], "running")
        self.assertTrue(receipt["idempotent_replay"])
        self.assertEqual(self.backend.launch_count, 0)

    def test_wrong_first_target_and_identity_replacement_never_signal(self):
        wrong_backend = FakeProcessBackend(wrong_target=True)
        wrong_broker = ProcessBroker(
            self.graph, self.registry, wrong_backend, self.admission,
            state_root=self.root / "wrong-state")
        claim = self.claim({"label": "wrong-target"})
        receipt = wrong_broker.launch(
            self.spec.spec_id, {"label": "wrong-target"},
            launch_idempotency_key=claim.idempotency_key,
            source_step_lease_id=str(claim.resource_lease_id),
            source_attempt_id=claim.attempt_id,
            source_worker_id=claim.worker_id,
            task_id=claim.task_id, step_id=claim.step_id,
            action_id=claim.action_id)
        self.assertEqual(receipt["state"], "identity_mismatch")
        self.assertFalse(receipt["verified"])
        self.assertEqual(self.workload_row(
            str(receipt["instance_id"]))["status"], "reconciling")

        # Use a fresh database-backed launch for post-start identity replacement.
        second = self.launch({"label": "replace"})
        instance_id = str(second["instance_id"])
        self.backend.replace_identity(instance_id)
        with self.assertRaises(ProcessIdentityError):
            self.broker.terminate(instance_id)
        self.assertEqual(self.backend.terminate_calls, [])
        self.assertEqual(self.instance_row(instance_id)["state"],
                         "identity_mismatch")

    def test_term_resistance_stays_nonterminal_until_separate_force(self):
        backend = FakeProcessBackend(resist_term=True)
        broker = ProcessBroker(
            self.graph, self.registry, backend, self.admission,
            state_root=self.root / "term-state")
        claim = self.claim({"label": "resistant"})
        running = broker.launch(
            self.spec.spec_id, {"label": "resistant"},
            launch_idempotency_key=claim.idempotency_key,
            source_step_lease_id=str(claim.resource_lease_id),
            source_attempt_id=claim.attempt_id,
            source_worker_id=claim.worker_id,
            task_id=claim.task_id, step_id=claim.step_id,
            action_id=claim.action_id)
        instance_id = str(running["instance_id"])

        pending = broker.terminate(instance_id)
        self.assertEqual(pending["state"], "stopping")
        self.assertEqual(self.workload_row(instance_id)["status"], "active")
        done = broker.terminate(instance_id, force=True)
        self.assertEqual(done["state"], "terminated")
        self.assertEqual(backend.terminate_calls, [False, True])
        self.assertEqual(self.workload_row(instance_id)["status"], "released")

    def test_identity_loss_after_stop_boundary_is_outcome_unknown(self):
        running = self.launch({"label": "ambiguous-stop"})
        instance_id = str(running["instance_id"])
        binding = self.broker.binding_for_instance(instance_id, "terminate")
        terminate_claim, _ = self.terminate_claim(
            instance_id, binding=binding, label="ambiguous-stop")
        original_terminate = self.backend.terminate

        def terminate_then_lose_identity(expected, *, force=False):
            original_terminate(expected, force=force)
            raise ProcessIdentityError()

        with mock.patch.object(
                self.backend, "terminate",
                side_effect=terminate_then_lose_identity):
            with self.assertRaises(ProcessBackendError) as raised:
                self.broker.terminate(
                    instance_id, expected_binding=binding,
                    operation_context=self.operation_context(terminate_claim))

        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(
            raised.exception.code,
            "process_termination_outcome_unknown")
        self.assertEqual(self.backend.terminate_calls, [False])
        self.assertEqual(self.instance_row(instance_id)["state"],
                         "stop_requested")

    def test_launch_projection_failure_after_effect_is_outcome_unknown(self):
        original_transition = self.broker._transition

        def fail_running_projection(instance_id, state, **kwargs):
            if state == "running":
                raise RuntimeError("injected post-launch projection failure")
            return original_transition(instance_id, state, **kwargs)

        with mock.patch.object(
                self.broker, "_transition",
                side_effect=fail_running_projection):
            with self.assertRaises(ProcessBackendError) as raised:
                self.launch({"label": "projection-failure"})

        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(
            raised.exception.code, "process_launch_outcome_unknown")
        self.assertEqual(self.backend.launch_count, 1)
        with self.graph._connect() as conn:
            row = conn.execute(
                """SELECT state FROM process_instances
                   ORDER BY created_at DESC LIMIT 1""").fetchone()
        self.assertEqual(row["state"], "starting")

    def test_launch_admission_failure_after_effect_is_outcome_unknown(self):
        private_error = "private heartbeat database detail"
        with mock.patch.object(
                self.admission, "heartbeat_workload_in_transaction",
                side_effect=RuntimeError(private_error)):
            with self.assertRaises(ProcessBackendError) as raised:
                self.launch({"label": "admission-projection-failure"})

        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(
            raised.exception.code, "process_launch_outcome_unknown")
        self.assertNotIn(private_error, str(raised.exception))
        self.assertEqual(self.backend.launch_count, 1)
        with self.graph._connect() as conn:
            process = conn.execute(
                """SELECT instance_id,state FROM process_instances
                   ORDER BY created_at DESC LIMIT 1""").fetchone()
            workload = conn.execute(
                """SELECT status FROM workload_resource_leases
                   WHERE instance_id=?""", (process["instance_id"],)).fetchone()
        self.assertEqual(process["state"], "starting")
        self.assertEqual(workload["status"], "active")

    def test_known_launch_rejection_survives_projection_failure(self):
        known = ProcessBackendError(
            "backend_launch_rejected", outcome_unknown=False)
        with (mock.patch.object(
                self.backend, "launch", side_effect=known),
              mock.patch.object(
                self.admission, "release_workload_in_transaction",
                side_effect=RuntimeError("injected release projection failure"))):
            with self.assertRaises(ProcessBackendError) as raised:
                self.launch({"label": "known-launch-rejection"})

        self.assertFalse(raised.exception.outcome_unknown)
        self.assertEqual(raised.exception.code, "backend_launch_rejected")
        self.assertEqual(self.backend.launch_count, 0)

    def test_positive_launch_observation_overrides_known_failure_claim(self):
        original_launch = self.backend.launch

        def launch_then_claim_rejection(request):
            original_launch(request)
            raise ProcessBackendError(
                "backend_launch_rejected", outcome_unknown=False)

        with (mock.patch.object(
                self.backend, "launch", side_effect=launch_then_claim_rejection),
              mock.patch.object(
                self.admission, "heartbeat_workload_in_transaction",
                side_effect=RuntimeError("injected live projection failure"))):
            with self.assertRaises(ProcessBackendError) as raised:
                self.launch({"label": "contradictory-known-rejection"})

        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(
            raised.exception.code, "process_launch_outcome_unknown")
        self.assertEqual(self.backend.launch_count, 1)

    def test_launch_replay_projection_failure_is_outcome_unknown(self):
        values = {"label": "replay-projection-failure"}
        claim = self.claim(values)
        running = self.launch(values, claim=claim)

        with mock.patch.object(
                self.admission, "heartbeat_workload_in_transaction",
                side_effect=RuntimeError("injected replay heartbeat failure")):
            with self.assertRaises(ProcessBackendError) as raised:
                self.launch(values, claim=claim)

        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(
            raised.exception.code, "process_launch_outcome_unknown")
        self.assertEqual(self.backend.launch_count, 1)
        self.assertEqual(
            self.instance_row(str(running["instance_id"]))["state"], "running")

    def test_termination_projection_failure_after_signal_is_outcome_unknown(self):
        running = self.launch({"label": "stop-projection-failure"})
        instance_id = str(running["instance_id"])
        binding = self.broker.binding_for_instance(instance_id, "terminate")
        terminate_claim, _ = self.terminate_claim(
            instance_id, binding=binding, label="projection-failure")
        original_transition = self.broker._transition

        def fail_terminal_projection(selected, state, **kwargs):
            if state == "terminated":
                raise RuntimeError("injected post-signal projection failure")
            return original_transition(selected, state, **kwargs)

        with mock.patch.object(
                self.broker, "_transition",
                side_effect=fail_terminal_projection):
            with self.assertRaises(ProcessBackendError) as raised:
                self.broker.terminate(
                    instance_id, expected_binding=binding,
                    operation_context=self.operation_context(terminate_claim))

        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(
            raised.exception.code, "process_termination_postcondition_unknown")
        self.assertEqual(self.backend.terminate_calls, [False])
        self.assertEqual(
            self.instance_row(instance_id)["state"], "stop_requested")

    def test_termination_release_failure_after_signal_is_outcome_unknown(self):
        running = self.launch({"label": "stop-release-failure"})
        instance_id = str(running["instance_id"])
        binding = self.broker.binding_for_instance(instance_id, "terminate")
        terminate_claim, _ = self.terminate_claim(
            instance_id, binding=binding, label="release-failure")
        private_error = "private workload release database detail"

        with mock.patch.object(
                self.admission, "release_workload_in_transaction",
                side_effect=RuntimeError(private_error)):
            with self.assertRaises(ProcessBackendError) as raised:
                self.broker.terminate(
                    instance_id, expected_binding=binding,
                    operation_context=self.operation_context(terminate_claim))

        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(
            raised.exception.code, "process_termination_postcondition_unknown")
        self.assertNotIn(private_error, str(raised.exception))
        self.assertEqual(self.backend.terminate_calls, [False])
        self.assertEqual(
            self.instance_row(instance_id)["state"], "stop_requested")
        self.assertEqual(self.workload_row(instance_id)["status"], "active")

    def test_known_termination_rejection_is_not_upgraded_to_unknown(self):
        running = self.launch({"label": "known-stop-rejection"})
        instance_id = str(running["instance_id"])
        binding = self.broker.binding_for_instance(instance_id, "terminate")
        terminate_claim, _ = self.terminate_claim(
            instance_id, binding=binding, label="known-rejection")
        known = ProcessBackendError(
            "backend_stop_rejected", outcome_unknown=False)

        with mock.patch.object(self.backend, "terminate", side_effect=known):
            with self.assertRaises(ProcessBackendError) as raised:
                self.broker.terminate(
                    instance_id, expected_binding=binding,
                    operation_context=self.operation_context(terminate_claim))

        self.assertFalse(raised.exception.outcome_unknown)
        self.assertEqual(raised.exception.code, "backend_stop_rejected")
        self.assertEqual(self.backend.terminate_calls, [])
        self.assertEqual(
            self.instance_row(instance_id)["state"], "stop_requested")

    def test_known_no_effect_rejection_is_authoritative_failure_evidence(self):
        running = self.launch({"label": "known-rejection-reconcile"})
        instance_id = str(running["instance_id"])
        claim, binding = self.terminate_claim(
            instance_id, label="known-rejection-reconcile")
        rejection = ProcessBackendError(
            "backend_stop_rejected", outcome_unknown=False)
        with mock.patch.object(
                self.backend, "terminate", side_effect=rejection):
            with self.assertRaises(ProcessBackendError):
                self.terminate_exact(
                    self.broker, instance_id, claim, binding)
        self.assertEqual(
            self.operation_row(claim.idempotency_key)["status"], "known_failed")
        self.tasks.mark_step_outcome_unknown(
            claim, reason_code="worker_crashed_before_recording_known_failure")
        candidate = self.tasks.reconciliation_candidate(claim.step_id)

        failure = self.broker.reconciliation_receipt(
            candidate.tool_name, candidate.executor_binding,
            candidate.args, candidate.idempotency_key,
            task_id=candidate.task_id, step_id=candidate.step_id,
            action_id=candidate.action_id, attempt_id=candidate.attempt_id)

        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["operation"], "terminate")
        self.assertEqual(failure["result_code"], "backend_stop_rejected")
        self.assertEqual(self.backend.terminate_calls, [])

    def test_task_termination_requires_exact_live_claim_before_dispatch(self):
        running = self.launch({"label": "exact-claim-gate"})
        instance_id = str(running["instance_id"])
        claim, binding = self.terminate_claim(
            instance_id, label="exact-claim-gate")

        with self.assertRaises(ProcessBindingError):
            self.broker.terminate(
                instance_id, expected_binding=binding)
        forged = self.operation_context(claim).model_copy(
            update={"worker_id": "forged-worker-id"})
        with self.assertRaises(ProcessBindingError):
            self.broker.terminate(
                instance_id, expected_binding=binding,
                operation_context=forged)

        self.assertEqual(self.backend.terminate_calls, [])
        self.assertEqual(self.graph.count("process_operations"), 0)
        receipt = self.terminate_exact(
            self.broker, instance_id, claim, binding)
        self.assertEqual(receipt["state"], "terminated")
        self.assertEqual(self.backend.terminate_calls, [False])

    def test_task_termination_requires_exact_durable_approval(self):
        running = self.launch({"label": "exact-approval-gate"})
        instance_id = str(running["instance_id"])
        binding = self.broker.binding_for_instance(instance_id, "terminate")

        for suffix, approval_status in (
                ("not-required", "not_required"),
                ("unbacked-approved", "approved")):
            with self.subTest(approval_status=approval_status):
                task_id, _ = self.tasks.create(
                    f"Reject {suffix} process termination",
                    {"terminated": True})
                batch_id, _ = self.tasks.stage_step_batch(task_id, [{
                    "tool_call_id": f"terminate-{suffix}",
                    "tool_name": "machine_terminate_process",
                    "args": {"instance_id": instance_id},
                    "risk": "high", "approval_status": approval_status,
                    "idempotency_class": "reconcilable",
                    "recovery_policy": "reconcile",
                    "executor_binding": binding.model_dump(mode="json"),
                    "resource_claims": {},
                }], round_index=0)
                claim = self.tasks.claim_next_step(
                    batch_id, f"approval-gate-{suffix}", lease_seconds=60)
                with self.assertRaises(ProcessBindingError):
                    self.broker.terminate(
                        instance_id, expected_binding=binding,
                        operation_context=self.operation_context(claim))
                with self.graph._connect() as conn:
                    operation_count = int(conn.execute(
                        "SELECT COUNT(*) FROM process_operations "
                        "WHERE idempotency_key=?",
                        (claim.idempotency_key,)).fetchone()[0])
                self.assertEqual(operation_count, 0)
                self.assertEqual(self.backend.terminate_calls, [])

    def test_unknown_dispatch_plus_natural_exit_never_settles_action(self):
        running = self.launch({"label": "unknown-natural-exit"})
        instance_id = str(running["instance_id"])
        claim, binding = self.terminate_claim(
            instance_id, label="unknown-natural-exit")
        ambiguous = ProcessBackendError(
            "systemd_command_unavailable", outcome_unknown=True)

        with mock.patch.object(
                self.backend, "terminate", side_effect=ambiguous):
            with self.assertRaises(ProcessBackendError) as raised:
                self.terminate_exact(
                    self.broker, instance_id, claim, binding)
        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(
            self.operation_row(claim.idempotency_key)["status"],
            "outcome_unknown")
        self.assertEqual(self.backend.terminate_calls, [])

        self.backend.natural_exit(instance_id)
        natural = self.broker.inspect(instance_id)
        self.assertEqual(natural["state"], "exited")
        self.assertFalse(self.broker.verify_receipt(
            "machine_terminate_process", natural,
            {"instance_id": instance_id}, claim.idempotency_key))
        self.tasks.mark_step_outcome_unknown(
            claim, reason_code="process_termination_outcome_unknown")
        candidate = self.tasks.reconciliation_candidate(claim.step_id)
        self.assertIsNone(self.broker.reconciliation_receipt(
            candidate.tool_name, candidate.executor_binding,
            candidate.args, candidate.idempotency_key,
            task_id=candidate.task_id, step_id=candidate.step_id,
            action_id=candidate.action_id, attempt_id=candidate.attempt_id))

    def test_crash_after_dispatch_journal_never_reenters_backend(self):
        running = self.launch({"label": "dispatch-crash"})
        instance_id = str(running["instance_id"])
        claim, binding = self.terminate_claim(
            instance_id, label="dispatch-crash")
        original_transition = self.broker._transition

        def crash_before_backend(selected, state, **kwargs):
            if state == "stop_requested":
                raise KeyboardInterrupt("simulated process crash")
            return original_transition(selected, state, **kwargs)

        with mock.patch.object(
                self.broker, "_transition", side_effect=crash_before_backend):
            with self.assertRaises(KeyboardInterrupt):
                self.terminate_exact(
                    self.broker, instance_id, claim, binding)

        self.assertEqual(
            self.operation_row(claim.idempotency_key)["status"], "dispatching")
        self.assertEqual(self.backend.terminate_calls, [])
        with self.assertRaises(ProcessBackendError) as replay:
            self.terminate_exact(
                self.broker, instance_id, claim, binding)
        self.assertTrue(replay.exception.outcome_unknown)
        self.assertEqual(self.backend.terminate_calls, [])

    def test_prepared_crash_can_resume_same_live_claim_once(self):
        running = self.launch({"label": "prepared-resume"})
        instance_id = str(running["instance_id"])
        claim, binding = self.terminate_claim(
            instance_id, label="prepared-resume")
        real_dispatch = self.broker._mark_termination_dispatching

        with mock.patch.object(
                self.broker, "_mark_termination_dispatching",
                side_effect=KeyboardInterrupt("simulated pre-dispatch crash")):
            with self.assertRaises(KeyboardInterrupt):
                self.terminate_exact(
                    self.broker, instance_id, claim, binding)
        self.assertEqual(
            self.operation_row(claim.idempotency_key)["status"], "prepared")
        self.assertEqual(self.backend.terminate_calls, [])

        with mock.patch.object(
                self.broker, "_mark_termination_dispatching",
                wraps=real_dispatch):
            receipt = self.terminate_exact(
                self.broker, instance_id, claim, binding)
        self.assertEqual(receipt["state"], "terminated")
        self.assertEqual(self.backend.terminate_calls, [False])

    def test_acknowledgement_survives_projection_crash_and_never_resignals(self):
        running = self.launch({"label": "ack-crash-recovery"})
        instance_id = str(running["instance_id"])
        claim, binding = self.terminate_claim(
            instance_id, label="ack-crash-recovery")
        original_transition = self.broker._transition

        def fail_terminal_projection(selected, state, **kwargs):
            if state == "terminated":
                raise RuntimeError("injected terminal projection crash")
            return original_transition(selected, state, **kwargs)

        with mock.patch.object(
                self.broker, "_transition",
                side_effect=fail_terminal_projection):
            with self.assertRaises(ProcessBackendError):
                self.terminate_exact(
                    self.broker, instance_id, claim, binding)
        self.assertEqual(
            self.operation_row(claim.idempotency_key)["status"],
            "effect_acknowledged")
        calls = list(self.backend.terminate_calls)
        self.tasks.mark_step_outcome_unknown(
            claim, reason_code="worker_crashed_after_effect_acknowledgement")
        candidate = self.tasks.reconciliation_candidate(claim.step_id)

        reconciled = self.broker.reconciliation_receipt(
            candidate.tool_name, candidate.executor_binding,
            candidate.args, candidate.idempotency_key,
            task_id=candidate.task_id, step_id=candidate.step_id,
            action_id=candidate.action_id, attempt_id=candidate.attempt_id)

        self.assertIsNotNone(reconciled)
        self.assertEqual(reconciled["state"], "terminated")
        self.assertEqual(self.backend.terminate_calls, calls)
        self.assertEqual(
            self.operation_row(claim.idempotency_key)["status"], "completed")
        self.assertTrue(self.broker.verify_receipt(
            candidate.tool_name, reconciled,
            candidate.args, candidate.idempotency_key))

    def test_completed_operation_replays_once_and_other_action_cannot_inherit(self):
        running = self.launch({"label": "two-actions-one-target"})
        instance_id = str(running["instance_id"])
        claim_a, binding = self.terminate_claim(
            instance_id, label="termination-a")
        claim_b, _ = self.terminate_claim(
            instance_id, binding=binding, label="termination-b")

        receipt_a = self.terminate_exact(
            self.broker, instance_id, claim_a, binding)
        calls = list(self.backend.terminate_calls)
        replay_a = self.terminate_exact(
            self.broker, instance_id, claim_a, binding)
        self.assertTrue(replay_a["idempotent_replay"])
        self.assertEqual(self.backend.terminate_calls, calls)
        self.assertTrue(self.broker.verify_receipt(
            "machine_terminate_process", receipt_a,
            {"instance_id": instance_id}, claim_a.idempotency_key))
        self.assertFalse(self.broker.verify_receipt(
            "machine_terminate_process", receipt_a,
            {"instance_id": instance_id}, claim_b.idempotency_key))
        with self.assertRaises(ProcessBindingError):
            self.terminate_exact(
                self.broker, instance_id, claim_b, binding)
        self.assertEqual(self.backend.terminate_calls, calls)

    def test_concurrent_exact_termination_signals_once(self):
        running = self.launch({"label": "concurrent-termination"})
        instance_id = str(running["instance_id"])
        claim, binding = self.terminate_claim(
            instance_id, label="concurrent-termination")

        def stop(_index):
            return self.terminate_exact(
                self.broker, instance_id, claim, binding)

        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(stop, range(2)))

        self.assertEqual(self.backend.terminate_calls, [False])
        self.assertEqual(
            sorted(bool(item["idempotent_replay"]) for item in receipts),
            [False, True])
        self.assertTrue(all(item["state"] == "terminated" for item in receipts))

    def test_forged_operation_event_sequence_invalidates_receipt(self):
        running = self.launch({"label": "forged-operation-event"})
        instance_id = str(running["instance_id"])
        claim, binding = self.terminate_claim(
            instance_id, label="forged-operation-event")
        receipt = self.terminate_exact(
            self.broker, instance_id, claim, binding)
        args = {"instance_id": instance_id}
        self.assertTrue(self.broker.verify_receipt(
            "machine_terminate_process", receipt, args,
            claim.idempotency_key))
        operation = self.operation_row(claim.idempotency_key)
        with self.graph.transaction() as conn:
            conn.execute(
                """UPDATE process_operations
                      SET postcondition_event_seq=?
                    WHERE idempotency_key=?""",
                (operation["prepared_event_seq"], claim.idempotency_key),
            )
        self.assertFalse(self.broker.verify_receipt(
            "machine_terminate_process", receipt, args,
            claim.idempotency_key))

    def test_operation_provenance_and_state_cannot_be_rewritten(self):
        running = self.launch({"label": "immutable-operation"})
        instance_id = str(running["instance_id"])
        claim, binding = self.terminate_claim(
            instance_id, label="immutable-operation")
        self.terminate_exact(self.broker, instance_id, claim, binding)

        mutations = (
            ("UPDATE task_steps SET executor_binding_json='{}' WHERE step_id=?",
             claim.step_id),
            ("UPDATE action_receipts SET args_sha256=? WHERE step_id=?",
             "f" * 64, claim.step_id),
            ("UPDATE action_attempts SET worker_id='other-worker' "
             "WHERE attempt_id=?", claim.attempt_id),
            ("UPDATE process_operations SET force=1 WHERE idempotency_key=?",
             claim.idempotency_key),
            ("UPDATE process_operations SET status='dispatching' "
             "WHERE idempotency_key=?", claim.idempotency_key),
        )
        for statement in mutations:
            with self.subTest(sql=statement[0]):
                with self.assertRaises(sqlite3.IntegrityError):
                    with self.graph.transaction() as conn:
                        conn.execute(statement[0], tuple(statement[1:]))

    def test_terminate_receipt_requires_exact_terminal_postcondition(self):
        running = self.launch({"label": "verify-termination"})
        instance_id = str(running["instance_id"])
        binding = self.broker.binding_for_instance(instance_id, "terminate")
        claim, _ = self.terminate_claim(
            instance_id, binding=binding, label="terminate-receipt")
        args = {"instance_id": instance_id}
        current = self.broker.inspect(instance_id)
        self.assertFalse(self.broker.verify_receipt(
            "machine_terminate_process", current, args,
            claim.idempotency_key))

        terminal = self.broker.terminate(
            instance_id, expected_binding=binding,
            operation_context=self.operation_context(claim))
        self.assertEqual(terminal["state"], "terminated")
        self.assertTrue(self.broker.verify_receipt(
            "machine_terminate_process", terminal, args,
            claim.idempotency_key))
        self.assertFalse(self.broker.verify_receipt(
            "machine_terminate_process", terminal, args, "act_forged"))

    def test_stopping_process_does_not_verify_as_terminated(self):
        backend = FakeProcessBackend(resist_term=True)
        broker = ProcessBroker(
            self.graph, self.registry, backend, self.admission,
            state_root=self.root / "stopping-verifier-state")
        launch_claim = self.claim({"label": "stopping-verifier"})
        launched = broker.launch(
            self.spec.spec_id, {"label": "stopping-verifier"},
            launch_idempotency_key=launch_claim.idempotency_key,
            source_step_lease_id=str(launch_claim.resource_lease_id),
            source_attempt_id=launch_claim.attempt_id,
            source_worker_id=launch_claim.worker_id,
            task_id=launch_claim.task_id, step_id=launch_claim.step_id,
            action_id=launch_claim.action_id)
        instance_id = str(launched["instance_id"])
        binding = broker.binding_for_instance(instance_id, "terminate")
        terminate_claim, _ = self.terminate_claim(
            instance_id, broker=broker, binding=binding,
            label="terminate-stopping")
        pending = broker.terminate(
            instance_id, expected_binding=binding,
            operation_context=self.operation_context(terminate_claim))
        self.assertEqual(pending["state"], "stopping")
        self.assertFalse(broker.verify_receipt(
            "machine_terminate_process", pending,
            {"instance_id": instance_id}, terminate_claim.idempotency_key))

    def test_process_reconciliation_probe_never_repeats_effect(self):
        launch_values = {"label": "reconcile-launch"}
        launch_claim = self.claim(launch_values)
        launched = self.launch(launch_values, claim=launch_claim)
        launch_count = self.backend.launch_count
        self.tasks.mark_step_outcome_unknown(
            launch_claim, reason_code="process_launch_outcome_unknown")
        launch_candidate = self.tasks.reconciliation_candidate(
            launch_claim.step_id)
        launch_receipt = self.broker.reconciliation_receipt(
            launch_candidate.tool_name, launch_candidate.executor_binding,
            launch_candidate.args, launch_candidate.idempotency_key,
            task_id=launch_candidate.task_id,
            step_id=launch_candidate.step_id,
            action_id=launch_candidate.action_id)
        self.assertIsNotNone(launch_receipt)
        self.assertEqual(self.backend.launch_count, launch_count)
        self.assertTrue(self.broker.verify_receipt(
            launch_candidate.tool_name, launch_receipt,
            launch_candidate.args, launch_candidate.idempotency_key))

        instance_id = str(launched["instance_id"])
        terminate_binding = self.broker.binding_for_instance(
            instance_id, "terminate")
        terminate_claim, _ = self.terminate_claim(
            instance_id, binding=terminate_binding,
            label="terminate-reconcile")
        self.broker.terminate(
            instance_id, expected_binding=terminate_binding,
            operation_context=self.operation_context(terminate_claim))
        terminate_calls = list(self.backend.terminate_calls)
        self.tasks.mark_step_outcome_unknown(
            terminate_claim, reason_code="process_terminate_outcome_unknown")
        terminate_candidate = self.tasks.reconciliation_candidate(
            terminate_claim.step_id)
        terminate_receipt = self.broker.reconciliation_receipt(
            terminate_candidate.tool_name,
            terminate_candidate.executor_binding,
            terminate_candidate.args,
            terminate_candidate.idempotency_key,
            task_id=terminate_candidate.task_id,
            step_id=terminate_candidate.step_id,
            action_id=terminate_candidate.action_id,
            attempt_id=terminate_candidate.attempt_id)
        self.assertIsNotNone(terminate_receipt)
        self.assertEqual(self.backend.terminate_calls, terminate_calls)
        self.assertTrue(self.broker.verify_receipt(
            terminate_candidate.tool_name, terminate_receipt,
            terminate_candidate.args,
            terminate_candidate.idempotency_key))

    def test_known_launch_failure_is_authoritative_reconciliation_evidence(self):
        values = {"label": "known-launch-failure"}
        claim = self.claim(values)
        with mock.patch.object(
                self.backend, "launch",
                side_effect=ProcessBackendError(
                    "backend_executable_identity_changed",
                    outcome_unknown=False)):
            failed = self.launch(values, claim=claim)
        self.assertEqual(failed["state"], "launch_failed")
        self.tasks.mark_step_outcome_unknown(
            claim, reason_code="worker_crashed_before_finish")
        candidate = self.tasks.reconciliation_candidate(claim.step_id)

        receipt = self.broker.reconciliation_receipt(
            candidate.tool_name, candidate.executor_binding,
            candidate.args, candidate.idempotency_key,
            task_id=candidate.task_id, step_id=candidate.step_id,
            action_id=candidate.action_id)

        self.assertIsNotNone(receipt)
        self.assertEqual(receipt["state"], "launch_failed")
        self.assertFalse(receipt["verified"])
        self.assertEqual(self.backend.launch_count, 0)

    def test_fast_exit_and_restart_natural_exit_preserve_fences(self):
        fast_backend = FakeProcessBackend(fast_exit=True)
        fast_broker = ProcessBroker(
            self.graph, self.registry, fast_backend, self.admission,
            state_root=self.root / "fast-state")
        claim = self.claim({"label": "fast"})
        fast = fast_broker.launch(
            self.spec.spec_id, {"label": "fast"},
            launch_idempotency_key=claim.idempotency_key,
            source_step_lease_id=str(claim.resource_lease_id),
            source_attempt_id=claim.attempt_id,
            source_worker_id=claim.worker_id,
            task_id=claim.task_id, step_id=claim.step_id,
            action_id=claim.action_id)
        fast_row = self.instance_row(str(fast["instance_id"]))
        self.assertEqual(fast_row["state"], "exited")
        for column in ("boot_id", "leader_pid", "start_ticks", "exe_device",
                       "exe_inode", "exe_sha256"):
            self.assertIsNone(fast_row[column])

        running = self.launch({"label": "natural"})
        instance_id = str(running["instance_id"])
        captured = self.instance_row(instance_id)
        self.backend.natural_exit(instance_id)
        with self.graph.transaction() as conn:
            conn.execute(
                """UPDATE workload_resource_leases
                   SET expires_at='2000-01-01T00:00:00.000000Z',
                       heartbeat_at='2000-01-01T00:00:00.000000Z'
                   WHERE instance_id=?""", (instance_id,))
        restarted_admission = ResourceAdmissionController(
            self.graph,
            AdmissionBudget(
                cpu_millis=8_000, ram_mib=16_384,
                concurrency_slots=16, network_slots=16,
                accelerator_vram_mib={}),
            snapshot_provider=lambda: ResourceSnapshot(
                available_cpu_millis=8_000,
                available_ram_mib=16_384,
                available_network_slots=16,
                available_accelerator_vram_mib={},
                captured_at=datetime.now(UTC)),
            snapshot_ttl_seconds=0,
            runtime_id="runtime-process-restart",
            profile_fingerprint="a" * 64,
        )
        restarted = ProcessBroker(
            self.graph, self.registry, self.backend, restarted_admission,
            state_root=self.root / "state")
        receipts = restarted.reconcile_active()
        terminal = next(item for item in receipts
                        if item["instance_id"] == instance_id)
        self.assertEqual(terminal["state"], "exited")
        after = self.instance_row(instance_id)
        for column in ("boot_id", "leader_pid", "start_ticks", "exe_device",
                       "exe_inode", "exe_sha256"):
            self.assertEqual(after[column], captured[column])
        self.assertEqual(self.workload_row(instance_id)["status"], "released")

    def test_terminal_unit_cleanup_is_durable_private_and_idempotent(self):
        running = self.launch({"label": "cleanup"})
        instance_id = str(running["instance_id"])
        before = self.naturally_exit(instance_id)

        self.assertEqual(self.cleanup_row(instance_id)["state"], "pending")
        first = self.broker.cleanup_retained()
        after = self.broker.inspect(instance_id)
        repeated = self.broker.cleanup_retained()

        self.assertEqual(first["completed_last_pass"], 1)
        self.assertNotIn("complete", first)
        self.assertEqual(repeated["attempted"], 0)
        self.assertEqual(self.backend.retire_calls, [
            ProcessBroker._unit_name(instance_id)])
        self.assertEqual(before, after)
        with self.graph._connect() as conn:
            payloads = [json.loads(row[0]) for row in conn.execute(
                """SELECT payload_json FROM graph_events
                    WHERE event_type LIKE 'process.cleanup_%'""")]
        self.assertTrue(payloads)
        encoded = json.dumps(payloads, sort_keys=True)
        for private in (
                "unit_name", "control_group", "invocation_id", "leader_pid",
                "identity_token", "argv", "environment"):
            self.assertNotIn(private, encoded)

    def test_cleanup_recovers_crashes_before_and_after_backend_effect(self):
        before_running = self.launch({"label": "cleanup-before"})
        before_id = str(before_running["instance_id"])
        self.naturally_exit(before_id)
        with mock.patch.object(
                self.backend, "retire_terminal",
                side_effect=ProcessBackendError("cleanup_transport_lost")):
            first = self.broker.cleanup_retained()
        self.assertEqual(first["pending"], 1)
        self.assertIn(ProcessBroker._unit_name(before_id),
                      self.backend.observations)
        with self.graph.transaction() as conn:
            conn.execute(
                """UPDATE process_unit_cleanups
                      SET next_attempt_at='2000-01-01T00:00:00Z'
                    WHERE instance_id=?""", (before_id,))
        retried = self.broker.cleanup_retained()
        self.assertEqual(retried["completed_last_pass"], 1)

        after_running = self.launch({"label": "cleanup-after"})
        after_id = str(after_running["instance_id"])
        self.naturally_exit(after_id)
        real_retire = self.backend.retire_terminal

        def retire_then_lose_ack(fence: BackendTerminalFence) -> None:
            real_retire(fence)
            raise ProcessBackendError("cleanup_ack_lost")

        with mock.patch.object(
                self.backend, "retire_terminal",
                side_effect=retire_then_lose_ack):
            uncertain = self.broker.cleanup_retained()
        self.assertEqual(uncertain["pending"], 1)
        self.assertNotIn(ProcessBroker._unit_name(after_id),
                         self.backend.observations)
        with self.graph.transaction() as conn:
            conn.execute(
                """UPDATE process_unit_cleanups
                      SET next_attempt_at='2000-01-01T00:00:00Z'
                    WHERE instance_id=?""", (after_id,))
        recovered = self.broker.cleanup_retained()
        self.assertEqual(recovered["completed_last_pass"], 1)
        self.assertEqual(self.cleanup_row(after_id)["state"], "complete")

    def test_retryable_cleanup_failure_stays_pending_with_capped_backoff(self):
        running = self.launch({"label": "cleanup-retry-forever"})
        instance_id = str(running["instance_id"])
        self.naturally_exit(instance_id)
        with self.graph.transaction() as conn:
            conn.execute(
                """UPDATE process_unit_cleanups
                      SET attempt_count=999,
                          next_attempt_at='2000-01-01T00:00:00Z'
                    WHERE instance_id=?""", (instance_id,))
        with mock.patch.object(
                self.backend, "retire_terminal",
                side_effect=ProcessBackendError("user_bus_unavailable")):
            result = self.broker.cleanup_retained()

        cleanup = self.cleanup_row(instance_id)
        self.assertEqual(result["pending"], 1)
        self.assertEqual(result["blocked"], 0)
        self.assertEqual(result["retrying"], 1)
        self.assertEqual(cleanup["state"], "pending")
        self.assertEqual(cleanup["attempt_count"], 1000)
        retry_at = datetime.fromisoformat(
            str(cleanup["next_attempt_at"]).replace("Z", "+00:00"))
        last_attempt = datetime.fromisoformat(
            str(cleanup["last_attempt_at"]).replace("Z", "+00:00"))
        self.assertGreaterEqual(
            (retry_at - last_attempt).total_seconds(), 256)
        self.assertLess(
            (retry_at - last_attempt).total_seconds(), 257)

    def test_expired_cleanup_claim_is_reclaimed_after_crash(self):
        running = self.launch({"label": "cleanup-claim-crash"})
        instance_id = str(running["instance_id"])
        self.naturally_exit(instance_id)
        claimed = self.broker._claim_terminal_cleanup(instance_id)
        self.assertIsNotNone(claimed)
        self.assertEqual(self.cleanup_row(instance_id)["state"], "cleaning")
        with self.graph.transaction() as conn:
            conn.execute(
                """UPDATE process_unit_cleanups
                      SET claim_expires_at='2000-01-01T00:00:00Z'
                    WHERE instance_id=?""", (instance_id,))

        recovered = self.broker.cleanup_retained()

        self.assertEqual(recovered["completed_last_pass"], 1)
        self.assertEqual(self.cleanup_row(instance_id)["attempt_count"], 2)

    def test_expired_claim_recovers_crash_after_backend_retirement(self):
        running = self.launch({"label": "cleanup-post-effect-crash"})
        instance_id = str(running["instance_id"])
        self.naturally_exit(instance_id)
        claimed = self.broker._claim_terminal_cleanup(instance_id)
        self.assertIsNotNone(claimed)
        row, _claim_token, _attempt = claimed

        self.backend.retire_terminal(self.broker._cleanup_fence(row))

        self.assertEqual(self.cleanup_row(instance_id)["state"], "cleaning")
        self.assertNotIn(ProcessBroker._unit_name(instance_id),
                         self.backend.observations)
        with self.graph.transaction() as conn:
            conn.execute(
                """UPDATE process_unit_cleanups
                      SET claim_expires_at='2000-01-01T00:00:00Z'
                    WHERE instance_id=?""", (instance_id,))
        recovered = self.broker.cleanup_retained()
        self.assertEqual(recovered["completed_last_pass"], 1)
        self.assertEqual(self.cleanup_row(instance_id)["state"], "complete")

    def test_stale_cleanup_claim_cannot_settle_after_reclaim(self):
        running = self.launch({"label": "cleanup-stale-claim"})
        instance_id = str(running["instance_id"])
        self.naturally_exit(instance_id)
        first = self.broker._claim_terminal_cleanup(instance_id)
        self.assertIsNotNone(first)
        _row, first_token, first_attempt = first
        with self.graph.transaction() as conn:
            conn.execute(
                """UPDATE process_unit_cleanups
                      SET claim_expires_at='2000-01-01T00:00:00Z'
                    WHERE instance_id=?""", (instance_id,))
        second = self.broker._claim_terminal_cleanup(instance_id)
        self.assertIsNotNone(second)
        second_row, second_token, second_attempt = second

        with self.assertRaises(ProcessBackendError) as raised:
            self.broker._settle_terminal_cleanup(
                instance_id, first_token, first_attempt, outcome="complete")
        self.assertEqual(
            raised.exception.code, "process_unit_cleanup_claim_lost")
        self.backend.retire_terminal(self.broker._cleanup_fence(second_row))
        settled = self.broker._settle_terminal_cleanup(
            instance_id, second_token, second_attempt, outcome="complete")
        self.assertEqual(settled, "complete")

    def test_cleanup_covers_terminated_launch_failed_and_fast_exit(self):
        terminated = self.launch({"label": "cleanup-terminated"})
        terminated_id = str(terminated["instance_id"])
        binding = self.broker.binding_for_instance(
            terminated_id, "terminate")
        terminate_claim, _ = self.terminate_claim(
            terminated_id, binding=binding, label="cleanup-terminated")
        stopped = self.broker.terminate(
            terminated_id, expected_binding=binding,
            operation_context=self.operation_context(terminate_claim))
        self.assertEqual(stopped["state"], "terminated")
        self.assertEqual(
            self.broker.cleanup_retained()["completed_last_pass"], 1)

        failed_claim = self.claim({"label": "cleanup-launch-failed"})
        with mock.patch.object(
                self.backend, "launch",
                side_effect=ProcessBackendError(
                    "backend_executable_identity_changed",
                    outcome_unknown=False)):
            failed = self.launch(
                {"label": "cleanup-launch-failed"}, claim=failed_claim)
        self.assertEqual(failed["state"], "launch_failed")
        self.assertEqual(
            self.broker.cleanup_retained()["completed_last_pass"], 1)

        fast_backend = FakeProcessBackend(fast_exit=True)
        fast_broker = ProcessBroker(
            self.graph, self.registry, fast_backend, self.admission,
            state_root=self.root / "fast-cleanup-state")
        fast_claim = self.claim({"label": "cleanup-fast-exit"})
        fast = fast_broker.launch(
            self.spec.spec_id, {"label": "cleanup-fast-exit"},
            launch_idempotency_key=fast_claim.idempotency_key,
            source_step_lease_id=str(fast_claim.resource_lease_id),
            source_attempt_id=fast_claim.attempt_id,
            source_worker_id=fast_claim.worker_id,
            task_id=fast_claim.task_id, step_id=fast_claim.step_id,
            action_id=fast_claim.action_id)
        self.assertEqual(fast["state"], "exited")
        self.assertEqual(
            fast_broker.cleanup_retained()["completed_last_pass"], 1)

    def test_concurrent_cleaners_claim_one_backend_effect(self):
        running = self.launch({"label": "cleanup-concurrent"})
        instance_id = str(running["instance_id"])
        self.naturally_exit(instance_id)
        entered = threading.Event()
        release = threading.Event()
        real_retire = self.backend.retire_terminal

        def blocking_retire(fence: BackendTerminalFence) -> None:
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            real_retire(fence)

        other = ProcessBroker(
            self.graph, self.registry, self.backend, self.admission,
            state_root=self.root / "independent-cleaner-state")
        with (mock.patch.object(
                self.backend, "retire_terminal", side_effect=blocking_retire),
              ThreadPoolExecutor(max_workers=1) as pool):
            future = pool.submit(self.broker.cleanup_retained)
            self.assertTrue(entered.wait(timeout=5))
            competing = other.cleanup_retained()
            release.set()
            winner = future.result(timeout=5)

        self.assertEqual(competing["attempted"], 0)
        self.assertEqual(winner["completed_last_pass"], 1)
        self.assertEqual(self.backend.retire_calls.count(
            ProcessBroker._unit_name(instance_id)), 1)

    def test_cleanup_identity_or_membership_mismatch_blocks_without_stop(self):
        identity_running = self.launch({"label": "cleanup-foreign"})
        identity_id = str(identity_running["instance_id"])
        self.naturally_exit(identity_id)
        self.backend.replace_identity(identity_id)

        identity_result = self.broker.cleanup_retained()

        self.assertEqual(identity_result["blocked_last_pass"], 1)
        self.assertEqual(self.cleanup_row(identity_id)["state"], "blocked")
        self.assertNotIn(ProcessBroker._unit_name(identity_id),
                         self.backend.retire_calls)

        members_running = self.launch({"label": "cleanup-members"})
        members_id = str(members_running["instance_id"])
        self.naturally_exit(members_id)
        unit_name = ProcessBroker._unit_name(members_id)
        with self.backend._lock:
            current = self.backend.observations[unit_name]
            self.backend.observations[unit_name] = current.model_copy(
                update={"cgroup_empty": False})

        members_result = self.broker.cleanup_retained()

        self.assertEqual(members_result["blocked_last_pass"], 1)
        self.assertEqual(self.cleanup_row(members_id)["state"], "blocked")
        self.assertNotIn(unit_name, self.backend.retire_calls)

    def test_cleanup_requires_released_workload_before_backend_effect(self):
        running = self.launch({"label": "cleanup-reservation"})
        instance_id = str(running["instance_id"])
        self.naturally_exit(instance_id)
        with self.graph.transaction() as conn:
            conn.execute(
                """UPDATE workload_resource_leases
                      SET status='active',released_at=NULL,release_reason=NULL
                    WHERE instance_id=?""", (instance_id,))

        result = self.broker.cleanup_retained()

        cleanup = self.cleanup_row(instance_id)
        self.assertEqual(result["blocked"], 1)
        self.assertEqual(cleanup["state"], "blocked")
        self.assertEqual(
            cleanup["last_error_code"],
            "process_unit_cleanup_workload_not_released")
        self.assertNotIn(ProcessBroker._unit_name(instance_id),
                         self.backend.retire_calls)

    def test_cleanup_rechecks_workload_release_after_backend_effect(self):
        running = self.launch({"label": "cleanup-release-race"})
        instance_id = str(running["instance_id"])
        self.naturally_exit(instance_id)
        real_retire = self.backend.retire_terminal

        def retire_then_corrupt_release(fence: BackendTerminalFence) -> None:
            real_retire(fence)
            with self.graph.transaction() as conn:
                conn.execute(
                    """UPDATE workload_resource_leases
                          SET status='active',released_at=NULL,
                              release_reason=NULL
                        WHERE instance_id=?""", (instance_id,))

        with mock.patch.object(
                self.backend, "retire_terminal",
                side_effect=retire_then_corrupt_release):
            result = self.broker.cleanup_retained()

        cleanup = self.cleanup_row(instance_id)
        self.assertEqual(result["blocked"], 1)
        self.assertEqual(cleanup["state"], "blocked")
        self.assertEqual(
            cleanup["last_error_code"],
            "process_unit_cleanup_workload_not_released")

    def test_running_launch_receipt_survives_exit_before_verification(self):
        values = {"label": "verify-race"}
        claim = self.claim(values)
        running = self.launch(values, claim=claim)
        instance_id = str(running["instance_id"])
        self.backend.natural_exit(instance_id)

        args = {"spec_id": self.spec.spec_id,
                "parameter_values": values}
        self.assertTrue(self.broker.verify_receipt(
            "machine_launch_process", running, args,
            claim.idempotency_key))
        self.assertFalse(self.broker.verify_receipt(
            "machine_launch_process",
            dict(running, result_code="forged"), args,
            claim.idempotency_key))
        self.assertEqual(self.instance_row(instance_id)["state"], "exited")

    def test_durable_launch_observation_survives_verifier_transport_loss(self):
        values = {"label": "verify-transport-loss"}
        claim = self.claim(values)
        running = self.launch(values, claim=claim)
        args = {"spec_id": self.spec.spec_id,
                "parameter_values": values}

        with mock.patch.object(
                self.backend, "inspect",
                side_effect=ProcessBackendError("systemd_inspection_failed")):
            self.assertTrue(self.broker.verify_receipt(
                "machine_launch_process", running, args,
                claim.idempotency_key))

    def test_operation_binding_fails_closed_after_state_drift(self):
        backend = FakeProcessBackend(resist_term=True)
        broker = ProcessBroker(
            self.graph, self.registry, backend, self.admission,
            state_root=self.root / "binding-state")
        claim = self.claim({"label": "binding"})
        receipt = broker.launch(
            self.spec.spec_id, {"label": "binding"},
            launch_idempotency_key=claim.idempotency_key,
            source_step_lease_id=str(claim.resource_lease_id),
            source_attempt_id=claim.attempt_id,
            source_worker_id=claim.worker_id,
            task_id=claim.task_id, step_id=claim.step_id,
            action_id=claim.action_id)
        instance_id = str(receipt["instance_id"])
        binding = broker.binding_for_instance(instance_id, "terminate")
        encoded = binding.model_dump(mode="json")
        self.assertFalse(set(encoded) & {
            "pid", "leader_pid", "unit", "unit_name", "control_group"})
        self.assertTrue(broker.verify_instance_binding(binding))

        terminate_claim, _ = self.terminate_claim(
            instance_id, broker=broker, binding=binding,
            label="binding-drift")
        first = broker.terminate(
            instance_id, expected_binding=binding,
            operation_context=self.operation_context(terminate_claim))
        self.assertEqual(first["state"], "stopping")
        calls_before = list(backend.terminate_calls)
        self.assertFalse(broker.verify_instance_binding(binding))
        replay = broker.terminate(
            instance_id, expected_binding=binding,
            operation_context=self.operation_context(terminate_claim))
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(backend.terminate_calls, calls_before)

    def test_receipts_and_spec_listing_are_private_and_independently_verified(self):
        literal = "private-looking-value; $(id)"
        claim = self.claim({"label": literal})
        receipt = self.launch({"label": literal}, claim=claim)
        instance_id = str(receipt["instance_id"])
        serialized = json.dumps({
            "receipt": receipt,
            "instances": self.broker.list_instances(),
            "specs": self.broker.list_specs(),
        }, sort_keys=True)
        for private in (literal, str(self.executable), "unit_name",
                        "leader_pid", "control_group", "workload_lease_id"):
            self.assertNotIn(private, serialized)

        args = {"spec_id": self.spec.spec_id,
                "parameter_values": {"label": literal}}
        self.assertTrue(self.broker.verify_receipt(
            "machine_launch_process", receipt, args, claim.idempotency_key))
        forged = dict(receipt, result_code="forged")
        self.assertFalse(self.broker.verify_receipt(
            "machine_launch_process", forged, args, claim.idempotency_key))

        specs = self.broker.list_specs()
        self.assertTrue(self.broker.verify_receipt(
            "machine_list_process_specs", specs, {}, None))
        wrapped = {"status": "ok", "verified": True, "specs": specs}
        self.assertTrue(self.broker.verify_receipt(
            "machine_list_process_specs", wrapped, {}, None))
        self.assertFalse(self.broker.verify_receipt(
            "machine_list_process_specs", specs + [{"forged": True}], {}, None))


class SystemdUserProcessBackendTests(unittest.TestCase):
    def test_loopback_listener_parser_requires_one_exact_same_uid_socket(self):
        header = "sl local_address rem_address st tx tr retr uid timeout inode\n"

        def row(inode: str, *, uid: int = 1000) -> str:
            return (
                "0: 0100007F:2407 00000000:0000 0A "
                "00000000:00000000 00:00000000 00000000 "
                f"{uid} 0 {inode} 1\n")

        def reader(value: str):
            def read(path: Path, *_args, **_kwargs):
                return header + value if path.name == "tcp" else header
            return read

        with (mock.patch.object(
                  Path, "read_text", autospec=True,
                  side_effect=reader(row("555"))),
              mock.patch("friday_core.processes.os.geteuid",
                         return_value=1000)):
            self.assertEqual(
                SystemdUserProcessBackend._loopback_listener_inode(9223),
                "555")

        with (mock.patch.object(
                  Path, "read_text", autospec=True,
                  side_effect=reader(row("555") + row("556"))),
              mock.patch("friday_core.processes.os.geteuid",
                         return_value=1000)):
            self.assertIsNone(
                SystemdUserProcessBackend._loopback_listener_inode(9223))

        with (mock.patch.object(
                  Path, "read_text", autospec=True,
                  side_effect=reader(row("555", uid=2000))),
              mock.patch("friday_core.processes.os.geteuid",
                         return_value=1000),
              self.assertRaises(ProcessIdentityError)):
            SystemdUserProcessBackend._loopback_listener_inode(9223)

    def test_loopback_listener_is_owned_only_by_exact_cgroup_execution(self):
        backend = object.__new__(SystemdUserProcessBackend)
        expected = BackendObservation(
            unit_name="friday-proc-" + "5" * 32 + ".service",
            identity_token="6" * 64, state="running",
            boot_id="boot", invocation_id="browser-invocation",
            control_group="/trusted/browser", leader_pid=1001,
            start_ticks=50_001, exe_device=8, exe_inode=1234,
            exe_sha256="7" * 64, cgroup_empty=False,
            result_code="success")
        descriptor = Path("/proc/2002/fd/9")
        with (mock.patch.object(
                  backend, "inspect", side_effect=[expected, expected]) as inspect,
              mock.patch.object(
                  backend, "_loopback_listener_inode", return_value="555"),
              mock.patch.object(
                  backend, "_cgroup_members", return_value=(2002,)),
              mock.patch.object(Path, "iterdir", return_value=[descriptor]),
              mock.patch("friday_core.processes.os.readlink",
                         return_value="socket:[555]"),
              mock.patch.object(
                  backend, "_pid_in_cgroup", return_value=True)):
            self.assertTrue(backend.owns_loopback_listener(expected, 9223))
        self.assertEqual(inspect.call_count, 2)

        with (mock.patch.object(backend, "inspect", return_value=expected),
              mock.patch.object(
                  backend, "_loopback_listener_inode", return_value="555"),
              mock.patch.object(
                  backend, "_cgroup_members", return_value=(9999,)),
              mock.patch.object(Path, "iterdir", return_value=[descriptor]),
              mock.patch("friday_core.processes.os.readlink",
                         return_value="socket:[foreign]"),
              mock.patch.object(backend, "_pid_in_cgroup") as membership):
            self.assertFalse(backend.owns_loopback_listener(expected, 9223))
        membership.assert_not_called()

        changed = expected.model_copy(update={
            "invocation_id": "replacement-browser"})
        with (mock.patch.object(
                  backend, "inspect", side_effect=[expected, changed]),
              mock.patch.object(
                  backend, "_loopback_listener_inode", return_value="555"),
              mock.patch.object(
                  backend, "_cgroup_members", return_value=(2002,)),
              mock.patch.object(Path, "iterdir", return_value=[descriptor]),
              mock.patch("friday_core.processes.os.readlink",
                         return_value="socket:[555]"),
              mock.patch.object(
                  backend, "_pid_in_cgroup", return_value=True)):
            with self.assertRaises(ProcessIdentityError):
                backend.owns_loopback_listener(expected, 9223)

    def test_member_identity_is_double_fenced_by_cgroup_and_execution(self):
        backend = object.__new__(SystemdUserProcessBackend)
        expected = BackendObservation(
            unit_name="friday-proc-" + "1" * 32 + ".service",
            identity_token="2" * 64, state="running",
            boot_id="boot", invocation_id="invocation",
            control_group="/trusted/cgroup", leader_pid=1001,
            start_ticks=50_001, exe_device=8, exe_inode=1234,
            exe_sha256="3" * 64, cgroup_empty=False,
            result_code="success")
        child_identity = (77_777, 9, 98765, "4" * 64)
        with (mock.patch.object(
                  backend, "inspect", side_effect=[expected, expected]) as inspect,
              mock.patch.object(
                  backend, "_pid_in_cgroup",
                  side_effect=[True, True]) as membership,
              mock.patch.object(
                  backend, "_process_identity",
                  return_value=child_identity)):
            observed = backend.member_identity(expected, 2002)

        self.assertEqual(observed, (2002, *child_identity))
        self.assertEqual(inspect.call_count, 2)
        self.assertEqual(membership.call_count, 2)

        with (mock.patch.object(backend, "inspect", return_value=expected),
              mock.patch.object(
                  backend, "_pid_in_cgroup", return_value=False),
              mock.patch.object(backend, "_process_identity") as identity):
            self.assertIsNone(backend.member_identity(expected, 9999))
        identity.assert_not_called()

        changed = expected.model_copy(update={
            "invocation_id": "replacement-invocation"})
        with (mock.patch.object(
                  backend, "inspect", side_effect=[expected, changed]),
              mock.patch.object(
                  backend, "_pid_in_cgroup", side_effect=[True, True]),
              mock.patch.object(
                  backend, "_process_identity", return_value=child_identity)):
            with self.assertRaises(ProcessIdentityError):
                backend.member_identity(expected, 2002)

    def test_wayland_access_is_backend_derived_and_socket_pinned(self):
        backend = object.__new__(SystemdUserProcessBackend)
        backend._session_environment = {
            "PATH": "/usr/bin:/bin",
            "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
            "DBUS_SESSION_BUS_ADDRESS":
                f"unix:path=/run/user/{os.getuid()}/bus",
        }
        backend._wayland_display = "wayland-7"
        socket = mock.Mock(
            st_mode=stat.S_IFSOCK | 0o600, st_uid=os.getuid())
        with mock.patch("friday_core.processes.os.lstat", return_value=socket):
            environment = backend._session_access_environment(
                ProcessSessionAccess(wayland=True, session_bus=True))
        self.assertEqual(environment, {
            "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
            "WAYLAND_DISPLAY": "wayland-7",
            "DBUS_SESSION_BUS_ADDRESS":
                f"unix:path=/run/user/{os.getuid()}/bus",
        })

        backend._wayland_display = "../attacker-socket"
        with self.assertRaises(ProcessBackendError) as raised:
            backend._session_access_environment(
                ProcessSessionAccess(wayland=True))
        self.assertFalse(raised.exception.outcome_unknown)

    def test_runner_gets_only_validated_session_transport_and_fixed_path(self):
        captured: list[dict[str, object]] = []

        def runner(command, **kwargs):
            captured.append({"command": list(command), **kwargs})
            return subprocess.CompletedProcess(command, 0, "", "")

        backend = SystemdUserProcessBackend(
            runner=runner,
            user_manager_control_group=USER_MANAGER_CGROUP)
        backend._run([backend.systemctl, "--user", "is-system-running"])

        self.assertEqual(captured[0]["env"], {
            "PATH": "/usr/bin:/bin",
            "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
            "DBUS_SESSION_BUS_ADDRESS":
                f"unix:path=/run/user/{os.getuid()}/bus",
        })
        self.assertNotIn("HOME", captured[0]["env"])
        with self.assertRaises(ProcessBackendError):
            SystemdUserProcessBackend(
                runner=runner, session_runtime_dir="/tmp",
                user_manager_control_group=USER_MANAGER_CGROUP)

    def test_systemd_launch_keeps_natural_exit_observable(self):
        commands: list[list[str]] = []

        def runner(command, **_kwargs):
            commands.append(list(command))
            if command[0].endswith("systemd-run"):
                return subprocess.CompletedProcess(command, 0, "", "")
            return subprocess.CompletedProcess(
                command, 0, "LoadState=not-found\n", "")

        backend = SystemdUserProcessBackend(
            runner=runner,
            user_manager_control_group=USER_MANAGER_CGROUP)
        _path, identity = __import__(
            "friday_core.processes", fromlist=["_executable_identity"]
        )._executable_identity("/usr/bin/true")
        request = BackendLaunchRequest(
            instance_id="process_" + "1" * 32,
            unit_name="friday-proc-" + "1" * 32 + ".service",
            identity_token="2" * 64,
            argv=("/usr/bin/true",),
            cwd="/tmp",
            environment=(),
            executable_identity=identity,
            limits=ProcessLimits(),
            sandbox=BubblewrapProfile(enabled=False),
        )
        with self.assertRaises(ProcessBackendError):
            backend.launch(request)
        launch_command = next(
            command for command in commands
            if command[0].endswith("systemd-run"))
        self.assertIn("--property=RemainAfterExit=yes", launch_command)
        self.assertIn("--property=KillMode=control-group", launch_command)
        self.assertIn("--property=SendSIGKILL=yes", launch_command)

    def test_active_mainpid_zero_is_a_durable_natural_exit(self):
        token = "3" * 64

        unit_name = "friday-proc-" + "4" * 32 + ".service"

        def runner(command, **_kwargs):
            output = "\n".join((
                "LoadState=loaded", "ActiveState=active", "SubState=exited",
                f"Description=friday-managed:{token}",
                "InvocationID=trusted-invocation", "ControlGroup=",
                "MainPID=0", "ExecMainCode=1", "ExecMainStatus=0",
                "Result=success", "Job=")) + "\n"
            return subprocess.CompletedProcess(command, 0, output, "")

        backend = SystemdUserProcessBackend(
            runner=runner,
            user_manager_control_group=USER_MANAGER_CGROUP)
        with (mock.patch.object(backend, "_boot_id", return_value="boot"),
              mock.patch.object(backend, "_cgroup_empty", return_value=True)):
            observed = backend.inspect(unit_name)
        self.assertIsNotNone(observed)
        self.assertEqual(observed.state, "exited")
        self.assertTrue(observed.cgroup_empty)
        self.assertIsNone(observed.leader_pid)

    def test_post_stop_inspection_failure_is_unknown_not_terminal(self):
        token = "8" * 64
        unit_name = "friday-proc-" + "9" * 32 + ".service"
        show_calls = 0
        commands: list[list[str]] = []

        def runner(command, **_kwargs):
            nonlocal show_calls
            commands.append(list(command))
            if "show" in command:
                show_calls += 1
                if show_calls >= 3:
                    return subprocess.CompletedProcess(command, 1, "", "bus down")
                output = "\n".join((
                    "LoadState=loaded", "ActiveState=active",
                    "SubState=running",
                    f"Description=friday-managed:{token}",
                    "InvocationID=trusted-invocation",
                    f"ControlGroup={unit_cgroup(unit_name)}",
                    "MainPID=111", "ExecMainCode=0",
                    "ExecMainStatus=0", "Result=success", "Job=")) + "\n"
                return subprocess.CompletedProcess(command, 0, output, "")
            return subprocess.CompletedProcess(command, 0, "", "")

        backend = SystemdUserProcessBackend(
            runner=runner,
            user_manager_control_group=USER_MANAGER_CGROUP)
        with (mock.patch.object(
                backend, "_process_identity",
                return_value=(10, 20, 30, "a" * 64)),
              mock.patch.object(backend, "_pid_in_cgroup", return_value=True),
              mock.patch.object(backend, "_boot_id", return_value="boot"),
              mock.patch.object(backend, "_cgroup_empty", return_value=False)):
            expected = backend.inspect(unit_name)
            with self.assertRaises(ProcessBackendError) as raised:
                backend.terminate(expected)

        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(raised.exception.code, "systemd_inspection_failed")
        self.assertTrue(any("stop" in command for command in commands))
        self.assertFalse(expected.cgroup_empty)

    def test_nonzero_stop_cannot_claim_a_coincident_terminal_state(self):
        unit_name = "friday-proc-" + "6" * 32 + ".service"
        expected = BackendObservation(
            unit_name=unit_name, identity_token="7" * 64,
            state="running", boot_id="boot",
            invocation_id="trusted-invocation",
            control_group=unit_cgroup(unit_name), leader_pid=111,
            start_ticks=10, exe_device=20, exe_inode=30,
            exe_sha256="a" * 64, cgroup_empty=False,
            result_code="success")
        coincident_exit = expected.model_copy(update={
            "state": "exited", "leader_pid": None,
            "start_ticks": None, "exe_device": None, "exe_inode": None,
            "exe_sha256": None, "cgroup_empty": True,
        })

        def runner(command, **_kwargs):
            return subprocess.CompletedProcess(
                command, 1, "", "stop request rejected")

        backend = SystemdUserProcessBackend(
            runner=runner,
            user_manager_control_group=USER_MANAGER_CGROUP)
        with mock.patch.object(
                backend, "inspect",
                side_effect=[expected, coincident_exit]) as inspect:
            with self.assertRaises(ProcessBackendError) as raised:
                backend.terminate(expected)

        self.assertEqual(raised.exception.code, "systemd_termination_failed")
        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(inspect.call_count, 1)

    def test_inspect_rejects_empty_duplicate_and_nonzero_show_results(self):
        unit_name = "friday-proc-" + "a" * 32 + ".service"
        for label, result in (
            ("empty", subprocess.CompletedProcess([], 0, "", "")),
            ("duplicate", subprocess.CompletedProcess(
                [], 0, "LoadState=loaded\nLoadState=not-found\n", "")),
            ("nonzero", subprocess.CompletedProcess(
                [], 1, "LoadState=not-found\n", "failed")),
        ):
            with self.subTest(label=label):
                backend = SystemdUserProcessBackend(
                    runner=lambda _command, **_kwargs: result,
                    user_manager_control_group=USER_MANAGER_CGROUP)
                with self.assertRaises(ProcessBackendError):
                    backend.inspect(unit_name)

    def test_sandbox_wrapper_resolves_only_one_nonwrapper_target(self):
        token = "5" * 64

        unit_name = "friday-proc-" + "7" * 32 + ".service"

        def runner(command, **_kwargs):
            output = "\n".join((
                "LoadState=loaded", "ActiveState=active", "SubState=running",
                f"Description=friday-managed:{token}",
                "InvocationID=trusted-invocation",
                f"ControlGroup={unit_cgroup(unit_name)}",
                "MainPID=111", "ExecMainCode=0", "ExecMainStatus=0",
                "Result=success", "Job=")) + "\n"
            return subprocess.CompletedProcess(command, 0, output, "")

        backend = SystemdUserProcessBackend(
            runner=runner,
            user_manager_control_group=USER_MANAGER_CGROUP)
        wrapper = backend._backend_identities[backend.bwrap_executable]
        target_hash = "6" * 64
        with (mock.patch.object(
                backend, "_process_identity",
                return_value=(10, wrapper.device, wrapper.inode,
                              wrapper.sha256)),
              mock.patch.object(
                  backend, "_sandbox_target_identity",
                  return_value=(222, 20, 30, 40, target_hash)),
              mock.patch.object(backend, "_pid_in_cgroup", return_value=True),
              mock.patch.object(backend, "_boot_id", return_value="boot"),
              mock.patch.object(backend, "_cgroup_empty", return_value=False)):
            observed = backend.inspect(unit_name)
        self.assertEqual(observed.leader_pid, 222)
        self.assertEqual(observed.start_ticks, 20)
        self.assertEqual(observed.exe_sha256, target_hash)

    def test_terminal_retirement_accepts_only_absent_empty_cgroup(self):
        unit_name = "friday-proc-" + "b" * 32 + ".service"
        commands: list[list[str]] = []

        def runner(command, **_kwargs):
            commands.append(list(command))
            return subprocess.CompletedProcess(
                command, 0, "LoadState=not-found\n", "")

        backend = SystemdUserProcessBackend(
            runner=runner,
            user_manager_control_group=USER_MANAGER_CGROUP)
        fence = BackendTerminalFence(
            unit_name=unit_name, identity_token="c" * 64)
        with mock.patch.object(backend, "_cgroup_empty", return_value=True):
            backend.retire_terminal(fence)
        self.assertFalse(any("stop" in command for command in commands))

        with mock.patch.object(backend, "_cgroup_empty", return_value=False):
            with self.assertRaises(ProcessCleanupBlocked) as raised:
                backend.retire_terminal(fence)
        self.assertEqual(
            raised.exception.code, "process_unit_cleanup_cgroup_not_empty")
        self.assertFalse(any("stop" in command for command in commands))

    def test_terminal_retirement_stops_exact_loaded_empty_unit_once(self):
        token = "d" * 64
        unit_name = "friday-proc-" + "e" * 32 + ".service"
        show_calls = 0
        commands: list[list[str]] = []

        def runner(command, **_kwargs):
            nonlocal show_calls
            commands.append(list(command))
            if "show" in command:
                show_calls += 1
                if show_calls > 1:
                    return subprocess.CompletedProcess(
                        command, 0, "LoadState=not-found\n", "")
                output = "\n".join((
                    "LoadState=loaded", "ActiveState=active",
                    "SubState=exited",
                    f"Description=friday-managed:{token}",
                    "InvocationID=cleanup-invocation", "ControlGroup=",
                    "MainPID=0", "ExecMainCode=1", "ExecMainStatus=0",
                    "Result=success", "Job=")) + "\n"
                return subprocess.CompletedProcess(command, 0, output, "")
            return subprocess.CompletedProcess(command, 0, "", "")

        backend = SystemdUserProcessBackend(
            runner=runner,
            user_manager_control_group=USER_MANAGER_CGROUP)
        fence = BackendTerminalFence(
            unit_name=unit_name, identity_token=token,
            invocation_id="cleanup-invocation",
            control_group=unit_cgroup(unit_name))
        with (mock.patch.object(backend, "_boot_id", return_value="boot"),
              mock.patch.object(backend, "_cgroup_empty", return_value=True)):
            backend.retire_terminal(fence)

        stop_commands = [command for command in commands if "stop" in command]
        self.assertEqual(stop_commands, [[
            backend.systemctl, "--user", "stop", "--job-mode=fail",
            unit_name]])

    def test_terminal_retirement_never_stops_foreign_loaded_unit(self):
        unit_name = "friday-proc-" + "f" * 32 + ".service"
        commands: list[list[str]] = []

        def runner(command, **_kwargs):
            commands.append(list(command))
            output = "\n".join((
                "LoadState=loaded", "ActiveState=active", "SubState=exited",
                f"Description=friday-managed:{'1' * 64}",
                "InvocationID=foreign-invocation", "ControlGroup=",
                "MainPID=0", "ExecMainCode=1", "ExecMainStatus=0",
                "Result=success", "Job=")) + "\n"
            return subprocess.CompletedProcess(command, 0, output, "")

        backend = SystemdUserProcessBackend(
            runner=runner,
            user_manager_control_group=USER_MANAGER_CGROUP)
        fence = BackendTerminalFence(
            unit_name=unit_name, identity_token="2" * 64,
            invocation_id="trusted-invocation",
            control_group=unit_cgroup(unit_name))
        with (mock.patch.object(backend, "_boot_id", return_value="boot"),
              mock.patch.object(backend, "_cgroup_empty", return_value=True)):
            with self.assertRaises(ProcessCleanupBlocked):
                backend.retire_terminal(fence)
        self.assertFalse(any("stop" in command for command in commands))

    def test_terminal_retirement_rejects_jobs_and_transitional_state(self):
        unit_name = "friday-proc-" + "0" * 32 + ".service"
        token = "4" * 64
        for label, active, sub, job, expected_code in (
            ("job", "active", "exited", "/job/42",
             "process_unit_cleanup_job_pending"),
            ("transition", "deactivating", "stop-sigterm", "",
             "process_unit_cleanup_state_not_terminal"),
        ):
            commands: list[list[str]] = []

            def runner(command, **_kwargs):
                commands.append(list(command))
                output = "\n".join((
                    "LoadState=loaded", f"ActiveState={active}",
                    f"SubState={sub}",
                    f"Description=friday-managed:{token}",
                    "InvocationID=cleanup-invocation", "ControlGroup=",
                    "MainPID=0", "ExecMainCode=1", "ExecMainStatus=0",
                    "Result=success", f"Job={job}")) + "\n"
                return subprocess.CompletedProcess(command, 0, output, "")

            with self.subTest(condition=label):
                backend = SystemdUserProcessBackend(
                    runner=runner,
                    user_manager_control_group=USER_MANAGER_CGROUP)
                fence = BackendTerminalFence(
                    unit_name=unit_name, identity_token=token,
                    invocation_id="cleanup-invocation",
                    control_group=unit_cgroup(unit_name))
                with (mock.patch.object(
                        backend, "_boot_id", return_value="boot"),
                      mock.patch.object(
                          backend, "_cgroup_empty", return_value=True)):
                    with self.assertRaises(ProcessBackendError) as raised:
                        backend.retire_terminal(fence)
                self.assertEqual(raised.exception.code, expected_code)
                self.assertFalse(any("stop" in item for item in commands))


if __name__ == "__main__":
    unittest.main()
