from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from friday_core.desktop import (
    DesktopActionError,
    DesktopApplicationLaunchBinding,
    DesktopBindingError,
    DesktopBroker,
    DesktopSnapshot,
    DesktopUnavailableError,
    DesktopWindowBinding,
    DesktopWindowObservation,
    HyprlandDesktopBackend,
)
from friday_core.cognition import ContractBuilder
from friday_core.graph import GraphStore, sha256_text
from friday_core.processes import ExecutableIdentity
from friday_core.processes import (BackendObservation, ProcessLaunchBinding,
                                   ProcessPresentation, ProcessResources)
from friday_core.tasks import TaskService


SESSION = sha256_text("test-hyprland-session")
EXECUTABLE = ExecutableIdentity(
    device=8, inode=12345, sha256=sha256_text("test-executable"),
    size=4096, mode=0o755)


def process_observation(*, pid: int = 2001,
                        executable: ExecutableIdentity = EXECUTABLE
                        ) -> BackendObservation:
    return BackendObservation(
        unit_name="friday-proc-" + "1" * 32 + ".service",
        identity_token="2" * 64,
        state="running", boot_id="boot",
        invocation_id="invocation", control_group="/trusted/cgroup",
        leader_pid=pid, start_ticks=pid * 10,
        exe_device=executable.device, exe_inode=executable.inode,
        exe_sha256=executable.sha256, cgroup_empty=False,
        result_code="success")


def process_launch_binding() -> ProcessLaunchBinding:
    return ProcessLaunchBinding(
        spec_id="app.test.v1", name="test-app", version=1,
        spec_fingerprint="3" * 64, sandbox_fingerprint="4" * 64,
        args_sha256="5" * 64, executable_identity=EXECUTABLE,
        resource_claim=ProcessResources(
            cpu_cores=1.0, ram_mib=512, network=True),
        persistent=False)


def window(
    address: str,
    *,
    pid: int,
    application_id: str,
    workspace_id: int = 1,
    active: bool = False,
    start_ticks: int | None = None,
    stable_id: str | None = None,
    executable: ExecutableIdentity = EXECUTABLE,
) -> DesktopWindowObservation:
    return DesktopWindowObservation(
        session_fingerprint=SESSION,
        session_signature="test_session_12345678",
        address=address,
        stable_id=stable_id or f"{pid:08x}",
        application_id=application_id,
        workspace_id=workspace_id,
        active=active,
        pid=pid,
        start_ticks=start_ticks if start_ticks is not None else pid * 10,
        executable_identity=executable,
    )


class FakeDesktopBackend:
    def __init__(self, windows):
        self.windows = list(windows)
        self.focus_calls: list[tuple[str, str]] = []
        self.close_calls: list[tuple[str, str]] = []
        self.session_fingerprint = SESSION
        self.session_signature = "test_session_12345678"
        self.inventory_complete = True
        self.present_windows = None

    def snapshot(self):
        return DesktopSnapshot(
            session_fingerprint=self.session_fingerprint,
            session_signature=self.session_signature,
            windows=tuple(self.windows),
            present_windows=(tuple(self.present_windows)
                             if self.present_windows is not None else None),
            inventory_complete=self.inventory_complete,
        )

    def focus_window(self, session_signature, address):
        self.focus_calls.append((session_signature, address))
        self.windows = [item.model_copy(update={
            "active": item.address == address,
        }) for item in self.windows]

    def close_window(self, session_signature, address):
        self.close_calls.append((session_signature, address))
        self.windows = [item for item in self.windows if item.address != address]
        if self.present_windows is not None:
            self.present_windows = [
                item for item in self.present_windows
                if item.address != address]


class DesktopBrokerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph = GraphStore(self.root / "friday.db")
        self.backend = FakeDesktopBackend([
            window("0xabc", pid=1001,
                   application_id="com.mitchellh.ghostty"),
            window("0xdef", pid=1002, application_id="chromium", active=True),
        ])
        self.broker = DesktopBroker(
            self.graph, self.backend, state_root=self.root / "desktop")

    def tearDown(self):
        self.temporary.cleanup()

    def _stage(self, tool_name: str, binding: DesktopWindowBinding):
        tasks = TaskService(self.graph)
        contract = ContractBuilder().build("desktop test", [tool_name])
        task_id, _ = tasks.create(
            "desktop test", contract.model_dump(mode="json"))
        _batch, steps = tasks.stage_step_batch(task_id, [{
            "tool_call_id": "call_desktop_1",
            "tool_name": tool_name,
            "args": {"window_id": binding.window_id},
            "risk": "high",
            "approval_status": "approved",
            "idempotency_class": "reconcilable",
            "recovery_policy": "reconcile",
            "executor_binding": binding.model_dump(mode="json"),
            "resource_claims": {},
        }], round_index=0)
        return steps[0]["idempotency_key"]

    def test_list_is_opaque_and_privacy_safe(self):
        receipt = self.broker.list_windows()
        self.assertEqual(receipt["status"], "ok")
        self.assertTrue(receipt["verified"])
        self.assertEqual(len(receipt["windows"]), 2)
        encoded = json.dumps(receipt)
        self.assertNotIn("0xabc", encoded)
        self.assertNotIn('"pid"', encoded)
        self.assertNotIn('"title"', encoded)
        self.assertRegex(receipt["windows"][0]["window_id"],
                         r"^win_[0-9a-f]{40}$")
        self.assertEqual(
            oct((self.root / "desktop" / "window-id-key").stat().st_mode & 0o777),
            "0o600")

    def test_application_launch_requires_exact_process_owned_window(self):
        presentation = ProcessPresentation(
            application_id="com.friday.managedterminal",
            application="Friday Terminal", startup_timeout_seconds=0.5)
        binding = self.broker.binding_for_application_launch(
            process_launch_binding(), presentation)
        self.assertIsInstance(binding, DesktopApplicationLaunchBinding)
        self.assertNotIn(
            presentation.application_id,
            json.dumps(binding.model_dump(mode="json")))

        launched = window(
            "0xaaa", pid=2001,
            application_id=presentation.application_id,
            active=True)
        self.backend.windows.append(launched)
        observation = process_observation()
        process_receipt = {
            "status": "ok", "verified": True,
            "instance_id": "process_" + "6" * 32,
            "spec_id": binding.process.spec_id, "state": "running",
        }
        receipt = self.broker.confirm_application_launch(
            binding, process_receipt, observation, presentation)

        self.assertEqual(receipt["presentation"]["application"],
                         "Friday Terminal")
        self.assertTrue(receipt["presentation"]["verified"])
        self.assertNotIn("pid", json.dumps(receipt))
        self.assertTrue(self.broker.verify_application_launch_receipt(
            binding, receipt, process_receipt, observation, presentation))
        forged = json.loads(json.dumps(receipt))
        forged["presentation"]["window_id"] = "win_" + "f" * 40
        self.assertFalse(self.broker.verify_application_launch_receipt(
            binding, forged, process_receipt, observation, presentation))

    def test_application_launch_can_require_exact_managed_cgroup_member(self):
        presentation = ProcessPresentation(
            application_id="com.example.ManagedBrowser",
            application="Managed Browser", startup_timeout_seconds=0.5,
            window_owner="managed_cgroup")
        binding = self.broker.binding_for_application_launch(
            process_launch_binding(), presentation)
        child_executable = ExecutableIdentity(
            device=8, inode=54321, sha256=sha256_text("renderer"),
            size=8192, mode=0o755)
        outside = window(
            "0xab1", pid=9001,
            application_id=presentation.application_id,
            executable=child_executable)
        child = window(
            "0xab2", pid=2002, start_ticks=77_777,
            application_id=presentation.application_id,
            executable=child_executable)
        self.backend.windows.extend([outside, child])
        observation = process_observation()
        process_receipt = {
            "status": "ok", "verified": True,
            "instance_id": "process_" + "9" * 32,
            "spec_id": binding.process.spec_id, "state": "running",
        }
        checked = []

        def runtime_owner(expected, candidate):
            self.assertTrue(observation.same_live_execution(expected))
            checked.append((candidate.pid, candidate.start_ticks,
                            candidate.executable_identity))
            return bool(
                candidate.pid == child.pid
                and candidate.start_ticks == child.start_ticks
                and candidate.executable_identity == child_executable)

        receipt = self.broker.confirm_application_launch(
            binding, process_receipt, observation, presentation,
            runtime_owner)

        self.assertEqual(receipt["presentation"]["application"],
                         "Managed Browser")
        self.assertEqual({item[0] for item in checked}, {9001, 2002})
        self.assertTrue(self.broker.verify_application_launch_receipt(
            binding, receipt, process_receipt, observation, presentation,
            runtime_owner))
        with self.assertRaisesRegex(
                DesktopBindingError, "cgroup_verifier_unavailable"):
            self.broker.reconciliation_application_receipt(
                binding, process_receipt, observation, presentation)

    def test_managed_cgroup_window_rejects_pid_reuse_and_ambiguity(self):
        presentation = ProcessPresentation(
            application_id="com.example.ManagedApp",
            application="Managed App", startup_timeout_seconds=0.5,
            window_owner="managed_cgroup")
        binding = self.broker.binding_for_application_launch(
            process_launch_binding(), presentation)
        observation = process_observation()
        process_receipt = {
            "status": "ok", "verified": True,
            "instance_id": "process_" + "a" * 32,
            "spec_id": binding.process.spec_id, "state": "running",
        }
        first = window(
            "0xab3", pid=2010, start_ticks=80_000,
            application_id=presentation.application_id)
        reused = first.model_copy(update={
            "address": "0xab4", "stable_id": "0000ab04",
            "start_ticks": 80_001,
        })
        self.backend.windows.extend([first, reused])

        def exact_owner(_expected, candidate):
            return candidate.pid == 2010 and candidate.start_ticks == 80_000

        recovered = self.broker.reconciliation_application_receipt(
            binding, process_receipt, observation, presentation, exact_owner)
        self.assertTrue(recovered["presentation"]["verified"])

        with self.assertRaisesRegex(
                DesktopBindingError, "desktop_application_window_ambiguous"):
            self.broker.reconciliation_application_receipt(
                binding, process_receipt, observation, presentation,
                lambda _expected, _candidate: True)

    def test_application_launch_timeout_is_unknown_and_reconcilable(self):
        presentation = ProcessPresentation(
            application_id="com.friday.managedterminal",
            application="Friday Terminal", startup_timeout_seconds=0.5)
        binding = self.broker.binding_for_application_launch(
            process_launch_binding(), presentation)
        observation = process_observation()
        process_receipt = {
            "status": "ok", "verified": True,
            "instance_id": "process_" + "7" * 32,
            "spec_id": binding.process.spec_id, "state": "running",
        }
        with (patch("friday_core.desktop.time.monotonic",
                    side_effect=[0.0, 1.0]),
              self.assertRaises(DesktopActionError) as raised):
            self.broker.confirm_application_launch(
                binding, process_receipt, observation, presentation)
        self.assertTrue(raised.exception.outcome_unknown)
        self.assertIsNone(self.broker.reconciliation_application_receipt(
            binding, process_receipt, observation, presentation))

        self.backend.windows.append(window(
            "0xaab", pid=2001,
            application_id=presentation.application_id))
        recovered = self.broker.reconciliation_application_receipt(
            binding, process_receipt, observation, presentation)
        self.assertTrue(recovered["presentation"]["verified"])

    def test_application_launch_rejects_wrong_session_pid_and_ambiguity(self):
        presentation = ProcessPresentation(
            application_id="com.friday.managedterminal",
            application="Friday Terminal", startup_timeout_seconds=0.5)
        binding = self.broker.binding_for_application_launch(
            process_launch_binding(), presentation)
        observation = process_observation()
        process_receipt = {
            "status": "ok", "verified": True,
            "instance_id": "process_" + "8" * 32,
            "spec_id": binding.process.spec_id, "state": "running",
        }
        self.backend.windows.extend([
            window("0xaac", pid=2999,
                   application_id=presentation.application_id),
            window("0xaad", pid=2001,
                   application_id=presentation.application_id),
            window("0xaae", pid=2001,
                   application_id=presentation.application_id),
        ])
        with self.assertRaisesRegex(
                DesktopBindingError, "desktop_application_window_ambiguous"):
            self.broker.reconciliation_application_receipt(
                binding, process_receipt, observation, presentation)

        self.backend.windows = self.backend.windows[:2]
        self.backend.session_fingerprint = sha256_text("new-session")
        with self.assertRaisesRegex(
                DesktopBindingError, "desktop_session_binding_changed"):
            self.broker.reconciliation_application_receipt(
                binding, process_receipt, observation, presentation)

    def test_opaque_ids_are_stable_but_bound_to_runtime_identity(self):
        first = self.broker.list_windows()["windows"]
        second = self.broker.list_windows()["windows"]
        self.assertEqual(first, second)
        prior = next(item["window_id"] for item in first
                     if item["application"] == "Terminal")
        self.backend.windows[0] = self.backend.windows[0].model_copy(
            update={"start_ticks": 999_999})
        changed = self.broker.list_windows()["windows"]
        changed_terminal = next(item["window_id"] for item in changed
                                if item["application"] == "Terminal")
        self.assertNotEqual(prior, changed_terminal)

    def test_focus_uses_exact_binding_and_verifies_postcondition(self):
        target = next(item for item in self.broker.list_windows()["windows"]
                      if item["application"] == "Terminal")
        binding = self.broker.binding_for_action(target["window_id"], "focus")
        receipt = self.broker.focus_window(
            target["window_id"], expected_binding=binding)
        self.assertEqual(receipt["state"], "focused")
        self.assertFalse(receipt["idempotent_replay"])
        self.assertEqual(len(self.backend.focus_calls), 1)
        replay = self.broker.focus_window(
            target["window_id"], expected_binding=binding)
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(len(self.backend.focus_calls), 1)

    def test_raw_focus_dispatch_failure_is_outcome_unknown(self):
        target = next(item for item in self.broker.list_windows()["windows"]
                      if item["application"] == "Terminal")
        binding = self.broker.binding_for_action(target["window_id"], "focus")
        original_focus = self.backend.focus_window

        def focus_then_fail(session_signature, address):
            original_focus(session_signature, address)
            raise OSError("injected compositor transport loss")

        with patch.object(
                self.backend, "focus_window", side_effect=focus_then_fail):
            with self.assertRaises(DesktopActionError) as raised:
                self.broker.focus_window(
                    target["window_id"], expected_binding=binding)

        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(len(self.backend.focus_calls), 1)

    def test_changed_window_identity_cannot_receive_approved_focus(self):
        target = next(item for item in self.broker.list_windows()["windows"]
                      if item["application"] == "Terminal")
        binding = self.broker.binding_for_action(target["window_id"], "focus")
        self.backend.windows[0] = self.backend.windows[0].model_copy(
            update={"start_ticks": 999_999})
        with self.assertRaisesRegex(
                DesktopBindingError, "desktop_window_unavailable"):
            self.broker.focus_window(
                target["window_id"], expected_binding=binding)
        self.assertEqual(self.backend.focus_calls, [])

    def test_session_change_cannot_be_mistaken_for_closed_window(self):
        target = next(item for item in self.broker.list_windows()["windows"]
                      if item["application"] == "Terminal")
        binding = self.broker.binding_for_action(target["window_id"], "close")
        key = self._stage("machine_close_window", binding)
        self.backend.session_fingerprint = sha256_text("replacement-session")
        self.backend.windows = [item.model_copy(update={
            "session_fingerprint": self.backend.session_fingerprint,
            "session_signature": "replacement_session_12345678",
        }) for item in self.backend.windows]
        with self.assertRaisesRegex(
                DesktopBindingError, "desktop_session_binding_changed"):
            self.broker.close_window(
                target["window_id"], expected_binding=binding)
        forged = self.broker._receipt(
            binding, "closed", idempotent_replay=True)
        self.assertFalse(self.broker.verify_receipt(
            "machine_close_window", forged,
            {"window_id": target["window_id"]}, key))
        self.assertIsNone(self.broker.reconciliation_receipt(binding))
        self.assertEqual(self.backend.close_calls, [])

    def test_session_change_cannot_receive_approved_focus(self):
        target = next(item for item in self.broker.list_windows()["windows"]
                      if item["application"] == "Terminal")
        binding = self.broker.binding_for_action(target["window_id"], "focus")
        self.backend.session_fingerprint = sha256_text("replacement-session")
        self.backend.windows = [item.model_copy(update={
            "session_fingerprint": self.backend.session_fingerprint,
            "session_signature": "replacement_session_12345678",
        }) for item in self.backend.windows]
        with self.assertRaisesRegex(
                DesktopBindingError, "desktop_session_binding_changed"):
            self.broker.focus_window(
                target["window_id"], expected_binding=binding)
        self.assertEqual(self.backend.focus_calls, [])

    def test_close_is_exact_and_idempotent(self):
        target = next(item for item in self.broker.list_windows()["windows"]
                      if item["application"] == "Terminal")
        binding = self.broker.binding_for_action(target["window_id"], "close")
        receipt = self.broker.close_window(
            target["window_id"], expected_binding=binding)
        self.assertEqual(receipt["state"], "closed")
        self.assertFalse(receipt["idempotent_replay"])
        replay = self.broker.close_window(
            target["window_id"], expected_binding=binding)
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(len(self.backend.close_calls), 1)

    def test_raw_close_verification_failure_is_outcome_unknown(self):
        target = next(item for item in self.broker.list_windows()["windows"]
                      if item["application"] == "Terminal")
        binding = self.broker.binding_for_action(target["window_id"], "close")
        original_snapshot = self.backend.snapshot
        snapshots = 0

        def fail_after_dispatch():
            nonlocal snapshots
            snapshots += 1
            if snapshots >= 2:
                raise RuntimeError("injected process identity read failure")
            return original_snapshot()

        with patch.object(
                self.backend, "snapshot", side_effect=fail_after_dispatch):
            with self.assertRaises(DesktopActionError) as raised:
                self.broker.close_window(
                    target["window_id"], expected_binding=binding)

        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(len(self.backend.close_calls), 1)

    def test_reconciliation_probe_is_read_only_and_requires_postcondition(self):
        listed = self.broker.list_windows()["windows"]
        terminal = next(item for item in listed
                        if item["application"] == "Terminal")
        chromium = next(item for item in listed
                        if item["application"] == "Chromium")
        close_binding = self.broker.binding_for_action(
            terminal["window_id"], "close")
        focus_binding = self.broker.binding_for_action(
            terminal["window_id"], "focus")
        active_binding = self.broker.binding_for_action(
            chromium["window_id"], "focus")

        self.assertIsNone(
            self.broker.reconciliation_receipt(close_binding))
        self.assertIsNone(
            self.broker.reconciliation_receipt(focus_binding))
        focused = self.broker.reconciliation_receipt(active_binding)
        self.assertEqual(focused["state"], "focused")
        self.backend.windows = [
            item for item in self.backend.windows if item.address != "0xabc"]
        closed = self.broker.reconciliation_receipt(close_binding)
        self.assertEqual(closed["state"], "closed")
        self.assertEqual(self.backend.focus_calls, [])
        self.assertEqual(self.backend.close_calls, [])

    def test_session_loss_after_dispatch_is_outcome_unknown(self):
        target = next(item for item in self.broker.list_windows()["windows"]
                      if item["application"] == "Terminal")
        binding = self.broker.binding_for_action(target["window_id"], "close")

        def swap_session(_signature, address):
            self.backend.close_calls.append((_signature, address))
            replacement = sha256_text("post-dispatch-replacement")
            self.backend.session_fingerprint = replacement
            self.backend.windows = [item.model_copy(update={
                "session_fingerprint": replacement,
                "session_signature": "replacement_session_12345678",
            }) for item in self.backend.windows]

        self.backend.close_window = swap_session
        with self.assertRaises(DesktopActionError) as raised:
            self.broker.close_window(
                target["window_id"], expected_binding=binding)
        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(len(self.backend.close_calls), 1)

    def test_binding_rejects_operation_or_argument_substitution(self):
        target = self.broker.list_windows()["windows"][0]
        binding = self.broker.binding_for_action(target["window_id"], "focus")
        with self.assertRaises(DesktopBindingError):
            self.broker.close_window(
                target["window_id"], expected_binding=binding)
        with self.assertRaises(ValidationError):
            DesktopWindowBinding.model_validate(
                binding.model_dump() | {"window_id": "0xabc"})

    def test_authoritative_verifier_binds_durable_focus_step(self):
        target = next(item for item in self.broker.list_windows()["windows"]
                      if item["application"] == "Terminal")
        binding = self.broker.binding_for_action(target["window_id"], "focus")
        key = self._stage("machine_focus_window", binding)
        receipt = self.broker.focus_window(
            target["window_id"], expected_binding=binding)
        args = {"window_id": target["window_id"]}
        self.assertTrue(self.broker.verify_receipt(
            "machine_focus_window", json.dumps(receipt), args, key))
        forged = dict(receipt)
        forged["application"] = "Forged"
        self.assertFalse(self.broker.verify_receipt(
            "machine_focus_window", forged, args, key))
        self.assertFalse(self.broker.verify_receipt(
            "machine_focus_window", receipt, args, "act_forged"))

    def test_authoritative_verifier_confirms_closed_postcondition(self):
        target = next(item for item in self.broker.list_windows()["windows"]
                      if item["application"] == "Terminal")
        binding = self.broker.binding_for_action(target["window_id"], "close")
        key = self._stage("machine_close_window", binding)
        receipt = self.broker.close_window(
            target["window_id"], expected_binding=binding)
        args = {"window_id": target["window_id"]}
        self.assertTrue(self.broker.verify_receipt(
            "machine_close_window", receipt, args, key))
        self.backend.windows.append(window(
            "0xabc", pid=1001, application_id="com.mitchellh.ghostty"))
        self.assertFalse(self.broker.verify_receipt(
            "machine_close_window", receipt, args, key))

    def test_incomplete_inventory_never_proves_window_absence(self):
        target = next(item for item in self.broker.list_windows()["windows"]
                      if item["application"] == "Terminal")
        binding = self.broker.binding_for_action(target["window_id"], "close")
        key = self._stage("machine_close_window", binding)
        alleged = self.broker._receipt(
            binding, "closed", idempotent_replay=True)
        self.backend.windows = []
        self.backend.inventory_complete = False

        with self.assertRaisesRegex(
                DesktopUnavailableError, "desktop_inventory_incomplete"):
            self.broker.close_window(
                target["window_id"], expected_binding=binding)
        self.assertIsNone(self.broker.reconciliation_receipt(binding))
        self.assertFalse(self.broker.verify_receipt(
            "machine_close_window", alleged,
            {"window_id": target["window_id"]}, key))
        self.assertEqual(self.backend.close_calls, [])

    def test_unmapped_target_is_present_until_exact_close_dispatch(self):
        target = next(item for item in self.broker.list_windows()["windows"]
                      if item["application"] == "Terminal")
        binding = self.broker.binding_for_action(target["window_id"], "close")
        key = self._stage("machine_close_window", binding)
        self.backend.present_windows = list(self.backend.windows)
        self.backend.windows = [
            item for item in self.backend.windows
            if self.broker._window_id(item) != target["window_id"]]
        alleged = self.broker._receipt(
            binding, "closed", idempotent_replay=True)

        self.assertIsNone(self.broker.reconciliation_receipt(binding))
        self.assertFalse(self.broker.verify_receipt(
            "machine_close_window", alleged,
            {"window_id": target["window_id"]}, key))
        receipt = self.broker.close_window(
            target["window_id"], expected_binding=binding)

        self.assertEqual(receipt["state"], "closed")
        self.assertFalse(receipt["idempotent_replay"])
        self.assertEqual(len(self.backend.close_calls), 1)

    def test_list_verifier_rejects_private_or_stale_receipts(self):
        receipt = self.broker.list_windows()
        self.assertTrue(self.broker.verify_receipt(
            "machine_list_windows", receipt, {}, None))
        private = dict(receipt) | {"pid": 123}
        self.assertFalse(self.broker.verify_receipt(
            "machine_list_windows", private, {}, None))
        self.backend.windows = self.backend.windows[:1]
        self.assertFalse(self.broker.verify_receipt(
            "machine_list_windows", receipt, {}, None))

    def test_property_parser_rejects_duplicate_and_malformed_values(self):
        self.assertEqual(
            HyprlandDesktopBackend._properties("Active=yes\nState=active\n"),
            {"Active": "yes", "State": "active"})
        for value in (
                "Active=yes\nActive=no\n", "not-a-property\n",
                "Bad_Key=value\n", "Active=" + "x" * 1025 + "\n"):
            with self.subTest(value=value[:32]):
                with self.assertRaisesRegex(
                        DesktopUnavailableError, "desktop_response_invalid"):
                    HyprlandDesktopBackend._properties(value)

    def test_compositor_snapshot_rejects_ambiguous_window_identities(self):
        base = {
            "mapped": True, "address": "0xabc", "stableId": "aaaaaaaa",
            "initialClass": "ghostty", "workspace": {"id": 1},
            "pid": 1001, "floating": False, "fullscreen": 0,
        }
        conflicts = (
            base | {"stableId": "bbbbbbbb", "pid": 1002},
            base | {"address": "0xdef", "pid": 1002},
        )
        for conflict in conflicts:
            with self.subTest(conflict=conflict):
                backend = object.__new__(HyprlandDesktopBackend)
                backend._session = lambda: ("test_session_12345678", SESSION)
                backend._json = lambda *args: (
                    [base, conflict] if args[-1] == "clients"
                    else {"address": "0xabc"})
                with patch(
                        "friday_core.desktop._proc_identity",
                        return_value=(10010, EXECUTABLE, "/usr/bin/ghostty")):
                    with self.assertRaisesRegex(
                            DesktopUnavailableError,
                            "desktop_response_ambiguous"):
                        backend.snapshot()

    def test_compositor_snapshot_bounds_clients_and_active_address(self):
        backend = object.__new__(HyprlandDesktopBackend)
        backend._session = lambda: ("test_session_12345678", SESSION)
        backend._json = lambda *args: (
            [{}] * 1025 if args[-1] == "clients" else {})
        with self.assertRaisesRegex(
                DesktopUnavailableError, "desktop_response_invalid"):
            backend.snapshot()

    def test_compositor_marks_skipped_mapped_client_inventory_incomplete(self):
        client = {
            "mapped": True, "address": "0xabc", "stableId": "aaaaaaaa",
            "initialClass": "ghostty", "workspace": {"id": 1},
            "pid": 1001, "floating": False, "fullscreen": 0,
        }
        backend = object.__new__(HyprlandDesktopBackend)
        backend._session = lambda: ("test_session_12345678", SESSION)
        backend._json = lambda *args: (
            [client] if args[-1] == "clients" else {"address": "0xabc"})
        with patch(
                "friday_core.desktop._proc_identity",
                side_effect=DesktopUnavailableError(
                    "desktop_process_identity_unavailable")):
            snapshot = backend.snapshot()

        self.assertEqual(snapshot.windows, ())
        self.assertFalse(snapshot.inventory_complete)

    def test_compositor_retains_unmapped_client_for_absence_proof(self):
        client = {
            "mapped": False, "address": "0xabc", "stableId": "aaaaaaaa",
            "initialClass": "ghostty", "workspace": {"id": 1},
            "pid": 1001, "floating": False, "fullscreen": 0,
        }
        backend = object.__new__(HyprlandDesktopBackend)
        backend._session = lambda: ("test_session_12345678", SESSION)
        backend._json = lambda *args: (
            [client] if args[-1] == "clients" else {})
        with patch(
                "friday_core.desktop._proc_identity",
                return_value=(10010, EXECUTABLE, "/usr/bin/ghostty")):
            snapshot = backend.snapshot()

        self.assertEqual(snapshot.windows, ())
        self.assertEqual(len(snapshot.present_windows), 1)
        self.assertTrue(snapshot.inventory_complete)

        backend._json = lambda *args: (
            [] if args[-1] == "clients"
            else {"address": "0xabc;close-everything"})
        with self.assertRaisesRegex(
                DesktopUnavailableError, "desktop_response_invalid"):
            backend.snapshot()

    def test_dispatcher_accepts_only_a_fixed_address_selector(self):
        backend = object.__new__(HyprlandDesktopBackend)
        backend.hyprctl = "/usr/bin/hyprctl"
        backend._session = lambda: ("test_session_12345678", SESSION)
        calls = []
        backend._run = lambda command: calls.append(tuple(command))

        with self.assertRaises(DesktopBindingError):
            backend.focus_window(
                "test_session_12345678", "0xabc'; close-everything")
        self.assertEqual(calls, [])

        backend.focus_window("test_session_12345678", "0xAbC")
        self.assertEqual(calls, [(
            "/usr/bin/hyprctl", "-i", "test_session_12345678", "eval",
            "hl.dispatch(hl.dsp.focus({ window = 'address:0xabc' }))",
        )])


if __name__ == "__main__":
    unittest.main()
