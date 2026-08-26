import json
import tempfile
import unittest
from pathlib import Path

from friday_core import (ContractBuilder, GraphStore, OutcomeVerifier, Planner,
                         PolicyEngine, TaskService, resource_claim_for)


class CognitiveKernelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.graph = GraphStore(Path(self.tmp.name) / "friday.db")
        self.tasks = TaskService(self.graph)
        self.contracts = ContractBuilder()
        self.verifier = OutcomeVerifier()

    def tearDown(self):
        self.tmp.cleanup()

    def _running(self, tool="list_files"):
        contract = self.contracts.build("Inspect the project", [tool])
        plan = Planner().build([{"name": tool}], contract)
        task_id, _ = self.tasks.create(
            contract.objective, contract.model_dump(mode="json"))
        self.tasks.transition(task_id, "interpreting")
        self.tasks.set_plan(task_id, plan)
        self.tasks.transition(task_id, "planned")
        self.tasks.transition(task_id, "running")
        return task_id, contract

    def test_versioned_task_cannot_complete_without_verification(self):
        task_id, _contract = self._running()
        self.tasks.transition(task_id, "verifying")
        with self.assertRaisesRegex(ValueError, "requires passed verification"):
            self.tasks.transition(task_id, "completed")

    def test_verified_receipt_satisfies_contract(self):
        task_id, contract = self._running()
        handle, _ = self.tasks.begin_action(
            task_id, "list_files", {"path": "."}, ordinal=1)
        action_check = self.verifier.verify_action(
            "list_files", "f server.py", succeeded=True)
        self.tasks.finish_action(
            handle, "f server.py", succeeded=True, verification=action_check)
        task_check = self.verifier.verify_task(
            contract, self.tasks.action_history(task_id))
        self.assertTrue(task_check.passed)
        self.tasks.transition(task_id, "verifying")
        self.tasks.record_verification(task_id, task_check)
        self.tasks.transition(task_id, "completed")
        self.assertEqual(self.tasks.get(task_id)["verification_status"], "passed")

    def test_unknown_external_receipt_is_uncertain_not_failed(self):
        task_id, contract = self._running("machine_close_window")
        batch_id, _ = self.tasks.stage_step_batch(task_id, [{
            "tool_call_id": "close-unknown",
            "tool_name": "machine_close_window",
            "args": {"window_id": "win_" + "a" * 40},
            "risk": "high", "approval_status": "approved",
            "idempotency_class": "reconcilable",
            "recovery_policy": "reconcile",
            "executor_binding": {}, "resource_claims": {},
        }], round_index=0)
        claim = self.tasks.claim_next_step(batch_id, "unknown-worker")
        self.tasks.mark_step_outcome_unknown(
            claim, reason_code="desktop_close_outcome_unknown")

        verification = self.verifier.verify_task(
            contract, self.tasks.action_history(task_id))

        self.assertEqual(verification.status.value, "uncertain")
        self.assertIn("authoritative outcome", " ".join(verification.missing))
        progress = self.tasks.record_verification(task_id, verification)
        self.assertEqual(progress["label"], "Outcome remains uncertain")

    def test_news_without_attributable_url_fails(self):
        result = json.dumps({"headlines": [
            {"title": "Claim", "source": "Wire"}
        ]})
        check = self.verifier.verify_action("fetch_news", result, succeeded=True)
        self.assertFalse(check.passed)
        self.assertTrue(check.missing)

    def test_plan_names_the_contract_verifier(self):
        contract = self.contracts.build("Search", ["web_search"])
        plan = Planner().build([{"name": "web_search"}], contract)
        self.assertEqual(plan.steps[0].verifier, "source_receipt")

    def test_core_upgrade_requires_approval_without_explicit_request(self):
        decision = PolicyEngine().decide("upgrade_core", explicitly_requested=False)
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.approval_required)
        self.assertEqual(decision.risk.value, "high")

    def test_project_write_requires_exact_user_approval(self):
        # Server treats each exact content proposal as a separate action from
        # the user's higher-level request to edit a file.
        decision = PolicyEngine().decide(
            "write_file", explicitly_requested=False)

        self.assertTrue(decision.allowed)
        self.assertTrue(decision.approval_required)

    def test_process_launch_and_termination_are_high_risk_approved_actions(self):
        for tool_name in ("machine_launch_process",
                          "machine_terminate_process"):
            decision = PolicyEngine().decide(
                tool_name, explicitly_requested=False)
            self.assertTrue(decision.allowed)
            self.assertTrue(decision.approval_required)
            self.assertEqual(decision.risk.value, "high")
            self.assertEqual(decision.permissions, ["process"])

    def test_process_termination_uses_only_the_reserved_control_lane_shape(self):
        claim = resource_claim_for(
            "machine_terminate_process", permissions=["process"])

        self.assertEqual(claim.cpu_cores, 0.0)
        self.assertEqual(claim.ram_mib, 0)
        self.assertEqual(claim.vram_mib, 0)
        self.assertEqual(claim.accelerator, "none")
        self.assertFalse(claim.network)
        self.assertEqual(claim.concurrency_slots, 1)
        self.assertEqual(claim.latency_class, "control")

    def test_process_contract_requires_specific_lifecycle_receipts(self):
        contract = self.contracts.build(
            "Launch then stop a curated application",
            ["machine_launch_process", "machine_terminate_process"],
        )
        plan = Planner().build([
            {"name": "machine_launch_process"},
            {"name": "machine_terminate_process"},
        ], contract)

        self.assertEqual(plan.steps[0].verifier, "process_launch_receipt")
        self.assertEqual(plan.steps[1].verifier, "process_terminate_receipt")
        self.assertTrue(contract.needs_user_confirmation)

    def test_process_receipts_fail_closed_without_authoritative_broker(self):
        plausible = json.dumps({
            "status": "ok", "verified": True,
            "instance_id": "process_0123456789abcdef0123456789abcdef",
            "spec_id": "proc.managed_wait.v1", "state": "running",
        })

        check = self.verifier.verify_action(
            "machine_launch_process", plausible, succeeded=True,
            args={"spec_id": "proc.managed_wait.v1",
                  "parameter_values": {"seconds": 30}},
            idempotency_key="act_0123456789abcdef")

        self.assertFalse(check.passed)

    def test_process_receipt_uses_injected_authoritative_verifier(self):
        calls = []

        def verify(tool_name, result, args, idempotency_key):
            calls.append((tool_name, result, args, idempotency_key))
            return True

        verifier = OutcomeVerifier(process_receipt_verifier=verify)
        result = json.dumps({"status": "ok", "verified": True, "specs": []})
        check = verifier.verify_action(
            "machine_list_process_specs", result, succeeded=True,
            args={}, idempotency_key=None)

        self.assertTrue(check.passed)
        self.assertEqual(calls, [(
            "machine_list_process_specs", result, {}, None)])

    def test_desktop_focus_and_close_require_exact_approval(self):
        for tool_name, expected_risk in (
                ("machine_focus_window", "medium"),
                ("machine_close_window", "high")):
            decision = PolicyEngine().decide(
                tool_name, explicitly_requested=False)
            self.assertTrue(decision.allowed)
            self.assertTrue(decision.approval_required)
            self.assertEqual(decision.risk.value, expected_risk)
            self.assertEqual(decision.permissions, ["desktop"])

    def test_desktop_contract_names_state_specific_verifiers(self):
        contract = self.contracts.build(
            "Observe, focus, then close a window",
            ["machine_list_windows", "machine_focus_window",
             "machine_close_window"])
        plan = Planner().build([
            {"name": "machine_list_windows"},
            {"name": "machine_focus_window"},
            {"name": "machine_close_window"},
        ], contract)

        self.assertEqual(plan.steps[0].verifier, "desktop_list_receipt")
        self.assertEqual(plan.steps[1].verifier, "desktop_focus_receipt")
        self.assertEqual(plan.steps[2].verifier, "desktop_close_receipt")
        self.assertTrue(contract.needs_user_confirmation)

    def test_desktop_receipts_fail_closed_without_authoritative_broker(self):
        plausible = json.dumps({
            "status": "ok", "verified": True, "operation": "focus",
            "window_id": "win_" + "a" * 40,
            "application": "Chromium",
            "workspace_id": 1, "state": "focused",
            "idempotent_replay": False,
        })

        check = self.verifier.verify_action(
            "machine_focus_window", plausible, succeeded=True,
            args={"window_id": "win_" + "a" * 40},
            idempotency_key="act_0123456789abcdef")

        self.assertFalse(check.passed)

    def test_desktop_receipt_uses_injected_authoritative_verifier(self):
        calls = []

        def verify(tool_name, result, args, idempotency_key):
            calls.append((tool_name, result, args, idempotency_key))
            return True

        verifier = OutcomeVerifier(desktop_receipt_verifier=verify)
        result = json.dumps({"status": "ok", "verified": True, "windows": []})
        check = verifier.verify_action(
            "machine_list_windows", result, succeeded=True,
            args={}, idempotency_key=None)

        self.assertTrue(check.passed)
        self.assertEqual(calls, [(
            "machine_list_windows", result, {}, None)])

    def test_dynamic_effects_require_exact_approval_and_drive_risk(self):
        decision = PolicyEngine().decide(
            "generated_sync", explicitly_requested=True,
            dynamic_permissions=["filesystem_read", "network", "process"],
            executor_identity="generated_sync@v3 (abc123)")

        self.assertTrue(decision.allowed)
        self.assertTrue(decision.approval_required)
        self.assertEqual(decision.risk.value, "high")
        self.assertEqual(
            decision.permissions,
            ["filesystem_read", "network", "process"])
        self.assertIn("generated_sync@v3", decision.reason)

    def test_dynamic_contract_includes_declared_permissions_and_confirmation(self):
        contract = self.contracts.build(
            "Run generated tool", ["generated_sync"],
            dynamic_permissions={
                "generated_sync": ["network", "filesystem_read"]})

        self.assertEqual(contract.risk.value, "high")
        self.assertEqual(contract.permissions, ["filesystem_read", "network"])
        self.assertTrue(contract.needs_user_confirmation)

    def test_core_upgrade_receipt_proves_reviewable_not_promoted(self):
        reviewable = self.verifier.verify_action(
            "upgrade_core", json.dumps({
                "status": "awaiting_review", "changed": ["server.py"]}),
            succeeded=True)
        unsafe_claim = self.verifier.verify_action(
            "upgrade_core", json.dumps({
                "status": "promoted", "changed": ["server.py"]}),
            succeeded=True)

        self.assertTrue(reviewable.passed)
        self.assertFalse(unsafe_claim.passed)


if __name__ == "__main__":
    unittest.main()
