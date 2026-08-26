import json
import math
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from friday_core import GraphStore, ResourceClaim, TaskService
from friday_core.graph import canonical_json


class TaskStepServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.graph = GraphStore(Path(self.tmp.name) / "friday.db")
        self.tasks = TaskService(self.graph)
        self.task_id, _ = self.tasks.create(
            "Run a durable tool batch",
            {"version": 0, "evidence": "durable step receipts"},
        )

    def tearDown(self):
        self.tmp.cleanup()

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
            context={"session_id": "session-test", "turn_id": "turn-test"},
        )

    def event_types(self):
        return [
            event["event_type"]
            for event in self.graph.events_since(task_id=self.task_id)
        ]

    def test_staged_steps_have_ordered_dependencies_and_claim_in_order(self):
        batch_id, steps = self.stage(
            self.call("call-1"),
            self.call("call-2"),
            self.call("call-3"),
        )

        self.assertEqual([step["ordinal"] for step in steps], [1, 2, 3])
        self.assertEqual(
            [step["depends_on"] for step in steps],
            [[], [steps[0]["step_id"]], [steps[1]["step_id"]]],
        )

        first = self.tasks.claim_next_step(batch_id, "worker-1")
        self.assertIsNotNone(first)
        self.assertEqual(first.step_id, steps[0]["step_id"])
        self.assertIsNone(self.tasks.claim_next_step(batch_id, "worker-2"))

        self.tasks.finish_step(first, {"entries": []}, succeeded=True)
        second = self.tasks.claim_next_step(batch_id, "worker-2")
        self.assertIsNotNone(second)
        self.assertEqual(second.step_id, steps[1]["step_id"])
        self.tasks.finish_step(second, {"entries": []}, succeeded=True)

        third = self.tasks.claim_next_step(batch_id, "worker-3")
        self.assertIsNotNone(third)
        self.assertEqual(third.step_id, steps[2]["step_id"])

    def test_two_workers_racing_only_produce_one_claim(self):
        batch_id, steps = self.stage(self.call("call-race"))
        ready = threading.Barrier(2)

        def race(worker_id):
            ready.wait(timeout=5)
            return self.tasks.claim_next_step(batch_id, worker_id)

        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(race, ("worker-a", "worker-b")))

        winners = [claim for claim in claims if claim is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0].step_id, steps[0]["step_id"])
        self.assertIn(winners[0].worker_id, {"worker-a", "worker-b"})
        self.assertEqual(self.graph.count("action_receipts"), 1)
        self.assertEqual(self.graph.count("action_attempts"), 1)

    def test_forced_recovery_fences_the_stale_lease_from_finishing(self):
        batch_id, _ = self.stage(self.call("call-stale"))
        stale_claim = self.tasks.claim_next_step(batch_id, "dead-worker")
        self.assertIsNotNone(stale_claim)

        recovered = self.tasks.recover_inflight_steps(
            force=True, dead_worker_id="dead-worker")

        self.assertEqual(recovered["retry"], [stale_claim.step_id])
        with self.assertRaisesRegex(PermissionError, "lease is stale"):
            self.tasks.finish_step(stale_claim, "late result", succeeded=True)
        self.assertNotIn("action.finished", self.event_types())
        self.assertNotIn("step.succeeded", self.event_types())

    def test_interrupted_read_only_claim_retries_same_dispatch_then_successor(self):
        original_args = {"path": "/tmp/input", "include_hidden": True}
        batch_id, steps = self.stage(
            self.call("call-retry", args=original_args),
            self.call("call-successor", args={"path": "/tmp/next"}),
        )
        first = self.tasks.claim_next_step(batch_id, "worker-before-restart")
        self.assertIsNotNone(first)
        self.assertEqual(first.attempt_number, 1)

        recovered = self.tasks.recover_inflight_steps(force=True)
        self.assertEqual(recovered, {"retry": [first.step_id], "reconcile": []})

        retry = self.tasks.claim_next_step(batch_id, "worker-after-restart")
        self.assertIsNotNone(retry)
        self.assertEqual(retry.step_id, first.step_id)
        self.assertEqual(retry.action_id, first.action_id)
        self.assertEqual(retry.idempotency_key, first.idempotency_key)
        self.assertEqual(retry.args, original_args)
        self.assertEqual(retry.attempt_number, 2)

        self.tasks.finish_step(retry, {"entries": ["one"]}, succeeded=True)
        successor = self.tasks.claim_next_step(batch_id, "worker-after-restart")
        self.assertIsNotNone(successor)
        self.assertEqual(successor.step_id, steps[1]["step_id"])
        self.assertEqual(successor.attempt_number, 1)

    def test_interrupted_non_repeatable_claim_requires_reconciliation(self):
        batch_id, steps = self.stage(
            self.call(
                "call-non-repeatable",
                tool_name="send_message",
                args={"message": "send exactly once"},
                idempotency_class="non_repeatable",
            )
        )
        claim = self.tasks.claim_next_step(batch_id, "dead-worker")
        self.assertIsNotNone(claim)

        recovered = self.tasks.recover_inflight_steps(force=True)

        self.assertEqual(recovered, {"retry": [], "reconcile": [claim.step_id]})
        self.assertEqual(
            self.tasks.list_steps(batch_id=batch_id)[0]["status"],
            "reconcile_required",
        )
        self.assertEqual(self.tasks.step_batch(batch_id)["status"],
                         "reconcile_required")
        self.assertEqual(steps[0]["recovery_policy"], "reconcile")
        self.assertIsNone(self.tasks.claim_next_step(batch_id, "new-worker"))

    def test_finish_records_exactly_one_action_and_step_success_pair(self):
        batch_id, _ = self.stage(self.call("call-finish"))
        claim = self.tasks.claim_next_step(batch_id, "worker-finish")
        self.assertIsNotNone(claim)

        self.tasks.finish_step(claim, {"entries": ["done"]}, succeeded=True)
        with self.assertRaisesRegex(PermissionError, "lease is stale"):
            self.tasks.finish_step(claim, {"entries": ["duplicate"]},
                                   succeeded=True)

        event_types = self.event_types()
        self.assertEqual(event_types.count("action.finished"), 1)
        self.assertEqual(event_types.count("step.succeeded"), 1)
        self.assertLess(
            event_types.index("action.finished"),
            event_types.index("step.succeeded"),
        )
        self.assertEqual(self.tasks.list_steps(batch_id=batch_id)[0]["status"],
                         "succeeded")

    def test_private_args_are_absent_from_database_dump_and_claim_repr(self):
        private_text = "browser-private-text-5f63bdf1"
        private_url = "https://private.example/?token=url-secret-9802"
        exact_args = {
            "selector": "#message",
            "text": private_text,
            "page_url": private_url,
        }
        batch_id, _ = self.stage(
            self.call(
                "call-private",
                tool_name="browser_type",
                args=exact_args,
                idempotency_class="non_repeatable",
            )
        )
        claim = self.tasks.claim_next_step(batch_id, "private-worker")
        self.assertIsNotNone(claim)

        with self.graph._connect() as conn:
            database_dump = "\n".join(conn.iterdump())
        claim_repr = repr(claim)

        self.assertEqual(claim.args, exact_args)
        for secret in (private_text, private_url):
            self.assertNotIn(secret, database_dump)
            self.assertNotIn(secret, claim_repr)
            self.assertNotIn("args=", claim_repr)
            self.assertIn(claim.step_id, claim_repr)
            self.assertIn("browser_type", claim_repr)

    def test_executor_version_and_resource_claims_are_durable_and_fingerprinted(self):
        binding = {
            "kind": "capability", "name": "generated_sync",
            "version_id": "capv_exact", "version": 3,
            "code_sha256": "a" * 64,
            "permissions": ["network"],
        }
        resources = {
            "cpu_cores": 1.0, "ram_mib": 512, "vram_mib": 0,
            "accelerator": "none", "network": True,
            "concurrency_slots": 1,
            "latency_class": "interactive",
        }
        batch_id, steps = self.tasks.stage_step_batch(
            self.task_id, [{
                **self.call("bound", tool_name="generated_sync"),
                "executor_binding": binding,
                "resource_claims": resources,
            }], round_index=0)

        self.assertEqual(steps[0]["executor_binding"], binding)
        self.assertEqual(steps[0]["resource_claims"], resources)
        claim = self.tasks.claim_next_step(batch_id, "bound-worker")
        self.assertEqual(claim.executor_binding, binding)
        self.assertEqual(claim.resource_claims, resources)

        with self.assertRaisesRegex(RuntimeError, "different durable batch"):
            self.tasks.stage_step_batch(
                self.task_id, [{
                    **self.call("bound", tool_name="generated_sync"),
                    "executor_binding": binding | {"version_id": "capv_other"},
                    "resource_claims": resources,
                }], round_index=0)

    def test_resource_claims_are_validated_and_canonicalized_before_staging(self):
        supplied = {"cpu_cores": 1, "ram_mib": 384, "network": True}
        expected = ResourceClaim.model_validate(supplied).model_dump(mode="json")

        _batch_id, steps = self.stage(
            self.call("canonical-resource-claim") | {
                "resource_claims": supplied,
            })

        self.assertEqual(steps[0]["resource_claims"], expected)
        with self.graph._connect() as conn:
            stored = conn.execute(
                "SELECT resource_claims_json FROM task_steps WHERE step_id=?",
                (steps[0]["step_id"],),
            ).fetchone()[0]
        self.assertEqual(json.loads(stored), expected)
        self.assertEqual(stored, canonical_json(expected))

    def test_invalid_resource_claims_are_rejected_before_any_step_is_staged(self):
        invalid_claims = {
            "negative CPU": {"cpu_cores": -0.01},
            "NaN CPU": {"cpu_cores": math.nan},
            "infinite CPU": {"cpu_cores": math.inf},
            "unknown field": {"cpu_cores": 1.0, "gpu_memory": 512},
            "VRAM without accelerator": {
                "vram_mib": 512,
                "accelerator": "none",
            },
        }

        for index, (label, resource_claims) in enumerate(
                invalid_claims.items()):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self.tasks.stage_step_batch(
                        self.task_id,
                        [self.call(f"invalid-resource-{index}") | {
                            "resource_claims": resource_claims,
                        }],
                        round_index=index,
                    )

        self.assertEqual(self.graph.count("task_steps"), 0)
        self.assertEqual(self.graph.count("task_step_batches"), 0)
        self.assertEqual(self.graph.count("action_receipts"), 0)

    @staticmethod
    def _bound_call(call_id="bound-integrity"):
        return {
            "tool_call_id": call_id,
            "tool_name": "generated_sync",
            "args": {"path": "/tmp/source"},
            "idempotency_class": "read_only",
            "executor_binding": {
                "kind": "capability",
                "name": "generated_sync",
                "version_id": "capv_integrity",
                "version": 4,
                "code_sha256": "b" * 64,
                "permissions": ["network"],
            },
            "resource_claims": {
                "cpu_cores": 1.0,
                "ram_mib": 512,
                "vram_mib": 0,
                "accelerator": "none",
                "network": True,
                "latency_class": "interactive",
            },
        }

    def _assert_mutated_batch_is_rejected_before_dispatch(
            self, column, mutated_value):
        batch_id, steps = self.tasks.stage_step_batch(
            self.task_id, [self._bound_call()], round_index=0)
        with self.graph.transaction() as conn:
            conn.execute(
                f"UPDATE task_steps SET {column}=? WHERE step_id=?",
                (canonical_json(mutated_value), steps[0]["step_id"]),
            )

        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            self.tasks.claim_next_step(batch_id, "integrity-worker")

        self.assertEqual(self.graph.count("action_receipts"), 0)
        self.assertEqual(self.graph.count("action_attempts"), 0)
        self.assertNotIn("action.started", self.event_types())
        self.assertEqual(
            self.tasks.list_steps(batch_id=batch_id)[0]["status"], "pending")

    def test_executor_binding_mutation_is_rejected_before_dispatch(self):
        self._assert_mutated_batch_is_rejected_before_dispatch(
            "executor_binding_json",
            self._bound_call()["executor_binding"] | {
                "version_id": "capv_attacker_substitution",
            },
        )

    def test_resource_claim_mutation_is_rejected_before_dispatch(self):
        self._assert_mutated_batch_is_rejected_before_dispatch(
            "resource_claims_json",
            self._bound_call()["resource_claims"] | {
                "ram_mib": 1,
                "network": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
