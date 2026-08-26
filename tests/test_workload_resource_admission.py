from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

from friday_core import GraphStore, ResourceClaim
from friday_core.admission import (
    AdmissionBudget,
    ResourceAdmissionController,
    ResourceSnapshot,
)


T0 = datetime(2026, 8, 24, 8, 0, tzinfo=UTC)


def budget(**overrides: object) -> AdmissionBudget:
    values: dict[str, object] = {
        "cpu_millis": 8_000,
        "ram_mib": 8_192,
        "concurrency_slots": 8,
        "network_slots": 8,
        "accelerator_vram_mib": {"cuda:0": 8_192},
    }
    values.update(overrides)
    return AdmissionBudget(**values)


def snapshot(**overrides: object) -> ResourceSnapshot:
    values: dict[str, object] = {
        "available_cpu_millis": 8_000,
        "available_ram_mib": 8_192,
        "available_network_slots": 8,
        "available_accelerator_vram_mib": {"cuda:0": 8_192},
        "captured_at": T0,
    }
    values.update(overrides)
    return ResourceSnapshot(**values)


class WorkloadResourceAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def graph(self, name: str = "friday.db") -> GraphStore:
        return GraphStore(Path(self.tmp.name) / name)

    @staticmethod
    def controller(
        graph: GraphStore,
        admission_budget: AdmissionBudget | None = None,
        *,
        runtime_id: str = "runtime-workloads",
        profile_fingerprint: str = "a" * 64,
        snapshot_provider=None,
        lease_ttl_seconds: int = 30,
    ) -> ResourceAdmissionController:
        return ResourceAdmissionController(
            graph,
            admission_budget or budget(),
            snapshot_provider=snapshot_provider,
            runtime_id=runtime_id,
            profile_fingerprint=profile_fingerprint,
            lease_ttl_seconds=lease_ttl_seconds,
        )

    @staticmethod
    def create_instance(
        graph: GraphStore,
        suffix: str,
        *,
        args_ciphertext: str = "enc:v1:test-args",
    ) -> str:
        spec_id = "process_spec_tests"
        instance_id = f"process_instance_{suffix}"
        now_text = T0.isoformat().replace("+00:00", "Z")
        with graph.transaction() as conn:
            if conn.execute(
                    "SELECT 1 FROM process_specs WHERE spec_id=?",
                    (spec_id,)).fetchone() is None:
                event_id, seq = graph.append_event(
                    conn, "process.spec_registered",
                    {"spec_id": spec_id, "name": "test-process",
                     "version": 1}, actor="test")
                graph.append_node(
                    conn, "process_spec",
                    {"spec_id": spec_id, "name": "test-process",
                     "version": 1}, event_id=event_id, node_id=spec_id)
                conn.execute(
                    """INSERT INTO process_specs
                       (spec_id,name,version,spec_ciphertext,display_json,
                        spec_sha256,sandbox_fingerprint,status,created_at,
                        updated_at,last_event_seq)
                       VALUES (?,?,1,?,?,?,?,?,?,?,?)""",
                    (spec_id, "test-process", "enc:v1:test-spec", "{}",
                     "1" * 64, "2" * 64, "active", now_text, now_text,
                     seq),
                )
            event_id, seq = graph.append_event(
                conn, "process.instance_prepared",
                {"instance_id": instance_id, "spec_id": spec_id},
                actor="test")
            graph.append_node(
                conn, "process_instance",
                {"instance_id": instance_id, "spec_id": spec_id},
                event_id=event_id, node_id=instance_id)
            conn.execute(
                """INSERT INTO process_instances
                   (instance_id,spec_id,launch_idempotency_key,
                    args_ciphertext,args_redacted_json,args_sha256,
                    spec_fingerprint,sandbox_fingerprint,state,unit_name,
                    prepared_at,created_at,updated_at,last_event_seq)
                   VALUES (?,?,?,?,?,?,?,?, 'prepared',?,?,?,?,?)""",
                (instance_id, spec_id, f"launch-{suffix}", args_ciphertext,
                 "{}", "3" * 64, "4" * 64, "2" * 64,
                 f"friday-workload-{suffix}.service", now_text, now_text,
                 now_text, seq),
            )
        return instance_id

    @staticmethod
    def set_instance_state(
        graph: GraphStore,
        instance_id: str,
        state: str,
        *,
        finished_at: datetime | None = None,
    ) -> None:
        instant = finished_at or T0
        now_text = instant.isoformat().replace("+00:00", "Z")
        with graph.transaction() as conn:
            _, seq = graph.append_event(
                conn, "process.instance_test_transition",
                {"instance_id": instance_id, "state": state}, actor="test")
            conn.execute(
                """UPDATE process_instances
                   SET state=?,finished_at=?,updated_at=?,last_event_seq=?
                   WHERE instance_id=?""",
                (state, now_text if finished_at is not None else None,
                 now_text, seq, instance_id),
            )

    @staticmethod
    def workload_rows(graph: GraphStore) -> list[dict[str, object]]:
        with graph._connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM workload_resource_leases ORDER BY lease_id"
            ).fetchall()]

    @staticmethod
    def step_rows(graph: GraphStore) -> list[dict[str, object]]:
        with graph._connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM resource_leases ORDER BY lease_id"
            ).fetchall()]

    def acquire_workload(
        self,
        controller: ResourceAdmissionController,
        claim: ResourceClaim,
        instance_id: str,
        *,
        observed: ResourceSnapshot | None = None,
        enforcement: dict[str, object] | str | None = None,
        now: datetime = T0,
    ):
        return controller.acquire_workload(
            claim, instance_id,
            enforcement or {"unit": f"{instance_id}.service"},
            now, snapshot=observed or snapshot())

    def test_workload_never_ttl_expires_and_remains_reserved(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "persistent")
        controller = self.controller(
            graph, budget(concurrency_slots=1), lease_ttl_seconds=5)
        claim = ResourceClaim(cpu_cores=0.5, ram_mib=512)

        admitted = self.acquire_workload(controller, claim, instance_id)
        reaped = controller.reap_stale(now=T0 + timedelta(days=1))
        blocked = controller.acquire(
            claim, step_id="step-after-workload",
            attempt_id="attempt-after-workload",
            worker_id="worker-after-workload",
            snapshot=snapshot(captured_at=T0 + timedelta(days=1)),
            now=T0 + timedelta(days=1))

        self.assertTrue(admitted.admitted)
        self.assertEqual(reaped, 0)
        self.assertEqual(blocked.status, "deferred")
        self.assertEqual(self.workload_rows(graph)[0]["status"], "active")
        status = controller.status()
        self.assertEqual(status["active"]["leases"], 1)

    def test_direct_workload_acquire_reaps_expired_step_reservations(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "expired-step")
        controller = self.controller(
            graph, budget(concurrency_slots=1), lease_ttl_seconds=5)
        claim = ResourceClaim(cpu_cores=0.5, ram_mib=512)
        step = controller.acquire(
            claim, step_id="step-expired-before-workload",
            attempt_id="attempt-expired-before-workload",
            worker_id="worker-expired-before-workload",
            snapshot=snapshot(), now=T0)

        workload = self.acquire_workload(
            controller, claim, instance_id,
            observed=snapshot(captured_at=T0 + timedelta(seconds=6)),
            now=T0 + timedelta(seconds=6))

        self.assertTrue(step.admitted)
        self.assertTrue(workload.admitted)
        self.assertEqual(self.step_rows(graph)[0]["status"], "expired")
        self.assertEqual(self.workload_rows(graph)[0]["status"], "active")

    def test_live_capacity_subtracts_existing_workload_reservation(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "live-subtraction")
        controller = self.controller(graph)
        claim = ResourceClaim(
            cpu_cores=0.6, ram_mib=600, vram_mib=600,
            accelerator="cuda:0", network=True)
        lower_live = snapshot(
            available_cpu_millis=1_000,
            available_ram_mib=1_000,
            available_network_slots=1,
            available_accelerator_vram_mib={"cuda:0": 1_000})

        first = self.acquire_workload(
            controller, claim, instance_id, observed=lower_live)
        second = controller.acquire(
            claim, step_id="step-live-blocked",
            attempt_id="attempt-live-blocked",
            worker_id="worker-live-blocked", snapshot=lower_live, now=T0)

        self.assertTrue(first.admitted)
        self.assertEqual(second.status, "deferred")
        for dimension in (
            "cpu_millis", "ram_mib", "network_slots", "vram_mib:cuda:0",
        ):
            self.assertGreater(second.deficits.get(dimension, 0), 0)

    def test_interactive_usage_does_not_consume_noninteractive_quota(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "interactive-first")
        controller = self.controller(
            graph, budget(concurrency_slots=2, network_slots=2))
        interactive = ResourceClaim(
            cpu_cores=0.1, ram_mib=1, network=True,
            latency_class="interactive")
        background = interactive.model_copy(
            update={"latency_class": "background"})

        first = self.acquire_workload(
            controller, interactive, instance_id)
        second = controller.acquire(
            background, step_id="step-background-after-interactive",
            attempt_id="attempt-background-after-interactive",
            worker_id="worker-background-after-interactive",
            snapshot=snapshot(), now=T0)
        third = controller.acquire(
            background, step_id="step-second-background",
            attempt_id="attempt-second-background",
            worker_id="worker-second-background",
            snapshot=snapshot(), now=T0)

        self.assertTrue(first.admitted)
        self.assertTrue(second.admitted)
        self.assertEqual(third.status, "deferred")
        self.assertGreater(third.deficits.get("concurrency_slots", 0), 0)
        self.assertGreater(third.deficits.get("network_slots", 0), 0)
        active = controller.status()["active"]
        self.assertEqual(active["concurrency_slots"], 2)
        self.assertEqual(active["network_slots"], 2)
        self.assertEqual(active["latency_classes"], {
            "interactive": 1,
            "background": 1,
        })

    def test_exact_acquire_is_idempotent_and_every_binding_mismatch_rejects(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "idempotent")
        controller = self.controller(graph)
        claim = ResourceClaim(
            cpu_cores=1.0, ram_mib=768, network=True,
            latency_class="background")
        enforcement = {"limits": {"MemoryMax": "768M", "CPUQuota": "100%"}}

        first = self.acquire_workload(
            controller, claim, instance_id, enforcement=enforcement)
        exact = self.acquire_workload(
            controller, claim, instance_id,
            enforcement='{"limits":{"CPUQuota":"100%","MemoryMax":"768M"}}')
        changed_claim = self.acquire_workload(
            controller, claim.model_copy(update={"ram_mib": 769}), instance_id,
            enforcement=enforcement)
        changed_enforcement = self.acquire_workload(
            controller, claim, instance_id,
            enforcement={"limits": {"MemoryMax": "769M"}})
        changed_profile = self.acquire_workload(
            self.controller(
                graph, runtime_id="runtime-workloads",
                profile_fingerprint="b" * 64),
            claim, instance_id, enforcement=enforcement)
        changed_runtime = self.acquire_workload(
            self.controller(graph, runtime_id="runtime-other"),
            claim, instance_id, enforcement=enforcement)

        self.assertTrue(first.admitted)
        self.assertEqual(exact.reason, "already_admitted")
        self.assertEqual(first.lease_id, exact.lease_id)
        for mismatch in (
            changed_claim, changed_enforcement, changed_profile,
            changed_runtime,
        ):
            self.assertEqual(mismatch.status, "rejected")
            self.assertFalse(mismatch.retryable)
        self.assertEqual(len(self.workload_rows(graph)), 1)

    def test_exact_reacquire_does_not_depend_on_live_telemetry(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "retry-without-telemetry")
        owner = self.controller(graph, runtime_id="runtime-idempotent")
        claim = ResourceClaim(cpu_cores=0.5, ram_mib=512)
        step = owner.acquire(
            claim, step_id="step-idempotent-telemetry",
            attempt_id="attempt-idempotent-telemetry",
            worker_id="worker-idempotent-telemetry",
            snapshot=snapshot(), now=T0)
        workload = self.acquire_workload(
            owner, claim, instance_id, enforcement={"MemoryMax": 512})

        def unavailable_sampler():
            raise RuntimeError("resource telemetry unavailable")

        retrying = self.controller(
            graph, runtime_id="runtime-idempotent",
            snapshot_provider=unavailable_sampler)
        step_retry = retrying.acquire(
            claim, step_id="step-idempotent-telemetry",
            attempt_id="attempt-idempotent-telemetry",
            worker_id="worker-idempotent-telemetry",
            now=T0 + timedelta(seconds=1))
        workload_retry = retrying.acquire_workload(
            claim, instance_id, {"MemoryMax": 512},
            T0 + timedelta(seconds=1))

        self.assertTrue(step.admitted)
        self.assertTrue(workload.admitted)
        self.assertEqual(step_retry.reason, "already_admitted")
        self.assertEqual(workload_retry.reason, "already_admitted")
        self.assertEqual(step_retry.lease_id, step.lease_id)
        self.assertEqual(workload_retry.lease_id, workload.lease_id)

    def test_delayed_step_lifecycle_messages_cannot_regress_time(self):
        graph = self.graph()
        controller = self.controller(graph, lease_ttl_seconds=10)
        claim = ResourceClaim(cpu_cores=0.5, ram_mib=512)
        admitted = controller.acquire(
            claim, step_id="step-monotonic", attempt_id="attempt-monotonic",
            worker_id="worker-monotonic", snapshot=snapshot(), now=T0)
        lease_id = str(admitted.lease_id)

        self.assertTrue(controller.heartbeat(
            lease_id, "attempt-monotonic", worker_id="worker-monotonic",
            now=T0 + timedelta(seconds=5), lease_ttl_seconds=10))
        before = self.step_rows(graph)[0]
        self.assertFalse(controller.heartbeat(
            lease_id, "attempt-monotonic", worker_id="worker-monotonic",
            now=T0 + timedelta(seconds=1), lease_ttl_seconds=10))
        self.assertFalse(controller.release(
            lease_id, "attempt-monotonic", worker_id="worker-monotonic",
            now=T0 + timedelta(seconds=4)))
        after = self.step_rows(graph)[0]

        self.assertEqual(after["heartbeat_at"], before["heartbeat_at"])
        self.assertEqual(after["expires_at"], before["expires_at"])
        self.assertEqual(after["status"], "active")
        self.assertTrue(controller.release(
            lease_id, "attempt-monotonic", worker_id="worker-monotonic",
            now=T0 + timedelta(seconds=6)))

    def test_stale_and_future_telemetry_fail_closed_before_workload_insert(self):
        for index, captured_at in enumerate((
            T0 - timedelta(hours=1), T0 + timedelta(hours=1),
        )):
            with self.subTest(captured_at=captured_at):
                graph = self.graph(f"telemetry-{index}.db")
                instance_id = self.create_instance(graph, f"telemetry-{index}")
                controller = self.controller(graph)
                with self.assertRaisesRegex(
                        RuntimeError, "(?i)(?:snapshot|telemetry)"):
                    self.acquire_workload(
                        controller,
                        ResourceClaim(cpu_cores=0.1, ram_mib=1),
                        instance_id,
                        observed=snapshot(captured_at=captured_at))
                self.assertEqual(self.workload_rows(graph), [])

    def test_monitor_loss_stays_counted_and_adoption_is_fenced(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "reconcile")
        profile = "c" * 64
        owner = self.controller(
            graph, budget(concurrency_slots=1), runtime_id="runtime-owner",
            profile_fingerprint=profile, lease_ttl_seconds=5)
        claim = ResourceClaim(cpu_cores=0.5, ram_mib=512)
        admitted = self.acquire_workload(owner, claim, instance_id)
        lease_id = str(admitted.lease_id)

        self.assertFalse(owner.heartbeat_workload(
            lease_id, instance_id, now=T0 - timedelta(seconds=1)))
        self.assertFalse(owner.heartbeat_workload(
            lease_id, instance_id, now=T0 + timedelta(seconds=5)))
        self.assertFalse(owner.heartbeat_workload(
            lease_id, instance_id, now=T0 + timedelta(seconds=6)))
        self.assertTrue(owner.mark_workload_reconciling(
            lease_id, instance_id, now=T0 + timedelta(seconds=5)))
        self.assertEqual(owner.reap_stale(now=T0 + timedelta(days=1)), 0)
        self.assertEqual(owner.status()["active"]["leases"], 1)
        self.assertEqual(self.workload_rows(graph)[0]["status"],
                         "reconciling")

        wrong_runtime = self.controller(
            graph, runtime_id="runtime-wrong", profile_fingerprint=profile)
        wrong_profile = self.controller(
            graph, runtime_id="runtime-recovery",
            profile_fingerprint="d" * 64)
        recovery = self.controller(
            graph, runtime_id="runtime-recovery",
            profile_fingerprint=profile)
        self.assertFalse(wrong_runtime.heartbeat_workload(
            lease_id, instance_id, now=T0 + timedelta(seconds=7)))
        self.assertFalse(wrong_profile.adopt_workload(
            lease_id, instance_id, previous_runtime_id="runtime-owner",
            now=T0 + timedelta(seconds=7)))
        self.assertFalse(recovery.adopt_workload(
            lease_id, instance_id, previous_runtime_id="runtime-not-owner",
            now=T0 + timedelta(seconds=7)))
        self.assertTrue(recovery.adopt_workload(
            lease_id, instance_id, previous_runtime_id="runtime-owner",
            now=T0 + timedelta(seconds=7)))
        row = self.workload_rows(graph)[0]
        self.assertEqual(row["status"], "active")
        self.assertEqual(row["runtime_id"], "runtime-recovery")

    def test_new_runtime_can_fence_exact_stale_owner_then_adopt(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "startup-reconcile")
        profile = "e" * 64
        owner = self.controller(
            graph, budget(concurrency_slots=1), runtime_id="runtime-old",
            profile_fingerprint=profile, lease_ttl_seconds=5)
        claim = ResourceClaim(cpu_cores=0.5, ram_mib=512)
        admitted = self.acquire_workload(owner, claim, instance_id)
        lease_id = str(admitted.lease_id)
        recovery = self.controller(
            graph, budget(concurrency_slots=1), runtime_id="runtime-new",
            profile_fingerprint=profile, lease_ttl_seconds=5)
        wrong_profile = self.controller(
            graph, budget(concurrency_slots=1), runtime_id="runtime-new",
            profile_fingerprint="f" * 64, lease_ttl_seconds=5)

        self.assertEqual(recovery.workloads_needing_reconciliation(
            now=T0 + timedelta(seconds=1)), [])
        self.assertFalse(recovery.mark_stale_workload_runtime_reconciling(
            lease_id, instance_id, "runtime-old",
            now=T0 + timedelta(seconds=1)))
        self.assertEqual(self.workload_rows(graph)[0]["status"], "active")
        pending = recovery.workloads_needing_reconciliation(
            now=T0 + timedelta(seconds=6))
        self.assertEqual(pending, [{
            "lease_id": lease_id,
            "instance_id": instance_id,
            "runtime_id": "runtime-old",
            "profile_fingerprint": profile,
            "status": "active",
            "expires_at": str(admitted.expires_at),
        }])
        self.assertEqual(wrong_profile.workloads_needing_reconciliation(
            now=T0 + timedelta(seconds=6)), [])
        self.assertEqual(recovery.status()["active"]["leases"], 1)
        self.assertFalse(recovery.mark_stale_workload_runtime_reconciling(
            lease_id, instance_id, "runtime-new",
            now=T0 + timedelta(seconds=6)))
        self.assertFalse(recovery.mark_stale_workload_runtime_reconciling(
            lease_id, instance_id, "runtime-not-owner",
            now=T0 + timedelta(seconds=6)))
        self.assertFalse(
            wrong_profile.mark_stale_workload_runtime_reconciling(
                lease_id, instance_id, "runtime-old",
                now=T0 + timedelta(seconds=6)))
        self.assertTrue(recovery.mark_stale_workload_runtime_reconciling(
            lease_id, instance_id, "runtime-old",
            now=T0 + timedelta(seconds=6)))
        row = self.workload_rows(graph)[0]
        self.assertEqual(row["status"], "reconciling")
        self.assertEqual(row["runtime_id"], "runtime-old")
        self.assertTrue(recovery.adopt_workload(
            lease_id, instance_id, previous_runtime_id="runtime-old",
            now=T0 + timedelta(seconds=7)))
        self.assertTrue(recovery.adopt_workload(
            lease_id, instance_id, previous_runtime_id="runtime-old",
            now=T0 + timedelta(seconds=7)))
        self.assertFalse(recovery.adopt_workload(
            lease_id, instance_id, previous_runtime_id="runtime-not-owner",
            now=T0 + timedelta(seconds=7)))

    def test_new_runtime_can_release_reconciling_terminal_workload(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "terminal-reconcile")
        profile = "9" * 64
        owner = self.controller(
            graph, runtime_id="runtime-terminal-old",
            profile_fingerprint=profile, lease_ttl_seconds=5)
        claim = ResourceClaim(cpu_cores=0.5, ram_mib=512)
        admitted = self.acquire_workload(owner, claim, instance_id)
        lease_id = str(admitted.lease_id)
        recovery = self.controller(
            graph, runtime_id="runtime-terminal-new",
            profile_fingerprint=profile, lease_ttl_seconds=5)

        self.assertFalse(recovery.release_workload(
            lease_id, instance_id, cgroup_empty=True,
            previous_runtime_id="runtime-terminal-old",
            now=T0 + timedelta(seconds=1)))
        self.assertTrue(recovery.mark_stale_workload_runtime_reconciling(
            lease_id, instance_id, "runtime-terminal-old",
            now=T0 + timedelta(seconds=5)))
        self.set_instance_state(
            graph, instance_id, "exited",
            finished_at=T0 + timedelta(seconds=5))
        self.assertFalse(recovery.release_workload(
            lease_id, instance_id, cgroup_empty=True,
            now=T0 + timedelta(seconds=6)))
        self.assertFalse(recovery.release_workload(
            lease_id, instance_id, cgroup_empty=True,
            previous_runtime_id="runtime-wrong",
            now=T0 + timedelta(seconds=6)))
        self.assertFalse(recovery.release_workload(
            lease_id, instance_id, cgroup_empty=False,
            previous_runtime_id="runtime-terminal-old",
            now=T0 + timedelta(seconds=6)))
        self.assertTrue(recovery.release_workload(
            lease_id, instance_id, cgroup_empty=True,
            previous_runtime_id="runtime-terminal-old", reason="exited",
            now=T0 + timedelta(seconds=6)))
        row = self.workload_rows(graph)[0]
        self.assertEqual(row["status"], "released")
        self.assertEqual(row["runtime_id"], "runtime-terminal-old")
        self.assertEqual(recovery.status()["active"]["leases"], 0)

    def test_release_requires_positive_empty_cgroup_proof(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "release-proof")
        controller = self.controller(
            graph, budget(concurrency_slots=1))
        claim = ResourceClaim(cpu_cores=0.5, ram_mib=512)
        admitted = self.acquire_workload(controller, claim, instance_id)
        lease_id = str(admitted.lease_id)

        self.assertFalse(controller.release_workload(
            lease_id, instance_id, cgroup_empty=False,
            now=T0 + timedelta(seconds=1)))
        self.assertEqual(self.workload_rows(graph)[0]["status"], "active")
        self.assertTrue(controller.release_workload(
            lease_id, instance_id, cgroup_empty=True,
            now=T0 + timedelta(seconds=1)))
        self.assertEqual(self.workload_rows(graph)[0]["status"], "released")

        available = controller.acquire(
            claim, step_id="step-after-release",
            attempt_id="attempt-after-release",
            worker_id="worker-after-release", snapshot=snapshot(), now=T0)
        self.assertTrue(available.admitted)

    def test_concurrent_exact_acquire_creates_one_durable_lease(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "concurrent")
        controllers = (
            self.controller(graph, runtime_id="runtime-race"),
            self.controller(graph, runtime_id="runtime-race"),
        )
        claim = ResourceClaim(cpu_cores=0.5, ram_mib=512)
        ready = threading.Barrier(2)

        def race(index: int):
            ready.wait(timeout=5)
            return self.acquire_workload(
                controllers[index], claim, instance_id,
                enforcement={"MemoryMax": 512})

        with ThreadPoolExecutor(max_workers=2) as pool:
            decisions = list(pool.map(race, (0, 1)))

        self.assertTrue(all(decision.admitted for decision in decisions))
        self.assertEqual(
            {decision.lease_id for decision in decisions},
            {decisions[0].lease_id})
        self.assertEqual(len(self.workload_rows(graph)), 1)
        self.assertEqual(graph.count_nodes("workload_resource_lease"), 1)

    def test_concurrent_distinct_workloads_cannot_oversubscribe_one_slot(self):
        graph = self.graph()
        instance_ids = (
            self.create_instance(graph, "capacity-race-a"),
            self.create_instance(graph, "capacity-race-b"),
        )
        controllers = (
            self.controller(
                graph, budget(concurrency_slots=1), runtime_id="runtime-race"),
            self.controller(
                graph, budget(concurrency_slots=1), runtime_id="runtime-race"),
        )
        claim = ResourceClaim(cpu_cores=0.5, ram_mib=512)
        ready = threading.Barrier(2)

        def race(index: int):
            ready.wait(timeout=5)
            return self.acquire_workload(
                controllers[index], claim, instance_ids[index])

        with ThreadPoolExecutor(max_workers=2) as pool:
            decisions = list(pool.map(race, (0, 1)))

        self.assertEqual(
            sorted(decision.status for decision in decisions),
            ["admitted", "deferred"])
        self.assertEqual(len(self.workload_rows(graph)), 1)

    def test_transfer_is_atomic_idempotent_and_never_double_counts(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "transfer")
        controller = self.controller(
            graph, budget(concurrency_slots=1), runtime_id="runtime-transfer")
        claim = ResourceClaim(cpu_cores=1.0, ram_mib=1_024)
        source = controller.acquire(
            claim, step_id="step-transfer", attempt_id="attempt-transfer",
            worker_id="worker-transfer", snapshot=snapshot(), now=T0)

        transferred = controller.transfer_step_to_workload(
            claim, instance_id, str(source.lease_id), "attempt-transfer",
            "worker-transfer", {"MemoryMax": 1_024}, T0)
        exact_retry = controller.transfer_step_to_workload(
            claim, instance_id, str(source.lease_id), "attempt-transfer",
            "worker-transfer", {"MemoryMax": 1_024}, T0)

        self.assertTrue(transferred.admitted)
        self.assertEqual(exact_retry.reason, "already_transferred")
        self.assertEqual(exact_retry.lease_id, transferred.lease_id)
        source_row = self.step_rows(graph)[0]
        self.assertEqual(source_row["status"], "released")
        self.assertEqual(source_row["release_reason"],
                         "transferred_to_workload")
        self.assertTrue(controller.is_step_lease_safely_discharged(
            str(source.lease_id), "attempt-transfer",
            worker_id="worker-transfer"))
        active = controller.status()["active"]
        self.assertEqual(active["leases"], 1)
        self.assertEqual(active["cpu_millis"], 1_000)
        self.assertEqual(active["ram_mib"], 1_024)
        self.assertEqual(active["concurrency_slots"], 1)

    def test_transfer_mismatch_leaves_the_source_active_without_a_gap(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "transfer-mismatch")
        controller = self.controller(graph, runtime_id="runtime-transfer")
        claim = ResourceClaim(cpu_cores=1.0, ram_mib=1_024)
        source = controller.acquire(
            claim, step_id="step-transfer-mismatch",
            attempt_id="attempt-transfer-mismatch",
            worker_id="worker-transfer-mismatch",
            snapshot=snapshot(), now=T0)

        mismatch = controller.transfer_step_to_workload(
            claim.model_copy(update={"ram_mib": 1_025}), instance_id,
            str(source.lease_id), "attempt-transfer-mismatch",
            "worker-transfer-mismatch", {}, T0)

        self.assertEqual(mismatch.status, "rejected")
        self.assertEqual(mismatch.reason, "source_step_lease_fence_mismatch")
        self.assertEqual(self.step_rows(graph)[0]["status"], "active")
        self.assertEqual(self.workload_rows(graph), [])
        self.assertEqual(controller.status()["active"]["leases"], 1)

    def test_transferred_terminal_instance_is_safely_discharged(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "terminal-transfer")
        controller = self.controller(graph, runtime_id="runtime-transfer")
        claim = ResourceClaim(cpu_cores=0.5, ram_mib=512)
        source = controller.acquire(
            claim, step_id="step-terminal-transfer",
            attempt_id="attempt-terminal-transfer",
            worker_id="worker-terminal-transfer", snapshot=snapshot(), now=T0)
        workload = controller.transfer_step_to_workload(
            claim, instance_id, str(source.lease_id),
            "attempt-terminal-transfer", "worker-terminal-transfer", {}, T0)
        self.assertTrue(controller.release_workload(
            str(workload.lease_id), instance_id, cgroup_empty=True,
            reason="launch_failed", now=T0 + timedelta(seconds=1)))
        self.set_instance_state(
            graph, instance_id, "launch_failed",
            finished_at=T0 + timedelta(seconds=1))

        self.assertTrue(controller.is_step_lease_safely_discharged(
            str(source.lease_id), "attempt-terminal-transfer",
            worker_id="worker-terminal-transfer"))

        # Identity mismatch is safe only while its workload reservation remains
        # active/reconciling; a terminal-looking released row is insufficient.
        other_instance = self.create_instance(graph, "identity-mismatch")
        other_source = controller.acquire(
            claim, step_id="step-identity-mismatch",
            attempt_id="attempt-identity-mismatch",
            worker_id="worker-identity-mismatch", snapshot=snapshot(), now=T0)
        other_workload = controller.transfer_step_to_workload(
            claim, other_instance, str(other_source.lease_id),
            "attempt-identity-mismatch", "worker-identity-mismatch", {}, T0)
        self.set_instance_state(
            graph, other_instance, "identity_mismatch",
            finished_at=T0 + timedelta(seconds=1))
        self.assertTrue(controller.is_step_lease_safely_discharged(
            str(other_source.lease_id), "attempt-identity-mismatch",
            worker_id="worker-identity-mismatch"))
        self.assertTrue(controller.fence_workload(
            str(other_workload.lease_id), other_instance, cgroup_empty=True,
            now=T0 + timedelta(seconds=2)))
        self.assertFalse(controller.is_step_lease_safely_discharged(
            str(other_source.lease_id), "attempt-identity-mismatch",
            worker_id="worker-identity-mismatch"))

    def test_status_aggregates_without_exposing_instance_or_enforcement_data(self):
        graph = self.graph()
        instance_id = self.create_instance(
            graph, "privacy", args_ciphertext="enc:v1:private-command")
        controller = self.controller(graph)
        claim = ResourceClaim(
            cpu_cores=1.25, ram_mib=768, network=True,
            vram_mib=512, accelerator="cuda:0",
            latency_class="background")
        secret = "private-unit-token"

        admitted = self.acquire_workload(
            controller, claim, instance_id,
            enforcement={"unit": "private.service", "token": secret})
        self.assertTrue(admitted.admitted)
        status = controller.status()
        rendered = json.dumps(status, sort_keys=True)

        self.assertEqual(status["active"]["cpu_millis"], 1_250)
        self.assertEqual(status["active"]["ram_mib"], 768)
        self.assertEqual(status["active"]["network_slots"], 1)
        self.assertEqual(status["active"]["accelerator_vram_mib"],
                         {"cuda:0": 512})
        self.assertEqual(status["active"]["latency_classes"],
                         {"background": 1})
        for private_value in (
            instance_id, "private-command", "private.service", secret,
            "enforcement_json", "source_worker_id",
        ):
            self.assertNotIn(private_value, rendered)
        acquired_event = next(
            event for event in graph.events_since()
            if event["event_type"] == "resource.workload_lease_acquired")
        self.assertNotIn(secret, json.dumps(acquired_event["payload"]))
        self.assertIn("enforcement_sha256", acquired_event["payload"])

    def test_control_lane_bypasses_saturation_but_cannot_launch_work(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "control")
        ordinary = self.controller(
            graph, budget(concurrency_slots=1), runtime_id="runtime-control")
        filled = ordinary.acquire(
            ResourceClaim(cpu_cores=0.1, ram_mib=1),
            step_id="step-fill", attempt_id="attempt-fill",
            worker_id="worker-fill", snapshot=snapshot(), now=T0)
        self.assertTrue(filled.admitted)

        def sampler_must_not_run():
            raise AssertionError("control admission consulted live sampler")

        controller = self.controller(
            graph, budget(concurrency_slots=1), runtime_id="runtime-control",
            snapshot_provider=sampler_must_not_run)
        control = ResourceClaim(
            cpu_cores=0, ram_mib=0, vram_mib=0, accelerator="none",
            network=False, concurrency_slots=1, latency_class="control")
        admitted = controller.acquire(
            control, step_id="step-terminate", attempt_id="attempt-terminate",
            worker_id="worker-terminate", now=T0)
        invalid = controller.acquire(
            control.model_copy(update={"cpu_cores": 0.1}),
            step_id="step-fake-control", attempt_id="attempt-fake-control",
            worker_id="worker-fake-control", now=T0)
        second_control = controller.acquire(
            control, step_id="step-second-control",
            attempt_id="attempt-second-control",
            worker_id="worker-second-control", now=T0)
        workload = controller.acquire_workload(
            control, instance_id, {}, T0)

        self.assertTrue(admitted.admitted)
        self.assertLessEqual(
            datetime.fromisoformat(
                str(admitted.expires_at).replace("Z", "+00:00")),
            T0 + timedelta(seconds=15))
        self.assertTrue(controller.heartbeat(
            str(admitted.lease_id), "attempt-terminate",
            worker_id="worker-terminate", now=T0 + timedelta(seconds=1),
            lease_ttl_seconds=3_600))
        control_row = next(
            row for row in self.step_rows(graph)
            if row["lease_id"] == admitted.lease_id)
        self.assertLessEqual(
            datetime.fromisoformat(
                str(control_row["expires_at"]).replace("Z", "+00:00")),
            T0 + timedelta(seconds=16))
        self.assertEqual(invalid.status, "rejected")
        self.assertEqual(second_control.status, "deferred")
        self.assertEqual(second_control.reason, "control_lane_busy")
        self.assertEqual(workload.status, "rejected")
        self.assertTrue(controller.control_lane_allows("inspect", control))
        self.assertTrue(controller.control_lane_allows("terminate", control))
        self.assertFalse(controller.control_lane_allows("launch", control))
        self.assertFalse(controller.control_lane_allows(
            "terminate", control.model_copy(update={"network": True})))
        self.assertTrue(ordinary.release(
            str(filled.lease_id), "attempt-fill", worker_id="worker-fill",
            now=T0 + timedelta(seconds=1)))
        replacement = ordinary.acquire(
            ResourceClaim(cpu_cores=0.1, ram_mib=1),
            step_id="step-replacement", attempt_id="attempt-replacement",
            worker_id="worker-replacement",
            snapshot=snapshot(captured_at=T0 + timedelta(seconds=1)),
            now=T0 + timedelta(seconds=1))
        self.assertTrue(replacement.admitted)
        status = ordinary.status()
        self.assertEqual(status["active"]["leases"], 2)
        self.assertEqual(status["active"]["latency_classes"], {
            "interactive": 1,
            "control": 1,
        })

    def test_enforcement_must_be_a_bounded_json_object(self):
        graph = self.graph()
        instance_id = self.create_instance(graph, "invalid-enforcement")
        controller = self.controller(graph)
        claim = ResourceClaim(cpu_cores=0.1, ram_mib=1)

        for invalid in ("not json", "[]", {"bad": float("nan")}):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "enforcement_json"):
                    controller.acquire_workload(
                        claim, instance_id, invalid, T0,
                        snapshot=snapshot())
        self.assertEqual(self.workload_rows(graph), [])


if __name__ == "__main__":
    unittest.main()
