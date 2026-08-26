from __future__ import annotations

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


def abundant_snapshot(**overrides: object) -> ResourceSnapshot:
    values: dict[str, object] = {
        "available_cpu_millis": 64_000,
        "available_ram_mib": 128 * 1024,
        "available_network_slots": 64,
        "available_accelerator_vram_mib": {
            "cuda:0": 24 * 1024,
            "cuda:1": 24 * 1024,
        },
        "captured_at": T0,
    }
    values.update(overrides)
    return ResourceSnapshot(**values)


def roomy_budget(**overrides: object) -> AdmissionBudget:
    values: dict[str, object] = {
        "cpu_millis": 64_000,
        "ram_mib": 128 * 1024,
        "concurrency_slots": 64,
        "network_slots": 64,
        "accelerator_vram_mib": {
            "cuda:0": 24 * 1024,
            "cuda:1": 24 * 1024,
        },
    }
    values.update(overrides)
    return AdmissionBudget(**values)


class ResourceAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def graph(self, name: str = "friday.db") -> GraphStore:
        return GraphStore(Path(self.tmp.name) / name)

    @staticmethod
    def lease_rows(graph: GraphStore) -> list[dict[str, object]]:
        with graph._connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM resource_leases ORDER BY acquired_at, lease_id"
            ).fetchall()]

    @staticmethod
    def acquire(
        controller: ResourceAdmissionController,
        claim: ResourceClaim,
        suffix: str,
        *,
        now: datetime = T0,
        snapshot: ResourceSnapshot | None = None,
        lease_ttl_seconds: int | None = None,
    ):
        return controller.acquire(
            claim,
            step_id=f"step-{suffix}",
            attempt_id=f"attempt-{suffix}",
            worker_id=f"worker-{suffix}",
            now=now,
            snapshot=snapshot,
            lease_ttl_seconds=lease_ttl_seconds,
        )

    def test_static_over_budget_is_permanently_rejected_without_a_lease(self):
        graph = self.graph()
        controller = ResourceAdmissionController(
            graph,
            AdmissionBudget(
                cpu_millis=2_000,
                ram_mib=2_048,
                concurrency_slots=2,
                network_slots=1,
                accelerator_vram_mib={"cuda:0": 4_096},
            ),
        )

        decision = self.acquire(
            controller,
            ResourceClaim(cpu_cores=2.5, ram_mib=128),
            "too-large",
            snapshot=abundant_snapshot(),
        )

        self.assertEqual(decision.status, "rejected")
        self.assertFalse(decision.retryable)
        self.assertIsNone(decision.lease_id)
        self.assertIsNone(decision.expires_at)
        self.assertGreater(decision.deficits.get("cpu_millis", 0), 0)
        self.assertEqual(self.lease_rows(graph), [])

    def test_active_leases_enforce_every_shared_capacity_dimension(self):
        cases = {
            "global": (
                roomy_budget(concurrency_slots=1),
                ResourceClaim(cpu_cores=0.1, ram_mib=1),
                ResourceClaim(cpu_cores=0.1, ram_mib=1),
                "concurrency_slots",
            ),
            "cpu": (
                roomy_budget(cpu_millis=2_000),
                ResourceClaim(cpu_cores=1.2, ram_mib=1),
                ResourceClaim(cpu_cores=1.0, ram_mib=1),
                "cpu_millis",
            ),
            "ram": (
                roomy_budget(ram_mib=2_048),
                ResourceClaim(cpu_cores=0.1, ram_mib=1_200),
                ResourceClaim(cpu_cores=0.1, ram_mib=1_000),
                "ram_mib",
            ),
            "network": (
                roomy_budget(network_slots=1),
                ResourceClaim(cpu_cores=0.1, ram_mib=1, network=True),
                ResourceClaim(cpu_cores=0.1, ram_mib=1, network=True),
                "network_slots",
            ),
            "same-gpu": (
                roomy_budget(accelerator_vram_mib={"cuda:0": 4_096}),
                ResourceClaim(
                    cpu_cores=0.1, ram_mib=1, vram_mib=3_000,
                    accelerator="cuda",
                ),
                ResourceClaim(
                    cpu_cores=0.1, ram_mib=1, vram_mib=2_000,
                    accelerator="cuda:0",
                ),
                "vram_mib:cuda:0",
            ),
        }

        for index, (label, (budget, first_claim, second_claim,
                            deficit_key)) in enumerate(cases.items()):
            with self.subTest(dimension=label):
                graph = self.graph(f"conflict-{index}.db")
                controller = ResourceAdmissionController(graph, budget)

                first = self.acquire(
                    controller, first_claim, f"{label}-first",
                    snapshot=abundant_snapshot(),
                )
                second = self.acquire(
                    controller, second_claim, f"{label}-second",
                    snapshot=abundant_snapshot(),
                )

                self.assertEqual(first.status, "admitted")
                self.assertEqual(second.status, "deferred")
                self.assertTrue(second.retryable)
                self.assertIsNone(second.lease_id)
                self.assertGreater(second.deficits.get(deficit_key, 0), 0)
                self.assertEqual(len(self.lease_rows(graph)), 1)

    def test_allocations_on_different_gpus_are_independent(self):
        graph = self.graph()
        controller = ResourceAdmissionController(
            graph,
            roomy_budget(
                accelerator_vram_mib={"cuda:0": 4_096, "cuda:1": 4_096}
            ),
        )

        gpu_zero = self.acquire(
            controller,
            ResourceClaim(
                cpu_cores=0.1, ram_mib=1, vram_mib=3_500,
                accelerator="cuda:0",
            ),
            "gpu-zero",
            snapshot=abundant_snapshot(),
        )
        gpu_one = self.acquire(
            controller,
            ResourceClaim(
                cpu_cores=0.1, ram_mib=1, vram_mib=3_500,
                accelerator="cuda:1",
            ),
            "gpu-one",
            snapshot=abundant_snapshot(),
        )

        self.assertEqual(gpu_zero.status, "admitted")
        self.assertEqual(gpu_one.status, "admitted")
        self.assertNotEqual(gpu_zero.lease_id, gpu_one.lease_id)
        self.assertEqual(len(self.lease_rows(graph)), 2)

    def test_low_live_ram_or_vram_temporarily_defers_without_a_lease(self):
        cases = (
            (
                "ram",
                ResourceClaim(cpu_cores=0.1, ram_mib=1_024),
                abundant_snapshot(available_ram_mib=512),
                "ram_mib",
            ),
            (
                "vram",
                ResourceClaim(
                    cpu_cores=0.1, ram_mib=1, vram_mib=2_048,
                    accelerator="cuda:0",
                ),
                abundant_snapshot(
                    available_accelerator_vram_mib={"cuda:0": 1_024}
                ),
                "vram_mib:cuda:0",
            ),
        )

        for index, (label, claim, snapshot, deficit_key) in enumerate(cases):
            with self.subTest(resource=label):
                graph = self.graph(f"live-{index}.db")
                controller = ResourceAdmissionController(graph, roomy_budget())

                decision = self.acquire(
                    controller, claim, f"live-{label}", snapshot=snapshot
                )

                self.assertEqual(decision.status, "deferred")
                self.assertTrue(decision.retryable)
                self.assertIsNone(decision.lease_id)
                self.assertGreater(decision.deficits.get(deficit_key, 0), 0)
                self.assertEqual(self.lease_rows(graph), [])

    def test_active_reservations_are_subtracted_from_lower_live_capacity(self):
        graph = self.graph("live-reservation-accounting.db")
        controller = ResourceAdmissionController(
            graph,
            AdmissionBudget(
                cpu_millis=8_000,
                ram_mib=8_192,
                concurrency_slots=8,
                network_slots=8,
                accelerator_vram_mib={"cuda:0": 8_192},
            ),
        )
        lower_live_capacity = ResourceSnapshot(
            available_cpu_millis=1_000,
            available_ram_mib=1_000,
            available_network_slots=1,
            available_accelerator_vram_mib={"cuda:0": 1_000},
            captured_at=T0,
        )
        claim = ResourceClaim(
            cpu_cores=0.6,
            ram_mib=600,
            vram_mib=600,
            accelerator="cuda:0",
            network=True,
        )

        first = self.acquire(
            controller,
            claim,
            "live-reservation-first",
            snapshot=lower_live_capacity,
        )
        blocked = self.acquire(
            controller,
            claim,
            "live-reservation-blocked",
            snapshot=lower_live_capacity,
        )

        self.assertEqual(first.status, "admitted")
        self.assertEqual(blocked.status, "deferred")
        self.assertTrue(blocked.retryable)
        self.assertIsNone(blocked.lease_id)
        for key in (
            "cpu_millis",
            "ram_mib",
            "network_slots",
            "vram_mib:cuda:0",
        ):
            self.assertGreater(blocked.deficits.get(key, 0), 0)
        self.assertEqual(len(self.lease_rows(graph)), 1)

    def test_stale_and_far_future_snapshots_fail_closed_before_a_lease(self):
        instant = datetime.now(UTC)
        cases = (
            ("stale", instant - timedelta(hours=1)),
            ("far-future", instant + timedelta(hours=1)),
        )
        for source_index, source in enumerate(("explicit", "provider")):
            for time_index, (label, captured_at) in enumerate(cases):
                with self.subTest(source=source, timestamp=label):
                    graph = self.graph(
                        f"untrusted-snapshot-{source_index}-{time_index}.db"
                    )
                    snapshot = abundant_snapshot(captured_at=captured_at)
                    provider = (
                        (lambda supplied=snapshot: supplied)
                        if source == "provider"
                        else None
                    )
                    controller = ResourceAdmissionController(
                        graph,
                        roomy_budget(),
                        snapshot_provider=provider,
                        snapshot_ttl_seconds=2,
                    )

                    with self.assertRaisesRegex(
                        RuntimeError, "(?i)(?:snapshot|telemetry)"
                    ):
                        self.acquire(
                            controller,
                            ResourceClaim(cpu_cores=0.1, ram_mib=1),
                            f"{source}-{label}",
                            now=instant,
                            snapshot=(snapshot if source == "explicit" else None),
                        )

                    self.assertEqual(self.lease_rows(graph), [])

    def test_noninteractive_work_preserves_final_interactive_slots(self):
        for index, latency_class in enumerate(("background", "batch")):
            with self.subTest(latency_class=latency_class):
                graph = self.graph(f"interactive-reserve-{index}.db")
                controller = ResourceAdmissionController(
                    graph,
                    roomy_budget(concurrency_slots=2, network_slots=2),
                )
                noninteractive = ResourceClaim(
                    cpu_cores=0.1,
                    ram_mib=1,
                    network=True,
                    latency_class=latency_class,
                )

                first = self.acquire(
                    controller,
                    noninteractive,
                    f"{latency_class}-first",
                    snapshot=abundant_snapshot(),
                )
                blocked = self.acquire(
                    controller,
                    noninteractive,
                    f"{latency_class}-blocked",
                    snapshot=abundant_snapshot(),
                )
                interactive = self.acquire(
                    controller,
                    ResourceClaim(
                        cpu_cores=0.1,
                        ram_mib=1,
                        network=True,
                        latency_class="interactive",
                    ),
                    f"{latency_class}-interactive",
                    snapshot=abundant_snapshot(),
                )

                self.assertEqual(first.status, "admitted")
                self.assertEqual(blocked.status, "deferred")
                self.assertTrue(blocked.retryable)
                self.assertIsNone(blocked.lease_id)
                self.assertGreater(
                    blocked.deficits.get("concurrency_slots", 0), 0
                )
                self.assertGreater(blocked.deficits.get("network_slots", 0), 0)
                self.assertEqual(interactive.status, "admitted")
                self.assertIsNotNone(interactive.lease_id)
                self.assertEqual(len(self.lease_rows(graph)), 2)

    def test_lease_persists_and_safely_reports_profile_and_latency_binding(self):
        graph = self.graph("binding-observability.db")
        fingerprint = "a" * 64
        controller = ResourceAdmissionController(
            graph,
            roomy_budget(),
            runtime_id="runtime-profile-binding",
            profile_fingerprint=fingerprint,
        )
        claim = ResourceClaim(
            cpu_cores=1.25,
            ram_mib=768,
            network=True,
            latency_class="background",
        )

        admitted = self.acquire(
            controller,
            claim,
            "bound-observability",
            snapshot=abundant_snapshot(),
        )

        self.assertEqual(admitted.status, "admitted")
        rows = self.lease_rows(graph)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["profile_fingerprint"], fingerprint)
        self.assertEqual(rows[0]["latency_class"], "background")

        acquired_event = next(
            event for event in graph.events_since()
            if event["event_type"] == "resource.lease_acquired"
        )
        self.assertEqual(acquired_event["payload"]["profile_fingerprint"],
                         fingerprint)
        self.assertEqual(acquired_event["payload"]["latency_class"],
                         "background")
        self.assertNotIn("args", acquired_event["payload"])

        status = controller.status()
        self.assertEqual(status["profile_fingerprint"], fingerprint)
        self.assertEqual(status["active"]["latency_classes"], {
            "background": 1,
        })
        # Status is aggregate telemetry; durable execution identities and tool
        # arguments must remain outside this endpoint-facing representation.
        rendered = str(status)
        self.assertNotIn("step-bound-observability", rendered)
        self.assertNotIn("attempt-bound-observability", rendered)
        self.assertNotIn("worker-bound-observability", rendered)
        self.assertNotIn("args", status)

    def test_exact_reacquire_rejects_a_different_profile_binding(self):
        graph = self.graph("profile-fence.db")
        runtime_id = "runtime-same-profile-fence"
        first_controller = ResourceAdmissionController(
            graph,
            roomy_budget(),
            runtime_id=runtime_id,
            profile_fingerprint="a" * 64,
        )
        changed_controller = ResourceAdmissionController(
            graph,
            roomy_budget(),
            runtime_id=runtime_id,
            profile_fingerprint="b" * 64,
        )
        claim = ResourceClaim(cpu_cores=0.5, ram_mib=256)
        acquire_args = {
            "step_id": "step-profile-fence",
            "attempt_id": "attempt-profile-fence",
            "worker_id": "worker-profile-fence",
            "now": T0,
            "snapshot": abundant_snapshot(),
        }

        admitted = first_controller.acquire(claim, **acquire_args)
        exact_reacquire = first_controller.acquire(claim, **acquire_args)
        changed_profile = changed_controller.acquire(claim, **acquire_args)

        self.assertEqual(admitted.status, "admitted")
        self.assertEqual(exact_reacquire.status, "admitted")
        self.assertEqual(exact_reacquire.reason, "already_admitted")
        self.assertEqual(changed_profile.status, "rejected")
        self.assertFalse(changed_profile.retryable)
        self.assertEqual(changed_profile.reason, "attempt_already_fenced")
        rows = self.lease_rows(graph)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["profile_fingerprint"], "a" * 64)

    def test_exact_reacquire_rejects_a_different_latency_binding(self):
        graph = self.graph("latency-fence.db")
        controller = ResourceAdmissionController(
            graph,
            roomy_budget(),
            runtime_id="runtime-same-latency-fence",
            profile_fingerprint="c" * 64,
        )
        interactive = ResourceClaim(
            cpu_cores=0.5,
            ram_mib=256,
            latency_class="interactive",
        )
        changed_latency = ResourceClaim(
            cpu_cores=0.5,
            ram_mib=256,
            latency_class="batch",
        )
        acquire_args = {
            "step_id": "step-latency-fence",
            "attempt_id": "attempt-latency-fence",
            "worker_id": "worker-latency-fence",
            "now": T0,
            "snapshot": abundant_snapshot(),
        }

        admitted = controller.acquire(interactive, **acquire_args)
        exact_reacquire = controller.acquire(interactive, **acquire_args)
        changed = controller.acquire(changed_latency, **acquire_args)

        self.assertEqual(admitted.status, "admitted")
        self.assertEqual(exact_reacquire.status, "admitted")
        self.assertEqual(exact_reacquire.reason, "already_admitted")
        self.assertEqual(changed.status, "rejected")
        self.assertFalse(changed.retryable)
        self.assertEqual(changed.reason, "attempt_already_fenced")
        rows = self.lease_rows(graph)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["latency_class"], "interactive")

    def test_concurrent_controllers_racing_for_one_slot_create_one_lease(self):
        graph = self.graph()
        budget = roomy_budget(concurrency_slots=1)
        first_controller = ResourceAdmissionController(graph, budget)
        second_controller = ResourceAdmissionController(graph, budget)
        ready = threading.Barrier(2)

        def race(index: int):
            ready.wait(timeout=5)
            return self.acquire(
                (first_controller, second_controller)[index],
                ResourceClaim(cpu_cores=0.1, ram_mib=1),
                f"race-{index}",
                snapshot=abundant_snapshot(),
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            decisions = list(pool.map(race, (0, 1)))

        self.assertEqual(
            sorted(decision.status for decision in decisions),
            ["admitted", "deferred"],
        )
        winner = next(
            decision for decision in decisions if decision.status == "admitted"
        )
        self.assertIsNotNone(winner.lease_id)
        self.assertEqual(len(self.lease_rows(graph)), 1)

    def test_heartbeat_and_release_are_exactly_fenced(self):
        graph = self.graph()
        controller = ResourceAdmissionController(
            graph, roomy_budget(), lease_ttl_seconds=10
        )
        admitted = self.acquire(
            controller,
            ResourceClaim(cpu_cores=0.1, ram_mib=1),
            "fenced",
            snapshot=abundant_snapshot(),
        )
        self.assertEqual(admitted.status, "admitted")
        lease_id = admitted.lease_id
        self.assertIsNotNone(lease_id)

        self.assertFalse(controller.heartbeat(
            lease_id, "attempt-wrong", worker_id="worker-fenced",
            now=T0 + timedelta(seconds=1),
        ))
        self.assertFalse(controller.heartbeat(
            lease_id, "attempt-fenced", worker_id="worker-wrong",
            now=T0 + timedelta(seconds=1),
        ))
        self.assertTrue(controller.heartbeat(
            lease_id, "attempt-fenced", worker_id="worker-fenced",
            now=T0 + timedelta(seconds=1), lease_ttl_seconds=30,
        ))
        with graph._connect() as conn:
            row = conn.execute(
                "SELECT expires_at,status FROM resource_leases WHERE lease_id=?",
                (lease_id,),
            ).fetchone()
        self.assertEqual(row["status"], "active")
        self.assertGreater(
            datetime.fromisoformat(str(row["expires_at"]).replace("Z", "+00:00")),
            T0 + timedelta(seconds=10),
        )

        self.assertFalse(controller.release(
            lease_id, "attempt-wrong", worker_id="worker-fenced",
            now=T0 + timedelta(seconds=2),
        ))
        self.assertFalse(controller.release(
            lease_id, "attempt-fenced", worker_id="worker-wrong",
            now=T0 + timedelta(seconds=2),
        ))
        self.assertTrue(controller.release(
            lease_id, "attempt-fenced", worker_id="worker-fenced",
            reason="completed", now=T0 + timedelta(seconds=2),
        ))
        self.assertFalse(controller.release(
            lease_id, "attempt-fenced", worker_id="worker-fenced",
            now=T0 + timedelta(seconds=3),
        ))
        self.assertFalse(controller.heartbeat(
            lease_id, "attempt-fenced", worker_id="worker-fenced",
            now=T0 + timedelta(seconds=3),
        ))

    def test_lifecycle_calls_are_fenced_by_profile_fingerprint(self):
        graph = self.graph("lifecycle-profile-fence.db")
        runtime_id = "runtime-shared-profile-fence"
        owner = ResourceAdmissionController(
            graph,
            roomy_budget(),
            runtime_id=runtime_id,
            profile_fingerprint="a" * 64,
            lease_ttl_seconds=10,
        )
        changed_profile = ResourceAdmissionController(
            graph,
            roomy_budget(),
            runtime_id=runtime_id,
            profile_fingerprint="b" * 64,
            lease_ttl_seconds=10,
        )
        admitted = self.acquire(
            owner,
            ResourceClaim(cpu_cores=0.1, ram_mib=1),
            "lifecycle-profile",
            snapshot=abundant_snapshot(),
        )
        self.assertEqual(admitted.status, "admitted")
        lease_id = admitted.lease_id
        self.assertIsNotNone(lease_id)
        before = self.lease_rows(graph)[0]

        self.assertFalse(changed_profile.heartbeat(
            lease_id,
            "attempt-lifecycle-profile",
            worker_id="worker-lifecycle-profile",
            now=T0 + timedelta(seconds=1),
            lease_ttl_seconds=120,
        ))
        self.assertFalse(changed_profile.release(
            lease_id,
            "attempt-lifecycle-profile",
            worker_id="worker-lifecycle-profile",
            now=T0 + timedelta(seconds=1),
        ))

        after = self.lease_rows(graph)[0]
        for field in (
            "status",
            "heartbeat_at",
            "expires_at",
            "released_at",
            "release_reason",
            "last_event_seq",
        ):
            self.assertEqual(after[field], before[field])
        self.assertTrue(owner.heartbeat(
            lease_id,
            "attempt-lifecycle-profile",
            worker_id="worker-lifecycle-profile",
            now=T0 + timedelta(seconds=2),
        ))
        self.assertTrue(owner.release(
            lease_id,
            "attempt-lifecycle-profile",
            worker_id="worker-lifecycle-profile",
            now=T0 + timedelta(seconds=3),
        ))

    def test_expired_runtime_lease_is_reaped_and_capacity_reused(self):
        graph = self.graph()
        controller = ResourceAdmissionController(
            graph, roomy_budget(concurrency_slots=1), lease_ttl_seconds=5
        )
        stale = self.acquire(
            controller,
            ResourceClaim(cpu_cores=0.1, ram_mib=1),
            "stale-runtime",
            snapshot=abundant_snapshot(),
        )
        self.assertEqual(stale.status, "admitted")

        reaped = controller.reap_stale(now=T0 + timedelta(seconds=6))

        self.assertEqual(reaped, 1)
        with graph._connect() as conn:
            row = conn.execute(
                "SELECT status FROM resource_leases WHERE lease_id=?",
                (stale.lease_id,),
            ).fetchone()
        self.assertEqual(row["status"], "expired")
        replacement = self.acquire(
            controller,
            ResourceClaim(cpu_cores=0.1, ram_mib=1),
            "replacement-runtime",
            now=T0 + timedelta(seconds=6),
            snapshot=abundant_snapshot(captured_at=T0 + timedelta(seconds=6)),
        )
        self.assertEqual(replacement.status, "admitted")
        self.assertNotEqual(replacement.lease_id, stale.lease_id)

    def test_dead_runtime_reaping_is_scoped_and_frees_only_its_leases(self):
        graph = self.graph()
        budget = roomy_budget(concurrency_slots=2)
        dead_controller = ResourceAdmissionController(
            graph, budget, runtime_id="runtime-dead"
        )
        live_controller = ResourceAdmissionController(
            graph, budget, runtime_id="runtime-live"
        )
        dead = self.acquire(
            dead_controller,
            ResourceClaim(cpu_cores=0.1, ram_mib=1),
            "dead-owner",
            snapshot=abundant_snapshot(),
        )
        live = self.acquire(
            live_controller,
            ResourceClaim(cpu_cores=0.1, ram_mib=1),
            "live-owner",
            snapshot=abundant_snapshot(),
        )
        self.assertEqual((dead.status, live.status), ("admitted", "admitted"))

        reaped = live_controller.reap_runtime(
            "runtime-dead", now=T0 + timedelta(seconds=1)
        )

        self.assertEqual(reaped, 1)
        with graph._connect() as conn:
            rows = {
                row["lease_id"]: dict(row)
                for row in conn.execute(
                    "SELECT lease_id,runtime_id,status FROM resource_leases"
                ).fetchall()
            }
        self.assertNotEqual(rows[dead.lease_id]["status"], "active")
        self.assertEqual(rows[live.lease_id]["status"], "active")
        self.assertEqual(rows[live.lease_id]["runtime_id"], "runtime-live")
        self.assertTrue(live_controller.heartbeat(
            live.lease_id,
            "attempt-live-owner",
            worker_id="worker-live-owner",
            now=T0 + timedelta(seconds=2),
        ))
        replacement = self.acquire(
            live_controller,
            ResourceClaim(cpu_cores=0.1, ram_mib=1),
            "dead-slot-replacement",
            now=T0 + timedelta(seconds=2),
            snapshot=abundant_snapshot(captured_at=T0 + timedelta(seconds=2)),
        )
        self.assertEqual(replacement.status, "admitted")


if __name__ == "__main__":
    unittest.main()
