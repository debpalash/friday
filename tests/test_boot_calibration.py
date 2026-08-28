import json
import os
import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import supervisor
import friday_core.calibration as calibration
from friday_core.calibration import (
    BOOT_STABILITY_SECONDS, BootCalibrationEvidence, BootRecoveryStore,
    LastKnownGoodStore, PendingCalibrationStore, PerformanceCalibrationStore,
    PerformancePortfolioStore, match_active_candidate,
    runtime_benchmark_candidates, runtime_boot_candidates,
)
from friday_core.hardware import (
    GIB, Accelerator, HardwareSnapshot, select_runtime_profile,
)


def snapshot(*memory_gib: int, cpu_count: int = 32,
             free_gib: tuple[int, ...] | None = None) -> HardwareSnapshot:
    free = free_gib or memory_gib
    return HardwareSnapshot(
        cpu_count=cpu_count,
        system_memory_bytes=64 * GIB,
        accelerators=tuple(
            Accelerator("cuda", index, f"GPU {index}", size * GIB,
                        free[index] * GIB)
            for index, size in enumerate(memory_gib)),
        cuda_probe="available",
    )


def evidence(context: int) -> BootCalibrationEvidence:
    return BootCalibrationEvidence(
        startup_ms=12_345,
        identity_probe_ms=11,
        tokenization_probe_ms=17,
        observed_context_tokens=context,
    )


def isolated_supervisor_state(root: Path) -> dict:
    return {
        "STATE": root,
        "FRIDAY_PID": root / "friday.pid",
        "QWEN_PID": root / "qwen.pid",
        "SUPERVISOR_PID": root / "supervisor.pid",
        "LIFECYCLE_LOCK": root / "lifecycle.lock",
        "FRIDAY_START_LOCK": root / "friday-start.lock",
        "QWEN_START_LOCK": root / "qwen-start.lock",
        "RUNTIME_PROFILE_FILE": root / "runtime-resolved.json",
        "QWEN_RUNTIME_BINDING_FILE": root / "qwen-runtime-binding.json",
        "LAST_KNOWN_GOOD_FILE": root / "runtime-last-known-good.json",
        "PENDING_CALIBRATION_FILE": root / "runtime-calibration-pending.json",
        "BOOT_RECOVERY_FILE": root / "runtime-boot-recovery.json",
        "PERFORMANCE_CALIBRATION_FILE": root / "runtime-performance.json",
        "PERFORMANCE_PORTFOLIO_FILE":
            root / "runtime-performance-portfolio.json",
        "FRIDAY_RUNTIME_FINGERPRINT_FILE": root / "friday-runtime-fingerprint",
    }


class IsolatedSupervisorStateTestCase(unittest.TestCase):
    """Fail-safe fixture for tests that exercise supervisor state writes."""

    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        patcher = mock.patch.multiple(
            supervisor, REPO=root / "repo", QWEN=root / "qwen",
            **isolated_supervisor_state(root))
        patcher.start()
        self.addCleanup(patcher.stop)


class LastKnownGoodCalibrationTests(unittest.TestCase):
    def test_native_vision_record_requires_scene_and_vram_evidence(self):
        profile = select_runtime_profile(snapshot(48), environment={})
        self.assertTrue(profile.native_vision_enabled)
        incomplete = evidence(profile.context_tokens)
        complete = BootCalibrationEvidence(
            startup_ms=12_345,
            identity_probe_ms=11,
            tokenization_probe_ms=17,
            observed_context_tokens=profile.context_tokens,
            native_vision_required=True,
            native_vision_verified=True,
            native_vision_score_verified=True,
            native_vision_probe_ms=250,
            native_vision_vram_mib=25_000,
            native_vision_vram_verified=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = LastKnownGoodStore(Path(temporary) / "lkg.json")
            with self.assertRaisesRegex(ValueError, "evidence"):
                store.record(profile, incomplete, now=100)
            self.assertTrue(store.record(profile, complete, now=100))
            self.assertEqual(store.resolve(profile, now=101).status, "usable")

    def test_hardware_fingerprint_ignores_load_but_binds_topology(self):
        busy = snapshot(24, free_gib=(1,))
        idle = snapshot(24, free_gib=(23,))
        larger = snapshot(48, free_gib=(23,))

        self.assertEqual(busy.fingerprint, idle.fingerprint)
        self.assertNotEqual(busy.fingerprint, larger.fingerprint)
        first_board = HardwareSnapshot(
            cpu_count=32, system_memory_bytes=64 * GIB,
            accelerators=(Accelerator(
                "cuda", 0, "same model", 24 * GIB, 20 * GIB, "GPU-A"),),
            cuda_probe="available")
        replacement_board = HardwareSnapshot(
            cpu_count=32, system_memory_bytes=64 * GIB,
            accelerators=(Accelerator(
                "cuda", 0, "same model", 24 * GIB, 20 * GIB, "GPU-B"),),
            cuda_probe="available")
        self.assertNotEqual(first_board.fingerprint,
                            replacement_board.fingerprint)

    def test_private_atomic_record_round_trips_exact_degraded_profile(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        degraded = runtime_boot_candidates(proposed)[1]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last-known-good.json"
            store = LastKnownGoodStore(path)

            self.assertTrue(store.record(degraded, evidence(200_000), now=100))
            resolved = store.resolve(proposed, now=101)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(resolved.status, "usable")
            self.assertEqual(resolved.profile.fingerprint, degraded.fingerprint)
            self.assertEqual(resolved.profile.source, "last-known-good")
            self.assertEqual(list(path.parent.glob(f".{path.name}.*")), [])

    def test_record_is_bound_to_hardware_family_profile_and_verified_evidence(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        degraded = runtime_boot_candidates(proposed)[1]
        different_hardware = select_runtime_profile(
            snapshot(24, cpu_count=48), environment={})
        different_family = select_runtime_profile(snapshot(24), environment={
            "FRIDAY_LOCAL_MODEL": "different-model",
        })
        with tempfile.TemporaryDirectory() as temporary:
            store = LastKnownGoodStore(Path(temporary) / "lkg.json")
            store.record(degraded, evidence(200_000), now=100)

            self.assertEqual(
                store.resolve(different_hardware, now=101).status,
                "hardware_mismatch")
            self.assertEqual(
                store.resolve(different_family, now=101).status,
                "overrides_active")
            with self.assertRaisesRegex(ValueError, "evidence"):
                store.record(degraded, BootCalibrationEvidence(
                    startup_ms=1, identity_probe_ms=1,
                    tokenization_probe_ms=1,
                    observed_context_tokens=1024), now=102)

    def test_insecure_or_tampered_record_fails_closed(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        degraded = runtime_boot_candidates(proposed)[1]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lkg.json"
            store = LastKnownGoodStore(path)
            store.record(degraded, evidence(200_000), now=100)
            path.chmod(0o644)
            self.assertEqual(store.resolve(proposed, now=101).status, "insecure")

            path.chmod(0o600)
            body = json.loads(path.read_text())
            body["tuning"]["context_tokens"] = 32_768
            path.write_text(json.dumps(body))
            path.chmod(0o600)
            self.assertEqual(
                store.resolve(proposed, now=101).status,
                "profile_fingerprint_mismatch")

    def test_explicit_overrides_disable_reuse_and_automatic_degradation(self):
        automatic = select_runtime_profile(snapshot(24), environment={})
        explicit = select_runtime_profile(snapshot(24), environment={
            "FRIDAY_MODEL_CONTEXT_TOKENS": "131072",
        })
        with tempfile.TemporaryDirectory() as temporary:
            store = LastKnownGoodStore(Path(temporary) / "lkg.json")
            store.record(automatic, evidence(200_000), now=100)

            resolution = store.resolve(explicit, now=101)
            candidates = runtime_boot_candidates(explicit, automatic)

        self.assertEqual(resolution.status, "overrides_active")
        self.assertEqual(candidates, (explicit,))

        launcher_override = select_runtime_profile(snapshot(24), environment={
            "FRIDAY_LLM_EXTRA_ARGS": "--enforce-eager",
        })
        self.assertIn("FRIDAY_LLM_EXTRA_ARGS", launcher_override.overrides)
        self.assertEqual(runtime_boot_candidates(launcher_override),
                         (launcher_override,))

    def test_degradation_ladder_is_bounded_monotonic_and_recognizable(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        candidates = runtime_boot_candidates(proposed)

        self.assertEqual(len(candidates), 3)
        self.assertIs(candidates[0], proposed)
        self.assertEqual(
            [item.context_tokens for item in candidates],
            [200_000, 131_072, 65_536])
        self.assertEqual([item.max_sequences for item in candidates], [8, 4, 2])
        active = {"fingerprint": candidates[1].fingerprint}
        self.assertEqual(match_active_candidate(candidates, active), candidates[1])

        lowest_lkg = candidates[-1]
        with_lkg = runtime_boot_candidates(proposed, lowest_lkg)
        self.assertEqual(
            [item.context_tokens for item in with_lkg],
            [200_000, 65_536, 32_768])
        self.assertEqual([item.max_sequences for item in with_lkg], [8, 2, 1])

        memory_reduced_lkg = replace(
            candidates[1], gpu_memory_utilization=0.80,
            llm_memory_budget_gib=19.2, unallocated_gpu_gib=4.8)
        after_memory_lkg = runtime_boot_candidates(
            proposed, memory_reduced_lkg)
        for previous, current in zip(
                after_memory_lkg, after_memory_lkg[1:]):
            self.assertLessEqual(
                current.gpu_memory_utilization,
                previous.gpu_memory_utilization)
            self.assertLessEqual(
                current.llm_memory_budget_gib,
                previous.llm_memory_budget_gib)
            self.assertGreaterEqual(
                current.unallocated_gpu_gib,
                previous.unallocated_gpu_gib)

    def test_candidate_limit_one_never_injects_a_last_known_good(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        last_known_good = runtime_boot_candidates(proposed)[1]

        candidates = runtime_boot_candidates(
            proposed, last_known_good, maximum=1)

        self.assertEqual(candidates, (proposed,))

    def test_memory_inconsistent_last_known_good_fails_closed(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        unsafe = replace(
            proposed, name="memory-inconsistent", context_tokens=131_072,
            max_sequences=4, cuda_graph_capture_size=16,
            llm_memory_budget_gib=10.0, unallocated_gpu_gib=15.0)
        with tempfile.TemporaryDirectory() as temporary:
            store = LastKnownGoodStore(Path(temporary) / "lkg.json")
            store.record(unsafe, evidence(200_000), now=100)

            resolution = store.resolve(proposed, now=101)

        self.assertEqual(resolution.status, "invalid")

    def test_pending_probation_preserves_stable_record_and_is_runtime_bound(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        higher, lower = runtime_boot_candidates(proposed)[1:]
        with tempfile.TemporaryDirectory() as temporary:
            stable = LastKnownGoodStore(Path(temporary) / "stable.json")
            pending = PendingCalibrationStore(Path(temporary) / "pending.json")
            stable.record(lower, evidence(200_000), now=100)
            pending.stage(
                higher, evidence(200_000), runtime_identity="runtime-a",
                now=101)

            before = stable.resolve(proposed, now=102)
            wrong_process = pending.promote(
                higher, stable, runtime_identity="runtime-b")
            promoted = pending.promote(
                higher, stable, runtime_identity="runtime-a")
            after = stable.resolve(proposed, now=103)

        self.assertEqual(before.profile.fingerprint, lower.fingerprint)
        self.assertFalse(wrong_process)
        self.assertTrue(promoted)
        self.assertEqual(after.profile.fingerprint, higher.fingerprint)

    def test_old_runtime_cannot_discard_same_profile_replacement_probation(self):
        profile = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            pending = PendingCalibrationStore(Path(temporary) / "pending.json")
            pending.stage(
                profile, evidence(200_000), runtime_identity="runtime-new")

            removed = pending.discard(
                profile.fingerprint, runtime_identity="runtime-old")

            self.assertFalse(removed)
            self.assertTrue(pending.path.exists())


class BootRecoveryTests(unittest.TestCase):
    def test_planned_stop_does_not_enter_crash_backoff(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        active = runtime_boot_candidates(proposed)[1]
        with tempfile.TemporaryDirectory() as temporary:
            store = BootRecoveryStore(Path(temporary) / "recovery.json")
            store.record_launch_failure(proposed, now=1)
            store.record_launch_success(
                proposed, active, runtime_identity="runtime-a", now=20)

            wrong = store.record_planned_stop(
                proposed, active, runtime_identity="runtime-b")
            stopped = store.record_planned_stop(
                proposed, active, runtime_identity="runtime-a")
            retry = store.observe(
                proposed, running=False, active=None, now=21)
            status = store.public_status(proposed, now=21)

        self.assertFalse(wrong)
        self.assertTrue(stopped)
        self.assertEqual(retry, 0)
        self.assertEqual(status, {
            "state": "ready",
            "consecutive_failures": 0,
            "retry_after_seconds": 0,
        })

    def test_signal_safe_planned_stop_uses_exact_persisted_identity(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            store = BootRecoveryStore(Path(temporary) / "recovery.json")
            store.record_launch_success(
                proposed, proposed, runtime_identity="runtime-a", now=20)

            wrong = store.record_planned_stop_identity(
                active_profile_fingerprint=proposed.fingerprint,
                runtime_identity="runtime-b")
            stopped = store.record_planned_stop_identity(
                active_profile_fingerprint=proposed.fingerprint,
                runtime_identity="runtime-a")
            status = store.public_status(proposed, now=21)

        self.assertFalse(wrong)
        self.assertTrue(stopped)
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["consecutive_failures"], 0)

    def test_early_runtime_loss_is_edge_triggered_and_exponentially_backed_off(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        active = runtime_boot_candidates(proposed)[1]
        with tempfile.TemporaryDirectory() as temporary:
            store = BootRecoveryStore(Path(temporary) / "recovery.json")
            store.record_launch_success(proposed, active, now=100)

            first_delay = store.observe(
                proposed, running=False, active=None, now=101)
            repeated_delay = store.observe(
                proposed, running=False, active=None, now=102)
            first_status = store.public_status(proposed, now=102)
            second_delay = store.record_launch_failure(proposed, now=116)
            second_status = store.public_status(proposed, now=116)

            self.assertEqual(stat.S_IMODE(store.path.stat().st_mode), 0o600)

        self.assertEqual(first_delay, 15)
        self.assertEqual(repeated_delay, 14)
        self.assertEqual(first_status["consecutive_failures"], 1)
        self.assertEqual(second_delay, 30)
        self.assertEqual(second_status["consecutive_failures"], 2)

    def test_stable_runtime_clears_failure_history(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            store = BootRecoveryStore(Path(temporary) / "recovery.json")
            store.record_launch_failure(proposed, now=1)
            store.record_launch_success(proposed, proposed, now=20)
            store.observe(
                proposed, running=True, active=proposed,
                now=20 + BOOT_STABILITY_SECONDS)
            status = store.public_status(
                proposed, now=20 + BOOT_STABILITY_SECONDS)

        self.assertEqual(status, {
            "state": "stable",
            "consecutive_failures": 0,
            "retry_after_seconds": 0,
        })

    def test_public_status_contains_no_paths_fingerprints_or_failure_details(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            lkg = LastKnownGoodStore(Path(temporary) / "secret-lkg.json")
            recovery = BootRecoveryStore(Path(temporary) / "secret-recovery.json")
            lkg.record(proposed, evidence(200_000), now=100)
            recovery.record_launch_failure(proposed, now=100)
            public = {
                "last_known_good": lkg.public_status(proposed, now=101),
                "recovery": recovery.public_status(proposed, now=101),
            }

        serialized = json.dumps(public)
        self.assertNotIn("secret-", serialized)
        self.assertNotIn(proposed.fingerprint, serialized)
        self.assertNotIn(proposed.hardware_fingerprint, serialized)
        self.assertNotIn(proposed.qwen_model, serialized)

    def test_malformed_matching_recovery_record_is_reported_without_crashing(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recovery.json"
            store = BootRecoveryStore(path)
            store.record_launch_failure(proposed, now=100)
            value = json.loads(path.read_text())
            value["consecutive_failures"] = "many"
            path.write_text(json.dumps(value))
            path.chmod(0o600)

            status = store.public_status(proposed, now=101)
            delay = store.observe(
                proposed, running=False, active=None, now=101)

        self.assertEqual(status["state"], "invalid")
        self.assertEqual(delay, 0)

    def test_reboot_discards_monotonic_backoff_from_the_prior_boot(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            store = BootRecoveryStore(Path(temporary) / "recovery.json")
            with mock.patch.object(
                    calibration, "_boot_session_id", return_value="boot-a"):
                store.record_launch_failure(proposed, now=100)
            with mock.patch.object(
                    calibration, "_boot_session_id", return_value="boot-b"):
                status = store.public_status(proposed, now=101)
                retry = store.observe(
                    proposed, running=False, active=None, now=101)

        self.assertEqual(status["state"], "new_profile")
        self.assertEqual(retry, 0)

    def test_clock_anomalies_are_clamped_and_restart_probation(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            store = BootRecoveryStore(Path(temporary) / "recovery.json")
            store.record_launch_failure(proposed, now=100)
            value = json.loads(store.path.read_text())
            value["next_retry_at"] = 100_000.0
            store.path.write_text(json.dumps(value))
            store.path.chmod(0o600)
            status = store.public_status(proposed, now=1)

            store.record_launch_success(
                proposed, proposed, runtime_identity="runtime-a", now=100)
            store.observe(
                proposed, running=True, active=proposed,
                runtime_identity="runtime-a", now=50)
            rebased = json.loads(store.path.read_text())

        self.assertEqual(status["retry_after_seconds"], 900)
        self.assertEqual(rebased["last_success_at"], 50)
        self.assertEqual(rebased["state"], "probation")

    def test_same_profile_replacement_gets_a_new_probation_window(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            store = BootRecoveryStore(Path(temporary) / "recovery.json")
            store.record_launch_success(
                proposed, proposed, runtime_identity="runtime-a", now=1)
            store.observe(
                proposed, running=True, active=proposed,
                runtime_identity="runtime-b", now=500)
            status = store.public_status(proposed, now=501)

        self.assertEqual(status["state"], "probation")


class PerformanceCalibrationTests(unittest.TestCase):
    @staticmethod
    def samples() -> list[dict]:
        return [
            {"first_token_ms": 60.0, "completion_tokens": 256,
             "decode_tokens_per_second": 105.0, "total_ms": 2490.0},
            {"first_token_ms": 58.0, "completion_tokens": 256,
             "decode_tokens_per_second": 107.0, "total_ms": 2440.0},
            {"first_token_ms": 59.0, "completion_tokens": 256,
             "decode_tokens_per_second": 106.0, "total_ms": 2460.0},
        ]

    def test_record_is_private_profile_bound_and_publicly_aggregate(self):
        profile = select_runtime_profile(snapshot(24), environment={})
        larger = select_runtime_profile(snapshot(48), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "performance.json"
            store = PerformanceCalibrationStore(path)
            status = store.record(
                profile, runtime_identity="a" * 64,
                samples=self.samples(), qwen_vram_mib=22_550, now=100)
            mismatch = store.public_status(larger, now=101)
            serialized = json.dumps(status)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(status, {
                "state": "measured", "age_seconds": 0,
                "sample_count": 3, "median_first_token_ms": 59.0,
                "median_decode_tokens_per_second": 106.0,
                "qwen_vram_mib": 22_550,
            })
            self.assertEqual(mismatch, {"state": "new_profile"})
            self.assertNotIn(profile.fingerprint, serialized)
            self.assertNotIn(profile.hardware_fingerprint, serialized)
            self.assertNotIn("a" * 64, serialized)

    def test_tampered_or_expired_measurement_is_not_trusted(self):
        profile = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "performance.json"
            store = PerformanceCalibrationStore(path)
            store.record(
                profile, runtime_identity="b" * 64,
                samples=self.samples(), qwen_vram_mib=22_550, now=100)
            expired = store.public_status(
                profile,
                now=100 + calibration.MAX_PERFORMANCE_AGE_SECONDS + 1)
            value = json.loads(path.read_text())
            value["median_decode_tokens_per_second"] = 9999
            path.write_text(json.dumps(value))
            path.chmod(0o600)
            tampered = store.public_status(profile, now=101)

        self.assertEqual(expired, {"state": "expired"})
        self.assertEqual(tampered, {"state": "invalid"})

    def test_benchmark_candidates_are_exact_bounded_launcher_modes(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        candidates = runtime_benchmark_candidates(proposed)

        self.assertEqual(len(candidates), 3)
        self.assertEqual([item.kv_mode for item in candidates],
                         ["huge", "long", "fast"])
        self.assertEqual([item.context_tokens for item in candidates],
                         [200_000, 150_000, 65_536])
        self.assertEqual(candidates[0].fingerprint, proposed.fingerprint)
        self.assertEqual(len({item.fingerprint for item in candidates}), 3)
        self.assertTrue(all(
            item.hardware_fingerprint == proposed.hardware_fingerprint
            and item.family_fingerprint == proposed.family_fingerprint
            for item in candidates))

        overridden = select_runtime_profile(snapshot(24), environment={
            "FRIDAY_KV_MODE": "fast",
        })
        self.assertEqual(runtime_benchmark_candidates(overridden),
                         (overridden,))

    def test_portfolio_separates_reasoning_and_throughput_recommendations(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        candidates = runtime_benchmark_candidates(proposed)
        decode = {"huge": 108.0, "long": 118.0, "fast": 132.0}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "portfolio.json"
            store = PerformancePortfolioStore(path)
            for index, profile in enumerate(candidates):
                samples = [{
                    "first_token_ms": 55.0 + index,
                    "completion_tokens": 256,
                    "decode_tokens_per_second": decode[profile.kv_mode],
                    "total_ms": 2500.0,
                } for _ in range(3)]
                store.record(
                    profile, runtime_identity=str(index + 1) * 64,
                    samples=samples, qwen_vram_mib=22_500 + index,
                    now=100 + index)
            status = store.public_status(
                proposed, candidates, now=200)
            reasoning = store.recommend(
                proposed, candidates, preference="reasoning", now=200)
            throughput = store.recommend(
                proposed, candidates, preference="throughput", now=200)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(status["state"], "measured")
            self.assertEqual(status["profile_count"], 3)
            self.assertEqual(status["reasoning"]["kv_mode"], "huge")
            self.assertEqual(status["throughput"]["kv_mode"], "fast")
            self.assertEqual(reasoning.profile.kv_mode, "huge")
            self.assertEqual(throughput.profile.kv_mode, "fast")
            serialized = json.dumps(status)
            self.assertNotIn(proposed.fingerprint, serialized)
            self.assertNotIn("1" * 64, serialized)

    def test_portfolio_tamper_expiry_and_hardware_change_fail_closed(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        candidates = runtime_benchmark_candidates(proposed)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "portfolio.json"
            store = PerformancePortfolioStore(path)
            store.record(
                proposed, runtime_identity="a" * 64,
                samples=self.samples(), qwen_vram_mib=22_550, now=100)
            expired = store.public_status(
                proposed, candidates,
                now=100 + calibration.MAX_PERFORMANCE_AGE_SECONDS + 1)
            changed = select_runtime_profile(snapshot(48), environment={})
            mismatch = store.public_status(
                changed, runtime_benchmark_candidates(changed), now=101)
            value = json.loads(path.read_text())
            value["entries"][0]["median_decode_tokens_per_second"] = 9999
            path.write_text(json.dumps(value))
            path.chmod(0o600)
            tampered = store.public_status(proposed, candidates, now=101)

        self.assertEqual(expired, {"state": "expired"})
        self.assertEqual(mismatch, {"state": "new_machine"})
        self.assertEqual(tampered, {"state": "invalid"})

    def test_new_measurement_repairs_but_never_carries_tampered_tuning(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        candidates = runtime_benchmark_candidates(proposed)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "portfolio.json"
            store = PerformancePortfolioStore(path)
            store.record(
                candidates[0], runtime_identity="a" * 64,
                samples=self.samples(), qwen_vram_mib=22_550, now=100)
            store.record(
                candidates[1], runtime_identity="b" * 64,
                samples=self.samples(), qwen_vram_mib=22_000, now=101)
            damaged = json.loads(path.read_text())
            damaged["entries"][0]["tuning"].pop("asr_threads")
            path.write_text(json.dumps(damaged))
            path.chmod(0o600)

            store.record(
                candidates[2], runtime_identity="c" * 64,
                samples=self.samples(), qwen_vram_mib=21_500, now=102)
            repaired = json.loads(path.read_text())
            status = store.public_status(proposed, candidates, now=103)

        self.assertEqual(len(repaired["entries"]), 1)
        self.assertEqual(
            repaired["entries"][0]["profile_fingerprint"],
            candidates[2].fingerprint)
        self.assertEqual(status["state"], "measured")
        self.assertEqual(status["profile_count"], 1)

    def test_public_portfolio_status_uses_one_validated_snapshot(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        candidates = runtime_benchmark_candidates(proposed)
        with tempfile.TemporaryDirectory() as temporary:
            store = PerformancePortfolioStore(
                Path(temporary) / "portfolio.json")
            for index, profile in enumerate(candidates):
                store.record(
                    profile, runtime_identity=str(index + 1) * 64,
                    samples=self.samples(), qwen_vram_mib=22_000,
                    now=100 + index)
            with mock.patch.object(
                    calibration, "_read_private_json",
                    wraps=calibration._read_private_json) as read:
                status = store.public_status(
                    proposed, candidates, now=200)

        self.assertEqual(status["state"], "measured")
        self.assertEqual(read.call_count, 1)


class SupervisorBootCalibrationTests(IsolatedSupervisorStateTestCase):
    def test_profile_benchmark_rejects_bounds_before_any_lifecycle_effect(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        with mock.patch.object(
                supervisor, "_service_start_lock") as start_lock:
            with self.assertRaisesRegex(ValueError, "bounds"):
                supervisor.benchmark_qwen_performance_profiles(
                    proposed, sample_count=0, max_tokens=128)
        start_lock.assert_not_called()

    def test_profile_benchmark_candidates_are_anchored_to_active_fallback(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        original = runtime_boot_candidates(proposed)[1]
        with (mock.patch.object(
                  supervisor, "_service_start_lock",
                  return_value=__import__("contextlib").nullcontext()),
              mock.patch.object(supervisor, "verified_pid", return_value=100),
              mock.patch.object(
                  supervisor, "read_active_runtime_profile",
                  return_value=original.to_dict()),
              mock.patch.object(
                  supervisor, "active_runtime_process_matches",
                  return_value=True),
              mock.patch.object(supervisor, "_require_compatible_qwen"),
              mock.patch.object(
                  supervisor, "runtime_benchmark_candidates",
                  return_value=(original,)) as benchmark_candidates):
            with self.assertRaisesRegex(
                    supervisor.NonDegradableBootError,
                    "multiple automatic candidates"):
                supervisor.benchmark_qwen_performance_profiles(proposed)

        benchmark_candidates.assert_called_once()
        self.assertEqual(
            benchmark_candidates.call_args.args[0].fingerprint,
            original.fingerprint)

    def test_profile_benchmark_is_bounded_and_restores_exact_runtime(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        state = {
            "profile": proposed, "qwen": True, "friday": True,
            "qwen_pid": 100, "friday_pid": 200,
        }
        starts = []
        measurements = []

        def verified(path, *_args, **_kwargs):
            if path == supervisor.QWEN_PID:
                return state["qwen_pid"] if state["qwen"] else None
            if path == supervisor.FRIDAY_PID:
                return state["friday_pid"] if state["friday"] else None
            return None

        def manifest():
            return {"fingerprint": state["profile"].fingerprint}

        def stop(path, *_args, **_kwargs):
            if path == supervisor.QWEN_PID:
                state["qwen"] = False
            elif path == supervisor.FRIDAY_PID:
                state["friday"] = False

        def start_qwen(profile, **_kwargs):
            starts.append(profile.kv_mode)
            if profile.kv_mode == "long":
                raise RuntimeError("injected mode-specific boot failure")
            state["profile"] = profile
            state["qwen"] = True
            state["qwen_pid"] += 1
            return state["qwen_pid"]

        def calibrate(profile, **_kwargs):
            measurements.append(profile.kv_mode)
            return {
                "median_first_token_ms": 50.0,
                "median_decode_tokens_per_second": 100.0,
                "qwen_vram_mib": 22_000,
            }

        def start_friday(*, profile, **_kwargs):
            self.assertEqual(profile.fingerprint, proposed.fingerprint)
            self.assertTrue(state["qwen"])
            state["friday"] = True
            return state["friday_pid"]

        with (mock.patch.object(supervisor, "verified_pid", side_effect=verified),
              mock.patch.object(
                  supervisor, "read_active_runtime_profile",
                  side_effect=manifest),
              mock.patch.object(
                  supervisor, "active_runtime_process_matches",
                  side_effect=lambda value, _pid: (
                      state["qwen"] and value.get("fingerprint")
                      == state["profile"].fingerprint)),
              mock.patch.object(
                  supervisor, "_require_compatible_qwen"),
              mock.patch.object(
                  supervisor, "_read_friday_fingerprint",
                  return_value=proposed.fingerprint),
              mock.patch.object(
                  supervisor, "healthy", return_value=True),
              mock.patch.object(supervisor, "stop_pid", side_effect=stop),
              mock.patch.object(
                  supervisor, "_start_qwen_locked", side_effect=start_qwen),
              mock.patch.object(
                  supervisor, "calibrate_qwen_performance",
                  side_effect=calibrate),
              mock.patch.object(
                  supervisor, "start_friday", side_effect=start_friday),
              mock.patch.object(
                  supervisor, "_service_start_lock",
                  return_value=__import__("contextlib").nullcontext())):
            result = supervisor.benchmark_qwen_performance_profiles(
                proposed, sample_count=2, max_tokens=128)

        self.assertEqual(starts, ["long", "fast", "huge"])
        self.assertEqual(measurements, ["fast", "huge"])
        self.assertEqual(state["profile"].fingerprint, proposed.fingerprint)
        self.assertTrue(state["qwen"])
        self.assertTrue(state["friday"])
        self.assertTrue(result["original_profile_restored"])
        self.assertTrue(result["friday_restored"])
        self.assertFalse(result["automatic_promotion"])
        self.assertEqual(
            [item["status"] for item in result["reports"]],
            ["failed", "measured", "measured"])

    def test_performance_calibration_persists_both_evidence_views(self):
        profile = select_runtime_profile(snapshot(24), environment={})
        active = profile.to_dict()
        sample = {
            "first_token_ms": 50.0,
            "completion_tokens": 128,
            "decode_tokens_per_second": 100.0,
            "total_ms": 1320.0,
        }
        with (mock.patch.object(
                  supervisor, "build_qwen_environment",
                  return_value={"VLLM_API_KEY": "secret"}),
              mock.patch.object(
                  supervisor, "read_active_runtime_profile",
                  return_value=active),
              mock.patch.object(
                  supervisor, "active_runtime_identity",
                  return_value="d" * 64),
              mock.patch.object(
                  supervisor, "_qwen_generation_sample",
                  return_value=sample) as generation,
              mock.patch.object(
                  supervisor, "_qwen_process_vram_mib",
                  return_value=22_550)):
            status = supervisor.calibrate_qwen_performance(
                profile, expected_pid=42, sample_count=2, max_tokens=128)

        legacy = json.loads(
            supervisor.PERFORMANCE_CALIBRATION_FILE.read_text())
        portfolio = json.loads(
            supervisor.PERFORMANCE_PORTFOLIO_FILE.read_text())
        public_portfolio = PerformancePortfolioStore(
            supervisor.PERFORMANCE_PORTFOLIO_FILE).public_status(
                profile, runtime_benchmark_candidates(profile))
        serialized = json.dumps({"legacy": legacy, "portfolio": portfolio})

        self.assertEqual(status["state"], "measured")
        self.assertEqual(public_portfolio["state"], "measured")
        self.assertEqual(public_portfolio["profile_count"], 1)
        self.assertEqual(generation.call_count, 3)
        self.assertEqual(generation.call_args_list[0].kwargs["max_tokens"], 32)
        self.assertNotIn("secret", serialized)
        self.assertNotIn("Friday hardware calibration canary", serialized)

    def test_watch_survives_recovery_state_write_failure(self):
        profile = select_runtime_profile(snapshot(24), environment={})
        recovery = mock.MagicMock()
        recovery.observe.side_effect = OSError("state media unavailable")
        with mock.patch.object(
                supervisor, "BootRecoveryStore", return_value=recovery):
            updated = supervisor._update_runtime_calibration_state(
                profile, profile, runtime_identity="runtime-a")

        self.assertFalse(updated)

    def test_watch_survives_pending_promotion_cleanup_failure(self):
        profile = select_runtime_profile(snapshot(24), environment={})
        recovery = mock.MagicMock()
        recovery.public_status.return_value = {"state": "stable"}
        pending = mock.MagicMock()
        pending.promote.side_effect = OSError("unlink unavailable")
        with mock.patch.object(
                supervisor, "BootRecoveryStore", return_value=recovery), \
             mock.patch.object(
                 supervisor, "PendingCalibrationStore", return_value=pending):
            updated = supervisor._update_runtime_calibration_state(
                profile, profile, runtime_identity="runtime-a")

        self.assertFalse(updated)

    def test_calibration_requires_endpoint_to_reject_a_wrong_credential(self):
        profile = select_runtime_profile(snapshot(24), environment={})
        denied = supervisor.urllib.error.HTTPError(
            profile.local_base_url + "/models", 401, "denied", {}, None)
        with mock.patch.object(supervisor, "_credentialed_urlopen",
                               side_effect=denied), \
             mock.patch.object(
                 supervisor, "_qwen_listener_binding",
                 return_value=("inode-a", os.geteuid(), 100)):
            self.assertTrue(supervisor.qwen_api_rejects_invalid_credential(
                profile, expected_pid=42))

        accepted = mock.MagicMock()
        accepted.__enter__.return_value = accepted
        with mock.patch.object(supervisor, "_credentialed_urlopen",
                               return_value=accepted), \
             mock.patch.object(
                 supervisor, "_qwen_listener_binding",
                 return_value=("inode-a", os.geteuid(), 100)):
            self.assertFalse(supervisor.qwen_api_rejects_invalid_credential(
                profile, expected_pid=42))

    def test_generation_sample_measures_without_retaining_generated_text(self):
        profile = select_runtime_profile(snapshot(24), environment={})
        response = mock.MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        response.__iter__.return_value = iter([
            b'data: {"choices":[{"text":"private canary output"}]}\n',
            b'data: {"choices":[],"usage":{"completion_tokens":128}}\n',
            b'data: [DONE]\n',
        ])
        with mock.patch.object(
                supervisor, "_qwen_listener_binding",
                return_value=("inode-a", os.geteuid(), 100,
                              "user-ns", "net-ns")), \
             mock.patch.object(supervisor, "_credentialed_urlopen",
                               return_value=response), \
             mock.patch.object(
                 supervisor.time, "monotonic",
                 side_effect=[10.0, 10.05, 12.05]):
            sample = supervisor._qwen_generation_sample(
                profile, expected_pid=42, credential="secret",
                max_tokens=128, timeout=5)

        self.assertEqual(sample, {
            "first_token_ms": 50.0,
            "completion_tokens": 128,
            "decode_tokens_per_second": 63.5,
            "total_ms": 2050.0,
        })
        self.assertNotIn("text", sample)

    def test_vram_observation_counts_only_exact_qwen_process_group(self):
        result = mock.Mock(
            stdout="1001, 22000\n1002, 550\n9999, 9000\n")
        process_groups = {1001: 42, 1002: 42, 9999: 9999}
        with mock.patch.object(supervisor.subprocess, "run",
                               return_value=result), \
             mock.patch.object(
                 supervisor.os, "getpgid",
                 side_effect=lambda pid: process_groups[pid]):
            observed = supervisor._qwen_process_vram_mib(42)

        self.assertEqual(observed, 22_550)

    def test_authenticated_tokenization_canary_measures_and_proves_context(self):
        profile = select_runtime_profile(snapshot(24), environment={})
        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({
            "count": 5,
            "tokens": [1, 2, 3, 4, 5],
            "max_model_len": 200_000,
        }).encode()
        response.__enter__.return_value = response
        clock = iter([10.0, 10.011, 10.020, 10.037, 10.040])

        with mock.patch.object(supervisor, "build_qwen_environment",
                               return_value={"VLLM_API_KEY": "secret"}), \
             mock.patch.object(supervisor, "qwen_api_compatible",
                               return_value=True), \
             mock.patch.object(supervisor,
                               "qwen_api_rejects_invalid_credential",
                               return_value=True), \
             mock.patch.object(supervisor, "_credentialed_urlopen",
                               return_value=response) as urlopen, \
             mock.patch.object(
                 supervisor, "_qwen_listener_binding",
                 return_value=("inode-a", os.geteuid(), 100)), \
             mock.patch.object(supervisor.time, "monotonic",
                               side_effect=lambda: next(clock)):
            proof = supervisor.qwen_boot_calibration(
                profile, startup_started_at=9.0, expected_pid=42)

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:18021/tokenize")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertTrue(proof.authenticated)
        self.assertEqual(proof.observed_context_tokens, 200_000)
        self.assertGreater(proof.startup_ms, 0)

    def test_native_vision_boot_calibration_measures_scene_and_vram(self):
        profile = select_runtime_profile(snapshot(48), environment={})
        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({
            "count": 5,
            "tokens": [1, 2, 3, 4, 5],
            "max_model_len": profile.context_tokens,
        }).encode()
        response.__enter__.return_value = response
        ticks = iter([10.0, 10.01, 10.02, 10.03, 10.04, 10.29, 10.30])
        with mock.patch.object(supervisor, "build_qwen_environment",
                               return_value={"VLLM_API_KEY": "secret"}), \
             mock.patch.object(supervisor, "qwen_api_compatible",
                               return_value=True), \
             mock.patch.object(supervisor,
                               "qwen_api_rejects_invalid_credential",
                               return_value=True), \
             mock.patch.object(supervisor, "_credentialed_urlopen",
                               return_value=response), \
             mock.patch.object(supervisor, "_qwen_listener_binding",
                               return_value=("inode-a", os.geteuid(), 100)), \
             mock.patch.object(supervisor, "qwen_native_vision_compatible",
                               return_value=True) as vision, \
             mock.patch.object(supervisor, "qwen_native_vision_score",
                               return_value=True) as vision_score, \
             mock.patch.object(supervisor, "_qwen_process_vram_mib",
                               return_value=25_000), \
             mock.patch.object(supervisor.time, "monotonic",
                               side_effect=lambda: next(ticks)):
            proof = supervisor.qwen_boot_calibration(
                profile, startup_started_at=9.0, expected_pid=42)

        vision.assert_called_once_with(
            profile, expected_pid=42, timeout=30.0)
        vision_score.assert_called_once_with(
            profile, None, expected_pid=42, timeout=30.0)
        self.assertTrue(proof.native_vision_required)
        self.assertTrue(proof.native_vision_verified)
        self.assertTrue(proof.native_vision_score_verified)
        self.assertEqual(proof.native_vision_probe_ms, 250)
        self.assertEqual(proof.native_vision_vram_mib, 25_000)
        self.assertTrue(proof.native_vision_vram_verified)

    def test_tokenization_canary_requires_exact_candidate_context(self):
        profile = select_runtime_profile(snapshot(24), environment={})
        for observed_context in (profile.context_tokens - 1,
                                 profile.context_tokens + 1):
            with self.subTest(observed_context=observed_context):
                response = mock.MagicMock()
                response.status = 200
                response.read.return_value = json.dumps({
                    "count": 5,
                    "tokens": [1, 2, 3, 4, 5],
                    "max_model_len": observed_context,
                }).encode()
                response.__enter__.return_value = response
                with mock.patch.object(
                        supervisor, "build_qwen_environment",
                        return_value={"VLLM_API_KEY": "secret"}), \
                     mock.patch.object(
                         supervisor, "qwen_api_compatible",
                         return_value=True), \
                     mock.patch.object(
                         supervisor, "qwen_api_rejects_invalid_credential",
                         return_value=True), \
                     mock.patch.object(
                         supervisor, "_credentialed_urlopen",
                         return_value=response), \
                     mock.patch.object(
                         supervisor, "_qwen_listener_binding",
                         return_value=("inode-a", os.geteuid(), 100)):
                    with self.assertRaisesRegex(
                            supervisor.NonDegradableBootError,
                            "did not prove"):
                        supervisor.qwen_boot_calibration(
                            profile, startup_started_at=1.0,
                            expected_pid=42)

    def test_calibrated_start_tries_proposal_then_records_actual_fallback(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            calls = []

            def launch(candidate, *, calibration_evidence):
                calls.append(candidate)
                if len(calls) == 1:
                    raise supervisor.DegradableCapacityBootError(
                        "candidate failed")
                calibration_evidence.append(evidence(200_000))
                return 707

            with mock.patch.multiple(
                    supervisor,
                    **isolated_supervisor_state(state)), \
                 mock.patch.object(supervisor, "healthy", return_value=False), \
                 mock.patch.object(supervisor, "read_active_runtime_profile",
                                   return_value={}), \
                 mock.patch.object(supervisor, "active_runtime_identity",
                                   return_value="runtime-707"), \
                 mock.patch.object(supervisor, "_start_qwen_locked",
                                   side_effect=launch):
                pid, active, proof = supervisor.start_qwen_calibrated(proposed)

            stored = LastKnownGoodStore(
                state / "runtime-last-known-good.json").resolve(proposed)
            pending = PendingCalibrationStore(
                state / "runtime-calibration-pending.json").public_status(active)
            recovery = BootRecoveryStore(
                state / "runtime-boot-recovery.json").public_status(proposed)

        self.assertEqual(pid, 707)
        self.assertEqual(len(calls), 2)
        self.assertEqual(active.fingerprint, calls[1].fingerprint)
        self.assertNotEqual(active.fingerprint, proposed.fingerprint)
        self.assertEqual(proof.observed_context_tokens, 200_000)
        self.assertEqual(stored.status, "missing")
        self.assertEqual(pending["state"], "probation")
        self.assertEqual(recovery["state"], "probation")

    def test_adopted_healthy_process_is_probed_but_not_claimed_as_measured_boot(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        adopted_proof = BootCalibrationEvidence(
            startup_ms=0, identity_probe_ms=2, tokenization_probe_ms=3,
            observed_context_tokens=200_000, startup_measured=False)
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            with mock.patch.multiple(
                    supervisor,
                    **isolated_supervisor_state(state)), \
                 mock.patch.object(supervisor, "healthy", return_value=True), \
                 mock.patch.object(supervisor, "read_active_runtime_profile",
                                   return_value=proposed.to_dict()), \
                 mock.patch.object(supervisor, "verified_pid", return_value=808), \
                 mock.patch.object(supervisor,
                                   "active_runtime_process_matches",
                                   return_value=True), \
                 mock.patch.object(supervisor, "active_runtime_identity",
                                   return_value="runtime-808"), \
                 mock.patch.object(supervisor, "_require_compatible_qwen"), \
                 mock.patch.object(supervisor, "qwen_boot_calibration",
                                   return_value=adopted_proof):
                pid, active, proof = supervisor.start_qwen_calibrated(proposed)

            self.assertFalse((state / "runtime-last-known-good.json").exists())
            recovery = BootRecoveryStore(
                state / "runtime-boot-recovery.json").public_status(proposed)

        self.assertEqual(pid, 808)
        self.assertIs(active, proposed)
        self.assertFalse(proof.startup_measured)
        self.assertEqual(recovery["state"], "probation")

    def test_identity_failure_is_not_retried_as_a_capacity_problem(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            with mock.patch.multiple(
                    supervisor,
                    **isolated_supervisor_state(state)), \
                 mock.patch.object(supervisor, "healthy", return_value=False), \
                 mock.patch.object(
                     supervisor, "_start_qwen_locked",
                     side_effect=supervisor.NonDegradableBootError(
                         "identity rejected")) as launch:
                with self.assertRaisesRegex(RuntimeError, "1 of 3"):
                    supervisor.start_qwen_calibrated(proposed)

            recovery = BootRecoveryStore(
                state / "runtime-boot-recovery.json").public_status(proposed)

        self.assertEqual(launch.call_count, 1)
        self.assertEqual(recovery["consecutive_failures"], 1)

    def test_status_names_verified_fallback_without_relabeling_proposal(self):
        proposed = select_runtime_profile(snapshot(24), environment={})
        fallback = runtime_boot_candidates(proposed)[1]
        active_manifest = fallback.to_dict()
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.multiple(
                 supervisor,
                 **isolated_supervisor_state(Path(temporary))), \
             mock.patch.object(supervisor, "read_active_runtime_profile",
                               return_value=active_manifest), \
             mock.patch.object(supervisor, "_read_friday_fingerprint",
                               return_value=fallback.fingerprint), \
             mock.patch.object(supervisor, "verified_pid", return_value=123), \
             mock.patch.object(supervisor, "active_runtime_process_matches",
                               return_value=True), \
             mock.patch.object(
                 supervisor, "_qwen_listener_binding",
                 return_value=("inode-a", os.geteuid(), 100)), \
             mock.patch.object(supervisor, "healthy", return_value=True), \
             mock.patch.object(supervisor, "qwen_api_compatible",
                               return_value=True) as compatible, \
             mock.patch.object(supervisor,
                               "qwen_api_rejects_invalid_credential",
                               return_value=True):
            result = supervisor.status(proposed)

        self.assertEqual(result["runtime_profile"]["fingerprint"],
                         proposed.fingerprint)
        self.assertEqual(result["active_runtime_profile"]["fingerprint"],
                         fallback.fingerprint)
        self.assertFalse(result["profile_matches"])
        self.assertTrue(result["boot_calibration"]["fallback_in_use"])
        self.assertTrue(result["friday"]["profile_matches_active"])
        compatible.assert_called_once_with(fallback, expected_pid=123)

    def test_failed_calibration_is_not_published_as_active(self):
        profile = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "qwen"
            launcher = root / "single-user" / "start_qwen.sh"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("#!/bin/sh\n")
            state = Path(temporary) / "state"
            manifest = state / "runtime-resolved.json"
            with mock.patch.multiple(
                    supervisor, QWEN=root,
                    **isolated_supervisor_state(state)), \
                 mock.patch.object(supervisor, "healthy", return_value=False), \
                 mock.patch.object(supervisor, "stop_pid") as stop, \
                 mock.patch.object(supervisor, "cleanup_orphaned_qwen"), \
                 mock.patch.object(supervisor, "discover_pid", return_value=None), \
                 mock.patch.object(supervisor, "wait_health"), \
                 mock.patch.object(supervisor, "wait_qwen_compatible"), \
                 mock.patch.object(supervisor, "owned", return_value=True), \
                 mock.patch.object(supervisor, "qwen_boot_calibration",
                                   side_effect=RuntimeError("canary failed")), \
                 mock.patch.object(
                     supervisor.subprocess, "Popen",
                     return_value=SimpleNamespace(pid=42)):
                with self.assertRaisesRegex(RuntimeError, "canary failed"):
                    supervisor._start_qwen_locked(
                        profile, calibration_evidence=[])

            pending_manifest = json.loads(manifest.read_text())
            self.assertEqual(pending_manifest["state"], "starting")
            self.assertNotIn("fingerprint", pending_manifest)
            stop.assert_called()

    def test_manifest_publication_failure_rolls_back_launched_process(self):
        profile = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "qwen"
            launcher = root / "single-user" / "start_qwen.sh"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("#!/bin/sh\n")
            state = Path(temporary) / "state"
            with mock.patch.multiple(
                    supervisor, QWEN=root,
                    **isolated_supervisor_state(state)), \
                 mock.patch.object(supervisor, "healthy", return_value=False), \
                 mock.patch.object(supervisor, "stop_pid") as stop, \
                 mock.patch.object(supervisor, "cleanup_orphaned_qwen"), \
                 mock.patch.object(supervisor, "discover_pid", return_value=None), \
                 mock.patch.object(supervisor, "wait_health"), \
                 mock.patch.object(supervisor, "wait_qwen_compatible"), \
                 mock.patch.object(supervisor, "owned", return_value=True), \
                 mock.patch.object(supervisor, "qwen_boot_calibration",
                                   return_value=evidence(200_000)), \
                 mock.patch.object(supervisor,
                                   "_write_runtime_process_binding"), \
                 mock.patch.object(supervisor, "write_runtime_profile",
                                   side_effect=OSError("disk unavailable")), \
                 mock.patch.object(
                     supervisor.subprocess, "Popen",
                     return_value=SimpleNamespace(pid=43)):
                with self.assertRaisesRegex(OSError, "disk unavailable"):
                    supervisor._start_qwen_locked(profile)

            self.assertGreaterEqual(stop.call_count, 2)
            self.assertEqual(stop.call_args.args[:3],
                             (state / "qwen.pid", root, "vllm"))
            self.assertTrue(stop.call_args.kwargs["group"])

    def test_pid_record_failure_rolls_back_exact_launcher_process(self):
        profile = select_runtime_profile(snapshot(24), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "qwen"
            launcher = root / "single-user" / "start_qwen.sh"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("#!/bin/sh\n")
            state = Path(temporary) / "state"
            bad_pid_path = state / "missing-parent" / "qwen.pid"

            def exact_identity(pid, cwd, marker):
                return (pid, cwd, marker) == (44, root, "start_qwen.sh")

            with mock.patch.multiple(
                    supervisor, QWEN=root,
                    **(isolated_supervisor_state(state) | {
                        "QWEN_PID": bad_pid_path})), \
                 mock.patch.object(supervisor, "healthy", return_value=False), \
                 mock.patch.object(supervisor, "stop_pid"), \
                 mock.patch.object(supervisor, "cleanup_orphaned_qwen"), \
                 mock.patch.object(supervisor, "discover_pid", return_value=None), \
                 mock.patch.object(supervisor, "owned",
                                   side_effect=exact_identity), \
                 mock.patch.object(supervisor, "_terminate_owned_process") as stop, \
                 mock.patch.object(
                     supervisor.subprocess, "Popen",
                     return_value=SimpleNamespace(pid=44)):
                with self.assertRaises(OSError):
                    supervisor._start_qwen_locked(profile)

            stop.assert_called_once_with(
                44, root, "start_qwen.sh", group=True)


if __name__ == "__main__":
    unittest.main()
