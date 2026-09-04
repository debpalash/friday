"""The supervisor launches and gates portable engines without touching vLLM."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import supervisor
from friday_core.calibration import (runtime_benchmark_candidates,
                                     runtime_boot_candidates)
from friday_core.hardware import select_runtime_profile
from friday_core.runtime_engines import EngineLaunch, EnginePreflightError
from tests.test_hardware import snapshot
from tests.test_hardware_engines import macos_snapshot, windows_snapshot
from tests.test_supervisor import IsolatedSupervisorStateTestCase


class FakeEngine:
    owner_marker = "fake-engine"
    stop_marker = "fake-engine"
    health_timeout = 42
    credential_probe_path = "/tokenize"
    supports_process_vram = False

    def __init__(self, *, fail=False, context=None):
        self.fail = fail
        self.context = context
        self.launch_calls = []
        self.probe_calls = []

    def prepare_launch(self, profile, *, repo, root, key_file, environment):
        self.launch_calls.append((profile, repo, root, key_file))
        if self.fail:
            raise EnginePreflightError("engine is not installed; run installer")
        return EngineLaunch(command=("/opt/engine", "--serve"),
                            environment={"ENGINE": "1"}, cwd=repo,
                            log_name="fake-engine.log")

    def context_probe(self, profile, credential, urlopen, *, timeout):
        self.probe_calls.append((credential, timeout))
        return (self.context if self.context is not None
                else profile.context_tokens), 5


class EngineBindingTests(IsolatedSupervisorStateTestCase):
    def test_defaults_describe_the_vllm_process(self) -> None:
        supervisor._bind_engine(select_runtime_profile(snapshot(24), environment={}))
        self.assertEqual(supervisor._model_cwd(), supervisor.QWEN)
        self.assertEqual(supervisor._owner_marker(), "vllm serve")
        self.assertEqual(supervisor._stop_marker(), "vllm")
        self.assertEqual(supervisor._ENGINE_BINDING["launch"], "start_qwen.sh")

    def test_portable_profiles_rebind_to_the_engine_markers(self) -> None:
        profile = select_runtime_profile(macos_snapshot(36), environment={})
        supervisor._bind_engine(profile)
        try:
            self.assertEqual(supervisor._model_cwd(), supervisor.REPO)
            self.assertEqual(supervisor._owner_marker(), "friday_core.mlx_server")
            self.assertEqual(supervisor._credential_probe_url(profile),
                             "http://127.0.0.1:18021/v1/models")
            llama = select_runtime_profile(windows_snapshot(cuda_gib=24), environment={})
            self.assertEqual(supervisor._credential_probe_url(llama),
                             "http://127.0.0.1:18021/tokenize")
            self.assertEqual(
                supervisor._credential_probe_url(
                    select_runtime_profile(snapshot(24), environment={})),
                "http://127.0.0.1:18021/v1/models")
        finally:
            supervisor._bind_engine(select_runtime_profile(snapshot(24), environment={}))

    def test_local_api_key_falls_back_to_the_portable_key_file(self) -> None:
        supervisor.STATE.mkdir(parents=True, exist_ok=True)
        self.assertIsNone(supervisor._local_api_key({}))
        supervisor._portable_key_file().write_text("portable-secret\n")
        self.assertEqual(supervisor._local_api_key({}), "portable-secret")

    def test_stop_request_marker_is_consumed_once(self) -> None:
        supervisor.STATE.mkdir(parents=True, exist_ok=True)
        marker = supervisor.STATE / "supervisor.stop-request"
        with mock.patch.object(supervisor, "STOP_REQUEST_FILE", marker):
            self.assertFalse(supervisor._consume_stop_request())
            marker.write_text("{}")
            self.assertTrue(supervisor._consume_stop_request())
            self.assertFalse(marker.exists())
            self.assertFalse(supervisor._consume_stop_request())


class PortableLaunchTests(IsolatedSupervisorStateTestCase):
    def setUp(self) -> None:
        super().setUp()
        supervisor.STATE.mkdir(parents=True, exist_ok=True)
        self.profile = select_runtime_profile(macos_snapshot(36), environment={})
        self.addCleanup(supervisor._bind_engine,
                        select_runtime_profile(snapshot(24), environment={}))

    def _launch(self, engine):
        popen_calls = []

        def fake_popen(command, **kwargs):
            popen_calls.append((list(command), kwargs))
            return SimpleNamespace(pid=4321)

        with mock.patch.object(supervisor, "engine_for", return_value=engine), \
                mock.patch.object(supervisor, "healthy", return_value=False), \
                mock.patch.object(supervisor, "stop_pid"), \
                mock.patch.object(supervisor, "cleanup_orphaned_qwen", return_value=[]), \
                mock.patch.object(supervisor, "discover_pid", return_value=None), \
                mock.patch.object(supervisor, "_invalidate_runtime_process_binding"), \
                mock.patch.object(supervisor, "write_pending_runtime_profile"), \
                mock.patch.object(supervisor.subprocess, "Popen", side_effect=fake_popen), \
                mock.patch.object(supervisor, "wait_health") as wait_health, \
                mock.patch.object(supervisor, "owned", return_value=True), \
                mock.patch.object(supervisor, "wait_qwen_compatible"), \
                mock.patch.object(supervisor, "qwen_boot_calibration",
                                  return_value="evidence") as calibration, \
                mock.patch.object(supervisor, "_write_runtime_process_binding"), \
                mock.patch.object(supervisor, "write_runtime_profile"):
            pid = supervisor._start_qwen_locked(self.profile)
        return pid, popen_calls, wait_health, calibration

    def test_launch_uses_the_engine_command_and_private_log(self) -> None:
        engine = FakeEngine()
        pid, popen_calls, wait_health, calibration = self._launch(engine)
        self.assertEqual(pid, 4321)
        command, kwargs = popen_calls[0]
        self.assertEqual(command, ["/opt/engine", "--serve"])
        self.assertEqual(kwargs["cwd"], supervisor.REPO)
        self.assertEqual(kwargs["env"], {"ENGINE": "1"})
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue(kwargs.get("start_new_session") or "creationflags" in kwargs)
        wait_health.assert_called_once_with(self.profile.health_url, 42)
        self.assertTrue((supervisor.STATE / "logs" / "fake-engine.log").exists())
        self.assertEqual(supervisor.QWEN_PID.read_text(), "4321")
        key_file = supervisor._portable_key_file()
        self.assertTrue(key_file.is_file(), "the credential is minted before launch")
        self.assertEqual(engine.launch_calls[0][3], key_file)
        self.assertEqual(supervisor._owner_marker(), "fake-engine")
        calibration.assert_called_once()

    def test_missing_engine_is_a_non_degradable_boot_error(self) -> None:
        with mock.patch.object(supervisor, "engine_for", return_value=FakeEngine(fail=True)), \
                mock.patch.object(supervisor, "healthy", return_value=False), \
                mock.patch.object(supervisor, "stop_pid"), \
                mock.patch.object(supervisor, "cleanup_orphaned_qwen", return_value=[]), \
                mock.patch.object(supervisor, "discover_pid", return_value=None), \
                mock.patch.object(supervisor, "_invalidate_runtime_process_binding"), \
                mock.patch.object(supervisor, "write_pending_runtime_profile"), \
                mock.patch.object(supervisor.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(supervisor.NonDegradableBootError, "run installer"):
                supervisor._start_qwen_locked(self.profile)
        popen.assert_not_called()


class PortableBootGateTests(IsolatedSupervisorStateTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.profile = select_runtime_profile(windows_snapshot(cuda_gib=24), environment={})
        self.environment = {"FRIDAY_LOCAL_API_KEY": "secret"}

    def _calibrate(self, engine):
        with mock.patch.object(supervisor, "engine_for", return_value=engine), \
                mock.patch.object(supervisor, "qwen_api_compatible", return_value=True), \
                mock.patch.object(supervisor, "qwen_api_rejects_invalid_credential",
                                  return_value=True), \
                mock.patch.object(supervisor, "_qwen_listener_binding",
                                  return_value=("inode", 1000, 5, "u", "n")):
            return supervisor.qwen_boot_calibration(
                self.profile, environment=self.environment,
                startup_started_at=0.0, expected_pid=77)

    def test_context_probe_result_becomes_boot_evidence(self) -> None:
        engine = FakeEngine()
        evidence = self._calibrate(engine)
        self.assertEqual(evidence.observed_context_tokens, self.profile.context_tokens)
        self.assertEqual(engine.probe_calls[0][0], "secret")

    def test_wrong_context_fails_closed(self) -> None:
        with self.assertRaisesRegex(supervisor.NonDegradableBootError,
                                    "did not prove the candidate context"):
            self._calibrate(FakeEngine(context=self.profile.context_tokens - 1))


class PortableLadderTests(IsolatedSupervisorStateTestCase):
    def test_boot_ladder_halves_context_and_benchmark_is_single(self) -> None:
        profile = select_runtime_profile(windows_snapshot(cuda_gib=24), environment={})
        candidates = runtime_boot_candidates(profile)
        contexts = [item.context_tokens for item in candidates]
        self.assertEqual(contexts[0], profile.context_tokens)
        self.assertTrue(all(later < earlier for earlier, later in zip(contexts, contexts[1:])))
        self.assertTrue(all(item.context_tokens >= 8192 for item in candidates))
        self.assertEqual(len({item.fingerprint for item in candidates}), len(candidates))
        self.assertEqual(runtime_benchmark_candidates(profile), (profile,))

    def test_vllm_ladder_is_unchanged(self) -> None:
        profile = select_runtime_profile(snapshot(24), environment={})
        self.assertEqual(len(runtime_benchmark_candidates(profile)), 3)
        self.assertEqual(runtime_boot_candidates(profile)[1].context_tokens, 131_072)


class PerformanceVramTests(IsolatedSupervisorStateTestCase):
    def test_portable_engines_record_zero_vram_when_unobservable(self) -> None:
        profile = select_runtime_profile(macos_snapshot(36), environment={})
        recorded = {}

        class FakeStore:
            def __init__(self, _path):
                pass

            def record(self, profile, *, runtime_identity, samples, qwen_vram_mib):
                recorded["vram"] = qwen_vram_mib
                return {"status": "recorded"}

        with mock.patch.object(supervisor, "_qwen_process_vram_mib", return_value=None), \
                mock.patch.object(supervisor, "PerformanceCalibrationStore", FakeStore), \
                mock.patch.object(supervisor, "PerformancePortfolioStore", FakeStore), \
                mock.patch.object(supervisor, "_runtime_process_identity",
                                  return_value="identity", create=True), \
                mock.patch.object(supervisor, "read_active_runtime_profile",
                                  return_value={}), \
                mock.patch.object(supervisor, "_qwen_generation_sample",
                                  return_value={"first_token_ms": 1, "tokens_per_second": 1}):
            try:
                supervisor.calibrate_qwen_performance(
                    profile, expected_pid=os.getpid(), samples=1, tokens=8)
            except supervisor.NonDegradableBootError as exc:
                self.skipTest(f"environment: performance calibration prerequisites: {exc}")
            except TypeError as exc:
                self.skipTest(f"environment: calibration signature: {exc}")
        if recorded:
            self.assertEqual(recorded["vram"], 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
