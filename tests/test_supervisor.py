import json
import os
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import supervisor


def isolated_supervisor_state(root: Path) -> dict:
    """Redirect every mutable supervisor record away from the live instance."""
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
        "FRIDAY_RUNTIME_FINGERPRINT_FILE": root / "friday-runtime-fingerprint",
    }


class IsolatedSupervisorStateTestCase(unittest.TestCase):
    """Fail-safe fixture: even an unexpected write is confined to a temp dir."""

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


class _Profile:
    def __init__(self, *, fingerprint="profile-a", served_model="friday-model",
                 llm_device=2, llm_devices=(), tts_device=3, available=True,
                 port=18021, native_vision=False):
        self.fingerprint = fingerprint
        self.served_model = served_model
        self.llm_cuda_device = llm_device
        self.effective_llm_cuda_devices = tuple(llm_devices) or (
            () if llm_device is None else (llm_device,))
        self.tts_cuda_device = tts_device
        self.tts_device = "cuda"
        self.local_runtime_available = available
        # Unit-test doubles are explicit profiles so automatic calibration does
        # not synthesize dataclass replacements from this lightweight object.
        self.overrides = ("TEST_PROFILE",)
        self.hardware_fingerprint = "hardware-a"
        self.family_fingerprint = "family-a"
        self.context_tokens = 200_000
        self.warnings = ("CUDA unavailable",) if not available else ()
        self.health_url = f"http://127.0.0.1:{port}/health"
        self.local_base_url = f"http://127.0.0.1:{port}/v1"
        self.llm_port = port
        self.qwen_model = "models/test"
        self.native_vision_enabled = native_vision
        self.native_vision_max_images = 1 if native_vision else 0
        self.native_vision_max_side = 1024 if native_vision else 0

    def qwen_environment(self):
        return {"MODEL": "models/test", "PORT": str(self.llm_port)}

    def assistant_environment(self):
        return {
            "FRIDAY_LOCAL_BASE_URL": self.local_base_url,
            "FRIDAY_LOCAL_MODEL": self.served_model,
        }

    def to_dict(self):
        return {"fingerprint": self.fingerprint,
                "served_model": self.served_model}


class SupervisorEnvironmentTests(IsolatedSupervisorStateTestCase):
    @staticmethod
    def _serve(handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def test_service_logs_are_created_private(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "server.log"
            path.write_text("old\n")
            path.chmod(0o644)
            with supervisor._private_log(path) as stream:
                stream.write("new\n")

            mode = stat.S_IMODE(path.stat().st_mode)

        self.assertEqual(mode, 0o600)

    def test_friday_health_probe_verifies_local_ca_without_proxy(self):
        response = mock.MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        opener = mock.MagicMock()
        opener.open.return_value = response
        context = object()
        with mock.patch.object(
                supervisor.ssl, "create_default_context",
                return_value=context) as create_context, \
             mock.patch.object(
                 supervisor.urllib.request, "build_opener",
                 return_value=opener) as build_opener, \
             mock.patch.object(
                 supervisor.urllib.request, "urlopen") as urlopen:
            result = supervisor.healthy(supervisor.FRIDAY_HEALTH_URL)

        self.assertTrue(result)
        create_context.assert_called_once_with(
            cafile=str(supervisor.STATE / "tls" / "friday-local-ca.crt"))
        opener.open.assert_called_once_with(
            supervisor.FRIDAY_HEALTH_URL, timeout=1.0)
        handlers = build_opener.call_args.args
        self.assertIsInstance(
            handlers[0], supervisor.urllib.request.ProxyHandler)
        self.assertIsInstance(
            handlers[1], supervisor.urllib.request.HTTPSHandler)
        urlopen.assert_not_called()

    def test_friday_health_fails_closed_when_ca_is_unavailable(self):
        with mock.patch.object(
                supervisor.ssl, "create_default_context",
                side_effect=FileNotFoundError):
            self.assertFalse(
                supervisor.healthy(supervisor.FRIDAY_HEALTH_URL))

    def test_qwen_uses_client_key_profile_device_and_mandatory_late_args(self):
        source = {
            "FRIDAY_LOCAL_API_KEY": "shared-secret",
            "FRIDAY_LLM_EXTRA_ARGS": (
                "--host 0.0.0.0 --served-model-name wrong-model"),
            "FRIDAY_LLM_CUDA_DEVICES": "7",
            "CUDA_VISIBLE_DEVICES": "8",
        }

        env = supervisor.build_qwen_environment(_Profile(), source)

        self.assertEqual(env["VLLM_API_KEY"], "shared-secret")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertTrue(env["EXTRA_ARGS"].endswith(
            "--host 127.0.0.1 --served-model-name friday-model "
            "--language-model-only"))
        self.assertEqual(source["CUDA_VISIBLE_DEVICES"], "8")

    def test_tensor_parallel_qwen_launch_is_exact_and_ambient_state_is_stripped(self):
        source = {
            "CUDA_VISIBLE_DEVICES": "7",
            "TENSOR_PARALLEL_SIZE": "9",
            "FRIDAY_LLM_EXTRA_ARGS": "--tensor-parallel-size 99",
        }

        env = supervisor.build_qwen_environment(
            _Profile(llm_device=0, llm_devices=(0, 1)), source)

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertNotEqual(env["TENSOR_PARALLEL_SIZE"], "9")
        self.assertEqual(env["TENSOR_PARALLEL_SIZE"], "2")
        self.assertTrue(env["EXTRA_ARGS"].endswith(
            "--host 127.0.0.1 --served-model-name friday-model "
            "--language-model-only --tensor-parallel-size 2"))
        self.assertEqual(source["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(source["TENSOR_PARALLEL_SIZE"], "9")

    def test_native_vision_launch_reasserts_bounded_multimodal_authority(self):
        profile = _Profile(native_vision=True)
        source = {"FRIDAY_LLM_EXTRA_ARGS": (
            "--language-model-only --limit-mm-per-prompt '{\"image\":99}'")}
        with mock.patch.object(
                supervisor, "_require_native_vision_checkpoint") as validate:
            env = supervisor.build_qwen_environment(profile, source)

        validate.assert_called_once_with(profile)
        self.assertTrue(env["EXTRA_ARGS"].endswith(
            "--no-language-model-only --limit-mm-per-prompt "
            "{\"image\":{\"count\":1,\"height\":1024,\"width\":1024},"
            "\"video\":0} --mm-processor-cache-gb 1 "
            "--no-skip-mm-profiling"))

    def test_native_vision_checkpoint_validation_requires_all_pinned_parts(self):
        profile = _Profile(native_vision=True)
        profile.qwen_model = "models/vision"
        with tempfile.TemporaryDirectory() as temporary:
            qwen = Path(temporary)
            model = qwen / "models" / "vision"
            model.mkdir(parents=True)
            (model / "config.json").write_text(json.dumps({
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "vision_config": {"model_type": "qwen3_5_vision"},
            }))
            (model / "preprocessor_config.json").write_text(json.dumps({
                "processor_class": "Qwen3VLProcessor",
            }))
            index = model / "model.safetensors.index.json"
            index.write_text(json.dumps({
                "weight_map": {"model.visual.patch_embed.weight": "part-1"},
            }))
            with mock.patch.object(supervisor, "QWEN", qwen):
                supervisor._require_native_vision_checkpoint(profile)
                index.write_text(json.dumps({
                    "weight_map": {"model.language_model.layers.0": "part-1"},
                }))
                with self.assertRaisesRegex(ValueError, "visual weights"):
                    supervisor._require_native_vision_checkpoint(profile)

    def test_native_vision_canary_is_text_free_and_listener_bound(self):
        raw = supervisor._vision_canary_png()
        self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(
            (int.from_bytes(raw[16:20], "big"),
             int.from_bytes(raw[20:24], "big"), raw[25]),
            (256, 160, 2))
        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "red square"}}],
        }).encode()
        response.__enter__.return_value = response
        profile = _Profile(native_vision=True)
        binding = ("inode-a", os.geteuid(), 100)
        with mock.patch.object(
                supervisor, "_require_native_vision_checkpoint"), \
             mock.patch.object(
                 supervisor, "_qwen_listener_binding",
                 side_effect=[binding, binding]), \
             mock.patch.object(
                 supervisor, "_credentialed_urlopen",
                 return_value=response) as urlopen:
            compatible = supervisor.qwen_native_vision_compatible(
                profile, {"FRIDAY_LOCAL_API_KEY": "secret"},
                expected_pid=42)

        self.assertTrue(compatible)
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertTrue(content[1]["image_url"]["url"].startswith(
            "data:image/png;base64,"))
        self.assertNotIn("7391", json.dumps(payload))

    def test_native_vision_score_is_profile_bound_and_uses_sanitized_images(self):
        captured = {}

        class FakeRunner:
            def __init__(self, _graph, complete, **kwargs):
                captured["complete"] = complete
                captured["kwargs"] = kwargs

            def run(self, path):
                captured["path"] = path
                captured["answer"] = captured["complete"](
                    "Which shape is left?", b"\x89PNG\r\n\x1a\nSANITIZED")
                return {"passed": 5, "total": 5}

        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "red square"}}],
        }).encode()
        response.__enter__.return_value = response
        binding = ("inode-a", os.geteuid(), 100)
        profile = _Profile(native_vision=True)
        with mock.patch.object(
                supervisor, "_require_native_vision_checkpoint"), \
             mock.patch.object(
                 supervisor, "NativeVisionEvalRunner", FakeRunner), \
             mock.patch.object(
                 supervisor, "_qwen_listener_binding",
                 side_effect=[binding, binding]), \
             mock.patch.object(
                 supervisor, "_credentialed_urlopen",
                 return_value=response) as urlopen:
            passed = supervisor.qwen_native_vision_score(
                profile, {"FRIDAY_LOCAL_API_KEY": "secret"}, expected_pid=42)

        self.assertTrue(passed)
        self.assertEqual(captured["answer"], "red square")
        self.assertEqual(captured["kwargs"], {
            "model": "friday-model",
            "runtime_fingerprint": "profile-a",
            "max_side": 1024,
        })
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))

    def test_explicit_client_key_file_reaches_both_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            key_file = Path(temporary) / "key.txt"
            key_file.write_text("file-secret\n")
            source = {"FRIDAY_LOCAL_API_KEY_FILE": str(key_file)}

            qwen = supervisor.build_qwen_environment(_Profile(), source)
            friday = supervisor.build_friday_environment(_Profile(), source)

        self.assertEqual(qwen["VLLM_API_KEY"], "file-secret")
        self.assertEqual(friday["FRIDAY_LOCAL_API_KEY"], "file-secret")

    def test_friday_uses_resolved_tts_device_not_raw_override(self):
        source = {
            "FRIDAY_TTS_CUDA_DEVICES": "6",
            "CUDA_VISIBLE_DEVICES": "7",
        }

        env = supervisor.build_friday_environment(
            _Profile(tts_device=1), source, activate_voice="voice-a")

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["FRIDAY_ACTIVATE_VOICE"], "voice-a")

    def test_invalid_served_model_cannot_enter_shell_expanded_arguments(self):
        with self.assertRaisesRegex(ValueError, "served model"):
            supervisor.build_qwen_environment(
                _Profile(served_model="model; touch /tmp/oops"), {})

    def test_api_compatibility_probe_uses_key_and_exact_served_model(self):
        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({
            "data": [{"id": "friday-model"}],
        }).encode()
        response.__enter__.return_value = response

        with mock.patch.object(supervisor, "_credentialed_urlopen",
                               return_value=response) as urlopen, \
             mock.patch.object(
                 supervisor, "_qwen_listener_binding",
                 return_value=("inode-a", os.geteuid(), 100)):
            compatible = supervisor.qwen_api_compatible(
                _Profile(), {"FRIDAY_LOCAL_API_KEY": "shared-secret"},
                expected_pid=42)

        request = urlopen.call_args.args[0]
        self.assertTrue(compatible)
        self.assertEqual(request.full_url,
                         "http://127.0.0.1:18021/v1/models")
        self.assertEqual(request.get_header("Authorization"),
                         "Bearer shared-secret")

    def test_credentialed_model_probe_rejects_same_origin_redirect(self):
        observed = []

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                observed.append((self.path, self.headers.get("Authorization")))
                if self.path == "/v1/models":
                    self.send_response(302)
                    self.send_header("Location", "/redirected")
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()

            def log_message(self, _format, *_args):
                return

        server, thread = self._serve(RedirectHandler)
        try:
            with mock.patch.object(supervisor, "owned", return_value=True):
                compatible = supervisor.qwen_api_compatible(
                    _Profile(port=server.server_port),
                    {"FRIDAY_LOCAL_API_KEY": "redirect-secret"},
                    expected_pid=os.getpid(), timeout=1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertFalse(compatible)
        self.assertEqual(observed, [
            ("/v1/models", "Bearer redirect-secret"),
        ])

    def test_tokenization_probe_rejects_cross_origin_redirect(self):
        source_requests = []
        redirected_requests = []

        class TargetHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                redirected_requests.append(
                    (self.path, self.headers.get("Authorization")))
                self.send_response(200)
                self.end_headers()

            do_POST = do_GET

            def log_message(self, _format, *_args):
                return

        target, target_thread = self._serve(TargetHandler)

        class SourceHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                source_requests.append(
                    (self.path, self.headers.get("Authorization")))
                self.send_response(307)
                self.send_header(
                    "Location",
                    f"http://127.0.0.1:{target.server_port}/capture")
                self.end_headers()

            def log_message(self, _format, *_args):
                return

        source, source_thread = self._serve(SourceHandler)
        try:
            with mock.patch.object(
                    supervisor, "qwen_api_compatible", return_value=True), \
                 mock.patch.object(
                     supervisor, "qwen_api_rejects_invalid_credential",
                     return_value=True), \
                 mock.patch.object(supervisor, "owned", return_value=True):
                with self.assertRaisesRegex(
                        supervisor.NonDegradableBootError,
                        "tokenization calibration"):
                    supervisor.qwen_boot_calibration(
                        _Profile(port=source.server_port),
                        startup_started_at=supervisor.time.monotonic(),
                        expected_pid=os.getpid(),
                        environment={"FRIDAY_LOCAL_API_KEY": "redirect-secret"},
                        timeout=1)
        finally:
            source.shutdown()
            source.server_close()
            source_thread.join(timeout=2)
            target.shutdown()
            target.server_close()
            target_thread.join(timeout=2)

        self.assertEqual(source_requests, [
            ("/tokenize", "Bearer redirect-secret"),
        ])
        self.assertEqual(redirected_requests, [])

    def test_wrong_key_probe_normalizes_malformed_redirect_failure(self):
        with mock.patch.object(
                supervisor, "_credentialed_urlopen",
                side_effect=ValueError("malformed redirect")), \
             mock.patch.object(
                 supervisor, "_qwen_listener_binding",
                 return_value=("inode-a", os.geteuid(), 100)):
            enforced = supervisor.qwen_api_rejects_invalid_credential(
                _Profile(), expected_pid=42)

        self.assertFalse(enforced)

    def test_hostile_prebound_listener_is_never_sent_the_valid_key(self):
        observed = []

        class HostileHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                observed.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()

            def log_message(self, _format, *_args):
                return

        server, thread = self._serve(HostileHandler)
        try:
            compatible = supervisor.qwen_api_compatible(
                _Profile(port=server.server_port),
                {"FRIDAY_LOCAL_API_KEY": "must-not-reach-hostile-listener"},
                expected_pid=1, timeout=1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertFalse(compatible)
        self.assertEqual(observed, [])

    def test_listener_swap_invalidates_an_otherwise_valid_response(self):
        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({
            "data": [{"id": "friday-model"}],
        }).encode()
        response.__enter__.return_value = response
        with mock.patch.object(
                supervisor, "_qwen_listener_binding",
                side_effect=[
                    ("inode-before", os.geteuid(), 100),
                    ("inode-after", os.geteuid(), 100),
                ]), \
             mock.patch.object(
                 supervisor, "_credentialed_urlopen",
                 return_value=response):
            compatible = supervisor.qwen_api_compatible(
                _Profile(), {"FRIDAY_LOCAL_API_KEY": "shared-secret"},
                expected_pid=42)

        self.assertFalse(compatible)

    def test_listener_swap_invalidates_wrong_key_rejection(self):
        denied = supervisor.urllib.error.HTTPError(
            "http://127.0.0.1:18021/v1/models", 401, "denied", {}, None)
        with mock.patch.object(
                supervisor, "_qwen_listener_binding",
                side_effect=[
                    ("inode-before", os.geteuid(), 100),
                    ("inode-after", os.geteuid(), 100),
                ]), \
             mock.patch.object(
                 supervisor, "_credentialed_urlopen", side_effect=denied):
            enforced = supervisor.qwen_api_rejects_invalid_credential(
                _Profile(), expected_pid=42)

        self.assertFalse(enforced)

    def test_listener_swap_invalidates_tokenization_canary(self):
        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({
            "count": 5,
            "max_model_len": 200_000,
        }).encode()
        response.__enter__.return_value = response
        with mock.patch.object(
                supervisor, "qwen_api_compatible", return_value=True), \
             mock.patch.object(
                 supervisor, "qwen_api_rejects_invalid_credential",
                 return_value=True), \
             mock.patch.object(
                 supervisor, "_qwen_listener_binding",
                 side_effect=[
                     ("inode-before", os.geteuid(), 100),
                     ("inode-after", os.geteuid(), 100),
                 ]), \
             mock.patch.object(
                 supervisor, "_credentialed_urlopen", return_value=response):
            with self.assertRaisesRegex(
                    supervisor.NonDegradableBootError,
                    "listener identity changed"):
                supervisor.qwen_boot_calibration(
                    _Profile(), startup_started_at=1.0, expected_pid=42,
                    environment={"FRIDAY_LOCAL_API_KEY": "shared-secret"})

    def test_ambiguous_listener_inodes_fail_closed(self):
        with mock.patch.object(
                supervisor, "_process_start_ticks",
                side_effect=[100, 100]), \
             mock.patch.object(
                 supervisor, "_process_effective_uid",
                 return_value=os.geteuid()), \
             mock.patch.object(
                 supervisor, "_process_namespace_identity",
                 return_value=("user:[1]", "net:[1]")), \
             mock.patch.object(supervisor, "owned", return_value=True), \
             mock.patch.object(
                 supervisor, "_loopback_listener_records",
                 return_value=[
                     ("inode-a", os.geteuid()),
                     ("inode-b", os.geteuid()),
                 ]), \
             mock.patch.object(
                 supervisor, "_process_socket_inodes",
                 return_value={"inode-a", "inode-b"}):
            binding = supervisor._qwen_listener_binding(_Profile(), 42)

        self.assertIsNone(binding)

    def test_process_start_change_during_listener_capture_fails_closed(self):
        with mock.patch.object(
                supervisor, "_process_start_ticks",
                side_effect=[100, 101]), \
             mock.patch.object(
                 supervisor, "_process_effective_uid",
                 return_value=os.geteuid()), \
             mock.patch.object(
                 supervisor, "_process_namespace_identity",
                 return_value=("user:[1]", "net:[1]")), \
             mock.patch.object(supervisor, "owned", return_value=True), \
             mock.patch.object(
                 supervisor, "_loopback_listener_records",
                 return_value=[("inode-a", os.geteuid())]), \
             mock.patch.object(
                 supervisor, "_process_socket_inodes",
                 return_value={"inode-a"}):
            binding = supervisor._qwen_listener_binding(_Profile(), 42)

        self.assertIsNone(binding)


class SupervisorProfilePersistenceTests(IsolatedSupervisorStateTestCase):
    def test_healthy_qwen_rejects_unknown_profile_without_relabeling(self):
        profile = _Profile()
        with mock.patch.object(supervisor, "healthy", return_value=True), \
             mock.patch.object(supervisor, "read_active_runtime_profile",
                               return_value={"fingerprint": "other"}), \
             mock.patch.object(supervisor, "verified_pid", return_value=42), \
             mock.patch.object(supervisor, "active_runtime_process_matches",
                               return_value=True), \
             mock.patch.object(supervisor, "write_runtime_profile") as write:
            with self.assertRaisesRegex(RuntimeError, "restart-all"):
                supervisor.start_qwen(profile)

        write.assert_not_called()

    def test_matching_manifest_still_rejects_wrong_key_or_served_model(self):
        profile = _Profile()
        with mock.patch.object(supervisor, "healthy", return_value=True), \
             mock.patch.object(supervisor, "read_active_runtime_profile",
                               return_value={"fingerprint": profile.fingerprint}), \
             mock.patch.object(supervisor, "verified_pid", return_value=42), \
             mock.patch.object(supervisor, "active_runtime_process_matches",
                               return_value=True), \
             mock.patch.object(supervisor, "qwen_api_compatible",
                               return_value=False):
            with self.assertRaisesRegex(RuntimeError, "credential|model"):
                supervisor.start_qwen(profile)

    def test_qwen_manifest_is_published_only_after_health_check(self):
        profile = _Profile()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "qwen"
            launcher = root / "single-user" / "start_qwen.sh"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("#!/bin/sh\n")
            state = Path(temporary) / "state"
            state.mkdir()
            manifest = state / "runtime-resolved.json"

            def healthy_then_publish(_url, _timeout):
                pending = json.loads(manifest.read_text())
                self.assertEqual(pending["state"], "starting")
                self.assertNotIn("fingerprint", pending)

            with mock.patch.multiple(
                    supervisor, QWEN=root,
                    **isolated_supervisor_state(state)), \
                 mock.patch.object(supervisor, "healthy", return_value=False), \
                 mock.patch.object(supervisor, "cleanup_orphaned_qwen"), \
                 mock.patch.object(supervisor, "discover_pid", return_value=None), \
                 mock.patch.object(supervisor, "wait_health",
                                   side_effect=healthy_then_publish), \
                 mock.patch.object(supervisor, "wait_qwen_compatible"), \
                 mock.patch.object(supervisor, "owned", return_value=True), \
                 mock.patch.object(supervisor, "qwen_boot_calibration"), \
                 mock.patch.object(
                     supervisor, "_write_runtime_process_binding"), \
                 mock.patch.object(
                     supervisor.subprocess, "Popen",
                     return_value=SimpleNamespace(pid=42)):
                supervisor.start_qwen(profile)

            active = json.loads(manifest.read_text())

        self.assertEqual(active["fingerprint"], profile.fingerprint)

    def test_friday_requires_matching_active_qwen_profile(self):
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.multiple(
                 supervisor,
                 **isolated_supervisor_state(Path(temporary))), \
             mock.patch.object(supervisor, "verified_pid", return_value=42), \
             mock.patch.object(supervisor, "read_active_runtime_profile",
                               return_value=None), \
             mock.patch.object(supervisor.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "restart-all"):
                supervisor.start_friday(profile=_Profile())

        launch.assert_not_called()

    def test_status_distinguishes_selected_active_and_friday_profiles(self):
        profile = _Profile()
        active = {"fingerprint": "other", "name": "old-profile"}
        with mock.patch.object(supervisor, "read_active_runtime_profile",
                               return_value=active), \
             mock.patch.object(supervisor, "_read_friday_fingerprint",
                               return_value="other"), \
             mock.patch.object(supervisor, "read_pid", return_value=None), \
             mock.patch.object(supervisor, "discover_pid", return_value=None), \
             mock.patch.object(supervisor, "healthy", return_value=False):
            result = supervisor.status(profile)

        self.assertEqual(result["runtime_profile"]["fingerprint"], "profile-a")
        self.assertEqual(result["active_runtime_profile"], active)
        self.assertFalse(result["profile_matches"])
        self.assertFalse(result["friday"]["profile_matches"])

    def test_unsupported_profile_fails_before_any_health_or_launch_probe(self):
        with mock.patch.object(supervisor, "healthy") as healthy, \
             mock.patch.object(supervisor.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                supervisor.start_qwen(_Profile(available=False))

        healthy.assert_not_called()
        launch.assert_not_called()

    def test_watch_backs_off_then_reprobes_an_unsupported_runtime(self):
        class StopWatch(Exception):
            pass

        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 2:
                raise StopWatch

        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch("sys.argv", ["supervisor.py", "watch"]), \
             mock.patch.multiple(
                 supervisor,
                 **isolated_supervisor_state(Path(temporary))), \
             mock.patch.object(supervisor, "read_pid", return_value=None), \
             mock.patch.object(supervisor, "lifecycle_locked", return_value=False), \
             mock.patch.object(
                 supervisor, "resolve_runtime_profile",
                 side_effect=[_Profile(available=False), _Profile()]) as resolve, \
             mock.patch.object(supervisor, "healthy", return_value=True), \
             mock.patch.object(supervisor, "verified_pid", return_value=42), \
             mock.patch.object(supervisor, "read_active_runtime_profile",
                               return_value={"fingerprint": "profile-a"}), \
             mock.patch.object(supervisor, "qwen_api_compatible",
                               return_value=True), \
             mock.patch.object(supervisor,
                               "qwen_api_rejects_invalid_credential",
                               return_value=True), \
             mock.patch.object(supervisor,
                               "active_runtime_process_matches",
                               return_value=True), \
             mock.patch.object(supervisor, "active_runtime_identity",
                               return_value="runtime-watch"), \
             mock.patch.object(supervisor.time, "sleep", side_effect=sleep):
            with self.assertRaises(StopWatch):
                supervisor.main()

        self.assertEqual(resolve.call_count, 2)
        self.assertEqual(sleeps, [60, 5])

    def test_restart_friday_preflights_profile_before_stopping_service(self):
        with mock.patch("sys.argv", ["supervisor.py", "restart-friday"]), \
             mock.patch.object(
                 supervisor, "resolve_runtime_profile", return_value=_Profile()), \
             mock.patch.object(supervisor, "verified_pid", return_value=42), \
             mock.patch.object(supervisor, "lifecycle_locked", return_value=False), \
             mock.patch.object(
                 supervisor, "_require_active_qwen_profile",
                 side_effect=RuntimeError("profile mismatch")), \
             mock.patch.object(supervisor, "stop_pid") as stop:
            with self.assertRaisesRegex(RuntimeError, "profile mismatch"):
                supervisor.main()

        stop.assert_not_called()


class SupervisorLifecycleFencingTests(IsolatedSupervisorStateTestCase):
    def test_sigterm_records_planned_stop_before_exiting(self):
        with mock.patch.object(
                supervisor, "_record_planned_watch_stop",
                return_value=True) as record:
            with self.assertRaises(SystemExit) as stopped:
                supervisor._request_watch_stop()

        self.assertEqual(stopped.exception.code, 0)
        record.assert_called_once_with()

    def test_planned_runtime_identity_rejects_stale_binding(self):
        active = {"fingerprint": "profile-a"}
        valid = {
            "schema_version": 1,
            "profile_fingerprint": "profile-a",
            "boot_id_hash": "boot-a",
            "pid": 42,
            "start_ticks": 99,
        }
        with mock.patch.object(
                supervisor, "_read_runtime_process_binding",
                return_value=valid), \
             mock.patch.object(supervisor, "_boot_id_hash",
                               return_value="boot-a"):
            identity = supervisor._planned_runtime_identity(active)
        with mock.patch.object(
                supervisor, "_read_runtime_process_binding",
                return_value=valid | {"boot_id_hash": "old-boot"}), \
             mock.patch.object(supervisor, "_boot_id_hash",
                               return_value="boot-a"):
            stale = supervisor._planned_runtime_identity(active)

        self.assertRegex(identity or "", r"^[0-9a-f]{64}$")
        self.assertIsNone(stale)

    def test_watch_reload_execs_in_place_and_unit_keeps_full_shutdown(self):
        unit = (Path(__file__).parents[1] / "ops" /
                "friday-supervisor.service").read_text()
        prior = supervisor._WATCH_RELOAD_REQUESTED
        self.addCleanup(
            setattr, supervisor, "_WATCH_RELOAD_REQUESTED", prior)
        supervisor._WATCH_RELOAD_REQUESTED = False
        supervisor._request_watch_reload()
        with mock.patch.object(supervisor.os, "execv") as execute:
            supervisor._reload_watch_if_requested()

        execute.assert_called_once_with(
            supervisor.sys.executable,
            [supervisor.sys.executable,
             str(Path(supervisor.__file__).resolve()), "watch"])
        self.assertIn("KillMode=control-group", unit)
        self.assertIn("ExecReload=/bin/kill -HUP $MAINPID", unit)

    def test_lifecycle_lock_is_kernel_owned_and_ignores_reused_pid_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "lifecycle.lock"
            lock_path.write_text(json.dumps({
                "pid": os.getpid(),
                "voice": "stale-voice",
            }))
            with mock.patch.object(supervisor, "LIFECYCLE_LOCK", lock_path):
                self.assertFalse(supervisor.lifecycle_locked())
                self.assertIsNone(supervisor.lifecycle_request())

            self.assertEqual(lock_path.read_text(), "")

    def test_lifecycle_operation_excludes_competitors_and_exposes_request(self):
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(
                 supervisor, "LIFECYCLE_LOCK",
                 Path(temporary) / "lifecycle.lock"):
            with supervisor.lifecycle_operation(voice="voice-exact"):
                self.assertTrue(supervisor.lifecycle_locked())
                request = supervisor.lifecycle_request()
                self.assertEqual(request["voice"], "voice-exact")
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with supervisor.lifecycle_operation():
                        self.fail("a second lifecycle owner entered")

            self.assertFalse(supervisor.lifecycle_locked())
            self.assertIsNone(supervisor.lifecycle_request())

    def test_stop_preserves_pid_file_replaced_during_shutdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary) / "friday.pid"
            pid_file.write_text("101")

            def terminate(*_args, **_kwargs):
                pid_file.write_text("202")

            with mock.patch.object(supervisor, "read_pid", return_value=101), \
                 mock.patch.object(supervisor, "owned", return_value=True), \
                 mock.patch.object(supervisor, "qwen_boot_calibration"), \
                 mock.patch.object(
                     supervisor, "_write_runtime_process_binding"), \
                 mock.patch.object(
                     supervisor, "_terminate_owned_process",
                     side_effect=terminate):
                supervisor.stop_pid(pid_file, Path(temporary), "server.py")

            self.assertEqual(pid_file.read_text(), "202")

    def test_stop_keeps_ownership_record_when_process_cannot_be_killed(self):
        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary) / "qwen.pid"
            pid_file.write_text("303")
            with mock.patch.object(supervisor, "read_pid", return_value=303), \
                 mock.patch.object(supervisor, "owned", return_value=True), \
                 mock.patch.object(
                     supervisor, "_terminate_owned_process",
                     side_effect=RuntimeError("still alive")):
                with self.assertRaisesRegex(RuntimeError, "still alive"):
                    supervisor.stop_pid(
                        pid_file, Path(temporary), "vllm", group=True)

            self.assertEqual(pid_file.read_text(), "303")

    def test_termination_escalates_to_sigkill_before_returning(self):
        with mock.patch.object(supervisor.os, "pidfd_open", return_value=9), \
             mock.patch.object(supervisor, "owned", return_value=True), \
             mock.patch.object(
                 supervisor, "_wait_pid_exit",
                 side_effect=[False, True]), \
             mock.patch.object(
                 supervisor.signal, "pidfd_send_signal") as send, \
             mock.patch.object(supervisor.os, "close"):
            supervisor._terminate_owned_process(
                404, Path("/service"), "server.py", group=False,
                grace_seconds=0, kill_seconds=0)

        self.assertEqual(
            [call.args[1] for call in send.call_args_list],
            [supervisor.signal.SIGTERM, supervisor.signal.SIGKILL],
        )

    def test_status_never_calls_an_unowned_endpoint_healthy(self):
        profile = _Profile()
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            qwen_pid = state / "qwen.pid"
            friday_pid = state / "friday.pid"
            qwen_pid.write_text(str(os.getpid()))
            friday_pid.write_text(str(os.getpid()))
            with mock.patch.multiple(
                    supervisor, QWEN_PID=qwen_pid, FRIDAY_PID=friday_pid), \
                 mock.patch.object(supervisor, "owned", return_value=False), \
                 mock.patch.object(supervisor, "discover_pid", return_value=None), \
                 mock.patch.object(supervisor, "healthy", return_value=True), \
                 mock.patch.object(supervisor, "read_active_runtime_profile",
                                   return_value={"fingerprint": profile.fingerprint}), \
                 mock.patch.object(supervisor, "_read_friday_fingerprint",
                                   return_value=profile.fingerprint), \
                 mock.patch.object(supervisor, "qwen_api_compatible") as compatible:
                result = supervisor.status(profile)

        self.assertFalse(result["qwen"]["healthy"])
        self.assertFalse(result["friday"]["healthy"])
        self.assertIsNone(result["qwen"]["pid"])
        self.assertIsNone(result["friday"]["pid"])
        compatible.assert_not_called()

    def test_concurrent_qwen_starts_launch_exactly_one_process(self):
        profile = _Profile()
        health = {"ready": False}
        launch_count = 0
        launch_guard = threading.Lock()
        ready = threading.Barrier(2)

        def run_start():
            ready.wait(timeout=5)
            return supervisor.start_qwen(profile)

        def launch(*_args, **_kwargs):
            nonlocal launch_count
            with launch_guard:
                launch_count += 1
            return SimpleNamespace(pid=505)

        def become_healthy(*_args, **_kwargs):
            health["ready"] = True

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "qwen"
            launcher = root / "single-user" / "start_qwen.sh"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("#!/bin/sh\n")
            state = Path(temporary) / "state"
            with mock.patch.multiple(
                    supervisor, QWEN=root,
                    **isolated_supervisor_state(state)), \
                 mock.patch.object(
                     supervisor, "healthy",
                     side_effect=lambda _url: health["ready"]), \
                 mock.patch.object(supervisor, "stop_pid"), \
                 mock.patch.object(supervisor, "cleanup_orphaned_qwen"), \
                 mock.patch.object(supervisor, "discover_pid", return_value=None), \
                 mock.patch.object(supervisor, "wait_health",
                                   side_effect=become_healthy), \
                 mock.patch.object(supervisor, "wait_qwen_compatible"), \
                 mock.patch.object(supervisor, "owned", return_value=True), \
                 mock.patch.object(
                     supervisor, "qwen_boot_calibration",
                     return_value=mock.MagicMock()), \
                 mock.patch.object(supervisor, "verified_pid", return_value=505), \
                 mock.patch.object(supervisor, "_require_active_qwen_profile"), \
                 mock.patch.object(supervisor, "_require_compatible_qwen"), \
                 mock.patch.object(
                     supervisor, "_write_runtime_process_binding"), \
                 mock.patch.object(supervisor, "write_runtime_profile"), \
                 mock.patch.object(supervisor.subprocess, "Popen",
                                   side_effect=launch):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _index: run_start(), range(2)))

        self.assertEqual(results, [505, 505])
        self.assertEqual(launch_count, 1)


if __name__ == "__main__":
    unittest.main()
