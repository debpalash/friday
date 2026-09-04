"""Portable engines launch, probe, and fail closed with exact contracts."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from friday_core import engine_assets, runtime_engines
from friday_core.hardware import select_runtime_profile
from friday_core.runtime_engines import (EnginePreflightError,
                                         LlamaServerEngine, MlxEngine,
                                         engine_for, ensure_local_api_key,
                                         runtime_root)
from friday_host.host import HostPlatform
from tests.test_hardware_engines import macos_snapshot, windows_snapshot

WINDOWS = HostPlatform(os="windows", arch="x86_64")
MACOS = HostPlatform(os="macos", arch="aarch64")
LINUX = HostPlatform(os="linux", arch="x86_64")


def _fake_asset(key: str, engine: str, entry: str) -> engine_assets.ModelAsset:
    files = ((entry, 11, "0" * 64),) if entry else (
        ("config.json", 11, "0" * 64), ("model.safetensors", 11, "0" * 64))
    return engine_assets.ModelAsset(
        key=key, engine=engine, size_label="8b", repo="Qwen/Test",
        revision="a" * 40, license="Apache-2.0", files=files,
        weights_bytes=11, kv_bytes_per_token=1, entry=entry)


class _Response:
    def __init__(self, payload, status=200):
        self.status = status
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class EngineFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "release"
        self.root = Path(self.temporary.name) / "runtime"
        self.key_file = Path(self.temporary.name) / "state" / "local-api-key"
        self.repo.mkdir()
        self.root.mkdir()

    def _install_model(self, asset: engine_assets.ModelAsset) -> Path:
        directory = self.repo / "models" / asset.directory
        directory.mkdir(parents=True)
        for name, size, _digest in asset.files:
            (directory / name).write_bytes(b"x" * size)
        return directory


class LlamaServerEngineTests(EngineFixture):
    def test_launch_argv_and_environment_are_exact(self) -> None:
        profile = select_runtime_profile(windows_snapshot(cuda_gib=24), environment={})
        asset = _fake_asset(profile.model_asset, "llama_server", "Qwen3-Test-Q4_K_M.gguf")
        model_dir = self._install_model(asset)
        binary = engine_assets.llama_server_binary("windows", "x86_64", "cuda")
        binary_path = self.root / "llama-server" / binary.directory / binary.executable
        binary_path.parent.mkdir(parents=True)
        binary_path.write_bytes(b"MZ")
        engine = LlamaServerEngine(WINDOWS)
        with mock.patch.object(runtime_engines, "model_asset", return_value=asset):
            launch = engine.prepare_launch(
                profile, repo=self.repo, root=self.root, key_file=self.key_file,
                environment={"PATH": "/usr/bin", "LLAMA_ARG_PORT": "9",
                             "CUDA_VISIBLE_DEVICES": "3"})
        command = list(launch.command)
        self.assertEqual(command[0], str(binary_path))
        self.assertEqual(command[command.index("--model") + 1],
                         str(model_dir / "Qwen3-Test-Q4_K_M.gguf"))
        self.assertEqual(command[command.index("--api-key-file") + 1], str(self.key_file))
        self.assertEqual(command[command.index("--ctx-size") + 1],
                         str(profile.context_tokens * profile.max_sequences))
        self.assertEqual(command[command.index("--parallel") + 1], str(profile.max_sequences))
        self.assertEqual(command[command.index("--alias") + 1], profile.served_model)
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "18021")
        self.assertIn("--reasoning-budget", command)
        self.assertNotIn("--api-key", command, "the secret never travels in argv")
        self.assertNotIn("LLAMA_ARG_PORT", launch.environment)
        self.assertEqual(launch.environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(launch.environment["PATH"], "/usr/bin")
        self.assertEqual(launch.cwd, self.repo)
        self.assertEqual(launch.log_name, "llama-server.log")

    def test_cpu_backend_drops_device_variables(self) -> None:
        profile = select_runtime_profile(macos_snapshot(16, arch="x86_64"), environment={})
        asset = _fake_asset(profile.model_asset, "llama_server", "m.gguf")
        self._install_model(asset)
        binary = engine_assets.llama_server_binary("macos", "x86_64", "cpu")
        path = self.root / "llama-server" / binary.directory / binary.executable
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
        with mock.patch.object(runtime_engines, "model_asset", return_value=asset):
            launch = LlamaServerEngine(HostPlatform(os="macos", arch="x86_64")).prepare_launch(
                profile, repo=self.repo, root=self.root, key_file=self.key_file,
                environment={"CUDA_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "0"})
        self.assertNotIn("CUDA_VISIBLE_DEVICES", launch.environment)
        self.assertNotIn("HIP_VISIBLE_DEVICES", launch.environment)
        self.assertEqual(list(launch.command)[list(launch.command).index("--n-gpu-layers") + 1], "0")

    def test_preflight_names_the_missing_piece(self) -> None:
        profile = select_runtime_profile(windows_snapshot(cuda_gib=24), environment={})
        engine = LlamaServerEngine(WINDOWS)
        with self.assertRaisesRegex(EnginePreflightError, "install_llama_server"):
            engine.preflight(profile, repo=self.repo, root=self.root)
        binary = engine_assets.llama_server_binary("windows", "x86_64", "cuda")
        path = self.root / "llama-server" / binary.directory / binary.executable
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
        with self.assertRaisesRegex(EnginePreflightError, "install_local_model"):
            engine.preflight(profile, repo=self.repo, root=self.root)
        asset = _fake_asset(profile.model_asset, "llama_server", "m.gguf")
        directory = self._install_model(asset)
        (directory / "m.gguf").write_bytes(b"short")
        with mock.patch.object(runtime_engines, "model_asset", return_value=asset), \
                self.assertRaisesRegex(EnginePreflightError, "size changed"):
            engine.preflight(profile, repo=self.repo, root=self.root)

    def test_context_probe_reads_props_and_tokenizes(self) -> None:
        profile = select_runtime_profile(windows_snapshot(cuda_gib=24), environment={})
        calls = []

        def urlopen(request, timeout):
            calls.append((request.full_url, request.get_method(),
                          request.get_header("Authorization")))
            if request.full_url.endswith("/props"):
                return _Response({"default_generation_settings": {
                    "n_ctx": profile.context_tokens}})
            return _Response({"tokens": [1, 2, 3, 4]})

        observed, count = LlamaServerEngine(WINDOWS).context_probe(
            profile, "secret", urlopen, timeout=5)
        self.assertEqual((observed, count), (profile.context_tokens, 4))
        self.assertEqual(calls[0][0], "http://127.0.0.1:18021/props")
        self.assertEqual(calls[1][0], "http://127.0.0.1:18021/tokenize")
        self.assertEqual(calls[1][1], "POST")
        self.assertTrue(all(auth == "Bearer secret" for _u, _m, auth in calls))

        def broken(request, timeout):
            return _Response({"default_generation_settings": {}})

        with self.assertRaisesRegex(RuntimeError, "context size"):
            LlamaServerEngine(WINDOWS).context_probe(profile, "s", broken, timeout=5)


class MlxEngineTests(EngineFixture):
    def test_launch_runs_the_wrapper_inside_the_pinned_venv(self) -> None:
        profile = select_runtime_profile(macos_snapshot(36), environment={})
        asset = _fake_asset(profile.model_asset, "mlx_lm", "")
        model_dir = self._install_model(asset)
        python = self.root / "mlx" / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"#!")
        with mock.patch.object(runtime_engines, "model_asset", return_value=asset):
            launch = MlxEngine(MACOS).prepare_launch(
                profile, repo=self.repo, root=self.root, key_file=self.key_file,
                environment={"PYTHONHOME": "/bad", "PATH": "/usr/bin"})
        command = list(launch.command)
        self.assertEqual(command[:3], [str(python), "-m", "friday_core.mlx_server"])
        self.assertEqual(command[command.index("--model") + 1], str(model_dir))
        self.assertEqual(command[command.index("--served-model") + 1], profile.served_model)
        self.assertEqual(command[command.index("--context-tokens") + 1],
                         str(profile.context_tokens))
        self.assertEqual(launch.environment["PYTHONPATH"], str(self.repo))
        self.assertEqual(launch.environment["HF_HUB_OFFLINE"], "1")
        self.assertNotIn("PYTHONHOME", launch.environment)
        self.assertEqual(MlxEngine(MACOS).credential_probe_path, "/v1/models")

    def test_preflight_requires_the_runtime(self) -> None:
        profile = select_runtime_profile(macos_snapshot(36), environment={})
        with self.assertRaisesRegex(EnginePreflightError, "install_mlx_runtime"):
            MlxEngine(MACOS).preflight(profile, repo=self.repo, root=self.root)

    def test_context_probe_uses_the_vllm_shaped_tokenize(self) -> None:
        profile = select_runtime_profile(macos_snapshot(36), environment={})

        def urlopen(request, timeout):
            body = json.loads(request.data)
            self.assertEqual(body["model"], profile.served_model)
            return _Response({"count": 7, "tokens": [1] * 7,
                              "max_model_len": profile.context_tokens})

        self.assertEqual(MlxEngine(MACOS).context_probe(profile, "k", urlopen, timeout=5),
                         (profile.context_tokens, 7))


class HelperTests(unittest.TestCase):
    def test_engine_for_rejects_vllm(self) -> None:
        from tests.test_hardware import snapshot

        with self.assertRaises(ValueError):
            engine_for(select_runtime_profile(snapshot(24), environment={}))
        self.assertIsInstance(
            engine_for(select_runtime_profile(macos_snapshot(36), environment={}), MACOS),
            MlxEngine)

    def test_local_api_key_is_minted_once_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state" / "local-api-key"
            first = ensure_local_api_key(path)
            second = ensure_local_api_key(path)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 48)
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_runtime_root_resolution(self) -> None:
        self.assertEqual(runtime_root({"FRIDAY_RUNTIME_ROOT": "/srv/rt"}), Path("/srv/rt"))
        self.assertEqual(runtime_root({}, qwen_root=Path("/opt/friday/runtime/qwen")),
                         Path("/opt/friday/runtime"))


if __name__ == "__main__":
    unittest.main()
