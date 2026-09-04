"""Engine selection keeps Linux/NVIDIA fingerprints and adds portable hosts."""

from __future__ import annotations

import json
import unittest

from friday_core import hardware
from friday_core.hardware import (GIB, Accelerator, HardwareSnapshot,
                                  detect_hardware, select_runtime_profile)
from tests.test_hardware import snapshot

# Recorded before the engine seam existed. Any change here means a proven
# Linux profile would lose its last-known-good calibration evidence.
GOLDEN = {
    24: ("reasoning-24gb",
         "7a9ffb04f6965c35a63a6db4e249093059a0845d485a551f984b8fb98e711b1b",
         "0cc9beebf048924d6c99ba37933dba79af31518c06e602698c6b3b44555c1845"),
    32: ("shared-32gb-native-vision",
         "63ab93a538eaf0641768e10c198c2a803244d28a5e5493321b8ffa5b4d3858ef",
         "fc79dacf425f1e907143344532a9dd2c01fc8f58e52f1d45c7deadf6fb9e72b1"),
    48: ("shared-48gb-native-vision",
         "7f56380b3df5d914327162091dfed5fff44537eb6680f9fd4f5e43f15e61fa8c",
         "f09f37425234c91d87301cad4e5237660eeebddb44da31be4d0d1e6b219b10f2"),
}


def macos_snapshot(memory_gib: int, *, arch: str = "aarch64",
                   cpu_count: int = 10) -> HardwareSnapshot:
    accelerators = ()
    unified = False
    if arch == "aarch64":
        accelerators = (Accelerator("metal", 0, "Apple M3", memory_gib * GIB,
                                    None, identity="Apple M3"),)
        unified = True
    return HardwareSnapshot(
        cpu_count=cpu_count, system_memory_bytes=memory_gib * GIB,
        accelerators=accelerators, cuda_probe="absent", platform="macos",
        arch=arch, unified_memory=unified)


def windows_snapshot(*, cuda_gib: int | None = None, vulkan_gib: int | None = None,
                     memory_gib: int = 32) -> HardwareSnapshot:
    accelerators = []
    if cuda_gib:
        accelerators.append(Accelerator("cuda", 0, "NVIDIA RTX", cuda_gib * GIB,
                                        cuda_gib * GIB, identity="GPU-1"))
    if vulkan_gib:
        accelerators.append(Accelerator("vulkan", 1, "AMD Radeon", vulkan_gib * GIB,
                                        None, identity="AMD Radeon"))
    return HardwareSnapshot(
        cpu_count=16, system_memory_bytes=memory_gib * GIB,
        accelerators=tuple(accelerators),
        cuda_probe="available" if cuda_gib else "absent",
        platform="windows", arch="x86_64")


class LinuxInvarianceTests(unittest.TestCase):
    def test_linux_cuda_fingerprints_are_unchanged_by_the_engine_seam(self) -> None:
        for gib, (name, fingerprint, family) in GOLDEN.items():
            with self.subTest(gib=gib):
                profile = select_runtime_profile(snapshot(gib), environment={})
                self.assertEqual(profile.name, name)
                self.assertEqual(profile.fingerprint, fingerprint)
                self.assertEqual(profile.family_fingerprint, family)
                self.assertEqual(profile.engine, "vllm")
                self.assertNotIn("engine_launch", profile.operational_config())
                self.assertNotIn("FRIDAY_LLM_ENGINE", profile.assistant_environment())
                self.assertEqual(profile.engine_launch()["engine"], "vllm")
                self.assertEqual(profile.to_dict()["engine"], "vllm")

    def test_linux_hardware_fingerprint_ignores_default_platform_fields(self) -> None:
        plain = snapshot(24)
        tagged = HardwareSnapshot(
            cpu_count=plain.cpu_count,
            system_memory_bytes=plain.system_memory_bytes,
            accelerators=plain.accelerators, cuda_probe=plain.cuda_probe,
            platform="linux", arch="x86_64", unified_memory=False)
        self.assertEqual(plain.fingerprint, tagged.fingerprint)
        self.assertNotEqual(plain.fingerprint, macos_snapshot(16).fingerprint)

    def test_linux_without_cuda_keeps_the_unsupported_profile_and_hints(self) -> None:
        bare = HardwareSnapshot(cpu_count=8, system_memory_bytes=32 * GIB,
                                cuda_probe="absent")
        profile = select_runtime_profile(bare, environment={})
        self.assertEqual(profile.name, "unsupported-local-runtime")
        self.assertEqual(profile.engine, "vllm")
        self.assertTrue(any("FRIDAY_LLM_ENGINE=llama_server" in w
                            for w in profile.warnings))
        explicit = select_runtime_profile(
            bare, environment={"FRIDAY_LLM_ENGINE": "llama_server"})
        self.assertEqual(explicit.engine, "llama_server")
        self.assertEqual(explicit.engine_backend, "cpu")
        self.assertTrue(explicit.local_runtime_available)
        self.assertIn("FRIDAY_LLM_ENGINE", explicit.overrides)

    def test_vllm_cannot_be_forced_without_cuda_or_off_linux(self) -> None:
        bare = HardwareSnapshot(cpu_count=8, system_memory_bytes=32 * GIB,
                                cuda_probe="absent")
        with self.assertRaises(ValueError):
            select_runtime_profile(bare, environment={"FRIDAY_LLM_ENGINE": "vllm"})
        with self.assertRaises(ValueError):
            select_runtime_profile(windows_snapshot(cuda_gib=24),
                                   environment={"FRIDAY_LLM_ENGINE": "vllm"})
        with self.assertRaises(ValueError):
            select_runtime_profile(snapshot(24), environment={"FRIDAY_LLM_ENGINE": "gpt"})


class PortableProfileTests(unittest.TestCase):
    def test_apple_silicon_selects_mlx_from_unified_memory(self) -> None:
        profile = select_runtime_profile(macos_snapshot(36), environment={})
        self.assertEqual(profile.engine, "mlx_lm")
        self.assertEqual(profile.engine_backend, "metal")
        self.assertEqual(profile.model_asset, "qwen3-30b-a3b-mlx-4bit")
        self.assertEqual(profile.served_model, "qwen3-30b-a3b")
        self.assertEqual(profile.tts_device, "cpu")
        self.assertIsNone(profile.llm_cuda_device)
        self.assertTrue(profile.local_runtime_available)
        self.assertEqual(profile.name, "mlx-lm-metal-qwen3-30b-a3b-mlx-4bit")
        launch = profile.engine_launch()
        self.assertEqual(launch["engine"], "mlx_lm")
        self.assertIn("--served-model", launch["args"])
        self.assertEqual(profile.assistant_environment()["FRIDAY_LLM_ENGINE"], "mlx_lm")
        self.assertIn("engine_launch", profile.operational_config())
        self.assertEqual(profile.to_dict()["model_path"], profile.qwen_model)
        self.assertTrue(profile.qwen_model.startswith("models/qwen3-30b-a3b-mlx-4bit-"))

    def test_intel_mac_uses_llama_server_on_cpu(self) -> None:
        profile = select_runtime_profile(macos_snapshot(16, arch="x86_64"), environment={})
        self.assertEqual(profile.engine, "llama_server")
        self.assertEqual(profile.engine_backend, "cpu")
        self.assertEqual(profile.gpu_layers, 0)
        self.assertEqual(profile.model_asset, "qwen3-8b-gguf-q4_k_m")

    def test_windows_prefers_cuda_then_vulkan_then_cpu(self) -> None:
        cuda = select_runtime_profile(windows_snapshot(cuda_gib=24), environment={})
        self.assertEqual((cuda.engine, cuda.engine_backend), ("llama_server", "cuda"))
        self.assertEqual(cuda.llm_cuda_device, 0)
        self.assertEqual(cuda.model_asset, "qwen3-30b-a3b-gguf-q4_k_m")
        self.assertEqual(cuda.context_tokens, 32_768)
        self.assertIn("cuda:0", cuda.admission_budget["vram_mib_by_accelerator"])
        args = cuda.engine_launch()["args"]
        self.assertIn("--reasoning-budget", args)
        self.assertEqual(args[args.index("--alias") + 1], "qwen3-30b-a3b")
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")

        vulkan = select_runtime_profile(windows_snapshot(vulkan_gib=16), environment={})
        self.assertEqual(vulkan.engine_backend, "vulkan")
        self.assertEqual(vulkan.model_asset, "qwen3-14b-gguf-q4_k_m")

        cpu = select_runtime_profile(windows_snapshot(memory_gib=16), environment={})
        self.assertEqual(cpu.engine_backend, "cpu")
        self.assertEqual(cpu.model_asset, "qwen3-8b-gguf-q4_k_m")

    def test_explicit_backend_and_engine_overrides(self) -> None:
        forced = select_runtime_profile(
            windows_snapshot(cuda_gib=24), environment={"FRIDAY_LLM_BACKEND": "cpu"})
        self.assertEqual(forced.engine_backend, "cpu")
        self.assertIn("FRIDAY_LLM_BACKEND", forced.overrides)
        with self.assertRaises(ValueError):
            select_runtime_profile(windows_snapshot(cuda_gib=24),
                                   environment={"FRIDAY_LLM_BACKEND": "metal"})
        with self.assertRaises(ValueError):
            select_runtime_profile(windows_snapshot(cuda_gib=24),
                                   environment={"FRIDAY_LLM_ENGINE": "mlx_lm"})
        llama_on_mac = select_runtime_profile(
            macos_snapshot(36), environment={"FRIDAY_LLM_ENGINE": "llama_server"})
        self.assertEqual((llama_on_mac.engine, llama_on_mac.engine_backend),
                         ("llama_server", "metal"))

    def test_vllm_only_overrides_are_rejected_for_portable_engines(self) -> None:
        for variable, value in (("FRIDAY_GPU_UTIL", "0.8"), ("FRIDAY_KV_MODE", "fast"),
                                ("FRIDAY_LLM_CUDA_DEVICES", "0"),
                                ("FRIDAY_NATIVE_VISION", "enabled"),
                                ("FRIDAY_TTS_DEVICE", "cuda")):
            with self.subTest(variable=variable), self.assertRaises(ValueError):
                select_runtime_profile(macos_snapshot(36), environment={variable: value})

    def test_portable_integer_overrides_apply(self) -> None:
        profile = select_runtime_profile(macos_snapshot(36), environment={
            "FRIDAY_MODEL_CONTEXT_TOKENS": "16384", "FRIDAY_MAX_SEQS": "1",
            "FRIDAY_LLM_THREADS": "6", "FRIDAY_LLM_PORT": "18022"})
        self.assertEqual(profile.context_tokens, 16384)
        self.assertEqual(profile.max_sequences, 1)
        self.assertEqual(profile.engine_threads, 6)
        self.assertEqual(profile.llm_port, 18022)
        for name in ("FRIDAY_MODEL_CONTEXT_TOKENS", "FRIDAY_MAX_SEQS",
                     "FRIDAY_LLM_THREADS", "FRIDAY_LLM_PORT"):
            self.assertIn(name, profile.overrides)

    def test_too_little_memory_is_reported_not_crashed(self) -> None:
        profile = select_runtime_profile(macos_snapshot(4), environment={})
        self.assertEqual(profile.name, "unsupported-local-runtime")
        self.assertFalse(profile.local_runtime_available)
        self.assertEqual(profile.engine, "mlx_lm")
        self.assertTrue(any("no Qwen3 tier fits" in w for w in profile.warnings))

    def test_portable_fingerprints_change_with_engine_facts(self) -> None:
        base = select_runtime_profile(macos_snapshot(36), environment={})
        other = select_runtime_profile(macos_snapshot(36),
                                       environment={"FRIDAY_LLM_ENGINE": "llama_server"})
        self.assertNotEqual(base.fingerprint, other.fingerprint)
        self.assertNotEqual(base.family_fingerprint, other.family_fingerprint)
        json.dumps(base.to_dict())


class PortableDetectionTests(unittest.TestCase):
    def test_darwin_probe_parsing(self) -> None:
        calls = []

        def probe(command):
            calls.append(command)
            if command[-1] == "hw.memsize":
                return "38654705664\n"
            if command[-1] == "machdep.cpu.brand_string":
                return "Apple M3 Pro\n"
            raise RuntimeError("unexpected probe")

        found = detect_hardware(probe=probe, platform_name="darwin",
                                machine="arm64", meminfo_path="/missing")
        self.assertEqual(found.platform, "macos")
        self.assertEqual(found.arch, "aarch64")
        self.assertTrue(found.unified_memory)
        self.assertEqual(found.system_memory_bytes, 38654705664)
        self.assertEqual(found.accelerators[0].backend, "metal")
        self.assertEqual(found.accelerators[0].name, "Apple M3 Pro")
        self.assertEqual(found.cuda_probe, "absent")
        self.assertFalse(any("nvidia-smi" in c[0] for c in calls))
        intel = detect_hardware(probe=probe, platform_name="darwin",
                                machine="x86_64", meminfo_path="/missing")
        self.assertEqual(intel.accelerators, ())
        self.assertFalse(intel.unified_memory)

    def test_windows_probe_parsing(self) -> None:
        def nvidia(command):
            if command[0] == "nvidia-smi":
                return "0, GPU-abc, NVIDIA GeForce RTX 4090, 24564, 20000\n"
            raise RuntimeError("unexpected")

        found = detect_hardware(probe=nvidia, platform_name="win32",
                                machine="AMD64", meminfo_path="/missing")
        self.assertEqual(found.platform, "windows")
        self.assertEqual(found.cuda_probe, "available")
        self.assertEqual(found.accelerators[0].backend, "cuda")

        def amd(command):
            if command[0] == "nvidia-smi":
                raise RuntimeError("not installed")
            return json.dumps([{"Name": "AMD Radeon RX 7800", "AdapterRAM": 4293918720},
                               {"Name": "NVIDIA something", "AdapterRAM": 1}])

        found = detect_hardware(probe=amd, platform_name="win32", machine="x86_64",
                                meminfo_path="/missing")
        self.assertEqual(found.accelerators[0].backend, "vulkan")
        self.assertEqual(found.accelerators[0].name, "AMD Radeon RX 7800")
        self.assertEqual(len(found.accelerators), 1)
        self.assertIn(found.cuda_probe, {"absent", "probe_failed"})
        self.assertTrue(found.system_memory_bytes > 0)

    def test_parsers_reject_garbage(self) -> None:
        with self.assertRaises(ValueError):
            hardware.parse_win32_video_controllers("not json")
        self.assertEqual(hardware.parse_win32_video_controllers(""), ())
        with self.assertRaises(ValueError):
            hardware.parse_sysctl_darwin("lots", "Apple", "aarch64")


if __name__ == "__main__":
    unittest.main()
