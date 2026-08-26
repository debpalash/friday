import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from friday_core import (Accelerator, HardwareSnapshot, detect_hardware,
                         select_runtime_profile, write_runtime_profile)
from friday_core.hardware import GIB, parse_nvidia_smi


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


class HardwareProfileTests(unittest.TestCase):
    def test_nvidia_inventory_parser_uses_binary_memory_units(self):
        devices = parse_nvidia_smi(
            "0, NVIDIA GeForce RTX 4090, 24564, 20200\n"
            "1, NVIDIA RTX A4000, 16376, 15000\n")

        self.assertEqual(devices[0].backend, "cuda")
        self.assertEqual(devices[0].total_memory_bytes, 24564 * 1024 ** 2)
        self.assertEqual(devices[1].index, 1)

    def test_nvidia_inventory_parser_captures_stable_device_identity(self):
        devices = parse_nvidia_smi(
            "0, GPU-1234, NVIDIA GeForce RTX 4090, 24564, 20200\n")

        self.assertEqual(devices[0].identity, "GPU-1234")

    def test_current_24gb_machine_defaults_to_reasoning_first_profile(self):
        profile = select_runtime_profile(snapshot(24), environment={})

        self.assertEqual(profile.name, "reasoning-24gb")
        self.assertEqual(profile.context_tokens, 200000)
        self.assertEqual(profile.max_sequences, 8)
        self.assertEqual(profile.kv_mode, "huge")
        self.assertEqual(profile.gpu_memory_utilization, 0.93)
        self.assertEqual(profile.asr_threads, 4)
        self.assertEqual(profile.tts_device, "cpu")
        self.assertIsNone(profile.tts_cuda_device)
        self.assertEqual(profile.tts_reserve_gib, 0.0)
        self.assertAlmostEqual(profile.unallocated_gpu_gib, 1.68)
        self.assertTrue(profile.local_runtime_available)

    def test_lower_edge_of_24gb_tier_still_preserves_speech_headroom(self):
        profile = select_runtime_profile(snapshot(23), environment={})

        self.assertEqual(profile.name, "reasoning-24gb")
        self.assertEqual(profile.context_tokens, 200000)
        self.assertEqual(profile.tts_device, "cpu")

    def test_larger_cards_buy_context_without_greedily_taking_all_vram(self):
        small = select_runtime_profile(snapshot(32), environment={})
        middle = select_runtime_profile(snapshot(48), environment={})
        large = select_runtime_profile(snapshot(80, cpu_count=64), environment={})

        self.assertEqual((small.context_tokens, small.kv_mode), (200000, "huge"))
        self.assertAlmostEqual(small.llm_memory_budget_gib, 25.5)
        self.assertEqual(small.tts_device, "cpu")
        self.assertIsNone(small.tts_cuda_device)
        self.assertEqual(small.tts_reserve_gib, 0.0)
        self.assertEqual(
            (small.native_vision_max_images, small.native_vision_max_side),
            (1, 1024))
        self.assertLess(small.gpu_memory_utilization, 0.80)
        self.assertEqual((middle.context_tokens, middle.kv_mode), (200000, "huge"))
        self.assertAlmostEqual(middle.llm_memory_budget_gib, 26.5)
        self.assertEqual(
            (middle.native_vision_max_images, middle.native_vision_max_side),
            (2, 1536))
        self.assertLess(middle.gpu_memory_utilization, 0.60)
        self.assertEqual((large.context_tokens, large.kv_mode), (200000, "huge"))
        self.assertAlmostEqual(large.llm_memory_budget_gib, 30.5)
        self.assertAlmostEqual(large.unallocated_gpu_gib, 49.5)
        self.assertEqual(
            (large.native_vision_max_images, large.native_vision_max_side),
            (4, 2048))
        self.assertEqual(large.tts_reserve_gib, 6.625)
        self.assertEqual(large.asr_threads, 8)

    def test_second_gpu_isolated_for_tts_before_tensor_parallelism(self):
        profile = select_runtime_profile(snapshot(24, 12), environment={})

        self.assertEqual(profile.name, "dedicated-llm-24gb")
        self.assertEqual(profile.llm_cuda_device, 0)
        self.assertEqual(profile.tts_cuda_device, 1)
        self.assertEqual(profile.gpu_memory_utilization, 0.93)
        self.assertEqual(profile.context_tokens, 200000)

    def test_explicit_cuda_tts_keeps_the_voice_balanced_24gb_escape_hatch(self):
        profile = select_runtime_profile(snapshot(24), environment={
            "FRIDAY_TTS_DEVICE": "cuda",
        })

        self.assertEqual(profile.name, "shared-24gb-voice")
        self.assertEqual(profile.context_tokens, 8192)
        self.assertEqual(profile.gpu_memory_utilization, 0.724)
        self.assertEqual(profile.tts_device, "cuda")

    def test_multi_gpu_placement_ignores_transient_free_memory(self):
        busy = select_runtime_profile(
            snapshot(24, 12, free_gib=(23, 1)), environment={})
        idle = select_runtime_profile(
            snapshot(24, 12, free_gib=(1, 11)), environment={})

        self.assertEqual((busy.llm_cuda_device, busy.tts_cuda_device), (0, 1))
        self.assertEqual((idle.llm_cuda_device, idle.tts_cuda_device), (0, 1))
        self.assertEqual(busy.fingerprint, idle.fingerprint)

    def test_explicit_cuda_placement_is_validated_and_controls_profile(self):
        dedicated = select_runtime_profile(snapshot(24, 48), environment={
            "FRIDAY_LLM_CUDA_DEVICES": "0",
            "FRIDAY_TTS_CUDA_DEVICES": "1",
        })
        shared = select_runtime_profile(snapshot(24, 48), environment={
            "FRIDAY_LLM_CUDA_DEVICES": "0",
            "FRIDAY_TTS_CUDA_DEVICES": "0",
        })

        self.assertEqual(dedicated.name, "dedicated-llm-24gb")
        self.assertEqual(
            (dedicated.llm_cuda_device, dedicated.tts_cuda_device), (0, 1))
        self.assertEqual(shared.name, "shared-24gb-voice")
        self.assertEqual((shared.llm_cuda_device, shared.tts_cuda_device), (0, 0))
        with self.assertRaisesRegex(ValueError, "detected devices"):
            select_runtime_profile(snapshot(24), environment={
                "FRIDAY_LLM_CUDA_DEVICES": "2"})

    def test_explicit_homogeneous_device_set_is_tensor_parallel_and_bounded(self):
        profile = select_runtime_profile(snapshot(24, 24), environment={
            "FRIDAY_LLM_CUDA_DEVICES": "1, 0",
        })

        self.assertEqual(profile.effective_llm_cuda_devices, (0, 1))
        self.assertEqual(profile.tensor_parallel_size, 2)
        self.assertEqual(profile.llm_cuda_device, 0)
        self.assertEqual(profile.tts_device, "cpu")
        self.assertIsNone(profile.tts_cuda_device)
        self.assertEqual(profile.context_tokens, 200_000)
        self.assertEqual(profile.gpu_memory_utilization, 0.93)
        self.assertEqual(profile.qwen_environment()["TENSOR_PARALLEL_SIZE"], "2")
        self.assertEqual(profile.to_dict()["llm_total_memory_budget_gib"], 44.64)
        self.assertEqual(profile.admission_budget["vram_mib_by_accelerator"], {
            "cuda:0": 982,
            "cuda:1": 982,
        })
        self.assertEqual(
            profile.operational_config()["llm_cuda_devices"], [0, 1])
        self.assertIn("FRIDAY_LLM_CUDA_DEVICES", profile.overrides)

    def test_tensor_parallel_devices_must_be_unique_detected_and_homogeneous(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_runtime_profile(snapshot(24, 24), environment={
                "FRIDAY_LLM_CUDA_DEVICES": "0,0"})
        with self.assertRaisesRegex(ValueError, "undetected"):
            select_runtime_profile(snapshot(24, 24), environment={
                "FRIDAY_LLM_CUDA_DEVICES": "0,2"})
        with self.assertRaisesRegex(ValueError, "identical total VRAM"):
            select_runtime_profile(snapshot(24, 48), environment={
                "FRIDAY_LLM_CUDA_DEVICES": "0,1"})

    def test_tensor_parallel_profile_uses_third_gpu_for_speech(self):
        profile = select_runtime_profile(snapshot(24, 24, 12), environment={
            "FRIDAY_LLM_CUDA_DEVICES": "0,1",
        })

        self.assertEqual(profile.effective_llm_cuda_devices, (0, 1))
        self.assertEqual(profile.tts_cuda_device, 2)
        self.assertEqual(profile.name,
                         "tensor-parallel-2x-dedicated-llm-24gb")
        self.assertEqual(profile.admission_budget["vram_mib_by_accelerator"], {
            "cuda:0": 982,
            "cuda:1": 982,
            "cuda:2": 4_992,
        })

    def test_single_device_operational_fingerprint_shape_is_backward_compatible(self):
        profile = select_runtime_profile(snapshot(24), environment={})

        self.assertNotIn("llm_cuda_devices", profile.operational_config())
        self.assertNotIn("tensor_parallel_size", profile.operational_config())
        self.assertNotIn("TENSOR_PARALLEL_SIZE", profile.qwen_environment())
        self.assertNotIn("native_vision", profile.operational_config())
        legacy_family = {
            "qwen_model": profile.qwen_model,
            "served_model": profile.served_model,
            "llm_host": profile.llm_host,
            "llm_port": profile.llm_port,
            "llm_cuda_device": profile.llm_cuda_device,
            "tts_device": profile.tts_device,
            "tts_cuda_device": profile.tts_cuda_device,
            "local_runtime_available": profile.local_runtime_available,
            "launch_override_fingerprint": profile.launch_override_fingerprint,
        }
        encoded = json.dumps(
            legacy_family, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(profile.family_fingerprint,
                         hashlib.sha256(encoded).hexdigest())

    def test_native_vision_is_hardware_adaptive_explicit_and_memory_bound(self):
        current = select_runtime_profile(snapshot(24), environment={})
        automatic = select_runtime_profile(snapshot(32), environment={})
        disabled = select_runtime_profile(snapshot(48), environment={
            "FRIDAY_NATIVE_VISION": "disabled",
        })
        custom_auto = select_runtime_profile(snapshot(48), environment={
            "FRIDAY_QWEN_MODEL": "models/text-only-custom",
        })
        custom_explicit = select_runtime_profile(snapshot(48), environment={
            "FRIDAY_QWEN_MODEL": "models/custom-vision",
            "FRIDAY_NATIVE_VISION": "enabled",
        })

        self.assertFalse(current.native_vision_enabled)
        self.assertTrue(automatic.native_vision_enabled)
        self.assertFalse(disabled.native_vision_enabled)
        self.assertFalse(custom_auto.native_vision_enabled)
        self.assertTrue(custom_explicit.native_vision_enabled)
        self.assertEqual(
            automatic.assistant_environment()["FRIDAY_NATIVE_VISION"], "1")
        self.assertEqual(automatic.native_vision_host_reserve_mib, 2048)
        with self.assertRaisesRegex(ValueError, "at least 30 GiB"):
            select_runtime_profile(snapshot(24), environment={
                "FRIDAY_NATIVE_VISION": "enabled",
            })
        with self.assertRaisesRegex(ValueError, "auto, enabled, or disabled"):
            select_runtime_profile(snapshot(48), environment={
                "FRIDAY_NATIVE_VISION": "perhaps",
            })

    def test_native_vision_respects_explicit_speech_and_model_memory_priority(self):
        speech_first = select_runtime_profile(snapshot(80), environment={
            "FRIDAY_TTS_RESERVE_GIB": "55",
        })
        model_limit = select_runtime_profile(snapshot(48), environment={
            "FRIDAY_GPU_UTIL": "0.50",
        })
        self.assertFalse(speech_first.native_vision_enabled)
        self.assertFalse(model_limit.native_vision_enabled)
        self.assertTrue(speech_first.warnings)
        self.assertTrue(model_limit.warnings)
        with self.assertRaisesRegex(ValueError, "shared speech reserves"):
            select_runtime_profile(snapshot(80), environment={
                "FRIDAY_TTS_RESERVE_GIB": "55",
                "FRIDAY_NATIVE_VISION": "enabled",
            })
        with self.assertRaisesRegex(ValueError, "memory envelope"):
            select_runtime_profile(snapshot(48), environment={
                "FRIDAY_GPU_UTIL": "0.50",
                "FRIDAY_NATIVE_VISION": "enabled",
            })

    def test_explicit_overrides_win_and_reach_both_process_environments(self):
        profile = select_runtime_profile(snapshot(48), environment={
            "FRIDAY_LOCAL_MODEL": "custom-model",
            "FRIDAY_LLM_PORT": "19000",
            "FRIDAY_MODEL_CONTEXT_TOKENS": "131072",
            "FRIDAY_MAX_SEQS": "12",
            "FRIDAY_GPU_UTIL": "0.61",
            "FRIDAY_KV_MODE": "long",
            "FRIDAY_TTS_DEVICE": "cpu",
            "FRIDAY_ASR_THREADS": "11",
        })

        self.assertEqual(profile.local_base_url, "http://127.0.0.1:19000/v1")
        self.assertEqual(profile.context_tokens, 131072)
        self.assertEqual(profile.kv_mode, "long")
        self.assertEqual(profile.gpu_memory_utilization, 0.61)
        self.assertEqual(profile.qwen_environment()["MAX_SEQS"], "12")
        self.assertEqual(profile.assistant_environment()["FRIDAY_LOCAL_MODEL"],
                         "custom-model")
        self.assertEqual(profile.assistant_environment()["FRIDAY_TTS_DEVICE"], "cpu")
        self.assertEqual(profile.assistant_environment()["FRIDAY_ASR_THREADS"], "11")

    def test_probe_failure_with_nvidia_device_falls_back_safely(self):
        with tempfile.TemporaryDirectory() as temporary:
            device_root = Path(temporary)
            (device_root / "nvidia0").touch()
            hardware = detect_hardware(
                probe=lambda _command: (_ for _ in ()).throw(
                    RuntimeError("driver offline")),
                meminfo_path="/missing/meminfo", drm_root="/missing/drm",
                device_root=device_root)
        profile = select_runtime_profile(hardware, environment={})

        self.assertEqual(hardware.cuda_probe, "probe_failed")
        self.assertEqual(profile.name, "fallback-24gb")
        self.assertEqual(profile.source, "fallback")
        self.assertTrue(profile.local_runtime_available)
        self.assertTrue(profile.warnings)
        self.assertTrue(profile.hardware.detection_errors)

    def test_confirmed_missing_cuda_disables_local_runtime(self):
        hardware = detect_hardware(
            probe=lambda _command: (_ for _ in ()).throw(
                RuntimeError("not installed")),
            meminfo_path="/missing/meminfo", drm_root="/missing/drm",
            device_root="/missing/dev")
        profile = select_runtime_profile(hardware, environment={})

        self.assertEqual(hardware.cuda_probe, "absent")
        self.assertEqual(profile.name, "unsupported-local-runtime")
        self.assertEqual(profile.source, "unsupported")
        self.assertFalse(profile.local_runtime_available)
        self.assertIsNone(profile.llm_cuda_device)
        self.assertTrue(profile.warnings)

    def test_unprobed_cuda_can_only_be_assumed_explicitly_when_absent(self):
        hardware = HardwareSnapshot(
            cpu_count=16, system_memory_bytes=32 * GIB, cuda_probe="absent")
        profile = select_runtime_profile(hardware, environment={
            "FRIDAY_ALLOW_UNPROBED_CUDA": "true",
            "FRIDAY_LLM_CUDA_DEVICES": "3",
            "FRIDAY_TTS_CUDA_DEVICES": "4",
        })

        self.assertTrue(profile.local_runtime_available)
        self.assertEqual(profile.name, "fallback-24gb")
        self.assertEqual((profile.llm_cuda_device, profile.tts_cuda_device), (3, 4))

    def test_tts_reserve_is_fixed_headroom_and_never_expands_tier_budget(self):
        default = select_runtime_profile(snapshot(80), environment={})
        same_reserve = select_runtime_profile(snapshot(80), environment={
            "FRIDAY_TTS_RESERVE_GIB": "6.625"})
        larger_reserve = select_runtime_profile(snapshot(80), environment={
            "FRIDAY_TTS_RESERVE_GIB": "55"})

        self.assertEqual(default.llm_memory_budget_gib, 30.5)
        self.assertEqual(default.unallocated_gpu_gib, 49.5)
        self.assertEqual(same_reserve.llm_memory_budget_gib, 30.5)
        self.assertEqual(same_reserve.unallocated_gpu_gib, 49.5)
        self.assertEqual(larger_reserve.llm_memory_budget_gib, 22.5)
        self.assertEqual(larger_reserve.unallocated_gpu_gib, 57.5)

    def test_gpu_util_override_recomputes_headroom_and_protects_shared_tts(self):
        profile = select_runtime_profile(snapshot(48), environment={
            "FRIDAY_GPU_UTIL": "0.61"})

        self.assertAlmostEqual(profile.llm_memory_budget_gib, 29.28)
        self.assertAlmostEqual(profile.unallocated_gpu_gib, 18.72)
        with self.assertRaisesRegex(ValueError, "TTS_RESERVE"):
            select_runtime_profile(snapshot(48), environment={
                "FRIDAY_GPU_UTIL": "0.90"})

    def test_admission_budget_reserves_cpu_ram_and_each_cuda_device(self):
        profile = select_runtime_profile(snapshot(24, 12), environment={})

        self.assertEqual(profile.admission_budget, {
            # 32 cores minus ceil(10%), with the reserve larger than 2 cores.
            "cpu_cores": 28.0,
            # 64 GiB minus ceil(10%), with the reserve larger than 2 GiB.
            "ram_mib": 58_982,
            # GPU 0: 24 GiB - ceil(22.32 GiB LLM) - ceil(3% guard).
            # GPU 1: 12 GiB - 6.625 GiB TTS - the 512 MiB guard floor.
            "vram_mib_by_accelerator": {
                "cuda:0": 982,
                "cuda:1": 4_992,
            },
            "concurrency_slots": 8,
            "network_slots": 4,
        })

    def test_admission_budget_applies_minimum_reserves_and_vram_floor(self):
        hardware = HardwareSnapshot(
            cpu_count=6,
            system_memory_bytes=8 * GIB,
            accelerators=(Accelerator(
                "cuda", 0, "GPU 0", 24 * GIB, 24 * GIB),),
            cuda_probe="available",
        )
        profile = select_runtime_profile(hardware, environment={
            "FRIDAY_TTS_DEVICE": "cuda",
        })

        self.assertEqual(profile.admission_budget, {
            "cpu_cores": 4.0,
            "ram_mib": 6_144,
            # The LLM, shared TTS reserve, and guard exceed remaining VRAM.
            "vram_mib_by_accelerator": {"cuda:0": 0},
            "concurrency_slots": 2,
            "network_slots": 2,
        })

    def test_admission_budget_is_serialized_and_changes_the_fingerprint(self):
        first = select_runtime_profile(snapshot(24, cpu_count=32), environment={})
        second = select_runtime_profile(snapshot(24, cpu_count=33), environment={})

        self.assertEqual(
            first.operational_config()["admission_budget"],
            first.admission_budget,
        )
        self.assertEqual(first.to_dict()["admission_budget"],
                         first.admission_budget)
        # Both profiles otherwise resolve the same model and assistant runtime;
        # the deterministic action budget is therefore what changes this hash.
        first_operations = first.operational_config() | {
            "admission_budget": second.admission_budget,
        }
        self.assertEqual(first_operations, second.operational_config())
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_admission_budget_ignores_transient_cuda_free_memory(self):
        busy = select_runtime_profile(
            snapshot(24, 12, free_gib=(1, 11)), environment={})
        idle = select_runtime_profile(
            snapshot(24, 12, free_gib=(23, 1)), environment={})

        self.assertEqual(busy.admission_budget, idle.admission_budget)
        self.assertEqual(busy.operational_config(), idle.operational_config())
        self.assertEqual(busy.fingerprint, idle.fingerprint)

    def test_profile_fingerprint_covers_operations_not_diagnostics(self):
        first = select_runtime_profile(snapshot(24, free_gib=(24,)), environment={})
        second_hardware = HardwareSnapshot(
            cpu_count=32, system_memory_bytes=64 * GIB,
            accelerators=(Accelerator(
                "cuda", 0, "renamed", 24 * GIB, 1 * GIB),),
            detection_errors=("transient diagnostic",), cuda_probe="available")
        second = select_runtime_profile(second_hardware, environment={})
        changed = select_runtime_profile(snapshot(24), environment={
            "FRIDAY_MODEL_CONTEXT_TOKENS": "16384"})

        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, changed.fingerprint)
        self.assertEqual(first.to_dict()["fingerprint"], first.fingerprint)
        self.assertIn("qwen_environment", first.operational_config())

    def test_invalid_override_is_rejected_instead_of_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "FRIDAY_KV_MODE"):
            select_runtime_profile(snapshot(24), environment={
                "FRIDAY_KV_MODE": "mystery"})
        with self.assertRaisesRegex(ValueError, "loopback"):
            select_runtime_profile(snapshot(24), environment={
                "FRIDAY_LLM_HOST": "192.168.1.10"})
        with self.assertRaisesRegex(ValueError, "safe model identifier"):
            select_runtime_profile(snapshot(24), environment={
                "FRIDAY_LOCAL_MODEL": "model --host 0.0.0.0"})
        with self.assertRaisesRegex(ValueError, "FRIDAY_TTS_DEVICE"):
            select_runtime_profile(snapshot(24), environment={
                "FRIDAY_TTS_DEVICE": "cuda:1"})

    def test_localhost_model_override_is_canonicalized_to_numeric_loopback(self):
        profile = select_runtime_profile(snapshot(24), environment={
            "FRIDAY_LLM_HOST": "localhost",
        })

        self.assertEqual(profile.llm_host, "127.0.0.1")
        self.assertIn("FRIDAY_LLM_HOST", profile.overrides)

    def test_resolved_manifest_contains_diagnostics_but_no_api_key(self):
        profile = select_runtime_profile(snapshot(24), environment={
            "FRIDAY_LOCAL_API_KEY": "must-not-leak"})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime-resolved.json"
            write_runtime_profile(path, profile)
            body = path.read_text()

        self.assertEqual(json.loads(body)["name"], "reasoning-24gb")
        self.assertNotIn("must-not-leak", body)
        self.assertNotIn("api_key", body.lower())

    def test_resolved_manifest_does_not_expose_stable_device_identity(self):
        hardware = HardwareSnapshot(
            cpu_count=32, system_memory_bytes=64 * GIB,
            accelerators=(Accelerator(
                "cuda", 0, "GPU", 24 * GIB, 20 * GIB,
                "GPU-private-uuid"),),
            cuda_probe="available")
        profile = select_runtime_profile(hardware, environment={})

        serialized = json.dumps(profile.to_dict())

        self.assertNotIn("GPU-private-uuid", serialized)
        self.assertNotIn(profile.hardware_fingerprint, serialized)
        self.assertNotIn(profile.family_fingerprint, serialized)


if __name__ == "__main__":
    unittest.main()
