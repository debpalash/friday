"""The pinned engine table is complete and the tier selector is monotonic."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from friday_core import engine_assets as assets

GIB = assets.GIB
HEX = re.compile(r"[0-9a-f]{64}")


class PinTableTests(unittest.TestCase):
    def test_every_binary_is_unique_per_host_and_fully_pinned(self) -> None:
        seen = set()
        for item in assets.llama_server_binaries():
            key = (item.platform, item.arch, item.backend)
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertTrue(HEX.fullmatch(item.sha256), item.name)
            self.assertGreater(item.size, 1_000_000)
            self.assertTrue(item.url.startswith(
                "https://github.com/ggml-org/llama.cpp/releases/download/"))
            self.assertIn(item.tag, item.url)
            self.assertEqual(item.executable.endswith(".exe"),
                             item.platform == "windows")
            for _name, url, size, digest in item.extra:
                self.assertTrue(HEX.fullmatch(digest))
                self.assertTrue(url.startswith("https://"))
                self.assertGreater(size, 0)
        for host in (("macos", "aarch64"), ("macos", "x86_64"),
                     ("linux", "x86_64"), ("windows", "x86_64"),
                     ("windows", "aarch64")):
            self.assertTrue(assets.engine_backends("llama_server", *host), host)
        self.assertIn("cuda", assets.engine_backends("llama_server", "windows", "x86_64"))
        self.assertIn("vulkan", assets.engine_backends("llama_server", "linux", "x86_64"))
        self.assertEqual(assets.engine_backends("llama_server", "macos", "aarch64"), {"metal"})

    def test_every_model_asset_is_pinned_and_licensed(self) -> None:
        keys = set()
        for item in assets.model_assets():
            self.assertNotIn(item.key, keys)
            keys.add(item.key)
            self.assertEqual(item.license, "Apache-2.0")
            self.assertTrue(re.fullmatch(r"[0-9a-f]{40}", item.revision), item.key)
            self.assertTrue(item.files)
            total = 0
            for name, size, digest in item.files:
                self.assertTrue(HEX.fullmatch(digest), (item.key, name))
                self.assertGreater(size, 0)
                self.assertFalse(name.startswith("/") or ".." in name)
                if name.endswith((".gguf", ".safetensors")):
                    total += size
            self.assertEqual(total, item.weights_bytes)
            self.assertGreater(item.kv_bytes_per_token, 0)
            if item.engine == "llama_server":
                self.assertTrue(item.entry.endswith("Q4_K_M.gguf"))
                self.assertTrue(item.repo.startswith("Qwen/"))
            else:
                self.assertEqual(item.entry, "")
                self.assertTrue(item.repo.startswith("mlx-community/"))
        self.assertEqual(assets.model_asset("qwen3-8b-gguf-q4_k_m").size_label, "8b")
        with self.assertRaises(KeyError):
            assets.model_asset("missing")

    def test_json_data_file_matches_the_loader(self) -> None:
        data = json.loads((Path(assets.__file__).with_name("engine_assets.json")).read_text())
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["llama_server"]["tag"], assets.llama_server_tag())
        self.assertEqual(assets.mlx_runtime_pins(), {"mlx": "0.32.2", "mlx-lm": "0.31.3"})
        self.assertEqual(assets.engine_backends("vllm", "linux", "x86_64"), {"cuda"})
        self.assertEqual(assets.engine_backends("vllm", "macos", "aarch64"), frozenset())
        self.assertEqual(assets.engine_backends("mlx_lm", "windows", "x86_64"), frozenset())


class TierSelectionTests(unittest.TestCase):
    def test_tiers_grow_with_the_budget_and_stay_within_it(self) -> None:
        previous_weights = 0
        for budget_gib in (4, 6, 9, 13, 20, 28, 45, 96):
            with self.subTest(budget_gib=budget_gib):
                tier = assets.select_model_tier("llama_server", budget_gib * GIB)
                if budget_gib < 5:
                    self.assertIsNone(tier)
                    continue
                asset, context, sequences = tier
                self.assertGreaterEqual(context, assets.MINIMUM_CONTEXT)
                self.assertEqual(context & (context - 1), 0, "power of two")
                self.assertIn(sequences, (1, 2, 4))
                needed = (asset.weights_bytes
                          + asset.kv_bytes_per_token * context * sequences
                          + assets.HEADROOM_BYTES)
                self.assertLessEqual(needed, budget_gib * GIB)
                self.assertGreaterEqual(asset.weights_bytes, previous_weights)
                previous_weights = asset.weights_bytes

    def test_expected_sizes_per_budget(self) -> None:
        expectations = {6: "4b", 9: "8b", 13: "14b", 20: "30b-a3b", 28: "32b", 45: "32b"}
        for budget_gib, size in expectations.items():
            tier = assets.select_model_tier("llama_server", budget_gib * GIB)
            self.assertIsNotNone(tier, budget_gib)
            self.assertEqual(tier[0].size_label, size, budget_gib)
        cpu_tier = assets.select_model_tier("llama_server", 45 * GIB, cpu_only=True)
        self.assertEqual(cpu_tier[0].size_label, "30b-a3b")
        mlx = assets.select_model_tier("mlx_lm", 24 * GIB)
        self.assertEqual(mlx[0].engine, "mlx_lm")
        self.assertEqual(mlx[0].size_label, "30b-a3b")
        self.assertIsNone(assets.select_model_tier("mlx_lm", 2 * GIB))

    def test_only_portable_engines_have_tiers(self) -> None:
        with self.assertRaises(ValueError):
            assets.select_model_tier("vllm", 24 * GIB)


if __name__ == "__main__":
    unittest.main()
