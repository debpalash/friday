"""Every platform lock is hash-pinned, complete, and bound to the ledger."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

from friday_core.dependency_review import parse_lock
from friday_host.host import HostPlatform

REPO = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO / "requirements"
LEDGER = json.loads((REPO / "compliance" / "dependency-review-v1.json").read_text())
PLATFORM_TAGS = {
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "linux-arm64": "aarch64-unknown-linux-gnu",
    "macos-arm64": "aarch64-apple-darwin",
    "windows-x86_64": "x86_64-pc-windows-msvc",
}
CUDA_ONLY = re.compile(r"^(nvidia-|triton==|torch==|torchaudio==|omnivoice==|cuda-)")


def _direct_pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, version = line.split("==", 1)
            pins[name.lower()] = version
    return pins


class LockShapeTests(unittest.TestCase):
    def test_every_lock_is_fully_hashed_and_targets_its_platform(self) -> None:
        for lock in sorted(REQUIREMENTS.glob("*.lock")):
            with self.subTest(lock=lock.name):
                text = lock.read_text()
                header = text.splitlines()[1]
                self.assertIn("--generate-hashes", header)
                for match in re.finditer(r"^[A-Za-z0-9_.-]+==[^\\\s]+", text, re.M):
                    self.assertIn("--hash=sha256:", text[match.end():match.end() + 200],
                                  match.group(0))
                for lock_id, tag in PLATFORM_TAGS.items():
                    if lock.name == f"runtime-{lock_id}.lock":
                        self.assertIn(f"--python-platform {tag}", header)
                if lock.name.startswith("runtime-macos") or lock.name == "mlx-runtime.lock":
                    self.assertIn("aarch64-apple-darwin", header)

    def test_portable_locks_carry_every_direct_pin_and_no_cuda_stack(self) -> None:
        direct = _direct_pins(REQUIREMENTS / "runtime.in")
        for lock_id in PLATFORM_TAGS:
            lock = REQUIREMENTS / f"runtime-{lock_id}.lock"
            with self.subTest(lock=lock.name):
                packages = parse_lock(lock)
                for name, version in direct.items():
                    self.assertEqual(packages.get(name), version, name)
                cuda = [name for name in packages if CUDA_ONLY.match(f"{name}==")]
                self.assertEqual(cuda, [], "CUDA packages leaked into a portable lock")
                self.assertNotIn("mlx", packages)
                self.assertNotIn("silero-vad", packages)

    def test_cuda_lock_is_a_superset_of_the_linux_portable_lock(self) -> None:
        portable = parse_lock(REQUIREMENTS / "runtime-linux-x86_64.lock")
        cuda = parse_lock(REQUIREMENTS / "cuda-linux-x86_64.lock")
        for name, version in portable.items():
            self.assertEqual(cuda.get(name), version, name)
        for name in _direct_pins(REQUIREMENTS / "cuda.in"):
            self.assertIn(name, cuda)

    def test_mlx_runtime_lock_pins_the_engine_table_versions(self) -> None:
        from friday_core.engine_assets import mlx_runtime_pins

        packages = parse_lock(REQUIREMENTS / "mlx-runtime.lock")
        for name, version in mlx_runtime_pins().items():
            self.assertEqual(packages.get(name), version, name)


class LedgerBindingTests(unittest.TestCase):
    def test_every_lock_file_has_a_ledger_entry_and_vice_versa(self) -> None:
        files = {f"requirements/{p.name}" for p in REQUIREMENTS.glob("*.lock")}
        entries = {value["path"] for value in LEDGER["locks"].values()}
        self.assertEqual(files, entries)
        self.assertEqual(LEDGER["review_version"], 2)
        for name, value in LEDGER["locks"].items():
            with self.subTest(name=name):
                path = REPO / value["path"]
                self.assertEqual(value["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(value["packages"], len(parse_lock(path)))
                self.assertIn(value["platform"], {*PLATFORM_TAGS, "macos-x86_64"})
                self.assertTrue(value["engines"])

    def test_binary_assets_match_the_engine_table(self) -> None:
        table = json.loads((REPO / "friday_core" / "engine_assets.json").read_text())
        binaries = {item["name"]: item for item in LEDGER["binary_assets"]}
        entry = next(iter(binaries.values()))
        self.assertEqual(entry["version"], table["llama_server"]["tag"])
        for row in table["llama_server"]["binaries"]:
            key = f"{row['platform']}-{row['arch']}-{row['backend']}"
            self.assertEqual(entry["artifacts"][key], row["sha256"])
        model_names = {item["name"] for item in LEDGER["models_and_assets"]}
        for row in table["models"]:
            self.assertIn(row["repo"], model_names)

    def test_supported_hosts_map_to_an_existing_lock(self) -> None:
        for host_os, arch in (("linux", "x86_64"), ("linux", "aarch64"),
                              ("macos", "aarch64"), ("windows", "x86_64")):
            host = HostPlatform(os=host_os, arch=arch)
            self.assertTrue((REQUIREMENTS / f"runtime-{host.lock_id}.lock").is_file(),
                            host.lock_id)


if __name__ == "__main__":
    unittest.main()
