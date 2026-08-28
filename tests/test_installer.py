from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ops import install_asr_model


ROOT = Path(__file__).resolve().parents[1]


class AsrInstallerTests(unittest.TestCase):
    def test_model_verification_is_exact_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / install_asr_model.MODEL_NAME
            target.mkdir()
            contents = {"encoder": b"one", "tokens": b"two"}
            assets = {
                name: (len(body), hashlib.sha256(body).hexdigest())
                for name, body in contents.items()
            }
            for name, body in contents.items():
                (target / name).write_bytes(body)
            with mock.patch.object(install_asr_model, "ASSETS", assets):
                self.assertTrue(install_asr_model.valid_model(target))
                (target / "tokens").unlink()
                (target / "tokens").symlink_to(target / "encoder")
                self.assertFalse(install_asr_model.valid_model(target))

    def test_archive_pin_is_the_upstream_release_digest(self):
        self.assertEqual(install_asr_model.ARCHIVE_SIZE, 487_170_055)
        self.assertEqual(
            install_asr_model.ARCHIVE_SHA256,
            "5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf",
        )


class InstallerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self._write_executable(self.fake_bin / "bwrap", "#!/bin/sh\nexit 0\n")
        self._write_executable(
            self.fake_bin / "systemctl",
            """#!/usr/bin/env bash
if [[ "$*" == *"show-environment"* ]]; then exit 0; fi
if [[ "$*" == *"is-active"* || "$*" == *"is-enabled"* ]]; then exit 1; fi
exit 0
""",
        )
        self.env = os.environ.copy()
        self.env.update({
            "HOME": str(self.home),
            "XDG_DATA_HOME": str(self.root / "data"),
            "XDG_STATE_HOME": str(self.root / "state"),
            "XDG_CONFIG_HOME": str(self.root / "config"),
            "XDG_CACHE_HOME": str(self.root / "cache"),
            "XDG_BIN_HOME": str(self.root / "user-bin"),
            "PATH": str(self.fake_bin) + os.pathsep + self.env["PATH"],
        })
        self.source = self.root / "source"
        self.llm = self.root / "qwen"
        self._make_source(self.source)
        self._make_llm(self.llm)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_executable(path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        path.chmod(0o755)

    def _make_source(self, target: Path) -> None:
        for relative in (
            "install.sh", "ops/fridayctl", "ops/friday.service.in",
            "ops/friday.desktop.in", "ops/provision_qwen_runtime.sh",
            "scripts/uninstall.sh", "assets/friday.svg",
        ):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        (target / "server.py").write_text("# fake Friday server\n")
        (target / "supervisor.py").write_text("# fake supervisor\n")
        (target / "ops" / "friday_doctor.py").write_text("# fake doctor\n")
        (target / "state").mkdir()
        (target / "state" / "friday.db").write_bytes(b"personal-state")
        (target / "skills").mkdir()
        (target / "skills" / "owned.txt").write_text("keep")
        (target / "persona" / "voices").mkdir(parents=True)
        (target / "persona" / "voices" / "voice.wav").write_bytes(b"voice")
        self._write_executable(
            target / "venv" / "bin" / "python",
            """#!/usr/bin/env bash
if [[ "$*" == *"friday_doctor.py"* && "${FAIL_DOCTOR:-0}" == 1 ]]; then exit 23; fi
exit 0
""",
        )

    def _make_llm(self, target: Path) -> None:
        self._write_executable(target / "venv" / "bin" / "vllm", "#!/bin/sh\nexit 0\n")
        self._write_executable(target / "single-user" / "start_qwen.sh", "#!/bin/sh\nexit 0\n")
        model = target / "models" / "Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        (target / "api_key.txt").write_text("private-test-key\n")

    def _install(self, source: Path, *, extra_env: dict[str, str] | None = None):
        env = self.env | (extra_env or {})
        return subprocess.run(
            [
                "bash", str(ROOT / "install.sh"), "--local", str(source),
                "--llm-root", str(self.llm), "--skip-assets",
                "--skip-hardware-check", "--no-start",
            ],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=60,
        )

    def test_clean_local_install_creates_private_loopback_release(self):
        result = self._install(self.source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        install_root = self.root / "data" / "friday"
        current = install_root / "current"
        self.assertTrue(current.is_symlink())
        self.assertTrue((current / "models").is_symlink())
        self.assertEqual((self.root / "state" / "friday" / "friday.db").read_bytes(), b"personal-state")
        environment = (self.root / "config" / "friday" / "friday.env").read_text()
        self.assertIn("FRIDAY_BIND_HOST='127.0.0.1'", environment)
        self.assertNotIn("0.0.0.0", environment)
        self.assertEqual((self.root / "user-bin" / "friday").stat().st_mode & 0o777, 0o755)
        self.assertEqual((self.root / "config" / "friday" / "friday.env").stat().st_mode & 0o777, 0o600)

    def test_failed_post_switch_doctor_restores_previous_release(self):
        first = self._install(self.source)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        current = self.root / "data" / "friday" / "current"
        previous = current.resolve()
        config = self.root / "config" / "friday" / "friday.env"
        unit = self.home / ".config" / "systemd" / "user" / "friday.service"
        config.write_text("RESTORED_CONFIG='yes'\n")
        unit.write_text("restored service\n")
        second_source = self.root / "source-two"
        self._make_source(second_source)
        second = self._install(second_source, extra_env={"FAIL_DOCTOR": "1"})
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(current.resolve(), previous)
        self.assertEqual(config.read_text(), "RESTORED_CONFIG='yes'\n")
        self.assertEqual(unit.read_text(), "restored service\n")
        self.assertTrue((self.root / "state" / "friday" / "friday.db").is_file())
        self.assertIn("restoring the previous Friday release", second.stdout + second.stderr)

    def test_failed_first_install_removes_managed_shell(self):
        result = self._install(self.source, extra_env={"FAIL_DOCTOR": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "data" / "friday" / "current").exists())
        self.assertFalse((self.root / "config" / "friday" / "friday.env").exists())
        self.assertFalse((self.root / "user-bin" / "friday").exists())
        self.assertFalse(
            (self.home / ".config" / "systemd" / "user" / "friday.service").exists()
        )

    def test_refuses_home_as_install_root(self):
        result = subprocess.run(
            ["bash", str(ROOT / "install.sh"), "--root", str(self.home), "--no-start"],
            cwd=ROOT, env=self.env, text=True, capture_output=True, timeout=20,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsafe install root", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
