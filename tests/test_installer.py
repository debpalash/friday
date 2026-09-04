from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ops import install_asr_model

from tests.platform_markers import require_platform

require_platform("linux")


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


class PortableRuntimeTests(unittest.TestCase):
    def test_runtime_uses_configured_state_root_not_release_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            configured = Path(temporary) / "private-state"
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import server, supervisor; "
                    "print(supervisor.STATE); print(supervisor.SERVER_LOG_FILE); "
                    "print(server.STATE_DIR); print(server.SESSION_FILE); "
                    "print(server.SERVER_LOG_FILE)",
                ],
                cwd=ROOT,
                env=os.environ | {"FRIDAY_STATE_DIR": str(configured)},
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.strip().splitlines()
            expected = str(configured.resolve())
            self.assertEqual(lines, [
                expected,
                str(configured.resolve() / "logs" / "server.log"),
                expected,
                str(configured.resolve() / "session.json"),
                str(configured.resolve() / "logs" / "server.log"),
            ])


class InstallerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self.systemctl_log = self.root / "systemctl.log"
        self.uv_log = self.root / "uv.log"
        self._write_executable(self.fake_bin / "bwrap", "#!/bin/sh\nexit 0\n")
        self._write_executable(
            self.fake_bin / "uv",
            """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "${UV_LOG:?}"
if [[ "${1:-}" == venv ]]; then
  target="${@: -1}"
  mkdir -p "$target/bin"
  cat > "$target/bin/python" <<'PY'
#!/bin/sh
exit 0
PY
  chmod 755 "$target/bin/python"
fi
exit 0
""",
        )
        self._write_executable(
            self.fake_bin / "systemctl",
            """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${SYSTEMCTL_LOG:?}"
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
            "SYSTEMCTL_LOG": str(self.systemctl_log),
            "UV_LOG": str(self.uv_log),
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
            "requirements/runtime-linux-x86_64.lock",
            "requirements/cuda-linux-x86_64.lock",
        ):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        (target / "server.py").write_text("# fake Friday server\n")
        (target / "supervisor.py").write_text("# fake supervisor\n")
        (target / "frontend").mkdir()
        (target / "frontend" / "index.html").write_text("<!doctype html>\n")
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
        self._write_executable(target / "verify.sh", "#!/bin/sh\nexit 0\n")
        model = target / "models" / "Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        (target / "api_key.txt").write_text("private-test-key\n")

    def test_broken_qwen_launcher_is_repaired_not_reused(self):
        self._write_executable(
            self.llm / "venv" / "bin" / "vllm",
            "#!/missing/staging/python\n",
        )
        self._write_executable(
            self.source / "ops" / "provision_qwen_runtime.sh",
            """#!/usr/bin/env bash
set -eu
runtime="$1"
printf '#!/bin/sh\nprintf repaired\\n\n' > "$runtime/venv/bin/vllm"
chmod 755 "$runtime/venv/bin/vllm"
touch "$runtime/provisioner-invoked"
""",
        )

        result = self._install(self.source)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.llm / "provisioner-invoked").is_file())
        launched = subprocess.run(
            [str(self.llm / "venv" / "bin" / "vllm"), "--version"],
            text=True, capture_output=True, timeout=10, check=False)
        self.assertEqual(launched.returncode, 0, launched.stderr)
        self.assertEqual(launched.stdout.strip(), "repaired")

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

    def _make_archive(self, source: Path, name: str = "candidate") -> tuple[Path, str]:
        archive = self.root / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for path in sorted(source.rglob("*")):
                relative = path.relative_to(source)
                if relative.parts[0] in {"venv", "state"}:
                    continue
                bundle.add(path, arcname=str(Path("friday-candidate") / relative),
                           recursive=False)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        return archive, digest

    def _install_archive(
        self, archive: Path, digest: str, *,
        extra_env: dict[str, str] | None = None,
    ):
        return subprocess.run(
            [
                "bash", str(ROOT / "install.sh"),
                "--archive", str(archive), "--source-sha256", digest,
                "--llm-root", str(self.llm), "--skip-assets",
                "--skip-hardware-check", "--no-start",
            ],
            cwd=ROOT, env=self.env | (extra_env or {}), text=True,
            capture_output=True, timeout=60,
        )

    def test_clean_local_install_creates_private_loopback_release(self):
        result = self._install(self.source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        install_root = self.root / "data" / "friday"
        current = install_root / "current"
        self.assertTrue(current.is_symlink())
        self.assertTrue((current / "models").is_symlink())
        self.assertFalse((current / "session.json").exists())
        self.assertFalse((current / "server.log").exists())
        self.assertTrue((current / "frontend" / "index.html").is_file())
        self.assertEqual((self.root / "state" / "friday" / "friday.db").read_bytes(), b"personal-state")
        environment = (self.root / "config" / "friday" / "friday.env").read_text()
        self.assertIn("FRIDAY_BIND_HOST='127.0.0.1'", environment)
        self.assertNotIn("FRIDAY_ALLOWED_HOSTS", environment)
        self.assertNotIn("FRIDAY_ALLOWED_ORIGINS", environment)
        self.assertNotIn("0.0.0.0", environment)
        self.assertEqual((self.root / "user-bin" / "friday").stat().st_mode & 0o777, 0o755)
        self.assertEqual((self.root / "config" / "friday" / "friday.env").stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.llm.joinpath("api_key.txt").stat().st_mode & 0o777, 0o600)
        self.assertTrue(any(
            call.startswith("pip sync ")
            for call in self.uv_log.read_text().splitlines()
        ))

    def test_verified_archive_install_uses_no_source_network_and_lifecycle_works(self):
        archive, digest = self._make_archive(self.source)
        curl_log = self.root / "curl.log"
        self._write_executable(
            self.fake_bin / "curl",
            """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${CURL_LOG:?}"
exit 0
""",
        )

        result = self._install_archive(
            archive, digest, extra_env={"CURL_LOG": str(curl_log)})

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(curl_log.exists())
        current = self.root / "data" / "friday" / "current"
        release = (current / "FRIDAY_RELEASE").read_text()
        self.assertIn(f"revision=archive-{digest[:12]}", release)
        ca = self.root / "state" / "friday" / "tls" / "friday-local-ca.crt"
        ca.parent.mkdir(parents=True)
        ca.write_text("synthetic rehearsal CA\n")
        cli = self.root / "user-bin" / "friday"
        for command in ("start", "restart", "stop"):
            lifecycle = subprocess.run(
                [str(cli), command], cwd=ROOT,
                env=self.env | {"CURL_LOG": str(curl_log)}, text=True,
                capture_output=True, timeout=20)
            self.assertEqual(
                lifecycle.returncode, 0,
                f"{command}: {lifecycle.stdout}{lifecycle.stderr}")
        calls = self.systemctl_log.read_text().splitlines()
        self.assertIn("--user start friday.service", calls)
        self.assertIn("--user restart friday.service", calls)
        self.assertIn("--user stop friday.service", calls)

    def test_archive_digest_failure_precedes_service_or_install_mutation(self):
        archive, digest = self._make_archive(self.source)

        result = self._install_archive(archive, "0" * 64)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA-256 does not match", result.stdout + result.stderr)
        self.assertFalse(self.systemctl_log.exists())
        self.assertFalse((self.root / "data" / "friday").exists())

    def test_archive_link_member_is_rejected_without_escape(self):
        archive = self.root / "hostile.tar.gz"
        link = tarfile.TarInfo("friday-candidate/frontend/index.html")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        with tarfile.open(archive, "w:gz") as bundle:
            root = tarfile.TarInfo("friday-candidate")
            root.type = tarfile.DIRTYPE
            bundle.addfile(root)
            bundle.addfile(link)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()

        result = self._install_archive(archive, digest)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("link or special member", result.stdout + result.stderr)
        self.assertFalse((self.root / "data" / "friday" / "current").exists())

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
        install_root = self.root / "data" / "friday"
        self.assertEqual(list((install_root / "releases").iterdir()), [])
        self.assertEqual(list(install_root.glob(".rollback-*")), [])
        calls = self.systemctl_log.read_text().splitlines()
        enabled = calls.index("--user enable friday.service")
        quiesced = calls.index("--user stop friday.service", enabled + 1)
        self.assertLess(enabled, quiesced)

    def test_uninstall_and_reinstall_preserve_personal_state(self):
        first = self._install(self.source)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        state = self.root / "state" / "friday" / "friday.db"
        before = hashlib.sha256(state.read_bytes()).hexdigest()
        current = self.root / "data" / "friday" / "current"
        uninstall = subprocess.run(
            ["bash", str(current / "scripts" / "uninstall.sh")],
            cwd=ROOT, env=self.env, text=True, capture_output=True, timeout=20)
        self.assertEqual(uninstall.returncode, 0,
                         uninstall.stdout + uninstall.stderr)
        self.assertFalse(current.exists())
        self.assertEqual(hashlib.sha256(state.read_bytes()).hexdigest(), before)

        second = self._install(self.source)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(hashlib.sha256(state.read_bytes()).hexdigest(), before)

    def test_refuses_home_as_install_root(self):
        result = subprocess.run(
            ["bash", str(ROOT / "install.sh"), "--root", str(self.home), "--no-start"],
            cwd=ROOT, env=self.env, text=True, capture_output=True, timeout=20,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsafe install root", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
