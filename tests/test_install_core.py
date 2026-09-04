"""The macOS install path: bash preflight, uv bootstrap, Python transaction.

These tests drive the real ``install.sh`` Darwin branch and
``ops/install_core.py`` on any POSIX host with fake ``uname``, ``sw_vers``,
``uv``, ``launchctl``, ``shasum``, and ``curl`` executables on PATH.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from tests.platform_markers import posix_only

ROOT = Path(__file__).resolve().parents[1]

FAKE_UV = """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "${UV_LOG:?}"
case "${1:-}" in
  python)
    if [[ "${2:-}" == find ]]; then printf '%s\\n' "${UV_PYTHON:?}"; fi
    ;;
  venv)
    target="${@: -1}"
    mkdir -p "$target/bin"
    cat > "$target/bin/python" <<'PY'
#!/usr/bin/env bash
if [[ "$*" == *"friday_doctor.py"* && "${FAIL_DOCTOR:-0}" == 1 ]]; then exit 23; fi
exit 0
PY
    chmod 755 "$target/bin/python"
    ;;
esac
exit 0
"""

FAKE_LAUNCHCTL = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "${LAUNCHCTL_LOG:?}"
if [[ "$1" == print ]]; then
  if [[ -f "${LAUNCHCTL_LOADED:?}" ]]; then printf 'state = running\\n'; exit 0; fi
  exit 113
fi
if [[ "$1" == bootstrap ]]; then touch "${LAUNCHCTL_LOADED:?}"; fi
if [[ "$1" == bootout ]]; then rm -f "${LAUNCHCTL_LOADED:?}"; fi
exit 0
"""

FAKE_SHASUM = """#!/usr/bin/env bash
# Emulates `shasum -a 256 -c -` with coreutils.
args=()
for arg in "$@"; do
  case "$arg" in -a|256) ;; *) args+=("$arg") ;; esac
done
exec sha256sum "${args[@]}"
"""


@posix_only
class MacInstallerLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(os.path.realpath(self.temporary.name))
        self.home = self.root / "home"
        self.home.mkdir()
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self.uv_log = self.root / "uv.log"
        self.launchctl_log = self.root / "launchctl.log"
        self._write_executable(self.fake_bin / "uname",
                               '#!/bin/sh\ncase "$1" in -m) echo arm64;; *) echo Darwin;; esac\n')
        self._write_executable(self.fake_bin / "sw_vers", "#!/bin/sh\necho 15.1\n")
        self._write_executable(self.fake_bin / "uv", FAKE_UV)
        self._write_executable(self.fake_bin / "launchctl", FAKE_LAUNCHCTL)
        self._write_executable(self.fake_bin / "shasum", FAKE_SHASUM)
        self.env = os.environ.copy()
        self.env.update({
            "HOME": str(self.home),
            "UV_LOG": str(self.uv_log),
            "UV_PYTHON": sys.executable,
            "LAUNCHCTL_LOG": str(self.launchctl_log),
            "LAUNCHCTL_LOADED": str(self.root / "launchctl.loaded"),
            "PATH": str(self.fake_bin) + os.pathsep + self.env["PATH"],
        })
        # The installer honours XDG_* on every platform so one harness can
        # drive each layout; a developer shell that exports them would point
        # this rehearsal at the real Linux install. Clear everything.
        for key in [k for k in self.env if k.startswith(("FRIDAY_", "XDG_"))]:
            self.env.pop(key)
        self.env["FRIDAY_INSTALL_REHEARSAL"] = "1"
        self.source = self.root / "source"
        self._make_source(self.source)
        self.app = self.home / "Library" / "Application Support" / "Friday" / "app"
        self.state = self.home / "Library" / "Application Support" / "Friday" / "state"
        self.config = self.home / "Library" / "Application Support" / "Friday" / "config"
        self.plist = self.home / "Library" / "LaunchAgents" / "dev.palash.friday.plist"
        self.cli = self.home / ".local" / "bin" / "friday"

    @staticmethod
    def _write_executable(path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        path.chmod(0o755)

    def _make_source(self, target: Path) -> None:
        for relative in ("install.sh", "ops/install_core.py", "ops/fridayctl.py",
                         "ops/friday_launch.py", "ops/friday.launchd.plist.in",
                         "ops/friday.service.in", "requirements/runtime-macos-arm64.lock"):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        shutil.copytree(ROOT / "friday_host", target / "friday_host",
                        ignore=shutil.ignore_patterns("__pycache__"))
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

    def _install(self, *arguments: str, env: dict[str, str] | None = None,
                 source: Path | None = None) -> subprocess.CompletedProcess:
        script = (source or self.source) / "install.sh"
        return subprocess.run(
            ["bash", str(script), *arguments, "--skip-assets", "--skip-hardware-check",
             "--no-start"],
            cwd=self.root, env=env or self.env, text=True, capture_output=True, timeout=300)

    def test_rehearsal_refuses_to_run_without_the_explicit_flag(self) -> None:
        env = dict(self.env)
        env.pop("FRIDAY_INSTALL_REHEARSAL")
        result = self._install("--local", str(self.source), env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FRIDAY_INSTALL_REHEARSAL", result.stderr)
        self.assertFalse((self.app / "releases").exists())

    def test_clean_local_install_registers_a_login_agent(self) -> None:
        result = self._install("--local", str(self.source))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("macOS 15.1 / Apple Silicon", result.stdout)
        self.assertTrue((self.app / "current").is_symlink())
        release = Path(os.path.realpath(self.app / "current"))
        self.assertTrue(release.is_relative_to(self.app / "releases"))
        self.assertTrue((release / "FRIDAY_RELEASE").is_file())
        self.assertEqual(os.readlink(release / "state"), str(self.state))
        self.assertEqual(os.readlink(release / "models"), str(self.app / "shared" / "models"))
        self.assertEqual((self.state / "friday.db").read_bytes(), b"personal-state")
        self.assertTrue((self.app / "shared" / "skills" / "owned.txt").is_file())
        self.assertTrue((self.app / "shared" / "persona" / "voices" / "voice.wav").is_file())

        env_text = (self.config / "friday.env").read_text()
        self.assertEqual((self.config / "friday.env").stat().st_mode & 0o777, 0o600)
        self.assertIn(f"FRIDAY_STATE_DIR='{self.state}'", env_text)
        self.assertIn("FRIDAY_DESKTOP_MODE='off'", env_text)
        self.assertIn("FRIDAY_LLM_ENGINE='auto'", env_text)
        self.assertIn(f"FRIDAY_LOCAL_API_KEY_FILE='{self.state / 'local-api-key'}'", env_text)
        self.assertIn(f"FRIDAY_RUNTIME_ROOT='{self.app / 'runtime'}'", env_text)

        plist = plistlib.loads(self.plist.read_bytes())
        self.assertEqual(plist["ProgramArguments"][0], str(self.app / "current/venv/bin/python"))
        self.assertEqual(plist["ProgramArguments"][-2:],
                         [str(self.app / "current/supervisor.py"), "watch"])
        launchctl = self.launchctl_log.read_text().splitlines()
        self.assertIn(f"bootstrap gui/{os.getuid()} {self.plist}", launchctl)
        self.assertIn(f"enable gui/{os.getuid()}/dev.palash.friday", launchctl)

        cli = self.cli.read_text()
        self.assertIn("ops/fridayctl.py", cli)
        self.assertTrue(self.cli.stat().st_mode & 0o100)
        uv = self.uv_log.read_text()
        self.assertIn("python find 3.12", uv)
        self.assertIn("pip sync --python", uv)
        self.assertIn("runtime-macos-arm64.lock", uv)
        self.assertFalse(list(self.app.glob(".rollback-*")))
        self.assertTrue((self.app / "install.log").is_file())

    def test_failed_doctor_rolls_back_to_the_previous_release(self) -> None:
        first = self._install("--local", str(self.source))
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        previous = os.path.realpath(self.app / "current")
        plist_before = self.plist.read_bytes()
        env = dict(self.env, FAIL_DOCTOR="1")
        second = self._install("--local", str(self.source), "--repair", env=env)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("rollback", second.stderr)
        self.assertEqual(os.path.realpath(self.app / "current"), previous)
        self.assertEqual(self.plist.read_bytes(), plist_before)
        self.assertEqual(len(list((self.app / "releases").iterdir())), 1)
        self.assertFalse(list(self.app.glob(".rollback-*")))

    def test_failed_first_install_removes_the_managed_shell(self) -> None:
        env = dict(self.env, FAIL_DOCTOR="1")
        result = self._install("--local", str(self.source), env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.plist.exists())
        self.assertFalse((self.config / "friday.env").exists())
        self.assertFalse(self.cli.exists())
        self.assertFalse((self.app / "current").exists())
        self.assertIn(f"bootout gui/{os.getuid()}/dev.palash.friday",
                      self.launchctl_log.read_text())

    def test_verified_archive_install_uses_shasum_and_validates_members(self) -> None:
        archive = self.root / "friday-src.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(self.source, arcname="friday-1.0")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        result = self._install("--archive", str(archive), "--source-sha256", digest)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("verified local release archive", result.stdout)
        release = Path(os.path.realpath(self.app / "current"))
        self.assertIn(f"revision=archive-{digest[:12]}", (release / "FRIDAY_RELEASE").read_text())

        wrong = self._install("--archive", str(archive), "--source-sha256", "0" * 64)
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("SHA-256 does not match", wrong.stderr)

        evil = self.root / "evil.tar.gz"
        with tarfile.open(evil, "w:gz") as tar:
            info = tarfile.TarInfo("friday/../escape.txt")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
        evil_digest = hashlib.sha256(evil.read_bytes()).hexdigest()
        rejected = self._install("--archive", str(evil), "--source-sha256", evil_digest)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("unsafe source archive member path", rejected.stderr)
        self.assertFalse((self.root / "escape.txt").exists())

    def test_intel_macs_are_refused_before_any_mutation(self) -> None:
        self._write_executable(self.fake_bin / "uname",
                               '#!/bin/sh\ncase "$1" in -m) echo x86_64;; *) echo Darwin;; esac\n')
        result = self._install("--local", str(self.source))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Apple Silicon", result.stderr)
        self.assertFalse(self.app.exists())
        self.assertFalse(self.launchctl_log.exists())

    def test_reinstalling_the_same_revision_is_a_no_op_without_repair(self) -> None:
        git = shutil.which("git")
        if git is None:
            self.skipTest("environment: git is not installed")
        subprocess.run([git, "init", "-q", str(self.source)], check=True)
        subprocess.run([git, "-C", str(self.source), "add", "."], check=True)
        subprocess.run([git, "-C", str(self.source), "-c", "user.name=t",
                        "-c", "user.email=t@example.invalid", "commit", "-qm", "init"],
                       check=True)
        first = self._install("--local", str(self.source))
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        second = self._install("--local", str(self.source))
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("already installed", second.stdout)
        self.assertEqual(len(list((self.app / "releases").iterdir())), 1)


class UvPinTests(unittest.TestCase):
    def test_install_sh_embeds_the_pinned_darwin_uv_digest(self) -> None:
        pins = json.loads((ROOT / "requirements" / "uv-pins.json").read_text())
        script = (ROOT / "install.sh").read_text()
        version = re.search(r"uv_version=([0-9.]+)", script).group(1)
        digest = re.search(r"uv_arm64_sha256=([0-9a-f]{64})", script).group(1)
        self.assertEqual(version, pins["version"])
        self.assertEqual(digest, pins["archives"]["aarch64-apple-darwin"]["sha256"])
        for target, entry in pins["archives"].items():
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["size"], 10_000_000)
            self.assertIn(target.split("-")[0], entry["file"])


if __name__ == "__main__":
    unittest.main()
