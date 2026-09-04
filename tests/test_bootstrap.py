"""The public install bootstrap only runs a checksum-verified release installer."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.platform_markers import require_platform

require_platform("linux")


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "site" / "public" / "install"
WINDOWS_BOOTSTRAP = ROOT / "site" / "public" / "install.ps1"
SITE_HEADERS = ROOT / "site" / "public" / "_headers"
TAG = "v9.9.9-test.1"


class BootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="friday-bootstrap-"))
        self.addCleanup(shutil.rmtree, self.temporary, True)
        self.release = self.temporary / TAG
        self.release.mkdir()
        self.marker = self.temporary / "installer-ran"
        installer = self.release / f"friday-installer-{TAG}.sh"
        installer.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' \"$@\" > '{self.marker}'\n",
            encoding="utf-8",
        )
        self.installer = installer
        self.write_sums(installer.read_bytes())

    def write_sums(self, installer_bytes: bytes) -> None:
        digest = hashlib.sha256(installer_bytes).hexdigest()
        (self.release / "SHA256SUMS").write_text(
            f"{digest}  {self.installer.name}\n"
            f"{'0' * 64}  friday-source-{TAG}.tar.gz\n",
            encoding="utf-8",
        )

    def run_bootstrap(self, *args: str, via_stdin: bool = False,
                      version: str | None = TAG) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "FRIDAY_RELEASE_BASE_URL": self.temporary.as_uri(),
            "FRIDAY_RELEASE_API_URL": "http://127.0.0.1:9/never",
        }
        env.pop("FRIDAY_VERSION", None)
        if version is not None:
            env["FRIDAY_VERSION"] = version
        if via_stdin:
            command = ["bash", "-s", "--", *args]
            stdin = BOOTSTRAP.read_text(encoding="utf-8")
        else:
            command = ["bash", str(BOOTSTRAP), *args]
            stdin = ""
        return subprocess.run(
            command, input=stdin, env=env, capture_output=True, text=True,
            timeout=60, check=False,
        )

    def test_runs_verified_installer_with_arguments(self) -> None:
        result = self.run_bootstrap("--local", "checkout", "--build-venv")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("checksum verified", result.stderr)
        self.assertEqual(
            self.marker.read_text(encoding="utf-8").split(),
            ["--local", "checkout", "--build-venv"],
        )

    def test_works_when_piped_into_bash_like_curl(self) -> None:
        result = self.run_bootstrap("--flag", via_stdin=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.marker.read_text(encoding="utf-8").strip(), "--flag")

    def test_refuses_installer_whose_checksum_does_not_match(self) -> None:
        self.write_sums(b"different bytes")
        result = self.run_bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum verification failed", result.stderr)
        self.assertFalse(self.marker.exists())

    def test_refuses_installer_missing_from_checksum_file(self) -> None:
        (self.release / "SHA256SUMS").write_text(
            f"{'0' * 64}  friday-source-{TAG}.tar.gz\n", encoding="utf-8")
        result = self.run_bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum verification failed", result.stderr)
        self.assertFalse(self.marker.exists())

    def test_refuses_missing_release_assets(self) -> None:
        (self.release / "SHA256SUMS").unlink()
        result = self.run_bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("could not download SHA256SUMS", result.stderr)
        self.assertFalse(self.marker.exists())

    def test_rejects_branch_names_as_versions(self) -> None:
        result = self.run_bootstrap(version="main")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("release tag", result.stderr)
        self.assertFalse(self.marker.exists())

    def test_bootstrap_documents_its_commands(self) -> None:
        text = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn("curl -fsSL https://friday.palash.dev/install | bash", text)
        self.assertTrue(text.rstrip().endswith('main "$@"'))
        self.assertIn("SHA256SUMS", text)
        subprocess.run(["bash", "-n", str(BOOTSTRAP)], check=True)

    @unittest.skipUnless(shutil.which("pwsh"), "environment: PowerShell is not installed")
    def test_windows_bootstrap_parses(self) -> None:
        subprocess.run(
            ["pwsh", "-NoProfile", "-Command",
             "$null = [scriptblock]::Create((Get-Content -Raw "
             f"'{WINDOWS_BOOTSTRAP}'))"],
            check=True, timeout=60,
        )

    def test_windows_bootstrap_hands_off_to_the_linux_bootstrap(self) -> None:
        text = WINDOWS_BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn("irm https://friday.palash.dev/install.ps1 | iex", text)
        self.assertIn("curl -fsSL https://friday.palash.dev/install | bash", text)
        self.assertIn("wsl.exe", text)
        self.assertIn("Read-Host", text)

    def test_public_bootstraps_are_served_as_plain_text(self) -> None:
        headers = SITE_HEADERS.read_text(encoding="utf-8")
        for path in ("/install", "/install.ps1"):
            self.assertIn(
                f"{path}\n  Content-Type: text/plain; charset=utf-8",
                headers)


if __name__ == "__main__":
    unittest.main()
