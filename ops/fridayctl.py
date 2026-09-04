#!/usr/bin/env python3
"""Friday lifecycle commands for hosts without the bash control script.

Linux keeps ``ops/fridayctl``; macOS installs a shim that runs this module
inside the release's Python environment. The subcommands mirror the bash
script so documentation applies to every platform.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_host import desktop_io, paths, procs  # noqa: E402
from friday_host.envfile import read_env_file  # noqa: E402
from friday_host.host import current_host  # noqa: E402
from friday_host.service import backend_for  # noqa: E402

USAGE = """Friday lifecycle commands

Usage: friday <command>

  start       Start Friday and wait for readiness
  stop        Stop Friday and unload its model
  restart     Restart Friday and its model
  status      Show service and runtime status
  doctor      Check hardware, assets, permissions, and health
  logs        Follow Friday's service log
  open        Start Friday and open its desktop window
  update      Install the newest configured Friday release
  repair      Rebuild the installed release from its local source
  export PATH Create a private, hash-bound data export at a new path
  verify-export PATH
              Verify a Friday private export without importing it
  delete ...  Preview or confirm selective private-data deletion
  trust-ca    Add Friday's private CA to your login trust store (asks first)
  untrust-ca  Remove Friday's private CA from your login trust store
  uninstall   Remove Friday's app/runtime; preserve personal data
  version     Print the installed release identity
"""


class Control:
    def __init__(self) -> None:
        self.host = current_host()
        self.install_root = paths.default_install_root()
        self.config_root = paths.default_config_root()
        self.env_file = self.config_root / "friday.env"
        self.values: dict[str, str] = {}
        if self.env_file.is_file():
            self.values = read_env_file(self.env_file)
            for key, value in self.values.items():
                os.environ.setdefault(key, value)
        self.state_root = Path(os.environ.get("FRIDAY_STATE_DIR")
                               or paths.default_state_root())
        self.current = self.install_root / "current"
        self.port = int(os.environ.get("FRIDAY_PORT", "8500"))
        self.url = f"https://127.0.0.1:{self.port}"
        self.ca_file = self.state_root / "tls" / "friday-local-ca.crt"
        self.backend = backend_for(self.host)
        self.python = paths.venv_python(self.current, self.host)

    # ---- helpers
    def need_install(self) -> None:
        if not (self.current.exists() and self.python.is_file()):
            raise SystemExit(f"Friday is not installed at {self.install_root}")

    def healthy(self) -> bool:
        if not self.ca_file.is_file():
            return False
        try:
            context = ssl.create_default_context(cafile=str(self.ca_file))
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}),
                urllib.request.HTTPSHandler(context=context))
            with opener.open(f"{self.url}/healthz", timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    def wait_ready(self) -> None:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.healthy():
                print(f"Friday is ready at {self.url}")
                return
            if not self.backend.is_active():
                print("Friday stopped before becoming ready.", file=sys.stderr)
                self._print_recent_logs()
                raise SystemExit(1)
            time.sleep(2)
        print("Friday did not become ready within 5 minutes.", file=sys.stderr)
        self._print_recent_logs()
        raise SystemExit(1)

    def _print_recent_logs(self) -> None:
        try:
            subprocess.run(self.backend.log_command(follow=False, lines=40),
                           check=False, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass

    def _run_python(self, *arguments: str) -> int:
        return subprocess.call([str(self.python), *arguments])

    # ---- commands
    def start(self) -> int:
        self.need_install()
        self.backend.start()
        self.wait_ready()
        return 0

    def stop(self) -> int:
        self.backend.stop()
        print("Friday stopped; its local model is unloaded.")
        return 0

    def restart(self) -> int:
        self.need_install()
        self.backend.restart()
        self.wait_ready()
        return 0

    def status(self) -> int:
        self.need_install()
        print(self.backend.status_text())
        self._run_python(str(self.current / "supervisor.py"), "status")
        return 0

    def doctor(self) -> int:
        self.need_install()
        return self._run_python(str(self.current / "ops" / "friday_doctor.py"),
                                "--expect-running")

    def logs(self) -> int:
        command = self.backend.log_command(follow=True)
        return subprocess.call(command)

    def _spki_pin(self) -> str | None:
        try:
            manifest = json.loads((self.state_root / "tls" / "active.json").read_text())
            certificate = self.state_root / "tls" / manifest["generation"] / "server-cert.pem"
            from cryptography import x509  # noqa: PLC0415
            from cryptography.hazmat.primitives import serialization  # noqa: PLC0415

            public = x509.load_pem_x509_certificate(certificate.read_bytes()).public_key()
            der = public.public_bytes(serialization.Encoding.DER,
                                      serialization.PublicFormat.SubjectPublicKeyInfo)
            return base64.b64encode(hashlib.sha256(der).digest()).decode("ascii")
        except Exception:
            return None

    def _app_browser(self) -> Path | None:
        candidates: list[Path] = []
        if self.host.is_macos:
            candidates = [
                Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
                Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
                Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            ]
        elif self.host.is_windows:
            for base in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData"):
                root = os.environ.get(base)
                if root:
                    candidates.append(Path(root) / "Google/Chrome/Application/chrome.exe")
                    candidates.append(Path(root) / "Microsoft/Edge/Application/msedge.exe")
        else:
            for name in ("chromium", "google-chrome"):
                found = shutil.which(name)
                if found:
                    candidates.append(Path(found))
        return next((item for item in candidates if item.is_file()), None)

    def open(self) -> int:
        self.need_install()
        if not self.backend.is_active():
            self.start()
        elif not self.healthy():
            self.wait_ready()
        browser = self._app_browser()
        pin = self._spki_pin()
        if browser is not None and pin is not None:
            profile = self.state_root / "ui-browser"
            profile.mkdir(parents=True, exist_ok=True)
            if os.name == "posix":
                profile.chmod(0o700)
            subprocess.Popen(
                [str(browser), f"--app={self.url}", "--class=Friday", "--no-first-run",
                 f"--user-data-dir={profile}",
                 f"--ignore-certificate-errors-spki-list={pin}"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, **procs.detached_popen_kwargs())
            return 0
        desktop_io.open_path(self.url, host=self.host)
        print("Opened Friday in your default browser. If it warns about the "
              "certificate, run `friday trust-ca` once, or install Chrome or "
              "Chromium for the pinned app window.")
        return 0

    def _install_args(self) -> list[str]:
        arguments = ["--root", str(self.install_root), "--config-root", str(self.config_root)]
        if os.environ.get("FRIDAY_STATE_DIR"):
            arguments += ["--state-root", os.environ["FRIDAY_STATE_DIR"]]
        if os.environ.get("FRIDAY_LLM_REPO"):
            arguments += ["--llm-root", os.environ["FRIDAY_LLM_REPO"]]
        return arguments

    def update(self) -> int:
        self.need_install()
        return subprocess.call(["bash", str(self.current / "install.sh"), *self._install_args()])

    def repair(self) -> int:
        self.need_install()
        return subprocess.call(["bash", str(self.current / "install.sh"), "--local",
                                str(self.current), "--repair", *self._install_args()])

    def data(self, verb: str, arguments: list[str]) -> int:
        self.need_install()
        script = str(self.current / "ops" / "friday_data.py")
        if verb == "verify-export":
            return self._run_python(script, "verify-export", *arguments)
        return self._run_python(script, "--state-dir", str(self.state_root), verb, *arguments)

    def uninstall(self, arguments: list[str]) -> int:
        self.need_install()
        return self._run_python(str(self.current / "ops" / "uninstall.py"), *arguments)

    def version(self) -> int:
        self.need_install()
        release = self.current / "FRIDAY_RELEASE"
        print(release.read_text().strip() if release.is_file()
              else self.current.resolve().name)
        return 0

    def trust_ca(self, *, assume_yes: bool) -> int:
        if not self.ca_file.is_file():
            raise SystemExit("Friday's local CA has not been created yet; start Friday first")
        if self.host.is_macos:
            command = ["security", "add-trusted-cert", "-r", "trustRoot", "-k",
                       str(Path.home() / "Library/Keychains/login.keychain-db"),
                       str(self.ca_file)]
        elif self.host.is_windows:
            command = ["certutil", "-user", "-addstore", "Root", str(self.ca_file)]
        else:
            certutil = shutil.which("certutil")
            if not certutil:
                raise SystemExit("certutil (NSS tools) is not installed; import "
                                 f"{self.ca_file} into your browser manually")
            command = [certutil, "-d", f"sql:{Path.home()}/.pki/nssdb", "-A", "-t", "C,,",
                       "-n", "Friday Local Controller CA", "-i", str(self.ca_file)]
        print("This adds Friday's private, loopback-only CA to your login trust store:")
        print("  " + " ".join(command))
        if not assume_yes:
            answer = input("Continue? [y/N] ").strip().lower()
            if not answer.startswith("y"):
                print("cancelled; nothing was changed")
                return 0
        return subprocess.call(command)

    def untrust_ca(self) -> int:
        if self.host.is_macos:
            command = ["security", "delete-certificate", "-c", "Friday Local Controller CA",
                       str(Path.home() / "Library/Keychains/login.keychain-db")]
        elif self.host.is_windows:
            command = ["certutil", "-user", "-delstore", "Root", "Friday Local Controller CA"]
        else:
            certutil = shutil.which("certutil")
            if not certutil:
                raise SystemExit("certutil (NSS tools) is not installed")
            command = [certutil, "-d", f"sql:{Path.home()}/.pki/nssdb", "-D", "-n",
                       "Friday Local Controller CA"]
        return subprocess.call(command)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0
    command, rest = argv[0], argv[1:]
    control = Control()
    simple = {
        "start": control.start, "stop": control.stop, "restart": control.restart,
        "status": control.status, "doctor": control.doctor, "logs": control.logs,
        "open": control.open, "update": control.update, "repair": control.repair,
        "version": control.version, "untrust-ca": control.untrust_ca,
    }
    if command in simple:
        return simple[command]()
    if command == "trust-ca":
        return control.trust_ca(assume_yes="--yes" in rest)
    if command in {"export", "verify-export"}:
        if len(rest) != 1:
            print(f"Usage: friday {command} PATH", file=sys.stderr)
            return 2
        return control.data(command, rest)
    if command == "delete":
        return control.data("delete", rest)
    if command == "uninstall":
        return control.uninstall(rest)
    print(f"Unknown command: {command}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
