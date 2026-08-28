"""Clean, synthetic-host rehearsal of the published-style installer lifecycle."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        item.stat(follow_symlinks=False).st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def _run(
    command: list[str], *, cwd: Path, environment: dict[str, str],
    timeout: int = 120,
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command, cwd=cwd, env=environment, text=True,
        capture_output=True, timeout=timeout, check=False)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return completed, elapsed_ms


class InstallerRehearsalRunner:
    """Exercise install through reinstall without real services or private data."""

    def __init__(self, repo: str | Path):
        self.repo = Path(repo).resolve()

    def run(self, archive: Path, source_sha256: str) -> dict[str, Any]:
        if (not archive.is_file() or archive.is_symlink()
                or hashlib.sha256(archive.read_bytes()).hexdigest()
                != source_sha256):
            raise ValueError("rehearsal source archive identity is invalid")
        root_path: Path | None = None
        with tempfile.TemporaryDirectory(prefix="friday-clean-host-") as value:
            root_path = Path(value)
            root_path.chmod(0o700)
            home = root_path / "home"
            fake_bin = root_path / "bin"
            home.mkdir(mode=0o700)
            fake_bin.mkdir(mode=0o700)
            systemctl_log = root_path / "systemctl.log"
            curl_log = root_path / "curl.log"
            _write_executable(fake_bin / "bwrap", "#!/bin/sh\nexit 0\n")
            _write_executable(
                fake_bin / "systemctl",
                """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${SYSTEMCTL_LOG:?}"
if [[ "$*" == *"show-environment"* ]]; then exit 0; fi
if [[ "$*" == *"is-active"* || "$*" == *"is-enabled"* ]]; then exit 1; fi
exit 0
""",
            )
            _write_executable(
                fake_bin / "curl",
                """#!/usr/bin/env bash
for value in "$@"; do
  case "$value" in http://*|https://*) printf '%s\n' "$value" >> "${CURL_LOG:?}";; esac
done
exit 0
""",
            )
            _write_executable(
                fake_bin / "uv",
                """#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == venv ]]; then
  target="${@: -1}"
  mkdir -p "$target/bin"
  cat > "$target/bin/python" <<'PY'
#!/usr/bin/env bash
if [[ "$*" == *"friday_doctor.py"* && "${FAIL_DOCTOR:-0}" == 1 ]]; then
  exit 23
fi
exit 0
PY
  chmod 755 "$target/bin/python"
fi
exit 0
""",
            )
            llm = root_path / "qwen"
            _write_executable(llm / "venv" / "bin" / "vllm",
                              "#!/bin/sh\nexit 0\n")
            _write_executable(llm / "single-user" / "start_qwen.sh",
                              "#!/bin/sh\nexit 0\n")
            _write_executable(llm / "verify.sh", "#!/bin/sh\nexit 0\n")
            model = (llm / "models"
                     / "Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound")
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (llm / "api_key.txt").write_text(
                "synthetic-rehearsal-key\n", encoding="utf-8")
            environment = os.environ.copy()
            environment.update({
                "HOME": str(home),
                "XDG_DATA_HOME": str(root_path / "data"),
                "XDG_STATE_HOME": str(root_path / "state"),
                "XDG_CONFIG_HOME": str(root_path / "config"),
                "XDG_CACHE_HOME": str(root_path / "cache"),
                "XDG_BIN_HOME": str(root_path / "user-bin"),
                "SYSTEMCTL_LOG": str(systemctl_log),
                "CURL_LOG": str(curl_log),
                "PATH": str(fake_bin) + os.pathsep + environment["PATH"],
            })
            install = [
                "bash", str(self.repo / "install.sh"),
                "--archive", str(archive),
                "--source-sha256", source_sha256,
                "--llm-root", str(llm), "--skip-assets",
                "--skip-hardware-check", "--no-start",
                "--owner", "Rehearsal",
            ]
            first, install_ms = _run(
                install, cwd=self.repo, environment=environment)
            if first.returncode != 0:
                raise RuntimeError("clean archive installation failed")
            install_root = root_path / "data" / "friday"
            state_root = root_path / "state" / "friday"
            config_root = root_path / "config" / "friday"
            current = install_root / "current"
            initial_release = current.resolve()
            state_file = state_root / "friday.db"
            state_file.write_bytes(b"synthetic-state-preservation-canary")
            state_sha256 = hashlib.sha256(state_file.read_bytes()).hexdigest()
            ca = state_root / "tls" / "friday-local-ca.crt"
            ca.parent.mkdir(parents=True)
            ca.write_text("synthetic rehearsal CA\n", encoding="utf-8")
            cli = root_path / "user-bin" / "friday"
            lifecycle: dict[str, float] = {}
            for command in ("start", "restart", "stop"):
                outcome, elapsed = _run(
                    [str(cli), command], cwd=self.repo,
                    environment=environment, timeout=30)
                if outcome.returncode != 0:
                    raise RuntimeError(f"rehearsal {command} failed")
                lifecycle[f"{command}_ms"] = round(elapsed, 3)
            disk = {
                "application_bytes": _tree_bytes(install_root),
                "state_bytes": _tree_bytes(state_root),
                "configuration_bytes": _tree_bytes(config_root),
            }
            failed, rollback_ms = _run(
                install, cwd=self.repo,
                environment=environment | {"FAIL_DOCTOR": "1"})
            rollback_passed = bool(
                failed.returncode != 0
                and current.resolve() == initial_release
                and hashlib.sha256(state_file.read_bytes()).hexdigest()
                    == state_sha256
                and "restoring the previous Friday release" in (
                    failed.stdout + failed.stderr))
            uninstall, uninstall_ms = _run(
                ["bash", str(current / "scripts" / "uninstall.sh")],
                cwd=self.repo, environment=environment, timeout=30)
            uninstall_preserved = bool(
                uninstall.returncode == 0 and not current.exists()
                and hashlib.sha256(state_file.read_bytes()).hexdigest()
                    == state_sha256)
            reinstalled, reinstall_ms = _run(
                install, cwd=self.repo, environment=environment)
            reinstall_preserved = bool(
                reinstalled.returncode == 0 and current.is_symlink()
                and hashlib.sha256(state_file.read_bytes()).hexdigest()
                    == state_sha256)
            contacted_urls = (
                curl_log.read_text(encoding="utf-8").splitlines()
                if curl_log.exists() else [])
            contacted_hosts = sorted({
                urllib.parse.urlsplit(url).hostname or ""
                for url in contacted_urls if url
            })
            external_hosts = [host for host in contacted_hosts
                              if host not in {"127.0.0.1", "localhost", "::1"}]
            systemctl_calls = (
                systemctl_log.read_text(encoding="utf-8").splitlines()
                if systemctl_log.exists() else [])
            checks = {
                "verified_archive_installed": current.is_symlink(),
                "first_boot": "--user start friday.service" in systemctl_calls,
                "restart": "--user restart friday.service" in systemctl_calls,
                "stop": "--user stop friday.service" in systemctl_calls,
                "failed_update_rolled_back": rollback_passed,
                "uninstall_preserved_state": uninstall_preserved,
                "reinstall_preserved_state": reinstall_preserved,
                "no_external_source_contact": not external_hosts,
            }
            result = {
                "environment": "isolated_linux_x86_64_systemd_fixture",
                "source_sha256": source_sha256,
                "archive_bytes": archive.stat().st_size,
                "timing": {
                    "install_ms": round(install_ms, 3),
                    **lifecycle,
                    "failed_update_rollback_ms": round(rollback_ms, 3),
                    "uninstall_ms": round(uninstall_ms, 3),
                    "reinstall_ms": round(reinstall_ms, 3),
                },
                "disk": disk,
                "contacted_hosts": contacted_hosts,
                "external_contacted_hosts": external_hosts,
                "checks": checks,
                "passed": all(checks.values()),
                "privacy": {
                    "fixture_only": True,
                    "user_content_used": False,
                    "raw_command_output_retained": False,
                },
            }
        result["privacy"]["fixture_cleanup_verified"] = bool(
            root_path is not None and not root_path.exists())
        result["passed"] = bool(
            result["passed"] and result["privacy"]["fixture_cleanup_verified"])
        return result
