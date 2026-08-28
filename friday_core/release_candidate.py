"""Private, hash-bound release-candidate gate orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dependency_review import run_dependency_review
from .installer_rehearsal import InstallerRehearsalRunner


EVALUATION_COMMANDS = (
    ("voice", "ops/run_voice_evals.py"),
    ("cognitive_core", "ops/run_cognitive_evals.py"),
    ("memory", "ops/run_memory_evals.py"),
    ("semantic_memory", "ops/run_semantic_memory_evals.py"),
    ("semantic_scale", "ops/run_semantic_scale_evals.py"),
    ("documents", "ops/run_document_reasoning_evals.py"),
    ("long_horizon_project", "ops/run_project_evals.py"),
    ("controller_browser", "ops/run_controller_browser_evals.py"),
    ("recovery", "ops/run_recovery_evals.py"),
    ("adversarial", "ops/run_adversarial_evals.py"),
    ("live_conversation", "ops/run_conversation_evals.py"),
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluation_passed(value: dict[str, Any]) -> bool:
    passed = value.get("passed")
    if isinstance(passed, bool):
        return passed
    total = value.get("total")
    return (isinstance(passed, int) and not isinstance(passed, bool)
            and isinstance(total, int) and not isinstance(total, bool)
            and passed == total and total > 0)


def write_private_candidate_report(path: Path, report: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("release-candidate report target already exists")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = parent.stat()
    if (metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077):
        raise PermissionError("release-candidate report directory must be private")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".release-candidate-", dir=parent)
    try:
        os.fchmod(descriptor, 0o600)
        payload = (json.dumps(report, indent=2, sort_keys=True)
                   + "\n").encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class ReleaseCandidateRunner:
    def __init__(
        self, repo: str | Path, *, app_python: Path, qwen_python: Path,
    ) -> None:
        self.repo = Path(repo).resolve()
        # A virtualenv's Python is commonly a symlink to the base interpreter.
        # Preserve the launcher path so Python retains the virtualenv prefix and
        # site-packages when a gate executes it.
        self.app_python = Path(os.path.abspath(app_python))
        self.qwen_python = Path(os.path.abspath(qwen_python))

    def _command(
        self, name: str, command: list[str], *, environment: dict[str, str],
        timeout: int,
    ) -> dict[str, Any]:
        started = time.perf_counter_ns()
        completed = subprocess.run(
            command, cwd=self.repo, env=environment,
            text=True, capture_output=True, timeout=timeout, check=False)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        output = (completed.stdout + completed.stderr).encode("utf-8")
        return {
            "name": name,
            "passed": completed.returncode == 0,
            "exit_code": completed.returncode,
            "duration_ms": round(elapsed_ms, 3),
            "output_sha256": hashlib.sha256(output).hexdigest(),
            "output_bytes": len(output),
        }

    def _evaluation(
        self, name: str, script: str, *, environment: dict[str, str],
    ) -> dict[str, Any]:
        started = time.perf_counter_ns()
        completed = subprocess.run(
            [str(self.app_python), str(self.repo / script)],
            cwd=self.repo, env=environment, text=True,
            capture_output=True, timeout=900, check=False)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        output = completed.stdout.encode("utf-8")
        try:
            parsed = json.loads(completed.stdout)
            graded = evaluation_passed(parsed)
            suite = str(parsed.get("suite") or name)[:160]
            version = parsed.get("version")
        except (json.JSONDecodeError, AttributeError, TypeError):
            graded = False
            suite = name
            version = None
        return {
            "name": name,
            "suite": suite,
            "version": version,
            "passed": completed.returncode == 0 and graded,
            "exit_code": completed.returncode,
            "duration_ms": round(elapsed_ms, 3),
            "result_sha256": hashlib.sha256(output).hexdigest(),
            "result_bytes": len(output),
        }

    def run(self) -> dict[str, Any]:
        for path in (self.app_python, self.qwen_python):
            if not path.is_file() or not os.access(path, os.X_OK):
                raise FileNotFoundError(f"reviewed Python environment missing: {path}")
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=self.repo,
            text=True, capture_output=True, timeout=20, check=True)
        if status.stdout:
            raise RuntimeError("release-candidate run requires a clean worktree")
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.repo,
            text=True, capture_output=True, timeout=20, check=True,
        ).stdout.strip()
        version = (self.repo / "VERSION").read_text(encoding="utf-8").strip()
        with tempfile.TemporaryDirectory(
                prefix="friday-release-candidate-") as value:
            temporary = Path(value)
            temporary.chmod(0o700)
            archive = temporary / "source.tar.gz"
            subprocess.run([
                "git", "archive", "--format=tar.gz",
                f"--prefix=friday-{commit[:12]}/", "-o", str(archive), commit,
            ], cwd=self.repo, timeout=60, check=True)
            source_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
            gate_environment = os.environ.copy()
            gate_environment["PYTHONHASHSEED"] = "0"
            evaluation_environment = gate_environment.copy()
            evaluation_state = temporary / "evaluation-state"
            evaluation_state.mkdir(mode=0o700)
            evaluation_environment["FRIDAY_STATE_DIR"] = str(evaluation_state)
            gates = [
                self._command(
                    "release_tree", ["scripts/check-release.sh"],
                    environment=gate_environment, timeout=120),
                self._command(
                    "full_tests",
                    [str(self.app_python), "-m", "unittest", "discover", "-v"],
                    environment=gate_environment, timeout=1200),
                self._command(
                    "full_history_secret_scan", ["scripts/scan-secrets.sh"],
                    environment=gate_environment, timeout=300),
            ]
            installer = InstallerRehearsalRunner(
                self.repo).run(archive, source_sha256)
            dependency = run_dependency_review(
                self.repo, app_python=self.app_python,
                qwen_python=self.qwen_python)
            evaluations = [
                self._evaluation(
                    name, script, environment=evaluation_environment)
                for name, script in EVALUATION_COMMANDS
            ]
            local_passed = bool(
                all(item["passed"] for item in gates)
                and installer["passed"] and dependency["passed"]
                and all(item["passed"] for item in evaluations))
            report = {
                "format_version": 1,
                "candidate": {
                    "version": version,
                    "commit": commit,
                    "source_sha256": source_sha256,
                    "source_bytes": archive.stat().st_size,
                },
                "local_gates_passed": local_passed,
                "gates": gates,
                "installer_rehearsal": installer,
                "dependency_review": {
                    "passed": dependency["passed"],
                    "policy_sha256": dependency["policy_sha256"],
                    "application_packages": dependency["environments"][
                        "application"]["locked_packages"],
                    "qwen_runtime_packages": dependency["environments"][
                        "qwen_runtime"]["locked_packages"],
                    "models_and_assets": len(dependency["models_and_assets"]),
                    "distribution_approval": dependency[
                        "distribution_approval"],
                },
                "evaluations": evaluations,
                "external_or_owner_gates": [
                    "owner_license_and_piper_compatibility_decision",
                    "owner_name_icon_screenshot_and_voice_rights_approval",
                    "independent_penetration_test",
                    "cross_device_advertised_hardware_matrix",
                    "native_vision_higher_memory_qualification",
                    "public_repository_protections_after_visibility_change",
                    "explicit_public_release_approval",
                ],
                "publication_performed": False,
                "privacy": {
                    "user_content_used": False,
                    "raw_gate_output_retained": False,
                    "raw_evaluation_output_retained": False,
                    "temporary_candidate_destroyed_after_report": True,
                },
                "ran_at": datetime.now(UTC).isoformat(
                    timespec="microseconds").replace("+00:00", "Z"),
            }
            report["report_payload_sha256"] = canonical_sha256(report)
            return report
