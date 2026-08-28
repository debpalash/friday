"""Receipt-grounded long-horizon project qualification for Friday."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from .graph import GraphStore, canonical_json, utc_now
from .tasks import TaskService
from .worker import DurableStepWorker, StepExecutionResult


MAX_PROJECT_SUITE_BYTES = 256_000
MAX_PROJECT_FILE_BYTES = 64_000
_TEST_COUNT = re.compile(r"Ran ([0-9]+) tests? in")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 160:
        raise ValueError("project file path is invalid")
    path = PurePosixPath(value)
    if (path.is_absolute() or value != path.as_posix()
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(character.isspace() for character in value)):
        raise ValueError("project file path is invalid")
    return value


class ProjectEvalRunner:
    """Run a recovered multi-step build and grade only durable evidence."""

    def __init__(self, graph: GraphStore):
        self.graph = graph

    @staticmethod
    def _load_suite(path: str | Path) -> tuple[dict[str, Any], str]:
        try:
            descriptor = os.open(
                Path(path), os.O_RDONLY | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise ValueError("project suite is unavailable") from exc
        try:
            metadata = os.fstat(descriptor)
            if (not stat.S_ISREG(metadata.st_mode)
                    or not 2 <= metadata.st_size <= MAX_PROJECT_SUITE_BYTES):
                raise ValueError("project suite must be a bounded regular file")
            encoded = os.read(descriptor, MAX_PROJECT_SUITE_BYTES + 1)
            if len(encoded) != metadata.st_size:
                raise ValueError("project suite changed while being read")
        finally:
            os.close(descriptor)
        try:
            suite = json.loads(
                encoded.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite value: {value}")),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("project suite is invalid JSON") from exc
        if (not isinstance(suite, dict)
                or set(suite) != {"name", "version", "gates", "files"}
                or suite.get("name") != "friday-long-horizon-project"
                or suite.get("version") != 1):
            raise ValueError("project suite metadata is invalid")
        gates = suite["gates"]
        if not isinstance(gates, dict) or set(gates) != {
            "minimum_files", "minimum_tests", "maximum_recovery_ms",
        }:
            raise ValueError("project gates are invalid")
        for field in ("minimum_files", "minimum_tests"):
            value = gates[field]
            if (isinstance(value, bool) or not isinstance(value, int)
                    or not 1 <= value <= 100):
                raise ValueError("project count gate is invalid")
        recovery = gates["maximum_recovery_ms"]
        if (isinstance(recovery, bool) or not isinstance(recovery, (int, float))
                or not math.isfinite(float(recovery))
                or not 1 <= recovery <= 60_000):
            raise ValueError("project recovery gate is invalid")
        files = suite["files"]
        if (not isinstance(files, list)
                or not gates["minimum_files"] <= len(files) <= 32):
            raise ValueError("project files are invalid")
        paths: set[str] = set()
        for item in files:
            if not isinstance(item, dict) or set(item) != {"path", "content"}:
                raise ValueError("project file is invalid")
            relative = _safe_relative(item["path"])
            content = item["content"]
            if (relative in paths or not isinstance(content, str)
                    or not 1 <= len(content.encode("utf-8"))
                    <= MAX_PROJECT_FILE_BYTES
                    or "\x00" in content):
                raise ValueError("project file is invalid")
            paths.add(relative)
        return suite, hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _atomic_write(root: Path, relative: str, content: str) -> dict[str, Any]:
        safe = _safe_relative(relative)
        destination = root.joinpath(*PurePosixPath(safe).parts)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if destination.exists() and destination.is_symlink():
            raise RuntimeError("project destination is a symlink")
        encoded = content.encode("utf-8")
        temporary = destination.with_name(destination.name + ".friday-new")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        return {
            "status": "succeeded",
            "bytes": len(encoded),
            "path_sha256": hashlib.sha256(safe.encode()).hexdigest(),
            "file_sha256": _sha256_bytes(encoded),
        }

    @staticmethod
    def _run_tests(root: Path) -> dict[str, Any]:
        started = time.perf_counter_ns()
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", ".",
             "-p", "test_*.py"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        combined = result.stdout + result.stderr
        match = _TEST_COUNT.search(combined.decode("utf-8", errors="replace"))
        return {
            "status": "succeeded" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
            "tests": int(match.group(1)) if match else 0,
            "duration_ms": round(elapsed_ms, 3),
            "output_sha256": _sha256_bytes(combined),
            "output_bytes": len(combined),
        }

    @staticmethod
    def _expected_files(suite: dict[str, Any]) -> dict[str, str]:
        return {
            item["path"]: _sha256_bytes(item["content"].encode("utf-8"))
            for item in suite["files"]
        }

    @staticmethod
    def _verify_files(root: Path, expected: dict[str, str]) -> dict[str, Any]:
        matched = 0
        observed: list[str] = []
        for relative, digest in expected.items():
            path = root.joinpath(*PurePosixPath(relative).parts)
            if path.is_file() and not path.is_symlink():
                actual = _sha256_bytes(path.read_bytes())
                observed.append(actual)
                matched += int(actual == digest)
        return {
            "expected": len(expected),
            "matched": matched,
            "artifact_set_sha256": hashlib.sha256(
                "".join(sorted(observed)).encode("ascii")).hexdigest(),
        }

    async def _run_project(
        self,
        suite: dict[str, Any],
        root: Path,
        durable_graph: GraphStore,
    ) -> dict[str, Any]:
        tasks = TaskService(durable_graph)
        contract = {
            "version": 1,
            "objective": "Build and independently verify a small local project",
            "intent_type": "action",
            "success_criteria": [
                {"criterion_id": "files", "description": "Files match",
                 "verifier": "sha256"},
                {"criterion_id": "tests", "description": "Tests pass",
                 "verifier": "unittest"},
                {"criterion_id": "recovery", "description": "Work recovers",
                 "verifier": "durable_attempts"},
            ],
            "required_tools": ["write_file", "project_test", "project_verify"],
            "permissions": ["filesystem_write", "process"],
            "risk": "medium",
            "freshness_seconds": None,
            "needs_user_confirmation": False,
        }
        task_id, _ = tasks.create(contract["objective"], contract)
        tasks.transition(task_id, "interpreting")
        tasks.set_plan(task_id, [
            "Write exact project artifacts",
            "Run the isolated test suite",
            "Re-read and hash every expected artifact",
        ])
        tasks.transition(task_id, "planned")
        tasks.transition(task_id, "running")
        calls = [{
            "tool_call_id": f"write-{index:02d}",
            "tool_name": "write_file",
            "args": {"path": item["path"], "content": item["content"]},
            "risk": "medium",
            "approval_status": "not_required",
            "idempotency_class": "idempotent",
            "recovery_policy": "retry",
            "verifier": "exact_file_hash",
        } for index, item in enumerate(suite["files"])]
        calls.extend(({
            "tool_call_id": "run-tests",
            "tool_name": "project_test",
            "args": {},
            "risk": "low",
            "idempotency_class": "read_only",
            "recovery_policy": "retry",
            "verifier": "unittest_exit_and_count",
        }, {
            "tool_call_id": "verify-files",
            "tool_name": "project_verify",
            "args": {},
            "risk": "read_only",
            "idempotency_class": "read_only",
            "recovery_policy": "retry",
            "verifier": "exact_file_hashes",
        }))
        batch_id, _ = tasks.stage_step_batch(
            task_id, calls, round_index=0,
            context={"session_id": "long-horizon-evaluation"})

        # Persist a dispatch and then abandon the worker before the executor is
        # invoked. The replacement must recover the same action and finish it
        # under a new fenced attempt.
        interrupted = tasks.claim_next_step(batch_id, "project-worker-before")
        if interrupted is None:
            raise RuntimeError("project interruption fixture was not dispatched")

        expected = self._expected_files(suite)
        write_counts = {relative: 0 for relative in expected}

        async def executor(claim):
            if claim.tool_name == "write_file":
                relative = _safe_relative(claim.args["path"])
                if relative not in expected:
                    raise RuntimeError("project write is outside the suite")
                result = self._atomic_write(
                    root, relative, str(claim.args["content"]))
                write_counts[relative] += 1
                passed = result["file_sha256"] == expected[relative]
                return StepExecutionResult(
                    result=result,
                    succeeded=passed,
                    verification={
                        "status": "passed" if passed else "failed",
                        "summary": "exact project file hash verified",
                        "evidence": [result["file_sha256"]],
                        "missing": [] if passed else ["expected file hash"],
                        "effects": [{"kind": "project_file", "verified": passed}],
                    },
                )
            if claim.tool_name == "project_test":
                result = await asyncio.to_thread(self._run_tests, root)
                passed = (result["exit_code"] == 0
                          and result["tests"] >= suite["gates"]["minimum_tests"])
                return StepExecutionResult(
                    result=result,
                    succeeded=passed,
                    verification={
                        "status": "passed" if passed else "failed",
                        "summary": "isolated project tests verified",
                        "evidence": [result["output_sha256"]],
                        "missing": [] if passed else ["passing project tests"],
                        "effects": [{"kind": "project_tests", "verified": passed,
                                     "count": result["tests"]}],
                    },
                )
            if claim.tool_name == "project_verify":
                result = self._verify_files(root, expected)
                passed = result["matched"] == result["expected"]
                return StepExecutionResult(
                    result=result,
                    succeeded=passed,
                    verification={
                        "status": "passed" if passed else "failed",
                        "summary": "all project artifacts independently verified",
                        "evidence": [result["artifact_set_sha256"]],
                        "missing": [] if passed else ["matching project files"],
                        "effects": [{"kind": "project_artifacts",
                                     "verified": passed,
                                     "count": result["matched"]}],
                    },
                )
            raise RuntimeError("project evaluator received an unknown tool")

        completed: asyncio.Queue = asyncio.Queue()

        async def completion_hook(outcome):
            await completed.put(outcome)

        recovery_started = time.perf_counter_ns()
        worker = DurableStepWorker(
            tasks, executor, worker_id="project-worker-after",
            completion_hook=completion_hook)
        resumed = await worker.start(dead_worker_id="project-worker-before")
        try:
            async with asyncio.timeout(30):
                outcome = await completed.get()
        finally:
            await worker.stop(timeout=2)
        recovery_ms = (time.perf_counter_ns() - recovery_started) / 1_000_000

        files = self._verify_files(root, expected)
        fresh_tests = await asyncio.to_thread(self._run_tests, root)
        receipts = tasks.action_history(task_id)
        steps = tasks.list_steps(task_id=task_id)
        with durable_graph._connect() as connection:
            attempts = int(connection.execute(
                "SELECT COUNT(*) FROM action_attempts WHERE step_id=?",
                (interrupted.step_id,),
            ).fetchone()[0])
        receipt_checks = [
            receipt["status"] == "succeeded"
            and isinstance(receipt["verification"], dict)
            and receipt["verification"].get("status") == "passed"
            for receipt in receipts
        ]
        checks = {
            "batch_succeeded": outcome.status == "succeeded",
            "files_verified": files["matched"] == files["expected"],
            "tests_verified": (
                fresh_tests["exit_code"] == 0
                and fresh_tests["tests"] >= suite["gates"]["minimum_tests"]),
            "recovery_observed": (
                batch_id in resumed and attempts == 2
                and recovery_ms <= suite["gates"]["maximum_recovery_ms"]),
            "receipts_verified": (
                len(receipts) == len(calls) and all(receipt_checks)),
            "no_duplicate_effects": all(count == 1 for count in write_counts.values()),
            "all_steps_succeeded": all(
                step["status"] == "succeeded" for step in steps),
        }
        verification = {
            "status": "passed" if all(checks.values()) else "failed",
            "summary": "project outcome derived from receipts and fresh probes",
            "evidence": [
                files["artifact_set_sha256"], fresh_tests["output_sha256"],
                hashlib.sha256(canonical_json([
                    receipt["idempotency_key"] for receipt in receipts
                ]).encode()).hexdigest(),
            ],
            "missing": [name for name, passed in checks.items() if not passed],
            "effects": [{"kind": "project", "verified": all(checks.values())}],
        }
        tasks.transition(task_id, "verifying")
        tasks.record_verification(task_id, verification)
        if all(checks.values()):
            tasks.transition(task_id, "completed")
        else:
            tasks.transition(task_id, "failed", error="project scorecard failed")
        final = tasks.get(task_id)
        visible = {
            "status": final["status"],
            "files_verified": files["matched"],
            "tests_passed": fresh_tests["tests"] if fresh_tests["exit_code"] == 0 else 0,
            "recovered": attempts == 2,
        }
        return {
            "checks": checks,
            "files": files,
            "tests": fresh_tests,
            "recovery": {
                "resumed_batches": len(resumed),
                "interrupted_step_attempts": attempts,
                "recovery_ms": round(recovery_ms, 3),
            },
            "receipts": {
                "expected": len(calls),
                "succeeded": sum(
                    receipt["status"] == "succeeded" for receipt in receipts),
                "verified": sum(receipt_checks),
            },
            "user_visible_outcome": visible,
            "passed": all(checks.values()) and visible["status"] == "completed",
        }

    async def run(self, suite_path: str | Path) -> dict[str, Any]:
        suite, suite_sha256 = self._load_suite(suite_path)
        workspace_path: Path | None = None
        with tempfile.TemporaryDirectory(
                prefix="friday-project-eval-") as workspace_value, \
             tempfile.TemporaryDirectory(
                prefix="friday-project-journal-") as journal_value:
            workspace_path = Path(workspace_value)
            os.chmod(workspace_path, 0o700)
            result = await self._run_project(
                suite, workspace_path,
                GraphStore(Path(journal_value) / "project.db"))
        cleanup_verified = bool(
            workspace_path is not None and not workspace_path.exists())
        checks = dict(result["checks"])
        checks["workspace_cleanup"] = cleanup_verified
        body = {
            "suite": suite["name"],
            "version": suite["version"],
            "suite_sha256": suite_sha256,
            "gates": suite["gates"],
            "files": result["files"],
            "tests": result["tests"],
            "recovery": result["recovery"],
            "receipts": result["receipts"],
            "user_visible_outcome": result["user_visible_outcome"],
            "checks": checks,
            "passed": result["passed"] and cleanup_verified,
            "privacy": {
                "workspace": "disposable",
                "file_contents_persisted": False,
                "test_output_persisted": False,
                "cleanup_verified": cleanup_verified,
            },
            "ran_at": utc_now(),
        }
        run_id = self.graph.record_node(
            "long_horizon_project_evaluation_run", body,
            actor="project_eval_runner",
            event_type="evaluation.long_horizon_project_completed",
        )
        return {"evaluation_run_id": run_id, **body}
