"""Injected-failure rollback and recovery qualification for Friday."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .calibration import BOOT_STABILITY_SECONDS, BootRecoveryStore
from .deployments import DeploymentManager
from .graph import GraphStore, utc_now
from .hardware import GIB, Accelerator, HardwareSnapshot, select_runtime_profile
from .operator import WebOperator
from .tasks import TaskService
from .worker import DurableStepWorker, StepExecutionResult


MAX_RECOVERY_SUITE_BYTES = 32_000


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 3)


class _BrowserPage:
    url = "https://93.184.216.34/recovery"

    def __init__(self):
        self.mutations = 0

    class _Locator:
        def __init__(self, page: "_BrowserPage"):
            self.page = page
            self.first = self

        def fill(self, _text: str, *, timeout: int):
            if timeout != 10_000:
                raise RuntimeError("browser timeout changed")
            self.page.mutations += 1

        @staticmethod
        def press(_key: str):
            return None

    def locator(self, _selector: str):
        return self._Locator(self)

    @staticmethod
    def wait_for_timeout(_milliseconds: int):
        return None

    @staticmethod
    def title() -> str:
        return "Recovery fixture"


class _Browser:
    def __init__(self, page: _BrowserPage):
        self.contexts = [type("Context", (), {"pages": [page]})()]


class _RecoveryWebOperator(WebOperator):
    def __init__(self, profile: Path, browser: _Browser):
        super().__init__(profile)
        self.browser = browser

    def _controlled(self, operation):
        self._verify_managed_runtime()
        result = operation(self.browser)
        self._verify_managed_runtime()
        return result


class RecoveryEvalRunner:
    def __init__(self, graph: GraphStore):
        self.graph = graph

    @staticmethod
    def _load_suite(path: str | Path) -> tuple[dict[str, Any], str]:
        try:
            descriptor = os.open(
                Path(path), os.O_RDONLY | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise ValueError("recovery suite is unavailable") from exc
        try:
            metadata = os.fstat(descriptor)
            if (not stat.S_ISREG(metadata.st_mode)
                    or not 2 <= metadata.st_size <= MAX_RECOVERY_SUITE_BYTES):
                raise ValueError("recovery suite must be a bounded regular file")
            encoded = os.read(descriptor, MAX_RECOVERY_SUITE_BYTES + 1)
            if len(encoded) != metadata.st_size:
                raise ValueError("recovery suite changed while being read")
        finally:
            os.close(descriptor)
        try:
            suite = json.loads(
                encoded.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite value: {value}")),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("recovery suite is invalid JSON") from exc
        if (not isinstance(suite, dict)
                or set(suite) != {"name", "version", "gates"}
                or suite.get("name") != "friday-injected-recovery"
                or suite.get("version") != 1):
            raise ValueError("recovery suite metadata is invalid")
        gates = suite["gates"]
        if not isinstance(gates, dict) or set(gates) != {
            "minimum_recovery_rate", "maximum_control_path_p95_ms",
            "maximum_model_retry_seconds",
        }:
            raise ValueError("recovery gates are invalid")
        rate = gates["minimum_recovery_rate"]
        latency = gates["maximum_control_path_p95_ms"]
        retry = gates["maximum_model_retry_seconds"]
        if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate)) or not 0 <= rate <= 1
                or isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or not math.isfinite(float(latency)) or not 1 <= latency <= 60_000
                or isinstance(retry, bool) or not isinstance(retry, int)
                or not 1 <= retry <= 900):
            raise ValueError("recovery gate is invalid")
        return suite, hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _model_failure(root: Path) -> dict[str, Any]:
        snapshot = HardwareSnapshot(
            cpu_count=32,
            system_memory_bytes=64 * GIB,
            accelerators=(Accelerator(
                "cuda", 0, "Evaluation GPU", 24 * GIB, 24 * GIB),),
            cuda_probe="available",
        )
        profile = select_runtime_profile(snapshot, environment={})
        store = BootRecoveryStore(root / "model-recovery.json")
        store.record_launch_success(
            profile, profile, runtime_identity="model-runtime-before", now=100)
        started = time.perf_counter_ns()
        retry_seconds = store.observe(
            profile, running=False, active=None, now=101)
        store.record_launch_success(
            profile, profile, runtime_identity="model-runtime-after",
            now=101 + retry_seconds)
        store.observe(
            profile, running=True, active=profile,
            runtime_identity="model-runtime-after",
            now=101 + retry_seconds + BOOT_STABILITY_SECONDS)
        status = store.public_status(
            profile, now=101 + retry_seconds + BOOT_STABILITY_SECONDS)
        control_ms = (time.perf_counter_ns() - started) / 1_000_000
        return {
            "name": "model",
            "recovered": (
                retry_seconds > 0 and status["state"] == "stable"
                and status["consecutive_failures"] == 0),
            "control_path_ms": round(control_ms, 3),
            "retry_policy_seconds": retry_seconds,
            "failure_injection": "early_runtime_loss",
        }

    @staticmethod
    async def _worker_failure(root: Path) -> dict[str, Any]:
        graph = GraphStore(root / "worker.db")
        tasks = TaskService(graph)
        task_id, _ = tasks.create(
            "Recover one interrupted worker action", {"version": 0})
        batch_id, _ = tasks.stage_step_batch(
            task_id, [{
                "tool_call_id": "worker-recovery-call",
                "tool_name": "list_files",
                "args": {"path": "."},
                "risk": "read_only",
                "idempotency_class": "read_only",
                "recovery_policy": "retry",
            }], round_index=0)
        interrupted = tasks.claim_next_step(batch_id, "worker-before-failure")
        if interrupted is None:
            raise RuntimeError("worker failure was not injected")
        effects = 0

        async def executor(_claim):
            nonlocal effects
            effects += 1
            return StepExecutionResult(
                result={"status": "succeeded"}, succeeded=True,
                verification={
                    "status": "passed", "summary": "worker result verified",
                    "evidence": ["recovered attempt"], "missing": [],
                    "effects": [{"kind": "read", "verified": True}],
                })

        completed: asyncio.Queue = asyncio.Queue()

        async def completion_hook(outcome):
            await completed.put(outcome)

        started = time.perf_counter_ns()
        worker = DurableStepWorker(
            tasks, executor, worker_id="worker-after-failure",
            completion_hook=completion_hook)
        resumed = await worker.start(dead_worker_id="worker-before-failure")
        try:
            async with asyncio.timeout(10):
                outcome = await completed.get()
        finally:
            await worker.stop(timeout=2)
        control_ms = (time.perf_counter_ns() - started) / 1_000_000
        with graph._connect() as connection:
            attempts = int(connection.execute(
                "SELECT COUNT(*) FROM action_attempts WHERE step_id=?",
                (interrupted.step_id,)).fetchone()[0])
        return {
            "name": "worker",
            "recovered": (
                outcome.status == "succeeded" and batch_id in resumed
                and attempts == 2 and effects == 1),
            "control_path_ms": round(control_ms, 3),
            "attempts": attempts,
            "effects": effects,
            "failure_injection": "abandoned_dispatched_attempt",
        }

    @staticmethod
    def _browser_failure(root: Path) -> dict[str, Any]:
        page = _BrowserPage()
        operator = _RecoveryWebOperator(
            root / "browser-profile", _Browser(page))
        expected_identity = "browser-execution-a"
        active_identity = "browser-execution-replaced"
        checks = 0

        def verify() -> bool:
            nonlocal checks
            checks += 1
            return active_identity == expected_identity

        operator.require_managed_runtime(verify)
        rejected = False
        started = time.perf_counter_ns()
        try:
            operator.type("#query", "private fixture", page_url=page.url)
        except RuntimeError:
            rejected = True
        active_identity = expected_identity
        receipt = operator.type(
            "#query", "private fixture", page_url=page.url, submit=True)
        control_ms = (time.perf_counter_ns() - started) / 1_000_000
        return {
            "name": "browser",
            "recovered": (
                rejected and page.mutations == 1
                and receipt["submitted"] is True and checks == 3),
            "control_path_ms": round(control_ms, 3),
            "runtime_checks": checks,
            "mutations": page.mutations,
            "failure_injection": "managed_runtime_replacement",
        }

    @staticmethod
    def _filesystem_failure(root: Path) -> dict[str, Any]:
        project = root / "filesystem-project"
        project.mkdir()
        target = project / "module.py"
        original = b"VALUE = 1\n"
        target.write_bytes(original)
        graph = GraphStore(project / "state" / "friday.db")
        manager = DeploymentManager(
            graph, project,
            [sys.executable, "-c", "raise SystemExit(7)"])
        rejected = False
        started = time.perf_counter_ns()
        try:
            manager.stage_write("module.py", "VALUE = 2\n")
        except RuntimeError:
            rejected = True
        control_ms = (time.perf_counter_ns() - started) / 1_000_000
        with graph._connect() as connection:
            status = connection.execute(
                "SELECT status FROM deployment_state").fetchone()[0]
        return {
            "name": "filesystem",
            "recovered": (
                rejected and target.read_bytes() == original
                and status == "rejected"),
            "control_path_ms": round(control_ms, 3),
            "source_restored": target.read_bytes() == original,
            "failure_injection": "verification_exit_7",
        }

    async def run(self, suite_path: str | Path) -> dict[str, Any]:
        suite, suite_sha256 = self._load_suite(suite_path)
        root_path: Path | None = None
        with tempfile.TemporaryDirectory(prefix="friday-recovery-eval-") as value:
            root_path = Path(value)
            os.chmod(root_path, 0o700)
            scenarios = [
                self._model_failure(root_path),
                await self._worker_failure(root_path),
                self._browser_failure(root_path),
                self._filesystem_failure(root_path),
            ]
        cleanup_verified = bool(root_path is not None and not root_path.exists())
        latencies = [item["control_path_ms"] for item in scenarios]
        recovered = sum(item["recovered"] for item in scenarios)
        recovery_rate = recovered / len(scenarios)
        metrics = {
            "scenarios": len(scenarios),
            "recovered": recovered,
            "recovery_rate": recovery_rate,
            "control_path_p50_ms": _percentile(latencies, 0.50),
            "control_path_p95_ms": _percentile(latencies, 0.95),
            "model_retry_policy_seconds": scenarios[0]["retry_policy_seconds"],
        }
        gates = suite["gates"]
        checks = {
            "recovery_rate": recovery_rate >= gates["minimum_recovery_rate"],
            "control_path_latency": (
                metrics["control_path_p95_ms"]
                <= gates["maximum_control_path_p95_ms"]),
            "model_retry_policy": (
                metrics["model_retry_policy_seconds"]
                <= gates["maximum_model_retry_seconds"]),
            "fixture_cleanup": cleanup_verified,
        }
        body = {
            "suite": suite["name"],
            "version": suite["version"],
            "suite_sha256": suite_sha256,
            "gates": gates,
            "scenarios": scenarios,
            "metrics": metrics,
            "checks": checks,
            "passed": all(checks.values()),
            "privacy": {
                "fixtures": "disposable",
                "raw_outputs_persisted": False,
                "cleanup_verified": cleanup_verified,
            },
            "ran_at": utc_now(),
        }
        run_id = self.graph.record_node(
            "recovery_evaluation_run", body,
            actor="recovery_eval_runner",
            event_type="evaluation.recovery_completed")
        return {"evaluation_run_id": run_id, **body}
