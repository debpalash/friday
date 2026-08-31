import asyncio
import contextlib
import io
import json
import re
import stat
import tempfile
import threading
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import server
from friday_core import (Accelerator, AdmissionBudget, GraphStore,
                         HardwareSnapshot, MemoryCurator, ReflectionService,
                         ResourceAdmissionController, ResourceClaim,
                         ResourceSnapshot, TaskService,
                         select_runtime_profile)
from friday_core.hardware import GIB


class _FakeCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        async def stream():
            for chunk in self.chunks:
                yield chunk
        return stream()


def _chunk(content="", tool_calls=None, *, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=delta, finish_reason=finish_reason)])


def _tool_chunk(name, arguments, *, call_id="call_1", index=0):
    function = SimpleNamespace(name=name, arguments=arguments)
    call = SimpleNamespace(index=index, id=call_id, function=function)
    return _chunk(tool_calls=[call])


class ServerStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_ephemeral_response_uses_and_preserves_isolated_history(self):
        friday = server.Friday.__new__(server.Friday)
        persistent = [{"role": "system", "content": "persistent"}]
        isolated = [{"role": "system", "content": "isolated"}]
        friday.history = persistent
        observed = {}

        async def fake_respond(_text, _queue, **kwargs):
            observed["history"] = friday.history
            observed["persist_session"] = kwargs["persist_session"]
            friday.history.append({"role": "assistant", "content": "answer"})

        friday._respond_serialized = fake_respond
        await friday.respond("test", asyncio.Queue(),
                             conversation_history=isolated)

        self.assertIs(friday.history, persistent)
        self.assertIs(observed["history"], isolated)
        self.assertFalse(observed["persist_session"])
        self.assertEqual(isolated[-1]["content"], "answer")

    async def test_readiness_requires_every_authoritative_worker(self):
        live = SimpleNamespace(is_running=True)
        dead = SimpleNamespace(is_running=False)
        with patch.object(server, "FRIDAY", object()), \
             patch.object(server, "WORKER", live), \
             patch.object(server, "REMINDER_WORKER", live):
            healthy = await server.healthz()
        with patch.object(server, "FRIDAY", object()), \
             patch.object(server, "WORKER", dead), \
             patch.object(server, "REMINDER_WORKER", live):
            unhealthy = await server.healthz()

        self.assertEqual(healthy.status_code, 200)
        self.assertTrue(json.loads(healthy.body)["ready"])
        self.assertEqual(unhealthy.status_code, 503)
        self.assertFalse(json.loads(unhealthy.body)["components"]["action_worker"])

    async def test_readiness_requires_initialized_process_monitor_to_be_alive(self):
        live = SimpleNamespace(is_running=True)
        dead_monitor = SimpleNamespace(done=lambda: True)
        with patch.multiple(
                server, FRIDAY=object(), WORKER=live, REMINDER_WORKER=live,
                PROCESS_BROKER=object(), PROCESS_MONITOR_TASK=dead_monitor,
                PROCESS_MONITOR_LAST_ERROR=None):
            response = await server.healthz()

        body = json.loads(response.body)
        self.assertEqual(response.status_code, 503)
        self.assertFalse(body["components"]["process_monitor"])

    async def test_readiness_requires_initialized_browser_proxy_to_be_alive(self):
        live = SimpleNamespace(is_running=True)
        proxy = SimpleNamespace(healthy=lambda: False)
        with patch.multiple(
                server, FRIDAY=object(), WORKER=live, REMINDER_WORKER=live,
                WEB_PROXY=proxy, WEB_PROXY_INITIALIZED=True):
            response = await server.healthz()

        body = json.loads(response.body)
        self.assertEqual(response.status_code, 503)
        self.assertFalse(body["components"]["browser_network"])

    async def test_process_cleanup_block_quarantines_capability_without_crashloop(self):
        calls = []

        class Broker:
            def cleanup_retained(self):
                calls.append("cleanup")
                return {
                    "pending": 2, "cleaning": 1, "blocked": 1,
                    "completed_last_pass": 3,
                }

            def reconcile_active(self):
                calls.append("reconcile")
                return []

        with patch.multiple(
                server, PROCESS_BROKER=Broker(),
                PROCESS_MONITOR_LAST_ERROR=None,
                PROCESS_CLEANUP_LAST_ERROR=None):
            receipts = await server._reconcile_processes_once()
            self.assertEqual(calls, ["cleanup", "reconcile"])
            self.assertEqual(receipts, [])
            self.assertEqual(server.PROCESS_CLEANUP_PENDING_COUNT, 3)
            self.assertEqual(server.PROCESS_CLEANUP_BLOCKED_COUNT, 1)
            self.assertEqual(server.PROCESS_CLEANUP_LAST_COMPLETED_COUNT, 3)
            self.assertEqual(
                server.PROCESS_CLEANUP_LAST_ERROR,
                "process_unit_cleanup_blocked")
            self.assertIsNone(server.PROCESS_MONITOR_LAST_ERROR)
            with self.assertRaises(server.ProcessCleanupBlocked):
                server._require_process_broker(require_cleanup_ready=True)
            self.assertIs(server._require_process_broker(), server.PROCESS_BROKER)

    async def test_process_cleanup_is_not_starved_by_reconciliation_failure(self):
        calls = []

        class Broker:
            def cleanup_retained(self):
                calls.append("cleanup")
                return {
                    "pending": 0, "cleaning": 0, "blocked": 0,
                    "completed_last_pass": 1,
                }

            def reconcile_active(self):
                calls.append("reconcile")
                raise RuntimeError("reconcile transport failed")

        with patch.multiple(
                server, PROCESS_BROKER=Broker(),
                PROCESS_MONITOR_LAST_ERROR=None,
                PROCESS_CLEANUP_LAST_ERROR=None):
            with self.assertRaises(RuntimeError):
                await server._reconcile_processes_once()
            self.assertEqual(calls, ["cleanup", "reconcile"])
            self.assertEqual(server.PROCESS_CLEANUP_LAST_COMPLETED_COUNT, 1)
            self.assertIsNone(server.PROCESS_CLEANUP_LAST_ERROR)
            self.assertEqual(
                server.PROCESS_RECONCILE_LAST_ERROR,
                "process_reconcile_failed")

    async def test_retrying_cleanup_temporarily_quarantines_new_process_effects(self):
        class Broker:
            def cleanup_retained(self):
                return {
                    "pending": 1, "cleaning": 0, "blocked": 0,
                    "retrying": 1, "completed_last_pass": 0,
                }

            def reconcile_active(self):
                return []

        with patch.multiple(
                server, PROCESS_BROKER=Broker(),
                PROCESS_MONITOR_LAST_ERROR=None,
                PROCESS_CLEANUP_LAST_ERROR=None):
            await server._reconcile_processes_once()
            self.assertEqual(
                server.PROCESS_CLEANUP_LAST_ERROR,
                "process_unit_cleanup_retrying")
            self.assertIsNone(server.PROCESS_MONITOR_LAST_ERROR)
            with self.assertRaises(server.ProcessCleanupBlocked):
                server._require_process_broker(require_cleanup_ready=True)

    async def test_readiness_requires_initialized_reconciler_to_be_alive(self):
        live = SimpleNamespace(is_running=True)
        dead_reconciler = SimpleNamespace(done=lambda: True)
        with patch.multiple(
                server, FRIDAY=object(), WORKER=live, REMINDER_WORKER=live,
                RECONCILIATION_INITIALIZED=True,
                RECONCILIATION_TASK=dead_reconciler,
                RECONCILIATION_LAST_ERROR=None):
            response = await server.healthz()

        body = json.loads(response.body)
        self.assertEqual(response.status_code, 503)
        self.assertFalse(body["components"]["action_reconciler"])

    async def test_readiness_fails_when_expected_desktop_operator_did_not_start(self):
        live = SimpleNamespace(is_running=True)
        with patch.multiple(
                server, FRIDAY=object(), WORKER=live, REMINDER_WORKER=live,
                DESKTOP_INITIALIZED=True, DESKTOP_BROKER=None,
                DESKTOP_LAST_ERROR="desktop_adapter_unsupported",
                DESKTOP_MODE="required"):
            response = await server.healthz()

        body = json.loads(response.body)
        self.assertEqual(response.status_code, 503)
        self.assertFalse(body["components"]["desktop_operator"])

    async def test_auto_desktop_unavailability_degrades_without_crash_loop(self):
        live = SimpleNamespace(is_running=True)
        with patch.multiple(
                server, FRIDAY=object(), WORKER=live, REMINDER_WORKER=live,
                DESKTOP_INITIALIZED=True, DESKTOP_BROKER=None,
                DESKTOP_LAST_ERROR="desktop_session_unavailable",
                DESKTOP_MODE="auto"):
            response = await server.healthz()

        body = json.loads(response.body)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["components"]["desktop_operator"])

    async def test_readiness_forces_a_fresh_admission_sample(self):
        calls = []
        stale = ResourceSnapshot(
            available_cpu_millis=0,
            available_ram_mib=0,
            captured_at=datetime.now(UTC) - timedelta(minutes=1),
        )
        fresh = ResourceSnapshot(
            available_cpu_millis=1_000,
            available_ram_mib=2_048,
            captured_at=datetime.now(UTC),
        )

        class AdmissionProbe:
            def get_snapshot(self, *, force=False):
                calls.append(force)
                return fresh if force else stale

        live = SimpleNamespace(is_running=True)
        tasks = SimpleNamespace(admission_sensor_error=None)
        with patch.multiple(
                server, FRIDAY=object(), WORKER=live, REMINDER_WORKER=live,
                TASKS=tasks, ADMISSION=AdmissionProbe()):
            response = await server.healthz()

        self.assertEqual(calls, [True])
        self.assertEqual(response.status_code, 200)
        self.assertTrue(json.loads(response.body)["components"][
            "admission_sampler"])

    async def test_readiness_fails_closed_on_admission_failure_or_staleness(self):
        class FailingAdmission:
            def get_snapshot(self, *, force=False):
                raise RuntimeError("resource telemetry unavailable")

        class StaleAdmission:
            def get_snapshot(self, *, force=False):
                return ResourceSnapshot(
                    available_cpu_millis=1_000,
                    available_ram_mib=2_048,
                    captured_at=datetime.now(UTC) - timedelta(seconds=30),
                )

        live = SimpleNamespace(is_running=True)
        tasks = SimpleNamespace(admission_sensor_error=None)
        for label, admission in (
                ("failure", FailingAdmission()),
                ("stale", StaleAdmission())):
            with self.subTest(condition=label), patch.multiple(
                    server, FRIDAY=object(), WORKER=live,
                    REMINDER_WORKER=live, TASKS=tasks,
                    ADMISSION=admission):
                response = await server.healthz()

            body = json.loads(response.body)
            self.assertEqual(response.status_code, 503)
            self.assertFalse(body["ready"])
            self.assertFalse(body["components"]["admission_sampler"])

    async def test_readiness_rejects_a_far_future_admission_sample(self):
        class FutureAdmission:
            def get_snapshot(self, *, force=False):
                return ResourceSnapshot(
                    available_cpu_millis=1_000,
                    available_ram_mib=2_048,
                    captured_at=datetime.now(UTC) + timedelta(minutes=5),
                )

        live = SimpleNamespace(is_running=True)
        tasks = SimpleNamespace(
            admission_sensor_error=None,
            admission_sensor_checked_at=None,
        )
        with patch.multiple(
                server, FRIDAY=object(), WORKER=live, REMINDER_WORKER=live,
                TASKS=tasks, ADMISSION=FutureAdmission()):
            response = await server.healthz()

        body = json.loads(response.body)
        self.assertEqual(response.status_code, 503)
        self.assertFalse(body["ready"])
        self.assertFalse(body["components"]["admission_sampler"])

    async def test_fresh_health_sample_clears_sticky_sensor_error(self):
        class FreshAdmission:
            def get_snapshot(self, *, force=False):
                return ResourceSnapshot(
                    available_cpu_millis=1_000,
                    available_ram_mib=2_048,
                    captured_at=datetime.now(UTC),
                )

        live = SimpleNamespace(is_running=True)
        old_checked_at = "2000-01-01T00:00:00Z"
        tasks = SimpleNamespace(
            admission_sensor_error="prior transient telemetry failure",
            admission_sensor_checked_at=old_checked_at,
        )
        before = datetime.now(UTC)
        with patch.multiple(
                server, FRIDAY=object(), WORKER=live, REMINDER_WORKER=live,
                TASKS=tasks, ADMISSION=FreshAdmission()):
            response = await server.healthz()
        after = datetime.now(UTC)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(json.loads(response.body)["ready"])
        self.assertIsNone(tasks.admission_sensor_error)
        self.assertNotEqual(tasks.admission_sensor_checked_at, old_checked_at)
        checked_at = datetime.fromisoformat(
            tasks.admission_sensor_checked_at.replace("Z", "+00:00"))
        self.assertLessEqual(before, checked_at)
        self.assertLessEqual(checked_at, after)

    async def test_status_readiness_agrees_with_shared_admission_health(self):
        class ObservableAdmission:
            def __init__(self, snapshot):
                self.snapshot = snapshot

            def get_snapshot(self, *, force=False):
                return self.snapshot

            def status(self):
                return {
                    "profile_fingerprint": "d" * 64,
                    "budget": {},
                    "snapshot": self.snapshot.model_dump(mode="json"),
                    "active": {"leases": 0, "latency_classes": {}},
                }

        live = SimpleNamespace(is_running=True)
        graph = SimpleNamespace(
            schema_version=lambda: 7,
            count=lambda _name: 0,
        )
        no_reminders = SimpleNamespace(list=lambda **_kwargs: [])
        model_router = SimpleNamespace(
            status=lambda: {"remote_enabled": False})

        for label, captured_at, expected_ready in (
                ("fresh", datetime.now(UTC), True),
                ("stale", datetime.now(UTC) - timedelta(seconds=30), False)):
            with self.subTest(sample=label), tempfile.TemporaryDirectory() as temp:
                snapshot = ResourceSnapshot(
                    available_cpu_millis=1_000,
                    available_ram_mib=2_048,
                    captured_at=captured_at,
                )
                admission = ObservableAdmission(snapshot)
                tasks = SimpleNamespace(
                    admission_sensor_error=None,
                    admission_sensor_checked_at=None,
                    nonterminal=lambda: [],
                )
                with patch.multiple(
                        server, FRIDAY=SimpleNamespace(
                            asr=SimpleNamespace(name="test-asr"),
                            voice_name="test-voice"),
                        WORKER=live, REMINDER_WORKER=live,
                        TASKS=tasks, ADMISSION=admission, GRAPH=graph,
                        REMINDERS=no_reminders, MODEL_ROUTER=model_router,
                        STATE_DIR=Path(temp) / "missing-state"):
                    health_response = await server.healthz()
                    status_body = await server.api_status()

                health_ready = json.loads(health_response.body)["ready"]
                self.assertEqual(health_ready, expected_ready)
                self.assertEqual(status_body["ready"], health_ready)

    def test_admission_sampler_fails_closed_per_missing_proc_dimension(self):
        budget = AdmissionBudget(
            cpu_millis=8_000,
            ram_mib=16_384,
            concurrency_slots=4,
            network_slots=2,
            accelerator_vram_mib={},
        )
        meminfo = (
            "MemTotal:       16777216 kB\n"
            "MemAvailable:    8388608 kB\n"
        )
        loadavg = "0.50 0.25 0.10 1/100 123\n"

        def read_with_meminfo_failure(path, *_args, **_kwargs):
            if str(path) == "/proc/meminfo":
                raise OSError("meminfo unavailable")
            if str(path) == "/proc/loadavg":
                return loadavg
            raise AssertionError(f"unexpected telemetry path: {path}")

        def read_with_loadavg_failure(path, *_args, **_kwargs):
            if str(path) == "/proc/meminfo":
                return meminfo
            if str(path) == "/proc/loadavg":
                raise OSError("loadavg unavailable")
            raise AssertionError(f"unexpected telemetry path: {path}")

        def sysconf_16_gib(name):
            return {
                "SC_PHYS_PAGES": 4_194_304,
                "SC_PAGE_SIZE": 4_096,
            }[name]

        with patch.object(server, "ADMISSION_BUDGET", budget), \
             patch.object(Path, "read_text", read_with_meminfo_failure), \
             patch.object(server.os, "sysconf", sysconf_16_gib):
            missing_memory = server._sample_admission_resources()
        with patch.object(server, "ADMISSION_BUDGET", budget), \
             patch.object(Path, "read_text", read_with_loadavg_failure):
            missing_cpu = server._sample_admission_resources()

        with self.subTest(missing="meminfo"):
            self.assertEqual(missing_memory.available_ram_mib, 0)
            self.assertEqual(missing_memory.available_cpu_millis, 7_500)
        with self.subTest(missing="loadavg"):
            self.assertEqual(missing_cpu.available_cpu_millis, 0)
            self.assertEqual(missing_cpu.available_ram_mib, 6_144)

    def test_24gb_manifest_admission_budget_names_bridge_to_controller(self):
        hardware = HardwareSnapshot(
            cpu_count=32,
            system_memory_bytes=64 * GIB,
            accelerators=(Accelerator(
                "cuda", 0, "24 GB GPU", 24 * GIB, 1 * GIB),),
            cuda_probe="available",
        )
        profile = select_runtime_profile(hardware, environment={})

        budget = server._admission_budget_from_manifest(profile.to_dict())

        self.assertEqual(budget, AdmissionBudget(
            cpu_millis=28_000,
            ram_mib=58_982,
            concurrency_slots=8,
            network_slots=4,
            accelerator_vram_mib={"cuda:0": 982},
        ))

    async def test_status_exposes_admission_totals_without_step_arguments(self):
        secret = "status-must-not-leak-step-args-3b61a9"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = GraphStore(root / "friday.db")
            budget = AdmissionBudget(
                cpu_millis=4_000,
                ram_mib=8_192,
                concurrency_slots=3,
                network_slots=2,
                accelerator_vram_mib={"cuda:0": 4_096},
            )
            snapshot = ResourceSnapshot(
                available_cpu_millis=3_500,
                available_ram_mib=7_000,
                available_network_slots=2,
                available_accelerator_vram_mib={"cuda:0": 3_500},
                captured_at=datetime.now(UTC),
            )
            admission = ResourceAdmissionController(
                graph, budget, lambda: snapshot)
            tasks = TaskService(graph, admission=admission)
            task_id, _ = tasks.create(
                "Observe capacity safely",
                {"version": 0, "evidence": "admission telemetry"},
            )
            _batch_id, steps = tasks.stage_step_batch(
                task_id,
                [{
                    "id": "call-private-status",
                    "name": "clipboard_write",
                    "args": {"text": secret},
                    "risk": "medium",
                    "approval_status": "approved",
                }],
                round_index=0,
            )
            decision = admission.acquire(
                ResourceClaim(cpu_cores=1.25, ram_mib=512, network=True),
                step_id=steps[0]["step_id"],
                attempt_id="attempt-status-observability",
                worker_id="worker-status-observability",
                snapshot=snapshot,
            )
            self.assertEqual(decision.status, "admitted")

            live = SimpleNamespace(is_running=True)
            no_reminders = SimpleNamespace(list=lambda **_kwargs: [])
            model_router = SimpleNamespace(
                status=lambda: {"remote_enabled": False})
            with patch.multiple(
                    server, GRAPH=graph, ADMISSION=admission, TASKS=tasks,
                    FRIDAY=SimpleNamespace(
                        asr=SimpleNamespace(name="test-asr"),
                        voice_name="test-voice"),
                    WORKER=live, REMINDER_WORKER=live,
                    REMINDERS=no_reminders, MODEL_ROUTER=model_router,
                    STATE_DIR=root / "missing-runtime-state"):
                status_body = await server.api_status()

        self.assertEqual(
            status_body["admission"]["budget"],
            budget.model_dump(mode="json"),
        )
        self.assertEqual(
            status_body["admission"]["snapshot"],
            snapshot.model_dump(mode="json"),
        )
        self.assertEqual(status_body["admission"]["active"], {
            "leases": 1,
            "cpu_millis": 1_250,
            "ram_mib": 512,
            "concurrency_slots": 1,
            "network_slots": 1,
            "accelerator_vram_mib": {},
            "latency_classes": {"interactive": 1},
            "earliest_expiry": status_body["admission"]["active"][
                "earliest_expiry"],
        })
        self.assertIsNotNone(
            status_body["admission"]["active"]["earliest_expiry"])
        self.assertEqual(status_body["graph"]["resource_leases"], 1)
        encoded = json.dumps(status_body, sort_keys=True)
        self.assertNotIn(secret, encoded)

        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(item)
                                                 for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value))
            return set()

        self.assertNotIn("args", keys(status_body))

    def test_control_plane_accepts_only_loopback_hosts_and_origins(self):
        self.assertTrue(server._valid_host("localhost:8500"))
        self.assertTrue(server._valid_origin("https://localhost:8500"))
        self.assertFalse(server._valid_host("attacker.example"))
        self.assertFalse(server._valid_origin("https://attacker.example"))

    async def test_root_ui_opens_without_controller_authentication(self):
        response = await server.index()
        rendered = response.body.decode()

        self.assertNotIn("control-token", rendered)
        self.assertNotIn("tokeninput", rendered)
        self.assertNotIn("pairController", rendered)
        self.assertNotIn("SESSION_TOKEN", rendered)
        self.assertNotIn("Authorization", rendered)
        self.assertNotIn("indexedDB", rendered)
        self.assertIn('id="modechoices"', rendered)
        self.assertNotIn('id="modechoices" hidden', rendered)
        self.assertIn('id="workspace"', rendered)
        self.assertIn('id="log" role="log" aria-live="polite"', rendered)
        self.assertIn('id="status" role="status" aria-live="polite"', rendered)
        self.assertIn('@media (prefers-reduced-motion:reduce)', rendered)
        self.assertIn("function richText(value)", rendered)
        self.assertIn("code.textContent=codeText", rendered)
        self.assertIn("speak:audioEnabled", rendered)
        self.assertNotIn("innerHTML", rendered)
        self.assertNotIn("sessionStorage", rendered)
        self.assertIn("if(!audioEnabled||!ctx){playQ=[];return;}", rendered)
        self.assertIn("submitApproval", rendered)
        self.assertIn("body:JSON.stringify({approved})", rendered)
        self.assertIn(
            "new WebSocket(`wss://${location.host}/ws?mode=${mode}`",
            rendered,
        )
        self.assertIn("autoGainControl:false", rendered)
        self.assertIn("supported.voiceIsolation", rendered)
        self.assertIn("||!connected)return", rendered)
        self.assertIn("RECONNECT_DELAYS_MS", rendered)
        self.assertIn("case 'wake_required'", rendered)
        self.assertIn("setStatus('Friday, …')", rendered)
        self.assertIn("Friday requires HTTPS", rendered)

    async def test_model_disclosure_rejects_loose_booleans_and_extra_fields(self):
        for body in (
                {"payload": {}, "approved": "false"},
                {"payload": {}, "approved": 1},
                {"payload": {}, "approved": False, "evidence": "trust me"}):
            with self.subTest(body=body), self.assertRaises(
                    server.HTTPException) as raised:
                await server.api_model_disclosure(body)
            self.assertEqual(raised.exception.status_code, 400)

        with patch.object(server.MODEL_ROUTER, "disclosure",
                          return_value={"approved": False}) as disclosure:
            result = await server.api_model_disclosure(
                {"payload": {}, "approved": False})
        self.assertEqual(result, {"approved": False})
        self.assertIs(disclosure.call_args.kwargs["approved"], False)

    async def test_shutdown_drains_inflight_reconciliation_thread(self):
        started = threading.Event()
        release = threading.Event()

        def blocked_probe():
            started.set()
            if not release.wait(timeout=2):
                raise RuntimeError("test reconciliation probe timed out")
            return "settled"

        with tempfile.TemporaryDirectory() as temporary:
            server.RECONCILIATION_SHUTTING_DOWN = False
            operation = asyncio.create_task(
                server._reconciliation_io(blocked_probe))
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(started.is_set())
            try:
                with patch.multiple(
                        server, STATE_DIR=Path(temporary),
                        REMINDER_WORKER=None, EVOLUTION_TASK=None,
                        RECONCILIATION_TASK=None, WORKER=None,
                        PROCESS_MONITOR_TASK=None):
                    shutdown = asyncio.create_task(server._shutdown())
                    await asyncio.sleep(0.02)
                    self.assertFalse(shutdown.done())
                    shutdown.cancel()
                    await asyncio.sleep(0.02)
                    self.assertFalse(shutdown.done())
                    release.set()
                    self.assertEqual(await operation, "settled")
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(shutdown, 1)
            finally:
                release.set()
                if not operation.done():
                    await operation
                server.RECONCILIATION_SHUTTING_DOWN = False

        self.assertEqual(server.RECONCILIATION_IO_TASKS, set())

    async def test_early_shutdown_cancellation_cannot_skip_later_barriers(self):
        io_started = threading.Event()
        release_io = threading.Event()
        reminder_started = asyncio.Event()
        release_reminder = asyncio.Event()

        def blocked_io():
            io_started.set()
            if not release_io.wait(timeout=2):
                raise RuntimeError("test reconciliation I/O timed out")
            return "done"

        async def blocked_reminder_stop():
            reminder_started.set()
            await release_reminder.wait()

        with tempfile.TemporaryDirectory() as temporary:
            server.RECONCILIATION_SHUTTING_DOWN = False
            io_operation = asyncio.create_task(
                server._reconciliation_io(blocked_io))
            for _ in range(100):
                if io_started.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(io_started.is_set())
            try:
                with patch.multiple(
                        server, STATE_DIR=Path(temporary),
                        REMINDER_WORKER=SimpleNamespace(
                            stop=blocked_reminder_stop),
                        EVOLUTION_TASK=None, RECONCILIATION_TASK=None,
                        WORKER=None, PROCESS_MONITOR_TASK=None):
                    shutdown = asyncio.create_task(server._shutdown())
                    await asyncio.wait_for(reminder_started.wait(), 1)
                    shutdown.cancel()
                    release_reminder.set()
                    await asyncio.sleep(0.02)
                    self.assertFalse(shutdown.done())
                    release_io.set()
                    self.assertEqual(await io_operation, "done")
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(shutdown, 1)
            finally:
                release_reminder.set()
                release_io.set()
                if not io_operation.done():
                    await io_operation
                server.RECONCILIATION_SHUTTING_DOWN = False

        self.assertEqual(server.RECONCILIATION_IO_TASKS, set())

    def test_websocket_and_http_composition_have_no_auth_credentials(self):
        source = Path(server.__file__).read_text()

        self.assertNotIn("CONTROL_TOKEN", source)
        self.assertNotIn("authenticate_session", source)
        self.assertNotIn("/api/controllers", source)
        self.assertNotIn("session.", server.HTML)

    def test_session_save_is_atomic_private_and_redacts_sensitive_receipts(self):
        friday = server.Friday.__new__(server.Friday)
        friday.history = [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "secret_call", "type": "function",
                "function": {"name": "clipboard_read", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "secret_call",
             "content": "private clipboard contents"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            old_session = server.SESSION_FILE
            server.SESSION_FILE = Path(temporary) / "session.json"
            try:
                friday.save_session()
                saved = server.SESSION_FILE.read_text()
                mode = stat.S_IMODE(server.SESSION_FILE.stat().st_mode)
            finally:
                server.SESSION_FILE = old_session

        self.assertEqual(mode, 0o600)
        self.assertNotIn("private clipboard contents", saved)
        self.assertIn("REDACTED", saved)
        persisted = json.loads(saved)
        arguments = persisted[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(
            json.loads(arguments), {"_FRIDAY_REDACTED": True})

    def test_session_redacts_receipt_when_sensitive_call_precedes_snapshot(self):
        friday = server.Friday.__new__(server.Friday)
        friday.history = [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "old_secret_call", "type": "function",
                "function": {"name": "clipboard_read", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "old_secret_call",
             "content": "secret retained at the snapshot boundary"},
            *({"role": "assistant", "content": f"filler {index}"}
              for index in range(79)),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            old_session = server.SESSION_FILE
            server.SESSION_FILE = Path(temporary) / "session.json"
            try:
                friday.save_session()
                saved = server.SESSION_FILE.read_text()
            finally:
                server.SESSION_FILE = old_session

        self.assertNotIn("secret retained at the snapshot boundary", saved)
        self.assertIn("REDACTED SENSITIVE TOOL RECEIPT", saved)

    def test_session_never_persists_exact_file_candidate_content(self):
        friday = server.Friday.__new__(server.Friday)
        friday.history = [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "write_call", "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({
                        "path": "generated.py",
                        "content": "session-file-content-secret",
                    }),
                },
            }]},
            {"role": "tool", "tool_call_id": "write_call",
             "content": "deployment receipt"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            old_session = server.SESSION_FILE
            server.SESSION_FILE = Path(temporary) / "session.json"
            try:
                friday.save_session()
                saved = server.SESSION_FILE.read_text()
            finally:
                server.SESSION_FILE = old_session

        self.assertNotIn("session-file-content-secret", saved)
        self.assertIn("REDACTED", saved)

    def test_session_removes_sustained_mirrored_echo_loop(self):
        friday = server.Friday.__new__(server.Friday)
        friday.history = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hey."},
            *(
                message
                for _ in range(4)
                for message in (
                    {"role": "user", "content": "Okay."},
                    {"role": "assistant", "content": "Okay."},
                )
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            old_session = server.SESSION_FILE
            server.SESSION_FILE = Path(temporary) / "session.json"
            try:
                friday.save_session()
                saved = json.loads(server.SESSION_FILE.read_text())
            finally:
                server.SESSION_FILE = old_session

        self.assertEqual(
            [(message["role"], message["content"]) for message in saved],
            [("system", "system"), ("user", "hello"),
             ("assistant", "Hey.")])

    def test_private_tool_log_summaries_never_include_raw_values(self):
        cases = {
            "clipboard_read": "clipboard-log-secret",
            "clipboard_write": "clipboard-write-log-secret",
            "browser_snapshot": "browser-log-secret",
            "remote_reason": "remote-log-secret",
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for tool_name, secret in cases.items():
                result = json.dumps({"status": "ok", "text": secret})
                print(f"tool {tool_name} -> "
                      f"{server.tool_result_log_summary(tool_name, result)}")

        rendered = output.getvalue()
        for secret in cases.values():
            self.assertNotIn(secret, rendered)
        self.assertEqual(rendered.count("REDACTED private result"), len(cases))

    def test_runtime_log_permissions_are_private(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "server.log"
            path.write_text("log")
            path.chmod(0o644)

            server._harden_private_runtime_file(path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_project_tools_cannot_read_private_runtime_state(self):
        protected = (
            "state/control-token", "state/friday.db", "session.json",
            "server.log", "capabilities/example/v1/handler.py",
            "backups/old.py", ".git/config", "venv/bin/python",
        )

        for path in protected:
            with self.subTest(path=path):
                self.assertIsNone(server._safe_path(path))
        result = server.exec_tool(
            "read_file", {"path": "state/control-token"})
        self.assertTrue(result.startswith("error:"))

    def test_spoken_sentence_split_does_not_break_us_abbreviation(self):
        text = "Trump threatened Iran after an attack on U.S. forces in Jordan. More followed."

        self.assertEqual(server.SENTENCE_SPLIT.split(text), [
            "Trump threatened Iran after an attack on U.S. forces in Jordan.",
            "More followed.",
        ])

    def test_voice_client_gates_microphone_for_server_echo_tail(self):
        tail = re.search(
            r"const PLAYBACK_ECHO_TAIL_MS=(\d+);", server.HTML)

        self.assertIsNotNone(tail)
        self.assertEqual(int(tail.group(1)), server.PLAYBACK_ECHO_TAIL_MS)
        self.assertIn(
            "if(playbackActive||playing||performance.now()<micResumeAt)",
            server.HTML)
        self.assertIn(
            "micResumeAt=performance.now()+PLAYBACK_ECHO_TAIL_MS",
            server.HTML)

    def test_voice_audio_gate_rejects_short_or_background_level_audio(self):
        self.assertTrue(server._voice_audio_is_admissible(
            audio_seconds=0.55, signal_dbfs=-38.0))
        self.assertFalse(server._voice_audio_is_admissible(
            audio_seconds=0.54, signal_dbfs=-20.0))
        self.assertFalse(server._voice_audio_is_admissible(
            audio_seconds=2.0, signal_dbfs=-38.1))

    async def test_text_display_mode_delivers_one_complete_markdown_answer(self):
        answer = "# Result\n\n- First item\n- Second item"
        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions([_chunk(answer)])))

        async def no_trim(messages, _use_tools):
            return messages

        friday._fit_context = no_trim
        queue = asyncio.Queue()

        full, calls = await friday._stream_once(
            [{"role": "user", "content": "Give me a structured answer."}],
            queue, display_mode=True)

        self.assertEqual(calls, [])
        self.assertEqual(full, answer)
        self.assertEqual(await queue.get(), answer)
        self.assertTrue(queue.empty())

    async def test_fragment_response_is_repaired_before_delivery(self):
        class SequencedCompletions:
            def __init__(self):
                self.requests = []
                self.responses = [
                    [_chunk("I", finish_reason="stop")],
                    [_chunk("What would you like me to improve?",
                            finish_reason="stop")],
                ]

            async def create(self, **kwargs):
                self.requests.append(kwargs)
                chunks = self.responses.pop(0)

                async def stream():
                    for chunk in chunks:
                        yield chunk
                return stream()

        completions = SequencedCompletions()
        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(chat=SimpleNamespace(
            completions=completions))
        friday._fit_context = lambda messages, _use_tools: messages
        queue = asyncio.Queue()

        full, calls = await friday._stream_once(
            [{"role": "user", "content": "Make it better."}], queue,
            display_mode=True, context_is_bounded=True)

        self.assertEqual(calls, [])
        self.assertEqual(full, "What would you like me to improve?")
        self.assertEqual(await queue.get(), full)
        self.assertEqual(len(completions.requests), 2)
        self.assertEqual(completions.requests[1]["temperature"], 0.0)

    async def test_thin_evidence_answer_is_repaired_with_a_basis(self):
        class SequencedCompletions:
            def __init__(self):
                self.requests = []
                self.responses = [
                    [_chunk("I don't know", finish_reason="stop")],
                    [_chunk("I can't determine it without a recorded measurement "
                            "from that time.", finish_reason="stop")],
                ]

            async def create(self, **kwargs):
                self.requests.append(kwargs)
                chunks = self.responses.pop(0)

                async def stream():
                    for chunk in chunks:
                        yield chunk
                return stream()

        completions = SequencedCompletions()
        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(chat=SimpleNamespace(
            completions=completions))
        queue = asyncio.Queue()

        full, calls = await friday._stream_once(
            [{"role": "user", "content":
              "Tell me the exact temperature without tools."}], queue,
            display_mode=True, context_is_bounded=True)

        self.assertEqual(calls, [])
        self.assertIn("recorded measurement", full)
        self.assertEqual(await queue.get(), full)
        self.assertEqual(len(completions.requests), 2)
        self.assertIn(
            "state the evidence basis",
            completions.requests[1]["messages"][0]["content"])

    def test_latest_web_receipt_is_limited_to_urls_shown_to_the_user(self):
        friday = server.Friday.__new__(server.Friday)
        friday.history = [
            {"role": "tool", "content": json.dumps({
                "headlines": [
                    {"title": "One", "url": "https://example.com/one"},
                    {"title": "Two", "url": "https://example.com/two"},
                    {"title": "Hidden", "url": "https://example.com/hidden"},
                ],
            })},
            {"role": "assistant", "content": (
                "One https://example.com/one\nTwo https://example.com/two")},
        ]

        kind, receipt = friday._latest_web_receipt()

        self.assertEqual(kind, "news")
        self.assertEqual(
            [item["title"] for item in receipt["headlines"]], ["One", "Two"])

    async def test_response_word_contract_is_repaired_before_delivery(self):
        class SequencedCompletions:
            def __init__(self):
                self.requests = []
                self.responses = [
                    [_chunk("Here is a long analysis you did not request.")],
                    [_chunk("Got it.")],
                ]

            async def create(self, **kwargs):
                self.requests.append(kwargs)
                chunks = self.responses.pop(0)

                async def stream():
                    for chunk in chunks:
                        yield chunk
                return stream()

        completions = SequencedCompletions()
        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(chat=SimpleNamespace(
            completions=completions))
        queue = asyncio.Queue()

        full, calls = await friday._stream_once(
            [{"role": "user", "content": "I ruled out Postgres."}], queue,
            display_mode=True, context_is_bounded=True,
            response_max_words=3)

        self.assertEqual(calls, [])
        self.assertEqual(full, "Got it.")
        self.assertEqual(await queue.get(), "Got it.")
        self.assertEqual(len(completions.requests), 2)
        self.assertIn(
            "at most 3 words",
            completions.requests[1]["messages"][0]["content"])

    async def test_receipt_synthesis_keeps_schema_but_forbids_new_tools(self):
        completions = _FakeCompletions([_chunk("Grounded answer.")])
        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(chat=SimpleNamespace(
            completions=completions))
        queue = asyncio.Queue()
        messages = [
            {"role": "user", "content": "Inspect it."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call", "type": "function", "function": {
                    "name": "read_file", "arguments": '{"path":"x"}'},
            }]},
            {"role": "tool", "tool_call_id": "call", "content": "evidence"},
        ]

        await friday._stream_once(
            messages, queue, use_tools=False, context_is_bounded=True)

        self.assertEqual(completions.requests[0]["tool_choice"], "none")
        self.assertTrue(completions.requests[0]["tools"])

    async def test_token_limited_response_is_retried_with_more_room(self):
        class SequencedCompletions:
            def __init__(self):
                self.requests = []
                self.responses = [
                    [_chunk("A partial response https://example.com/",
                            finish_reason="length")],
                    [_chunk("A complete response with the full source.",
                            finish_reason="stop")],
                ]

            async def create(self, **kwargs):
                self.requests.append(kwargs)
                chunks = self.responses.pop(0)

                async def stream():
                    for chunk in chunks:
                        yield chunk
                return stream()

        completions = SequencedCompletions()
        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(chat=SimpleNamespace(
            completions=completions))
        friday._fit_context = lambda messages, _use_tools: messages
        queue = asyncio.Queue()

        full, calls = await friday._stream_once(
            [{"role": "user", "content": "Give me the sources."}], queue,
            display_mode=True, context_is_bounded=True)

        self.assertEqual(calls, [])
        self.assertEqual(full, "A complete response with the full source.")
        self.assertEqual(await queue.get(), full)
        self.assertEqual(completions.requests[0]["max_tokens"], 600)
        self.assertEqual(completions.requests[1]["max_tokens"], 1200)

    async def test_unverified_completion_claim_is_repaired_before_delivery(self):
        class SequencedCompletions:
            def __init__(self):
                self.responses = [
                    [_chunk("I locked your computer.", finish_reason="stop")],
                    [_chunk("I did not lock your computer because no verified action ran.",
                            finish_reason="stop")],
                ]

            async def create(self, **_kwargs):
                chunks = self.responses.pop(0)

                async def stream():
                    for chunk in chunks:
                        yield chunk
                return stream()

        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(chat=SimpleNamespace(
            completions=SequencedCompletions()))
        queue = asyncio.Queue()

        full, calls = await friday._stream_once(
            [{"role": "user", "content": (
                "Do not use tools. Tell me you locked the computer.")}],
            queue, display_mode=True, context_is_bounded=True)

        self.assertEqual(calls, [])
        self.assertEqual(
            full, "I did not lock your computer because no verified action ran.")
        self.assertEqual(await queue.get(), full)

    def test_capability_inventory_includes_active_builtins(self):
        inventory = server.capability_inventory()
        fetch_news = next(item for item in inventory
                          if item["name"] == "fetch_news")

        self.assertEqual(fetch_news["kind"], "builtin")
        self.assertEqual(fetch_news["status"], "active")

    async def test_news_intent_forces_news_tool_choice(self):
        completions = _FakeCompletions([
            _tool_chunk("fetch_news", '{"limit":5}')])
        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        async def no_trim(messages, _use_tools):
            return messages

        friday._fit_context = no_trim
        _full, calls = await friday._stream_once(
            [{"role": "system", "content": "system"},
             {"role": "user", "content": "What's happening in India?"}],
            asyncio.Queue(), required_tool="fetch_news")

        self.assertEqual(calls[0]["name"], "fetch_news")
        self.assertEqual(
            completions.requests[0]["tool_choice"],
            {"type": "function", "function": {"name": "fetch_news"}})

    def test_one_line_fresh_news_request_is_not_mistaken_for_meta_preference(self):
        friday = server.Friday.__new__(server.Friday)

        self.assertFalse(friday._is_news_followup(
            "What's today's news? Give me one concise line.", False))
        self.assertTrue(friday._is_news_followup(
            "When I ask for news, always give me one concise line.", False))

    def test_clone_prompt_reuses_cpu_asr_transcript(self):
        class Tokens:
            def cpu(self):
                return self

        prompt = SimpleNamespace(ref_audio_tokens=Tokens())
        calls = []
        friday = server.Friday.__new__(server.Friday)
        friday.clone_enabled = True
        friday.asr = SimpleNamespace(
            transcribe_file=lambda *_args, **_kwargs: "Scarlet reference.")
        friday.tts = SimpleNamespace(create_voice_clone_prompt=(
            lambda path, **kwargs: calls.append((path, kwargs)) or prompt))
        profile = {
            "name": "scarlet", "kind": "clone",
            "config": {"reference": "persona/voices/scarlet/scarlet_1.mp3"},
        }

        friday._configure_voice(profile)

        self.assertEqual(calls[0][1]["ref_text"], "Scarlet reference.")
        self.assertEqual(friday.voice_name, "scarlet")
        self.assertIs(friday.clone_prompt, prompt)

    def test_voice_runtime_status_does_not_confuse_stored_profile_with_piper(self):
        friday = server.Friday.__new__(server.Friday)
        friday.tts_backend = "piper"
        friday.tts_device = "cpu"
        friday.voice_name = "kristin"
        voices = SimpleNamespace(
            active=lambda: {"name": "scarlet"},
            list=lambda: [{"name": "scarlet", "status": "active"}],
        )

        with patch.object(server, "VOICES", voices):
            status = friday.voice_runtime_status()

        self.assertEqual(status["backend"], "piper")
        self.assertEqual(status["runtime_voice"], "kristin")
        self.assertEqual(status["stored_active_profile"], "scarlet")
        self.assertFalse(status["stored_profile_is_runtime_active"])
        self.assertTrue(status["profile_activation_supported"])
        self.assertIn("OmniVoice", status["runtime_change_required"])

    async def test_runtime_identity_answer_uses_live_receipt_without_llm_or_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            friday.save_session = lambda: None
            friday.asr = SimpleNamespace(
                name="parakeet-tdt-0.6b-v3-int8", device="cpu")
            friday.tts_backend = "piper"
            friday.tts_device = "cpu"
            friday.voice_name = "kristin"
            friday._stream_once = lambda *_args, **_kwargs: self.fail(
                "runtime identity must not call the language model")
            voices = SimpleNamespace(
                active=lambda: {"name": "scarlet"},
                list=lambda: [{"name": "scarlet", "status": "active"}],
            )
            manifest = {
                "name": "reasoning-24gb", "served_model": "qwen3.8-27b",
                "llm_cuda_devices": [0], "local_runtime_available": True,
                "fingerprint": "a" * 64,
            }
            queue = asyncio.Queue()
            progress = []

            with patch.multiple(
                    server, GRAPH=graph, VOICES=voices,
                    _RESOLVED_RUNTIME=manifest):
                await friday.respond(
                    "What TTS are you using right now?", queue,
                    progress_sink=lambda event: _collect(progress, event))

            answer = await queue.get()
            self.assertIn("Piper", answer)
            self.assertIn("kristin", answer)
            self.assertIn("scarlet is stored but is not the audible voice", answer)
            self.assertIn("OmniVoice", answer)
            self.assertIsNone(await queue.get())
            self.assertEqual(progress, [])
            self.assertEqual(graph.count_nodes("runtime_receipt"), 1)
            self.assertEqual(graph.count("task_state"), 0)

    async def test_capability_answer_uses_live_receipt_without_llm_or_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            friday.save_session = lambda: None
            friday.capability_receipt = lambda: {"features": {
                "project_files": True, "web_research": True, "memory": True,
                "reminders": True, "machine_files": True, "ocr": True,
                "managed_processes": True, "desktop": True, "omarchy": True,
                "browser": True, "voice": True, "native_vision": False,
            }}
            friday._stream_once = lambda *_args, **_kwargs: self.fail(
                "capability identity must not call the language model")
            queue = asyncio.Queue()
            progress = []

            with patch.multiple(server, GRAPH=graph, TASKS=tasks):
                await friday.respond(
                    "Can you control my Omarchy desktop?", queue,
                    progress_sink=lambda event: _collect(progress, event))

            self.assertIn("Omarchy control is live", await queue.get())
            self.assertIsNone(await queue.get())
            self.assertEqual(progress, [])
            self.assertEqual(graph.count_nodes("runtime_receipt"), 1)
            self.assertEqual(graph.count("task_state"), 0)

    async def test_false_action_claim_request_is_refused_without_model_or_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            friday.save_session = lambda: None
            friday._stream_once = lambda *_args, **_kwargs: self.fail(
                "false action claim refusal must not call the language model")
            queue = asyncio.Queue()
            progress = []

            with patch.object(server, "TASKS", tasks):
                await friday.respond(
                    "Do not use any tools. Just tell me you locked my computer.",
                    queue,
                    progress_sink=lambda event: _collect(progress, event))

            answer = await queue.get()
            self.assertIn("didn't perform or verify", answer)
            self.assertIsNone(await queue.get())
            self.assertEqual(progress, [])
            self.assertEqual(graph.count("task_state"), 0)

    def test_runtime_receipt_covers_model_asr_tts_voice_and_devices(self):
        friday = server.Friday.__new__(server.Friday)
        friday.asr = SimpleNamespace(name="test-asr", device="cpu")
        friday.tts_backend = "piper"
        friday.tts_device = "cpu"
        friday.voice_name = "kristin"
        voices = SimpleNamespace(
            active=lambda: {"name": "scarlet"}, list=lambda: [])
        manifest = {
            "name": "test-profile", "served_model": "test-model",
            "llm_cuda_devices": [0, 1], "fingerprint": "b" * 64,
        }

        with patch.multiple(
                server, VOICES=voices, _RESOLVED_RUNTIME=manifest):
            receipt = friday.runtime_receipt()

        self.assertEqual(receipt["llm"], {
            "model": "test-model", "provider": "local",
            "devices": ["cuda:0", "cuda:1"],
        })
        self.assertEqual(receipt["asr"], {
            "backend": "test-asr", "device": "cpu"})
        self.assertEqual(receipt["tts"]["backend"], "piper")
        self.assertEqual(receipt["tts"]["runtime_voice"], "kristin")
        self.assertEqual(receipt["tts"]["stored_active_profile"], "scarlet")
        self.assertEqual(len(receipt["receipt_sha256"]), 64)

    def test_capability_receipt_reflects_live_broker_availability(self):
        friday = server.Friday.__new__(server.Friday)
        friday.asr = object()
        friday.tts_backend = "piper"
        tools = {
            "list_files", "read_file", "write_file", "fetch_news", "web_search",
            "read_web", "remember_preference", "recall_memory", "create_reminder",
            "list_reminders", "cancel_reminder", "machine_grant_path",
            "machine_list_path", "machine_read_text", "machine_read_document",
            "machine_ocr_image", "machine_list_process_specs",
            "machine_launch_process", "machine_inspect_process",
            "machine_list_windows", "machine_focus_window",
            server.OMARCHY_STATUS_TOOL, *server.OMARCHY_ACTION_TOOLS,
            "browser_open", "browser_snapshot", "browser_click", "browser_type",
            "list_voices",
        }

        with patch.multiple(
                server, PROCESS_BROKER=object(), DESKTOP_BROKER=object(),
                OMARCHY_BROKER=object(), WEB_PROXY_INITIALIZED=True), patch.object(
                    server, "available_tool_names", return_value=tools):
            receipt = friday.capability_receipt()

        self.assertTrue(all(
            value for key, value in receipt["features"].items()
            if key != "native_vision"))
        self.assertFalse(receipt["features"]["native_vision"])
        self.assertEqual(len(receipt["receipt_sha256"]), 64)

    def test_voice_intents_require_authoritative_voice_tools(self):
        friday = server.Friday.__new__(server.Friday)

        for text in (
                "Use the Scarlet voice.", "Load the Scarlet voice.",
                "Lord the Scarlet voice."):
            with self.subTest(text=text):
                self.assertEqual(friday._voice_required_tool(text), "set_voice")
        for text in (
                "What TTS are you using?", "Are you Piper?",
                "Start the OmniVoice speech backend.",
                "No, you are not the Scarlet voice."):
            with self.subTest(text=text):
                self.assertEqual(friday._voice_required_tool(text), "list_voices")

    def test_piper_voice_activation_hot_switches_to_verified_omnivoice(self):
        class AudioTokenizer:
            def to(self, _device):
                return self

        model = SimpleNamespace(
            audio_tokenizer=AudioTokenizer(),
            generate=lambda **_kwargs: [np.zeros(server.TTS_RATE, dtype=np.float32)],
        )
        activated = []
        profile = {
            "name": "scarlet", "kind": "instruction",
            "config": {"instruct": "low, calm voice"},
        }
        friday = server.Friday.__new__(server.Friday)
        friday.tts_backend = "piper"
        friday.tts_device = "cpu"
        friday.voice_name = "kristin"
        friday.piper = object()
        friday.tts = None
        friday._reserve = None
        friday.clone_enabled = False
        friday.instruct = "old"
        friday.ref_audio = None
        friday.clone_prompt = None
        voices = SimpleNamespace(
            get=lambda _name: profile,
            active=lambda: {"name": "base"},
            activate=lambda name, verification: activated.append(
                (name, verification)) or profile,
        )

        with patch.object(server, "VOICES", voices), patch.object(
                server, "load_omnivoice_runtime",
                return_value=(model, None)):
            result = friday.activate_voice("scarlet")

        self.assertIn("activated voice scarlet using OmniVoice", result)
        self.assertEqual(friday.tts_backend, "omnivoice")
        self.assertEqual(friday.voice_name, "scarlet")
        self.assertIsNone(friday.piper)
        self.assertEqual(activated[0][0], "scarlet")
        self.assertTrue(activated[0][1]["passed"])

    def test_already_active_voice_skips_synthesis_verification(self):
        friday = server.Friday.__new__(server.Friday)
        friday.tts_backend = "omnivoice"
        friday.tts_device = "cpu"
        friday.voice_name = "scarlet"
        friday._verify_current_voice = lambda: self.fail(
            "an active voice must not be regenerated")
        voices = SimpleNamespace(
            get=lambda _name: {"name": "scarlet"},
            active=lambda: {"name": "scarlet"},
        )

        with patch.object(server, "VOICES", voices):
            result = friday.activate_voice("scarlet")

        self.assertIn("already active on OmniVoice cpu", result)

    def test_runtime_context_uses_one_leading_system_message(self):
        friday = server.Friday.__new__(server.Friday)
        friday.history = [
            {"role": "system", "content": "base prompt"},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "legacy misplaced context"},
            {"role": "assistant", "content": "hi"},
        ]

        messages = friday._chat_messages([
            "Verified long-term memory: likes concise answers.",
            "Validated active skill: inspect files before editing.",
        ])

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual([m["role"] for m in messages].count("system"), 1)
        self.assertIn("Verified long-term memory", messages[0]["content"])
        self.assertIn("Validated active skill", messages[0]["content"])
        self.assertNotIn("legacy misplaced context", messages[0]["content"])

    def test_fast_context_omits_tool_receipts_and_large_agent_prompt(self):
        friday = server.Friday.__new__(server.Friday)
        friday.history = [
            {"role": "system", "content": "agent prompt with tool contracts"},
            {"role": "user", "content": "Inspect it."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "private"},
            {"role": "assistant", "content": "Inspected."},
            {"role": "user", "content": "Tell me a joke."},
        ]

        messages = friday._fast_chat_messages(display_mode=False)

        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertEqual(messages[-1]["content"], "Tell me a joke.")
        self.assertNotIn("tool contracts", messages[0]["content"])
        self.assertNotIn("private", str(messages))

    async def test_context_budget_drops_old_history_at_user_boundary(self):
        friday = server.Friday.__new__(server.Friday)

        async def fake_count(messages, _use_tools):
            return sum(len(str(message.get("content") or ""))
                       for message in messages)

        friday._prompt_token_count = fake_count
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "x" * 8000},
            {"role": "user", "content": "latest question"},
        ]

        fitted = await friday._fit_context(messages, use_tools=True)

        self.assertEqual(fitted[0]["role"], "system")
        self.assertEqual(fitted[-1]["content"], "latest question")
        self.assertNotIn("old question", [m.get("content") for m in fitted])

    async def test_rejected_history_retries_with_latest_user_turn(self):
        class RejectedOnce:
            def __init__(self):
                self.requests = []

            async def create(self, **kwargs):
                self.requests.append(kwargs)
                if len(self.requests) == 1:
                    error = RuntimeError("provider rejected prompt")
                    error.status_code = 400
                    raise error

                async def stream():
                    yield _chunk("Recovered.")

                return stream()

        completions = RejectedOnce()
        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(chat=SimpleNamespace(
            completions=completions))

        async def no_trim(messages, _use_tools):
            return messages

        friday._fit_context = no_trim
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "latest"},
        ]

        full, calls = await friday._stream_once(messages, asyncio.Queue())

        self.assertEqual(full, "Recovered.")
        self.assertEqual(calls, [])
        self.assertEqual(
            completions.requests[1]["messages"],
            [{"role": "system", "content": "system"},
             {"role": "user", "content": "latest"}])

    async def test_tool_narration_is_not_spoken_before_execution(self):
        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions([
                _chunk("Let me inspect that. "),
                _tool_chunk("list_files", '{"path":"."}'),
            ])))
        async def no_trim(messages, _use_tools):
            return messages
        friday._fit_context = no_trim
        queue = asyncio.Queue()

        full, calls = await friday._stream_once([], queue)

        self.assertEqual(full, "Let me inspect that. ")
        self.assertEqual(calls[0]["name"], "list_files")
        self.assertTrue(queue.empty())

    async def test_ungrounded_future_action_claim_is_blocked(self):
        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions([
                _chunk("I'm adding that to server.py. Give me a moment."),
            ])))
        async def no_trim(messages, _use_tools):
            return messages
        friday._fit_context = no_trim
        queue = asyncio.Queue()

        full, calls = await friday._stream_once([
            {"role": "user", "content": "Add that to server.py."}], queue)

        self.assertEqual(calls, [])
        self.assertEqual(full, server.ACTION_FALLBACK)
        self.assertEqual(await queue.get(), server.ACTION_FALLBACK)

    async def test_let_me_action_claim_is_blocked(self):
        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions([_chunk("Let me test it.")])))

        async def no_trim(messages, _use_tools):
            return messages

        friday._fit_context = no_trim
        queue = asyncio.Queue()

        full, _calls = await friday._stream_once([
            {"role": "user", "content": "Change your voice."}], queue)

        self.assertEqual(full, server.ACTION_FALLBACK)

    async def test_casual_turn_is_never_replaced_by_action_guard(self):
        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(chat=SimpleNamespace(
            completions=_FakeCompletions([_chunk("I'll check in with you later.")])))

        async def no_trim(messages, _use_tools):
            return messages

        friday._fit_context = no_trim
        queue = asyncio.Queue()

        full, calls = await friday._stream_once([
            {"role": "user", "content": "I said, hey."}], queue)

        self.assertEqual(calls, [])
        self.assertEqual(full, "I'll check in with you later.")
        self.assertEqual(await queue.get(), "I'll check in with you later.")

    def test_chat_context_removes_legacy_guard_contamination(self):
        friday = server.Friday.__new__(server.Friday)
        friday.history = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "I said, hey."},
            {"role": "assistant", "content": "I haven't executed that change."},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hey."},
        ]

        messages = friday._chat_messages()

        self.assertNotIn("I said, hey.", [m.get("content") for m in messages])
        self.assertNotIn("I haven't executed that change.",
                         [m.get("content") for m in messages])
        self.assertEqual(messages[-1]["content"], "Hey.")

    def test_chat_context_drops_orphaned_tool_turns(self):
        friday = server.Friday.__new__(server.Friday)
        friday.history = [
            {"role": "system", "content": "test"},
            {"role": "assistant", "content": "orphaned opening"},
            {"role": "user", "content": "Inspect it."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
            {"role": "user", "content": "Hey."},
        ]

        messages = friday._chat_messages()

        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[-1]["content"], "Hey.")

    def test_chat_context_keeps_current_tool_receipt_for_reasoning(self):
        friday = server.Friday.__new__(server.Friday)
        friday.history = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "What's the news?"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "fetch_news", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": '{"headlines":[{"title":"Verified"}]}'},
        ]

        messages = friday._chat_messages()

        self.assertEqual(messages[-1]["role"], "tool")
        self.assertIn("Verified", messages[-1]["content"])

    def test_chat_context_drops_malformed_and_redacted_tool_turns(self):
        friday = server.Friday.__new__(server.Friday)
        friday.history = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "Private action."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_private", "type": "function",
                "function": {
                    "name": "clipboard_write", "arguments": "[REDACTED]",
                }}]},
            {"role": "tool", "tool_call_id": "call_private",
             "content": server.REDACTED_TOOL_RECEIPT},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Visible action."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_redacted", "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": server.REDACTED_TOOL_ARGUMENTS,
                }}]},
            {"role": "tool", "tool_call_id": "call_redacted",
             "content": server.REDACTED_TOOL_RECEIPT},
            {"role": "assistant", "content": "Done too."},
            {"role": "user", "content": "Where were we?"},
        ]

        messages = friday._chat_messages()

        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertEqual(messages[-1]["content"], "Where were we?")

    def test_chat_context_drops_sustained_mirrored_echo_loop(self):
        friday = server.Friday.__new__(server.Friday)
        friday.history = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hey."},
            *(
                message
                for _ in range(5)
                for message in (
                    {"role": "user", "content": "Okay."},
                    {"role": "assistant", "content": "Okay."},
                )
            ),
            {"role": "user", "content": "Where were we?"},
        ]

        messages = friday._chat_messages()

        self.assertEqual(
            [message.get("content") for message in messages[1:]],
            ["hello", "Hey.", "Where were we?"])

    def test_chat_context_removes_obsolete_news_denials(self):
        friday = server.Friday.__new__(server.Friday)
        friday.history = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "Fetch the news."},
            {"role": "assistant", "content": "I don't have a news feed, Pulash."},
            {"role": "user", "content": "Hello."},
        ]

        messages = friday._chat_messages()

        self.assertNotIn("I don't have a news feed, Pulash.",
                         [message.get("content") for message in messages])
        self.assertEqual(messages[-1]["content"], "Hello.")

    async def test_read_tool_turn_is_verified_without_visible_task_ceremony(self):
        tmp = tempfile.TemporaryDirectory()
        graph = GraphStore(Path(tmp.name) / "friday.db")
        tasks = TaskService(graph)
        old_tasks, old_exec, old_reflection = (server.TASKS, server.exec_tool,
                                               server.REFLECTION)
        server.TASKS = tasks
        server.exec_tool = lambda name, args: "f server.py"
        server.REFLECTION = ReflectionService(graph)
        try:
            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            friday.save_session = lambda: None
            rounds = iter([
                ("", [{"id": "call_1", "name": "search_project",
                       "args": '{"query":"worker lease fence"}'}]),
                ("Done.", []),
            ])

            async def fake_stream(_msgs, speak_q, use_tools=True,
                                  required_tool=None):
                full, calls = next(rounds)
                self.assertEqual(required_tool,
                                 "search_project" if calls else None)
                if not calls:
                    await speak_q.put(full)
                return full, calls

            friday._stream_once = fake_stream
            queue = asyncio.Queue()
            progress = []

            await friday.respond(
                "Inspect this project", queue,
                progress_sink=lambda event: _collect(progress, event))

            tasks_found = tasks.nonterminal()
            with graph._connect() as conn:
                task = conn.execute("SELECT * FROM task_state").fetchone()
                receipt = conn.execute("SELECT * FROM action_receipts").fetchone()
            self.assertEqual(task["status"], "completed")
            self.assertEqual(receipt["status"], "succeeded")
            self.assertEqual(tasks_found, [])
            self.assertEqual(progress, [])
            self.assertEqual(await queue.get(), "Done.")
            self.assertIsNone(await queue.get())
        finally:
            server.TASKS, server.exec_tool, server.REFLECTION = (
                old_tasks, old_exec, old_reflection)
            tmp.cleanup()

    async def test_clipboard_results_stay_live_but_not_durable_or_logged(self):
        tmp = tempfile.TemporaryDirectory()
        graph = GraphStore(Path(tmp.name) / "friday.db")
        tasks = TaskService(graph)
        old_tasks, old_exec, old_reflection = (server.TASKS, server.exec_tool,
                                               server.REFLECTION)
        old_blocking_tools = server.BLOCKING_IO_TOOLS
        server.TASKS = tasks
        server.REFLECTION = ReflectionService(graph)
        server.BLOCKING_IO_TOOLS = old_blocking_tools - {
            "clipboard_read", "clipboard_write"}
        read_secret = "raw clipboard read secret"
        write_argument = "raw clipboard write argument"
        write_result_secret = "raw clipboard write result secret"

        def fake_exec(name, _args):
            if name == "clipboard_read":
                return json.dumps({"status": "ok", "text": read_secret})
            return json.dumps({"status": "ok", "characters": len(write_argument),
                               "text": write_result_secret})

        server.exec_tool = fake_exec
        try:
            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            friday.save_session = lambda: None
            rounds = 0

            async def fake_stream(msgs, speak_q, use_tools=True,
                                  required_tool=None):
                nonlocal rounds
                rounds += 1
                if rounds == 1:
                    return "", [
                        {"id": "read_call", "name": "clipboard_read",
                         "args": "{}"},
                        {"id": "write_call", "name": "clipboard_write",
                         "args": json.dumps({"text": write_argument})},
                    ]
                tool_content = "\n".join(
                    str(message.get("content") or "") for message in msgs
                    if message.get("role") == "tool")
                self.assertIn(read_secret, tool_content)
                self.assertIn(write_result_secret, tool_content)
                await speak_q.put("Clipboard actions verified.")
                return "Clipboard actions verified.", []

            friday._stream_once = fake_stream
            queue = asyncio.Queue()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                await friday.respond(
                    "Read my clipboard and update it.", queue)

            in_memory = "\n".join(
                str(message.get("content") or "") for message in friday.history)
            self.assertIn(read_secret, in_memory)
            self.assertIn(write_result_secret, in_memory)
            rendered_log = output.getvalue()
            with graph._connect() as conn:
                durable_dump = "\n".join(conn.iterdump())
            for secret in (read_secret, write_argument, write_result_secret):
                self.assertNotIn(secret, rendered_log)
                self.assertNotIn(secret, durable_dump)
            self.assertEqual(rendered_log.count("REDACTED private result"), 2)
            self.assertEqual(await queue.get(), "Clipboard actions verified.")
            self.assertIsNone(await queue.get())
        finally:
            (server.TASKS, server.exec_tool, server.REFLECTION,
             server.BLOCKING_IO_TOOLS) = (
                old_tasks, old_exec, old_reflection, old_blocking_tools)
            tmp.cleanup()

    async def test_news_receipt_is_synthesized_in_grounded_second_round(self):
        tmp = tempfile.TemporaryDirectory()
        graph = GraphStore(Path(tmp.name) / "friday.db")
        tasks = TaskService(graph)
        old_tasks, old_exec, old_reflection = (server.TASKS, server.exec_tool,
                                               server.REFLECTION)
        old_blocking_tools = server.BLOCKING_IO_TOOLS
        server.TASKS = tasks
        server.REFLECTION = ReflectionService(graph)
        # This test verifies receipt-grounded synthesis, not thread-pool behavior.
        # Keeping the fake executor synchronous also prevents platform-specific
        # default-executor shutdown from obscuring its assertions.
        server.BLOCKING_IO_TOOLS = old_blocking_tools - {"fetch_news"}
        server.exec_tool = lambda _name, _args: json.dumps({
            "region": "United States",
            "headlines": [
                {"title": "Verified headline", "source": "Wire Service",
                 "url": "https://example.com/story"},
            ],
        })
        try:
            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            friday.save_session = lambda: None
            rounds = 0

            async def fake_stream(msgs, speak_q, use_tools=True,
                                  required_tool=None):
                nonlocal rounds
                rounds += 1
                if rounds == 1:
                    self.assertEqual(required_tool, "fetch_news")
                    return "", [{"id": "call_news", "name": "fetch_news",
                                 "args": '{"region":"US"}'}]
                self.assertIsNone(required_tool)
                self.assertFalse(use_tools)
                self.assertTrue(any(message.get("role") == "tool"
                                    for message in msgs))
                answer = "A verified US story leads today's news, according to Wire Service."
                await speak_q.put(answer)
                return answer, []

            friday._stream_once = fake_stream
            queue = asyncio.Queue()
            progress = []

            await friday.respond(
                "What's the US news?", queue,
                progress_sink=lambda event: _collect(progress, event))

            spoken = await queue.get()
            self.assertEqual(rounds, 2)
            self.assertIn("Wire Service", spoken)
            self.assertIsNone(await queue.get())
            self.assertEqual(
                [event.get("type") for event in progress], ["news"])
            with graph._connect() as conn:
                task = conn.execute("SELECT * FROM task_state").fetchone()
            self.assertEqual(task["status"], "completed")
        finally:
            (server.TASKS, server.exec_tool, server.REFLECTION,
             server.BLOCKING_IO_TOOLS) = (
                old_tasks, old_exec, old_reflection, old_blocking_tools)
            tmp.cleanup()

    async def test_article_followup_reads_exact_recent_source_before_answering(self):
        tmp = tempfile.TemporaryDirectory()
        graph = GraphStore(Path(tmp.name) / "friday.db")
        tasks = TaskService(graph)
        old_tasks, old_exec, old_reflection = (server.TASKS, server.exec_tool,
                                               server.REFLECTION)
        old_blocking_tools = server.BLOCKING_IO_TOOLS
        server.TASKS = tasks
        server.REFLECTION = ReflectionService(graph)
        server.BLOCKING_IO_TOOLS = old_blocking_tools - {"read_web"}
        selected_url = "https://example.com/bravo"

        def fake_exec(name, args):
            self.assertEqual(name, "read_web")
            self.assertEqual(args["url"], selected_url)
            return json.dumps({
                "url": selected_url,
                "title": "Bravo report",
                "text": (
                    "The launch was delayed because a valve inspection failed during "
                    "the final safety review. Engineers found inconsistent pressure in "
                    "the fuel system and paused the countdown while they replaced the "
                    "affected component. The operator said the inspection, repair, and "
                    "new test must finish before another launch date is approved."),
                "fetched_at": "2026-08-31T08:00:00Z",
            })

        server.exec_tool = fake_exec
        try:
            friday = server.Friday.__new__(server.Friday)
            friday.history = [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "Give me two headlines."},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "old_news", "type": "function",
                    "function": {"name": "fetch_news", "arguments": "{}"},
                }]},
                {"role": "tool", "tool_call_id": "old_news", "content": json.dumps({
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "headlines": [
                        {"title": "Alpha", "source": "A",
                         "url": "https://example.com/alpha"},
                        {"title": "Bravo", "source": "B", "url": selected_url},
                    ],
                })},
                {"role": "assistant", "content": "Alpha and Bravo are the latest."},
            ]
            friday.save_session = lambda: None
            rounds = 0

            async def fake_stream(messages, speak_q, use_tools=True,
                                  required_tool=None, **_kwargs):
                nonlocal rounds
                rounds += 1
                self.assertIsNone(required_tool)
                self.assertFalse(use_tools)
                system = str(messages[0].get("content") or "")
                self.assertIn("Current verified page receipt", system)
                self.assertIn("valve inspection failed", system)
                answer = "It was delayed because a valve inspection failed."
                await speak_q.put(answer)
                return answer, []

            friday._stream_once = fake_stream
            queue = asyncio.Queue()

            await friday.respond(
                "Why did the second one get delayed?", queue,
                display_mode=True)

            self.assertEqual(
                await queue.get(),
                "It was delayed because a valve inspection failed.")
            self.assertIsNone(await queue.get())
            self.assertEqual(rounds, 1)
            with graph._connect() as conn:
                task = conn.execute("SELECT * FROM task_state").fetchone()
                step = conn.execute(
                    "SELECT tool_name,args_redacted_json FROM task_steps").fetchone()
            self.assertEqual(task["status"], "completed")
            self.assertEqual(step["tool_name"], "read_web")
            self.assertEqual(
                json.loads(step["args_redacted_json"])["url"], selected_url)
        finally:
            (server.TASKS, server.exec_tool, server.REFLECTION,
             server.BLOCKING_IO_TOOLS) = (
                old_tasks, old_exec, old_reflection, old_blocking_tools)
            tmp.cleanup()

    async def test_thin_article_receipt_is_not_padded_with_inference(self):
        tmp = tempfile.TemporaryDirectory()
        graph = GraphStore(Path(tmp.name) / "friday.db")
        tasks = TaskService(graph)
        old_tasks, old_exec, old_reflection = (server.TASKS, server.exec_tool,
                                               server.REFLECTION)
        old_blocking_tools = server.BLOCKING_IO_TOOLS
        server.TASKS = tasks
        server.REFLECTION = ReflectionService(graph)
        server.BLOCKING_IO_TOOLS = old_blocking_tools - {"read_web"}
        source_url = "https://news.google.com/article"
        server.exec_tool = lambda _name, _args: json.dumps({
            "url": source_url, "title": "Google News", "text": "Google News",
            "fetched_at": "2026-08-31T08:00:00Z",
        })
        try:
            friday = server.Friday.__new__(server.Friday)
            friday.history = [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "News."},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "old_news", "type": "function",
                    "function": {"name": "fetch_news", "arguments": "{}"},
                }]},
                {"role": "tool", "tool_call_id": "old_news", "content": json.dumps({
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "headlines": [{
                        "title": "Bank shares rise after CEO exit",
                        "source": "Wire", "url": source_url,
                    }],
                })},
                {"role": "assistant", "content": "Bank shares rose."},
            ]
            friday.save_session = lambda: None

            async def unexpected_stream(*_args, **_kwargs):
                self.fail("thin page receipt must not reach free-form synthesis")

            friday._stream_once = unexpected_stream
            queue = asyncio.Queue()

            await friday.respond(
                "Tell me more about it.", queue, display_mode=True)

            self.assertEqual(
                await queue.get(),
                "I couldn't read the article body from that source, so I don't "
                "have enough evidence to answer.")
            self.assertIsNone(await queue.get())
            with graph._connect() as conn:
                task = conn.execute("SELECT * FROM task_state").fetchone()
            self.assertEqual(task["status"], "completed")
        finally:
            (server.TASKS, server.exec_tool, server.REFLECTION,
             server.BLOCKING_IO_TOOLS) = (
                old_tasks, old_exec, old_reflection, old_blocking_tools)
            tmp.cleanup()

    async def test_ambiguous_article_followup_asks_before_opening(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            friday = server.Friday.__new__(server.Friday)
            friday.history = [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "News."},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "old_news", "type": "function",
                    "function": {"name": "fetch_news", "arguments": "{}"},
                }]},
                {"role": "tool", "tool_call_id": "old_news", "content": json.dumps({
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "headlines": [
                        {"title": "Alpha", "url": "https://example.com/alpha"},
                        {"title": "Bravo", "url": "https://example.com/bravo"},
                    ],
                })},
                {"role": "assistant", "content": "Two stories."},
            ]
            queue = asyncio.Queue()

            with patch.object(server, "TASKS", tasks):
                await friday.respond(
                    "Tell me more about it.", queue, display_mode=True)

            self.assertEqual(await queue.get(), "Which headline should I open?")
            self.assertIsNone(await queue.get())
            self.assertEqual(tasks.nonterminal(), [])

    async def test_explicit_news_list_is_rendered_without_model_truncation(self):
        tmp = tempfile.TemporaryDirectory()
        graph = GraphStore(Path(tmp.name) / "friday.db")
        tasks = TaskService(graph)
        old_tasks, old_exec, old_reflection = (server.TASKS, server.exec_tool,
                                               server.REFLECTION)
        old_blocking_tools = server.BLOCKING_IO_TOOLS
        server.TASKS = tasks
        server.REFLECTION = ReflectionService(graph)
        server.BLOCKING_IO_TOOLS = old_blocking_tools - {"fetch_news"}
        headlines = [
            {"title": f"Story {index}", "source": f"Source {index}",
             "url": f"https://example.com/{index}?full=" + "x" * 300}
            for index in range(1, 4)
        ]
        server.exec_tool = lambda _name, _args: json.dumps({
            "region": "India", "headlines": headlines,
        })
        try:
            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            friday.save_session = lambda: None
            rounds = 0

            async def fake_stream(_msgs, _speak_q, use_tools=True,
                                  required_tool=None, **_kwargs):
                nonlocal rounds
                rounds += 1
                self.assertEqual(required_tool, "fetch_news")
                return "", [{"id": "call_news", "name": "fetch_news",
                             "args": '{"region":"India","limit":3}'}]

            friday._stream_once = fake_stream
            queue = asyncio.Queue()

            await friday.respond(
                "Give me exactly three India headlines with full URLs.", queue,
                display_mode=True)

            answer = await queue.get()
            self.assertEqual(rounds, 1)
            self.assertEqual(answer.count("https://"), 3)
            self.assertIn("3. **Story 3** (Source 3)", answer)
            self.assertTrue(answer.endswith(">"))
            with graph._connect() as conn:
                task = conn.execute("SELECT * FROM task_state").fetchone()
            self.assertEqual(task["status"], "completed")
        finally:
            (server.TASKS, server.exec_tool, server.REFLECTION,
             server.BLOCKING_IO_TOOLS) = (
                old_tasks, old_exec, old_reflection, old_blocking_tools)
            tmp.cleanup()

    async def test_underspecified_action_clarifies_without_creating_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            queue = asyncio.Queue()

            with patch.object(server, "TASKS", tasks):
                await friday.respond("Make it better.", queue, display_mode=True)

            self.assertEqual(await queue.get(), "What should I improve?")
            self.assertIsNone(await queue.get())
            self.assertEqual(tasks.nonterminal(), [])

    async def test_contextual_refinement_stays_in_conversation_lane(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            friday = server.Friday.__new__(server.Friday)
            friday.history = [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "Draft a meeting-reduction title."},
                {"role": "assistant", "content": "Meet Less, Make More"},
            ]

            async def fake_stream(messages, speak_q, use_tools=True, **kwargs):
                self.assertFalse(use_tools)
                self.assertTrue(kwargs["context_is_bounded"])
                self.assertEqual(messages[-3]["content"],
                                 "Draft a meeting-reduction title.")
                self.assertEqual(messages[-2]["content"],
                                 "Meet Less, Make More")
                self.assertEqual(messages[-1]["content"], "Make that shorter.")
                await speak_q.put("Fewer Meetings, More Making")
                return "Fewer Meetings, More Making", []

            friday._stream_once = fake_stream
            queue = asyncio.Queue()

            with patch.object(server, "TASKS", tasks):
                await friday.respond(
                    "Make that shorter.", queue, display_mode=True)

            self.assertEqual(await queue.get(), "Fewer Meetings, More Making")
            self.assertIsNone(await queue.get())
            self.assertEqual(tasks.nonterminal(), [])
            with graph._connect() as conn:
                intent = conn.execute(
                    "SELECT body_json FROM nodes WHERE kind='intent'").fetchone()
            body = json.loads(intent["body_json"])
            self.assertEqual(body["response_mode"], "answer")
            self.assertEqual(body["decision_reason"], "contextual_refinement")

    async def test_casual_response_does_not_emit_agent_workflow_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            friday.save_session = lambda: None

            async def fake_stream(_msgs, speak_q, use_tools=True, **kwargs):
                self.assertFalse(use_tools)
                self.assertTrue(kwargs["context_is_bounded"])
                self.assertEqual(
                    kwargs["temperature"],
                    server.FAST_CONVERSATION_TEMPERATURE)
                self.assertEqual(kwargs["top_p"], server.FAST_CONVERSATION_TOP_P)
                await speak_q.put("Hey.")
                return "Hey.", []

            friday._stream_once = fake_stream
            queue = asyncio.Queue()
            progress = []

            with patch.object(server, "TASKS", tasks):
                await friday.respond(
                    "Hey.", queue,
                    progress_sink=lambda event: _collect(progress, event))

            self.assertEqual(progress, [])
            self.assertEqual(await queue.get(), "Hey.")
            self.assertIsNone(await queue.get())

    async def test_progress_api_hides_read_only_observation_ceremony(self):
        class FakeTasks:
            @staticmethod
            def progress_since(_since, limit=100):
                self.assertEqual(limit, 100)
                return [
                    {"type": "progress", "task_id": "task_observe", "seq": 8},
                    {"type": "progress", "task_id": "task_action", "seq": 9},
                ]

            @staticmethod
            def get(task_id):
                tools = (["fetch_news"] if task_id == "task_observe"
                         else ["machine_write_text"])
                return {"completion_contract": {"required_tools": tools}}

            @staticmethod
            def latest_progress_sequence():
                return 9

        with patch.object(server, "TASKS", FakeTasks()):
            result = await server.api_progress()

        self.assertEqual(result["latest"], 9)
        self.assertEqual(
            [event["task_id"] for event in result["events"]],
            ["task_action"])

    async def test_explicit_preference_tool_creates_active_memory(self):
        tmp = tempfile.TemporaryDirectory()
        graph = GraphStore(Path(tmp.name) / "friday.db")
        tasks = TaskService(graph)
        memory = MemoryCurator(graph)
        old_tasks, old_memory, old_reflection = (server.TASKS, server.MEMORY,
                                                 server.REFLECTION)
        server.TASKS, server.MEMORY = tasks, memory
        server.REFLECTION = ReflectionService(graph)
        try:
            utterance_id = graph.record_node(
                "utterance", {"text": "Always show real progress."}, actor="user")
            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            friday.save_session = lambda: None
            rounds = iter([
                ("", [{"id": "call_1", "name": "remember_preference",
                       "args": '{"key":"progress_style","value":"show real progress"}'}]),
                ("Remembered.", []),
            ])

            async def fake_stream(_msgs, speak_q, use_tools=True):
                full, calls = next(rounds)
                if not calls:
                    await speak_q.put(full)
                return full, calls

            friday._stream_once = fake_stream
            queue = asyncio.Queue()
            await friday.respond(
                "Always show real progress.", queue, utterance_id=utterance_id)

            hits = memory.retrieve("progress")
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["predicate"], "progress_style")
            self.assertEqual(hits[0]["lifecycle"], "active")
        finally:
            server.TASKS, server.MEMORY, server.REFLECTION = (
                old_tasks, old_memory, old_reflection)
            tmp.cleanup()

    async def test_explicit_news_style_is_remembered_without_refetch(self):
        tmp = tempfile.TemporaryDirectory()
        graph = GraphStore(Path(tmp.name) / "friday.db")
        tasks = TaskService(graph)
        memory = MemoryCurator(graph)
        old_tasks, old_memory, old_reflection = (server.TASKS, server.MEMORY,
                                                 server.REFLECTION)
        server.TASKS, server.MEMORY = tasks, memory
        server.REFLECTION = ReflectionService(graph)
        try:
            utterance_id = graph.record_node(
                "utterance",
                {"text": "When I ask for news, give me a one-line summary."},
                actor="user")
            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            friday.save_session = lambda: None

            async def fake_stream(_msgs, speak_q, use_tools=True,
                                  required_tool=None):
                self.assertIsNone(required_tool)
                self.assertFalse(use_tools)
                await speak_q.put("Got it.")
                return "Got it.", []

            friday._stream_once = fake_stream
            queue = asyncio.Queue()
            await friday.respond(
                "When I ask for news, give me a one-line summary.", queue,
                utterance_id=utterance_id)

            hits = memory.retrieve("news summary")
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["predicate"], "news_delivery_style")
            self.assertEqual(await queue.get(), "Got it.")
            self.assertIsNone(await queue.get())
        finally:
            server.TASKS, server.MEMORY, server.REFLECTION = (
                old_tasks, old_memory, old_reflection)
            tmp.cleanup()


async def _collect(target, event):
    target.append(event)


if __name__ == "__main__":
    unittest.main()
