"""Process-level acceptance tests for durable step restart recovery.

These tests intentionally terminate a Python worker with SIGKILL.  The small
child modes below use the real SQLite database and TaskService API; the append
journal represents the external tool invocation boundary that SQLite cannot
roll back.  A stdout handshake makes each crash window deterministic.
"""

from __future__ import annotations

import argparse
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / "venv" / "bin" / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from friday_core import GraphStore, TaskService  # noqa: E402
from friday_core.graph import canonical_json, sha256_text  # noqa: E402

from tests.platform_markers import require_platform

require_platform('linux', 'darwin')


def _claim_record(claim: Any) -> dict[str, Any]:
    """Return the non-secret invocation identity written before side effects."""
    return {
        "step_id": claim.step_id,
        "batch_id": claim.batch_id,
        "task_id": claim.task_id,
        "ordinal": claim.ordinal,
        "tool_call_id": claim.tool_call_id,
        "tool_name": claim.tool_name,
        "args": claim.args,
        "args_sha256": sha256_text(canonical_json(claim.args)),
        "action_id": claim.action_id,
        "idempotency_key": claim.idempotency_key,
        "attempt_id": claim.attempt_id,
        "attempt_number": claim.attempt_number,
        "worker_id": claim.worker_id,
    }


def _append_invocation(journal_path: Path, claim: Any) -> dict[str, Any]:
    """Append and fsync the external invocation boundary before finishing."""
    record = _claim_record(claim)
    encoded = (canonical_json(record) + "\n").encode("utf-8")
    fd = os.open(journal_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("invocation journal write made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return record


def _handshake(phase: str, **payload: Any) -> None:
    print(canonical_json({"phase": phase, **payload}), flush=True)


def _wait_for_sigkill() -> None:
    # The parent owns the only intended exit from these modes.  Avoid sleeps so
    # the test cannot accidentally advance past the selected crash boundary.
    while True:
        signal.pause()


def _open_services(database: Path) -> tuple[GraphStore, TaskService]:
    graph = GraphStore(database)
    return graph, TaskService(graph)


def _child_a(database: Path, journal: Path, batch_id: str,
             *, finish_before_handshake: bool) -> int:
    _, tasks = _open_services(database)
    claim = tasks.claim_next_step(batch_id, "child-a", lease_seconds=3600)
    if claim is None:
        raise RuntimeError("child A could not claim the first durable step")
    record = _append_invocation(journal, claim)
    if finish_before_handshake:
        tasks.finish_step(
            claim,
            {"status": "ok", "invoked": claim.tool_call_id},
            succeeded=True,
        )
        _handshake("step1-finish-committed", claim=record)
    else:
        _handshake("step1-invoked-before-finish", claim=record)
    _wait_for_sigkill()
    return 99  # pragma: no cover - SIGKILL is the expected exit.


def _child_b(database: Path, journal: Path, batch_id: str) -> int:
    _, tasks = _open_services(database)
    recovered = tasks.recover_inflight_steps(
        force=True,
        dead_worker_id="child-a",
        actor="restart-test",
    )
    claims: list[dict[str, Any]] = []
    while True:
        claim = tasks.claim_next_step(
            batch_id, "child-b", lease_seconds=3600,
            actor="restart-test",
        )
        if claim is None:
            break
        claims.append(_append_invocation(journal, claim))
        tasks.finish_step(
            claim,
            {"status": "ok", "invoked": claim.tool_call_id},
            succeeded=True,
            actor="restart-test",
        )
    batch = tasks.step_batch(batch_id)
    _handshake(
        "restart-complete",
        recovered=recovered,
        claims=claims,
        batch_status=batch["status"] if batch else None,
    )
    return 0


def _run_child_mode(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("child-a-running", "child-a-finished", "child-b"))
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args(argv)
    if args.mode == "child-a-running":
        return _child_a(
            args.database, args.journal, args.batch_id,
            finish_before_handshake=False,
        )
    if args.mode == "child-a-finished":
        return _child_a(
            args.database, args.journal, args.batch_id,
            finish_before_handshake=True,
        )
    return _child_b(args.database, args.journal, args.batch_id)


class DurableStepSigkillRestartTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        if not VENV_PYTHON.is_file():
            self.fail(f"project virtualenv Python is missing: {VENV_PYTHON}")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.database = self.root / "restart.db"
        self.journal = self.root / "invocations.jsonl"
        self.graph = GraphStore(self.database)
        self.tasks = TaskService(self.graph)
        self.task_id, _ = self.tasks.create(
            "Prove exact durable restart recovery",
            {"version": 0, "evidence": "process-level SIGKILL drill"},
        )
        self.tasks.transition(self.task_id, "interpreting")
        self.tasks.set_plan(self.task_id, ["Inspect alpha", "Inspect beta"])
        self.tasks.transition(self.task_id, "planned")
        self.tasks.transition(self.task_id, "running")
        self.expected_args = (
            {"path": "/virtual/alpha", "include_hidden": True},
            {"path": "/virtual/beta", "include_hidden": False},
        )
        self.batch_id, self.staged_steps = self.tasks.stage_step_batch(
            self.task_id,
            [
                {
                    "tool_call_id": "read-alpha",
                    "tool_name": "list_files",
                    "args": self.expected_args[0],
                    "risk": "read_only",
                    "idempotency_class": "read_only",
                },
                {
                    "tool_call_id": "read-beta",
                    "tool_name": "list_files",
                    "args": self.expected_args[1],
                    "risk": "read_only",
                    "idempotency_class": "read_only",
                },
            ],
            round_index=0,
            context={"session_id": "restart-test-session",
                     "turn_id": "restart-test-turn"},
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _command(self, mode: str) -> list[str]:
        return [
            str(VENV_PYTHON), str(Path(__file__).resolve()),
            "--crash-child", mode,
            "--database", str(self.database),
            "--journal", str(self.journal),
            "--batch-id", self.batch_id,
        ]

    def _spawn_child_a(self, mode: str, expected_phase: str) -> dict[str, Any]:
        process = subprocess.Popen(
            self._command(mode),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.assertIsNotNone(process.stdout)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + 15
        line = ""
        try:
            while time.monotonic() < deadline:
                ready = selector.select(max(0.0, deadline - time.monotonic()))
                if not ready:
                    break
                line = process.stdout.readline()
                if line:
                    break
                if process.poll() is not None:
                    break
        finally:
            selector.close()

        if not line:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
            self.fail(
                "child A exited or stalled before its crash handshake\n"
                f"returncode={process.returncode}\nstdout={stdout}\nstderr={stderr}")
        try:
            handshake = json.loads(line)
        except json.JSONDecodeError:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
            self.fail(
                f"invalid child A handshake: {line!r}\n"
                f"stdout={stdout}\nstderr={stderr}")
        if handshake.get("phase") != expected_phase:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
            self.fail(
                f"unexpected child A phase: {handshake!r}\n"
                f"stdout={stdout}\nstderr={stderr}")

        os.kill(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(
            process.returncode, -signal.SIGKILL,
            f"child A was not terminated by SIGKILL; stdout={stdout!r} "
            f"stderr={stderr!r}",
        )
        return handshake

    def _run_child_b(self) -> dict[str, Any]:
        completed = subprocess.run(
            self._command("child-b"),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(
            completed.returncode, 0,
            f"restart child failed\nstdout={completed.stdout}\n"
            f"stderr={completed.stderr}",
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1, completed.stdout)
        result = json.loads(lines[0])
        self.assertEqual(result["phase"], "restart-complete")
        self.assertEqual(result["batch_status"], "succeeded")
        return result

    def _journal_records(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.journal.read_text().splitlines()
                if line.strip()]

    def _events(self) -> list[dict[str, Any]]:
        return self.graph.events_since(task_id=self.task_id, limit=1000)

    def _assert_one_finish_pair_per_step(self) -> None:
        events = self._events()
        for step in self.staged_steps:
            step_id = step["step_id"]
            action_finishes = [
                event for event in events
                if event["event_type"] == "action.finished"
                and event["payload"].get("step_id") == step_id
            ]
            step_finishes = [
                event for event in events
                if event["event_type"] == "step.succeeded"
                and event["payload"].get("step_id") == step_id
            ]
            self.assertEqual(len(action_finishes), 1, step_id)
            self.assertEqual(len(step_finishes), 1, step_id)
            self.assertLess(action_finishes[0]["seq"], step_finishes[0]["seq"])

    def _assert_final_database_state(self, expected_attempts: tuple[int, int]) -> None:
        batch = self.tasks.step_batch(self.batch_id)
        self.assertIsNotNone(batch)
        self.assertEqual(batch["status"], "succeeded")
        self.assertEqual([step["status"] for step in batch["steps"]],
                         ["succeeded", "succeeded"])
        self.assertEqual(
            tuple(step["attempt_count"] for step in batch["steps"]),
            expected_attempts,
        )
        # A finished tool batch is handed back to the conversation continuation;
        # it has no active durable step but does not itself finalize the task.
        task = self.tasks.get(self.task_id)
        self.assertEqual(task["status"], "running")
        self.assertIsNone(task["active_step"])
        with self.graph._connect() as conn:
            receipt_rows = conn.execute(
                "SELECT step_id,status FROM action_receipts "
                "WHERE task_id=? ORDER BY created_at", (self.task_id,)
            ).fetchall()
            self.assertEqual(len(receipt_rows), 2)
            self.assertTrue(all(row["status"] == "succeeded"
                                for row in receipt_rows))
            self.assertEqual(
                [row[0] for row in conn.execute("PRAGMA integrity_check")],
                ["ok"],
            )
            self.assertEqual(list(conn.execute("PRAGMA foreign_key_check")), [])
        self._assert_one_finish_pair_per_step()

    def test_sigkill_running_step_retries_exact_step_then_successor(self) -> None:
        first = self._spawn_child_a(
            "child-a-running", "step1-invoked-before-finish")
        restarted = self._run_child_b()

        step_ids = [step["step_id"] for step in self.staged_steps]
        self.assertEqual(restarted["recovered"],
                         {"retry": [step_ids[0]], "reconcile": []})
        self.assertEqual([claim["step_id"] for claim in restarted["claims"]],
                         step_ids)
        records = self._journal_records()
        self.assertEqual([record["step_id"] for record in records],
                         [step_ids[0], step_ids[0], step_ids[1]])
        self.assertEqual([record["attempt_number"] for record in records],
                         [1, 2, 1])
        self.assertEqual([record["args"] for record in records],
                         [self.expected_args[0], self.expected_args[0],
                          self.expected_args[1]])
        self.assertEqual(records[0]["action_id"], records[1]["action_id"])
        self.assertEqual(records[0]["idempotency_key"],
                         records[1]["idempotency_key"])
        self.assertEqual(records[0]["tool_call_id"],
                         records[1]["tool_call_id"])
        self.assertNotEqual(records[0]["attempt_id"], records[1]["attempt_id"])
        self.assertEqual(first["claim"], records[0])
        self.assertEqual(restarted["claims"], records[1:])

        with self.graph._connect() as conn:
            attempts = [dict(row) for row in conn.execute(
                "SELECT step_id,attempt_number,status FROM action_attempts "
                "ORDER BY step_id,attempt_number")]
        first_attempts = [row for row in attempts if row["step_id"] == step_ids[0]]
        second_attempts = [row for row in attempts if row["step_id"] == step_ids[1]]
        self.assertEqual(
            [(row["attempt_number"], row["status"]) for row in first_attempts],
            [(1, "abandoned"), (2, "succeeded")],
        )
        self.assertEqual(
            [(row["attempt_number"], row["status"]) for row in second_attempts],
            [(1, "succeeded")],
        )
        self._assert_final_database_state((2, 1))

    def test_sigkill_after_finish_commit_does_not_reinvoke_completed_step(self) -> None:
        first = self._spawn_child_a(
            "child-a-finished", "step1-finish-committed")
        restarted = self._run_child_b()

        step_ids = [step["step_id"] for step in self.staged_steps]
        self.assertEqual(restarted["recovered"],
                         {"retry": [], "reconcile": []})
        self.assertEqual([claim["step_id"] for claim in restarted["claims"]],
                         [step_ids[1]])
        records = self._journal_records()
        self.assertEqual([record["step_id"] for record in records], step_ids)
        self.assertEqual([record["attempt_number"] for record in records], [1, 1])
        self.assertEqual([record["args"] for record in records],
                         list(self.expected_args))
        self.assertEqual(first["claim"], records[0])
        self.assertEqual(restarted["claims"], records[1:])
        self._assert_final_database_state((1, 1))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--crash-child":
        raise SystemExit(_run_child_mode(sys.argv[2:]))
    unittest.main()
