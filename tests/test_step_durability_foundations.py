import base64
import json
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path

from friday_core.db_migrations import (
    LATEST_SCHEMA_VERSION,
    MIGRATIONS,
    apply_schema_migrations,
)
from friday_core.graph import GraphStore, canonical_json, sha256_text, utc_now
from friday_core.step_payloads import StepPayloadCipher


class StepPayloadCipherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.key_path = self.root / "state" / "step-payload.key"
        self.cipher = StepPayloadCipher(self.key_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_seal_round_trip_is_context_bound_and_contains_no_plaintext(self):
        arguments = {
            "text": "ultra-private clipboard phrase 827364",
            "options": {"replace": True, "retries": 2},
        }
        payload = self.cipher.seal(arguments, context="task-1:step-1")

        self.assertEqual(
            StepPayloadCipher(self.key_path).open(
                payload, context="task-1:step-1"),
            arguments,
        )
        self.assertNotIn(arguments["text"], payload)
        with self.assertRaisesRegex(RuntimeError, "authentication failed"):
            self.cipher.open(payload, context="task-1:step-2")

    def test_ciphertext_tampering_is_rejected(self):
        payload = self.cipher.seal(
            {"text": "do not execute a modified step"},
            context="task-2:step-1",
        )
        body = json.loads(payload)
        ciphertext = bytearray(base64.b64decode(body["ciphertext"]))
        ciphertext[0] ^= 0x01
        body["ciphertext"] = base64.b64encode(ciphertext).decode("ascii")

        with self.assertRaisesRegex(RuntimeError, "authentication failed"):
            self.cipher.open(
                canonical_json(body), context="task-2:step-1")

    def test_key_file_is_created_private_and_rehardened_when_loaded(self):
        payload = self.cipher.seal({"value": 1}, context="task-3:step-1")
        self.assertEqual(stat.S_IMODE(self.key_path.stat().st_mode), 0o600)

        self.key_path.chmod(0o644)
        reopened = StepPayloadCipher(self.key_path)
        self.assertEqual(
            reopened.open(payload, context="task-3:step-1"), {"value": 1})
        self.assertEqual(stat.S_IMODE(self.key_path.stat().st_mode), 0o600)


class DurableStepSchemaTests(unittest.TestCase):
    EXPECTED_STEP_COLUMNS = {
        "step_id",
        "task_id",
        "batch_id",
        "round_index",
        "ordinal",
        "tool_call_id",
        "tool_name",
        "args_ciphertext",
        "args_redacted_json",
        "args_sha256",
        "context_json",
        "idempotency_key",
        "idempotency_class",
        "recovery_policy",
        "status",
        "depends_on_json",
        "verifier",
        "risk",
        "approval_status",
        "approval_id",
        "action_id",
        "lease_id",
        "lease_expires_at",
        "attempt_count",
        "max_attempts",
        "last_error",
        "created_at",
        "updated_at",
        "last_event_seq",
    }

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}

    def test_fresh_database_has_v2_durable_step_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = GraphStore(Path(tmp) / "friday.db")

            self.assertEqual(graph.schema_version(), LATEST_SCHEMA_VERSION)
            with graph._connect() as conn:
                versions = [int(row[0]) for row in conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version")]
                step_columns = self._columns(conn, "task_steps")
                receipt_columns = self._columns(conn, "action_receipts")
                approval_columns = self._columns(conn, "approval_state")
                lease_columns = {
                    str(row[1]): row
                    for row in conn.execute(
                        "PRAGMA table_info(resource_leases)")
                }
                indexes = {str(row[1]) for row in conn.execute(
                    "PRAGMA index_list(task_steps)")}

            self.assertEqual(versions, list(range(1, LATEST_SCHEMA_VERSION + 1)))
            self.assertTrue(self.EXPECTED_STEP_COLUMNS.issubset(step_columns))
            self.assertNotIn("args_json", step_columns)
            self.assertIn("step_id", receipt_columns)
            self.assertIn("step_id", approval_columns)
            self.assertIn("task_steps_task_status", indexes)
            self.assertIn("task_steps_lease", indexes)
            for name, default in (
                ("profile_fingerprint", "'legacy'"),
                ("latency_class", "'interactive'"),
            ):
                self.assertIn(name, lease_columns)
                self.assertEqual(int(lease_columns[name][3]), 1)
                self.assertEqual(str(lease_columns[name][4]), default)

    def test_existing_v6_resource_lease_upgrades_to_v7_without_data_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "friday-v6.db"
            schema_path = (
                Path(__file__).parents[1] / "friday_core" / "schema.sql"
            )
            with sqlite3.connect(path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.executescript(schema_path.read_text())
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS schema_migrations (
                           version INTEGER PRIMARY KEY,
                           applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                       )"""
                )
                for version, migration in MIGRATIONS:
                    if version > 6:
                        break
                    migration(conn)
                    conn.execute(
                        "INSERT INTO schema_migrations(version) VALUES (?)",
                        (version,),
                    )
                conn.execute("PRAGMA user_version = 6")
                cursor = conn.execute(
                    """INSERT INTO graph_events
                       (event_id, occurred_at, actor, event_type, payload_json,
                        payload_sha256)
                       VALUES ('event_existing_lease',
                               '2026-08-24T08:00:00Z',
                               'resource_admission', 'resource.lease_acquired',
                               '{}', 'hash_existing_event')"""
                )
                event_seq = int(cursor.lastrowid)
                conn.execute(
                    """INSERT INTO nodes
                       (node_id, kind, created_event_id, body_json, body_sha256)
                       VALUES ('resource_lease_existing', 'resource_lease',
                               'event_existing_lease', '{}',
                               'hash_existing_node')"""
                )
                conn.execute(
                    """INSERT INTO resource_leases
                       (lease_id, step_id, attempt_id, worker_id, runtime_id,
                        cpu_millis, ram_mib, concurrency_slots, network_slots,
                        accelerator, vram_mib, status, acquired_at,
                        heartbeat_at, expires_at, last_event_seq)
                       VALUES ('resource_lease_existing', 'step_existing',
                               'attempt_existing', 'worker_existing',
                               'runtime_existing', 750, 1536, 1, 1,
                               'cuda:0', 2048, 'active',
                               '2026-08-24T08:00:00Z',
                               '2026-08-24T08:00:01Z',
                               '2026-08-24T08:01:01Z', ?)""",
                    (event_seq,),
                )
                conn.commit()

                self.assertEqual(
                    apply_schema_migrations(conn), LATEST_SCHEMA_VERSION
                )
                self.assertEqual(
                    apply_schema_migrations(conn), LATEST_SCHEMA_VERSION
                )
                conn.commit()

                lease_columns = {
                    str(row[1]): row
                    for row in conn.execute(
                        "PRAGMA table_info(resource_leases)")
                }
                row = conn.execute(
                    """SELECT lease_id,step_id,attempt_id,worker_id,runtime_id,
                              cpu_millis,ram_mib,concurrency_slots,network_slots,
                              accelerator,vram_mib,status,profile_fingerprint,
                              latency_class
                       FROM resource_leases
                       WHERE lease_id='resource_lease_existing'"""
                ).fetchone()
                versions = [
                    int(item[0])
                    for item in conn.execute(
                        "SELECT version FROM schema_migrations ORDER BY version"
                    )
                ]
                user_version = int(
                    conn.execute("PRAGMA user_version").fetchone()[0]
                )

            self.assertEqual(user_version, LATEST_SCHEMA_VERSION)
            self.assertEqual(
                versions, list(range(1, LATEST_SCHEMA_VERSION + 1))
            )
            for name, default in (
                ("profile_fingerprint", "'legacy'"),
                ("latency_class", "'interactive'"),
            ):
                self.assertEqual(int(lease_columns[name][3]), 1)
                self.assertEqual(str(lease_columns[name][4]), default)
            self.assertEqual(
                row,
                (
                    "resource_lease_existing",
                    "step_existing",
                    "attempt_existing",
                    "worker_existing",
                    "runtime_existing",
                    750,
                    1536,
                    1,
                    1,
                    "cuda:0",
                    2048,
                    "active",
                    "legacy",
                    "interactive",
                ),
            )

    def test_encrypted_arguments_leave_no_plaintext_in_task_step_dump(self):
        secret = "ultra-private clipboard phrase 827364"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = GraphStore(root / "friday.db")
            cipher = StepPayloadCipher(root / "step-payload.key")
            task_id = "task_privacy"
            step_id = "step_privacy"
            context = f"{task_id}:{step_id}"
            arguments = {"text": secret, "replace": True}
            encrypted = cipher.seal(arguments, context=context)
            now = utc_now()

            with graph.transaction() as conn:
                task_event, task_seq = graph.append_event(
                    conn, "task.created", {"objective": "private operation"},
                    task_id=task_id,
                )
                graph.append_node(
                    conn, "task", {"objective": "private operation"},
                    event_id=task_event, node_id=task_id,
                )
                conn.execute(
                    """INSERT INTO task_state
                       (task_id, objective, completion_contract_json, status,
                        plan_json, created_at, updated_at, last_event_seq)
                       VALUES (?, ?, '{}', 'running', '[]', ?, ?, ?)""",
                    (task_id, "private operation", now, now, task_seq),
                )
                step_event, step_seq = graph.append_event(
                    conn, "task.step.recorded",
                    {"step_id": step_id, "args": {"text": "[private]"}},
                    task_id=task_id,
                )
                graph.append_node(
                    conn, "task_step",
                    {"step_id": step_id, "args": {"text": "[private]"}},
                    event_id=step_event, node_id=step_id,
                )
                conn.execute(
                    """INSERT INTO task_steps
                       (step_id, task_id, batch_id, round_index, ordinal,
                        tool_call_id, tool_name, args_ciphertext,
                        args_redacted_json, args_sha256, context_json,
                        idempotency_key, idempotency_class, recovery_policy,
                        status, depends_on_json, verifier, risk,
                        approval_status, created_at, updated_at, last_event_seq)
                       VALUES (?, ?, ?, 0, 0, ?, ?, ?, ?, ?, '{}', ?, ?, ?,
                               'pending', '[]', ?, ?, 'not_required', ?, ?, ?)""",
                    (
                        step_id,
                        task_id,
                        "batch_privacy",
                        "call_privacy",
                        "clipboard_write",
                        encrypted,
                        canonical_json({"text": "[private]", "replace": True}),
                        sha256_text(canonical_json(arguments)),
                        "idem_privacy",
                        "consequential",
                        "reconcile",
                        "clipboard_contains_expected_text",
                        "high",
                        now,
                        now,
                        step_seq,
                    ),
                )

            with sqlite3.connect(graph.path) as conn:
                stored = conn.execute(
                    "SELECT args_ciphertext, args_redacted_json "
                    "FROM task_steps WHERE step_id=?", (step_id,)).fetchone()
                dump = "\n".join(conn.iterdump())

            self.assertEqual(stored[0], encrypted)
            self.assertNotIn(secret, stored[0])
            self.assertNotIn(secret, stored[1])
            self.assertNotIn(secret, dump)
            self.assertEqual(cipher.open(stored[0], context=context), arguments)

    def test_existing_v1_database_upgrades_without_losing_receipts_or_approvals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "friday-v1.db"
            with sqlite3.connect(path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE graph_events (seq INTEGER PRIMARY KEY);
                    CREATE TABLE nodes (node_id TEXT PRIMARY KEY);
                    CREATE TABLE task_state (
                        task_id TEXT PRIMARY KEY REFERENCES nodes(node_id)
                    );
                    CREATE TABLE action_receipts (
                        idempotency_key TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL REFERENCES task_state(task_id),
                        action_id TEXT NOT NULL REFERENCES nodes(node_id),
                        tool_name TEXT NOT NULL,
                        args_sha256 TEXT NOT NULL,
                        status TEXT NOT NULL,
                        observation_id TEXT REFERENCES nodes(node_id),
                        result_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        effects_json TEXT NOT NULL DEFAULT '[]',
                        verification_json TEXT,
                        risk TEXT NOT NULL DEFAULT 'low',
                        approval_status TEXT NOT NULL DEFAULT 'not_required'
                    );
                    CREATE TABLE approval_state (
                        approval_id TEXT PRIMARY KEY REFERENCES nodes(node_id),
                        task_id TEXT NOT NULL REFERENCES task_state(task_id),
                        tool_name TEXT NOT NULL,
                        args_json TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        decided_at TEXT,
                        last_event_seq INTEGER NOT NULL REFERENCES graph_events(seq)
                    );
                    CREATE TABLE schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    INSERT INTO schema_migrations(version) VALUES (1);
                    PRAGMA user_version = 1;

                    INSERT INTO graph_events(seq) VALUES (1);
                    INSERT INTO nodes(node_id) VALUES
                        ('task_existing'), ('action_existing'), ('approval_existing');
                    INSERT INTO task_state(task_id) VALUES ('task_existing');
                    INSERT INTO action_receipts
                        (idempotency_key, task_id, action_id, tool_name,
                         args_sha256, status, result_json, created_at, updated_at)
                    VALUES
                        ('idem_existing', 'task_existing', 'action_existing',
                         'read_file', 'hash_existing', 'succeeded', '{}',
                         '2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z');
                    INSERT INTO approval_state
                        (approval_id, task_id, tool_name, args_json, reason,
                         status, created_at, last_event_seq)
                    VALUES
                        ('approval_existing', 'task_existing', 'write_file',
                         '{}', 'test', 'pending',
                         '2026-01-01T00:00:00Z', 1);
                    """
                )

                self.assertEqual(
                    apply_schema_migrations(conn), LATEST_SCHEMA_VERSION)
                self.assertEqual(
                    apply_schema_migrations(conn), LATEST_SCHEMA_VERSION)
                conn.commit()

                versions = [int(row[0]) for row in conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version")]
                receipt = conn.execute(
                    "SELECT status, step_id FROM action_receipts "
                    "WHERE idempotency_key='idem_existing'").fetchone()
                approval = conn.execute(
                    "SELECT status, step_id FROM approval_state "
                    "WHERE approval_id='approval_existing'").fetchone()
                step_columns = self._columns(conn, "task_steps")
                user_version = int(conn.execute(
                    "PRAGMA user_version").fetchone()[0])

            self.assertEqual(versions, list(range(1, LATEST_SCHEMA_VERSION + 1)))
            self.assertEqual(user_version, LATEST_SCHEMA_VERSION)
            self.assertEqual(receipt, ("succeeded", None))
            self.assertEqual(approval, ("pending", None))
            self.assertTrue(self.EXPECTED_STEP_COLUMNS.issubset(step_columns))


if __name__ == "__main__":
    unittest.main()
