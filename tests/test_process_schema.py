import sqlite3
import stat
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import friday_core.db_migrations as db_migrations
from friday_core.db_migrations import (
    LATEST_SCHEMA_VERSION,
    MIGRATIONS,
    apply_schema_migrations,
)
from friday_core.graph import GraphStore


SPEC_COLUMNS = {
    "spec_id",
    "name",
    "version",
    "spec_ciphertext",
    "display_json",
    "spec_sha256",
    "sandbox_fingerprint",
    "status",
    "source_task_id",
    "created_at",
    "updated_at",
    "last_event_seq",
}

INSTANCE_COLUMNS = {
    "instance_id",
    "spec_id",
    "task_id",
    "step_id",
    "action_id",
    "launch_idempotency_key",
    "args_ciphertext",
    "args_redacted_json",
    "args_sha256",
    "spec_fingerprint",
    "sandbox_fingerprint",
    "state",
    "unit_name",
    "boot_id",
    "invocation_id",
    "control_group",
    "leader_pid",
    "start_ticks",
    "exe_device",
    "exe_inode",
    "exe_sha256",
    "persistent",
    "exit_code",
    "exit_signal",
    "result_code",
    "stdout_bytes",
    "stdout_sha256",
    "stdout_truncated",
    "stderr_bytes",
    "stderr_sha256",
    "stderr_truncated",
    "prepared_at",
    "started_at",
    "heartbeat_at",
    "stop_requested_at",
    "finished_at",
    "created_at",
    "updated_at",
    "last_event_seq",
}

WORKLOAD_LEASE_COLUMNS = {
    "lease_id",
    "instance_id",
    "source_step_lease_id",
    "source_attempt_id",
    "source_worker_id",
    "runtime_id",
    "profile_fingerprint",
    "latency_class",
    "cpu_millis",
    "ram_mib",
    "concurrency_slots",
    "network_slots",
    "accelerator",
    "vram_mib",
    "enforcement_json",
    "status",
    "acquired_at",
    "heartbeat_at",
    "expires_at",
    "reconcile_started_at",
    "released_at",
    "release_reason",
    "last_event_seq",
}

CLEANUP_COLUMNS = {
    "instance_id",
    "state",
    "attempt_count",
    "requested_at",
    "last_attempt_at",
    "next_attempt_at",
    "claim_token",
    "claim_expires_at",
    "completed_at",
    "last_error_code",
    "last_event_seq",
}

OPERATION_COLUMNS = {
    "operation_id", "idempotency_key", "task_id", "step_id", "action_id",
    "attempt_id", "attempt_number", "step_lease_id", "worker_id",
    "tool_name", "args_sha256", "instance_id",
    "executor_binding_sha256", "executor_binding_json",
    "target_boundary_sha256", "force", "status", "prepared_event_seq",
    "dispatch_event_seq", "outcome_event_seq", "postcondition_event_seq",
    "error_code", "created_at", "updated_at", "completed_at",
    "last_event_seq",
}


class ProcessSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "friday.db"
        self.graph = GraphStore(self.path)
        self._node_number = 0

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {str(row[1]) for row in conn.execute(
            f"PRAGMA table_info({table})")}

    @staticmethod
    def _foreign_keys(
        conn: sqlite3.Connection, table: str,
    ) -> set[tuple[str, str, str]]:
        return {
            (str(row[3]), str(row[2]), str(row[4]))
            for row in conn.execute(f"PRAGMA foreign_key_list({table})")
        }

    def _append_node(
        self, conn: sqlite3.Connection, node_id: str, kind: str,
    ) -> int:
        self._node_number += 1
        event_id, seq = self.graph.append_event(
            conn,
            f"{kind}.test_recorded",
            {"node_id": node_id, "number": self._node_number},
            idempotency_key=f"process-schema-event-{self._node_number}",
        )
        self.graph.append_node(
            conn, kind, {"node_id": node_id}, event_id=event_id,
            node_id=node_id,
        )
        return seq

    def _insert_spec(
        self, conn: sqlite3.Connection, spec_id: str, **overrides,
    ) -> None:
        seq = self._append_node(conn, spec_id, "process_spec")
        values = {
            "spec_id": spec_id,
            "name": spec_id,
            "version": 1,
            "spec_ciphertext": "sealed-spec-envelope",
            "display_json": '{"executable":"[approved executable]"}',
            "spec_sha256": "a" * 64,
            "sandbox_fingerprint": "b" * 64,
            "status": "active",
            "source_task_id": None,
            "created_at": "2026-08-24T08:00:00Z",
            "updated_at": "2026-08-24T08:00:00Z",
            "last_event_seq": seq,
        }
        values.update(overrides)
        conn.execute(
            f"INSERT INTO process_specs ({','.join(values)}) "
            f"VALUES ({','.join('?' for _ in values)})",
            tuple(values.values()),
        )

    def _insert_instance(
        self,
        conn: sqlite3.Connection,
        instance_id: str,
        *,
        spec_id: str = "spec_main",
        **overrides,
    ) -> None:
        seq = self._append_node(conn, instance_id, "process_instance")
        values = {
            "instance_id": instance_id,
            "spec_id": spec_id,
            "task_id": None,
            "step_id": None,
            "action_id": None,
            "launch_idempotency_key": f"launch-{instance_id}",
            "args_ciphertext": "sealed-args-envelope",
            "args_redacted_json": '{"argv":["[approved argument]"]}',
            "args_sha256": "c" * 64,
            "spec_fingerprint": "d" * 64,
            "sandbox_fingerprint": "b" * 64,
            "state": "prepared",
            "unit_name": f"friday-{instance_id}.service",
            "boot_id": None,
            "invocation_id": None,
            "control_group": None,
            "leader_pid": None,
            "start_ticks": None,
            "exe_device": None,
            "exe_inode": None,
            "exe_sha256": None,
            "persistent": 0,
            "exit_code": None,
            "exit_signal": None,
            "result_code": None,
            "stdout_bytes": 0,
            "stdout_sha256": None,
            "stdout_truncated": 0,
            "stderr_bytes": 0,
            "stderr_sha256": None,
            "stderr_truncated": 0,
            "prepared_at": "2026-08-24T08:01:00Z",
            "started_at": None,
            "heartbeat_at": None,
            "stop_requested_at": None,
            "finished_at": None,
            "created_at": "2026-08-24T08:01:00Z",
            "updated_at": "2026-08-24T08:01:00Z",
            "last_event_seq": seq,
        }
        values.update(overrides)
        conn.execute(
            f"INSERT INTO process_instances ({','.join(values)}) "
            f"VALUES ({','.join('?' for _ in values)})",
            tuple(values.values()),
        )

    def _insert_workload_lease(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        instance_id: str,
        **overrides,
    ) -> None:
        seq = self._append_node(conn, lease_id, "workload_resource_lease")
        values = {
            "lease_id": lease_id,
            "instance_id": instance_id,
            "source_step_lease_id": None,
            "source_attempt_id": None,
            "source_worker_id": None,
            "runtime_id": "runtime-process-schema",
            "profile_fingerprint": "e" * 64,
            "latency_class": "interactive",
            "cpu_millis": 1000,
            "ram_mib": 512,
            "concurrency_slots": 1,
            "network_slots": 0,
            "accelerator": "none",
            "vram_mib": 0,
            "enforcement_json": '{"cgroup":"enforced"}',
            "status": "active",
            "acquired_at": "2026-08-24T08:01:01Z",
            "heartbeat_at": "2026-08-24T08:01:01Z",
            "expires_at": "2026-08-24T08:02:01Z",
            "reconcile_started_at": None,
            "released_at": None,
            "release_reason": None,
            "last_event_seq": seq,
        }
        values.update(overrides)
        conn.execute(
            f"INSERT INTO workload_resource_leases ({','.join(values)}) "
            f"VALUES ({','.join('?' for _ in values)})",
            tuple(values.values()),
        )

    def test_fresh_database_has_v14_contract_indexes_fks_and_safe_modes(self):
        self.assertEqual(self.graph.schema_version(), 14)
        self.assertEqual(LATEST_SCHEMA_VERSION, 14)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

        with self.graph._connect() as conn:
            versions = [int(row[0]) for row in conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version")]
            modes = (
                str(conn.execute("PRAGMA journal_mode").fetchone()[0]),
                int(conn.execute("PRAGMA foreign_keys").fetchone()[0]),
            )
            self.assertEqual(self._columns(conn, "process_specs"), SPEC_COLUMNS)
            self.assertEqual(
                self._columns(conn, "process_instances"), INSTANCE_COLUMNS)
            self.assertEqual(
                self._columns(conn, "workload_resource_leases"),
                WORKLOAD_LEASE_COLUMNS,
            )
            self.assertEqual(
                self._columns(conn, "process_unit_cleanups"), CLEANUP_COLUMNS)
            self.assertEqual(
                self._columns(conn, "process_operations"), OPERATION_COLUMNS)
            self.assertEqual(
                self._foreign_keys(conn, "process_specs"),
                {
                    ("spec_id", "nodes", "node_id"),
                    ("source_task_id", "task_state", "task_id"),
                    ("last_event_seq", "graph_events", "seq"),
                },
            )
            instance_fks = self._foreign_keys(conn, "process_instances")
            self.assertEqual(
                instance_fks,
                {
                    ("instance_id", "nodes", "node_id"),
                    ("spec_id", "process_specs", "spec_id"),
                    ("task_id", "task_state", "task_id"),
                    ("step_id", "task_steps", "step_id"),
                    ("action_id", "nodes", "node_id"),
                    ("last_event_seq", "graph_events", "seq"),
                },
            )
            self.assertNotIn(
                "workload_resource_leases",
                {referenced for _, referenced, _ in instance_fks},
            )
            self.assertEqual(
                self._foreign_keys(conn, "workload_resource_leases"),
                {
                    ("lease_id", "nodes", "node_id"),
                    ("instance_id", "process_instances", "instance_id"),
                    ("source_step_lease_id", "resource_leases", "lease_id"),
                    ("last_event_seq", "graph_events", "seq"),
                },
            )
            self.assertEqual(
                self._foreign_keys(conn, "process_unit_cleanups"),
                {
                    ("instance_id", "process_instances", "instance_id"),
                    ("last_event_seq", "graph_events", "seq"),
                },
            )
            operation_fks = self._foreign_keys(conn, "process_operations")
            self.assertTrue({
                ("instance_id", "process_instances", "instance_id"),
                ("prepared_event_seq", "graph_events", "seq"),
                ("dispatch_event_seq", "graph_events", "seq"),
                ("outcome_event_seq", "graph_events", "seq"),
                ("postcondition_event_seq", "graph_events", "seq"),
                ("last_event_seq", "graph_events", "seq"),
                ("executor_binding_json", "task_steps",
                 "executor_binding_json"),
                ("idempotency_key", "action_receipts", "idempotency_key"),
                ("attempt_id", "action_attempts", "attempt_id"),
                ("step_lease_id", "action_attempts", "lease_id"),
            }.issubset(operation_fks))

            instance_indexes = {
                str(row[1]): (int(row[2]), int(row[4]))
                for row in conn.execute("PRAGMA index_list(process_instances)")
            }
            workload_indexes = {
                str(row[1]): (int(row[2]), int(row[4]))
                for row in conn.execute(
                    "PRAGMA index_list(workload_resource_leases)")
            }
            cleanup_indexes = {
                str(row[1]): (int(row[2]), int(row[4]))
                for row in conn.execute(
                    "PRAGMA index_list(process_unit_cleanups)")
            }
            operation_indexes = {
                str(row[1]): (int(row[2]), int(row[4]))
                for row in conn.execute("PRAGMA index_list(process_operations)")
            }
            identity_columns = [str(row[2]) for row in conn.execute(
                "PRAGMA index_info(process_instances_active_runtime_identity)")]

        self.assertEqual(versions, list(range(1, 15)))
        self.assertEqual(modes, ("wal", 1))
        self.assertEqual(
            instance_indexes["process_instances_active_runtime_identity"],
            (1, 1),
        )
        self.assertEqual(identity_columns, ["boot_id", "leader_pid", "start_ticks"])
        self.assertEqual(
            workload_indexes["workload_resource_leases_reserving"], (0, 1))
        self.assertIn("workload_resource_leases_profile", workload_indexes)
        self.assertEqual(
            cleanup_indexes["process_unit_cleanups_unresolved"], (0, 1))
        self.assertEqual(
            operation_indexes["process_operations_accepted_instance"], (1, 1))
        self.assertEqual(
            operation_indexes["process_operations_unresolved"], (0, 1))
        for table in (
            "process_specs", "process_instances", "workload_resource_leases",
            "process_unit_cleanups", "process_operations",
        ):
            self.assertEqual(self.graph.count(table), 0)

    def test_terminal_transition_journals_one_cleanup_intent(self):
        with self.graph.transaction() as conn:
            self._insert_spec(conn, "spec_main")
            self._insert_instance(conn, "instance_cleanup")
            row = conn.execute(
                "SELECT last_event_seq FROM process_instances "
                "WHERE instance_id='instance_cleanup'"
            ).fetchone()
            conn.execute(
                """UPDATE process_instances
                      SET state='exited',finished_at=?,updated_at=?,
                          last_event_seq=?
                    WHERE instance_id='instance_cleanup'""",
                ("2026-08-24T08:03:00Z", "2026-08-24T08:03:00Z", row[0]),
            )
            conn.execute(
                "UPDATE process_instances SET state='exited' "
                "WHERE instance_id='instance_cleanup'"
            )
            cleanup = conn.execute(
                "SELECT * FROM process_unit_cleanups "
                "WHERE instance_id='instance_cleanup'"
            ).fetchone()
        self.assertEqual(cleanup["state"], "pending")
        self.assertEqual(cleanup["attempt_count"], 0)
        self.assertEqual(cleanup["requested_at"], "2026-08-24T08:03:00Z")
        self.assertIsNone(cleanup["claim_token"])
        self.assertEqual(self.graph.count("process_unit_cleanups"), 1)

    def test_cleanup_claim_state_constraints_fail_closed(self):
        with self.graph.transaction() as conn:
            self._insert_spec(conn, "spec_main")
            self._insert_instance(
                conn, "instance_cleanup_constraints", state="exited",
                finished_at="2026-08-24T08:03:00Z")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """UPDATE process_unit_cleanups
                          SET state='cleaning',claim_token=NULL,
                              claim_expires_at=NULL
                        WHERE instance_id='instance_cleanup_constraints'"""
                )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """UPDATE process_unit_cleanups
                          SET state='complete',completed_at=NULL
                        WHERE instance_id='instance_cleanup_constraints'"""
                )
            event_seq = conn.execute(
                "SELECT last_event_seq FROM process_instances "
                "WHERE instance_id='instance_cleanup_constraints'"
            ).fetchone()[0]
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """INSERT INTO process_unit_cleanups
                           (instance_id,state,attempt_count,requested_at,
                            last_event_seq)
                       VALUES (NULL,'pending',0,?,?)""",
                    ("2026-08-24T08:04:00Z", event_seq),
                )

    def test_process_schema_has_no_raw_output_or_plaintext_argument_columns(self):
        with self.graph._connect() as conn:
            columns = self._columns(conn, "process_instances")
            sql = str(conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' "
                "AND name='process_instances'"
            ).fetchone()[0]).lower()

        for stream in ("stdout", "stderr"):
            self.assertEqual(
                {name for name in columns if name.startswith(stream)},
                {f"{stream}_bytes", f"{stream}_sha256", f"{stream}_truncated"},
            )
            for forbidden in ("text", "content", "data", "blob", "ciphertext"):
                self.assertNotIn(f"{stream}_{forbidden}", sql)
        self.assertIn("args_ciphertext", columns)
        self.assertIn("args_redacted_json", columns)
        self.assertIn("args_sha256", columns)
        self.assertNotIn("args_json", columns)
        self.assertNotIn("executable", columns)
        self.assertNotIn("cwd", columns)

    def test_checks_uniqueness_and_active_runtime_identity_fail_closed(self):
        with self.graph.transaction() as conn:
            self._insert_spec(conn, "spec_main", name="editor")

            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_spec(
                    conn, "spec_bad_version", name="bad-version", version=0)
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_spec(
                    conn, "spec_bad_status", name="bad-status", status="draft")
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_spec(
                    conn, "spec_bad_json", name="bad-json", display_json="secret")
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_spec(conn, "spec_duplicate", name="editor")

            self._insert_instance(conn, "instance_main")
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_instance(
                    conn, "instance_bad_state", state="unknown")
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_instance(
                    conn, "instance_bad_bool", persistent=2)
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_instance(
                    conn, "instance_partial_identity",
                    boot_id="boot-a", leader_pid=1234)
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_instance(
                    conn, "instance_terminal_without_time", state="exited")
            self._insert_instance(
                conn,
                "instance_unresolved_identity",
                state="identity_mismatch",
            )
            unresolved = conn.execute(
                "SELECT state,finished_at FROM process_instances "
                "WHERE instance_id='instance_unresolved_identity'"
            ).fetchone()
            self.assertEqual(tuple(unresolved), ("identity_mismatch", None))
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_instance(
                    conn,
                    "instance_duplicate_launch",
                    launch_idempotency_key="launch-instance_main",
                )

            identity = {
                "state": "running",
                "boot_id": "boot-identity",
                "invocation_id": "invocation-identity",
                "control_group": "/friday/identity",
                "leader_pid": 4242,
                "start_ticks": 99001,
                "exe_device": 8,
                "exe_inode": 88001,
                "exe_sha256": "9" * 64,
            }
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_instance(
                    conn,
                    "instance_running_partial",
                    state="running",
                    boot_id="boot-partial",
                    leader_pid=4343,
                    start_ticks=99101,
                )
            self._insert_instance(conn, "instance_identity_owner", **identity)
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_instance(conn, "instance_identity_race", **identity)
            historical_identity = dict(identity)
            historical_identity["state"] = "exited"
            self._insert_instance(
                conn,
                "instance_identity_history",
                **historical_identity,
                finished_at="2026-08-24T08:02:00Z",
            )

            self._insert_workload_lease(
                conn, "workload_main", "instance_main")
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_workload_lease(
                    conn, "workload_duplicate_instance", "instance_main")

            self._insert_instance(conn, "instance_bad_resource")
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_workload_lease(
                    conn,
                    "workload_bad_resource",
                    "instance_bad_resource",
                    cpu_millis=-1,
                )
            self._insert_instance(conn, "instance_bad_lease_state")
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_workload_lease(
                    conn,
                    "workload_bad_state",
                    "instance_bad_lease_state",
                    status="expired",
                )
            self._insert_instance(conn, "instance_bad_reconcile")
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_workload_lease(
                    conn,
                    "workload_bad_reconcile",
                    "instance_bad_reconcile",
                    status="reconciling",
                )
            self._insert_instance(conn, "instance_bad_release")
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_workload_lease(
                    conn,
                    "workload_bad_release",
                    "instance_bad_release",
                    status="released",
                )
            self._insert_instance(conn, "instance_bad_provenance")
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_workload_lease(
                    conn,
                    "workload_bad_provenance",
                    "instance_bad_provenance",
                    source_attempt_id="attempt-without-lease",
                    source_worker_id="worker-without-lease",
                )

    def test_foreign_keys_and_atomic_transfer_provenance_are_enforced(self):
        with self.graph.transaction() as conn:
            self._insert_spec(conn, "spec_main")
            self._insert_instance(conn, "instance_transfer")
            self._insert_instance(conn, "instance_transfer_race")

            source_seq = self._append_node(
                conn, "resource_lease_source", "resource_lease")
            conn.execute(
                """INSERT INTO resource_leases
                   (lease_id,step_id,attempt_id,worker_id,runtime_id,
                    profile_fingerprint,latency_class,cpu_millis,ram_mib,
                    concurrency_slots,network_slots,accelerator,vram_mib,status,
                    acquired_at,heartbeat_at,expires_at,last_event_seq)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'active',?,?,?,?)""",
                (
                    "resource_lease_source",
                    "step-source",
                    "attempt-source",
                    "worker-source",
                    "runtime-process-schema",
                    "e" * 64,
                    "interactive",
                    1000,
                    512,
                    1,
                    0,
                    "none",
                    0,
                    "2026-08-24T08:00:00Z",
                    "2026-08-24T08:00:00Z",
                    "2026-08-24T08:01:00Z",
                    source_seq,
                ),
            )
            self._insert_workload_lease(
                conn,
                "workload_transfer",
                "instance_transfer",
                source_step_lease_id="resource_lease_source",
                source_attempt_id="attempt-source",
                source_worker_id="worker-source",
            )
            with self.assertRaises(sqlite3.IntegrityError):
                self._insert_workload_lease(
                    conn,
                    "workload_transfer_race",
                    "instance_transfer_race",
                    source_step_lease_id="resource_lease_source",
                    source_attempt_id="attempt-source",
                    source_worker_id="worker-source",
                )

            orphan_seq = self._append_node(
                conn, "instance_orphan", "process_instance")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """INSERT INTO process_instances
                       (instance_id,spec_id,launch_idempotency_key,
                        args_ciphertext,args_redacted_json,args_sha256,
                        spec_fingerprint,sandbox_fingerprint,state,unit_name,
                        prepared_at,created_at,updated_at,last_event_seq)
                       VALUES (?,'missing-spec',?,'sealed','{}',?,?,?,
                               'prepared',?,?,?, ?,?)""",
                    (
                        "instance_orphan",
                        "launch-orphan",
                        "c" * 64,
                        "d" * 64,
                        "b" * 64,
                        "friday-instance-orphan.service",
                        "2026-08-24T08:00:00Z",
                        "2026-08-24T08:00:00Z",
                        "2026-08-24T08:00:00Z",
                        orphan_seq,
                    ),
                )


class ProcessSchemaMigrationTests(unittest.TestCase):
    @staticmethod
    def _create_v8(path: Path) -> None:
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
                if version > 8:
                    break
                migration(conn)
                conn.execute(
                    "INSERT INTO schema_migrations(version) VALUES (?)",
                    (version,),
                )
            conn.execute("PRAGMA user_version = 8")
            conn.commit()

    @classmethod
    def _create_v10(cls, path: Path) -> None:
        cls._create_v8(path)
        with sqlite3.connect(path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            for version, migration in MIGRATIONS:
                if version < 9:
                    continue
                if version > 10:
                    break
                migration(conn)
                conn.execute(
                    "INSERT INTO schema_migrations(version) VALUES (?)",
                    (version,),
                )
            conn.execute("PRAGMA user_version = 10")
            conn.commit()

    @classmethod
    def _create_v12(cls, path: Path) -> None:
        cls._create_v10(path)
        with sqlite3.connect(path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            for version, migration in MIGRATIONS:
                if version < 11:
                    continue
                if version > 12:
                    break
                migration(conn)
                conn.execute(
                    "INSERT INTO schema_migrations(version) VALUES (?)",
                    (version,),
                )
            conn.execute("PRAGMA user_version = 12")
            conn.commit()

    @staticmethod
    def _insert_v8_terminal(path: Path) -> str:
        instance_id = "process_" + "9" * 32
        nonterminal_id = "process_" + "8" * 32
        with sqlite3.connect(path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            cursor = conn.execute(
                """INSERT INTO graph_events
                   (event_id,occurred_at,actor,event_type,payload_json,
                    payload_sha256)
                   VALUES ('event_v8_terminal','2026-08-24T08:00:00Z','test',
                           'process.exited','{}','event-hash')"""
            )
            seq = int(cursor.lastrowid)
            conn.execute(
                """INSERT INTO nodes
                   (node_id,kind,created_event_id,body_json,body_sha256)
                   VALUES ('spec_v8','process_spec','event_v8_terminal','{}',
                           'spec-node-hash'),
                          (?,'process_instance','event_v8_terminal','{}',
                           'instance-node-hash'),
                          (?,'process_instance','event_v8_terminal','{}',
                           'prepared-node-hash'),
                          ('workload_v8','workload_resource_lease',
                           'event_v8_terminal','{}','workload-node-hash')""",
                (instance_id, nonterminal_id),
            )
            conn.execute(
                """INSERT INTO process_specs
                   (spec_id,name,version,spec_ciphertext,display_json,
                    spec_sha256,sandbox_fingerprint,status,created_at,
                    updated_at,last_event_seq)
                   VALUES ('spec_v8','legacy',1,'sealed','{}',?,?,
                           'active','2026-08-24T08:00:00Z',
                           '2026-08-24T08:00:00Z',?)""",
                ("a" * 64, "b" * 64, seq),
            )
            conn.execute(
                """INSERT INTO process_instances
                   (instance_id,spec_id,launch_idempotency_key,args_ciphertext,
                    args_redacted_json,args_sha256,spec_fingerprint,
                    sandbox_fingerprint,state,unit_name,result_code,
                    prepared_at,finished_at,created_at,updated_at,last_event_seq)
                   VALUES (?,'spec_v8','launch-v8-terminal','sealed','{}',?,?,?,
                           'exited',?,'success','2026-08-24T08:00:00Z',
                           '2026-08-24T08:01:00Z','2026-08-24T08:00:00Z',
                           '2026-08-24T08:01:00Z',?)""",
                (instance_id, "c" * 64, "a" * 64, "b" * 64,
                 "friday-proc-" + "9" * 32 + ".service", seq),
            )
            conn.execute(
                """INSERT INTO process_instances
                   (instance_id,spec_id,launch_idempotency_key,args_ciphertext,
                    args_redacted_json,args_sha256,spec_fingerprint,
                    sandbox_fingerprint,state,unit_name,prepared_at,created_at,
                    updated_at,last_event_seq)
                   VALUES (?,'spec_v8','launch-v8-prepared','sealed','{}',?,?,?,
                           'prepared',?,'2026-08-24T08:00:00Z',
                           '2026-08-24T08:00:00Z','2026-08-24T08:00:00Z',?)""",
                (nonterminal_id, "d" * 64, "a" * 64, "b" * 64,
                 "friday-proc-" + "8" * 32 + ".service", seq),
            )
            conn.execute(
                """INSERT INTO workload_resource_leases
                   (lease_id,instance_id,runtime_id,profile_fingerprint,
                    latency_class,cpu_millis,ram_mib,concurrency_slots,
                    network_slots,accelerator,vram_mib,enforcement_json,status,
                    acquired_at,heartbeat_at,expires_at,released_at,
                    release_reason,last_event_seq)
                   VALUES ('workload_v8',?,'runtime_v8',?,'background',1000,512,
                           1,0,'none',0,'{}','released',
                           '2026-08-24T08:00:00Z','2026-08-24T08:00:30Z',
                           '2026-08-24T08:01:00Z','2026-08-24T08:01:00Z',
                           'exited',?)""",
                (instance_id, "e" * 64, seq),
            )
            conn.commit()
        return instance_id

    def test_genuine_v8_terminal_rows_are_backfilled_idempotently(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "friday-v8.db"
            self._create_v8(path)
            instance_id = self._insert_v8_terminal(path)
            with sqlite3.connect(path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                self.assertEqual(apply_schema_migrations(conn), 14)
                self.assertEqual(apply_schema_migrations(conn), 14)
                cleanup = conn.execute(
                    "SELECT state,attempt_count,requested_at,last_event_seq "
                    "FROM process_unit_cleanups WHERE instance_id=?",
                    (instance_id,),
                ).fetchone()
                cleanup_count = int(conn.execute(
                    "SELECT COUNT(*) FROM process_unit_cleanups"
                ).fetchone()[0])
                prepared_cleanup = conn.execute(
                    "SELECT 1 FROM process_unit_cleanups WHERE instance_id=?",
                    ("process_" + "8" * 32,),
                ).fetchone()
                workload = conn.execute(
                    "SELECT status,released_at FROM workload_resource_leases "
                    "WHERE instance_id=?", (instance_id,),
                ).fetchone()
                versions = [int(row[0]) for row in conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version")]
            self.assertEqual(
                cleanup,
                ("pending", 0, "2026-08-24T08:01:00Z", cleanup[3]))
            self.assertGreater(cleanup[3], 0)
            self.assertEqual(cleanup_count, 1)
            self.assertIsNone(prepared_cleanup)
            self.assertEqual(
                workload, ("released", "2026-08-24T08:01:00Z"))
            self.assertEqual(versions, list(range(1, 15)))

    def test_v9_ddl_backfill_and_marker_roll_back_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "friday-v8-rollback.db"
            self._create_v8(path)
            self._insert_v8_terminal(path)
            migration_9 = dict(MIGRATIONS)[9]

            def fail_after_backfill(conn):
                migration_9(conn)
                raise RuntimeError("injected migration failure")

            replaced = tuple(
                (version, fail_after_backfill if version == 9 else migration)
                for version, migration in MIGRATIONS)
            with sqlite3.connect(path) as conn, mock.patch.object(
                    db_migrations, "MIGRATIONS", replaced):
                with self.assertRaises(RuntimeError):
                    apply_schema_migrations(conn)
                table = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='process_unit_cleanups'"
                ).fetchone()
                marker = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version=9"
                ).fetchone()
                user_version = int(
                    conn.execute("PRAGMA user_version").fetchone()[0])
            self.assertIsNone(table)
            self.assertIsNone(marker)
            self.assertEqual(user_version, 8)

    def test_v10_upgrades_to_empty_operation_and_controller_state_idempotently(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "friday-v10.db"
            self._create_v10(path)
            with sqlite3.connect(path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                self.assertEqual(apply_schema_migrations(conn), 14)
                self.assertEqual(apply_schema_migrations(conn), 14)
                count = int(conn.execute(
                    "SELECT COUNT(*) FROM process_operations").fetchone()[0])
                controller_count = int(conn.execute(
                    "SELECT COUNT(*) FROM controller_identities").fetchone()[0])
                versions = [int(row[0]) for row in conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version")]
                user_version = int(
                    conn.execute("PRAGMA user_version").fetchone()[0])
            self.assertEqual(count, 0)
            self.assertEqual(controller_count, 0)
            self.assertEqual(versions, list(range(1, 15)))
            self.assertEqual(user_version, 14)

    def test_v11_ddl_and_marker_roll_back_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "friday-v10-rollback.db"
            self._create_v10(path)
            migration_11 = dict(MIGRATIONS)[11]

            def fail_after_ddl(conn):
                migration_11(conn)
                raise RuntimeError("injected v11 migration failure")

            replaced = tuple(
                (version, fail_after_ddl if version == 11 else migration)
                for version, migration in MIGRATIONS)
            with sqlite3.connect(path) as conn, mock.patch.object(
                    db_migrations, "MIGRATIONS", replaced):
                conn.execute("PRAGMA foreign_keys = ON")
                with self.assertRaises(RuntimeError):
                    apply_schema_migrations(conn)
                table = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='process_operations'").fetchone()
                marker = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version=11").fetchone()
                user_version = int(
                    conn.execute("PRAGMA user_version").fetchone()[0])
            self.assertIsNone(table)
            self.assertIsNone(marker)
            self.assertEqual(user_version, 10)

    def test_concurrent_v10_upgraders_share_one_v11_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concurrent-v10.db"
            self._create_v10(path)

            def upgrade(_index: int) -> int:
                with sqlite3.connect(path, timeout=15) as conn:
                    conn.execute("PRAGMA foreign_keys = ON")
                    conn.execute("PRAGMA busy_timeout = 15000")
                    return apply_schema_migrations(conn)

            with ThreadPoolExecutor(max_workers=8) as pool:
                versions = list(pool.map(upgrade, range(16)))
            with sqlite3.connect(path) as conn:
                marker_count = int(conn.execute(
                    "SELECT COUNT(*) FROM schema_migrations WHERE version=11"
                ).fetchone()[0])
                operation_count = int(conn.execute(
                    "SELECT COUNT(*) FROM process_operations").fetchone()[0])
                integrity = str(conn.execute(
                    "PRAGMA quick_check").fetchone()[0])
            self.assertEqual(versions, [14] * 16)
            self.assertEqual(marker_count, 1)
            self.assertEqual(operation_count, 0)
            self.assertEqual(integrity, "ok")

    def test_v13_ddl_and_marker_roll_back_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "friday-v12-rollback.db"
            self._create_v12(path)
            migration_13 = dict(MIGRATIONS)[13]

            def fail_after_ddl(conn):
                migration_13(conn)
                raise RuntimeError("injected v13 migration failure")

            replaced = tuple(
                (version, fail_after_ddl if version == 13 else migration)
                for version, migration in MIGRATIONS)
            with sqlite3.connect(path) as conn, mock.patch.object(
                    db_migrations, "MIGRATIONS", replaced):
                conn.execute("PRAGMA foreign_keys = ON")
                with self.assertRaises(RuntimeError):
                    apply_schema_migrations(conn)
                tables = {str(row[0]) for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name IN ('controller_task_authorities',"
                    "'controller_approval_requests',"
                    "'controller_approval_decisions',"
                    "'controller_effect_uses')")}
                indexes = {str(row[0]) for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name IN ('controller_sessions_authority_identity',"
                    "'approval_state_exact_request')")}
                marker = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version=13"
                ).fetchone()
                user_version = int(
                    conn.execute("PRAGMA user_version").fetchone()[0])
            self.assertEqual(tables, set())
            self.assertEqual(indexes, set())
            self.assertIsNone(marker)
            self.assertEqual(user_version, 12)

    def test_concurrent_v12_upgraders_share_one_v13_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concurrent-v12.db"
            self._create_v12(path)

            def upgrade(_index: int) -> int:
                with sqlite3.connect(path, timeout=15) as conn:
                    conn.execute("PRAGMA foreign_keys = ON")
                    conn.execute("PRAGMA busy_timeout = 15000")
                    return apply_schema_migrations(conn)

            with ThreadPoolExecutor(max_workers=8) as pool:
                versions = list(pool.map(upgrade, range(16)))
            with sqlite3.connect(path) as conn:
                marker_count = int(conn.execute(
                    "SELECT COUNT(*) FROM schema_migrations WHERE version=13"
                ).fetchone()[0])
                authority_counts = [int(conn.execute(
                    f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                    for table in (
                        "controller_task_authorities",
                        "controller_approval_requests",
                        "controller_approval_decisions",
                        "controller_effect_uses",
                    )]
                integrity = str(conn.execute(
                    "PRAGMA quick_check").fetchone()[0])
                foreign_keys = conn.execute(
                    "PRAGMA foreign_key_check").fetchall()
            self.assertEqual(versions, [14] * 16)
            self.assertEqual(marker_count, 1)
            self.assertEqual(authority_counts, [0, 0, 0, 0])
            self.assertEqual(integrity, "ok")
            self.assertEqual(foreign_keys, [])

    def test_runtime_rejects_future_or_gapped_schema_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "future.db"
            graph = GraphStore(path)
            with graph.transaction() as conn:
                conn.execute(
                    "INSERT INTO schema_migrations(version) VALUES (15)")
                conn.execute("PRAGMA user_version = 15")
            with graph._connect() as conn:
                with self.assertRaisesRegex(
                        RuntimeError, "schema is newer"):
                    apply_schema_migrations(conn)
            with graph.transaction() as conn:
                conn.execute("DELETE FROM schema_migrations WHERE version=15")
                conn.execute("DELETE FROM schema_migrations WHERE version=8")
                conn.execute("PRAGMA user_version = 14")
            with graph._connect() as conn:
                with self.assertRaisesRegex(
                        RuntimeError, "history is incomplete"):
                    apply_schema_migrations(conn)

    def test_concurrent_fresh_initializers_serialize_schema_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concurrent.db"

            def initialize(_index: int) -> int:
                return GraphStore(path).schema_version()

            with ThreadPoolExecutor(max_workers=8) as pool:
                versions = list(pool.map(initialize, range(16)))
            with sqlite3.connect(path) as conn:
                markers = [int(row[0]) for row in conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version")]
                integrity = str(conn.execute(
                    "PRAGMA quick_check").fetchone()[0])
            lock_path = Path(str(path) + ".schema.lock")
            self.assertEqual(versions, [14] * 16)
            self.assertEqual(markers, list(range(1, 15)))
            self.assertEqual(integrity, "ok")
            self.assertEqual(stat.S_IMODE(lock_path.stat().st_mode), 0o600)

    def test_concurrent_v8_upgraders_share_one_v9_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concurrent-v8.db"
            self._create_v8(path)
            self._insert_v8_terminal(path)

            def upgrade(_index: int) -> int:
                with sqlite3.connect(path, timeout=15) as conn:
                    conn.execute("PRAGMA foreign_keys = ON")
                    conn.execute("PRAGMA busy_timeout = 15000")
                    return apply_schema_migrations(conn)

            with ThreadPoolExecutor(max_workers=8) as pool:
                versions = list(pool.map(upgrade, range(16)))
            with sqlite3.connect(path) as conn:
                marker_count = int(conn.execute(
                    "SELECT COUNT(*) FROM schema_migrations WHERE version=9"
                ).fetchone()[0])
                cleanup_count = int(conn.execute(
                    "SELECT COUNT(*) FROM process_unit_cleanups"
                ).fetchone()[0])
                integrity = str(conn.execute(
                    "PRAGMA quick_check").fetchone()[0])
            self.assertEqual(versions, [14] * 16)
            self.assertEqual(marker_count, 1)
            self.assertEqual(cleanup_count, 1)
            self.assertEqual(integrity, "ok")

    def test_genuine_v7_database_upgrades_twice_without_data_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "friday-v7.db"
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
                    if version > 7:
                        break
                    migration(conn)
                    conn.execute(
                        "INSERT INTO schema_migrations(version) VALUES (?)",
                        (version,),
                    )
                conn.execute("PRAGMA user_version = 7")

                cursor = conn.execute(
                    """INSERT INTO graph_events
                       (event_id,occurred_at,actor,event_type,payload_json,
                        payload_sha256)
                       VALUES ('event_v7','2026-08-24T08:00:00Z','test',
                               'v7.fixture','{}','event-hash')"""
                )
                seq = int(cursor.lastrowid)
                conn.execute(
                    """INSERT INTO nodes
                       (node_id,kind,created_event_id,body_json,body_sha256)
                       VALUES ('task_v7','task','event_v7','{}','task-hash'),
                              ('resource_lease_v7','resource_lease','event_v7',
                               '{}','lease-hash')"""
                )
                conn.execute(
                    """INSERT INTO task_state
                       (task_id,objective,completion_contract_json,status,
                        plan_json,created_at,updated_at,last_event_seq)
                       VALUES ('task_v7','preserve me','{}','running','[]',
                               '2026-08-24T08:00:00Z',
                               '2026-08-24T08:00:00Z',?)""",
                    (seq,),
                )
                conn.execute(
                    """INSERT INTO resource_leases
                       (lease_id,step_id,attempt_id,worker_id,runtime_id,
                        profile_fingerprint,latency_class,cpu_millis,ram_mib,
                        concurrency_slots,network_slots,accelerator,vram_mib,
                        status,acquired_at,heartbeat_at,expires_at,last_event_seq)
                       VALUES ('resource_lease_v7','step_v7','attempt_v7',
                               'worker_v7','runtime_v7',?,'background',750,1536,
                               1,1,'cuda:0',2048,'active',
                               '2026-08-24T08:00:00Z',
                               '2026-08-24T08:00:01Z',
                               '2026-08-24T08:01:01Z',?)""",
                    ("f" * 64, seq),
                )
                conn.commit()

                self.assertIsNone(conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='process_specs'"
                ).fetchone())
                self.assertEqual(
                    int(conn.execute("PRAGMA user_version").fetchone()[0]), 7)

                self.assertEqual(apply_schema_migrations(conn), 14)
                self.assertEqual(apply_schema_migrations(conn), 14)
                conn.commit()

                versions = [int(row[0]) for row in conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version")]
                task = conn.execute(
                    "SELECT objective,status FROM task_state WHERE task_id='task_v7'"
                ).fetchone()
                lease = conn.execute(
                    """SELECT step_id,attempt_id,worker_id,runtime_id,
                              profile_fingerprint,latency_class,cpu_millis,
                              ram_mib,accelerator,vram_mib,status
                       FROM resource_leases
                       WHERE lease_id='resource_lease_v7'"""
                ).fetchone()
                new_counts = tuple(int(conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]) for table in (
                    "process_specs",
                    "process_instances",
                    "workload_resource_leases",
                    "process_unit_cleanups",
                ))
                user_version = int(
                    conn.execute("PRAGMA user_version").fetchone()[0])
                fk_mode = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
                journal_mode = str(
                    conn.execute("PRAGMA journal_mode").fetchone()[0])

            self.assertEqual(versions, list(range(1, 15)))
            self.assertEqual(user_version, 14)
            self.assertEqual(task, ("preserve me", "running"))
            self.assertEqual(
                lease,
                (
                    "step_v7",
                    "attempt_v7",
                    "worker_v7",
                    "runtime_v7",
                    "f" * 64,
                    "background",
                    750,
                    1536,
                    "cuda:0",
                    2048,
                    "active",
                ),
            )
            self.assertEqual(new_counts, (0, 0, 0, 0))
            self.assertEqual((journal_mode, fk_mode), ("wal", 1))


if __name__ == "__main__":
    unittest.main()
