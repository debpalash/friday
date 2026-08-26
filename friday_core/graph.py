"""Append-only SQLite event graph.

The graph tables are immutable. Mutable tables are explicitly projections that can be
rebuilt from events. All write operations use BEGIN IMMEDIATE so an event and its graph
objects are committed together.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .db_migrations import apply_schema_migrations


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class GraphStore:
    """Small transactional API over Friday's append-only graph."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._schema = Path(__file__).with_name("schema.sql").read_text()
        self._init_lock = threading.Lock()
        self._initialized = False
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        self._harden_files()
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 15000")
        return conn

    def _harden_files(self) -> None:
        for path in (self.path, Path(str(self.path) + "-wal"),
                     Path(str(self.path) + "-shm"),
                     Path(str(self.path) + ".schema.lock")):
            try:
                os.chmod(path, 0o600)
            except FileNotFoundError:
                pass

    def initialize(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            lock_path = Path(str(self.path) + ".schema.lock")
            flags = (os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
                     | getattr(os, "O_NOFOLLOW", 0))
            descriptor = os.open(lock_path, flags, 0o600)
            try:
                observed = os.fstat(descriptor)
                if (not stat.S_ISREG(observed.st_mode)
                        or observed.st_uid != os.getuid()
                        or observed.st_nlink != 1):
                    raise RuntimeError("database schema lock identity is invalid")
                os.fchmod(descriptor, 0o600)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                with self._connect() as conn:
                    conn.executescript(self._schema)
                    apply_schema_migrations(conn)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            self._harden_files()
            self._initialized = True

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
            self._harden_files()

    def append_event(
        self,
        conn: sqlite3.Connection,
        event_type: str,
        payload: dict[str, Any],
        *,
        actor: str = "friday",
        session_id: str | None = None,
        turn_id: str | None = None,
        task_id: str | None = None,
        idempotency_key: str | None = None,
        occurred_at: str | None = None,
    ) -> tuple[str, int]:
        event_id = new_id("evt")
        body = canonical_json(payload)
        cur = conn.execute(
            """INSERT INTO graph_events
               (event_id, occurred_at, actor, session_id, turn_id, task_id,
                event_type, payload_json, payload_sha256, idempotency_key)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_id, occurred_at or utc_now(), actor, session_id, turn_id,
             task_id, event_type, body, sha256_text(body), idempotency_key),
        )
        return event_id, int(cur.lastrowid)

    def append_node(
        self,
        conn: sqlite3.Connection,
        kind: str,
        body: dict[str, Any],
        *,
        event_id: str,
        node_id: str | None = None,
    ) -> str:
        node_id = node_id or new_id(kind)
        encoded = canonical_json(body)
        conn.execute(
            """INSERT INTO nodes
               (node_id, kind, created_event_id, body_json, body_sha256)
               VALUES (?, ?, ?, ?, ?)""",
            (node_id, kind, event_id, encoded, sha256_text(encoded)),
        )
        return node_id

    def append_edge(
        self,
        conn: sqlite3.Connection,
        from_node_id: str,
        relation: str,
        to_node_id: str,
        *,
        event_id: str,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        edge_id = new_id("edge")
        conn.execute(
            """INSERT INTO edges
               (edge_id, from_node_id, relation, to_node_id, created_event_id,
                attributes_json) VALUES (?, ?, ?, ?, ?, ?)""",
            (edge_id, from_node_id, relation, to_node_id, event_id,
             canonical_json(attributes or {})),
        )
        return edge_id

    def record_node(
        self,
        kind: str,
        body: dict[str, Any],
        *,
        actor: str = "friday",
        session_id: str | None = None,
        turn_id: str | None = None,
        task_id: str | None = None,
        event_type: str | None = None,
        links: list[tuple[str, str]] | None = None,
    ) -> str:
        """Append one node and optional (new node, relation, existing node) edges."""
        with self.transaction() as conn:
            event_id, _ = self.append_event(
                conn, event_type or f"{kind}.recorded", body, actor=actor,
                session_id=session_id, turn_id=turn_id, task_id=task_id,
            )
            node_id = self.append_node(conn, kind, body, event_id=event_id)
            for relation, target_id in links or []:
                self.append_edge(conn, node_id, relation, target_id,
                                 event_id=event_id)
            return node_id

    def record_edge(
        self,
        from_node_id: str,
        relation: str,
        to_node_id: str,
        *,
        actor: str = "friday",
        task_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        payload = {"from": from_node_id, "relation": relation, "to": to_node_id,
                   "attributes": attributes or {}}
        with self.transaction() as conn:
            event_id, _ = self.append_event(conn, "edge.recorded", payload,
                                             actor=actor, task_id=task_id)
            return self.append_edge(conn, from_node_id, relation, to_node_id,
                                    event_id=event_id, attributes=attributes)

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE node_id = ?",
                               (node_id,)).fetchone()
        if row is None:
            return None
        return {"node_id": row["node_id"], "kind": row["kind"],
                "created_event_id": row["created_event_id"],
                "body": json.loads(row["body_json"]),
                "body_sha256": row["body_sha256"]}

    def events_since(self, seq: int = 0, *, task_id: str | None = None,
                     limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT * FROM graph_events WHERE seq > ?"
        params: list[Any] = [seq]
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += " ORDER BY seq LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) | {"payload": json.loads(row["payload_json"])}
                for row in rows]

    def count(self, table: str) -> int:
        if table not in {"graph_events", "nodes", "edges", "task_state",
                         "action_receipts", "claim_state", "progress_outbox",
                         "skill_state", "skill_versions", "deployment_state",
                         "capability_state", "capability_versions", "voice_profiles",
                         "core_upgrade_state", "task_verifications", "feedback_state",
                         "transcript_corrections", "reminder_state", "approval_state",
                         "model_disclosures", "schema_migrations", "task_steps",
                         "task_step_batches", "action_attempts",
                         "operator_grants", "resource_leases",
                         "process_specs", "process_instances",
                         "workload_resource_leases",
                         "process_unit_cleanups", "process_operations",
                         "controller_identities", "controller_pairings",
                         "controller_session_challenges",
                         "controller_sessions",
                         "controller_task_authorities",
                         "controller_approval_requests",
                         "controller_approval_decisions",
                         "controller_effect_uses",
                         "memory_embedding_index"}:
            raise ValueError("unsupported table")
        with self._connect() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def count_nodes(self, kind: str) -> int:
        with self._connect() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE kind=?", (kind,)
            ).fetchone()[0])

    def schema_version(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])
