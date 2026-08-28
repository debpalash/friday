from __future__ import annotations

import json
import os
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from friday_core import data_lifecycle
from friday_core.data_lifecycle import (
    delete_private_data,
    export_private_data,
    plan_private_deletion,
    verify_private_export,
)
from friday_core import MemoryCurator, TaskService
from friday_core.graph import GraphStore


class PrivateDataExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "friday.db"
        self.graph = GraphStore(self.database)
        self.node_id = self.graph.record_node(
            "conversation_message",
            {"role": "user", "content": "private fixture"},
            session_id="session-fixture",
            turn_id="turn-fixture",
        )
        self.export = self.root / "private-export"

    def tearDown(self):
        self.temporary.cleanup()

    def test_export_is_private_complete_and_verifiable(self):
        manifest = export_private_data(self.database, self.export)
        verified = verify_private_export(self.export)
        self.assertEqual(verified, manifest)
        self.assertEqual(manifest["format"], "friday-private-export")
        self.assertGreaterEqual(manifest["schema_version"], 14)
        table_names = {item["name"] for item in manifest["tables"]}
        self.assertIn("graph_events", table_names)
        self.assertIn("memory_fts", table_names)
        self.assertNotIn("memory_fts_data", table_names)
        self.assertEqual(stat.S_IMODE(self.export.stat().st_mode), 0o700)
        for filename in ("friday.sqlite3", "manifest.json"):
            self.assertEqual(
                stat.S_IMODE((self.export / filename).stat().st_mode), 0o600)
        with sqlite3.connect(self.export / "friday.sqlite3") as snapshot:
            body = snapshot.execute(
                "SELECT body_json FROM nodes WHERE node_id = ?", (self.node_id,)
            ).fetchone()[0]
        self.assertEqual(json.loads(body)["content"], "private fixture")

    def test_database_tampering_is_detected(self):
        export_private_data(self.database, self.export)
        database = self.export / "friday.sqlite3"
        with database.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(RuntimeError, "size does not match"):
            verify_private_export(self.export)

    def test_manifest_tampering_is_detected(self):
        export_private_data(self.database, self.export)
        manifest_path = self.export / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tables"] = []
        manifest_path.write_text(json.dumps(manifest))
        manifest_path.chmod(0o600)
        with self.assertRaisesRegex(RuntimeError, "inventory does not match"):
            verify_private_export(self.export)

    def test_unexpected_export_entries_are_rejected(self):
        export_private_data(self.database, self.export)
        extra = self.export / "notes.txt"
        extra.write_text("not part of the format")
        extra.chmod(0o600)
        with self.assertRaisesRegex(RuntimeError, "unexpected entries"):
            verify_private_export(self.export)

    def test_export_and_verifier_reject_symlink_targets(self):
        real = self.root / "real"
        real.mkdir(mode=0o700)
        self.export.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            export_private_data(self.database, self.export)
        with self.assertRaisesRegex(RuntimeError, "directory is invalid"):
            verify_private_export(self.export)

    def test_export_rejects_public_source_database(self):
        self.database.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, "file is invalid"):
            export_private_data(self.database, self.export)

    def test_failed_export_removes_private_partial_directory(self):
        with mock.patch.object(
            data_lifecycle,
            "_backup_database",
            side_effect=RuntimeError("injected backup failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected backup failure"):
                export_private_data(self.database, self.export)
        self.assertFalse(self.export.exists())
        self.assertEqual(list(self.root.glob(".private-export.partial-*")), [])


class SelectiveDeletionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _database(self, name: str) -> tuple[Path, GraphStore]:
        path = self.root / name / "friday.db"
        path.parent.mkdir(mode=0o700)
        return path, GraphStore(path)

    @staticmethod
    def _assert_database_valid(path: Path) -> None:
        with sqlite3.connect(path) as connection:
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
            assert connection.execute("PRAGMA integrity_check").fetchall() == [
                ("ok",)]

    def test_conversation_deletion_is_compacted_and_preserves_other_session(self):
        path, graph = self._database("conversation")
        removed_session = graph.record_node("session", {"label": "remove-alpha"})
        removed_turn = graph.record_node(
            "turn", {"text": "remove-alpha-private"},
            session_id=removed_session, turn_id="turn_remove_alpha")
        graph.record_edge(removed_session, "contains", removed_turn)
        retained_session = graph.record_node("session", {"label": "keep-bravo"})
        retained_turn = graph.record_node(
            "turn", {"text": "keep-bravo-private"},
            session_id=retained_session, turn_id="turn_keep_bravo")
        graph.record_edge(retained_session, "contains", retained_turn)

        preview = plan_private_deletion(
            path, "conversation", value=removed_session)
        result = delete_private_data(
            path, "conversation", value=removed_session, runtime_stopped=True)

        self.assertTrue(preview["matched"])
        self.assertEqual(result["rows"], preview["rows"])
        self.assertEqual(result["status"], "deleted")
        self.assertNotIn(b"remove-alpha-private", path.read_bytes())
        self.assertIn(b"keep-bravo-private", path.read_bytes())
        with sqlite3.connect(path) as connection:
            self.assertIsNone(connection.execute(
                "SELECT 1 FROM nodes WHERE node_id=?", (removed_session,)
            ).fetchone())
            self.assertIsNotNone(connection.execute(
                "SELECT 1 FROM nodes WHERE node_id=?", (retained_session,)
            ).fetchone())
            tombstone = connection.execute(
                "SELECT scope,selector_sha256,rows_json "
                "FROM deletion_tombstones"
            ).fetchone()
        self.assertEqual(tombstone[0], "conversation")
        self.assertNotIn(removed_session, "".join(map(str, tombstone)))
        self._assert_database_valid(path)

        repeated = delete_private_data(
            path, "conversation", value=removed_session, runtime_stopped=True)
        self.assertEqual(repeated["status"], "already_deleted")
        self.assertEqual(repeated["deletion_id"], result["deletion_id"])

    def test_task_deletion_removes_task_graph_and_keeps_other_task(self):
        path, graph = self._database("task")
        tasks = TaskService(graph)
        removed, _ = tasks.create(
            "remove-charlie-objective", {"evidence": "remove-charlie-contract"})
        retained, _ = tasks.create(
            "keep-delta-objective", {"evidence": "keep-delta-contract"})

        result = delete_private_data(
            path, "task", value=removed, runtime_stopped=True)

        self.assertEqual(result["status"], "deleted")
        self.assertNotIn(b"remove-charlie-objective", path.read_bytes())
        self.assertIn(b"keep-delta-objective", path.read_bytes())
        with sqlite3.connect(path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT task_id FROM task_state").fetchall(), [(retained,)])
        self._assert_database_valid(path)

    def test_artifact_deletion_removes_exact_node_and_provenance(self):
        path, graph = self._database("artifact")
        removed = graph.record_node(
            "artifact", {"content": "remove-echo-artifact"})
        retained = graph.record_node(
            "artifact", {"content": "keep-foxtrot-artifact"})

        result = delete_private_data(
            path, "artifact", value=removed, runtime_stopped=True)

        self.assertEqual(result["status"], "deleted")
        self.assertNotIn(b"remove-echo-artifact", path.read_bytes())
        self.assertIn(b"keep-foxtrot-artifact", path.read_bytes())
        with sqlite3.connect(path) as connection:
            self.assertEqual(
                connection.execute("SELECT node_id FROM nodes").fetchall(),
                [(retained,)],
            )
        self._assert_database_valid(path)

    def test_memory_claim_deletion_removes_search_projection(self):
        path, graph = self._database("memory")
        memory = MemoryCurator(graph)
        source = graph.record_node(
            "utterance", {"text": "memory source golf"}, actor="user")
        claim = memory.propose(
            subject="remove-golf-subject",
            predicate="prefers",
            object_value="remove-golf-object",
            scope="user_preference",
            evidence_class="user_explicit",
            source_node_ids=[source],
            confidence=1.0,
            retention_reason="remove-golf-reason",
        )
        self.assertTrue(memory.evaluate(claim).promoted)

        result = delete_private_data(
            path, "memory_claim", value=claim, runtime_stopped=True)

        self.assertEqual(result["status"], "deleted")
        self.assertNotIn(b"remove-golf-subject", path.read_bytes())
        with sqlite3.connect(path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT 1 FROM claim_state WHERE claim_id=?", (claim,)
                ).fetchall(), [])
            self.assertEqual(
                connection.execute(
                    "SELECT 1 FROM memory_fts WHERE claim_id=?", (claim,)
                ).fetchall(), [])
        self._assert_database_valid(path)

    def test_time_range_deletion_uses_inclusive_start_exclusive_end(self):
        path, graph = self._database("time")
        identifiers: list[str] = []
        for index, occurred_at in enumerate((
            "2026-08-01T00:00:00.000000Z",
            "2026-08-15T00:00:00.000000Z",
            "2026-09-01T00:00:00.000000Z",
        )):
            with graph.transaction() as connection:
                event_id, _seq = graph.append_event(
                    connection,
                    "fixture.recorded",
                    {"content": f"time-marker-{index}"},
                    occurred_at=occurred_at,
                )
                identifiers.append(graph.append_node(
                    connection,
                    "artifact",
                    {"content": f"time-marker-{index}"},
                    event_id=event_id,
                ))

        result = delete_private_data(
            path,
            "time_range",
            start="2026-08-01T00:00:00Z",
            end="2026-09-01T00:00:00Z",
            runtime_stopped=True,
        )

        self.assertEqual(result["status"], "deleted")
        with sqlite3.connect(path) as connection:
            remaining = {
                str(row[0]) for row in connection.execute(
                    "SELECT node_id FROM nodes")
            }
        self.assertEqual(remaining, {identifiers[2]})
        self.assertNotIn(b"time-marker-0", path.read_bytes())
        self.assertNotIn(b"time-marker-1", path.read_bytes())
        self.assertIn(b"time-marker-2", path.read_bytes())
        self._assert_database_valid(path)

    def test_confirmation_and_nonempty_selection_are_required(self):
        path, graph = self._database("guards")
        graph.record_node("artifact", {"content": "guard-hotel"})
        with self.assertRaisesRegex(RuntimeError, "service to be stopped"):
            delete_private_data(path, "artifact", value="missing")
        with self.assertRaisesRegex(RuntimeError, "matched no durable records"):
            delete_private_data(
                path, "artifact", value="missing", runtime_stopped=True)
        with sqlite3.connect(path) as connection:
            bodies = " ".join(str(row[0]) for row in connection.execute(
                "SELECT body_json FROM nodes"))
        self.assertIn("guard-hotel", bodies)

    def test_failed_rewrite_leaves_source_unchanged_and_removes_partial(self):
        path, graph = self._database("failure")
        artifact = graph.record_node(
            "artifact", {"content": "failure-india-private"})
        with mock.patch.object(
            data_lifecycle,
            "_delete_selected_rows",
            side_effect=RuntimeError("injected rewrite failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected rewrite failure"):
                delete_private_data(
                    path, "artifact", value=artifact, runtime_stopped=True)
        with sqlite3.connect(path) as connection:
            bodies = " ".join(str(row[0]) for row in connection.execute(
                "SELECT body_json FROM nodes"))
        self.assertIn("failure-india-private", bodies)
        self.assertEqual(list(path.parent.glob(".friday.db.delete-*")), [])
        self._assert_database_valid(path)


if __name__ == "__main__":
    unittest.main()
