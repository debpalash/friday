import json
import stat
import sqlite3
import tempfile
import unittest
from pathlib import Path

from friday_core import GraphStore


class GraphStoreTests(unittest.TestCase):
    def test_database_and_wal_are_private(self):
        with self.graph.transaction() as conn:
            self.graph.append_event(conn, "privacy.test", {"ok": True})

        paths = [self.graph.path, Path(str(self.graph.path) + "-wal")]
        for path in paths:
            if path.exists():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.graph = GraphStore(Path(self.tmp.name) / "friday.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_recorded_nodes_and_events_are_hashed_and_linked(self):
        session_id = self.graph.record_node("session", {"client": "test"})
        turn_id = self.graph.record_node(
            "turn", {"text": "hello"}, links=[("contains", session_id)])

        self.assertEqual(self.graph.count("nodes"), 2)
        self.assertEqual(self.graph.count("edges"), 1)
        self.assertEqual(self.graph.count("graph_events"), 2)
        turn = self.graph.get_node(turn_id)
        self.assertEqual(turn["body"], {"text": "hello"})
        self.assertEqual(len(turn["body_sha256"]), 64)

    def test_core_graph_is_append_only(self):
        node_id = self.graph.record_node("observation", {"value": 1})
        with self.graph._connect() as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("UPDATE nodes SET kind='claim' WHERE node_id=?", (node_id,))
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("DELETE FROM nodes WHERE node_id=?", (node_id,))

    def test_event_payload_has_canonical_hash(self):
        self.graph.record_node("observation", {"b": 2, "a": 1})
        event = self.graph.events_since()[0]
        self.assertEqual(json.loads(event["payload_json"]), {"a": 1, "b": 2})
        self.assertEqual(event["payload"], {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
