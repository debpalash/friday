import json
import tempfile
import unittest
from pathlib import Path

from friday_core import GraphStore, migrate_session_json


class MigrationTests(unittest.TestCase):
    def test_legacy_session_import_is_journal_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "session.json"
            session.write_text(json.dumps([
                {"role": "system", "content": "prompt"},
                {"role": "user", "content": "I prefer concise answers."},
                {"role": "assistant", "content": "Noted."},
            ]))
            graph = GraphStore(root / "friday.db")

            imported = migrate_session_json(graph, session)
            repeated = migrate_session_json(graph, session)

            self.assertEqual(imported, 2)
            self.assertEqual(repeated, 0)
            self.assertEqual(graph.count_nodes("legacy_session"), 1)
            self.assertEqual(graph.count("claim_state"), 0)
            with graph._connect() as conn:
                bodies = [json.loads(row[0]) for row in conn.execute(
                    "SELECT body_json FROM nodes WHERE kind='utterance'")]
            self.assertEqual(bodies[0]["knowledge_layer"], "journal_only")


if __name__ == "__main__":
    unittest.main()
