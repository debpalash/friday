import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from friday_core import GraphStore, MemoryCurator


class FakeEmbedder:
    fingerprint = "e" * 64
    dimension = 3

    def __init__(self, *, fail=False):
        self.fail = fail
        self.passage_calls = 0
        self.query_calls = 0

    def encode(self, texts, *, kind):
        if self.fail:
            raise RuntimeError("offline encoder unavailable")
        if kind == "query":
            self.query_calls += 1
        else:
            self.passage_calls += 1
        output = []
        for text in texts:
            lowered = text.casefold()
            if kind == "query":
                output.append([1.0, 0.0, 0.0])
            elif "visible progress" in lowered or "audio progress" in lowered:
                output.append([1.0, 0.0, 0.0])
            elif "short direct" in lowered:
                output.append([0.0, 1.0, 0.0])
            else:
                output.append([0.0, 0.0, 1.0])
        return np.asarray(output, dtype="<f4")


class MemoryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.graph = GraphStore(Path(self.tmp.name) / "friday.db")
        self.memory = MemoryCurator(self.graph)

    def tearDown(self):
        self.tmp.cleanup()

    def test_assistant_output_cannot_promote_itself(self):
        source = self.graph.record_node(
            "assistant_message", {"text": "Pulash prefers every answer in Latin."})
        claim_id = self.memory.propose(
            subject="Pulash", predicate="prefers", object_value="Latin",
            scope="user_preference", evidence_class="assistant_utterance",
            source_node_ids=[source], confidence=0.99,
            retention_reason="assistant said so",
        )

        decision = self.memory.evaluate(claim_id)

        self.assertFalse(decision.promoted)
        self.assertEqual(self.memory.retrieve("Latin"), [])

    def test_explicit_user_preference_is_retrievable(self):
        source = self.graph.record_node(
            "utterance", {"text": "Keep progress visible."}, actor="user")
        claim_id = self.memory.propose(
            subject="Pulash", predicate="prefers",
            object_value="visible progress", scope="user_preference",
            evidence_class="user_explicit", source_node_ids=[source], confidence=1.0,
            retention_reason="explicit preference",
        )

        decision = self.memory.evaluate(claim_id)
        hits = self.memory.retrieve("progress")

        self.assertTrue(decision.promoted)
        self.assertEqual([hit["claim_id"] for hit in hits], [claim_id])

    def test_candidate_requires_existing_provenance(self):
        with self.assertRaises(ValueError):
            self.memory.propose(
                subject="x", predicate="is", object_value="y", scope="project",
                evidence_class="model_inference", source_node_ids=["missing"],
                confidence=0.5, retention_reason="guess",
            )

    def test_evidence_label_cannot_forge_provenance(self):
        source = self.graph.record_node(
            "assistant_message", {"text": "I observed a passing test."})
        claim_id = self.memory.propose(
            subject="server", predicate="tests", object_value="pass",
            scope="project", evidence_class="deterministic_test",
            source_node_ids=[source], confidence=1.0,
            retention_reason="forged evidence class",
        )
        self.assertFalse(self.memory.evaluate(claim_id).promoted)

    def test_user_evidence_requires_a_user_authored_utterance(self):
        forged_source = self.graph.record_node(
            "utterance", {"text": "The user prefers Latin."}, actor="assistant")
        forged = self.memory.propose(
            subject="Pulash", predicate="prefers", object_value="Latin",
            scope="user_preference", evidence_class="user_explicit",
            source_node_ids=[forged_source], confidence=1.0,
            retention_reason="forged utterance kind")

        decision = self.memory.evaluate(forged)

        self.assertFalse(decision.promoted)
        self.assertEqual(self.memory.retrieve("Latin"), [])

    def test_expired_memory_is_neither_promoted_nor_retrieved(self):
        source = self.graph.record_node(
            "utterance", {"text": "Use amber alerts."}, actor="user")
        past = "2000-01-01T00:00:00.000000Z"
        expired = self.memory.propose(
            subject="Pulash", predicate="alert_style", object_value="amber",
            scope="user_preference", evidence_class="user_explicit",
            source_node_ids=[source], confidence=1.0,
            retention_reason="temporary preference", valid_until=past)
        self.assertFalse(self.memory.evaluate(expired).promoted)

        future = (datetime.now(UTC) + timedelta(days=2)).isoformat(
            timespec="microseconds").replace("+00:00", "Z")
        active = self.memory.propose(
            subject="Pulash", predicate="alert_style", object_value="amber",
            scope="user_preference", evidence_class="user_explicit",
            source_node_ids=[source], confidence=1.0,
            retention_reason="temporary preference", valid_until=future)
        self.assertTrue(self.memory.evaluate(active).promoted)
        self.assertEqual(self.memory.retrieve("amber")[0]["claim_id"], active)
        after = (datetime.now(UTC) + timedelta(days=3)).isoformat(
            timespec="microseconds").replace("+00:00", "Z")
        self.assertEqual(self.memory.retrieve("amber", now=after), [])

    def test_duplicate_refresh_leaves_one_active_claim(self):
        first_source = self.graph.record_node(
            "utterance", {"text": "Keep progress visible."}, actor="user")
        second_source = self.graph.record_node(
            "utterance", {"text": "Still keep progress visible."}, actor="user")
        claim_ids = []
        for source in (first_source, second_source):
            claim_id = self.memory.propose(
                subject="Pulash", predicate="progress_style",
                object_value="visible progress", scope="user_preference",
                evidence_class="user_explicit", source_node_ids=[source],
                confidence=1.0, retention_reason="explicit preference")
            self.assertTrue(self.memory.evaluate(claim_id).promoted)
            claim_ids.append(claim_id)

        hits = self.memory.retrieve("visible progress")

        self.assertEqual([item["claim_id"] for item in hits], [claim_ids[-1]])
        self.assertEqual(
            [item["lifecycle"] for item in self.memory.list(limit=10)].count(
                "active"), 1)

    def test_retrieval_is_ranked_bounded_and_fts_syntax_safe(self):
        memories = (
            ("progress_style", "visible progress updates"),
            ("answer_style", "show progress through detailed answers"),
            ("notification_color", "blue notifications"),
        )
        expected = None
        for predicate, value in memories:
            source = self.graph.record_node(
                "utterance", {"text": value}, actor="user")
            claim_id = self.memory.propose(
                subject="Pulash", predicate=predicate, object_value=value,
                scope="user_preference", evidence_class="user_explicit",
                source_node_ids=[source], confidence=1.0,
                retention_reason="explicit preference")
            self.memory.evaluate(claim_id)
            if predicate == "progress_style":
                expected = claim_id

        hits = self.memory.retrieve("How should progress update notifications appear?")

        self.assertEqual(hits[0]["claim_id"], expected)
        self.assertEqual(len(hits), 1)
        self.assertGreater(hits[0]["relevance_score"], 0)
        self.assertIn("progress", hits[0]["matched_terms"])
        self.assertEqual(self.memory.retrieve('\" OR * NEAR( ) --'), [])
        with self.assertRaises(ValueError):
            self.memory.retrieve("x" * 2_001)

    def test_memory_values_and_timestamps_are_bounded_finite_data(self):
        source = self.graph.record_node("utterance", {"text": "x"}, actor="user")
        for object_value, valid_until in (
                (float("nan"), None),
                ("x", "tomorrow"),
                ("x" * 40_000, None)):
            with self.assertRaises(ValueError):
                self.memory.propose(
                    subject="Pulash", predicate="test", object_value=object_value,
                    scope="user_preference", evidence_class="user_explicit",
                    source_node_ids=[source], confidence=1.0,
                    retention_reason="test", valid_until=valid_until)

    def test_semantic_fallback_is_cached_ranked_and_expiry_safe(self):
        embedder = FakeEmbedder()
        memory = MemoryCurator(self.graph, embedder=embedder)
        target = None
        for predicate, value in (
                ("progress_style", "visible progress updates"),
                ("answer_style", "short direct answers"),
                ("interface_color", "blue controls")):
            source = self.graph.record_node(
                "utterance", {"text": value}, actor="user")
            claim_id = memory.propose(
                subject="Pulash", predicate=predicate, object_value=value,
                scope="user_preference", evidence_class="user_explicit",
                source_node_ids=[source], confidence=1.0,
                retention_reason="explicit preference",
                valid_until=("2099-01-01T00:00:00.000000Z"
                             if predicate == "progress_style" else None))
            memory.evaluate(claim_id)
            if predicate == "progress_style":
                target = claim_id

        query = "¿Cómo debo mantenerte informado mientras trabajas?"
        first = memory.retrieve(query)
        second = memory.retrieve(query)

        self.assertEqual(first[0]["claim_id"], target)
        self.assertEqual(first[0]["retrieval_mode"], "semantic_fallback")
        self.assertEqual(second[0]["claim_id"], target)
        self.assertEqual(embedder.passage_calls, 1)
        self.assertEqual(embedder.query_calls, 2)
        self.assertEqual(self.graph.count("memory_embedding_index"), 3)
        self.assertEqual(memory.retrieve(
            query, now="2100-01-01T00:00:00.000000Z"), [])

    def test_semantic_projection_repairs_tamper_and_drops_superseded_vector(self):
        embedder = FakeEmbedder()
        memory = MemoryCurator(self.graph, embedder=embedder)
        old_source = self.graph.record_node(
            "utterance", {"text": "visible progress updates"}, actor="user")
        old = memory.propose(
            subject="Pulash", predicate="progress_style",
            object_value="visible progress updates", scope="user_preference",
            evidence_class="user_explicit", source_node_ids=[old_source],
            confidence=1.0, retention_reason="explicit preference")
        memory.evaluate(old)
        query = "¿Cómo debo mantenerte informado mientras trabajas?"
        self.assertEqual(memory.retrieve(query)[0]["claim_id"], old)
        with self.graph.transaction() as conn:
            conn.execute(
                "UPDATE memory_embedding_index SET vector=? WHERE claim_id=?",
                (bytes(12), old))
        self.assertEqual(memory.retrieve(query)[0]["claim_id"], old)
        self.assertEqual(embedder.passage_calls, 2)

        new_source = self.graph.record_node(
            "utterance", {"text": "audio progress announcements"}, actor="user")
        new = memory.propose(
            subject="Pulash", predicate="progress_style",
            object_value="audio progress announcements", scope="user_preference",
            evidence_class="user_explicit", source_node_ids=[new_source],
            confidence=1.0, retention_reason="explicit correction")
        memory.evaluate(new)

        with self.graph._connect() as conn:
            old_vector = conn.execute(
                "SELECT 1 FROM memory_embedding_index WHERE claim_id=?", (old,)
            ).fetchone()
        self.assertIsNone(old_vector)
        self.assertEqual(memory.retrieve(query)[0]["claim_id"], new)

    def test_semantic_provider_failure_degrades_to_lexical_retrieval(self):
        memory = MemoryCurator(self.graph, embedder=FakeEmbedder(fail=True))
        source = self.graph.record_node(
            "utterance", {"text": "visible progress"}, actor="user")
        claim_id = memory.propose(
            subject="Pulash", predicate="progress_style",
            object_value="visible progress", scope="user_preference",
            evidence_class="user_explicit", source_node_ids=[source],
            confidence=1.0, retention_reason="explicit preference")
        memory.evaluate(claim_id)

        self.assertEqual(memory.retrieve("visible progress")[0]["claim_id"],
                         claim_id)
        self.assertEqual(memory.retrieve("¿Cómo debo informarte?"), [])

    def test_new_user_preference_supersedes_old_without_deleting_it(self):
        old_source = self.graph.record_node(
            "utterance", {"text": "Use long answers."}, actor="user")
        old_id = self.memory.propose(
            subject="Pulash", predicate="answer_style", object_value="long",
            scope="user_preference", evidence_class="user_explicit",
            source_node_ids=[old_source], confidence=1.0,
            retention_reason="explicit preference")
        self.memory.evaluate(old_id)

        new_source = self.graph.record_node(
            "utterance", {"text": "Use short answers instead."}, actor="user")
        new_id = self.memory.propose(
            subject="Pulash", predicate="answer_style", object_value="short",
            scope="user_preference", evidence_class="user_explicit",
            source_node_ids=[new_source], confidence=1.0,
            retention_reason="explicit correction")
        self.memory.evaluate(new_id)

        with self.graph._connect() as conn:
            old_state = conn.execute(
                "SELECT lifecycle FROM claim_state WHERE claim_id=?", (old_id,)
            ).fetchone()[0]
            edge = conn.execute(
                """SELECT 1 FROM edges WHERE from_node_id=? AND relation='supersedes'
                   AND to_node_id=?""", (new_id, old_id)
            ).fetchone()
        self.assertEqual(old_state, "superseded")
        self.assertIsNotNone(self.graph.get_node(old_id))
        self.assertIsNotNone(edge)
        self.assertEqual(self.memory.retrieve("answer style short")[0]["claim_id"],
                         new_id)


if __name__ == "__main__":
    unittest.main()
