"""Exact-graded local semantic-memory evaluation with disposable facts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

from .embeddings import LocalTextEmbedder
from .graph import GraphStore, utc_now
from .memory import MemoryCurator


MAX_SEMANTIC_SUITE_BYTES = 128_000
_EXPECTED = {
    "english_paraphrase": {
        "target_first": True, "semantic_mode": True,
        "no_lexical_match": True},
    "spanish_crosslingual": {
        "target_first": True, "semantic_mode": True},
    "hindi_crosslingual": {
        "target_first": True, "semantic_mode": True},
    "german_crosslingual": {
        "target_first": True, "semantic_mode": True},
    "irrelevant_abstention": {
        "weather_empty": True, "cuisine_empty": True, "ram_empty": True},
    "semantic_expiry": {
        "present_before": True, "absent_after": True},
    "semantic_correction": {
        "old_retrieved_before": True, "new_retrieved_after": True,
        "old_retrieved_after": False, "old_vector_removed": True},
    "bounded_large_corpus": {
        "target_first": True, "indexed_claims": 129,
        "cache_count_stable": True},
}


class SemanticMemoryEvalRunner:
    def __init__(self, graph: GraphStore, embedder: LocalTextEmbedder):
        self.graph = graph
        self.embedder = embedder
        if (not isinstance(getattr(embedder, "fingerprint", None), str)
                or len(embedder.fingerprint) != 64
                or not isinstance(getattr(embedder, "dimension", None), int)
                or not 1 <= embedder.dimension <= 4_096):
            raise ValueError("semantic evaluation provider identity is invalid")

    @staticmethod
    def _load_suite(suite_path: str | Path) -> tuple[dict[str, Any], str]:
        try:
            descriptor = os.open(
                Path(suite_path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if (not stat.S_ISREG(metadata.st_mode)
                        or not 2 <= metadata.st_size <= MAX_SEMANTIC_SUITE_BYTES):
                    raise ValueError(
                        "semantic-memory suite must be a bounded regular file")
                encoded = stream.read(MAX_SEMANTIC_SUITE_BYTES + 1)
        except OSError as exc:
            raise ValueError(
                "semantic-memory suite must be a bounded regular file") from exc
        if len(encoded) != metadata.st_size:
            raise ValueError("semantic-memory suite changed while being read")

        def reject_constant(_value: str):
            raise ValueError("semantic-memory suite contains a non-finite number")

        suite = json.loads(encoded.decode("utf-8"), parse_constant=reject_constant)
        if (not isinstance(suite, dict)
                or set(suite) != {"name", "version", "coverage", "cases"}
                or suite.get("name") != "friday-semantic-memory"
                or suite.get("version") != 1
                or not isinstance(suite.get("coverage"), list)
                or not 1 <= len(suite["coverage"]) <= 16
                or any(not isinstance(item, str) or not 1 <= len(item) <= 80
                       for item in suite["coverage"])
                or len(set(suite["coverage"])) != len(suite["coverage"])
                or not isinstance(suite.get("cases"), list)
                or len(suite["cases"]) != len(_EXPECTED)):
            raise ValueError("semantic-memory suite metadata is invalid")
        names: set[str] = set()
        for case in suite["cases"]:
            if (not isinstance(case, dict)
                    or set(case) != {"name", "scenario", "expected"}
                    or not isinstance(case["name"], str)
                    or not 1 <= len(case["name"]) <= 160
                    or case["name"] in names
                    or case["scenario"] not in _EXPECTED
                    or case["expected"] != _EXPECTED[case["scenario"]]):
                raise ValueError("semantic-memory case metadata is invalid")
            names.add(case["name"])
        if {case["scenario"] for case in suite["cases"]} != set(_EXPECTED):
            raise ValueError("semantic-memory suite coverage is incomplete")
        return suite, hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _remember(memory: MemoryCurator, graph: GraphStore, predicate: str,
                  value: str, *, valid_until: str | None = None) -> str:
        source = graph.record_node(
            "utterance", {"text": value}, actor="user")
        claim_id = memory.propose(
            subject="user", predicate=predicate, object_value=value,
            scope="user_preference", evidence_class="user_explicit",
            source_node_ids=[source], confidence=1.0,
            retention_reason="semantic evaluation evidence",
            valid_until=valid_until)
        if not memory.evaluate(claim_id).promoted:
            raise RuntimeError("synthetic semantic memory did not promote")
        return claim_id

    @classmethod
    def _base(cls, memory: MemoryCurator,
              graph: GraphStore) -> dict[str, str]:
        values = (
            ("progress_style", "keep progress visible with status updates"),
            ("answer_style", "give short direct answers"),
            ("interface_color", "use blue controls"),
            ("news_delivery", "give one concise spoken news summary"),
            ("voice_style", "use a calm young female voice"),
        )
        return {
            predicate: cls._remember(memory, graph, predicate, value)
            for predicate, value in values}

    @staticmethod
    def _top(memory: MemoryCurator, query: str, target: str,
             *, now: str | None = None) -> tuple[bool, bool, bool]:
        hits = memory.retrieve(query, limit=3, now=now)
        first = hits[0] if hits else None
        return (
            bool(first and first["claim_id"] == target),
            bool(first and first.get("retrieval_mode") == "semantic_fallback"),
            bool(first and not first.get("matched_terms")),
        )

    def _scenario(self, name: str, memory: MemoryCurator,
                  graph: GraphStore) -> dict[str, Any]:
        if name == "english_paraphrase":
            self._base(memory, graph)
            target = self._remember(
                memory, graph, "background_notice",
                "send a desktop notification when a long task finishes")
            first, semantic, no_lexical = self._top(
                memory, "Alert me upon completion of lengthy work.", target)
            return {"target_first": first, "semantic_mode": semantic,
                    "no_lexical_match": no_lexical}
        if name in {"spanish_crosslingual", "hindi_crosslingual",
                    "german_crosslingual"}:
            ids = self._base(memory, graph)
            query = {
                "spanish_crosslingual":
                    "¿Cómo debo mostrar las actualizaciones de progreso?",
                "hindi_crosslingual":
                    "मुझे प्रगति अपडेट कैसे दिखाने चाहिए?",
                "german_crosslingual":
                    "Wie soll ich Fortschrittsmeldungen anzeigen?",
            }[name]
            first, semantic, _no_lexical = self._top(
                memory, query, ids["progress_style"])
            return {"target_first": first, "semantic_mode": semantic}
        if name == "irrelevant_abstention":
            self._base(memory, graph)
            return {
                "weather_empty": not memory.retrieve(
                    "What is the weather forecast for tomorrow?"),
                "cuisine_empty": not memory.retrieve(
                    "Which cuisine should I order tonight?"),
                "ram_empty": not memory.retrieve("How much RAM is free?"),
            }
        if name == "semantic_expiry":
            self._base(memory, graph)
            target = self._remember(
                memory, graph, "background_notice",
                "send a desktop notification when a long task finishes",
                valid_until="2099-01-01T00:00:00.000000Z")
            query = "Alert me upon completion of lengthy work."
            before, _semantic, _ = self._top(
                memory, query, target,
                now="2098-01-01T00:00:00.000000Z")
            after = memory.retrieve(
                query, now="2100-01-01T00:00:00.000000Z")
            return {"present_before": before,
                    "absent_after": target not in {
                        item["claim_id"] for item in after}}
        if name == "semantic_correction":
            self._base(memory, graph)
            old = self._remember(
                memory, graph, "background_notice",
                "send a desktop notification when a long task finishes")
            old_query = "Alert me upon completion of lengthy work."
            old_before, _semantic, _ = self._top(memory, old_query, old)
            new = self._remember(
                memory, graph, "background_notice",
                "play a chime after extended work completes")
            hits = memory.retrieve(
                "Make a noise once a lengthy operation is done.")
            with graph._connect() as conn:
                old_vector = conn.execute(
                    "SELECT 1 FROM memory_embedding_index WHERE claim_id=?",
                    (old,)).fetchone()
            ids = {item["claim_id"] for item in hits}
            return {
                "old_retrieved_before": old_before,
                "new_retrieved_after": bool(hits and hits[0]["claim_id"] == new),
                "old_retrieved_after": old in ids,
                "old_vector_removed": old_vector is None,
            }
        if name == "bounded_large_corpus":
            target = self._remember(
                memory, graph, "background_notice",
                "send a desktop notification when a long task finishes")
            for index in range(128):
                self._remember(
                    memory, graph, f"archive_label_{index:03d}",
                    f"catalog section {index:03d} stores synthetic marker "
                    f"{(index * 7919) % 104729:06d}")
            query = "Alert me upon completion of lengthy work."
            first = memory.retrieve(query, limit=3)
            before = graph.count("memory_embedding_index")
            second = memory.retrieve(query, limit=3)
            after = graph.count("memory_embedding_index")
            return {
                "target_first": bool(
                    first and second and first[0]["claim_id"] == target
                    and second[0]["claim_id"] == target),
                "indexed_claims": before,
                "cache_count_stable": before == after,
            }
        raise ValueError("unknown semantic-memory scenario")

    def run(self, suite_path: str | Path) -> dict[str, Any]:
        suite, suite_sha256 = self._load_suite(suite_path)
        results: list[dict[str, Any]] = []
        for case in suite["cases"]:
            started = time.monotonic()
            try:
                with tempfile.TemporaryDirectory(
                        prefix="friday-semantic-memory-eval-") as temporary:
                    graph = GraphStore(Path(temporary) / "scenario.db")
                    observed = self._scenario(
                        case["scenario"],
                        MemoryCurator(graph, embedder=self.embedder), graph)
                passed = observed == case["expected"]
                failure = None
            except Exception as exc:
                observed = {}
                passed = False
                failure = type(exc).__name__
            result = {
                "name": case["name"], "scenario": case["scenario"],
                "passed": passed,
                "duration_ms": round((time.monotonic() - started) * 1_000),
                "observation_sha256": hashlib.sha256(
                    json.dumps(observed, sort_keys=True, separators=(",", ":"))
                    .encode("utf-8")).hexdigest(),
            }
            if failure:
                result["failure"] = failure
            results.append(result)
        passed = sum(int(item["passed"]) for item in results)
        body = {
            "suite": suite["name"], "version": suite["version"],
            "suite_sha256": suite_sha256,
            "embedding_fingerprint": self.embedder.fingerprint,
            "embedding_dimension": self.embedder.dimension,
            "embedding_batch_size": self.embedder.batch_size,
            "backend": "local_cpu_transformers",
            "coverage": list(suite["coverage"]),
            "passed": passed, "total": len(results),
            "pass_rate": passed / len(results), "results": results,
            "ran_at": utc_now(),
        }
        if not math.isfinite(body["pass_rate"]):
            raise RuntimeError("semantic-memory evaluation score is non-finite")
        run_id = self.graph.record_node(
            "semantic_memory_evaluation_run", body,
            actor="semantic_memory_eval_runner",
            event_type="evaluation.semantic_memory_completed")
        return {"evaluation_run_id": run_id, **body}
