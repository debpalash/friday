"""Held-out quality, correction-lift, and scale evidence for semantic memory."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .graph import GraphStore, canonical_json, utc_now
from .memory import MAX_SEMANTIC_INDEX_CLAIMS, MemoryCurator


MAX_SCALE_SUITE_BYTES = 256_000
LEGACY_SEMANTIC_SCAN_LIMIT = 4_096
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 3)


def _bounded_text(value: Any, field: str, *, maximum: int = 1_000) -> str:
    if (not isinstance(value, str) or not 1 <= len(value) <= maximum
            or any(ord(character) < 32 and character not in "\t\n\r"
                   for character in value)):
        raise ValueError(f"semantic-scale {field} is invalid")
    return value


class SemanticScaleEvalRunner:
    def __init__(self, graph: GraphStore, embedder: Any):
        self.graph = graph
        self.embedder = embedder
        if (re.fullmatch(r"[0-9a-f]{64}", str(
                getattr(embedder, "fingerprint", ""))) is None
                or isinstance(getattr(embedder, "dimension", None), bool)
                or not isinstance(getattr(embedder, "dimension", None), int)
                or not 1 <= embedder.dimension <= 4_096):
            raise ValueError("semantic-scale provider identity is invalid")

    @staticmethod
    def _load_suite(path: str | Path) -> tuple[dict[str, Any], str]:
        try:
            descriptor = os.open(
                Path(path), os.O_RDONLY | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise ValueError("semantic-scale suite is unavailable") from exc
        try:
            metadata = os.fstat(descriptor)
            if (not stat.S_ISREG(metadata.st_mode)
                    or not 2 <= metadata.st_size <= MAX_SCALE_SUITE_BYTES):
                raise ValueError("semantic-scale suite must be a bounded regular file")
            encoded = os.read(descriptor, MAX_SCALE_SUITE_BYTES + 1)
            if len(encoded) != metadata.st_size:
                raise ValueError("semantic-scale suite changed while being read")
        finally:
            os.close(descriptor)
        try:
            suite = json.loads(
                encoded.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite value: {value}")),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("semantic-scale suite is invalid JSON") from exc
        if (not isinstance(suite, dict)
                or set(suite) != {
                    "name", "version", "gates", "scale_sizes", "memories",
                    "negative_queries", "corrections",
                }
                or suite.get("name") != "friday-semantic-scale"
                or suite.get("version") != 1):
            raise ValueError("semantic-scale suite metadata is invalid")
        SemanticScaleEvalRunner._validate_gates(suite["gates"])
        SemanticScaleEvalRunner._validate_scale_sizes(
            suite["scale_sizes"], suite["gates"])
        SemanticScaleEvalRunner._validate_memories(suite["memories"])
        SemanticScaleEvalRunner._validate_negative_queries(
            suite["negative_queries"])
        SemanticScaleEvalRunner._validate_corrections(suite["corrections"])
        return suite, hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _validate_gates(value: Any) -> None:
        if not isinstance(value, dict) or set(value) != {
            "precision", "recall", "abstention", "corrected_task_lift",
            "warm_p95_ms", "maximum_corpus_minimum",
        }:
            raise ValueError("semantic-scale gates are invalid")
        for field in (
                "precision", "recall", "abstention", "corrected_task_lift"):
            item = value[field]
            if (isinstance(item, bool) or not isinstance(item, (int, float))
                    or not math.isfinite(float(item)) or not 0 <= item <= 1):
                raise ValueError("semantic-scale quality gate is invalid")
        latency = value["warm_p95_ms"]
        corpus = value["maximum_corpus_minimum"]
        if (isinstance(latency, bool) or not isinstance(latency, (int, float))
                or not math.isfinite(float(latency)) or not 1 <= latency <= 10_000
                or isinstance(corpus, bool) or not isinstance(corpus, int)
                or not 16 <= corpus <= MAX_SEMANTIC_INDEX_CLAIMS):
            raise ValueError("semantic-scale capacity gate is invalid")

    @staticmethod
    def _validate_scale_sizes(value: Any, gates: dict[str, Any]) -> None:
        if (not isinstance(value, list) or not 2 <= len(value) <= 5
                or any(isinstance(item, bool) or not isinstance(item, int)
                       for item in value)
                or value != sorted(set(value))
                or value[0] < 16
                or value[-1] > MAX_SEMANTIC_INDEX_CLAIMS
                or value[-1] < gates["maximum_corpus_minimum"]):
            raise ValueError("semantic-scale corpus sizes are invalid")

    @staticmethod
    def _validate_memories(value: Any) -> None:
        if not isinstance(value, list) or not 4 <= len(value) <= 64:
            raise ValueError("semantic-scale memories are invalid")
        identifiers: set[str] = set()
        predicates: set[str] = set()
        query_count = 0
        for item in value:
            if (not isinstance(item, dict)
                    or set(item) != {"id", "predicate", "value", "queries"}
                    or _IDENTIFIER.fullmatch(str(item.get("id", ""))) is None
                    or item["id"] in identifiers
                    or _IDENTIFIER.fullmatch(str(
                        item.get("predicate", ""))) is None
                    or item["predicate"] in predicates
                    or not isinstance(item.get("queries"), list)
                    or not 1 <= len(item["queries"]) <= 8):
                raise ValueError("semantic-scale memory case is invalid")
            _bounded_text(item["value"], "memory value")
            for query in item["queries"]:
                _bounded_text(query, "positive query")
            identifiers.add(item["id"])
            predicates.add(item["predicate"])
            query_count += len(item["queries"])
        if query_count < 8:
            raise ValueError("semantic-scale positive coverage is incomplete")

    @staticmethod
    def _validate_negative_queries(value: Any) -> None:
        if (not isinstance(value, list) or not 4 <= len(value) <= 64
                or len(set(value)) != len(value)):
            raise ValueError("semantic-scale negative queries are invalid")
        for query in value:
            _bounded_text(query, "negative query")

    @staticmethod
    def _validate_corrections(value: Any) -> None:
        if not isinstance(value, list) or not 2 <= len(value) <= 16:
            raise ValueError("semantic-scale corrections are invalid")
        identifiers: set[str] = set()
        for item in value:
            if (not isinstance(item, dict)
                    or set(item) != {
                        "id", "predicate", "initial", "corrected", "query",
                    }
                    or _IDENTIFIER.fullmatch(str(item.get("id", ""))) is None
                    or item["id"] in identifiers
                    or _IDENTIFIER.fullmatch(str(
                        item.get("predicate", ""))) is None
                    or item.get("initial") == item.get("corrected")):
                raise ValueError("semantic-scale correction case is invalid")
            for field in ("initial", "corrected", "query"):
                _bounded_text(item[field], f"correction {field}")
            identifiers.add(item["id"])

    @staticmethod
    def _remember(
        memory: MemoryCurator,
        graph: GraphStore,
        predicate: str,
        value: str,
    ) -> str:
        source = graph.record_node(
            "utterance", {"text": value}, actor="user")
        claim_id = memory.propose(
            subject="user",
            predicate=predicate,
            object_value=value,
            scope="user_preference",
            evidence_class="user_explicit",
            source_node_ids=[source],
            confidence=1.0,
            retention_reason="semantic-scale evaluation evidence",
        )
        if not memory.evaluate(claim_id).promoted:
            raise RuntimeError("semantic-scale fixture did not promote")
        return claim_id

    @staticmethod
    def _add_distractors(
        graph: GraphStore,
        *,
        first: int,
        count: int,
    ) -> None:
        for batch_start in range(first, first + count, 512):
            batch_end = min(first + count, batch_start + 512)
            with graph.transaction() as connection:
                for index in range(batch_start, batch_end):
                    claim_id = f"claim_scale_archive_{index:06d}"
                    value = (
                        f"catalog section {index:06d} stores synthetic marker "
                        f"{(index * 7919) % 104729:06d}")
                    event_id, sequence = graph.append_event(
                        connection,
                        "memory.semantic_scale_fixture",
                        {"claim_id": claim_id},
                        actor="semantic_scale_eval_runner",
                    )
                    graph.append_node(
                        connection,
                        "memory_claim",
                        {"synthetic": True},
                        event_id=event_id,
                        node_id=claim_id,
                    )
                    timestamp = utc_now()
                    connection.execute(
                        """INSERT INTO claim_state
                           (claim_id,subject,predicate,object_json,scope,lifecycle,
                            confidence,evidence_class,retention_reason,valid_until,
                            created_at,updated_at,last_event_seq)
                           VALUES (?,?,?,?,?,'active',1.0,'deterministic_test',?,
                                   NULL,?,?,?)""",
                        (
                            claim_id,
                            "synthetic_archive",
                            f"archive_label_{index:06d}",
                            canonical_json(value),
                            "evaluation_fixture",
                            "semantic-scale synthetic distractor",
                            timestamp,
                            timestamp,
                            sequence,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO memory_fts(claim_id,text) VALUES (?,?)",
                        (claim_id, f"synthetic_archive {value}"),
                    )

    def _scale_and_quality(
        self,
        suite: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        with tempfile.TemporaryDirectory(
                prefix="friday-semantic-scale-") as temporary:
            graph = GraphStore(Path(temporary) / "scale.db")
            memory = MemoryCurator(graph, embedder=self.embedder)
            targets = {
                item["id"]: self._remember(
                    memory, graph, item["predicate"], item["value"])
                for item in suite["memories"]
            }
            current_corpus = len(targets)
            checkpoints: list[dict[str, Any]] = []
            for target_size in suite["scale_sizes"]:
                if target_size < current_corpus:
                    raise RuntimeError("semantic-scale corpus is smaller than fixtures")
                self._add_distractors(
                    graph,
                    first=current_corpus,
                    count=target_size - current_corpus,
                )
                current_corpus = target_size
                canary = suite["memories"][0]
                started = time.perf_counter_ns()
                cold_hits = memory._semantic_rank(
                    canary["queries"][0], datetime.now(UTC), 3)
                cold_ms = (time.perf_counter_ns() - started) / 1_000_000
                warm_latencies: list[float] = []
                for item in suite["memories"][:4]:
                    started = time.perf_counter_ns()
                    memory.retrieve(item["queries"][0], limit=3)
                    warm_latencies.append(
                        (time.perf_counter_ns() - started) / 1_000_000)
                indexed = graph.count("memory_embedding_index")
                checkpoints.append({
                    "corpus": target_size,
                    "cold_index_ms": round(cold_ms, 3),
                    "warm_p50_ms": _percentile(warm_latencies, 0.50),
                    "warm_p95_ms": _percentile(warm_latencies, 0.95),
                    "indexed_claims": indexed,
                    "index_complete": indexed == target_size,
                    "oldest_target_first": bool(
                        cold_hits
                        and cold_hits[0]["claim_id"] == targets[canary["id"]]),
                })

            true_positive = 0
            false_positive = 0
            false_negative = 0
            latency_ms: list[float] = []
            positive_queries = 0
            for item in suite["memories"]:
                for query in item["queries"]:
                    positive_queries += 1
                    started = time.perf_counter_ns()
                    hits = memory.retrieve(query, limit=3)
                    latency_ms.append(
                        (time.perf_counter_ns() - started) / 1_000_000)
                    if hits and hits[0]["claim_id"] == targets[item["id"]]:
                        true_positive += 1
                    else:
                        false_negative += 1
                        if hits:
                            false_positive += 1
            negative_empty = 0
            for query in suite["negative_queries"]:
                started = time.perf_counter_ns()
                hits = memory.retrieve(query, limit=3)
                latency_ms.append(
                    (time.perf_counter_ns() - started) / 1_000_000)
                if hits:
                    false_positive += 1
                else:
                    negative_empty += 1
            precision_denominator = true_positive + false_positive
            quality = {
                "positive_queries": positive_queries,
                "negative_queries": len(suite["negative_queries"]),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": (
                    true_positive / precision_denominator
                    if precision_denominator else 0.0),
                "recall": true_positive / positive_queries,
                "abstention": negative_empty / len(suite["negative_queries"]),
                "warm_p50_ms": _percentile(latency_ms, 0.50),
                "warm_p95_ms": _percentile(latency_ms, 0.95),
            }
            return checkpoints, quality

    def _correction_lift(self, suite: dict[str, Any]) -> dict[str, Any]:
        before_success = 0
        after_success = 0
        old_reappeared = 0
        vector_retained = 0
        for item in suite["corrections"]:
            with tempfile.TemporaryDirectory(
                    prefix="friday-semantic-correction-") as temporary:
                graph = GraphStore(Path(temporary) / "correction.db")
                memory = MemoryCurator(graph, embedder=self.embedder)
                old = self._remember(
                    memory, graph, item["predicate"], item["initial"])
                before = memory.retrieve(item["query"], limit=3)
                if (before and before[0]["object"] == item["corrected"]):
                    before_success += 1
                new = self._remember(
                    memory, graph, item["predicate"], item["corrected"])
                after = memory.retrieve(item["query"], limit=3)
                if (after and after[0]["claim_id"] == new
                        and after[0]["object"] == item["corrected"]):
                    after_success += 1
                if old in {hit["claim_id"] for hit in after}:
                    old_reappeared += 1
                with graph._connect() as connection:
                    if connection.execute(
                        "SELECT 1 FROM memory_embedding_index WHERE claim_id=?",
                        (old,),
                    ).fetchone() is not None:
                        vector_retained += 1
        total = len(suite["corrections"])
        before_rate = before_success / total
        after_rate = after_success / total
        return {
            "tasks": total,
            "before_success": before_success,
            "after_success": after_success,
            "before_rate": before_rate,
            "after_rate": after_rate,
            "corrected_task_lift": after_rate - before_rate,
            "old_reappeared": old_reappeared,
            "superseded_vectors_retained": vector_retained,
        }

    def run(self, suite_path: str | Path) -> dict[str, Any]:
        suite, suite_sha256 = self._load_suite(suite_path)
        checkpoints, quality = self._scale_and_quality(suite)
        correction = self._correction_lift(suite)
        gates = suite["gates"]
        checks = {
            "precision": quality["precision"] >= gates["precision"],
            "recall": quality["recall"] >= gates["recall"],
            "abstention": quality["abstention"] >= gates["abstention"],
            "warm_p95": quality["warm_p95_ms"] <= gates["warm_p95_ms"],
            "correction_lift": (
                correction["corrected_task_lift"]
                >= gates["corrected_task_lift"]),
            "correction_retirement": (
                correction["old_reappeared"] == 0
                and correction["superseded_vectors_retained"] == 0),
            "index_complete": all(
                checkpoint["index_complete"] for checkpoint in checkpoints),
            "oldest_target_recall": all(
                checkpoint["oldest_target_first"] for checkpoint in checkpoints),
            "beyond_legacy_limit": (
                checkpoints[-1]["corpus"] > LEGACY_SEMANTIC_SCAN_LIMIT),
        }
        decision = {
            "index_strategy": "sharded_exact",
            "legacy_scan_replaced": True,
            "ann_required": not checks["warm_p95"],
            "reason": (
                "the legacy claim cap failed the measured corpus requirement; "
                "the complete sharded exact index meets the warm-latency gate"
                if checks["warm_p95"] else
                "the complete sharded exact index exceeds the warm-latency gate"),
        }
        body = {
            "suite": suite["name"],
            "version": suite["version"],
            "suite_sha256": suite_sha256,
            "embedding_fingerprint": self.embedder.fingerprint,
            "embedding_dimension": self.embedder.dimension,
            "scale": checkpoints,
            "quality": quality,
            "correction": correction,
            "gates": gates,
            "checks": checks,
            "decision": decision,
            "passed": all(checks.values()),
            "ran_at": utc_now(),
        }
        run_id = self.graph.record_node(
            "semantic_scale_evaluation_run",
            body,
            actor="semantic_scale_eval_runner",
            event_type="evaluation.semantic_scale_completed",
        )
        return {"evaluation_run_id": run_id, **body}
