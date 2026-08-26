"""Artifact-backed, exact-graded long-term-memory evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from .graph import GraphStore, utc_now
from .memory import MemoryCurator


MAX_MEMORY_SUITE_BYTES = 128_000
_EXPECTED = {
    "precision_at_one": {
        "top_is_target": True, "returned": 1, "precision_at_one": 1.0},
    "correction_propagation": {
        "new_retrieved": True, "old_retrieved": False,
        "one_active_version": True},
    "stale_rejection": {
        "retrieved_before_expiry": True, "retrieved_after_expiry": False,
        "past_claim_promoted": False},
    "actor_provenance": {
        "forged_promoted": False, "forged_retrieved": False,
        "user_promoted": True, "user_retrieved": True},
    "hostile_query": {
        "syntax_safe": True, "stop_words_empty": True,
        "oversized_rejected": True},
    "duplicate_refresh": {
        "latest_retrieved": True, "one_active_version": True},
    "morphology_recall": {
        "notification_recalled": True, "matched_progress": True},
}


class MemoryEvalRunner:
    """Run synthetic memory cases in isolated graphs and journal aggregates only."""

    def __init__(self, graph: GraphStore):
        self.graph = graph

    @staticmethod
    def _load_suite(suite_path: str | Path) -> tuple[dict[str, Any], str]:
        try:
            descriptor = os.open(
                Path(suite_path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if (not stat.S_ISREG(metadata.st_mode)
                        or not 2 <= metadata.st_size <= MAX_MEMORY_SUITE_BYTES):
                    raise ValueError(
                        "memory evaluation suite must be a bounded regular file")
                encoded = stream.read(MAX_MEMORY_SUITE_BYTES + 1)
        except OSError as exc:
            raise ValueError(
                "memory evaluation suite must be a bounded regular file") from exc
        if len(encoded) != metadata.st_size:
            raise ValueError("memory evaluation suite changed while being read")

        def reject_constant(_value: str):
            raise ValueError("memory evaluation suite contains a non-finite number")

        suite = json.loads(encoded.decode("utf-8"), parse_constant=reject_constant)
        if (not isinstance(suite, dict)
                or set(suite) != {"name", "version", "coverage", "cases"}
                or suite.get("name") != "friday-memory-retrieval"
                or suite.get("version") != 1
                or not isinstance(suite.get("coverage"), list)
                or not 1 <= len(suite["coverage"]) <= 16
                or any(not isinstance(item, str) or not 1 <= len(item) <= 80
                       for item in suite["coverage"])
                or len(set(suite["coverage"])) != len(suite["coverage"])
                or not isinstance(suite.get("cases"), list)
                or len(suite["cases"]) != len(_EXPECTED)):
            raise ValueError("memory evaluation suite metadata is invalid")
        seen: set[str] = set()
        for case in suite["cases"]:
            if (not isinstance(case, dict)
                    or set(case) != {"name", "scenario", "expected"}
                    or not isinstance(case["name"], str)
                    or not 1 <= len(case["name"]) <= 160
                    or case["name"] in seen
                    or case["scenario"] not in _EXPECTED
                    or case["expected"] != _EXPECTED[case["scenario"]]):
                raise ValueError("memory evaluation case metadata is invalid")
            seen.add(case["name"])
        if {case["scenario"] for case in suite["cases"]} != set(_EXPECTED):
            raise ValueError("memory evaluation suite coverage is incomplete")
        return suite, hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _remember(memory: MemoryCurator, graph: GraphStore, predicate: str,
                  value: str, *, actor: str = "user",
                  valid_until: str | None = None) -> tuple[str, bool]:
        source = graph.record_node(
            "utterance", {"text": value}, actor=actor)
        claim_id = memory.propose(
            subject="user", predicate=predicate, object_value=value,
            scope="user_preference", evidence_class="user_explicit",
            source_node_ids=[source], confidence=1.0,
            retention_reason="synthetic evaluation evidence",
            valid_until=valid_until)
        return claim_id, memory.evaluate(claim_id).promoted

    @classmethod
    def _scenario(cls, name: str, memory: MemoryCurator,
                  graph: GraphStore) -> dict[str, Any]:
        if name == "precision_at_one":
            cls._remember(
                memory, graph, "answer_style",
                "show progress through detailed answers")
            target, _ = cls._remember(
                memory, graph, "progress_style", "visible progress updates")
            cls._remember(
                memory, graph, "notification_color", "blue notifications")
            hits = memory.retrieve(
                "How should progress update notifications appear?", limit=3)
            relevant = [item for item in hits if item["claim_id"] == target]
            return {
                "top_is_target": bool(hits and hits[0]["claim_id"] == target),
                "returned": len(hits),
                "precision_at_one": float(bool(
                    hits and hits[0]["claim_id"] == target and relevant)),
            }
        if name == "correction_propagation":
            old, _ = cls._remember(memory, graph, "answer_style", "long answers")
            new, _ = cls._remember(memory, graph, "answer_style", "short answers")
            hits = memory.retrieve("answer style", limit=8)
            active = memory.list(lifecycle="active", limit=20)
            return {
                "new_retrieved": new in {item["claim_id"] for item in hits},
                "old_retrieved": old in {item["claim_id"] for item in hits},
                "one_active_version": len(active) == 1,
            }
        if name == "stale_rejection":
            active, _ = cls._remember(
                memory, graph, "temporary_style", "amber alerts",
                valid_until="2099-01-01T00:00:00.000000Z")
            _past, past_promoted = cls._remember(
                memory, graph, "expired_style", "violet alerts",
                valid_until="2000-01-01T00:00:00.000000Z")
            before = memory.retrieve(
                "amber", now="2029-01-01T00:00:00.000000Z")
            after = memory.retrieve(
                "amber", now="2100-01-01T00:00:00.000000Z")
            return {
                "retrieved_before_expiry": active in {
                    item["claim_id"] for item in before},
                "retrieved_after_expiry": bool(after),
                "past_claim_promoted": past_promoted,
            }
        if name == "actor_provenance":
            forged, forged_promoted = cls._remember(
                memory, graph, "language", "Latin", actor="assistant")
            valid, user_promoted = cls._remember(
                memory, graph, "progress_style", "visible progress")
            return {
                "forged_promoted": forged_promoted,
                "forged_retrieved": forged in {
                    item["claim_id"] for item in memory.retrieve("Latin")},
                "user_promoted": user_promoted,
                "user_retrieved": valid in {
                    item["claim_id"] for item in memory.retrieve("progress")},
            }
        if name == "hostile_query":
            cls._remember(memory, graph, "progress_style", "visible progress")
            try:
                syntax_safe = memory.retrieve('\" OR * NEAR( ) --') == []
            except Exception:
                syntax_safe = False
            try:
                oversized_rejected = False
                memory.retrieve("x" * 2_001)
            except ValueError:
                oversized_rejected = True
            return {
                "syntax_safe": syntax_safe,
                "stop_words_empty": memory.retrieve("what is it to me") == [],
                "oversized_rejected": oversized_rejected,
            }
        if name == "duplicate_refresh":
            _old, _ = cls._remember(
                memory, graph, "progress_style", "visible progress")
            latest, _ = cls._remember(
                memory, graph, "progress_style", "visible progress")
            hits = memory.retrieve("visible progress")
            return {
                "latest_retrieved": [item["claim_id"] for item in hits] == [latest],
                "one_active_version": len(memory.list(
                    lifecycle="active", limit=20)) == 1,
            }
        if name == "morphology_recall":
            target, _ = cls._remember(
                memory, graph, "notification_style", "visible progress notifications")
            hits = memory.retrieve("show progress notification")
            match = next(
                (item for item in hits if item["claim_id"] == target), None)
            return {
                "notification_recalled": match is not None,
                "matched_progress": bool(
                    match and "progress" in match["matched_terms"]),
            }
        raise ValueError("unknown memory evaluation scenario")

    def run(self, suite_path: str | Path) -> dict[str, Any]:
        suite, suite_sha256 = self._load_suite(suite_path)
        results: list[dict[str, Any]] = []
        for case in suite["cases"]:
            try:
                with tempfile.TemporaryDirectory(
                        prefix="friday-memory-eval-") as temporary:
                    graph = GraphStore(Path(temporary) / "scenario.db")
                    observed = self._scenario(
                        case["scenario"], MemoryCurator(graph), graph)
                passed = observed == case["expected"]
                failure = None
            except Exception as exc:
                observed = {}
                passed = False
                failure = type(exc).__name__
            result = {
                "name": case["name"], "scenario": case["scenario"],
                "passed": passed,
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
            "coverage": list(suite["coverage"]),
            "passed": passed, "total": len(results),
            "pass_rate": passed / len(results), "results": results,
            "ran_at": utc_now(),
        }
        if not math.isfinite(body["pass_rate"]):
            raise RuntimeError("memory evaluation produced a non-finite score")
        run_id = self.graph.record_node(
            "memory_evaluation_run", body, actor="memory_eval_runner",
            event_type="evaluation.memory_completed")
        return {"evaluation_run_id": run_id, **body}
