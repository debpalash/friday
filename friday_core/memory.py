"""Evidence-gated memory promotion and retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from .graph import GraphStore, canonical_json, new_id, utc_now

if TYPE_CHECKING:
    from .embeddings import LocalTextEmbedder


@dataclass(frozen=True)
class PromotionDecision:
    claim_id: str
    promoted: bool
    lifecycle: str
    reason: str


MAX_MEMORY_TEXT_CHARS = 4_000
MAX_MEMORY_OBJECT_BYTES = 32_000
MAX_MEMORY_SOURCES = 32
MAX_RETRIEVAL_QUERY_CHARS = 2_000
MAX_RETRIEVAL_TERMS = 24
MAX_SEMANTIC_INDEX_CLAIMS = 65_536
SEMANTIC_EMBEDDING_REQUEST_SIZE = 512
SEMANTIC_SCORE_SHARD_SIZE = 1_024
MIN_SEMANTIC_SCORE = 0.775
MIN_SEMANTIC_CENTERED_MARGIN = 0.010
MIN_SEMANTIC_NULL_MARGIN = 0.005
SEMANTIC_NULL_PASSAGES = (
    "The request asks for current weather information, not a stored memory.",
    "The request asks for a sports result, not a stored memory.",
    "The request asks for a currency conversion or market price, not a stored "
    "memory.",
    "The request asks for a recipe or cooking instructions, not a stored memory.",
    "The request asks a geography fact, not a stored memory.",
    "The request asks for current computer status, not a stored memory.",
    "The request asks for travel search or booking, not a stored memory.",
    "The request asks for a scientific explanation of the natural world, not "
    "a stored memory.",
    "The request asks for current news or public information, not a stored memory.",
)
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "our",
    "please", "should", "that", "the", "their", "them", "this", "to", "us",
    "was", "we", "what", "when", "where", "which", "who", "with", "you",
    "your",
})


def _bounded_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"memory {field} must be text")
    value = unicodedata.normalize("NFKC", value).strip()
    if not 1 <= len(value) <= MAX_MEMORY_TEXT_CHARS or any(
            ord(character) < 32 and character not in "\t\n\r"
            for character in value):
        raise ValueError(f"memory {field} is invalid")
    return value


def _utc_timestamp(value: str, field: str) -> datetime:
    if not isinstance(value, str) or not 20 <= len(value) <= 40:
        raise ValueError(f"memory {field} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"memory {field} timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != datetime.min.replace(
            tzinfo=UTC).utcoffset():
        raise ValueError(f"memory {field} timestamp must be UTC")
    return parsed.astimezone(UTC)


def _stem(term: str) -> str:
    if len(term) > 5 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 5 and term.endswith("ing"):
        root = term[:-3]
        if len(root) > 3 and root[-1] == root[-2]:
            root = root[:-1]
        return root
    if len(term) > 4 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 4 and term.endswith(("sses", "xes", "zes", "ches", "shes")):
        return term[:-2]
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def _terms(value: str, *, limit: int = MAX_RETRIEVAL_TERMS) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    output: list[str] = []
    for raw in _WORD.findall(normalized):
        if len(raw) > 64 or raw in _STOP_WORDS:
            continue
        term = _stem(raw)
        if len(term) < 2 or term in output:
            continue
        output.append(term)
        if len(output) >= limit:
            break
    return output


def _object_json(value: Any) -> str:
    try:
        encoded = canonical_json(value)
        json.loads(
            encoded,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("memory object contains a non-finite number")))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("memory object must be bounded finite JSON") from exc
    if not 1 <= len(encoded.encode("utf-8")) <= MAX_MEMORY_OBJECT_BYTES:
        raise ValueError("memory object must be bounded finite JSON")
    return encoded


class MemoryCurator:
    IMMEDIATE_EVIDENCE = {"tool_observation", "deterministic_test"}

    def __init__(self, graph: GraphStore,
                 embedder: LocalTextEmbedder | None = None):
        self.graph = graph
        self.embedder = embedder
        self._semantic_null_vectors: np.ndarray | None = None

    def propose(
        self,
        *,
        subject: str,
        predicate: str,
        object_value: Any,
        scope: str,
        evidence_class: str,
        source_node_ids: list[str],
        confidence: float,
        retention_reason: str,
        actor: str = "curator",
        valid_until: str | None = None,
    ) -> str:
        subject = _bounded_text(subject, "subject")
        predicate = _bounded_text(predicate, "predicate")
        scope = _bounded_text(scope, "scope")
        evidence_class = _bounded_text(evidence_class, "evidence class")
        retention_reason = _bounded_text(retention_reason, "retention reason")
        object_json = _object_json(object_value)
        if (not isinstance(source_node_ids, list)
                or not 1 <= len(source_node_ids) <= MAX_MEMORY_SOURCES
                or len(set(source_node_ids)) != len(source_node_ids)
                or any(not isinstance(item, str) or not 1 <= len(item) <= 160
                       for item in source_node_ids)):
            raise ValueError("a memory candidate requires provenance")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("confidence must be between 0 and 1")
        if valid_until is not None:
            parsed_until = _utc_timestamp(valid_until, "valid-until")
            valid_until = parsed_until.isoformat(
                timespec="microseconds").replace("+00:00", "Z")
        sources = [self.graph.get_node(node_id) for node_id in source_node_ids]
        if any(source is None for source in sources):
            raise ValueError("memory source node does not exist")

        body = {
            "subject": subject,
            "predicate": predicate,
            "object": object_value,
            "scope": scope,
            "evidence_class": evidence_class,
            "confidence": confidence,
            "retention_reason": retention_reason,
            "valid_until": valid_until,
        }
        now = utc_now()
        with self.graph.transaction() as conn:
            event_id, seq = self.graph.append_event(
                conn, "memory.candidate_proposed", body, actor=actor)
            claim_id = self.graph.append_node(
                conn, "claim", body, event_id=event_id, node_id=new_id("claim"))
            for source_id in source_node_ids:
                self.graph.append_edge(conn, claim_id, "derived_from", source_id,
                                       event_id=event_id)
            conn.execute(
                """INSERT INTO claim_state
                   (claim_id, subject, predicate, object_json, scope, lifecycle,
                    confidence, evidence_class, retention_reason, valid_until,
                    created_at, updated_at, last_event_seq)
                   VALUES (?, ?, ?, ?, ?, 'candidate', ?, ?, ?, ?, ?, ?, ?)""",
                (claim_id, subject, predicate, object_json, scope,
                 confidence, evidence_class, retention_reason, valid_until,
                 now, now, seq),
            )
        return claim_id

    def evaluate(self, claim_id: str, *, actor: str = "curator") -> PromotionDecision:
        with self.graph.transaction() as conn:
            claim = conn.execute(
                "SELECT * FROM claim_state WHERE claim_id = ?", (claim_id,)
            ).fetchone()
            if claim is None:
                raise ValueError("claim does not exist")
            source_rows = conn.execute(
                """SELECT n.kind,g.actor FROM edges e
                   JOIN nodes n ON n.node_id=e.to_node_id
                   JOIN graph_events g ON g.event_id=n.created_event_id
                   WHERE e.from_node_id=? AND e.relation='derived_from'""",
                (claim_id,),
            ).fetchall()
            source_kinds = {row["kind"] for row in source_rows}

            evidence = claim["evidence_class"]
            scope = claim["scope"]
            promoted = False
            expired = False
            if claim["valid_until"] is not None:
                try:
                    expired = _utc_timestamp(
                        claim["valid_until"], "valid-until") <= _utc_timestamp(
                            utc_now(), "current")
                except ValueError:
                    expired = True
            user_utterances = [
                row for row in source_rows
                if row["kind"] == "utterance" and row["actor"] == "user"]
            if expired:
                reason = "evidence validity expired"
            elif source_kinds and source_kinds <= {"assistant_message"}:
                reason = "assistant output cannot validate its own claim"
            elif (evidence == "user_explicit" and scope == "user_preference"
                  and len(user_utterances) == len(source_rows) > 0):
                promoted = True
                reason = "explicit user preference"
            elif (evidence in self.IMMEDIATE_EVIDENCE
                  and source_kinds & {"observation", "evaluation", "artifact"}):
                promoted = True
                reason = f"supported by {evidence}"
            elif evidence == "assistant_utterance":
                reason = "assistant utterances are not evidence"
            else:
                reason = "requires corroboration"

            lifecycle = "active" if promoted else "candidate"
            payload = {"claim_id": claim_id, "promoted": promoted,
                       "lifecycle": lifecycle, "reason": reason}
            event_id, seq = self.graph.append_event(
                conn, "memory.promotion_evaluated", payload, actor=actor)
            evaluation_id = self.graph.append_node(
                conn, "evaluation", payload, event_id=event_id)
            conn.execute(
                """UPDATE claim_state SET lifecycle=?, updated_at=?, last_event_seq=?
                   WHERE claim_id=?""",
                (lifecycle, utc_now(), seq, claim_id),
            )
            if promoted:
                text = " ".join((claim["subject"], claim["predicate"],
                                 str(json.loads(claim["object_json"]))))
                conn.execute("DELETE FROM memory_fts WHERE claim_id = ?", (claim_id,))
                conn.execute("INSERT INTO memory_fts(claim_id, text) VALUES (?, ?)",
                             (claim_id, text))
                old_claims = conn.execute(
                    """SELECT claim_id, object_json FROM claim_state
                       WHERE subject=? AND predicate=? AND scope=?
                         AND lifecycle='active' AND claim_id<>?""",
                    (claim["subject"], claim["predicate"], scope, claim_id),
                ).fetchall()
                for old in old_claims:
                    self.graph.append_edge(conn, claim_id, "supersedes",
                                           old["claim_id"], event_id=event_id)
                    conn.execute(
                        """UPDATE claim_state SET lifecycle='superseded', updated_at=?,
                           last_event_seq=? WHERE claim_id=?""",
                        (utc_now(), seq, old["claim_id"]),
                    )
                    conn.execute("DELETE FROM memory_fts WHERE claim_id=?",
                                 (old["claim_id"],))
                    conn.execute(
                        "DELETE FROM memory_embedding_index WHERE claim_id=?",
                        (old["claim_id"],))
                self.graph.append_edge(conn, claim_id, "verified_by", evaluation_id,
                                       event_id=event_id,
                                       attributes={"decision": reason})
        return PromotionDecision(claim_id, promoted, lifecycle, reason)

    @staticmethod
    def _semantic_passage(row: Any, object_value: Any) -> str:
        value = (object_value if isinstance(object_value, str)
                 else canonical_json(object_value))
        return (
            f"The {row['scope']} memory for subject {row['subject']} has category "
            f"{row['predicate']}. Its verified value is: {value}")[:4_000]

    def _semantic_rank(self, query: str, current: datetime,
                       limit: int) -> list[dict[str, Any]]:
        embedder = self.embedder
        if embedder is None:
            return []
        fingerprint = str(getattr(embedder, "fingerprint", ""))
        dimension = getattr(embedder, "dimension", None)
        if (re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
                or isinstance(dimension, bool) or not isinstance(dimension, int)
                or not 1 <= dimension <= 4_096):
            raise RuntimeError("semantic memory provider identity is invalid")
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM claim_state
                   WHERE lifecycle='active'
                   ORDER BY updated_at DESC,claim_id
                   LIMIT ?""", (MAX_SEMANTIC_INDEX_CLAIMS + 1,)).fetchall()
        if len(rows) > MAX_SEMANTIC_INDEX_CLAIMS:
            raise RuntimeError("semantic memory index exceeds its bounded capacity")
        candidates: list[tuple[dict[str, Any], str, str]] = []
        missing: list[tuple[str, str, str]] = []
        for row in rows:
            try:
                if (row["valid_until"] is not None
                        and _utc_timestamp(row["valid_until"], "valid-until")
                        <= current):
                    continue
                object_value = json.loads(
                    row["object_json"], parse_constant=lambda _value: (
                        _ for _ in ()).throw(ValueError("non-finite object")))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            passage = self._semantic_passage(row, object_value)
            content_sha256 = hashlib.sha256(passage.encode("utf-8")).hexdigest()
            item = dict(row) | {"object": object_value}
            candidates.append((item, passage, content_sha256))
        if not candidates:
            return []

        cached: dict[str, Any] = {}
        candidate_ids = [item[0]["claim_id"] for item in candidates]
        with self.graph._connect() as conn:
            for offset in range(0, len(candidate_ids), 400):
                batch = candidate_ids[offset:offset + 400]
                placeholders = ",".join("?" for _ in batch)
                for record in conn.execute(
                    "SELECT claim_id,content_sha256,dimension,vector "
                    "FROM memory_embedding_index WHERE model_fingerprint=? "
                    f"AND claim_id IN ({placeholders})",
                    (fingerprint, *batch),
                ):
                    cached[record["claim_id"]] = record

        vectors: dict[str, np.ndarray] = {}
        for item, passage, content_sha256 in candidates:
            record = cached.get(item["claim_id"])
            try:
                vector = np.frombuffer(
                    record["vector"], dtype="<f4").copy() if record else None
                valid = bool(
                    record and record["content_sha256"] == content_sha256
                    and record["dimension"] == dimension
                    and vector is not None and vector.shape == (dimension,)
                    and np.isfinite(vector).all()
                    and 0.99 <= float(np.linalg.norm(vector)) <= 1.01)
            except (TypeError, ValueError):
                valid = False
            if valid:
                vectors[item["claim_id"]] = vector
            else:
                missing.append((item["claim_id"], passage, content_sha256))
        for offset in range(0, len(missing), SEMANTIC_EMBEDDING_REQUEST_SIZE):
            batch = missing[offset:offset + SEMANTIC_EMBEDDING_REQUEST_SIZE]
            encoded = embedder.encode(
                [item[1] for item in batch], kind="passage")
            if encoded.shape != (len(batch), dimension):
                raise RuntimeError("semantic memory index shape is invalid")
            prepared: list[tuple[str, str, bytes]] = []
            for (claim_id, _passage, content_sha256), vector in zip(
                    batch, encoded, strict=True):
                normalized = np.asarray(vector, dtype="<f4")
                if (normalized.shape != (dimension,)
                        or not np.isfinite(normalized).all()
                        or not 0.99 <= float(
                            np.linalg.norm(normalized)) <= 1.01):
                    raise RuntimeError("semantic memory vector is invalid")
                prepared.append((
                    claim_id, content_sha256, normalized.tobytes(order="C")))
                vectors[claim_id] = normalized
            with self.graph.transaction() as conn:
                conn.executemany(
                    """INSERT INTO memory_embedding_index
                       (claim_id,model_fingerprint,content_sha256,dimension,
                        vector,indexed_at) VALUES (?,?,?,?,?,?)
                       ON CONFLICT(claim_id,model_fingerprint) DO UPDATE SET
                         content_sha256=excluded.content_sha256,
                         dimension=excluded.dimension,
                         vector=excluded.vector,indexed_at=excluded.indexed_at""",
                    [
                        (claim_id, fingerprint, content_sha256, dimension,
                         vector, utc_now())
                        for claim_id, content_sha256, vector in prepared
                    ],
                )
        query_vector = embedder.encode([query], kind="query")[0]
        null_vectors = self._semantic_null_vectors
        if null_vectors is None:
            null_vectors = embedder.encode(
                SEMANTIC_NULL_PASSAGES, kind="passage")
            if (null_vectors.shape
                    != (len(SEMANTIC_NULL_PASSAGES), dimension)
                    or not np.isfinite(null_vectors).all()
                    or not np.allclose(
                        np.linalg.norm(null_vectors, axis=1), 1.0,
                        atol=0.01)):
                raise RuntimeError("semantic null calibration is invalid")
            self._semantic_null_vectors = np.asarray(
                null_vectors, dtype="<f4")
        null_score = float(np.max(null_vectors @ query_vector))

        ordered_vectors = [
            vectors[item[0]["claim_id"]]
            for item in candidates if item[0]["claim_id"] in vectors
        ]
        if not ordered_vectors:
            return []
        centroid = np.zeros(dimension, dtype=np.float64)
        for vector in ordered_vectors:
            centroid += vector
        centroid = (centroid / len(ordered_vectors)).astype("<f4")
        centered_query = query_vector - centroid
        centered_query_norm = float(np.linalg.norm(centered_query))
        use_centered = centered_query_norm > 1e-6 and len(ordered_vectors) > 1
        if use_centered:
            centered_query = centered_query / centered_query_norm

        scored: list[tuple[float, float, dict[str, Any]]] = []
        for offset in range(0, len(candidates), SEMANTIC_SCORE_SHARD_SIZE):
            shard = candidates[offset:offset + SEMANTIC_SCORE_SHARD_SIZE]
            present = [
                (item, vectors[item["claim_id"]])
                for item, _passage, _content_sha256 in shard
                if item["claim_id"] in vectors
            ]
            if not present:
                continue
            matrix = np.stack([vector for _item, vector in present])
            raw_scores = matrix @ query_vector
            if use_centered:
                centered = matrix - centroid
                norms = np.linalg.norm(centered, axis=1)
                valid = norms > 1e-6
                centered_scores = np.full(len(present), -1.0, dtype=np.float32)
                centered_scores[valid] = (
                    centered[valid] / norms[valid, None]) @ centered_query
            else:
                centered_scores = raw_scores
            for (item, _vector), centered_value, raw_value in zip(
                    present, centered_scores, raw_scores, strict=True):
                centered_score = float(centered_value)
                raw_score = float(raw_value)
                if math.isfinite(centered_score) and math.isfinite(raw_score):
                    scored.append((centered_score, raw_score, item))
        scored.sort(key=lambda value: (-value[0], value[2]["claim_id"]))
        if (not scored or scored[0][1] < MIN_SEMANTIC_SCORE
                or scored[0][1] - null_score < MIN_SEMANTIC_NULL_MARGIN):
            return []
        if (len(scored) > 1
                and scored[0][0] - scored[1][0]
                < MIN_SEMANTIC_CENTERED_MARGIN):
            return []
        selected = []
        cutoff = scored[0][0] - 0.01
        for centered_score, raw_score, item in scored:
            if centered_score < cutoff or len(selected) >= limit:
                break
            selected.append(item | {
                "semantic_score": round(raw_score, 6),
                "semantic_centered_score": round(centered_score, 6),
                "relevance_score": round(raw_score, 6),
                "matched_terms": [],
                "retrieval_mode": "semantic_fallback",
            })
        return selected

    def retrieve(self, query: str, *, limit: int = 8,
                 now: str | None = None) -> list[dict[str, Any]]:
        if (not isinstance(query, str)
                or len(query) > MAX_RETRIEVAL_QUERY_CHARS
                or isinstance(limit, bool) or not isinstance(limit, int)
                or not 1 <= limit <= 100):
            raise ValueError("memory retrieval request is invalid")
        query_terms = _terms(query)
        if not query_terms:
            return []
        current = _utc_timestamp(now or utc_now(), "current")
        fts_query = " OR ".join(f'"{term}"*' for term in query_terms)
        candidate_limit = min(256, max(32, limit * 12))
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT c.*,bm25(memory_fts) AS lexical_rank
                   FROM memory_fts f
                   JOIN claim_state c ON c.claim_id=f.claim_id
                   WHERE memory_fts MATCH ? AND c.lifecycle='active'
                   ORDER BY lexical_rank,c.confidence DESC,c.updated_at DESC
                   LIMIT ?""",
                (fts_query, candidate_limit),
            ).fetchall()
        ranked: list[dict[str, Any]] = []
        query_set = set(query_terms)
        for row in rows:
            try:
                if (row["valid_until"] is not None
                        and _utc_timestamp(row["valid_until"], "valid-until")
                        <= current):
                    continue
                object_value = json.loads(
                    row["object_json"], parse_constant=lambda _value: (
                        _ for _ in ()).throw(ValueError("non-finite object")))
                subject_terms = set(_terms(row["subject"], limit=64))
                predicate_terms = set(_terms(row["predicate"], limit=64))
                object_terms = set(_terms(str(object_value), limit=128))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            candidate_terms = subject_terms | predicate_terms | object_terms
            matched = query_set & candidate_terms
            if not matched:
                continue
            field_score = sum(
                3.0 if term in object_terms else
                2.0 if term in predicate_terms else 0.5
                for term in matched)
            coverage = len(matched) / len(query_set)
            score = field_score + coverage + float(row["confidence"]) * 0.25
            item = dict(row) | {
                "object": object_value,
                "relevance_score": round(score, 6),
                "matched_terms": sorted(matched),
            }
            item.pop("lexical_rank", None)
            ranked.append(item)
        ranked.sort(key=lambda item: (
            -item["relevance_score"], -float(item["confidence"]),
            item["claim_id"]))
        # A verbose query often contains one generic term shared by unrelated
        # memories. When the best candidate has multiple independent matches,
        # retain only candidates in the same relevance band instead of leaking
        # every one-token FTS hit into the model context.
        if (len(query_set) >= 3 and ranked
                and len(ranked[0]["matched_terms"]) >= 2):
            cutoff = ranked[0]["relevance_score"] * 0.60
            ranked = [
                item for item in ranked
                if item["relevance_score"] >= cutoff]
        semantic_needed = bool(
            self.embedder is not None
            and (not ranked or (len(query_set) >= 3
                                and len(ranked[0]["matched_terms"]) < 2)))
        if semantic_needed:
            try:
                semantic = self._semantic_rank(query, current, limit)
            except (OSError, RuntimeError, TypeError, ValueError):
                semantic = []
            if semantic:
                return semantic
            return []
        for item in ranked:
            item["retrieval_mode"] = "lexical"
        return ranked[:limit]

    def list(self, *, lifecycle: str | None = None,
             limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM claim_state"
        params: list[Any] = []
        if lifecycle:
            sql += " WHERE lifecycle=?"
            params.append(lifecycle)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self.graph._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) | {"object": json.loads(row["object_json"])}
                for row in rows]
