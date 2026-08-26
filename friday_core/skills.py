"""Immutable, test-gated skill versions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .graph import GraphStore, canonical_json, new_id, utc_now


_RETRIEVAL_STOPWORDS = {
    "about", "active", "adapt", "after", "again", "anything", "arguments",
    "before", "being", "current", "every", "example", "from", "have",
    "historical", "into", "just", "linked", "never", "objectives", "only",
    "outputs", "permissions", "relevant", "request", "sequence", "should",
    "that", "their", "then", "this", "tool", "tools", "trigger", "use",
    "used", "using", "verified", "verify", "what", "when", "where", "which",
    "with", "would", "your",
}


def _retrieval_terms(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in _RETRIEVAL_STOPWORDS
    }


class SkillManager:
    def __init__(self, graph: GraphStore, root: str | Path):
        self.graph = graph
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slug(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if not slug:
            raise ValueError("skill name has no usable characters")
        return slug

    def create_version(self, name: str, instructions: str,
                       manifest: dict[str, Any], tests: list[dict[str, Any]], *,
                       source_node_ids: list[str], actor: str = "friday") -> str:
        if not instructions.strip() or not tests:
            raise ValueError("a skill requires instructions and tests")
        if not source_node_ids or any(self.graph.get_node(x) is None
                                      for x in source_node_ids):
            raise ValueError("a skill requires valid provenance")
        slug = self._slug(name)
        now = utc_now()
        with self.graph.transaction() as conn:
            state = conn.execute("SELECT * FROM skill_state WHERE name=?",
                                 (slug,)).fetchone()
            if state:
                skill_id = state["skill_id"]
                version = int(conn.execute(
                    "SELECT COALESCE(MAX(version),0)+1 FROM skill_versions WHERE skill_id=?",
                    (skill_id,)).fetchone()[0])
            else:
                event_id, seq = self.graph.append_event(
                    conn, "skill.created", {"name": slug}, actor=actor)
                skill_id = self.graph.append_node(
                    conn, "skill", {"name": slug}, event_id=event_id,
                    node_id=new_id("skill"))
                conn.execute(
                    """INSERT INTO skill_state(skill_id,name,status,created_at,updated_at,
                       last_event_seq) VALUES (?,?,'proposed',?,?,?)""",
                    (skill_id, slug, now, now, seq))
                version = 1
            body = {"name": slug, "version": version, "instructions": instructions,
                    "manifest": manifest, "tests": tests}
            event_id, seq = self.graph.append_event(
                conn, "skill.version_drafted", body, actor=actor)
            version_id = self.graph.append_node(
                conn, "skill_version", body, event_id=event_id,
                node_id=new_id("skillv"))
            self.graph.append_edge(conn, skill_id, "contains", version_id,
                                   event_id=event_id)
            for source_id in source_node_ids:
                self.graph.append_edge(conn, version_id, "derived_from", source_id,
                                       event_id=event_id)
            conn.execute(
                """INSERT INTO skill_versions(version_id,skill_id,version,instructions,
                   manifest_json,tests_json,status,created_at,last_event_seq)
                   VALUES (?,?,?,?,?,?,'drafted',?,?)""",
                (version_id, skill_id, version, instructions,
                 canonical_json(manifest), canonical_json(tests), now, seq))
            conn.execute(
                "UPDATE skill_state SET status='drafted',updated_at=?,last_event_seq=? "
                "WHERE skill_id=?", (now, seq, skill_id))
        version_dir = self.root / slug / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=False)
        (version_dir / "SKILL.md").write_text(instructions)
        (version_dir / "manifest.json").write_text(
            json.dumps(manifest | {"name": slug, "version": version}, indent=2) + "\n")
        (version_dir / "tests.json").write_text(json.dumps(tests, indent=2) + "\n")
        return version_id

    def evaluate(self, version_id: str, results: list[dict[str, Any]], *,
                 actor: str = "verifier") -> bool:
        passed = bool(results) and all(result.get("passed") is True for result in results)
        with self.graph.transaction() as conn:
            version = conn.execute("SELECT * FROM skill_versions WHERE version_id=?",
                                   (version_id,)).fetchone()
            if version is None:
                raise ValueError("skill version does not exist")
            body = {"version_id": version_id, "passed": passed, "results": results}
            event_id, seq = self.graph.append_event(
                conn, "skill.evaluated", body, actor=actor)
            evaluation_id = self.graph.append_node(conn, "evaluation", body,
                                                   event_id=event_id)
            self.graph.append_edge(conn, version_id, "verified_by", evaluation_id,
                                   event_id=event_id)
            status = "validated" if passed else "quarantined"
            conn.execute("UPDATE skill_versions SET status=?,last_event_seq=? "
                         "WHERE version_id=?", (status, seq, version_id))
            conn.execute("UPDATE skill_state SET status=?,updated_at=?,last_event_seq=? "
                         "WHERE skill_id=?", (status, utc_now(), seq,
                                              version["skill_id"]))
        return passed

    def activate(self, version_id: str, *, actor: str = "friday") -> None:
        with self.graph.transaction() as conn:
            version = conn.execute("SELECT * FROM skill_versions WHERE version_id=?",
                                   (version_id,)).fetchone()
            if version is None or version["status"] != "validated":
                raise ValueError("only a validated skill version can be activated")
            event_id, seq = self.graph.append_event(
                conn, "skill.activated", {"version_id": version_id}, actor=actor)
            conn.execute("UPDATE skill_versions SET status='active',last_event_seq=? "
                         "WHERE version_id=?", (seq, version_id))
            conn.execute(
                """UPDATE skill_state SET status='active',active_version_id=?,updated_at=?,
                   last_event_seq=? WHERE skill_id=?""",
                (version_id, utc_now(), seq, version["skill_id"]))
            self.graph.append_edge(conn, version["skill_id"], "activated_as", version_id,
                                   event_id=event_id)

    def quarantine(self, version_id: str, reason: str, *,
                   actor: str = "verifier") -> None:
        """Deactivate a skill version while retaining its complete provenance."""
        if not reason.strip():
            raise ValueError("quarantine requires a reason")
        with self.graph.transaction() as conn:
            version = conn.execute("SELECT * FROM skill_versions WHERE version_id=?",
                                   (version_id,)).fetchone()
            if version is None:
                raise ValueError("skill version does not exist")
            _event_id, seq = self.graph.append_event(
                conn, "skill.quarantined",
                {"version_id": version_id, "reason": reason}, actor=actor)
            conn.execute("UPDATE skill_versions SET status='quarantined',last_event_seq=? "
                         "WHERE version_id=?", (seq, version_id))
            conn.execute(
                """UPDATE skill_state SET status='quarantined',active_version_id=NULL,
                   updated_at=?,last_event_seq=? WHERE skill_id=?""",
                (utc_now(), seq, version["skill_id"]))

    def list(self) -> list[dict[str, Any]]:
        with self.graph._connect() as conn:
            rows = conn.execute("SELECT * FROM skill_state ORDER BY name").fetchall()
        return [dict(row) for row in rows]

    def active_context(self, limit: int = 8, *,
                       available_tools: set[str] | None = None) -> list[dict[str, Any]]:
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT s.name,v.version_id,v.instructions,v.manifest_json
                   FROM skill_state s JOIN skill_versions v
                     ON v.version_id=s.active_version_id
                   WHERE s.status='active' ORDER BY s.updated_at DESC LIMIT ?""",
                (limit,)).fetchall()
        skills = [dict(row) | {"manifest": json.loads(row["manifest_json"])}
                  for row in rows]
        if available_tools is None:
            return skills
        return [skill for skill in skills
                if set(skill["manifest"].get("permissions", [])) <= available_tools]

    def relevant_context(self, query: str, limit: int = 8, *,
                         available_tools: set[str] | None = None
                         ) -> list[dict[str, Any]]:
        """Return only active skills with a lexical reason to match this request.

        Skills are optional hints, not global personality instructions. A zero-score
        skill is therefore safer to omit than to inject into an unrelated turn.
        """
        query_terms = _retrieval_terms(query)
        if not query_terms:
            return []
        candidates = self.active_context(
            max(limit * 4, limit), available_tools=available_tools)
        ranked: list[tuple[int, dict[str, Any]]] = []
        for skill in candidates:
            manifest = skill["manifest"]
            searchable = " ".join([
                skill["name"], skill["instructions"],
                str(manifest.get("trigger", "")),
                " ".join(str(item) for item in
                         manifest.get("example_objectives", [])),
            ])
            score = len(query_terms & _retrieval_terms(searchable))
            if score:
                ranked.append((score, skill))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [skill for _score, skill in ranked[:limit]]
