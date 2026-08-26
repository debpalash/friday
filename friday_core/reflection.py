"""Candidate-only reflection recording and deterministic consolidation."""

from __future__ import annotations

from typing import Any

from .graph import GraphStore


class ReflectionService:
    def __init__(self, graph: GraphStore):
        self.graph = graph

    def record(self, task_id: str, summary: str, lessons: list[str], *,
               actor: str = "reflector") -> str:
        task = self.graph.get_node(task_id)
        if task is None or task["kind"] != "task":
            raise ValueError("reflection requires a task")
        with self.graph._connect() as conn:
            existing = conn.execute(
                """SELECT n.node_id FROM nodes n JOIN edges e
                     ON e.from_node_id=n.node_id
                   WHERE n.kind='reflection' AND e.relation='derived_from'
                     AND e.to_node_id=? LIMIT 1""", (task_id,)).fetchone()
        if existing:
            return existing[0]
        body: dict[str, Any] = {
            "summary": summary.strip(),
            "lessons": [lesson.strip() for lesson in lessons if lesson.strip()],
            "lifecycle": "candidate",
            "knowledge_layer": "candidate",
        }
        reflection_id = self.graph.record_node(
            "reflection", body, actor=actor, task_id=task_id,
            event_type="reflection.recorded", links=[("derived_from", task_id)])
        for lesson in body["lessons"]:
            lesson_id = self.graph.record_node(
                "lesson", {"text": lesson, "lifecycle": "candidate",
                           "knowledge_layer": "candidate"},
                actor=actor, task_id=task_id,
                event_type="lesson.candidate_recorded",
                links=[("derived_from", reflection_id)])
        return reflection_id
