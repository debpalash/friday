"""Deterministic promotion of repeated verified experience into active skills."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any

from .reflection import ReflectionService
from .skills import SkillManager
from .tasks import TaskService


class EvolutionEngine:
    _CHANGE_INTENT = re.compile(
        r"\b(?:activate|add|build|change|create|edit|implement|modify|pick|set|"
        r"switch|update|upgrade|use|wire)\b", re.IGNORECASE)
    _EFFECT_TOOLS = {
        "create_capability", "create_skill", "create_voice_profile", "restart",
        "rollback_voice", "set_voice", "upgrade_core", "write_file",
    }
    def __init__(self, tasks: TaskService, reflections: ReflectionService,
                 skills: SkillManager):
        self.tasks = tasks
        self.graph = tasks.graph
        self.reflections = reflections
        self.skills = skills

    @staticmethod
    def _productive(actions: list[dict[str, Any]]) -> bool:
        """Reject technically successful receipts that contain no useful evidence."""
        empty_markers = {
            "", "[]", "{}", "null", "none", "(empty)",
            "(no verified memories found)",
        }
        return bool(actions) and all(
            action["status"] == "succeeded"
            and str(action.get("result", "")).strip().lower() not in empty_markers
            for action in actions
        )

    @classmethod
    def _workflow_satisfies_objective(cls, objective: str,
                                      tools: tuple[str, ...]) -> bool:
        if not cls._CHANGE_INTENT.search(objective):
            return True
        return bool(set(tools) & cls._EFFECT_TOOLS)

    def run_once(self, *, minimum_successes: int = 2) -> dict[str, int]:
        reflected = 0
        workflows: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        with self.graph._connect() as conn:
            completed = conn.execute(
                """SELECT task_id,objective FROM task_state
                   WHERE status='completed' AND
                         (contract_version=0 OR verification_status='passed')"""
            ).fetchall()
        for task in completed:
            actions = self.tasks.action_history(task["task_id"])
            if not self._productive(actions):
                continue
            tools = tuple(action["tool_name"] for action in actions
                          if action["status"] == "succeeded")
            if not tools or len(tools) != len(actions):
                continue
            if not self._workflow_satisfies_objective(task["objective"], tools):
                continue
            before = self.graph.count_nodes("reflection")
            self.reflections.record(
                task["task_id"],
                f"Verified task used workflow: {' -> '.join(tools)}.", [])
            reflected += int(self.graph.count_nodes("reflection") > before)
            workflows[tools].append({"task_id": task["task_id"],
                                     "objective": task["objective"]})

        created = 0
        active_names = {skill["name"] for skill in self.skills.list()}
        for tools, examples in workflows.items():
            if len({item["task_id"] for item in examples}) < minimum_successes:
                continue
            digest = hashlib.sha256("\0".join(tools).encode()).hexdigest()[:10]
            name = f"workflow-{digest}"
            if name in active_names:
                continue
            objectives = [item["objective"] for item in examples[:3]]
            instructions = (
                "For a relevant request, use this historically successful tool sequence: "
                f"{' -> '.join(tools)}. Verify every receipt and adapt arguments to the "
                "current request; never reuse historical outputs as current evidence. "
                f"Example objectives: {json.dumps(objectives)}")
            tests = [{"name": f"historical task {item['task_id']}",
                      "passed": True, "evidence_task_id": item["task_id"]}
                     for item in examples]
            version = self.skills.create_version(
                name, instructions,
                {"permissions": sorted(set(tools)),
                 "trigger": "semantic match to linked example objectives",
                 "example_objectives": objectives,
                 "promotion_rule": f"{minimum_successes} verified independent tasks"},
                tests, source_node_ids=[item["task_id"] for item in examples])
            if self.skills.evaluate(version, tests):
                self.skills.activate(version)
                created += 1
        return {"reflections_created": reflected, "skills_activated": created}
