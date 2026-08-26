"""Versioned deterministic evaluation runner for Friday's cognitive kernel."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from .cognition import (ContractBuilder, IntentInterpreter, OutcomeVerifier,
                        Planner, PolicyEngine)
from .graph import GraphStore, utc_now
from .hardware import Accelerator, GIB, HardwareSnapshot, select_runtime_profile
from .memory import MemoryCurator
from .public_http import normalize_public_http_url
from .tasks import TaskService


MAX_EVAL_SUITE_BYTES = 256_000
MAX_EVAL_CASES = 256
_BASIC_KINDS = frozenset({"intent", "contract", "policy", "verification"})
_SCENARIOS = frozenset({
    "durable_readonly_restart",
    "false_completion_gate",
    "hardware_tensor_parallel",
    "memory_provenance",
    "nonrepeatable_reconciliation",
    "public_network_boundary",
})


class CognitiveEvalRunner:
    def __init__(self, graph: GraphStore):
        self.graph = graph
        self.intents = IntentInterpreter()
        self.contracts = ContractBuilder()
        self.policy = PolicyEngine()
        self.verifier = OutcomeVerifier()

    @staticmethod
    def _load_suite(suite_path: str | Path) -> dict[str, Any]:
        path = Path(suite_path)
        try:
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if (not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_size < 2
                        or metadata.st_size > MAX_EVAL_SUITE_BYTES):
                    raise ValueError(
                        "evaluation suite must be a bounded regular file")
                encoded = stream.read(MAX_EVAL_SUITE_BYTES + 1)
        except OSError as exc:
            raise ValueError(
                "evaluation suite must be a bounded regular file") from exc
        if len(encoded) != metadata.st_size:
            raise ValueError("evaluation suite changed while being read")

        def reject_constant(_value: str):
            raise ValueError("evaluation suite contains a non-finite number")

        suite = json.loads(encoded.decode("utf-8"),
                           parse_constant=reject_constant)
        if not isinstance(suite, dict):
            raise ValueError("evaluation suite must be an object")
        name = suite.get("name")
        version = suite.get("version")
        cases = suite.get("cases")
        coverage = suite.get("coverage", [])
        if (not isinstance(name, str) or not 1 <= len(name) <= 128
                or isinstance(version, bool) or not isinstance(version, int)
                or not 1 <= version <= 1_000_000
                or not isinstance(cases, list)
                or not 1 <= len(cases) <= MAX_EVAL_CASES
                or not isinstance(coverage, list)
                or len(coverage) > 64
                or any(not isinstance(item, str) or not 1 <= len(item) <= 80
                       for item in coverage)
                or len(set(coverage)) != len(coverage)):
            raise ValueError("evaluation suite metadata is invalid")
        seen: set[str] = set()
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("evaluation case must be an object")
            case_name = case.get("name")
            kind = case.get("kind")
            if (not isinstance(case_name, str)
                    or not 1 <= len(case_name) <= 160
                    or case_name in seen or "expected" not in case
                    or kind not in (_BASIC_KINDS | {"scenario"})):
                raise ValueError("evaluation case metadata is invalid")
            seen.add(case_name)
            if kind == "scenario" and case.get("scenario") not in _SCENARIOS:
                raise ValueError("evaluation scenario is not allowlisted")
            if kind == "scenario" and not isinstance(case["expected"], dict):
                raise ValueError("scenario expectation must be an object")
            if kind in {"intent", "contract"}:
                text = case.get("text")
                tools = case.get("tools", [])
                if (not isinstance(text, str) or not 1 <= len(text) <= 4_000
                        or not isinstance(tools, list) or len(tools) > 64
                        or any(not isinstance(tool, str)
                               or not 1 <= len(tool) <= 128 for tool in tools)
                        or kind == "contract" and not tools):
                    raise ValueError("cognitive evaluation input is invalid")
            elif kind == "policy":
                if (not isinstance(case.get("tool"), str)
                        or not 1 <= len(case["tool"]) <= 128
                        or not isinstance(
                            case.get("explicitly_requested", True), bool)):
                    raise ValueError("policy evaluation input is invalid")
            elif kind == "verification":
                if (not isinstance(case.get("tool"), str)
                        or not 1 <= len(case["tool"]) <= 128
                        or not isinstance(case.get("succeeded", True), bool)):
                    raise ValueError("verification evaluation input is invalid")
        return suite

    @staticmethod
    def _isolated_graph() -> tuple[tempfile.TemporaryDirectory, GraphStore]:
        temporary = tempfile.TemporaryDirectory(prefix="friday-eval-")
        return temporary, GraphStore(Path(temporary.name) / "evaluation.db")

    def _scenario_false_completion_gate(self, _case: dict[str, Any]) -> dict:
        temporary, graph = self._isolated_graph()
        try:
            tasks = TaskService(graph)
            contract = self.contracts.build("Inspect the project", ["list_files"])
            plan = Planner().build([{"name": "list_files"}], contract)
            task_id, _ = tasks.create(
                contract.objective, contract.model_dump(mode="json"))
            for state in ("interpreting",):
                tasks.transition(task_id, state)
            tasks.set_plan(task_id, plan)
            for state in ("planned", "running", "verifying"):
                tasks.transition(task_id, state)
            blocked = False
            try:
                tasks.transition(task_id, "completed")
            except ValueError:
                blocked = True
            return {
                "completion_blocked": blocked,
                "final_status": tasks.get(task_id)["status"],
                "verification_status": tasks.get(task_id)["verification_status"],
            }
        finally:
            temporary.cleanup()

    def _scenario_durable_readonly_restart(self, _case: dict[str, Any]) -> dict:
        temporary, graph = self._isolated_graph()
        try:
            tasks = TaskService(graph)
            task_id, _ = tasks.create(
                "Recover an exact two-step batch",
                {"version": 0, "evidence": "held-out restart scenario"})
            first_args = {"path": "/evaluation/first", "hidden": True}
            batch_id, steps = tasks.stage_step_batch(task_id, [{
                "tool_call_id": "eval-call-1", "tool_name": "list_files",
                "args": first_args, "idempotency_class": "read_only",
            }, {
                "tool_call_id": "eval-call-2", "tool_name": "list_files",
                "args": {"path": "/evaluation/second"},
                "idempotency_class": "read_only",
            }], round_index=0)
            stale = tasks.claim_next_step(batch_id, "eval-worker-before")
            recovered = tasks.recover_inflight_steps(
                force=True, dead_worker_id="eval-worker-before")
            retry = tasks.claim_next_step(batch_id, "eval-worker-after")
            stale_fenced = False
            try:
                tasks.finish_step(stale, {"entries": ["late"]}, succeeded=True)
            except PermissionError:
                stale_fenced = True
            same_dispatch = bool(retry and stale and (
                retry.step_id, retry.action_id, retry.idempotency_key, retry.args)
                == (stale.step_id, stale.action_id, stale.idempotency_key,
                    first_args))
            if retry is not None:
                tasks.finish_step(retry, {"entries": ["first"]}, succeeded=True)
            successor = tasks.claim_next_step(batch_id, "eval-worker-after")
            return {
                "recovered_retry_count": len(recovered["retry"]),
                "same_dispatch": same_dispatch,
                "retry_attempt": retry.attempt_number if retry else None,
                "stale_fenced": stale_fenced,
                "successor_ordered": bool(
                    successor and successor.step_id == steps[1]["step_id"]),
            }
        finally:
            temporary.cleanup()

    def _scenario_nonrepeatable_reconciliation(
            self, _case: dict[str, Any]) -> dict:
        temporary, graph = self._isolated_graph()
        try:
            tasks = TaskService(graph)
            task_id, _ = tasks.create(
                "Fence one nonrepeatable effect",
                {"version": 0, "evidence": "held-out recovery scenario"})
            batch_id, _ = tasks.stage_step_batch(task_id, [{
                "tool_call_id": "eval-send", "tool_name": "send_message",
                "args": {"message": "exactly once"},
                "idempotency_class": "non_repeatable",
            }], round_index=0)
            claim = tasks.claim_next_step(batch_id, "eval-dead-worker")
            recovered = tasks.recover_inflight_steps(
                force=True, dead_worker_id="eval-dead-worker")
            retry = tasks.claim_next_step(batch_id, "eval-new-worker")
            step = tasks.list_steps(batch_id=batch_id)[0]
            return {
                "retry_count": len(recovered["retry"]),
                "reconcile_count": len(recovered["reconcile"]),
                "redispatch_blocked": retry is None,
                "state": step["status"],
                "same_step": bool(
                    claim and recovered["reconcile"] == [claim.step_id]),
            }
        finally:
            temporary.cleanup()

    def _scenario_memory_provenance(self, _case: dict[str, Any]) -> dict:
        temporary, graph = self._isolated_graph()
        try:
            memory = MemoryCurator(graph)
            guessed_source = graph.record_node(
                "assistant_message", {"text": "The user prefers Latin."})
            guessed = memory.propose(
                subject="user", predicate="prefers", object_value="Latin",
                scope="user_preference", evidence_class="assistant_utterance",
                source_node_ids=[guessed_source], confidence=0.99,
                retention_reason="assistant inference")
            guessed_decision = memory.evaluate(guessed)
            user_source = graph.record_node(
                "utterance", {"text": "Keep progress visible."}, actor="user")
            verified = memory.propose(
                subject="user", predicate="prefers",
                object_value="visible progress", scope="user_preference",
                evidence_class="user_explicit", source_node_ids=[user_source],
                confidence=1.0, retention_reason="explicit preference")
            verified_decision = memory.evaluate(verified)
            return {
                "assistant_claim_promoted": guessed_decision.promoted,
                "assistant_claim_retrieved": bool(memory.retrieve("Latin")),
                "user_claim_promoted": verified_decision.promoted,
                "user_claim_retrieved": [
                    item["claim_id"] for item in memory.retrieve("progress")
                ] == [verified],
            }
        finally:
            temporary.cleanup()

    @staticmethod
    def _scenario_public_network_boundary(case: dict[str, Any]) -> dict:
        inputs = case.get("input", {})
        private_urls = inputs.get("private_urls", [])
        public_url = inputs.get("public_url")
        if (not isinstance(private_urls, list) or len(private_urls) > 16
                or any(not isinstance(item, str) for item in private_urls)
                or not isinstance(public_url, str)):
            raise ValueError("public boundary scenario input is invalid")
        blocked = 0
        for url in private_urls:
            try:
                normalize_public_http_url(url)
            except ValueError:
                blocked += 1
        normalized = normalize_public_http_url(public_url)
        return {
            "private_blocked": blocked,
            "private_total": len(private_urls),
            "public_normalized": normalized,
        }

    @staticmethod
    def _scenario_hardware_tensor_parallel(case: dict[str, Any]) -> dict:
        inputs = case.get("input", {})
        capacities = inputs.get("cuda_gib", [])
        selected = inputs.get("llm_devices")
        if (not isinstance(capacities, list) or not 1 <= len(capacities) <= 16
                or any(isinstance(item, bool) or not isinstance(item, int)
                       or not 8 <= item <= 1024 for item in capacities)
                or not isinstance(selected, str)):
            raise ValueError("hardware scenario input is invalid")
        snapshot = HardwareSnapshot(
            cpu_count=64, system_memory_bytes=128 * GIB,
            accelerators=tuple(Accelerator(
                "cuda", index, f"Evaluation GPU {index}", size * GIB,
                size * GIB) for index, size in enumerate(capacities)),
            cuda_probe="available")
        profile = select_runtime_profile(snapshot, environment={
            "FRIDAY_LLM_CUDA_DEVICES": selected,
        })
        return {
            "llm_devices": list(profile.effective_llm_cuda_devices),
            "tensor_parallel_size": profile.tensor_parallel_size,
            "tts_device": profile.tts_device,
            "tts_cuda_device": profile.tts_cuda_device,
            "per_rank_budget_gib": profile.llm_memory_budget_gib,
            "total_budget_gib": profile.to_dict()[
                "llm_total_memory_budget_gib"],
            "native_vision_enabled": profile.native_vision_enabled,
            "native_vision_max_images": profile.native_vision_max_images,
            "native_vision_max_side": profile.native_vision_max_side,
            "rank_remaining_mib": {
                f"cuda:{index}": profile.admission_budget[
                    "vram_mib_by_accelerator"][f"cuda:{index}"]
                for index in profile.effective_llm_cuda_devices
            },
        }

    def _run_scenario(self, case: dict[str, Any]) -> dict:
        name = case["scenario"]
        method = getattr(self, f"_scenario_{name}", None)
        if method is None:
            raise ValueError("evaluation scenario is not implemented")
        return method(case)

    def run(self, suite_path: str | Path) -> dict[str, Any]:
        suite = self._load_suite(suite_path)
        results = []
        for case in suite["cases"]:
            kind = case["kind"]
            expected = case["expected"]
            try:
                if kind == "intent":
                    actual = self.intents.interpret(
                        case["text"], case.get("tools", [])).value
                elif kind == "contract":
                    actual = self.contracts.build(
                        case["text"], case["tools"]).model_dump(mode="json")
                    actual = {key: actual[key] for key in expected}
                elif kind == "policy":
                    actual = self.policy.decide(
                        case["tool"], explicitly_requested=case.get(
                            "explicitly_requested", True),
                    ).model_dump(mode="json")
                    actual = {key: actual[key] for key in expected}
                elif kind == "verification":
                    check = self.verifier.verify_action(
                        case["tool"], json.dumps(case.get("result")),
                        succeeded=case.get("succeeded", True))
                    actual = {
                        "status": check.status.value,
                        "missing_evidence": bool(check.missing),
                    }
                else:
                    actual = self._run_scenario(case)
            except Exception as exc:
                actual = {"scenario_error": type(exc).__name__}
            results.append({"name": case["name"], "passed": actual == expected,
                            "expected": expected, "actual": actual})
        passed = sum(int(item["passed"]) for item in results)
        body = {"suite": suite["name"], "version": suite["version"],
                "coverage": list(suite.get("coverage", [])), "passed": passed,
                "total": len(results), "pass_rate": passed / len(results) if results else 0,
                "results": results, "ran_at": utc_now()}
        run_id = self.graph.record_node(
            "evaluation_run", body, actor="eval_runner",
            event_type="evaluation.run_completed")
        return {"evaluation_run_id": run_id, **body}
