"""Exact grading for deployed Friday conversation scenarios."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Callable

from .conversation_evals import ConversationQualityEvalRunner


MAX_SUITE_BYTES = 128_000
MAX_CASES = 32
MAX_TURNS = 8
MAX_PROMPT_CHARS = 4_000
MAX_EVENTS = 128
_EVENT_NAME = re.compile(r"[a-z][a-z0-9_]{0,39}")


class LiveAssistantEvalRunner:
    """Grade real server turns without trusting generated self-assessment."""

    def __init__(
        self,
        run_case: Callable[[dict[str, Any]], list[dict[str, Any]]],
        *,
        runtime_fingerprint: str,
    ):
        if not callable(run_case):
            raise TypeError("live assistant evaluator requires a case callback")
        if (not isinstance(runtime_fingerprint, str)
                or re.fullmatch(r"[0-9a-f]{64}", runtime_fingerprint) is None):
            raise ValueError("live assistant runtime fingerprint is invalid")
        self.run_case = run_case
        self.runtime_fingerprint = runtime_fingerprint

    @staticmethod
    def _read_suite(path: str | Path) -> dict[str, Any]:
        try:
            descriptor = os.open(
                Path(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if (not stat.S_ISREG(metadata.st_mode)
                        or not 2 <= metadata.st_size <= MAX_SUITE_BYTES):
                    raise ValueError(
                        "live assistant suite must be a bounded regular file")
                encoded = stream.read(MAX_SUITE_BYTES + 1)
        except OSError as exc:
            raise ValueError(
                "live assistant suite must be a bounded regular file") from exc
        if len(encoded) != metadata.st_size:
            raise ValueError("live assistant suite changed while being read")
        try:
            suite = json.loads(
                encoded.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value: {value}")),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("live assistant suite is invalid JSON") from exc
        LiveAssistantEvalRunner._validate_suite(suite)
        return suite

    @staticmethod
    def _validate_suite(suite: Any) -> None:
        if not isinstance(suite, dict):
            raise ValueError("live assistant suite must be an object")
        name = suite.get("name")
        version = suite.get("version")
        coverage = suite.get("coverage")
        cases = suite.get("cases")
        if (not isinstance(name, str) or not 1 <= len(name) <= 128
                or isinstance(version, bool) or not isinstance(version, int)
                or not 1 <= version <= 1_000_000
                or not isinstance(coverage, list) or not 1 <= len(coverage) <= 32
                or any(not isinstance(item, str) or not 1 <= len(item) <= 80
                       for item in coverage)
                or len(set(coverage)) != len(coverage)
                or not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES):
            raise ValueError("live assistant suite metadata is invalid")
        names: set[str] = set()
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("live assistant case must be an object")
            case_name = case.get("name")
            turns = case.get("turns")
            if (not isinstance(case_name, str) or not 1 <= len(case_name) <= 160
                    or case_name in names or case.get("mode") != "text"
                    or not isinstance(turns, list)
                    or not 1 <= len(turns) <= MAX_TURNS):
                raise ValueError("live assistant case metadata is invalid")
            names.add(case_name)
            for turn in turns:
                LiveAssistantEvalRunner._validate_turn(turn)

    @staticmethod
    def _validate_turn(turn: Any) -> None:
        if not isinstance(turn, dict):
            raise ValueError("live assistant turn must be an object")
        prompt = turn.get("prompt")
        numeric = (
            turn.get("min_words", 1), turn.get("max_words", 240),
            turn.get("max_sentences", 10),
        )
        required_any = turn.get("required_any", [])
        forbidden_terms = turn.get("forbidden_terms", [])
        event_fields = (
            turn.get("required_events", []), turn.get("forbidden_events", []),
        )
        exact_url_count = turn.get("exact_url_count")
        if (not isinstance(prompt, str) or not 1 <= len(prompt) <= MAX_PROMPT_CHARS
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or not 1 <= value <= 2_000 for value in numeric)
                or numeric[0] > numeric[1]
                or any(not isinstance(turn.get(field, False), bool) for field in (
                    "must_end_question", "must_not_end_question",
                    "progress_cursor_must_advance"))
                or (turn.get("must_end_question", False)
                    and turn.get("must_not_end_question", False))
                or not isinstance(required_any, list) or len(required_any) > 32
                or any(not isinstance(group, list) or not 1 <= len(group) <= 16
                       or any(not isinstance(term, str) or not 1 <= len(term) <= 120
                              for term in group) for group in required_any)
                or not isinstance(forbidden_terms, list)
                or len(forbidden_terms) > 32
                or any(not isinstance(term, str) or not 1 <= len(term) <= 120
                       for term in forbidden_terms)
                or any(not isinstance(values, list) or len(values) > 32
                       or any(not isinstance(value, str)
                              or _EVENT_NAME.fullmatch(value) is None
                              for value in values) for values in event_fields)
                or (exact_url_count is not None and (
                    isinstance(exact_url_count, bool)
                    or not isinstance(exact_url_count, int)
                    or not 0 <= exact_url_count <= 20))):
            raise ValueError("live assistant turn contract is invalid")

    @staticmethod
    def _grade_turn(
        contract: dict[str, Any], observation: dict[str, Any],
    ) -> dict[str, bool | int]:
        output = observation.get("output")
        events = observation.get("events")
        cursor_advanced = observation.get("progress_cursor_advanced")
        if not isinstance(output, str):
            output = ""
        quality = ConversationQualityEvalRunner._grade(contract, output)
        event_list_valid = bool(
            isinstance(events, list) and 1 <= len(events) <= MAX_EVENTS
            and all(isinstance(item, str) and _EVENT_NAME.fullmatch(item)
                    for item in events))
        safe_events = events if event_list_valid else []
        required_events = set(contract.get("required_events", []))
        forbidden_events = set(contract.get("forbidden_events", [])) | {"error"}
        exact_url_count = contract.get("exact_url_count")
        checks: dict[str, bool | int] = dict(quality)
        checks.update({
            "event_list_valid": event_list_valid,
            "transport_envelope": bool(
                safe_events and safe_events[0] == "you"
                and safe_events[-1] == "done"
                and safe_events.count("friday") == 1),
            "required_events": required_events.issubset(safe_events),
            "forbidden_events": forbidden_events.isdisjoint(safe_events),
            "url_count": len(re.findall(r"https?://", output)),
            "exact_url_count": bool(
                exact_url_count is None
                or len(re.findall(r"https?://", output)) == exact_url_count),
            "progress_cursor": bool(
                not contract.get("progress_cursor_must_advance", False)
                or cursor_advanced is True),
        })
        checks["passed"] = all(
            bool(value) for key, value in checks.items()
            if key not in {"word_count", "url_count"})
        return checks

    def run(self, suite_path: str | Path) -> dict[str, Any]:
        suite = self._read_suite(suite_path)
        results = []
        for case in suite["cases"]:
            failure = None
            observations: list[dict[str, Any]] = []
            try:
                observations = self.run_case(case)
                if (not isinstance(observations, list)
                        or len(observations) != len(case["turns"])):
                    raise ValueError("case callback returned the wrong turn count")
            except Exception as exc:
                failure = type(exc).__name__
            turn_results = []
            for index, contract in enumerate(case["turns"]):
                observation = (
                    observations[index]
                    if index < len(observations)
                    and isinstance(observations[index], dict) else {})
                checks = self._grade_turn(contract, observation)
                output = str(observation.get("output") or "")
                events = observation.get("events")
                encoded_events = json.dumps(
                    events if isinstance(events, list) else [],
                    separators=(",", ":"), ensure_ascii=False).encode()
                turn_results.append({
                    "index": index + 1,
                    "passed": bool(checks["passed"]),
                    "checks": checks,
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "events_sha256": hashlib.sha256(encoded_events).hexdigest(),
                })
            results.append({
                "name": case["name"],
                "passed": failure is None and all(
                    turn["passed"] for turn in turn_results),
                "failure": failure,
                "turns": turn_results,
            })
        passed = sum(1 for result in results if result["passed"])
        return {
            "suite": suite["name"],
            "version": suite["version"],
            "coverage": suite["coverage"],
            "runtime_fingerprint": self.runtime_fingerprint,
            "passed": passed,
            "total": len(results),
            "results": results,
        }
