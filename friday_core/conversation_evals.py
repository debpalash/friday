"""Held-out, exact-graded evaluation for Friday's conversational output."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from pathlib import Path
from typing import Any, Callable

from .conversation import fast_system_prompt


MAX_SUITE_BYTES = 128_000
MAX_CASES = 64
MAX_PROMPT_CHARS = 4_000
MAX_OUTPUT_CHARS = 8_000
MAX_TURNS_PER_CASE = 8
_MODES = frozenset({"voice", "text"})
_CEREMONY = re.compile(
    r"\b(?:task (?:created|planned|completed|failed)|verified actions?|"
    r"response produced|choosing the next verified step|actions selected|"
    r"plan (?:ready|recorded)|executing a recorded|receipt recorded|"
    r"as an ai|great question|i(?:'d| would) be happy to|certainly[!,])\b",
    re.IGNORECASE,
)
_MARKDOWN = re.compile(
    r"(?:^|\n)\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s)|```|"
    r"\[[^\]\n]+\]\([^\)\n]+\)",
)


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def _sentence_keys(value: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", value.strip())
    return [key for sentence in sentences
            if (key := re.sub(r"[^\w]+", " ", sentence.casefold()).strip())]


def _contains_exact_term(normalized: str, term: str) -> bool:
    value = _normalized(term)
    return re.search(
        r"(?<!\w)" + re.escape(value) + r"(?!\w)", normalized) is not None


class ConversationQualityEvalRunner:
    def __init__(self, complete: Callable[[str, str], str], *, model: str,
                 runtime_fingerprint: str):
        if not callable(complete):
            raise TypeError("conversation evaluator requires a completion callback")
        if not isinstance(model, str) or not 1 <= len(model) <= 160:
            raise ValueError("conversation evaluator model identity is invalid")
        if (not isinstance(runtime_fingerprint, str)
                or re.fullmatch(r"[0-9a-f]{64}", runtime_fingerprint) is None):
            raise ValueError("conversation evaluator runtime fingerprint is invalid")
        self.complete = complete
        self.model = model
        self.runtime_fingerprint = runtime_fingerprint

    @staticmethod
    def _load_suite(suite_path: str | Path) -> dict[str, Any]:
        try:
            descriptor = os.open(
                Path(suite_path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if (not stat.S_ISREG(metadata.st_mode)
                        or not 2 <= metadata.st_size <= MAX_SUITE_BYTES):
                    raise ValueError(
                        "conversation evaluation suite must be a bounded regular file")
                encoded = stream.read(MAX_SUITE_BYTES + 1)
        except OSError as exc:
            raise ValueError(
                "conversation evaluation suite must be a bounded regular file") from exc
        if len(encoded) != metadata.st_size:
            raise ValueError("conversation evaluation suite changed while being read")

        def reject_constant(_value: str):
            raise ValueError("conversation suite contains a non-finite number")

        suite = json.loads(encoded.decode("utf-8"), parse_constant=reject_constant)
        if not isinstance(suite, dict):
            raise ValueError("conversation evaluation suite must be an object")
        name = suite.get("name")
        version = suite.get("version")
        coverage = suite.get("coverage", [])
        cases = suite.get("cases")
        if (not isinstance(name, str) or not 1 <= len(name) <= 128
                or isinstance(version, bool) or not isinstance(version, int)
                or not 1 <= version <= 1_000_000
                or not isinstance(coverage, list) or len(coverage) > 32
                or any(not isinstance(item, str) or not 1 <= len(item) <= 80
                       for item in coverage)
                or len(set(coverage)) != len(coverage)
                or not isinstance(cases, list)
                or not 1 <= len(cases) <= MAX_CASES):
            raise ValueError("conversation evaluation suite metadata is invalid")

        seen: set[str] = set()
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("conversation evaluation case must be an object")
            case_name = case.get("name")
            prompt = case.get("prompt")
            required_any = case.get("required_any", [])
            forbidden = case.get("forbidden_terms", [])
            numeric = [case.get("min_words", 1), case.get("max_words", 200),
                       case.get("max_sentences", 8)]
            if (not isinstance(case_name, str) or not 1 <= len(case_name) <= 160
                    or case_name in seen or case.get("mode") not in _MODES
                    or not isinstance(prompt, str)
                    or not 1 <= len(prompt) <= MAX_PROMPT_CHARS
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           or not 1 <= value <= 2_000 for value in numeric)
                    or numeric[0] > numeric[1]
                    or not isinstance(case.get("forbid_markdown", False), bool)
                    or not isinstance(case.get("must_end_question", False), bool)
                    or not isinstance(required_any, list) or len(required_any) > 32
                    or any(not isinstance(group, list) or not 1 <= len(group) <= 16
                           or any(not isinstance(term, str)
                                  or not 1 <= len(term) <= 120 for term in group)
                           for group in required_any)
                    or not isinstance(forbidden, list) or len(forbidden) > 32
                    or any(not isinstance(term, str) or not 1 <= len(term) <= 120
                           for term in forbidden)):
                raise ValueError("conversation evaluation case is invalid")
            seen.add(case_name)
        return suite

    @staticmethod
    def _grade(case: dict[str, Any], output: str) -> dict[str, bool | int]:
        bounded_output = len(output) <= MAX_OUTPUT_CHARS
        graded_output = output if bounded_output else output[:MAX_OUTPUT_CHARS + 1]
        normalized = _normalized(graded_output)
        words = re.findall(r"\b[\w'-]+\b", graded_output, re.UNICODE)
        sentence_keys = _sentence_keys(graded_output)
        required_groups = case.get("required_any", [])
        forbidden = case.get("forbidden_terms", [])
        checks: dict[str, bool | int] = {
            "nonempty": bool(normalized),
            "bounded_output": bounded_output,
            "word_count": len(words),
            "word_range": (
                int(case.get("min_words", 1)) <= len(words)
                <= int(case.get("max_words", 200))),
            "sentence_limit": (
                len(sentence_keys) <= int(case.get("max_sentences", 8))),
            "required_terms": all(
                any(_normalized(term) in normalized for term in group)
                for group in required_groups),
            "forbidden_terms": not any(
                _contains_exact_term(normalized, term) for term in forbidden),
            "no_task_ceremony": _CEREMONY.search(graded_output) is None,
            "no_repeated_sentences": len(sentence_keys) == len(set(sentence_keys)),
            "markdown_policy": (
                not bool(case.get("forbid_markdown"))
                or _MARKDOWN.search(graded_output) is None),
            "question_when_ambiguous": (
                not bool(case.get("must_end_question"))
                or graded_output.rstrip().endswith("?")),
            "answer_when_context_clear": (
                not bool(case.get("must_not_end_question"))
                or not graded_output.rstrip().endswith("?")),
        }
        checks["passed"] = all(
            bool(value) for key, value in checks.items() if key != "word_count")
        return checks

    def run(self, suite_path: str | Path) -> dict[str, Any]:
        suite = self._load_suite(suite_path)
        results = []
        for case in suite["cases"]:
            output = ""
            failure = None
            try:
                output = self.complete(
                    fast_system_prompt(
                        owner_name="the user", display_mode=case["mode"] == "text"),
                    case["prompt"])
                if not isinstance(output, str):
                    raise TypeError("completion output must be text")
                checks = self._grade(case, output)
            except Exception as exc:
                failure = type(exc).__name__
                checks = {"passed": False}
            results.append({
                "name": case["name"], "mode": case["mode"],
                "passed": bool(checks["passed"]), "checks": checks,
                "failure": failure,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            })
        passed = sum(1 for result in results if result["passed"])
        return {
            "suite": suite["name"], "version": suite["version"],
            "coverage": suite.get("coverage", []),
            "model": self.model,
            "runtime_fingerprint": self.runtime_fingerprint,
            "passed": passed, "total": len(results), "results": results,
        }


class ConversationContinuityEvalRunner:
    """Exact-graded, stateful evaluation across related conversation turns."""

    def __init__(self, complete: Callable[[str, list[dict]], str], *, model: str,
                 runtime_fingerprint: str):
        if not callable(complete):
            raise TypeError("continuity evaluator requires a completion callback")
        if not isinstance(model, str) or not 1 <= len(model) <= 160:
            raise ValueError("continuity evaluator model identity is invalid")
        if (not isinstance(runtime_fingerprint, str)
                or re.fullmatch(r"[0-9a-f]{64}", runtime_fingerprint) is None):
            raise ValueError("continuity evaluator runtime fingerprint is invalid")
        self.complete = complete
        self.model = model
        self.runtime_fingerprint = runtime_fingerprint

    @staticmethod
    def _load_suite(suite_path: str | Path) -> dict[str, Any]:
        try:
            descriptor = os.open(
                Path(suite_path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if (not stat.S_ISREG(metadata.st_mode)
                        or not 2 <= metadata.st_size <= MAX_SUITE_BYTES):
                    raise ValueError(
                        "continuity evaluation suite must be a bounded regular file")
                encoded = stream.read(MAX_SUITE_BYTES + 1)
        except OSError as exc:
            raise ValueError(
                "continuity evaluation suite must be a bounded regular file") from exc
        if len(encoded) != metadata.st_size:
            raise ValueError("continuity evaluation suite changed while being read")

        def reject_constant(_value: str):
            raise ValueError("continuity suite contains a non-finite number")

        suite = json.loads(encoded.decode("utf-8"), parse_constant=reject_constant)
        if not isinstance(suite, dict):
            raise ValueError("continuity evaluation suite must be an object")
        name = suite.get("name")
        version = suite.get("version")
        coverage = suite.get("coverage", [])
        cases = suite.get("cases")
        if (not isinstance(name, str) or not 1 <= len(name) <= 128
                or isinstance(version, bool) or not isinstance(version, int)
                or not 1 <= version <= 1_000_000
                or not isinstance(coverage, list) or len(coverage) > 32
                or any(not isinstance(item, str) or not 1 <= len(item) <= 80
                       for item in coverage)
                or len(set(coverage)) != len(coverage)
                or not isinstance(cases, list)
                or not 1 <= len(cases) <= MAX_CASES):
            raise ValueError("continuity evaluation suite metadata is invalid")

        seen: set[str] = set()
        total_turns = 0
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("continuity evaluation case must be an object")
            case_name = case.get("name")
            turns = case.get("turns")
            if (not isinstance(case_name, str) or not 1 <= len(case_name) <= 160
                    or case_name in seen or case.get("mode") not in _MODES
                    or not isinstance(turns, list)
                    or not 2 <= len(turns) <= MAX_TURNS_PER_CASE):
                raise ValueError("continuity evaluation case is invalid")
            seen.add(case_name)
            total_turns += len(turns)
            for turn in turns:
                if not isinstance(turn, dict):
                    raise ValueError("continuity evaluation turn must be an object")
                prompt = turn.get("prompt")
                required_any = turn.get("required_any", [])
                forbidden = turn.get("forbidden_terms", [])
                numeric = [turn.get("min_words", 1), turn.get("max_words", 200),
                           turn.get("max_sentences", 8)]
                if (not isinstance(prompt, str)
                        or not 1 <= len(prompt) <= MAX_PROMPT_CHARS
                        or any(isinstance(value, bool) or not isinstance(value, int)
                               or not 1 <= value <= 2_000 for value in numeric)
                        or numeric[0] > numeric[1]
                        or any(not isinstance(turn.get(field, False), bool)
                               for field in ("forbid_markdown", "must_end_question",
                                             "must_not_end_question"))
                        or (turn.get("must_end_question", False)
                            and turn.get("must_not_end_question", False))
                        or not isinstance(required_any, list)
                        or len(required_any) > 32
                        or any(not isinstance(group, list)
                               or not 1 <= len(group) <= 16
                               or any(not isinstance(term, str)
                                      or not 1 <= len(term) <= 120
                                      for term in group)
                               for group in required_any)
                        or not isinstance(forbidden, list) or len(forbidden) > 32
                        or any(not isinstance(term, str)
                               or not 1 <= len(term) <= 120 for term in forbidden)):
                    raise ValueError("continuity evaluation turn is invalid")
        if total_turns > MAX_CASES * MAX_TURNS_PER_CASE:
            raise ValueError("continuity evaluation suite has too many turns")
        return suite

    def run(self, suite_path: str | Path) -> dict[str, Any]:
        suite = self._load_suite(suite_path)
        results = []
        for case in suite["cases"]:
            system = fast_system_prompt(
                owner_name="the user", display_mode=case["mode"] == "text")
            history: list[dict] = []
            turn_results = []
            for index, turn in enumerate(case["turns"]):
                output = ""
                failure = None
                history.append({"role": "user", "content": turn["prompt"]})
                try:
                    output = self.complete(system, list(history))
                    if not isinstance(output, str):
                        raise TypeError("completion output must be text")
                    checks = ConversationQualityEvalRunner._grade(turn, output)
                except Exception as exc:
                    failure = type(exc).__name__
                    checks = {"passed": False}
                history.append({"role": "assistant", "content": output})
                turn_results.append({
                    "index": index + 1,
                    "passed": bool(checks["passed"]),
                    "checks": checks,
                    "failure": failure,
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                })
            results.append({
                "name": case["name"], "mode": case["mode"],
                "passed": all(turn["passed"] for turn in turn_results),
                "turns": turn_results,
            })
        passed = sum(1 for result in results if result["passed"])
        return {
            "suite": suite["name"], "version": suite["version"],
            "coverage": suite.get("coverage", []),
            "model": self.model,
            "runtime_fingerprint": self.runtime_fingerprint,
            "passed": passed, "total": len(results), "results": results,
        }
