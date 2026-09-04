#!/usr/bin/env python3
"""Run the unit tests and account for every platform-conditional skip.

A ``platform:`` skip must be listed for this host in
``tests/platform_expectations.json`` and every listed skip must occur, so
the allowlist cannot go stale. Linux is required to skip nothing for
platform reasons. ``environment:`` skips are reported; unlisted ones are
warnings because developer machines differ. Any skip without a classifier
prefix fails the run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTATIONS = ROOT / "tests" / "platform_expectations.json"
HOST = ("linux" if sys.platform.startswith("linux") else "darwin"
        if sys.platform == "darwin" else "win32" if sys.platform == "win32"
        else sys.platform)


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_records: list[tuple[str, str]] = []

    def addSkip(self, test, reason):  # noqa: N802 - unittest API
        super().addSkip(test, reason)
        self.skip_records.append((test.id(), str(reason)))


def _matches(test_id: str, expected: str) -> bool:
    return test_id == expected or test_id.startswith(expected + ".")


def classify(records: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    groups: dict[str, list[tuple[str, str]]] = {
        "platform": [], "environment": [], "unclassified": []}
    for test_id, reason in records:
        if reason.startswith("platform:"):
            groups["platform"].append((test_id, reason))
        elif reason.startswith("environment:"):
            groups["environment"].append((test_id, reason))
        else:
            groups["unclassified"].append((test_id, reason))
    return groups


def check_expectations(groups: dict, expectations: dict, *,
                       require_no_platform_skips: bool) -> list[str]:
    problems: list[str] = []
    host = expectations.get(HOST, {"platform": [], "environment": []})
    expected_platform = list(host.get("platform", []))
    expected_environment = list(host.get("environment", []))
    for test_id, reason in groups["unclassified"]:
        problems.append(f"skip without classifier: {test_id}: {reason}")
    for test_id, _reason in groups["platform"]:
        if not any(_matches(test_id, item) for item in expected_platform):
            problems.append(f"unexpected platform skip on {HOST}: {test_id}")
    for item in expected_platform:
        if not any(_matches(test_id, item) for test_id, _r in groups["platform"]):
            problems.append(f"stale platform expectation on {HOST}: {item}")
    if require_no_platform_skips and groups["platform"]:
        problems.append(
            f"{len(groups['platform'])} platform skip(s) on a host that must skip none")
    for test_id, _reason in groups["environment"]:
        if not any(_matches(test_id, item) for item in expected_environment):
            print(f"warning: environment skip not listed for {HOST}: {test_id}",
                  file=sys.stderr)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, help="write a JSON summary here")
    parser.add_argument("--github-summary", action="store_true",
                        help="append a table to $GITHUB_STEP_SUMMARY")
    parser.add_argument("--require-no-platform-skips", action="store_true")
    parser.add_argument("--start", default="tests")
    parser.add_argument("--pattern", default="test*.py")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(args.start, pattern=args.pattern,
                                                top_level_dir=str(ROOT))
    runner = unittest.TextTestRunner(verbosity=args.verbose,
                                     resultclass=RecordingResult)
    started = time.monotonic()
    result = runner.run(suite)
    duration = round(time.monotonic() - started, 1)
    groups = classify(result.skip_records)
    expectations = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    problems = check_expectations(
        groups, expectations,
        require_no_platform_skips=args.require_no_platform_skips)

    summary = {
        "host": HOST, "tests": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "duration_seconds": duration,
        "skips": {key: [{"test": t, "reason": r} for t, r in value]
                  for key, value in groups.items()},
        "problems": problems,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    if args.github_summary and os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(f"### Tests on {HOST}\n\n")
            stream.write("| Ran | Failures | Errors | Platform skips | Environment skips |\n")
            stream.write("|---|---|---|---|---|\n")
            stream.write(f"| {result.testsRun} | {len(result.failures)} | "
                         f"{len(result.errors)} | {len(groups['platform'])} | "
                         f"{len(groups['environment'])} |\n")
            for problem in problems:
                stream.write(f"- :x: {problem}\n")
    for problem in problems:
        print(f"skip accounting: {problem}", file=sys.stderr)
    if problems:
        return 2
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
