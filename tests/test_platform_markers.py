"""Skip accounting: every conditional skip is classified and expected."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path

from tests import platform_markers

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
EXPECTATIONS = json.loads((TESTS / "platform_expectations.json").read_text())
SKIP_CALLS = {"skipTest", "skipUnless", "skipIf", "SkipTest", "skip"}


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_tests", ROOT / "scripts" / "run_tests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExpectationFileTests(unittest.TestCase):
    def test_linux_expects_zero_platform_skips(self) -> None:
        self.assertEqual(EXPECTATIONS["linux"]["platform"], [])

    def test_every_host_has_sorted_unique_lists(self) -> None:
        for host in ("linux", "darwin", "win32"):
            self.assertIn(host, EXPECTATIONS)
            for kind in ("platform", "environment"):
                items = EXPECTATIONS[host][kind]
                self.assertEqual(items, sorted(set(items)), (host, kind))

    def test_expectation_ids_resolve_to_test_modules(self) -> None:
        for host, kinds in EXPECTATIONS.items():
            for items in kinds.values():
                for item in items:
                    module = item.split(".")[1] if item.startswith("tests.") else None
                    self.assertIsNotNone(module, item)
                    self.assertTrue((TESTS / f"{module}.py").is_file(), item)


class SkipReasonTests(unittest.TestCase):
    def test_every_literal_skip_reason_carries_a_classifier(self) -> None:
        offenders = []
        for path in sorted(TESTS.glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = (node.func.attr if isinstance(node.func, ast.Attribute)
                        else node.func.id if isinstance(node.func, ast.Name) else "")
                if name not in SKIP_CALLS:
                    continue
                reasons = [arg for arg in node.args if isinstance(arg, ast.Constant)
                           and isinstance(arg.value, str)]
                for reason in reasons:
                    if not re.match(r"(platform|environment):", reason.value):
                        offenders.append(f"{path.name}:{node.lineno}: {reason.value!r}")
        self.assertEqual(offenders, [])


class RunnerTests(unittest.TestCase):
    def test_runner_classifies_and_audits_skips(self) -> None:
        runner = _load_runner()
        groups = runner.classify([
            ("tests.test_a.T.test_x", "platform: linux only"),
            ("tests.test_b.T.test_y", "environment: missing font"),
            ("tests.test_c.T.test_z", "no reason"),
        ])
        self.assertEqual([t for t, _ in groups["platform"]], ["tests.test_a.T.test_x"])
        self.assertEqual([t for t, _ in groups["unclassified"]], ["tests.test_c.T.test_z"])
        expectations = {runner.HOST: {"platform": ["tests.test_a"], "environment": []}}
        problems = runner.check_expectations(
            groups, expectations, require_no_platform_skips=False)
        self.assertEqual(len(problems), 1)
        self.assertIn("skip without classifier", problems[0])
        stale = {runner.HOST: {"platform": ["tests.test_a", "tests.test_missing"],
                               "environment": []}}
        problems = runner.check_expectations(
            {"platform": groups["platform"], "environment": [], "unclassified": []},
            stale, require_no_platform_skips=True)
        self.assertTrue(any("stale platform expectation" in p for p in problems))
        self.assertTrue(any("must skip none" in p for p in problems))

    def test_markers_match_this_host(self) -> None:
        self.assertEqual(platform_markers.HOST_PLATFORM in {"linux", "darwin", "win32"},
                         True)
        self.assertEqual(platform_markers.IS_LINUX, sys.platform.startswith("linux"))


if __name__ == "__main__":
    unittest.main()
