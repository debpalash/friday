"""Operating-system-specific modules are imported only inside friday_host."""

from __future__ import annotations

import ast
import importlib
import pkgutil
import unittest
from pathlib import Path

from friday_host.host import HostPlatform
from friday_host.platform_capabilities import compute_capabilities

ROOT = Path(__file__).resolve().parents[1]
POSIX_ONLY = {"fcntl", "pwd", "grp", "termios", "resource", "msvcrt", "winreg",
              "_winapi"}
ALLOWED = {ROOT / "friday_host" / "fs.py"}
SCANNED = [ROOT / "server.py", ROOT / "supervisor.py", ROOT / "friday.py",
           *sorted((ROOT / "friday_core").glob("*.py")),
           *sorted((ROOT / "friday_host").glob("*.py")),
           *sorted((ROOT / "ops").glob("*.py"))]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


class ImportBoundaryTests(unittest.TestCase):
    def test_posix_only_modules_stay_inside_the_platform_layer(self) -> None:
        offenders = {}
        for path in SCANNED:
            if path in ALLOWED:
                continue
            found = _imports(path) & POSIX_ONLY
            if found:
                offenders[str(path.relative_to(ROOT))] = sorted(found)
        self.assertEqual(offenders, {})

    def test_friday_host_never_imports_the_application_core(self) -> None:
        for path in sorted((ROOT / "friday_host").glob("*.py")):
            self.assertFalse(
                {"friday_core", "server", "supervisor"} & _imports(path),
                f"{path.name} imports the application core")

    def test_every_core_module_imports_on_this_host(self) -> None:
        for finder, name, _is_package in pkgutil.iter_modules([str(ROOT / "friday_core")]):
            with self.subTest(module=name):
                importlib.import_module(f"friday_core.{name}")
        for finder, name, _is_package in pkgutil.iter_modules([str(ROOT / "friday_host")]):
            with self.subTest(module=name):
                importlib.import_module(f"friday_host.{name}")

    def test_unsupported_action_classes_are_reported_not_hidden(self) -> None:
        for host in (HostPlatform(os="macos", arch="aarch64"),
                     HostPlatform(os="windows", arch="x86_64"),
                     HostPlatform(os="linux", arch="x86_64")):
            capabilities = compute_capabilities(host)
            for name in ("desktop", "managed_processes", "managed_browser"):
                self.assertFalse(getattr(capabilities, name), (host.os, name))
                self.assertIn(name, capabilities.reasons, (host.os, name))
            self.assertTrue(capabilities.unavailable_tools)


if __name__ == "__main__":
    unittest.main()
