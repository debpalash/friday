"""Host-conditional test markers.

Every skip reason starts with a classifier prefix that ``scripts/run_tests.py``
parses:

* ``platform:`` the test cannot run on this operating system by nature. The
  runner requires these to match ``tests/platform_expectations.json`` exactly
  and requires zero of them on Linux.
* ``environment:`` an optional local dependency is absent (a font, a sandbox,
  a downloaded checkpoint). These are allowed but reported.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

IS_LINUX = sys.platform.startswith("linux")
IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"
IS_POSIX = os.name == "posix"
HOST_PLATFORM = ("linux" if IS_LINUX else "darwin" if IS_MACOS
                 else "win32" if IS_WINDOWS else sys.platform)

linux_only = unittest.skipUnless(IS_LINUX, "platform: linux only")
macos_only = unittest.skipUnless(IS_MACOS, "platform: macos only")
windows_only = unittest.skipUnless(IS_WINDOWS, "platform: windows only")
posix_only = unittest.skipUnless(IS_POSIX, "platform: posix only")


def require_platform(*names: str) -> None:
    """Skip a whole module at import time unless the host is one of ``names``."""
    if HOST_PLATFORM not in names:
        raise unittest.SkipTest(f"platform: requires {' or '.join(names)}")


def requires_executable(name: str):
    return unittest.skipUnless(shutil.which(name), f"environment: missing {name}")


def _symlinks_work() -> bool:
    try:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target"
            target.write_text("x")
            (Path(temporary) / "link").symlink_to(target)
            return True
    except (OSError, NotImplementedError):
        return False


requires_symlinks = unittest.skipUnless(
    _symlinks_work(), "environment: symlinks unavailable")


def sandbox_available() -> bool:
    """True when bubblewrap can build Friday's minimal sandbox on this host."""
    if not IS_LINUX or not Path("/usr/bin/bwrap").is_file():
        return False
    import subprocess

    try:
        return subprocess.run(
            ["/usr/bin/bwrap", "--ro-bind", "/usr", "/usr", "--symlink",
             "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64", "--proc",
             "/proc", "--dev", "/dev", "--unshare-all", "--die-with-parent",
             "/usr/bin/true"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


requires_sandbox = unittest.skipUnless(
    sandbox_available(), "environment: bubblewrap sandbox is unavailable")


def assert_private_file(testcase: unittest.TestCase, path: Path) -> None:
    """Owner-only file: exact 0600 on POSIX; ACL-bounded on Windows."""
    if IS_POSIX:
        import stat

        testcase.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, path)
    else:
        testcase.assertTrue(path.is_file(), path)


def assert_private_dir(testcase: unittest.TestCase, path: Path) -> None:
    if IS_POSIX:
        import stat

        testcase.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700, path)
    else:
        testcase.assertTrue(path.is_dir(), path)


__all__ = [
    "HOST_PLATFORM", "IS_LINUX", "IS_MACOS", "IS_POSIX", "IS_WINDOWS",
    "assert_private_dir", "assert_private_file", "linux_only", "macos_only",
    "posix_only", "require_platform", "requires_executable",
    "requires_sandbox", "requires_symlinks", "sandbox_available", "windows_only",
]
