"""Process probes identify same-user processes on this host."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from friday_host import procs


class IdentityTests(unittest.TestCase):
    def test_this_process_is_owned_by_its_working_directory_and_argv(self) -> None:
        cwd = Path.cwd()
        identity = procs.process_identity(os.getpid())
        self.assertEqual(identity.pid, os.getpid())
        self.assertEqual(identity.cwd, cwd.resolve())
        self.assertTrue(identity.cmdline)
        self.assertTrue(identity.comm)
        marker = identity.cmdline.split()[0]
        self.assertTrue(procs.owned(os.getpid(), cwd, marker))
        self.assertFalse(procs.owned(os.getpid(), cwd, "definitely-not-in-argv"))
        self.assertFalse(procs.owned(os.getpid(), cwd / "elsewhere", "python"))

    def test_discover_pid_finds_a_child_by_cwd_and_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = "friday-host-discover-marker"
            child = subprocess.Popen(
                [sys.executable, "-c",
                 f"import time; '{marker}'; time.sleep(30)"],
                cwd=temporary, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 5
                found = None
                while time.monotonic() < deadline and found is None:
                    found = procs.discover_pid(Path(temporary), marker)
                    time.sleep(0.05)
                self.assertEqual(found, child.pid)
                self.assertTrue(procs.pid_exists(child.pid))
            finally:
                child.kill()
                child.wait(timeout=10)
            self.assertTrue(procs.wait_pid_exit(child.pid, 5))

    def test_unknown_pid_is_not_owned(self) -> None:
        self.assertFalse(procs.owned(2 ** 22 - 1, Path.cwd(), "python"))
        with self.assertRaises(OSError):
            procs.process_identity(2 ** 22 - 1)


class MachineProbeTests(unittest.TestCase):
    def test_memory_info_reports_sane_totals(self) -> None:
        total, available = procs.memory_info_mib(live=True)
        self.assertGreaterEqual(total, 128)
        self.assertLessEqual(available, total)
        static_total, static_available = procs.memory_info_mib()
        self.assertEqual(static_total, total)
        self.assertLessEqual(static_available, total)

    def test_memory_info_falls_back_when_meminfo_is_missing(self) -> None:
        total, available = procs.memory_info_mib(
            live=True, meminfo_path=Path("/nonexistent/meminfo"))
        self.assertGreaterEqual(total, 128)
        self.assertGreaterEqual(available, 0)

    def test_physical_memory_and_load(self) -> None:
        self.assertGreater(procs.physical_memory_bytes(), 0)
        self.assertGreaterEqual(procs.load_average_1m(), 0.0)

    def test_boot_id_hash_is_a_stable_sha256(self) -> None:
        first = procs.boot_id_hash()
        self.assertEqual(len(first), 64)
        self.assertEqual(first, procs.boot_id_hash())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "boot_id"
            path.write_text("abc\n")
            self.assertEqual(
                procs.boot_id_hash(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            path.write_text("")
            with self.assertRaises(RuntimeError):
                procs.boot_id_hash(path)
            with self.assertRaises(RuntimeError):
                procs.boot_id_hash(Path(temporary) / "missing")

    def test_spawn_kwargs(self) -> None:
        kwargs = procs.detached_popen_kwargs()
        if procs.IS_WINDOWS:
            self.assertIn("creationflags", kwargs)
        else:
            self.assertEqual(kwargs, {"start_new_session": True})
            self.assertEqual(procs.hidden_console_kwargs(), {})
            self.assertEqual(procs.runtime_dir(1000), Path("/run/user/1000")
                             if procs.IS_LINUX else None)


class TerminateTreeTests(unittest.TestCase):
    def test_terminate_tree_stops_a_child_and_its_descendants(self) -> None:
        try:
            import psutil  # noqa: F401
        except ImportError:
            self.skipTest("environment: psutil is not installed")
        script = (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(60)\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        grandchild = int(parent.stdout.readline().strip())
        procs.terminate_tree(parent.pid, grace_seconds=5, kill_seconds=5)
        self.assertTrue(procs.wait_pid_exit(parent.pid, 5))
        self.assertTrue(procs.wait_pid_exit(grandchild, 5))
        parent.wait(timeout=5)
        procs.terminate_tree(parent.pid)


if __name__ == "__main__":
    unittest.main()
