"""Private-file helpers keep their POSIX semantics and lock exclusively."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from friday_host import fs

REPO = Path(__file__).resolve().parents[1]


class LockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.lock_path = Path(self.temporary.name) / "test.lock"

    def _hold_in_subprocess(self) -> subprocess.Popen:
        script = (
            "import os, sys, time\n"
            "sys.path.insert(0, sys.argv[2])\n"
            "from friday_host import fs\n"
            "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
            "fs.lock_exclusive(fd)\n"
            "print('held', flush=True)\n"
            "sys.stdin.readline()\n"
            "fs.unlock(fd)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(self.lock_path), str(REPO)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.addCleanup(process.kill)
        self.assertEqual(process.stdout.readline().strip(), "held")
        return process

    def test_second_holder_is_refused_until_release(self) -> None:
        holder = self._hold_in_subprocess()
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        self.addCleanup(os.close, descriptor)
        with self.assertRaises(BlockingIOError):
            fs.lock_exclusive(descriptor, blocking=False)
        with self.assertRaises(BlockingIOError):
            fs.lock_exclusive(descriptor, timeout=0.3)
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder.wait(timeout=10)
        fs.lock_exclusive(descriptor, timeout=5)
        fs.unlock(descriptor)

    def test_context_manager_releases_on_exit(self) -> None:
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        self.addCleanup(os.close, descriptor)
        with fs.exclusive_lock(descriptor):
            other = os.open(self.lock_path, os.O_RDWR)
            try:
                with self.assertRaises(BlockingIOError):
                    fs.lock_exclusive(other, blocking=False)
            finally:
                os.close(other)
        other = os.open(self.lock_path, os.O_RDWR)
        try:
            fs.lock_exclusive(other, blocking=False)
            fs.unlock(other)
        finally:
            os.close(other)

    @unittest.skipIf(fs.IS_WINDOWS, "platform: posix lock implementation")
    def test_posix_lock_uses_flock(self) -> None:
        with mock.patch.object(fs.fcntl, "flock") as flock:
            fs.lock_exclusive(7)
            fs.lock_exclusive(7, blocking=False)
            fs.unlock(7)
        self.assertEqual(flock.call_args_list[0].args, (7, fs.fcntl.LOCK_EX))
        self.assertEqual(flock.call_args_list[1].args,
                         (7, fs.fcntl.LOCK_EX | fs.fcntl.LOCK_NB))
        self.assertEqual(flock.call_args_list[2].args, (7, fs.fcntl.LOCK_UN))


class PrivateFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_private_open_flags_match_the_pre_port_expression(self) -> None:
        expected = getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        if not fs.IS_WINDOWS:
            self.assertEqual(fs.PRIVATE_OPEN_FLAGS, expected)
        self.assertEqual(fs.PRIVATE_OPEN_FLAGS & os.O_RDONLY, 0)

    def test_fsync_directory_accepts_a_real_directory(self) -> None:
        fs.fsync_directory(self.root)
        with self.assertRaises(OSError):
            fs.fsync_directory(self.root / "missing")

    @unittest.skipIf(fs.IS_WINDOWS, "platform: posix mode bits")
    def test_posix_identity_checks(self) -> None:
        private = self.root / "private"
        private.write_bytes(b"x")
        os.chmod(private, 0o600)
        loose = self.root / "loose"
        loose.write_bytes(b"x")
        os.chmod(loose, 0o644)
        self.assertTrue(fs.is_private_regular(private.lstat()))
        self.assertFalse(fs.is_private_regular(loose.lstat()))
        self.assertTrue(fs.owned_by_caller(private.lstat()))
        self.assertTrue(fs.private_mode_ok(private.lstat()))
        self.assertFalse(fs.private_mode_ok(loose.lstat()))
        self.assertTrue(fs.private_mode_ok(loose.lstat(), mask=0o022))
        os.chmod(self.root, 0o700)
        self.assertTrue(fs.is_private_directory(self.root.lstat()))
        self.assertFalse(fs.is_private_directory(private.lstat()))
        self.assertEqual(fs.current_uid(), os.geteuid())
        fs.chmod_private(loose, 0o600)
        self.assertEqual(loose.stat().st_mode & 0o777, 0o600)
        descriptor = os.open(loose, os.O_RDONLY)
        try:
            fs.chmod_private(descriptor, 0o400)
        finally:
            os.close(descriptor)
        self.assertEqual(loose.stat().st_mode & 0o777, 0o400)


class WindowsBranchTests(unittest.TestCase):
    """Drive the Windows branch on any host with a stubbed msvcrt."""

    def test_windows_lock_uses_msvcrt_byte_locking(self) -> None:
        calls = []

        class FakeMsvcrt:
            LK_NBLCK = 2
            LK_UNLCK = 0
            busy = True

            def locking(self, descriptor, mode, size):
                calls.append(("locking", descriptor, mode, size))
                if mode == self.LK_NBLCK and self.busy:
                    self.busy = False
                    raise OSError(36, "busy")

        seeks = []
        with mock.patch.object(fs, "IS_WINDOWS", True), \
                mock.patch.object(fs, "msvcrt", FakeMsvcrt(), create=True), \
                mock.patch.object(fs.os, "lseek",
                                  lambda *args: seeks.append(args)), \
                mock.patch.object(fs.time, "sleep", lambda _s: None):
            with self.assertRaises(BlockingIOError):
                fs.lock_exclusive(5, blocking=False)
            calls.clear()
            fs.lock_exclusive(5)
            fs.unlock(5)
        self.assertEqual(calls, [("locking", 5, 2, 1), ("locking", 5, 0, 1)])
        self.assertTrue(all(args[1:] == (0, os.SEEK_SET) for args in seeks))

    def test_windows_identity_checks_defer_to_the_profile_acl(self) -> None:
        metadata = os.stat(__file__)
        with mock.patch.object(fs, "IS_WINDOWS", True):
            self.assertTrue(fs.owned_by_caller(metadata))
            self.assertTrue(fs.private_mode_ok(metadata))
            fs.chmod_private(Path(__file__), 0o600)
            fs.fsync_directory(Path(__file__).parent / "missing")


if __name__ == "__main__":
    unittest.main()
