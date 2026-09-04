"""Private-file primitives with one POSIX path and one Windows path.

On POSIX every function reproduces the exact ``fcntl``/``os`` expression the
Linux code used before the port. On Windows, file modes and numeric owners
are meaningless; the user's profile ACL is the privacy boundary and the
checks that depend on ``st_uid``/``st_mode`` report success. ``msvcrt`` byte
locks replace ``flock``.
"""

from __future__ import annotations

import contextlib
import os
import stat
import sys
import time
from pathlib import Path
from typing import Iterator

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:  # pragma: no cover - exercised on Windows runners
    import msvcrt
else:
    import fcntl


PRIVATE_OPEN_FLAGS = (
    getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_NOINHERIT", 0)
)


def lock_exclusive(descriptor: int, *, blocking: bool = True,
                   timeout: float | None = None) -> None:
    """Take an exclusive advisory lock on an open descriptor.

    Raises ``BlockingIOError`` when ``blocking`` is false (or ``timeout``
    elapses) and another holder owns the lock.
    """
    if not IS_WINDOWS:
        if blocking and timeout is None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if not blocking or (deadline is not None
                                    and time.monotonic() >= deadline):
                    raise
                time.sleep(0.1)
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:  # pragma: no cover - Windows only
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if not blocking or (deadline is not None
                                and time.monotonic() >= deadline):
                raise BlockingIOError(str(exc)) from exc
            time.sleep(0.1)


def unlock(descriptor: int) -> None:
    if not IS_WINDOWS:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return
    os.lseek(descriptor, 0, os.SEEK_SET)  # pragma: no cover - Windows only
    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def exclusive_lock(descriptor: int, *, blocking: bool = True,
                   timeout: float | None = None) -> Iterator[None]:
    lock_exclusive(descriptor, blocking=blocking, timeout=timeout)
    try:
        yield
    finally:
        unlock(descriptor)


def fsync_directory(path: Path | str) -> None:
    """Persist a directory entry; a no-op where directories cannot be opened."""
    if IS_WINDOWS:  # pragma: no cover - Windows only
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def current_uid() -> int:
    if hasattr(os, "geteuid"):
        return os.geteuid()
    return 0  # pragma: no cover - Windows only


def owned_by_caller(metadata: os.stat_result) -> bool:
    """True when the file belongs to the effective user (always on Windows)."""
    if IS_WINDOWS:  # pragma: no cover - Windows only
        return True
    return metadata.st_uid == os.geteuid()


def private_mode_ok(metadata: os.stat_result, *, mask: int = 0o077) -> bool:
    """True when no masked permission bit is set (always on Windows)."""
    if IS_WINDOWS:  # pragma: no cover - Windows only
        return True
    return not (metadata.st_mode & mask)


def chmod_private(target: Path | str | int, mode: int) -> None:
    """``chmod``/``fchmod`` on POSIX; nothing to do on Windows."""
    if IS_WINDOWS:  # pragma: no cover - Windows only
        return
    if isinstance(target, int):
        os.fchmod(target, mode)
    else:
        os.chmod(target, mode)


def is_private_regular(metadata: os.stat_result) -> bool:
    return (stat.S_ISREG(metadata.st_mode) and owned_by_caller(metadata)
            and metadata.st_nlink == 1 and private_mode_ok(metadata))


def is_private_directory(metadata: os.stat_result) -> bool:
    return (stat.S_ISDIR(metadata.st_mode) and owned_by_caller(metadata)
            and private_mode_ok(metadata))


__all__ = [
    "IS_WINDOWS", "PRIVATE_OPEN_FLAGS", "chmod_private", "current_uid",
    "exclusive_lock", "fsync_directory", "is_private_directory",
    "is_private_regular", "lock_exclusive", "owned_by_caller",
    "private_mode_ok", "unlock",
]
