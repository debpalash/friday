"""Process and machine probes that differ by operating system.

Linux uses procfs exactly as Friday did before the port. macOS and Windows
use ``psutil`` (an existing application dependency) which is imported lazily
so this module stays importable from a bare interpreter.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

IS_LINUX = sys.platform.startswith("linux")
IS_WINDOWS = sys.platform == "win32"


def _psutil():
    import psutil  # noqa: PLC0415 - optional off-Linux dependency

    return psutil


def detached_popen_kwargs() -> dict[str, object]:
    """Keyword arguments that detach a child into its own process group."""
    if IS_WINDOWS:  # pragma: no cover - Windows only
        return {
            "creationflags": (subprocess.CREATE_NEW_PROCESS_GROUP
                              | subprocess.CREATE_NO_WINDOW),
        }
    return {"start_new_session": True}


def hidden_console_kwargs() -> dict[str, object]:
    """Keyword arguments that keep helper subprocesses off the desktop."""
    if IS_WINDOWS:  # pragma: no cover - Windows only
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def physical_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(
            os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    try:
        return int(_psutil().virtual_memory().total)
    except Exception:
        return 0


def memory_info_mib(*, live: bool = False,
                    meminfo_path: Path = Path("/proc/meminfo")) -> tuple[int, int]:
    """Return (total, available) MiB; ``available`` is 0 when unobservable."""
    values: dict[str, int] = {}
    try:
        for line in meminfo_path.read_text().splitlines():
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0]) // 1024
    except (OSError, ValueError, IndexError):
        if not IS_LINUX:
            try:
                memory = _psutil().virtual_memory()
                total = max(128, int(memory.total) // (1024 ** 2))
                return total, (max(0, int(memory.available) // (1024 ** 2))
                               if live else total)
            except Exception:
                pass
        fallback = max(128, physical_memory_bytes() // (1024 ** 2))
        return fallback, 0 if live else fallback
    total = max(128, values.get("MemTotal", 128))
    if live and "MemAvailable" not in values:
        return total, 0
    return total, max(0, values.get("MemAvailable", total))


def load_average_1m(loadavg_path: Path = Path("/proc/loadavg")) -> float:
    """One-minute load average; raises ``OSError`` when unobservable."""
    if IS_LINUX:
        return float(loadavg_path.read_text().split()[0])
    if hasattr(os, "getloadavg"):
        return float(os.getloadavg()[0])
    try:  # pragma: no cover - Windows only
        return float(_psutil().getloadavg()[0])
    except Exception as exc:  # pragma: no cover - Windows only
        raise OSError("load average is unavailable") from exc


def runtime_dir(uid: int | None = None) -> Path | None:
    """The per-user runtime directory; Linux only."""
    if not IS_LINUX:
        return None
    return Path(f"/run/user/{os.getuid() if uid is None else uid}")


def boot_id_hash(path: Path = Path("/proc/sys/kernel/random/boot_id")) -> str:
    """Stable hash of the current OS boot; never the raw identifier."""
    if IS_LINUX or path != Path("/proc/sys/kernel/random/boot_id"):
        try:
            boot_id = path.read_text().strip()
        except OSError as exc:
            raise RuntimeError("cannot read kernel boot identity") from exc
        if not boot_id:
            raise RuntimeError("kernel boot identity is empty")
        return hashlib.sha256(boot_id.encode()).hexdigest()
    try:
        boot_time = int(_psutil().boot_time())
    except Exception as exc:
        raise RuntimeError("cannot read boot identity") from exc
    return hashlib.sha256(f"boot-time:{boot_time}".encode()).hexdigest()


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    cwd: Path | None
    cmdline: str
    comm: str


def process_identity(pid: int) -> ProcessIdentity:
    """Return the working directory, command line, and name of a process."""
    if IS_LINUX:
        cwd = Path(f"/proc/{pid}/cwd").resolve()
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(
            b"\0", b" ").decode()
        comm = Path(f"/proc/{pid}/comm").read_text().strip()
        return ProcessIdentity(pid, cwd, cmdline, comm)
    psutil = _psutil()
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            raw_cwd = process.cwd()
            cmdline = " ".join(process.cmdline())
            comm = process.name()
    except (psutil.Error, OSError) as exc:
        raise OSError(f"process {pid} is unavailable") from exc
    return ProcessIdentity(
        pid, Path(raw_cwd).resolve() if raw_cwd else None, cmdline, comm)


def owned(pid: int, cwd: Path, marker: str) -> bool:
    """True when ``pid`` runs in ``cwd`` with ``marker`` in its command line."""
    if IS_LINUX:
        try:
            actual_cwd = Path(f"/proc/{pid}/cwd").resolve()
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(
                b"\0", b" ").decode()
            return actual_cwd == cwd.resolve() and marker in cmdline
        except OSError:
            return False
    try:
        identity = process_identity(pid)
    except OSError:
        return False
    return identity.cwd == cwd.resolve() and marker in identity.cmdline


def iter_pids() -> Iterator[int]:
    if IS_LINUX:
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                yield int(entry.name)
        return
    yield from _psutil().pids()


def discover_pid(cwd: Path, marker: str) -> int | None:
    for pid in iter_pids():
        if owned(pid, cwd, marker):
            return pid
    return None


def pid_exists(pid: int) -> bool:
    if IS_WINDOWS:  # pragma: no cover - Windows only
        return _psutil().pid_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_pid_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_exists(pid):
            return True
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    return not pid_exists(pid)


def terminate_tree(pid: int, *, grace_seconds: float = 20.0,
                   kill_seconds: float = 5.0) -> None:
    """Portable termination for platforms without process-group signals.

    Linux keeps its own ``killpg``/``pidfd`` implementation in the supervisor;
    this is the macOS and Windows path.
    """
    psutil = _psutil()
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    processes = [root, *root.children(recursive=True)]
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(processes, timeout=grace_seconds)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(alive, timeout=kill_seconds)
    if alive:
        raise RuntimeError(f"service PID {pid} did not exit after kill")


__all__ = [
    "IS_LINUX", "IS_WINDOWS", "ProcessIdentity", "boot_id_hash",
    "detached_popen_kwargs", "discover_pid", "hidden_console_kwargs",
    "iter_pids", "load_average_1m", "memory_info_mib", "owned",
    "physical_memory_bytes", "pid_exists", "process_identity", "runtime_dir",
    "terminate_tree", "wait_pid_exit",
]
