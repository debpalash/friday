"""Per-user secret storage backed by the operating system's keyring.

Linux keeps the exact ``secret-tool`` attributes Friday used before the port
so existing items are found. macOS uses the login keychain through
``security`` in interactive mode so the secret never appears in a process
list. Windows uses the Credential Manager through ``advapi32`` and needs no
extra packages. When no keyring is usable the store fails closed.
"""

from __future__ import annotations

import base64
import secrets
import subprocess
from typing import Callable

from .host import HostPlatform, current_host
from .procs import hidden_console_kwargs

Runner = Callable[..., subprocess.CompletedProcess]

_SERVICE = "friday"
_LABEL = "Friday {purpose}"


class SecretStoreUnavailable(RuntimeError):
    """The host keyring refused or cannot hold Friday's secret."""


def _linux_get(purpose: str, runner: Runner) -> bytes | None:
    lookup = runner(
        ["secret-tool", "lookup", "application", _SERVICE, "purpose", purpose],
        capture_output=True, timeout=5)
    return bytes(lookup.stdout).strip() or None


def _linux_set(purpose: str, value: bytes, runner: Runner) -> None:
    stored = runner(
        ["secret-tool", "store", f"--label={_LABEL.format(purpose=purpose)}",
         "application", _SERVICE, "purpose", purpose],
        input=value + b"\n", capture_output=True, timeout=10)
    if stored.returncode:
        raise SecretStoreUnavailable("desktop keyring refused the secret")


def _macos_get(purpose: str, runner: Runner) -> bytes | None:
    found = runner(
        ["security", "find-generic-password", "-s", _SERVICE, "-a", purpose, "-w"],
        capture_output=True, timeout=10)
    if found.returncode:
        return None
    return bytes(found.stdout).strip() or None


def _macos_set(purpose: str, value: bytes, runner: Runner) -> None:
    # `security -i` reads commands from standard input, keeping the secret
    # out of argv. -U updates an existing item in place.
    command = (
        f"add-generic-password -U -s {_SERVICE} -a {purpose} "
        f"-l \"{_LABEL.format(purpose=purpose)}\" -w {value.decode('ascii')}\n")
    stored = runner(["security", "-i"], input=command.encode("ascii"),
                    capture_output=True, timeout=10)
    if stored.returncode:
        raise SecretStoreUnavailable("login keychain refused the secret")


def _windows_target(purpose: str) -> str:
    return f"{_SERVICE}/{purpose}"


def _windows_get(purpose: str) -> bytes | None:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.windll.advapi32  # type: ignore[attr-defined]

    class CREDENTIAL(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p), ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    pointer = ctypes.POINTER(CREDENTIAL)()
    if not advapi32.CredReadW(_windows_target(purpose), 1, 0, ctypes.byref(pointer)):
        return None
    try:
        record = pointer.contents
        size = int(record.CredentialBlobSize)
        return bytes(record.CredentialBlob[:size]).strip() or None
    finally:
        advapi32.CredFree(pointer)


def _windows_set(purpose: str, value: bytes) -> None:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.windll.advapi32  # type: ignore[attr-defined]

    class CREDENTIAL(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p), ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    blob = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    record = CREDENTIAL()
    record.Type = 1  # CRED_TYPE_GENERIC
    record.TargetName = _windows_target(purpose)
    record.Comment = _LABEL.format(purpose=purpose)
    record.CredentialBlobSize = len(value)
    record.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
    record.Persist = 2  # CRED_PERSIST_LOCAL_MACHINE
    record.UserName = _SERVICE
    if not advapi32.CredWriteW(ctypes.byref(record), 0):
        raise SecretStoreUnavailable("Credential Manager refused the secret")


def _windows_delete(purpose: str) -> bool:  # pragma: no cover - Windows only
    import ctypes

    advapi32 = ctypes.windll.advapi32  # type: ignore[attr-defined]
    return bool(advapi32.CredDeleteW(_windows_target(purpose), 1, 0))


class SecretStore:
    """Read, create, and remove one named secret in the host keyring."""

    def __init__(self, *, host: HostPlatform | None = None,
                 runner: Runner = subprocess.run) -> None:
        self.host = host or current_host()
        self._runner = runner

    def _run(self, command, **kwargs):
        try:
            return self._runner(command, **kwargs, **hidden_console_kwargs())
        except (OSError, subprocess.SubprocessError) as exc:
            raise SecretStoreUnavailable(
                f"host keyring is unavailable ({type(exc).__name__})") from exc

    def get(self, purpose: str) -> bytes | None:
        if self.host.is_linux:
            return _linux_get(purpose, self._run)
        if self.host.is_macos:
            return _macos_get(purpose, self._run)
        return _windows_get(purpose)

    def set(self, purpose: str, value: bytes) -> None:
        if not value or any(byte <= 0x20 or byte > 0x7e for byte in value):
            raise ValueError("secret values must be printable ASCII")
        if self.host.is_linux:
            _linux_set(purpose, value, self._run)
        elif self.host.is_macos:
            _macos_set(purpose, value, self._run)
        else:
            _windows_set(purpose, value)

    def delete(self, purpose: str) -> bool:
        if self.host.is_linux:
            result = self._run(
                ["secret-tool", "clear", "application", _SERVICE,
                 "purpose", purpose], capture_output=True, timeout=10)
            return result.returncode == 0
        if self.host.is_macos:
            result = self._run(
                ["security", "delete-generic-password", "-s", _SERVICE,
                 "-a", purpose], capture_output=True, timeout=10)
            return result.returncode == 0
        return _windows_delete(purpose)

    def get_or_create(self, purpose: str, length: int) -> bytes:
        """Return ``length`` key bytes, minting and storing them on first use."""
        value = self.get(purpose)
        if not value:
            value = base64.urlsafe_b64encode(secrets.token_bytes(length))
            self.set(purpose, value)
        try:
            key = base64.urlsafe_b64decode(value)
        except Exception as exc:
            raise SecretStoreUnavailable(
                f"stored {purpose} secret is invalid") from exc
        if len(key) < length:
            raise SecretStoreUnavailable(f"stored {purpose} secret is too short")
        return key[:length]


__all__ = ["SecretStore", "SecretStoreUnavailable"]
