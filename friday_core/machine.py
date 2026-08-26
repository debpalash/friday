"""Permissioned, receipt-producing local machine operations.

The model never receives ambient filesystem authority. A user-approved grant
bounds every operation to one canonical directory and a small permission set.
Exact grant targets and rollback payloads are encrypted at rest. File access is
performed through directory file descriptors with ``O_NOFOLLOW`` so a symlink
swap cannot redirect a previously authorized action outside its grant.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .documents import (MAX_DOCUMENT_CHARS, document_source_sha256,
                        extract_document)
from .graph import GraphStore, canonical_json, new_id, sha256_text, utc_now
from .images import (MAX_OCR_CHARS, extract_image_text, image_source_sha256,
                     prepare_native_vision_image)
from .step_payloads import StepPayloadCipher


GRANT_PERMISSIONS = {"inspect", "list", "read", "write"}
_OPERATION_ID = re.compile(r"[A-Za-z0-9_.:-]{8,200}\Z")
_SENSITIVE_COMPONENTS = {
    ".aws", ".azure", ".gnupg", ".kube", ".password-store", ".ssh",
    "keyrings", "secrets", "state",
}
_SENSITIVE_NAMES = {
    ".env", "credentials", "credentials.json", "id_rsa", "id_ed25519",
    "session.json",
}
_VIRTUAL_ROOTS = tuple(Path(item) for item in ("/dev", "/proc", "/run", "/sys"))


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("grant expiry must include a timezone")
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class AuthorizedPath:
    grant_id: str
    root: Path
    target: Path
    allow_sensitive: bool


@dataclass(frozen=True, repr=False)
class NativeVisionInput:
    """Ephemeral sanitized image bytes plus independently verified provenance."""

    grant_id: str
    path: str
    encoded: bytes = field(repr=False)
    provenance: dict[str, Any] = field(repr=False)

    def __repr__(self) -> str:
        return (f"NativeVisionInput(grant_id={self.grant_id!r}, "
                f"image_bytes={len(self.encoded)})")


class OperatorGrantService:
    """Durable encrypted path grants, separate from per-action approvals."""

    def __init__(self, graph: GraphStore, project_root: str | Path, *,
                 home: str | Path | None = None,
                 state_root: str | Path | None = None,
                 cipher: StepPayloadCipher | None = None):
        self.graph = graph
        self.project_root = Path(project_root).expanduser().resolve()
        self.home = Path(home or Path.home()).expanduser().resolve()
        self.state_root = Path(
            state_root or graph.path.parent).expanduser().resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.state_root, 0o700)
        self._cipher = cipher or StepPayloadCipher(
            self.state_root / "operator-grants.key")

    @staticmethod
    def _context(grant_id: str, target_sha256: str) -> str:
        return f"operator-grant\0{grant_id}\0{target_sha256}"

    def _scope_kind(self, target: Path) -> str:
        if _is_within(target, self.project_root):
            return "project"
        if _is_within(target, self.home):
            return "user"
        return "system"

    def _redacted_target(self, target: Path, scope_kind: str) -> str:
        if scope_kind == "project":
            relative = target.relative_to(self.project_root)
            return "$PROJECT" + (f"/{relative}" if relative.parts else "")
        if scope_kind == "user":
            relative = target.relative_to(self.home)
            return "$HOME" + (f"/{relative}" if relative.parts else "")
        return f"$SYSTEM/{target.name or 'root'}"

    @staticmethod
    def _validate_permissions(permissions: list[str]) -> list[str]:
        normalized = sorted(set(str(item) for item in permissions))
        if not normalized:
            raise ValueError("an operator grant needs at least one permission")
        unknown = set(normalized) - GRANT_PERMISSIONS
        if unknown:
            raise ValueError(f"unsupported operator permissions: {sorted(unknown)}")
        return normalized

    def _validate_root(self, value: str | Path, *,
                       allow_sensitive: bool) -> Path:
        raw = Path(value).expanduser()
        if not raw.is_absolute():
            raise ValueError("operator grant targets must be absolute paths")
        lexical = Path(os.path.abspath(raw))
        target = raw.resolve(strict=True)
        if lexical != target:
            raise ValueError("symbolic-link grant targets are not accepted")
        if not target.is_dir():
            raise ValueError("operator path grants must target an existing directory")
        if target == Path("/"):
            raise ValueError("granting the filesystem root is not allowed")
        if any(_is_within(target, root) for root in _VIRTUAL_ROOTS):
            raise ValueError("virtual kernel/device filesystems cannot be granted")
        if self.is_sensitive(target) and not allow_sensitive:
            raise ValueError(
                "this target is sensitive; an exact sensitive grant is required")
        return target

    def is_sensitive(self, target: Path) -> bool:
        lowered = {part.casefold() for part in target.parts}
        name = target.name.casefold()
        if lowered & _SENSITIVE_COMPONENTS:
            return True
        if (name in _SENSITIVE_NAMES or name.startswith(".env.")
                or "secret" in name or "token" in name or "api_key" in name):
            return True
        # Friday's runtime authority is sensitive even though it is under the
        # otherwise ordinary project directory.
        return _is_within(target, self.project_root / "state")

    def grant_path(self, path: str | Path, permissions: list[str], *,
                   allow_sensitive: bool = False,
                   source_task_id: str | None = None,
                   expires_at: str | None = None,
                   actor: str = "user") -> dict[str, Any]:
        allowed = self._validate_permissions(permissions)
        target = self._validate_root(path, allow_sensitive=allow_sensitive)
        expiry = _parse_time(expires_at)
        if expiry is not None and expiry <= datetime.now(UTC):
            raise ValueError("operator grant expiry must be in the future")
        if source_task_id is not None and self.graph.get_node(source_task_id) is None:
            raise ValueError("operator grant source task does not exist")
        grant_id = new_id("grant")
        target_text = str(target)
        target_hash = sha256_text(target_text)
        scope_kind = self._scope_kind(target)
        preview = self._redacted_target(target, scope_kind)
        encrypted = self._cipher.seal(
            {"target": target_text},
            context=self._context(grant_id, target_hash))
        now = utc_now()
        body = {
            "grant_id": grant_id, "scope_kind": scope_kind,
            "target": preview, "target_sha256": target_hash,
            "permissions": allowed, "allow_sensitive": bool(allow_sensitive),
            "expires_at": expires_at,
        }
        with self.graph.transaction() as conn:
            event_id, seq = self.graph.append_event(
                conn, "operator.grant_created", body, actor=actor,
                task_id=source_task_id)
            self.graph.append_node(
                conn, "operator_grant", body, event_id=event_id,
                node_id=grant_id)
            if source_task_id:
                self.graph.append_edge(
                    conn, source_task_id, "authorizes", grant_id,
                    event_id=event_id)
            conn.execute(
                """INSERT INTO operator_grants
                   (grant_id,scope_kind,target_ciphertext,target_redacted,
                    target_sha256,permissions_json,allow_sensitive,status,
                    source_task_id,expires_at,created_at,updated_at,last_event_seq)
                   VALUES (?,?,?,?,?,?,?,'active',?,?,?,?,?)""",
                (grant_id, scope_kind, encrypted, preview, target_hash,
                 canonical_json(allowed), int(bool(allow_sensitive)),
                 source_task_id, expires_at, now, now, seq))
        return body | {"status": "active", "target": target_text}

    def _target(self, row: Any) -> Path:
        payload = self._cipher.open(
            row["target_ciphertext"],
            context=self._context(row["grant_id"], row["target_sha256"]))
        target = str(payload.get("target") or "")
        if sha256_text(target) != row["target_sha256"]:
            raise RuntimeError("operator grant target hash does not match")
        return Path(target)

    @staticmethod
    def _expired(row: Any) -> bool:
        expiry = _parse_time(row["expires_at"])
        return expiry is not None and expiry <= datetime.now(UTC)

    def list_grants(self, *, reveal_targets: bool = False) -> list[dict[str, Any]]:
        with self.graph._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM operator_grants ORDER BY created_at").fetchall()
        values = []
        for row in rows:
            status = "expired" if row["status"] == "active" and self._expired(row) \
                else row["status"]
            item = {
                "grant_id": row["grant_id"], "scope_kind": row["scope_kind"],
                "target": (str(self._target(row)) if reveal_targets
                           else row["target_redacted"]),
                "target_sha256": row["target_sha256"],
                "permissions": json.loads(row["permissions_json"]),
                "allow_sensitive": bool(row["allow_sensitive"]),
                "status": status, "expires_at": row["expires_at"],
                "created_at": row["created_at"],
            }
            values.append(item)
        return values

    def revoke(self, grant_id: str, *, actor: str = "user") -> dict[str, Any]:
        with self.graph.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM operator_grants WHERE grant_id=?", (grant_id,)
            ).fetchone()
            if row is None:
                raise ValueError("operator grant does not exist")
            if row["status"] != "active":
                return {"grant_id": grant_id, "status": str(row["status"]),
                        "already_inactive": True}
            body = {"grant_id": grant_id, "status": "revoked"}
            event_id, seq = self.graph.append_event(
                conn, "operator.grant_revoked", body, actor=actor,
                task_id=row["source_task_id"])
            conn.execute(
                """UPDATE operator_grants SET status='revoked',updated_at=?,
                   last_event_seq=? WHERE grant_id=?""",
                (utc_now(), seq, grant_id))
        return body

    def authorize(self, target: Path, permission: str) -> AuthorizedPath:
        if permission not in GRANT_PERMISSIONS:
            raise ValueError("unsupported operator permission")
        canonical = target.resolve(strict=False)
        with self.graph._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM operator_grants WHERE status='active'").fetchall()
        candidates: list[tuple[int, Any, Path]] = []
        for row in rows:
            if self._expired(row):
                continue
            permissions = set(json.loads(row["permissions_json"]))
            if permission not in permissions:
                continue
            root = self._target(row)
            if _is_within(canonical, root):
                candidates.append((len(root.parts), row, root))
        if not candidates:
            raise PermissionError(
                f"no active {permission} grant covers the requested path")
        _, row, root = max(candidates, key=lambda item: item[0])
        allow_sensitive = bool(row["allow_sensitive"])
        if self.is_sensitive(canonical) and not allow_sensitive:
            raise PermissionError(
                "the path is sensitive and this grant does not include it")
        return AuthorizedPath(
            grant_id=str(row["grant_id"]), root=root, target=canonical,
            allow_sensitive=allow_sensitive)


class MachineOperator:
    """Race-resistant bounded filesystem broker with independent receipts."""

    MAX_READ_BYTES = 256_000
    MAX_WRITE_BYTES = 1_048_576
    MAX_BACKUP_BYTES = 3_000_000
    MAX_LIST_ENTRIES = 500

    def __init__(self, grants: OperatorGrantService, *,
                 state_root: str | Path | None = None,
                 backup_cipher: StepPayloadCipher | None = None):
        self.grants = grants
        self.state_root = Path(
            state_root or grants.state_root).expanduser().resolve()
        self.backup_root = self.state_root / "operator-backups"
        self.backup_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.backup_root, 0o700)
        self._backup_cipher = backup_cipher or StepPayloadCipher(
            self.state_root / "operator-backups.key")

    @staticmethod
    def _lexical_path(value: str | Path) -> Path:
        raw = Path(value).expanduser()
        if not raw.is_absolute():
            raise ValueError("machine paths must be absolute")
        return Path(os.path.abspath(raw))

    def _authorize(self, value: str | Path, permission: str) -> AuthorizedPath:
        lexical = self._lexical_path(value)
        canonical = lexical.resolve(strict=False)
        if lexical != canonical:
            raise PermissionError("symbolic-link paths are not accepted")
        return self.grants.authorize(canonical, permission)

    @staticmethod
    def _open_root(root: Path) -> int:
        return os.open(
            root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0))

    def _open_parent(self, authorized: AuthorizedPath) -> tuple[int, str]:
        relative = authorized.target.relative_to(authorized.root)
        if not relative.parts:
            raise ValueError("the grant root itself is not a file target")
        current = self._open_root(authorized.root)
        try:
            for component in relative.parts[:-1]:
                if component in {"", ".", ".."}:
                    raise PermissionError("invalid path component")
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current)
                os.close(current)
                current = next_fd
            return current, relative.parts[-1]
        except Exception:
            os.close(current)
            raise

    def _open_target(self, authorized: AuthorizedPath, *,
                     directory: bool = False) -> int:
        relative = authorized.target.relative_to(authorized.root)
        if not relative.parts:
            return self._open_root(authorized.root)
        parent, leaf = self._open_parent(authorized)
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        if directory:
            flags |= os.O_DIRECTORY
        try:
            return os.open(leaf, flags, dir_fd=parent)
        finally:
            os.close(parent)

    @staticmethod
    def _kind(mode: int) -> str:
        if stat.S_ISREG(mode):
            return "file"
        if stat.S_ISDIR(mode):
            return "directory"
        if stat.S_ISLNK(mode):
            return "symlink"
        return "special"

    @staticmethod
    def _read_fd(fd: int, limit: int) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while total <= limit:
            chunk = os.read(fd, min(65_536, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        data = b"".join(chunks)
        if len(data) > limit:
            raise ValueError(f"file exceeds the {limit}-byte safety limit")
        return data

    def inspect(self, path: str | Path) -> dict[str, Any]:
        authorized = self._authorize(path, "inspect")
        # O_PATH avoids blocking on a FIFO/device while still producing metadata.
        relative = authorized.target.relative_to(authorized.root)
        if not relative.parts:
            fd = os.open(
                authorized.root, getattr(os, "O_PATH", os.O_RDONLY)
                | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
        else:
            parent, leaf = self._open_parent(authorized)
            try:
                fd = os.open(
                    leaf, getattr(os, "O_PATH", os.O_RDONLY) | os.O_CLOEXEC
                    | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
            finally:
                os.close(parent)
        try:
            info = os.fstat(fd)
        finally:
            os.close(fd)
        return {
            "status": "ok", "verified": True,
            "grant_id": authorized.grant_id, "path": str(authorized.target),
            "kind": self._kind(info.st_mode), "bytes": int(info.st_size),
            "mode": stat.S_IMODE(info.st_mode),
            "mtime_ns": int(info.st_mtime_ns),
        }

    def list_path(self, path: str | Path, *, limit: int = 200) -> dict[str, Any]:
        if not 1 <= int(limit) <= self.MAX_LIST_ENTRIES:
            raise ValueError(
                f"list limit must be between 1 and {self.MAX_LIST_ENTRIES}")
        authorized = self._authorize(path, "list")
        fd = self._open_target(authorized, directory=True)
        try:
            names = sorted(os.listdir(fd), key=str.casefold)
            entries = []
            omitted_sensitive = 0
            for name in names:
                child = authorized.target / name
                if self.grants.is_sensitive(child) and not authorized.allow_sensitive:
                    omitted_sensitive += 1
                    continue
                try:
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                except OSError:
                    continue
                entries.append({
                    "name": name, "kind": self._kind(info.st_mode),
                    "bytes": int(info.st_size),
                })
                if len(entries) >= int(limit):
                    break
        finally:
            os.close(fd)
        return {
            "status": "ok", "verified": True,
            "grant_id": authorized.grant_id, "path": str(authorized.target),
            "entries": entries, "truncated": len(entries) < len(names),
            "omitted_sensitive": omitted_sensitive,
        }

    def read_text(self, path: str | Path, *,
                  max_bytes: int = 64_000) -> dict[str, Any]:
        if not 1 <= int(max_bytes) <= self.MAX_READ_BYTES:
            raise ValueError(
                f"read limit must be between 1 and {self.MAX_READ_BYTES}")
        authorized = self._authorize(path, "read")
        fd = self._open_target(authorized)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("only regular files can be read")
            data = self._read_fd(fd, int(max_bytes))
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns) != (
                after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns):
            raise RuntimeError("file changed while it was being read")
        if b"\0" in data:
            raise ValueError("binary files are not available through read_text")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("file is not valid UTF-8 text") from exc
        return {
            "status": "ok", "verified": True,
            "grant_id": authorized.grant_id, "path": str(authorized.target),
            "text": text, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def read_document(self, path: str | Path, *,
                      max_chars: int = 80_000) -> dict[str, Any]:
        if (isinstance(max_chars, bool) or not isinstance(max_chars, int)
                or not 1 <= max_chars <= MAX_DOCUMENT_CHARS):
            raise ValueError(
                f"document character limit must be between 1 and "
                f"{MAX_DOCUMENT_CHARS}")
        authorized = self._authorize(path, "read")
        fd = self._open_target(authorized)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("only regular document files can be read")
            extracted = extract_document(
                fd, authorized.target.name, max_chars=max_chars)
            after_hash = document_source_sha256(fd, int(before.st_size))
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                after.st_ctime_ns):
            raise RuntimeError("document changed while it was being extracted")
        if (extracted.get("source_bytes") != before.st_size
                or extracted.get("source_sha256") != after_hash):
            raise RuntimeError("document content changed during extraction")
        return {
            "status": "ok", "verified": True,
            "grant_id": authorized.grant_id, "path": str(authorized.target),
            **extracted,
        }

    def ocr_image(self, path: str | Path, *,
                  max_chars: int = 80_000) -> dict[str, Any]:
        if (isinstance(max_chars, bool) or not isinstance(max_chars, int)
                or not 1 <= max_chars <= MAX_OCR_CHARS):
            raise ValueError(
                f"OCR character limit must be between 1 and {MAX_OCR_CHARS}")
        authorized = self._authorize(path, "read")
        fd = self._open_target(authorized)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("only regular image files can be read")
            extracted = extract_image_text(
                fd, authorized.target.name, max_chars=max_chars)
            after_hash = image_source_sha256(fd, int(before.st_size))
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                after.st_ctime_ns):
            raise RuntimeError("image changed while OCR was running")
        if (extracted.get("source_bytes") != before.st_size
                or extracted.get("source_sha256") != after_hash):
            raise RuntimeError("image content changed during OCR")
        return {
            "status": "ok", "verified": True,
            "grant_id": authorized.grant_id, "path": str(authorized.target),
            **extracted,
        }

    def native_vision_image(self, path: str | Path, *,
                            max_side: int) -> NativeVisionInput:
        """Capture one exact-granted image as a sandbox-sanitized snapshot."""
        authorized = self._authorize(path, "read")
        fd = self._open_target(authorized)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("only regular image files can be understood")
            encoded, provenance = prepare_native_vision_image(
                fd, authorized.target.name, max_side=max_side)
            after_hash = image_source_sha256(fd, int(before.st_size))
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                after.st_ctime_ns):
            raise RuntimeError("image changed while it was being sanitized")
        if (provenance.get("source_bytes") != before.st_size
                or provenance.get("source_sha256") != after_hash):
            raise RuntimeError("image content changed during sanitization")
        return NativeVisionInput(
            grant_id=authorized.grant_id, path=str(authorized.target),
            encoded=encoded, provenance=provenance)

    @staticmethod
    def _backup_name(operation_id: str) -> str:
        if not _OPERATION_ID.fullmatch(operation_id):
            raise ValueError("operation_id has an invalid format")
        return hashlib.sha256(operation_id.encode()).hexdigest() + ".enc"

    @staticmethod
    def _backup_context(operation_id: str) -> str:
        return f"machine-backup\0{operation_id}"

    def _backup_path(self, operation_id: str) -> Path:
        return self.backup_root / self._backup_name(operation_id)

    def _load_backup(self, operation_id: str) -> dict[str, Any] | None:
        path = self._backup_path(operation_id)
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except FileNotFoundError:
            return None
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError("rollback checkpoint is not a regular file")
            payload = self._read_fd(fd, self.MAX_BACKUP_BYTES).decode("utf-8")
        finally:
            os.close(fd)
        return self._backup_cipher.open(
            payload, context=self._backup_context(operation_id))

    def _store_backup(self, operation_id: str, value: dict[str, Any]) -> None:
        target = self._backup_path(operation_id)
        payload = self._backup_cipher.seal(
            value, context=self._backup_context(operation_id))
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
                 | getattr(os, "O_NOFOLLOW", 0))
        fd = os.open(target, flags, 0o600)
        try:
            encoded = payload.encode()
            offset = 0
            while offset < len(encoded):
                offset += os.write(fd, encoded[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)

    def _update_backup(self, operation_id: str, value: dict[str, Any]) -> None:
        """Atomically advance the authenticated operation journal."""
        name = self._backup_name(operation_id)
        payload = self._backup_cipher.seal(
            value, context=self._backup_context(operation_id)).encode("utf-8")
        root_fd = self._open_root(self.backup_root)
        try:
            self._atomic_replace(root_fd, name, payload, 0o600)
        finally:
            os.close(root_fd)

    @staticmethod
    def _existing_bytes(parent: int, leaf: str) -> tuple[bool, bytes, int]:
        try:
            fd = os.open(
                leaf, os.O_RDONLY | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
        except FileNotFoundError:
            return False, b"", 0o600
        except OSError as exc:
            try:
                observed = os.stat(
                    leaf, dir_fd=parent, follow_symlinks=False)
            except OSError:
                raise exc
            if not stat.S_ISREG(observed.st_mode):
                raise ValueError(
                    "write target must be a regular file or absent") from exc
            raise
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("write target must be a regular file or absent")
            if before.st_size > MachineOperator.MAX_WRITE_BYTES:
                raise ValueError(
                    "existing file is too large for reversible replacement")
            data = MachineOperator._read_fd(fd, MachineOperator.MAX_WRITE_BYTES)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                after.st_ctime_ns):
            raise RuntimeError("file changed while its rollback checkpoint was read")
        return True, data, stat.S_IMODE(before.st_mode)

    @staticmethod
    def _atomic_replace(parent: int, leaf: str, data: bytes, mode: int) -> None:
        stage = f".friday-stage-{secrets.token_hex(12)}"
        fd = os.open(
            stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0), mode, dir_fd=parent)
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(fd, data[offset:])
            os.fsync(fd)
        except Exception:
            try:
                os.unlink(stage, dir_fd=parent)
            except OSError:
                pass
            raise
        finally:
            os.close(fd)
        os.replace(stage, leaf, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)

    def write_text(self, path: str | Path, content: str, *,
                   operation_id: str) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("write content must be text")
        data = content.encode("utf-8")
        if len(data) > self.MAX_WRITE_BYTES:
            raise ValueError(
                f"write content exceeds {self.MAX_WRITE_BYTES} bytes")
        self._backup_name(operation_id)
        authorized = self._authorize(path, "write")
        parent, leaf = self._open_parent(authorized)
        idempotent_replay = False
        try:
            existed, before, mode = self._existing_bytes(parent, leaf)
            before_hash = hashlib.sha256(before).hexdigest() if existed else None
            after_hash = hashlib.sha256(data).hexdigest()
            backup = self._load_backup(operation_id)
            if backup is None:
                backup = {
                    "path": str(authorized.target), "grant_id": authorized.grant_id,
                    "existed": existed,
                    "before_b64": base64.b64encode(before).decode("ascii"),
                    "before_sha256": before_hash, "before_mode": mode,
                    "after_sha256": after_hash, "state": "prepared",
                }
                self._store_backup(operation_id, backup)
            elif (backup.get("path") != str(authorized.target)
                  or backup.get("after_sha256") != after_hash):
                raise RuntimeError(
                    "operation_id is already bound to a different exact write")
            state = str(backup.get("state") or "prepared")
            if state == "rolled_back":
                raise RuntimeError("operation_id has already been rolled back")
            if state not in {"prepared", "applied"}:
                raise RuntimeError("operation journal has an invalid state")
            original_existed = bool(backup.get("existed"))
            expected_before_hash = backup.get("before_sha256")
            current_hash = hashlib.sha256(before).hexdigest() if existed else None
            matches_before = (existed == original_existed
                              and current_hash == expected_before_hash)
            matches_after = existed and current_hash == after_hash
            if state == "applied":
                if not matches_after:
                    raise RuntimeError(
                        "write replay refused because the file changed afterward")
                idempotent_replay = True
            elif matches_after:
                # The atomic replacement completed before a prior process could
                # advance its journal. Reconcile it without invoking the effect.
                backup["state"] = "applied"
                self._update_backup(operation_id, backup)
                idempotent_replay = True
            elif not matches_before:
                raise RuntimeError(
                    "write refused because the target changed after checkpointing")
            else:
                self._atomic_replace(
                    parent, leaf, data, int(backup.get("before_mode", mode)))
            verified_exists, verified, _ = self._existing_bytes(parent, leaf)
        finally:
            os.close(parent)
        verified_hash = hashlib.sha256(verified).hexdigest()
        if not verified_exists or verified_hash != after_hash:
            raise RuntimeError("filesystem write verification failed")
        if backup.get("state") != "applied":
            backup["state"] = "applied"
            self._update_backup(operation_id, backup)
        return {
            "status": "ok", "verified": True,
            "grant_id": authorized.grant_id, "path": str(authorized.target),
            "bytes": len(data), "before_sha256": backup.get("before_sha256"),
            "after_sha256": after_hash, "rollback_operation_id": operation_id,
            "idempotent_replay": idempotent_replay,
        }

    def rollback(self, operation_id: str) -> dict[str, Any]:
        backup = self._load_backup(operation_id)
        if backup is None:
            raise ValueError("rollback checkpoint does not exist")
        authorized = self._authorize(str(backup["path"]), "write")
        parent, leaf = self._open_parent(authorized)
        try:
            exists, current, _ = self._existing_bytes(parent, leaf)
            current_hash = hashlib.sha256(current).hexdigest() if exists else None
            before_hash = backup.get("before_sha256")
            after_hash = backup.get("after_sha256")
            original_existed = bool(backup.get("existed"))
            already_restored = (
                (original_existed and exists and current_hash == before_hash)
                or (not original_existed and not exists))
            state = str(backup.get("state") or "prepared")
            if state not in {"prepared", "applied", "rolled_back"}:
                raise RuntimeError("operation journal has an invalid state")
            if not already_restored:
                if not exists or current_hash != after_hash:
                    raise RuntimeError(
                        "rollback refused because the file changed after Friday's write")
                if original_existed:
                    original = base64.b64decode(
                        str(backup["before_b64"]), validate=True)
                    if hashlib.sha256(original).hexdigest() != before_hash:
                        raise RuntimeError("rollback checkpoint hash does not match")
                    self._atomic_replace(
                        parent, leaf, original, int(backup["before_mode"]))
                else:
                    os.unlink(leaf, dir_fd=parent)
                    os.fsync(parent)
            final_exists, final, _ = self._existing_bytes(parent, leaf)
        finally:
            os.close(parent)
        final_hash = hashlib.sha256(final).hexdigest() if final_exists else None
        if final_exists != original_existed or final_hash != before_hash:
            raise RuntimeError("rollback verification failed")
        if backup.get("state") != "rolled_back":
            backup["state"] = "rolled_back"
            self._update_backup(operation_id, backup)
        return {
            "status": "ok", "verified": True,
            "grant_id": authorized.grant_id, "path": str(authorized.target),
            "restored_sha256": final_hash,
            "restored_absence": not original_existed,
            "already_restored": already_restored,
            "operation_id": operation_id,
        }
