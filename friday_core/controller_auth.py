"""Durable paired-controller authentication for Friday's control plane.

Pairing and session proofs use a controller-owned P-256 key.  Browser private
keys can therefore remain non-extractable in WebCrypto/IndexedDB.  SQLite stores
only public keys, hashes, lifecycle state, and expiring session-token digests;
raw pairing and session bearers are returned once and never journaled.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import stat
import subprocess
import tempfile
import unicodedata
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .graph import GraphStore, canonical_json, new_id


_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_OPAQUE_RE = re.compile(r"[A-Za-z0-9_-]{8,160}\Z")
_SECRET_RE = re.compile(r"[A-Za-z0-9_-]{20,160}\Z")
_PAIRING_PREFIX = "fpair1"
_SESSION_PREFIX = "fsess1"
_KEY_ALGORITHM = "ecdsa-p256-sha256"
_SPKI_P256_PREFIX = bytes.fromhex(
    "3059301306072a8648ce3d020106082a8648ce3d03010703420004")


class ControllerAuthError(PermissionError):
    """Stable, privacy-safe paired-controller failure."""

    def __init__(self, code: str = "controller_authorization_failed") -> None:
        self.code = code if re.fullmatch(r"[a-z0-9_.:-]{1,80}", code) else (
            "controller_authorization_failed")
        super().__init__(self.code)


@dataclass(frozen=True)
class ControllerPrincipal:
    controller_id: str
    session_id: str
    public_key_sha256: str
    controller_epoch: int
    origin_sha256: str
    transport_binding_sha256: str
    issued_at: str
    idle_expires_at: str
    absolute_expires_at: str


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(
        timespec="microseconds").replace("+00:00", "Z")


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _b64url_decode(value: str, *, exact_bytes: int) -> bytes:
    if not isinstance(value, str) or "=" in value or not _SECRET_RE.fullmatch(value):
        raise ControllerAuthError("controller_key_invalid")
    try:
        decoded = base64.urlsafe_b64decode(
            value + "=" * ((4 - len(value) % 4) % 4))
    except (ValueError, TypeError) as exc:
        raise ControllerAuthError("controller_key_invalid") from exc
    if len(decoded) != exact_bytes:
        raise ControllerAuthError("controller_key_invalid")
    return decoded


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _validate_hash(name: str, value: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_opaque(name: str, value: str) -> str:
    if not isinstance(value, str) or not _OPAQUE_RE.fullmatch(value):
        raise ControllerAuthError(f"{name}_invalid")
    return value


def _controller_label(value: str) -> str:
    label = " ".join(str(value or "").strip().split())
    if (not 1 <= len(label) <= 80
            or len(label.encode("utf-8")) > 240
            or any(unicodedata.category(char).startswith("C") for char in label)):
        raise ValueError("controller label must contain 1 through 80 visible characters")
    return label


def normalize_https_origin(value: str) -> str:
    """Return one canonical HTTPS origin, rejecting proxy-style ambiguity."""
    try:
        parsed = urllib.parse.urlsplit(str(value or ""))
        port = parsed.port
    except ValueError as exc:
        raise ControllerAuthError("controller_origin_invalid") from exc
    if (parsed.scheme.lower() != "https" or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ControllerAuthError("controller_origin_invalid")
    host = parsed.hostname.lower()
    if (any(ord(character) > 127 or character.isspace() for character in host)
            or "%" in host or host.endswith(".") or len(host) > 253):
        raise ControllerAuthError("controller_origin_invalid")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        labels = host.split(".")
        if any(
            not label or len(label) > 63
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
            for label in labels
        ):
            raise ControllerAuthError("controller_origin_invalid")
        authority_host = host
    else:
        host = address.compressed
        authority_host = f"[{host}]" if address.version == 6 else host
    effective_port = 443 if port is None else port
    if not 1 <= effective_port <= 65535:
        raise ControllerAuthError("controller_origin_invalid")
    return f"https://{authority_host}" + (
        "" if effective_port == 443 else f":{effective_port}")


def normalize_public_jwk(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ControllerAuthError("controller_key_invalid")
    allowed = {"kty", "crv", "x", "y", "ext", "key_ops"}
    if set(value) - allowed:
        raise ControllerAuthError("controller_key_invalid")
    if value.get("kty") != "EC" or value.get("crv") != "P-256":
        raise ControllerAuthError("controller_key_invalid")
    x = str(value.get("x") or "")
    y = str(value.get("y") or "")
    _b64url_decode(x, exact_bytes=32)
    _b64url_decode(y, exact_bytes=32)
    if "ext" in value and value.get("ext") is not True:
        raise ControllerAuthError("controller_key_invalid")
    if "key_ops" in value and value.get("key_ops") not in ([], ["verify"]):
        raise ControllerAuthError("controller_key_invalid")
    return {"crv": "P-256", "kty": "EC", "x": x, "y": y}


def public_key_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_text(canonical_json(normalize_public_jwk(value)))


def _raw_ecdsa_to_der(raw: bytes) -> bytes:
    if len(raw) != 64:
        raise ControllerAuthError("controller_signature_invalid")

    def integer(value: bytes) -> bytes:
        stripped = value.lstrip(b"\x00") or b"\x00"
        if stripped[0] & 0x80:
            stripped = b"\x00" + stripped
        return b"\x02" + bytes([len(stripped)]) + stripped

    body = integer(raw[:32]) + integer(raw[32:])
    return b"\x30" + bytes([len(body)]) + body


def verify_p256_signature(
    public_jwk: Mapping[str, Any], payload: bytes, signature_b64url: str, *,
    openssl_path: str = "/usr/bin/openssl",
) -> bool:
    """Verify a WebCrypto P-256 SHA-256 signature with argv-only OpenSSL."""
    try:
        normalized = normalize_public_jwk(public_jwk)
        x = _b64url_decode(normalized["x"], exact_bytes=32)
        y = _b64url_decode(normalized["y"], exact_bytes=32)
        raw_signature = _b64url_decode(signature_b64url, exact_bytes=64)
        signature = _raw_ecdsa_to_der(raw_signature)
    except (ControllerAuthError, ValueError):
        return False
    executable = Path(openssl_path)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError("OpenSSL is required for controller proof verification")
    public_der = _SPKI_P256_PREFIX + x + y
    encoded_public = base64.b64encode(public_der)
    public_pem = (
        b"-----BEGIN PUBLIC KEY-----\n"
        + b"\n".join(encoded_public[index:index + 64]
                       for index in range(0, len(encoded_public), 64))
        + b"\n-----END PUBLIC KEY-----\n")
    with tempfile.TemporaryDirectory(prefix="friday-controller-proof-") as root:
        directory = Path(root)
        public_path = directory / "public.pem"
        signature_path = directory / "signature.der"
        public_path.write_bytes(public_pem)
        signature_path.write_bytes(signature)
        os.chmod(public_path, 0o600)
        os.chmod(signature_path, 0o600)
        environment = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}
        try:
            checked = subprocess.run(
                [str(executable), "pkey", "-pubin", "-in", str(public_path),
                 "-pubcheck", "-noout"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=environment, timeout=3,
                check=False)
            if checked.returncode != 0:
                return False
            verified = subprocess.run(
                [str(executable), "dgst", "-sha256", "-verify",
                 str(public_path), "-signature", str(signature_path)],
                input=payload, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=environment, timeout=3,
                check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "controller proof verifier failed") from exc
        return verified.returncode == 0


def _load_or_create_auth_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        key = secrets.token_bytes(32)
        create_flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
                        | getattr(os, "O_NOFOLLOW", 0))
        try:
            descriptor = os.open(path, create_flags, 0o600)
        except FileExistsError:
            return _load_or_create_auth_key(path)
        try:
            os.write(descriptor, key)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | os.O_CLOEXEC
            | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return key
    except OSError as exc:
        raise RuntimeError(
            "controller authentication key identity is invalid") from exc
    try:
        observed = os.fstat(descriptor)
        if (not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.getuid()
                or observed.st_nlink != 1 or observed.st_size != 32):
            raise RuntimeError("controller authentication key identity is invalid")
        os.fchmod(descriptor, 0o600)
        key = os.read(descriptor, 33)
    finally:
        os.close(descriptor)
    if len(key) != 32:
        raise RuntimeError("controller authentication key is invalid")
    return key


class ControllerAuthService:
    def __init__(
        self,
        graph: GraphStore,
        state_root: str | Path,
        *,
        key_provider: Callable[[], bytes] | None = None,
        clock: Callable[[], datetime] | None = None,
        openssl_path: str = "/usr/bin/openssl",
        pairing_ttl_seconds: int = 300,
        challenge_ttl_seconds: int = 60,
        idle_session_ttl_seconds: int = 1800,
        absolute_session_ttl_seconds: int = 43_200,
    ) -> None:
        if not 30 <= pairing_ttl_seconds <= 900:
            raise ValueError("pairing TTL must be between 30 and 900 seconds")
        if not 15 <= challenge_ttl_seconds <= 300:
            raise ValueError("challenge TTL must be between 15 and 300 seconds")
        if not 60 <= idle_session_ttl_seconds <= 3600:
            raise ValueError("idle session TTL must be between 60 and 3600 seconds")
        if not idle_session_ttl_seconds <= absolute_session_ttl_seconds <= 86_400:
            raise ValueError("absolute session TTL is invalid")
        self.graph = graph
        self.state_root = Path(state_root)
        provided = key_provider() if key_provider is not None else (
            _load_or_create_auth_key(self.state_root / "controller-auth.key"))
        if not isinstance(provided, bytes) or len(provided) != 32:
            raise RuntimeError("controller authentication key provider is invalid")
        self._key = provided
        self._clock = clock or (lambda: datetime.now(UTC))
        self.openssl_path = openssl_path
        self.pairing_ttl_seconds = pairing_ttl_seconds
        self.challenge_ttl_seconds = challenge_ttl_seconds
        self.idle_session_ttl_seconds = idle_session_ttl_seconds
        self.absolute_session_ttl_seconds = absolute_session_ttl_seconds

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime):
            raise RuntimeError("controller authentication clock is invalid")
        if value.tzinfo is None:
            raise RuntimeError("controller authentication clock must be timezone-aware")
        return value.astimezone(UTC)

    def _digest(self, kind: str, value: str) -> str:
        return hmac.new(
            self._key, f"friday-controller-{kind}\0{value}".encode("utf-8"),
            hashlib.sha256).hexdigest()

    @staticmethod
    def _parse_pairing_token(value: str) -> tuple[str, str, str]:
        parts = str(value or "").split(".")
        if (len(parts) != 4 or parts[0] != _PAIRING_PREFIX
                or not parts[1].startswith("pairing_")
                or not _OPAQUE_RE.fullmatch(parts[1])
                or not _SECRET_RE.fullmatch(parts[2])
                or not _SECRET_RE.fullmatch(parts[3])):
            raise ControllerAuthError("controller_pairing_invalid")
        return parts[1], parts[2], parts[3]

    @staticmethod
    def _parse_session_token(value: str) -> tuple[str, str]:
        parts = str(value or "").split(".")
        if (len(parts) != 3 or parts[0] != _SESSION_PREFIX
                or not parts[1].startswith("controller_session_")
                or not _OPAQUE_RE.fullmatch(parts[1])
                or not _SECRET_RE.fullmatch(parts[2])):
            raise ControllerAuthError()
        return parts[1], parts[2]

    @staticmethod
    def pairing_proof_payload(
        *, pairing_id: str, challenge: str, label: str,
        public_jwk: Mapping[str, Any], origin: str,
        transport_binding_sha256: str,
    ) -> str:
        return canonical_json({
            "challenge": challenge,
            "controller_label": _controller_label(label),
            "origin": normalize_https_origin(origin),
            "pairing_id": _validate_opaque("controller_pairing", pairing_id),
            "public_jwk": normalize_public_jwk(public_jwk),
            "schema_version": 1,
            "transport_binding_sha256": _validate_hash(
                "transport binding", transport_binding_sha256),
        })

    @staticmethod
    def session_proof_payload(
        *, challenge_id: str, challenge: str, controller_id: str,
        controller_key_sha256: str, origin: str,
        transport_binding_sha256: str,
    ) -> str:
        return canonical_json({
            "challenge": challenge,
            "challenge_id": _validate_opaque(
                "controller_challenge", challenge_id),
            "controller_id": _validate_opaque(
                "controller_identity", controller_id),
            "controller_key_sha256": _validate_hash(
                "controller key", controller_key_sha256),
            "origin": normalize_https_origin(origin),
            "schema_version": 1,
            "transport_binding_sha256": _validate_hash(
                "transport binding", transport_binding_sha256),
        })

    def create_pairing(self, transport_binding_sha256: str) -> dict[str, Any]:
        binding = _validate_hash("transport binding", transport_binding_sha256)
        pairing_id = new_id("pairing")
        code = secrets.token_urlsafe(24)
        challenge = secrets.token_urlsafe(32)
        token = f"{_PAIRING_PREFIX}.{pairing_id}.{code}.{challenge}"
        now_value = self._now()
        now = _iso(now_value)
        expires = _iso(now_value + timedelta(seconds=self.pairing_ttl_seconds))
        with self.graph.transaction() as conn:
            body = {
                "pairing_id": pairing_id, "status": "pending",
                "expires_at": expires,
            }
            _, seq = self.graph.append_event(
                conn, "controller_pairing.created", body,
                actor="local_operator")
            conn.execute(
                """INSERT INTO controller_pairings
                   (pairing_id,code_digest,challenge_sha256,
                    transport_binding_sha256,status,attempts_remaining,
                    created_at,expires_at,created_event_seq,last_event_seq)
                   VALUES (?,?,?,?,'pending',5,?,?,?,?)""",
                (pairing_id, self._digest("pairing-code", code),
                 _sha256_text(challenge), binding, now, expires, seq, seq),
            )
        return {
            "pairing_id": pairing_id,
            "pairing_token": token,
            "expires_at": expires,
            "transport_binding_sha256": binding,
        }

    def _pairing_snapshot(
        self, pairing_token: str, transport_binding_sha256: str,
    ) -> tuple[dict[str, Any], str]:
        binding = _validate_hash("transport binding", transport_binding_sha256)
        pairing_id, code, challenge = self._parse_pairing_token(pairing_token)
        with self.graph._connect() as conn:
            row = conn.execute(
                "SELECT * FROM controller_pairings WHERE pairing_id=?",
                (pairing_id,),
            ).fetchone()
        if row is None:
            raise ControllerAuthError("controller_pairing_invalid")
        current = dict(row)
        expires = _parse_time(current["expires_at"])
        now_value = self._now()
        if (current["status"] == "pending" and expires is not None
                and now_value >= expires):
            self._expire_pairing(pairing_id, now_value)
            raise ControllerAuthError("controller_pairing_invalid")
        valid = bool(
            current["status"] == "pending"
            and int(current["attempts_remaining"]) > 0
            and expires is not None
            and current["transport_binding_sha256"] == binding
            and current["challenge_sha256"] == _sha256_text(challenge)
            and hmac.compare_digest(
                str(current["code_digest"]),
                self._digest("pairing-code", code)))
        if not valid:
            self._fail_pairing_attempt(pairing_id)
            raise ControllerAuthError("controller_pairing_invalid")
        return current, challenge

    def _expire_pairing(self, pairing_id: str, now_value: datetime) -> None:
        with self.graph.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM controller_pairings WHERE pairing_id=?",
                (pairing_id,),
            ).fetchone()
            expires = _parse_time(row["expires_at"] if row else None)
            if (row is None or row["status"] != "pending"
                    or expires is None or now_value < expires):
                return
            body = {"pairing_id": pairing_id, "status": "expired"}
            _, seq = self.graph.append_event(
                conn, "controller_pairing.expired", body,
                actor="controller_auth")
            conn.execute(
                """UPDATE controller_pairings
                      SET status='expired',last_event_seq=?
                    WHERE pairing_id=? AND status='pending'""",
                (seq, pairing_id),
            )

    def _fail_pairing_attempt(self, pairing_id: str) -> None:
        with self.graph.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM controller_pairings WHERE pairing_id=?",
                (pairing_id,),
            ).fetchone()
            if row is None or row["status"] != "pending":
                return
            remaining = max(0, int(row["attempts_remaining"]) - 1)
            status = "cancelled" if remaining == 0 else "pending"
            body = {"pairing_id": pairing_id, "status": status,
                    "attempts_remaining": remaining}
            _, seq = self.graph.append_event(
                conn, "controller_pairing.attempt_rejected", body,
                actor="controller_auth")
            conn.execute(
                """UPDATE controller_pairings
                      SET attempts_remaining=?,status=?,last_event_seq=?
                    WHERE pairing_id=? AND status='pending'""",
                (remaining, status, seq, pairing_id),
            )

    def prepare_pairing(
        self, pairing_token: str, label: str,
        public_jwk: Mapping[str, Any], *, origin: str,
        transport_binding_sha256: str,
    ) -> dict[str, Any]:
        row, challenge = self._pairing_snapshot(
            pairing_token, transport_binding_sha256)
        normalized = normalize_public_jwk(public_jwk)
        payload = self.pairing_proof_payload(
            pairing_id=str(row["pairing_id"]), challenge=challenge,
            label=label, public_jwk=normalized, origin=origin,
            transport_binding_sha256=transport_binding_sha256)
        return {
            "pairing_id": str(row["pairing_id"]),
            "proof_payload": payload,
            "public_key_sha256": public_key_sha256(normalized),
            "expires_at": str(row["expires_at"]),
        }

    def _new_session_values(
        self, now_value: datetime, *, controller_id: str,
    ) -> tuple[str, str, str, str, str]:
        session_id = new_id("controller_session")
        secret = secrets.token_urlsafe(32)
        token = f"{_SESSION_PREFIX}.{session_id}.{secret}"
        idle = _iso(now_value + timedelta(
            seconds=self.idle_session_ttl_seconds))
        absolute = _iso(now_value + timedelta(
            seconds=self.absolute_session_ttl_seconds))
        return session_id, token, self._digest("session-token", token), idle, absolute

    def _insert_session(
        self, conn: Any, *, controller: Mapping[str, Any], origin_sha256: str,
        transport_binding_sha256: str, proof_challenge_sha256: str,
        proof_signature_sha256: str, now_value: datetime,
    ) -> tuple[ControllerPrincipal, str]:
        now = _iso(now_value)
        session_id, token, digest, idle, absolute = self._new_session_values(
            now_value, controller_id=str(controller["controller_id"]))
        body = {
            "session_id": session_id,
            "controller_id": str(controller["controller_id"]),
            "status": "active", "expires_at": idle,
        }
        event_id, seq = self.graph.append_event(
            conn, "controller_session.issued", body, actor="controller_auth")
        self.graph.append_node(
            conn, "controller_session", body, event_id=event_id,
            node_id=session_id)
        conn.execute(
            """INSERT INTO controller_sessions
               (session_id,controller_id,controller_key_sha256,
                controller_epoch,token_digest,origin_sha256,
                transport_binding_sha256,proof_challenge_sha256,
                proof_signature_sha256,status,issued_at,idle_expires_at,
                absolute_expires_at,last_seen_at,issued_event_seq,last_event_seq)
               VALUES (?,?,?,?,?,?,?,?,?,'active',?,?,?,?,?,?)""",
            (session_id, controller["controller_id"],
             controller["public_key_sha256"], int(controller["auth_epoch"]),
             digest, origin_sha256, transport_binding_sha256,
             proof_challenge_sha256, proof_signature_sha256,
             now, idle, absolute, now, seq, seq),
        )
        principal = ControllerPrincipal(
            controller_id=str(controller["controller_id"]),
            session_id=session_id,
            public_key_sha256=str(controller["public_key_sha256"]),
            controller_epoch=int(controller["auth_epoch"]),
            origin_sha256=origin_sha256,
            transport_binding_sha256=transport_binding_sha256,
            issued_at=now, idle_expires_at=idle,
            absolute_expires_at=absolute)
        return principal, token

    def complete_pairing(
        self, pairing_token: str, label: str,
        public_jwk: Mapping[str, Any], signature_b64url: str, *,
        origin: str, transport_binding_sha256: str,
    ) -> dict[str, Any]:
        prepared = self.prepare_pairing(
            pairing_token, label, public_jwk, origin=origin,
            transport_binding_sha256=transport_binding_sha256)
        normalized = normalize_public_jwk(public_jwk)
        if not verify_p256_signature(
                normalized, prepared["proof_payload"].encode("utf-8"),
                signature_b64url, openssl_path=self.openssl_path):
            self._fail_pairing_attempt(str(prepared["pairing_id"]))
            raise ControllerAuthError("controller_proof_invalid")
        pairing_id, code, challenge = self._parse_pairing_token(pairing_token)
        code_digest = self._digest("pairing-code", code)
        challenge_sha256 = _sha256_text(challenge)
        key_json = canonical_json(normalized)
        key_sha256 = _sha256_text(key_json)
        signature_sha256 = _sha256_text(signature_b64url)
        canonical_origin = normalize_https_origin(origin)
        origin_sha256 = _sha256_text(canonical_origin)
        binding = _validate_hash("transport binding", transport_binding_sha256)
        now_value = self._now()
        now = _iso(now_value)
        controller_id = new_id("controller")
        with self.graph.transaction() as conn:
            pairing = conn.execute(
                "SELECT * FROM controller_pairings WHERE pairing_id=?",
                (pairing_id,),
            ).fetchone()
            expires = _parse_time(pairing["expires_at"] if pairing else None)
            if (pairing is None or pairing["status"] != "pending"
                    or expires is None or now_value >= expires
                    or pairing["transport_binding_sha256"] != binding
                    or pairing["challenge_sha256"] != challenge_sha256
                    or not hmac.compare_digest(
                        str(pairing["code_digest"]), code_digest)):
                raise ControllerAuthError("controller_pairing_invalid")
            duplicate = conn.execute(
                "SELECT controller_id FROM controller_identities "
                "WHERE public_key_sha256=?", (key_sha256,)).fetchone()
            if duplicate is not None:
                raise ControllerAuthError("controller_key_already_paired")
            controller_body = {
                "controller_id": controller_id,
                "label": _controller_label(label), "status": "active",
            }
            controller_event, controller_seq = self.graph.append_event(
                conn, "controller.paired", controller_body,
                actor="local_pairing")
            self.graph.append_node(
                conn, "controller", controller_body,
                event_id=controller_event, node_id=controller_id)
            conn.execute(
                """INSERT INTO controller_identities
                   (controller_id,label,key_algorithm,public_jwk_json,
                    public_key_sha256,transport_binding_sha256,status,
                    auth_epoch,paired_at,paired_event_seq,last_event_seq)
                   VALUES (?,?,?,?,?,?,'active',1,?,?,?)""",
                (controller_id, controller_body["label"], _KEY_ALGORITHM,
                 key_json, key_sha256, binding, now, controller_seq,
                 controller_seq),
            )
            controller = conn.execute(
                "SELECT * FROM controller_identities WHERE controller_id=?",
                (controller_id,),
            ).fetchone()
            principal, session_token = self._insert_session(
                conn, controller=controller, origin_sha256=origin_sha256,
                transport_binding_sha256=binding,
                proof_challenge_sha256=challenge_sha256,
                proof_signature_sha256=signature_sha256,
                now_value=now_value)
            pairing_body = {
                "pairing_id": pairing_id, "controller_id": controller_id,
                "status": "consumed",
            }
            _, pairing_seq = self.graph.append_event(
                conn, "controller_pairing.consumed", pairing_body,
                actor="controller_auth")
            changed = conn.execute(
                """UPDATE controller_pairings
                      SET status='consumed',proposed_public_jwk_json=?,
                          proposed_key_sha256=?,proof_signature_sha256=?,
                          controller_id=?,consumed_at=?,last_event_seq=?
                    WHERE pairing_id=? AND status='pending'""",
                (key_json, key_sha256, signature_sha256, controller_id, now,
                 pairing_seq, pairing_id),
            ).rowcount
            if changed != 1:
                raise ControllerAuthError("controller_pairing_invalid")
        return {
            "controller_id": controller_id,
            "controller_label": _controller_label(label),
            "public_key_sha256": key_sha256,
            "session_id": principal.session_id,
            "session_token": session_token,
            "idle_expires_at": principal.idle_expires_at,
            "absolute_expires_at": principal.absolute_expires_at,
        }

    def create_session_challenge(
        self, controller_id: str, *, origin: str,
        transport_binding_sha256: str,
    ) -> dict[str, Any]:
        controller_id = _validate_opaque("controller_identity", controller_id)
        canonical_origin = normalize_https_origin(origin)
        origin_sha256 = _sha256_text(canonical_origin)
        binding = _validate_hash("transport binding", transport_binding_sha256)
        challenge_id = new_id("controller_challenge")
        challenge = secrets.token_urlsafe(32)
        now_value = self._now()
        now = _iso(now_value)
        expires = _iso(now_value + timedelta(
            seconds=self.challenge_ttl_seconds))
        with self.graph.transaction() as conn:
            controller = conn.execute(
                "SELECT * FROM controller_identities WHERE controller_id=?",
                (controller_id,),
            ).fetchone()
            if (controller is None or controller["status"] != "active"
                    or controller["transport_binding_sha256"] != binding):
                raise ControllerAuthError()
            payload = self.session_proof_payload(
                challenge_id=challenge_id, challenge=challenge,
                controller_id=controller_id,
                controller_key_sha256=str(controller["public_key_sha256"]),
                origin=canonical_origin, transport_binding_sha256=binding)
            body = {
                "challenge_id": challenge_id,
                "controller_id": controller_id,
                "status": "pending", "expires_at": expires,
            }
            _, seq = self.graph.append_event(
                conn, "controller_session_challenge.created", body,
                actor="controller_auth")
            conn.execute(
                """INSERT INTO controller_session_challenges
                   (challenge_id,controller_id,controller_key_sha256,
                    challenge_sha256,origin_sha256,transport_binding_sha256,
                    status,attempts_remaining,created_at,expires_at,
                    created_event_seq,last_event_seq)
                   VALUES (?,?,?,?,?,?,'pending',5,?,?,?,?)""",
                (challenge_id, controller_id, controller["public_key_sha256"],
                 _sha256_text(challenge), origin_sha256, binding,
                 now, expires, seq, seq),
            )
        return {
            "challenge_id": challenge_id, "challenge": challenge,
            "proof_payload": payload, "expires_at": expires,
        }

    def _fail_session_challenge(self, challenge_id: str) -> None:
        with self.graph.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM controller_session_challenges "
                "WHERE challenge_id=?", (challenge_id,)).fetchone()
            if row is None or row["status"] != "pending":
                return
            now_value = self._now()
            expires = _parse_time(row["expires_at"])
            if expires is not None and now_value >= expires:
                body = {"challenge_id": challenge_id, "status": "expired"}
                _, seq = self.graph.append_event(
                    conn, "controller_session_challenge.expired", body,
                    actor="controller_auth")
                conn.execute(
                    """UPDATE controller_session_challenges
                          SET status='expired',last_event_seq=?
                        WHERE challenge_id=? AND status='pending'""",
                    (seq, challenge_id),
                )
                return
            remaining = max(0, int(row["attempts_remaining"]) - 1)
            status = "cancelled" if remaining == 0 else "pending"
            body = {"challenge_id": challenge_id, "status": status,
                    "attempts_remaining": remaining}
            _, seq = self.graph.append_event(
                conn, "controller_session_challenge.attempt_rejected", body,
                actor="controller_auth")
            conn.execute(
                """UPDATE controller_session_challenges
                      SET attempts_remaining=?,status=?,last_event_seq=?
                    WHERE challenge_id=? AND status='pending'""",
                (remaining, status, seq, challenge_id),
            )

    def complete_session(
        self, challenge_id: str, challenge: str, proof_payload: str,
        signature_b64url: str,
    ) -> dict[str, Any]:
        challenge_id = _validate_opaque("controller_challenge", challenge_id)
        if not isinstance(challenge, str) or not _SECRET_RE.fullmatch(challenge):
            self._fail_session_challenge(challenge_id)
            raise ControllerAuthError("controller_proof_invalid")
        if not isinstance(proof_payload, str) or len(proof_payload) > 4096:
            self._fail_session_challenge(challenge_id)
            raise ControllerAuthError("controller_proof_invalid")
        try:
            payload_value = json.loads(proof_payload)
        except json.JSONDecodeError as exc:
            self._fail_session_challenge(challenge_id)
            raise ControllerAuthError("controller_proof_invalid") from exc
        if (not isinstance(payload_value, Mapping)
                or canonical_json(payload_value) != proof_payload):
            self._fail_session_challenge(challenge_id)
            raise ControllerAuthError("controller_proof_invalid")
        with self.graph._connect() as conn:
            row = conn.execute(
                """SELECT c.*,i.public_jwk_json,i.status AS controller_status,
                          i.auth_epoch,i.public_key_sha256,
                          i.transport_binding_sha256 AS identity_binding
                     FROM controller_session_challenges c
                     JOIN controller_identities i
                       ON i.controller_id=c.controller_id
                    WHERE c.challenge_id=?""",
                (challenge_id,),
            ).fetchone()
        expires = _parse_time(row["expires_at"] if row else None)
        if (row is None or row["status"] != "pending"
                or row["controller_status"] != "active"
                or expires is None or self._now() >= expires
                or row["challenge_sha256"] != _sha256_text(challenge)
                or row["identity_binding"] != row["transport_binding_sha256"]):
            self._fail_session_challenge(challenge_id)
            raise ControllerAuthError()
        try:
            expected = self.session_proof_payload(
                challenge_id=challenge_id, challenge=challenge,
                controller_id=str(row["controller_id"]),
                controller_key_sha256=str(row["controller_key_sha256"]),
                origin=str(payload_value.get("origin") or ""),
                transport_binding_sha256=str(
                    row["transport_binding_sha256"]))
            payload_origin_sha256 = _sha256_text(normalize_https_origin(
                str(payload_value.get("origin") or "")))
        except (ControllerAuthError, ValueError):
            self._fail_session_challenge(challenge_id)
            raise ControllerAuthError("controller_proof_invalid") from None
        if (expected != proof_payload
                or payload_origin_sha256 != row["origin_sha256"]):
            self._fail_session_challenge(challenge_id)
            raise ControllerAuthError("controller_proof_invalid")
        public_jwk = json.loads(str(row["public_jwk_json"]))
        if not verify_p256_signature(
                public_jwk, proof_payload.encode("utf-8"), signature_b64url,
                openssl_path=self.openssl_path):
            self._fail_session_challenge(challenge_id)
            raise ControllerAuthError("controller_proof_invalid")
        now_value = self._now()
        now = _iso(now_value)
        with self.graph.transaction() as conn:
            challenge_row = conn.execute(
                "SELECT * FROM controller_session_challenges "
                "WHERE challenge_id=?", (challenge_id,)).fetchone()
            controller = conn.execute(
                "SELECT * FROM controller_identities WHERE controller_id=?",
                (row["controller_id"],),
            ).fetchone()
            challenge_expires = _parse_time(
                challenge_row["expires_at"] if challenge_row else None)
            if (challenge_row is None or challenge_row["status"] != "pending"
                    or challenge_expires is None or now_value >= challenge_expires
                    or controller is None or controller["status"] != "active"
                    or int(controller["auth_epoch"]) != int(row["auth_epoch"])
                    or challenge_row["challenge_sha256"]
                        != _sha256_text(challenge)):
                raise ControllerAuthError()
            principal, token = self._insert_session(
                conn, controller=controller,
                origin_sha256=str(challenge_row["origin_sha256"]),
                transport_binding_sha256=str(
                    challenge_row["transport_binding_sha256"]),
                proof_challenge_sha256=str(
                    challenge_row["challenge_sha256"]),
                proof_signature_sha256=_sha256_text(signature_b64url),
                now_value=now_value)
            body = {"challenge_id": challenge_id, "status": "consumed",
                    "session_id": principal.session_id}
            _, seq = self.graph.append_event(
                conn, "controller_session_challenge.consumed", body,
                actor="controller_auth")
            changed = conn.execute(
                """UPDATE controller_session_challenges
                      SET status='consumed',consumed_at=?,last_event_seq=?
                    WHERE challenge_id=? AND status='pending'""",
                (now, seq, challenge_id),
            ).rowcount
            if changed != 1:
                raise ControllerAuthError()
        return {
            "controller_id": principal.controller_id,
            "session_id": principal.session_id,
            "session_token": token,
            "idle_expires_at": principal.idle_expires_at,
            "absolute_expires_at": principal.absolute_expires_at,
        }

    def authenticate_session(
        self, session_token: str, *, origin: str,
        transport_binding_sha256: str,
    ) -> ControllerPrincipal:
        session_id, _ = self._parse_session_token(session_token)
        canonical_origin = normalize_https_origin(origin)
        origin_sha256 = _sha256_text(canonical_origin)
        binding = _validate_hash("transport binding", transport_binding_sha256)
        digest = self._digest("session-token", session_token)
        now_value = self._now()
        now = _iso(now_value)
        expired_session = False
        with self.graph.transaction() as conn:
            row = conn.execute(
                """SELECT s.*,i.status AS controller_status,
                          i.auth_epoch AS current_epoch,
                          i.public_key_sha256 AS current_key
                     FROM controller_sessions s
                     JOIN controller_identities i
                       ON i.controller_id=s.controller_id
                    WHERE s.session_id=? AND s.token_digest=?""",
                (session_id, digest),
            ).fetchone()
            idle = _parse_time(row["idle_expires_at"] if row else None)
            absolute = _parse_time(
                row["absolute_expires_at"] if row else None)
            identity_valid = bool(
                row is not None and row["status"] == "active"
                and row["controller_status"] == "active"
                and int(row["controller_epoch"]) == int(row["current_epoch"])
                and row["controller_key_sha256"] == row["current_key"]
                and row["origin_sha256"] == origin_sha256
                and row["transport_binding_sha256"] == binding)
            # Keep expiry durable when a legitimate bearer is presented after
            # its deadline, without letting malformed credentials mutate state.
            if (identity_valid and idle is not None and absolute is not None
                    and (now_value >= idle or now_value >= absolute)):
                body = {
                    "session_id": session_id,
                    "controller_id": str(row["controller_id"]),
                    "status": "expired",
                }
                _, seq = self.graph.append_event(
                    conn, "controller_session.expired", body,
                    actor="controller_auth")
                conn.execute(
                    """UPDATE controller_sessions
                          SET status='expired',ended_at=?,last_event_seq=?
                        WHERE session_id=? AND status='active'""",
                    (now, seq, session_id),
                )
                expired_session = True
            else:
                if (not identity_valid or idle is None or absolute is None
                        or now_value >= idle or now_value >= absolute):
                    raise ControllerAuthError()
                last_seen = _parse_time(row["last_seen_at"])
                if last_seen is None:
                    raise ControllerAuthError()
                seen_value = max(now_value, last_seen)
                refreshed_value = min(
                    seen_value + timedelta(
                        seconds=self.idle_session_ttl_seconds),
                    absolute,
                )
                if refreshed_value < idle:
                    refreshed_value = idle
                seen = _iso(seen_value)
                refreshed = _iso(refreshed_value)
                changed = conn.execute(
                    """UPDATE controller_sessions
                          SET last_seen_at=?,idle_expires_at=?
                        WHERE session_id=? AND status='active'
                          AND controller_epoch=?""",
                    (seen, refreshed, session_id, int(row["current_epoch"])),
                ).rowcount
                if changed != 1:
                    raise ControllerAuthError()
                conn.execute(
                    """UPDATE controller_identities
                          SET last_authenticated_at=CASE
                              WHEN last_authenticated_at IS NULL
                                   OR last_authenticated_at < ? THEN ?
                              ELSE last_authenticated_at END
                        WHERE controller_id=? AND status='active'
                          AND auth_epoch=?""",
                    (seen, seen, row["controller_id"],
                     int(row["current_epoch"])),
                )
                return ControllerPrincipal(
                    controller_id=str(row["controller_id"]),
                    session_id=str(row["session_id"]),
                    public_key_sha256=str(row["controller_key_sha256"]),
                    controller_epoch=int(row["controller_epoch"]),
                    origin_sha256=str(row["origin_sha256"]),
                    transport_binding_sha256=str(
                        row["transport_binding_sha256"]),
                    issued_at=str(row["issued_at"]),
                    idle_expires_at=refreshed,
                    absolute_expires_at=str(row["absolute_expires_at"]),
                )
        if expired_session:
            raise ControllerAuthError()
        raise ControllerAuthError()

    def _require_principal(
        self, conn: Any, principal: ControllerPrincipal,
        now_value: datetime,
    ) -> Mapping[str, Any]:
        if not isinstance(principal, ControllerPrincipal):
            raise ControllerAuthError()
        row = conn.execute(
            """SELECT s.*,i.status AS controller_status,
                      i.auth_epoch AS current_epoch,
                      i.public_jwk_json AS controller_public_jwk_json
                 FROM controller_sessions s
                 JOIN controller_identities i
                   ON i.controller_id=s.controller_id
                WHERE s.session_id=? AND s.controller_id=?
                  AND s.controller_key_sha256=?
                  AND s.origin_sha256=?
                  AND s.transport_binding_sha256=?
                  AND s.issued_at=? AND s.absolute_expires_at=?""",
            (principal.session_id, principal.controller_id,
             principal.public_key_sha256, principal.origin_sha256,
             principal.transport_binding_sha256, principal.issued_at,
             principal.absolute_expires_at),
        ).fetchone()
        idle = _parse_time(row["idle_expires_at"] if row else None)
        absolute = _parse_time(row["absolute_expires_at"] if row else None)
        if (row is None or row["status"] != "active"
                or row["controller_status"] != "active"
                or int(row["controller_epoch"]) != principal.controller_epoch
                or int(row["current_epoch"]) != principal.controller_epoch
                or idle is None or absolute is None
                or now_value >= idle or now_value >= absolute):
            raise ControllerAuthError()
        return row

    def require_principal_in_transaction(
        self, conn: Any, principal: ControllerPrincipal, *,
        now_value: datetime | None = None,
    ) -> Mapping[str, Any]:
        """Fence a controller principal inside the caller's write transaction."""
        return self._require_principal(
            conn, principal, now_value or self._now())

    def current_time(self) -> datetime:
        """Return the service's authoritative UTC clock value."""
        return self._now()

    def verify_principal_signature(
        self, principal: ControllerPrincipal, payload: str,
        signature_b64url: str,
    ) -> bool:
        if not isinstance(payload, str) or len(payload) > 16_384:
            return False
        with self.graph._connect() as conn:
            row = self._require_principal(conn, principal, self._now())
        try:
            public_jwk = json.loads(
                str(row["controller_public_jwk_json"]))
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "controller public key projection is invalid") from exc
        return verify_p256_signature(
            public_jwk, payload.encode("utf-8"), signature_b64url,
            openssl_path=self.openssl_path)

    def revalidate_principal(self, principal: ControllerPrincipal) -> None:
        with self.graph._connect() as conn:
            self._require_principal(conn, principal, self._now())

    def revoke_session(
        self, principal: ControllerPrincipal, session_id: str | None = None,
    ) -> None:
        target = _validate_opaque(
            "controller_session", session_id or principal.session_id)
        now_value = self._now()
        now = _iso(now_value)
        with self.graph.transaction() as conn:
            self._require_principal(conn, principal, now_value)
            row = conn.execute(
                "SELECT * FROM controller_sessions WHERE session_id=?",
                (target,),
            ).fetchone()
            if row is None or row["controller_id"] != principal.controller_id:
                raise ControllerAuthError()
            if row["status"] != "active":
                return
            body = {"session_id": target,
                    "controller_id": principal.controller_id,
                    "status": "revoked"}
            _, seq = self.graph.append_event(
                conn, "controller_session.revoked", body,
                actor=principal.controller_id)
            conn.execute(
                """UPDATE controller_sessions
                      SET status='revoked',ended_at=?,last_event_seq=?
                    WHERE session_id=? AND status='active'""",
                (now, seq, target),
            )

    def revoke_controller(
        self, principal: ControllerPrincipal, controller_id: str,
    ) -> None:
        target = _validate_opaque("controller_identity", controller_id)
        now_value = self._now()
        now = _iso(now_value)
        with self.graph.transaction() as conn:
            self._require_principal(conn, principal, now_value)
            row = conn.execute(
                "SELECT * FROM controller_identities WHERE controller_id=?",
                (target,),
            ).fetchone()
            if row is None:
                raise ControllerAuthError()
            if row["status"] != "active":
                return
            body = {"controller_id": target, "status": "revoked"}
            _, seq = self.graph.append_event(
                conn, "controller.revoked", body,
                actor=principal.controller_id)
            conn.execute(
                """UPDATE controller_identities
                      SET status='revoked',auth_epoch=auth_epoch+1,
                          revoked_at=?,last_event_seq=?
                    WHERE controller_id=? AND status='active'""",
                (now, seq, target),
            )
            conn.execute(
                """UPDATE controller_sessions
                      SET status='revoked',ended_at=?,last_event_seq=?
                    WHERE controller_id=? AND status='active'""",
                (now, seq, target),
            )

    def list_controllers(self) -> list[dict[str, Any]]:
        now = _iso(self._now())
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT i.controller_id,i.label,i.status,i.paired_at,
                          i.last_authenticated_at,i.revoked_at,
                          COUNT(CASE WHEN s.status='active'
                                      AND s.controller_epoch=i.auth_epoch
                                      AND s.idle_expires_at>?
                                      AND s.absolute_expires_at>?
                                     THEN 1 END)
                              AS active_sessions
                     FROM controller_identities i
                     LEFT JOIN controller_sessions s
                       ON s.controller_id=i.controller_id
                    GROUP BY i.controller_id
                    ORDER BY i.paired_at""",
                (now, now),
            ).fetchall()
        return [dict(row) for row in rows]


__all__ = [
    "ControllerAuthError", "ControllerAuthService", "ControllerPrincipal",
    "normalize_https_origin", "normalize_public_jwk", "public_key_sha256",
    "verify_p256_signature",
]
