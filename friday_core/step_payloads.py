"""Authenticated local encryption for durable step arguments.

The SQLite graph intentionally stores only redacted argument previews.  Exact
arguments are encrypted so a worker can resume after process death without
putting clipboard contents, form input, code, or provider prompts into an
iterdump, progress event, or log.

The per-install key is a mode-0600 local secret.  This protects against
accidental disclosure and copied databases; it is not intended to defend
against an attacker already executing as the Friday OS user.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import subprocess
from pathlib import Path
from typing import Any

from .graph import canonical_json


class StepPayloadCipher:
    """AES-256-CTR with encrypt-then-MAC and context-bound payloads."""

    VERSION = 1

    def __init__(self, key_path: str | Path, *, key: bytes | None = None):
        self.key_path = Path(key_path)
        self._key = key
        if key is not None and len(key) != 64:
            raise ValueError("step payload key must contain exactly 64 bytes")

    def _load_key(self) -> bytes:
        if self._key is not None:
            return self._key
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.key_path, flags)
        except FileNotFoundError:
            encoded = base64.urlsafe_b64encode(secrets.token_bytes(64)) + b"\n"
            create_flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                            | getattr(os, "O_NOFOLLOW", 0))
            try:
                descriptor = os.open(self.key_path, create_flags, 0o600)
            except FileExistsError:
                return self._load_key()
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            raw = base64.urlsafe_b64decode(encoded.strip())
            self._key = raw
            return raw
        with os.fdopen(descriptor, "rb") as stream:
            encoded = stream.read().strip()
        try:
            raw = base64.urlsafe_b64decode(encoded)
        except Exception as exc:
            raise RuntimeError("durable step payload key is invalid") from exc
        if len(raw) != 64:
            raise RuntimeError("durable step payload key is truncated")
        os.chmod(self.key_path, 0o600)
        self._key = raw
        return raw

    @staticmethod
    def _authenticated_bytes(context: str, iv: bytes, ciphertext: bytes) -> bytes:
        return context.encode("utf-8") + b"\0" + iv + ciphertext

    def seal(self, value: dict[str, Any], *, context: str) -> str:
        master = self._load_key()
        iv = secrets.token_bytes(16)
        plaintext = canonical_json(value).encode("utf-8")
        ciphertext = subprocess.run(
            ["openssl", "enc", "-aes-256-ctr", "-K", master[:32].hex(),
             "-iv", iv.hex()], input=plaintext, capture_output=True,
            timeout=10, check=True).stdout
        digest = hmac.new(
            master[32:], self._authenticated_bytes(context, iv, ciphertext),
            hashlib.sha256).digest()
        return canonical_json({
            "version": self.VERSION,
            "cipher": "aes-256-ctr+hmac-sha256",
            "iv": base64.b64encode(iv).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "mac": base64.b64encode(digest).decode("ascii"),
        })

    def open(self, payload: str, *, context: str) -> dict[str, Any]:
        master = self._load_key()
        try:
            body = json.loads(payload)
            if (body.get("version") != self.VERSION
                    or body.get("cipher") != "aes-256-ctr+hmac-sha256"):
                raise ValueError("unsupported payload format")
            iv = base64.b64decode(body["iv"], validate=True)
            ciphertext = base64.b64decode(body["ciphertext"], validate=True)
            supplied_mac = base64.b64decode(body["mac"], validate=True)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("durable step payload is malformed") from exc
        expected_mac = hmac.new(
            master[32:], self._authenticated_bytes(context, iv, ciphertext),
            hashlib.sha256).digest()
        if not hmac.compare_digest(supplied_mac, expected_mac):
            raise RuntimeError("durable step payload authentication failed")
        plaintext = subprocess.run(
            ["openssl", "enc", "-d", "-aes-256-ctr",
             "-K", master[:32].hex(), "-iv", iv.hex()],
            input=ciphertext, capture_output=True, timeout=10,
            check=True).stdout
        try:
            value = json.loads(plaintext)
        except json.JSONDecodeError as exc:
            raise RuntimeError("durable step payload plaintext is invalid") from exc
        if not isinstance(value, dict):
            raise RuntimeError("durable step arguments must decode to an object")
        return value
