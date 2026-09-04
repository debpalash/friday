"""Encrypted local storage for audio retained only after transcript correction."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Callable

from friday_host.secret_store import SecretStore, SecretStoreUnavailable

from .local_cipher import aes256_ctr


KeyProvider = Callable[[], bytes]


class CorrectedAudioStore:
    """AES-256-CTR plus encrypt-then-MAC, with the master key in the host keyring."""

    def __init__(self, root: str | Path, key_provider: KeyProvider | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key_provider = key_provider or self._desktop_key

    @staticmethod
    def _desktop_key() -> bytes:
        try:
            return SecretStore().get_or_create("corrected-audio", 64)
        except SecretStoreUnavailable as exc:
            raise RuntimeError(str(exc)) from exc

    def store(self, artifact_id: str, pcm: bytes, metadata: dict, *,
              iv: bytes | None = None) -> str:
        if not pcm:
            raise ValueError("audio evidence is empty")
        master = self.key_provider()
        encryption_key, mac_key = master[:32], master[32:64]
        iv = secrets.token_bytes(16) if iv is None else bytes(iv)
        if len(iv) != 16:
            raise ValueError("audio evidence IV must contain 16 bytes")
        encrypted = aes256_ctr(encryption_key, iv, pcm)
        authenticated = iv + encrypted
        digest = hmac.new(mac_key, authenticated, hashlib.sha256).digest()
        payload = {"version": 1, "cipher": "aes-256-ctr+hmac-sha256",
                   "iv": base64.b64encode(iv).decode(),
                   "ciphertext": base64.b64encode(encrypted).decode(),
                   "mac": base64.b64encode(digest).decode(),
                   "metadata": metadata}
        path = self.root / f"{artifact_id}.enc.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        os.chmod(path, 0o600)
        return str(path)

    @staticmethod
    def decrypt(payload: dict, master: bytes) -> bytes:
        """Authenticate and decrypt one stored artifact payload."""
        if (payload.get("version") != 1
                or payload.get("cipher") != "aes-256-ctr+hmac-sha256"):
            raise ValueError("unsupported audio artifact format")
        iv = base64.b64decode(payload["iv"], validate=True)
        encrypted = base64.b64decode(payload["ciphertext"], validate=True)
        supplied = base64.b64decode(payload["mac"], validate=True)
        expected = hmac.new(master[32:64], iv + encrypted, hashlib.sha256).digest()
        if not hmac.compare_digest(supplied, expected):
            raise ValueError("audio artifact authentication failed")
        return aes256_ctr(master[:32], iv, encrypted)

    def delete(self, artifact_path: str | Path) -> bool:
        path = Path(artifact_path).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("audio artifact is outside the evidence store")
        if not path.exists():
            return False
        path.unlink()
        return True
