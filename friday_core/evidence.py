"""Encrypted local storage for audio retained only after transcript correction."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import subprocess
from pathlib import Path
from typing import Callable


KeyProvider = Callable[[], bytes]


class CorrectedAudioStore:
    """AES-256-CTR plus encrypt-then-MAC, with the master key in Secret Service."""

    def __init__(self, root: str | Path, key_provider: KeyProvider | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key_provider = key_provider or self._desktop_key

    @staticmethod
    def _desktop_key() -> bytes:
        lookup = subprocess.run(
            ["secret-tool", "lookup", "application", "friday",
             "purpose", "corrected-audio"], capture_output=True, timeout=5)
        value = lookup.stdout.strip()
        if not value:
            value = base64.urlsafe_b64encode(secrets.token_bytes(64))
            stored = subprocess.run(
                ["secret-tool", "store", "--label=Friday corrected audio",
                 "application", "friday", "purpose", "corrected-audio"],
                input=value + b"\n", capture_output=True, timeout=10)
            if stored.returncode:
                raise RuntimeError("desktop keyring refused the corrected-audio key")
        try:
            key = base64.urlsafe_b64decode(value)
        except Exception as exc:
            raise RuntimeError("corrected-audio key is invalid") from exc
        if len(key) < 64:
            raise RuntimeError("corrected-audio key is too short")
        return key[:64]

    def store(self, artifact_id: str, pcm: bytes, metadata: dict) -> str:
        if not pcm:
            raise ValueError("audio evidence is empty")
        master = self.key_provider()
        encryption_key, mac_key = master[:32], master[32:64]
        iv = secrets.token_bytes(16)
        encrypted = subprocess.run(
            ["openssl", "enc", "-aes-256-ctr", "-K", encryption_key.hex(),
             "-iv", iv.hex()], input=pcm, capture_output=True, timeout=15,
             check=True).stdout
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

    def delete(self, artifact_path: str | Path) -> bool:
        path = Path(artifact_path).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("audio artifact is outside the evidence store")
        if not path.exists():
            return False
        path.unlink()
        return True
