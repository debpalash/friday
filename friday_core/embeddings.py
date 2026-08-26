"""Pinned, offline, CPU-first text embeddings for private local retrieval."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import threading
from pathlib import Path
from typing import Sequence

import numpy as np


MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_DIRECTORY = f"multilingual-e5-small-{MODEL_REVISION[:8]}"
MODEL_DIMENSION = 384
MAX_EMBEDDING_TEXTS = 512
MAX_EMBEDDING_TEXT_CHARS = 4_000
_ASSETS = {
    "config.json": (
        655, "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959"),
    "model.safetensors": (
        470_641_600,
        "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"),
    "sentencepiece.bpe.model": (
        5_069_051,
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"),
    "special_tokens_map.json": (
        167, "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7"),
    "tokenizer.json": (
        17_082_730,
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
    "tokenizer_config.json": (
        443, "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b"),
}
MODEL_FINGERPRINT = hashlib.sha256(
    (MODEL_ID + "\0" + MODEL_REVISION + "\0" + "\0".join(
        f"{name}:{size}:{digest}"
        for name, (size, digest) in sorted(_ASSETS.items()))).encode("utf-8")
).hexdigest()


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _host_capacity() -> tuple[int, int]:
    cpus = max(1, int(os.cpu_count() or 1))
    try:
        memory = int(os.sysconf("SC_PHYS_PAGES")) * int(
            os.sysconf("SC_PAGE_SIZE"))
    except (OSError, TypeError, ValueError):
        memory = 0
    return cpus, memory


class LocalTextEmbedder:
    """Lazy local E5 encoder; no hub/network path exists after construction."""

    dimension = MODEL_DIMENSION
    fingerprint = MODEL_FINGERPRINT

    def __init__(self, model_path: str | Path, *, batch_size: int | None = None):
        self.model_path = Path(model_path).resolve()
        cpus, memory = _host_capacity()
        automatic_batch = 16 if cpus >= 16 and memory >= 24 * 1024 ** 3 else (
            8 if cpus >= 8 and memory >= 8 * 1024 ** 3 else 4)
        self.batch_size = automatic_batch if batch_size is None else batch_size
        if (isinstance(self.batch_size, bool)
                or not isinstance(self.batch_size, int)
                or not 1 <= self.batch_size <= 64):
            raise ValueError("embedding batch size is invalid")
        self._lock = threading.RLock()
        self._tokenizer = None
        self._model = None
        self._verified = False

    def _verify(self) -> None:
        if self._verified:
            return
        try:
            metadata = self.model_path.stat()
        except OSError as exc:
            raise RuntimeError("pinned local embedding model is unavailable") from exc
        if (not stat.S_ISDIR(metadata.st_mode) or self.model_path.is_symlink()
                or metadata.st_uid != os.getuid()):
            raise RuntimeError("pinned local embedding directory is invalid")
        for name, (expected_size, expected_hash) in _ASSETS.items():
            path = self.model_path / name
            try:
                item = path.stat()
            except OSError as exc:
                raise RuntimeError(
                    f"pinned embedding asset is unavailable: {name}") from exc
            if (not stat.S_ISREG(item.st_mode) or path.is_symlink()
                    or item.st_uid != os.getuid() or item.st_size != expected_size
                    or _sha256_file(path) != expected_hash):
                raise RuntimeError(f"pinned embedding asset is invalid: {name}")
        self._verified = True

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            self._verify()
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=False)
            model = AutoModel.from_pretrained(
                self.model_path, local_files_only=True,
                trust_remote_code=False).to("cpu").eval()
            hidden = int(getattr(model.config, "hidden_size", 0))
            if hidden != self.dimension:
                raise RuntimeError("pinned embedding dimension is incompatible")
            self._tokenizer, self._model = tokenizer, model

    @staticmethod
    def _validated(texts: Sequence[str], kind: str) -> list[str]:
        if (kind not in {"query", "passage"} or not isinstance(texts, Sequence)
                or isinstance(texts, (str, bytes))
                or not 1 <= len(texts) <= MAX_EMBEDDING_TEXTS):
            raise ValueError("embedding request is invalid")
        output = []
        for value in texts:
            if (not isinstance(value, str)
                    or not 1 <= len(value) <= MAX_EMBEDDING_TEXT_CHARS
                    or any(ord(character) < 32 and character not in "\t\n\r"
                           for character in value)):
                raise ValueError("embedding text is invalid")
            output.append(f"{kind}: {value}")
        return output

    def encode(self, texts: Sequence[str], *, kind: str) -> np.ndarray:
        values = self._validated(texts, kind)
        self._load()
        import torch
        chunks: list[np.ndarray] = []
        with self._lock, torch.inference_mode():
            for offset in range(0, len(values), self.batch_size):
                batch = self._tokenizer(
                    values[offset:offset + self.batch_size], padding=True,
                    truncation=True, max_length=256, return_tensors="pt")
                result = self._model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (result * mask).sum(dim=1) / mask.sum(dim=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                chunks.append(pooled.detach().cpu().numpy().astype("<f4"))
        encoded = np.concatenate(chunks, axis=0)
        if (encoded.shape != (len(values), self.dimension)
                or not np.isfinite(encoded).all()):
            raise RuntimeError("local embedding output is invalid")
        return encoded


def configured_local_embedder(repo: str | Path,
                              env: dict[str, str] | None = None
                              ) -> LocalTextEmbedder | None:
    environment = os.environ if env is None else env
    configured = str(environment.get("FRIDAY_EMBEDDING_MODEL", "auto")).strip()
    if configured.casefold() in {"disabled", "none", "off", "0"}:
        return None
    default = Path(repo) / "models" / MODEL_DIRECTORY
    path = default if configured.casefold() in {"", "auto"} else Path(configured)
    if not path.is_absolute():
        path = Path(repo) / path
    if configured.casefold() in {"", "auto"} and not path.is_dir():
        return None
    batch_value = str(environment.get("FRIDAY_EMBEDDING_BATCH_SIZE", "")).strip()
    if batch_value and re.fullmatch(r"[0-9]{1,2}", batch_value) is None:
        raise ValueError("FRIDAY_EMBEDDING_BATCH_SIZE is invalid")
    batch_size = int(batch_value) if batch_value else None
    return LocalTextEmbedder(path, batch_size=batch_size)
