"""Pinned engine binaries, model checkpoints, and the memory tier table.

The pins live in ``engine_assets.json`` next to this module so the release
review and the installers read the same digests. ``select_model_tier`` is
pure: it maps a memory budget to the largest Qwen3 checkpoint whose weights
and key-value cache fit, so profile selection stays testable without
hardware.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

GIB = 1024 ** 3
_DATA = Path(__file__).with_name("engine_assets.json")
ENGINES = ("vllm", "llama_server", "mlx_lm")
MINIMUM_CONTEXT = 8192
PREFERRED_CONTEXT = 16_384
HEADROOM_BYTES = 1 * GIB


@dataclass(frozen=True)
class ModelAsset:
    key: str
    engine: str
    size_label: str
    repo: str
    revision: str
    license: str
    files: tuple[tuple[str, int, str], ...]
    weights_bytes: int
    kv_bytes_per_token: int
    entry: str

    @property
    def directory(self) -> str:
        return f"{self.key}-{self.revision[:8]}"

    @property
    def model_file(self) -> str:
        """Path relative to the model directory that the engine loads."""
        return self.entry or "."


@dataclass(frozen=True)
class EngineBinary:
    platform: str
    arch: str
    backend: str
    tag: str
    name: str
    url: str
    size: int
    sha256: str
    executable: str
    extra: tuple[tuple[str, str, int, str], ...] = ()

    @property
    def directory(self) -> str:
        return f"{self.tag}-{self.backend}"


@lru_cache(maxsize=1)
def _load() -> dict:
    data = json.loads(_DATA.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise RuntimeError("engine asset table schema is unsupported")
    return data


@lru_cache(maxsize=1)
def llama_server_binaries() -> tuple[EngineBinary, ...]:
    entries = []
    for row in _load()["llama_server"]["binaries"]:
        entries.append(EngineBinary(
            platform=row["platform"], arch=row["arch"], backend=row["backend"],
            tag=row["tag"], name=row["name"], url=row["url"], size=int(row["size"]),
            sha256=row["sha256"], executable=row["executable"],
            extra=tuple((item["name"], item["url"], int(item["size"]),
                         item["sha256"]) for item in row.get("extra", ()))))
    return tuple(entries)


def llama_server_tag() -> str:
    return str(_load()["llama_server"]["tag"])


def mlx_runtime_pins() -> dict[str, str]:
    value = _load()["mlx_runtime"]
    return {"mlx": str(value["mlx"]), "mlx-lm": str(value["mlx-lm"])}


@lru_cache(maxsize=1)
def model_assets() -> tuple[ModelAsset, ...]:
    entries = []
    for row in _load()["models"]:
        entries.append(ModelAsset(
            key=row["key"], engine=row["engine"], size_label=row["size_label"],
            repo=row["repo"], revision=row["revision"], license=row["license"],
            files=tuple((name, int(size), digest)
                        for name, size, digest in row["files"]),
            weights_bytes=int(row["weights_bytes"]),
            kv_bytes_per_token=int(row["kv_bytes_per_token"]),
            entry=str(row.get("entry") or "")))
    return tuple(entries)


def model_asset(key: str) -> ModelAsset:
    for item in model_assets():
        if item.key == key:
            return item
    raise KeyError(f"unknown model asset: {key}")


def engine_backends(engine: str, platform: str, arch: str) -> frozenset[str]:
    """Backends with a pinned binary (or built-in runtime) for this host."""
    if engine == "mlx_lm":
        return frozenset({"metal"}) if (platform, arch) == ("macos", "aarch64") \
            else frozenset()
    if engine == "llama_server":
        return frozenset(
            item.backend for item in llama_server_binaries()
            if item.platform == platform and item.arch == arch)
    if engine == "vllm":
        return frozenset({"cuda"}) if platform == "linux" else frozenset()
    raise ValueError(f"unknown engine: {engine}")


def llama_server_binary(platform: str, arch: str, backend: str) -> EngineBinary:
    for item in llama_server_binaries():
        if (item.platform, item.arch, item.backend) == (platform, arch, backend):
            return item
    raise KeyError(f"no llama-server binary for {platform}/{arch}/{backend}")


# Ordered from smallest to largest; the selector walks it downward.
_TIER_ORDER = ("4b", "8b", "14b", "30b-a3b", "32b")
_CONTEXT_CEILING = {"4b": 32_768, "8b": 32_768, "14b": 32_768,
                    "30b-a3b": 32_768, "32b": 65_536}


def _fits(asset: ModelAsset, context: int, sequences: int, budget: int) -> bool:
    return (asset.weights_bytes + asset.kv_bytes_per_token * context * sequences
            + HEADROOM_BYTES) <= budget


def select_model_tier(engine: str, budget_bytes: int, *, cpu_only: bool = False
                      ) -> tuple[ModelAsset, int, int] | None:
    """Return ``(asset, context_tokens, max_sequences)`` or ``None``.

    Picks the largest checkpoint for ``engine`` that fits with a comfortable
    16K context, falling back to the largest that fits at the 8K minimum,
    then grows the context in powers of two up to a per-size ceiling. CPU-only
    hosts prefer the sparse 30B-A3B over the dense 32B because active
    parameters, not total parameters, bound decode speed.
    """
    if engine not in ("llama_server", "mlx_lm"):
        raise ValueError(f"tier selection applies to portable engines, not {engine}")
    candidates = [item for item in model_assets() if item.engine == engine]
    by_size = {item.size_label: item for item in candidates}
    order = [size for size in _TIER_ORDER if size in by_size]
    if cpu_only and "32b" in order:
        order.remove("32b")
    for required in (PREFERRED_CONTEXT, MINIMUM_CONTEXT):
        for size in reversed(order):
            asset = by_size[size]
            if not _fits(asset, required, 1, budget_bytes):
                continue
            context = required
            while (context * 2 <= _CONTEXT_CEILING[size]
                   and _fits(asset, context * 2, 1, budget_bytes)):
                context *= 2
            sequences = 1
            while sequences * 2 <= 4 and _fits(asset, context, sequences * 2,
                                               budget_bytes):
                sequences *= 2
            return asset, context, sequences
    return None


__all__ = [
    "ENGINES", "EngineBinary", "GIB", "HEADROOM_BYTES", "MINIMUM_CONTEXT",
    "PREFERRED_CONTEXT",
    "ModelAsset", "engine_backends", "llama_server_binaries",
    "llama_server_binary", "llama_server_tag", "mlx_runtime_pins",
    "model_asset", "model_assets", "select_model_tier",
]
