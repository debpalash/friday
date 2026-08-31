"""Bounded, local speech synthesis backends for Friday."""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from math import gcd
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.signal import resample_poly


PIPER_VOICE_REVISION = "375a0fe641dea077c2a47b4e9a056d6da521eed3"
PIPER_VOICE_DIRECTORY = f"piper-en_US-kristin-medium-{PIPER_VOICE_REVISION[:8]}"
PIPER_MODEL_NAME = "en_US-kristin-medium.onnx"
PIPER_CONFIG_NAME = f"{PIPER_MODEL_NAME}.json"
PIPER_ASSETS = {
    PIPER_MODEL_NAME: (
        63_531_379,
        "5849957f929cbf720c258f8458692d6103fff2f0e3d3b19c8259474bb06a18d4",
    ),
    PIPER_CONFIG_NAME: (
        4_968,
        "5681426d4aead22195de70531eeeeddb46493cfaffc5764b2ea3db73428b651c",
    ),
}
OMNIVOICE_MODEL_ID = "khaledmezdour/omnivoice-singing"
OMNIVOICE_MODEL_REVISION = "31927d2ac03a2a7259f4f5ca02329457d89cb353"
OMNIVOICE_MODEL_DIRECTORY = (
    f"omnivoice-singing-{OMNIVOICE_MODEL_REVISION[:8]}")
OMNIVOICE_ASSETS = {
    "config.json": (
        2_238,
        "7da35a3d312085444b2809c63d4d94e377b732637fbe40f47a0dd8959a5a59b2",
    ),
    "model.safetensors": (
        2_450_344_144,
        "1141123172e28971fc97a59f0dfbb5356574c2730b141e2486ee01da089f98b6",
    ),
    "tokenizer.json": (
        11_423_986,
        "408f669b7e2b045fdf54201d815bd364e6667dbd845115da81239c40bc6dcfd1",
    ),
    "tokenizer_config.json": (
        533,
        "49f78845596a82bf15c83673794bdf9f76f812b11f60ab6a2239d9be65b00676",
    ),
    "chat_template.jinja": (
        4_168,
        "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    ),
    "audio_tokenizer/config.json": (
        2_531,
        "eefb20806f7104e77c9a5277c9df0f9bb8826b08eb1d4e8ab2b9829b6ef9fac1",
    ),
    "audio_tokenizer/model.safetensors": (
        805_665_628,
        "fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d",
    ),
    "audio_tokenizer/preprocessor_config.json": (
        206,
        "ae61eea88558608ee2fa86d2aec9fce8d99a5ff75d09cb7651ccce21ae1d9084",
    ),
}
MAX_SPEECH_TEXT_CHARS = 4_000
MAX_SOURCE_AUDIO_SECONDS = 120


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_piper_model_path(repo: Path) -> Path:
    return (repo / "models" / PIPER_VOICE_DIRECTORY / PIPER_MODEL_NAME).resolve()


def verify_pinned_piper_voice(repo: Path) -> Path:
    """Return the model path only when every immutable voice asset is valid."""
    directory = repo / "models" / PIPER_VOICE_DIRECTORY
    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError("the pinned Piper voice directory is unavailable")
    for name, (expected_size, expected_digest) in PIPER_ASSETS.items():
        path = directory / name
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise RuntimeError(f"the pinned Piper asset is unavailable: {name}") from exc
        if (not stat.S_ISREG(metadata.st_mode) or path.is_symlink()
                or metadata.st_uid != os.geteuid()
                or metadata.st_size != expected_size
                or metadata.st_mode & 0o022):
            raise RuntimeError(f"the pinned Piper asset boundary is invalid: {name}")
        if _sha256(path) != expected_digest:
            raise RuntimeError(f"the pinned Piper asset digest is invalid: {name}")
    return directory / PIPER_MODEL_NAME


def pinned_omnivoice_model_path(repo: Path) -> Path:
    directory = repo / "models" / OMNIVOICE_MODEL_DIRECTORY
    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError("the pinned OmniVoice model directory is unavailable")
    for name, (expected_size, expected_digest) in OMNIVOICE_ASSETS.items():
        path = directory / name
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"the pinned OmniVoice asset is unavailable: {name}") from exc
        if (not stat.S_ISREG(metadata.st_mode) or path.is_symlink()
                or metadata.st_uid != os.geteuid()
                or metadata.st_size != expected_size
                or metadata.st_mode & 0o022):
            raise RuntimeError(
                f"the pinned OmniVoice asset boundary is invalid: {name}")
        if _sha256(path) != expected_digest:
            raise RuntimeError(
                f"the pinned OmniVoice asset digest is invalid: {name}")
    return directory


def choose_speech_backend(
        *, tts_device: str, repo: Path,
        environment: Mapping[str, str] | None = None) -> str:
    """Choose fast CPU speech without changing the GPU runtime envelope."""
    env = os.environ if environment is None else environment
    requested = env.get("FRIDAY_TTS_BACKEND", "auto").strip().lower()
    if requested not in {"auto", "omnivoice", "piper"}:
        raise RuntimeError("FRIDAY_TTS_BACKEND must be auto, omnivoice, or piper")
    if requested == "omnivoice":
        return requested
    if requested == "piper":
        verify_pinned_piper_voice(repo)
        return requested
    if tts_device.strip().lower() == "cpu":
        try:
            verify_pinned_piper_voice(repo)
        except RuntimeError:
            return "omnivoice"
        return "piper"
    return "omnivoice"


def load_omnivoice_runtime(repo: Path, device: str):
    """Load the pinned local OmniVoice model and its bounded CUDA reserve."""
    import torch
    from omnivoice.models.omnivoice import OmniVoice

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "the selected runtime profile requires CUDA for speech synthesis")
    model = OmniVoice.from_pretrained(
        str(pinned_omnivoice_model_path(repo)), device_map=device).eval()
    model.audio_tokenizer.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(20260821)
    reserve = None
    if device.startswith("cuda"):
        _ = (torch.zeros(64, 64, device=device)
             @ torch.zeros(64, 64, device=device))
        reserve = torch.empty(
            96 * 1024 * 1024 // 4, dtype=torch.float32, device=device)
    return model, reserve


class PiperSpeechSynthesizer:
    """Persistent CPU Piper voice with bounded, validated float output."""

    backend_name = "piper"
    voice_name = "kristin"

    def __init__(self, repo: Path, *, output_rate: int = 24_000):
        if not 8_000 <= output_rate <= 96_000:
            raise ValueError("speech output rate is outside the supported range")
        model_path = verify_pinned_piper_voice(repo)
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise RuntimeError(
                "Piper is unavailable; sync requirements/runtime.lock") from exc
        self._voice = PiperVoice.load(str(model_path), use_cuda=False)
        self._lock = threading.Lock()
        self.output_rate = output_rate
        self.model_path = model_path

    def synthesize(self, text: str) -> np.ndarray:
        value = " ".join(str(text).split())
        if not value:
            return np.zeros(0, dtype=np.float32)
        if len(value) > MAX_SPEECH_TEXT_CHARS:
            raise ValueError("speech text exceeds the bounded input limit")

        arrays: list[np.ndarray] = []
        source_rate: int | None = None
        source_samples = 0
        with self._lock:
            for chunk in self._voice.synthesize(value):
                if chunk.sample_width != 2 or chunk.sample_channels != 1:
                    raise RuntimeError("Piper returned an unsupported audio format")
                if source_rate is None:
                    source_rate = int(chunk.sample_rate)
                elif int(chunk.sample_rate) != source_rate:
                    raise RuntimeError("Piper changed sample rate within one utterance")
                audio = np.asarray(chunk.audio_float_array, dtype=np.float32)
                if audio.ndim != 1 or not np.all(np.isfinite(audio)):
                    raise RuntimeError("Piper returned malformed audio")
                source_samples += len(audio)
                if (source_rate <= 0
                        or source_samples > source_rate * MAX_SOURCE_AUDIO_SECONDS):
                    raise RuntimeError("Piper audio exceeded the bounded output limit")
                arrays.append(audio)

        if not arrays or source_rate is None:
            raise RuntimeError("Piper returned no audio")
        result = np.concatenate(arrays)
        if source_rate != self.output_rate:
            divisor = gcd(source_rate, self.output_rate)
            result = resample_poly(
                result, self.output_rate // divisor,
                source_rate // divisor).astype(np.float32, copy=False)
        result = np.clip(result, -1.0, 1.0).astype(np.float32, copy=False)
        if (not np.all(np.isfinite(result))
                or float(np.sqrt(np.mean(np.square(result)))) < 1e-5):
            raise RuntimeError("Piper returned silent or invalid audio")
        return result
