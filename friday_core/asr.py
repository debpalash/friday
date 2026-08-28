"""Speech-recognition backends used by Friday.

Parakeet TDT v3 is the primary recognizer.  Faster-Whisper remains available as
a boot-time fallback so a missing/corrupt local model does not take the whole
assistant offline.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
PARAKEET_MODEL_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"


def _mono_16khz(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return contiguous mono float32 samples at Parakeet's native rate."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"expected mono or channel-last audio, got {audio.shape}")
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    if sample_rate != SAMPLE_RATE and audio.size:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        audio = resample_poly(
            audio, SAMPLE_RATE // divisor, int(sample_rate) // divisor
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(audio.reshape(-1), dtype=np.float32)


class ParakeetASR:
    """CPU int8 Parakeet TDT v3 through Sherpa-ONNX."""

    name = "parakeet-tdt-0.6b-v3-int8"
    device = "cpu"

    def __init__(self, model_dir: Path | str, *, recognizer=None, num_threads: int = 4):
        self.model_dir = Path(model_dir).resolve()
        self._lock = threading.Lock()
        if recognizer is not None:
            self._recognizer = recognizer
            return

        assets = {
            "encoder": self.model_dir / "encoder.int8.onnx",
            "decoder": self.model_dir / "decoder.int8.onnx",
            "joiner": self.model_dir / "joiner.int8.onnx",
            "tokens": self.model_dir / "tokens.txt",
        }
        missing = [path.name for path in assets.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Parakeet model is incomplete at {self.model_dir}: "
                + ", ".join(missing)
            )

        import sherpa_onnx

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(assets["encoder"]),
            decoder=str(assets["decoder"]),
            joiner=str(assets["joiner"]),
            tokens=str(assets["tokens"]),
            num_threads=max(1, int(num_threads)),
            provider="cpu",
            decoding_method="greedy_search",
            model_type="nemo_transducer",
        )

    def transcribe_samples(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        audio = _mono_16khz(samples, sample_rate)
        if not audio.size:
            return ""
        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            self._recognizer.decode_stream(stream)
            return str(stream.result.text or "").strip()

    def transcribe_file(self, path: Path | str) -> str:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        return self.transcribe_samples(audio, int(sample_rate))


class FasterWhisperASR:
    """Compatibility fallback for systems without a usable Parakeet model."""

    name = "faster-whisper-small-fallback"
    device = "cpu"

    def __init__(self):
        from faster_whisper import WhisperModel

        self._model = WhisperModel("small", device="cpu", compute_type="int8")
        self._lock = threading.Lock()

    def _transcribe(self, source) -> str:
        with self._lock:
            segments, _ = self._model.transcribe(
                source, language="en", beam_size=1, vad_filter=True
            )
            return " ".join(segment.text for segment in segments).strip()

    def transcribe_samples(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        return self._transcribe(_mono_16khz(samples, sample_rate))

    def transcribe_file(self, path: Path | str) -> str:
        return self._transcribe(str(path))


def load_asr(model_dir: Path | str):
    """Load the configured ASR, falling back only when explicitly permitted."""
    backend = os.environ.get("FRIDAY_ASR", "parakeet").strip().lower()
    if backend in {"whisper", "faster-whisper"}:
        return FasterWhisperASR()
    if backend not in {"parakeet", "parakeet-tdt"}:
        raise ValueError(f"unsupported FRIDAY_ASR backend: {backend}")
    try:
        threads = int(os.environ.get("FRIDAY_ASR_THREADS", "4"))
        return ParakeetASR(model_dir, num_threads=threads)
    except Exception as exc:
        allow_fallback = os.environ.get("FRIDAY_ASR_FALLBACK", "1").lower() not in {
            "0", "false", "off", "no"
        }
        if not allow_fallback:
            raise
        print(f"Parakeet ASR unavailable; loading Whisper fallback: {exc}", flush=True)
        return FasterWhisperASR()
