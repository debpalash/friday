"""Silero voice-activity detection on onnxruntime.

This is the same model Friday ran through TorchScript before the port, using
the ONNX file from the pinned upstream release, so it needs no torch and runs
on every platform. The streaming state machine mirrors Silero's own ONNX
wrapper: 64 samples of context are prepended to each 512-sample frame and
the recurrent state is carried between calls.
"""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

import numpy as np

from friday_host import fs

VAD_MODEL_REPOSITORY = "snakers4/silero-vad"
VAD_MODEL_REVISION = "7e30209a3e901f9842f81b225f3e93d8199902b1"  # tag v6.2.1
VAD_MODEL_DIRECTORY = f"silero-vad-{VAD_MODEL_REVISION[:8]}"
VAD_MODEL_FILE = "silero_vad.onnx"
VAD_MODEL_SIZE = 2_327_524
VAD_MODEL_SHA256 = (
    "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3")
VAD_MODEL_URL = (
    f"https://raw.githubusercontent.com/{VAD_MODEL_REPOSITORY}/"
    f"{VAD_MODEL_REVISION}/src/silero_vad/data/{VAD_MODEL_FILE}")

FRAME_SAMPLES = {16000: 512, 8000: 256}
CONTEXT_SAMPLES = {16000: 64, 8000: 32}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_vad_model_path(repo: str | Path) -> Path:
    """Return the verified pinned ONNX file or raise ``RuntimeError``."""
    path = Path(repo) / "models" / VAD_MODEL_DIRECTORY / VAD_MODEL_FILE
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(
            "the pinned Silero VAD model is unavailable; run "
            "ops/install_vad_model.py") from exc
    if (not stat.S_ISREG(metadata.st_mode) or path.is_symlink()
            or not fs.owned_by_caller(metadata)
            or metadata.st_size != VAD_MODEL_SIZE
            or not fs.private_mode_ok(metadata, mask=0o022)
            or _sha256_file(path) != VAD_MODEL_SHA256):
        raise RuntimeError("the pinned Silero VAD model boundary is invalid")
    return path


class SileroOnnxVad:
    """Streaming speech probability for exact 512-sample 16 kHz frames."""

    def __init__(self, model_path: str | Path, *, threads: int = 1,
                 session=None) -> None:
        self.model_path = Path(model_path)
        if session is None:
            import onnxruntime  # noqa: PLC0415 - heavy import stays lazy

            options = onnxruntime.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = max(1, int(threads))
            session = onnxruntime.InferenceSession(
                str(self.model_path), sess_options=options,
                providers=["CPUExecutionProvider"])
        self.session = session
        self._sample_rate = 0
        self.reset_states()

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context: np.ndarray | None = None
        self._sample_rate = 0

    def __call__(self, samples: np.ndarray, sample_rate: int) -> float:
        if sample_rate != 16000 and sample_rate % 16000 == 0:
            samples = np.asarray(samples)[:: sample_rate // 16000]
            sample_rate = 16000
        if sample_rate not in FRAME_SAMPLES:
            raise ValueError("supported sampling rates: 8000 and 16000")
        frame = np.ascontiguousarray(
            np.asarray(samples, dtype=np.float32).reshape(-1))
        expected = FRAME_SAMPLES[sample_rate]
        if frame.shape[0] != expected:
            raise ValueError(
                f"provided number of samples is {frame.shape[0]} "
                f"(expected {expected} for {sample_rate} Hz)")
        context_size = CONTEXT_SAMPLES[sample_rate]
        if self._sample_rate and self._sample_rate != sample_rate:
            self.reset_states()
        self._sample_rate = sample_rate
        if self._context is None:
            self._context = np.zeros((1, context_size), dtype=np.float32)
        window = np.concatenate([self._context, frame[None, :]], axis=1)
        output, state = self.session.run(None, {
            "input": window,
            "state": self._state,
            "sr": np.array(sample_rate, dtype=np.int64),
        })
        self._state = np.asarray(state, dtype=np.float32)
        self._context = window[:, -context_size:]
        return float(np.asarray(output).reshape(-1)[0])


__all__ = [
    "SileroOnnxVad", "VAD_MODEL_DIRECTORY", "VAD_MODEL_FILE",
    "VAD_MODEL_REVISION", "VAD_MODEL_SHA256", "VAD_MODEL_SIZE",
    "VAD_MODEL_URL", "pinned_vad_model_path",
]
