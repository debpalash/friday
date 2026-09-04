"""The ONNX voice-activity detector reproduces the TorchScript model's output."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from friday_core import vad
from friday_core.vad import SileroOnnxVad, pinned_vad_model_path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vad" / "silero_probabilities_v1.json"


def _fixture_audio(fixture: dict) -> np.ndarray:
    """Decode the recorded 16 kHz int16 speech sample."""
    import base64

    raw = base64.b64decode(fixture["audio_int16_b64"])
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0


class PinnedModelTests(unittest.TestCase):
    def test_missing_or_unpinned_model_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "install_vad_model"):
                pinned_vad_model_path(temporary)
            bad = Path(temporary) / "models" / vad.VAD_MODEL_DIRECTORY
            bad.mkdir(parents=True)
            (bad / vad.VAD_MODEL_FILE).write_bytes(b"not a model")
            with self.assertRaisesRegex(RuntimeError, "boundary is invalid"):
                pinned_vad_model_path(temporary)

    def test_pin_constants_are_exact(self) -> None:
        self.assertEqual(len(vad.VAD_MODEL_SHA256), 64)
        self.assertIn(vad.VAD_MODEL_REVISION, vad.VAD_MODEL_URL)
        self.assertTrue(vad.VAD_MODEL_URL.startswith("https://raw.githubusercontent.com/snakers4/silero-vad/"))


class FakeSession:
    """Records the exact tensors Friday feeds the model."""

    def __init__(self) -> None:
        self.calls = []

    def run(self, _outputs, feeds):
        self.calls.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        state = feeds["state"] + 1
        return np.array([[0.25]], dtype=np.float32), state


class StreamingContractTests(unittest.TestCase):
    def test_context_and_state_are_carried_between_frames(self) -> None:
        session = FakeSession()
        detector = SileroOnnxVad("unused.onnx", session=session)
        first = np.arange(512, dtype=np.float32)
        second = np.arange(512, 1024, dtype=np.float32)
        self.assertEqual(detector(first, 16000), 0.25)
        detector(second, 16000)
        self.assertEqual(session.calls[0]["input"].shape, (1, 576))
        self.assertTrue(np.all(session.calls[0]["input"][0, :64] == 0))
        np.testing.assert_array_equal(session.calls[0]["input"][0, 64:], first)
        np.testing.assert_array_equal(session.calls[1]["input"][0, :64], first[-64:])
        self.assertEqual(session.calls[0]["sr"].dtype, np.int64)
        self.assertEqual(session.calls[0]["state"].shape, (2, 1, 128))
        self.assertTrue(np.all(session.calls[1]["state"] == 1))
        detector.reset_states()
        detector(first, 16000)
        self.assertTrue(np.all(session.calls[2]["state"] == 0))
        self.assertTrue(np.all(session.calls[2]["input"][0, :64] == 0))

    def test_frame_size_and_rate_are_validated(self) -> None:
        detector = SileroOnnxVad("unused.onnx", session=FakeSession())
        with self.assertRaises(ValueError):
            detector(np.zeros(500, dtype=np.float32), 16000)
        with self.assertRaises(ValueError):
            detector(np.zeros(512, dtype=np.float32), 44100)
        detector(np.zeros(1024, dtype=np.float32), 32000)


class TorchParityTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.model_path = pinned_vad_model_path(REPO)
        except RuntimeError as exc:
            self.skipTest(f"environment: {exc}")

    def test_onnx_probabilities_match_the_recorded_torchscript_run(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        audio = _fixture_audio(fixture)
        detector = SileroOnnxVad(self.model_path)
        expected = np.asarray(fixture["probabilities"], dtype=np.float64)
        observed = []
        for offset in range(0, len(audio) - 511, 512):
            observed.append(detector(audio[offset:offset + 512], 16000))
        observed = np.asarray(observed)
        self.assertEqual(observed.shape, expected.shape)
        self.assertLess(float(np.max(np.abs(observed - expected))), 0.02)
        self.assertGreater(float(expected.max()), 0.5, "fixture contains speech-like frames")
        self.assertLess(float(expected.min()), 0.2, "fixture contains silent frames")


if __name__ == "__main__":
    unittest.main()
