import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from friday_core.asr import ParakeetASR, _mono_16khz


class _FakeStream:
    def __init__(self):
        self.received = None
        self.result = SimpleNamespace(text="  understood speech.  ")

    def accept_waveform(self, sample_rate, samples):
        self.received = (sample_rate, samples)


class _FakeRecognizer:
    def __init__(self):
        self.stream = _FakeStream()
        self.decoded = False

    def create_stream(self):
        return self.stream

    def decode_stream(self, stream):
        self.decoded = stream is self.stream


class ParakeetBackendTests(unittest.TestCase):
    def test_transcribe_uses_16khz_contiguous_float32(self):
        recognizer = _FakeRecognizer()
        backend = ParakeetASR("unused-for-injected-recognizer", recognizer=recognizer)

        text = backend.transcribe_samples(np.ones((48_000, 2)), sample_rate=48_000)

        sample_rate, samples = recognizer.stream.received
        self.assertEqual(text, "understood speech.")
        self.assertTrue(recognizer.decoded)
        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(samples.dtype, np.float32)
        self.assertTrue(samples.flags.c_contiguous)
        self.assertAlmostEqual(len(samples) / sample_rate, 1.0, places=2)

    def test_missing_model_assets_fail_with_names(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError) as raised:
                ParakeetASR(Path(directory))

        message = str(raised.exception)
        self.assertIn("encoder.int8.onnx", message)
        self.assertIn("tokens.txt", message)

    def test_empty_audio_is_empty_transcript(self):
        recognizer = _FakeRecognizer()
        backend = ParakeetASR("unused-for-injected-recognizer", recognizer=recognizer)

        self.assertEqual(backend.transcribe_samples(np.array([], dtype=np.float32)), "")
        self.assertFalse(recognizer.decoded)

    def test_resampler_rejects_invalid_shape(self):
        with self.assertRaises(ValueError):
            _mono_16khz(np.zeros((2, 2, 2), dtype=np.float32), 16_000)
