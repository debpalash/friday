import hashlib
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from friday_core import speech


class SpeechBackendTests(unittest.TestCase):
    def test_auto_uses_omnivoice_for_cuda_without_piper_probe(self):
        with mock.patch.object(
                speech, "verify_pinned_piper_voice",
                side_effect=AssertionError("must not inspect CPU voice")):
            selected = speech.choose_speech_backend(
                tts_device="cuda", repo=Path("/missing"), environment={})

        self.assertEqual(selected, "omnivoice")

    def test_auto_cpu_falls_back_when_pinned_voice_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            selected = speech.choose_speech_backend(
                tts_device="cpu", repo=Path(temporary), environment={})

        self.assertEqual(selected, "omnivoice")

    def test_explicit_piper_fails_closed_without_pinned_voice(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
                RuntimeError, "voice directory"):
            speech.choose_speech_backend(
                tts_device="cpu", repo=Path(temporary),
                environment={"FRIDAY_TTS_BACKEND": "piper"})

    def test_invalid_backend_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "must be auto"):
            speech.choose_speech_backend(
                tts_device="cpu", repo=Path("/missing"),
                environment={"FRIDAY_TTS_BACKEND": "remote"})

    def test_voice_asset_verifier_accepts_only_exact_private_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            voice_dir = repo / "models" / speech.PIPER_VOICE_DIRECTORY
            voice_dir.mkdir(parents=True)
            model = voice_dir / speech.PIPER_MODEL_NAME
            config = voice_dir / speech.PIPER_CONFIG_NAME
            model.write_bytes(b"model")
            config.write_bytes(b"config")
            assets = {
                model.name: (5, hashlib.sha256(b"model").hexdigest()),
                config.name: (6, hashlib.sha256(b"config").hexdigest()),
            }
            with mock.patch.object(speech, "PIPER_ASSETS", assets):
                self.assertEqual(
                    speech.verify_pinned_piper_voice(repo), model)
                os.chmod(config, 0o666)
                with self.assertRaisesRegex(RuntimeError, "boundary"):
                    speech.verify_pinned_piper_voice(repo)

    def test_omnivoice_verifier_rejects_modified_or_writable_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            model_dir = repo / "models" / speech.OMNIVOICE_MODEL_DIRECTORY
            model_dir.mkdir(parents=True)
            config = model_dir / "config.json"
            weights = model_dir / "audio_tokenizer" / "model.safetensors"
            weights.parent.mkdir()
            config.write_bytes(b"config")
            weights.write_bytes(b"weights")
            assets = {
                "config.json": (6, hashlib.sha256(b"config").hexdigest()),
                "audio_tokenizer/model.safetensors": (
                    7, hashlib.sha256(b"weights").hexdigest()),
            }
            with mock.patch.object(speech, "OMNIVOICE_ASSETS", assets):
                self.assertEqual(
                    speech.pinned_omnivoice_model_path(repo), model_dir)
                weights.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "digest"):
                    speech.pinned_omnivoice_model_path(repo)
                weights.write_bytes(b"weights")
                os.chmod(weights, 0o666)
                with self.assertRaisesRegex(RuntimeError, "boundary"):
                    speech.pinned_omnivoice_model_path(repo)

    def test_synthesizer_rejects_silent_and_oversized_output(self):
        synthesizer = speech.PiperSpeechSynthesizer.__new__(
            speech.PiperSpeechSynthesizer)
        synthesizer.output_rate = 24_000
        synthesizer._lock = threading.Lock()
        synthesizer._voice = SimpleNamespace(synthesize=lambda _text: iter([
            SimpleNamespace(
                sample_width=2, sample_channels=1, sample_rate=24_000,
                audio_float_array=np.zeros(2400, dtype=np.float32)),
        ]))

        with self.assertRaisesRegex(RuntimeError, "silent"):
            synthesizer.synthesize("hello")
        with self.assertRaisesRegex(ValueError, "bounded input"):
            synthesizer.synthesize("x" * (speech.MAX_SPEECH_TEXT_CHARS + 1))

    def test_real_pinned_cpu_voice_is_audible_and_resampled(self):
        repo = Path(__file__).resolve().parents[1]
        try:
            synthesizer = speech.PiperSpeechSynthesizer(
                repo, output_rate=24_000)
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"pinned Piper runtime unavailable: {exc}")

        audio = synthesizer.synthesize("Friday speech is ready.")

        self.assertEqual(audio.dtype, np.float32)
        self.assertGreater(len(audio), 24_000 // 2)
        self.assertLessEqual(float(np.max(np.abs(audio))), 1.0)
        self.assertGreater(float(np.sqrt(np.mean(np.square(audio)))), 0.01)


if __name__ == "__main__":
    unittest.main()
