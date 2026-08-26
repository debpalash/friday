import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from friday_core.embeddings import (
    MODEL_DIRECTORY, LocalTextEmbedder, configured_local_embedder)


REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "models" / MODEL_DIRECTORY


class LocalEmbeddingTests(unittest.TestCase):
    def test_configuration_is_explicit_bounded_and_has_safe_auto_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(configured_local_embedder(
                temporary, {"FRIDAY_EMBEDDING_MODEL": "auto"}))
        self.assertIsNone(configured_local_embedder(
            REPO, {"FRIDAY_EMBEDDING_MODEL": "disabled"}))
        with self.assertRaises(ValueError):
            configured_local_embedder(REPO, {
                "FRIDAY_EMBEDDING_MODEL": str(MODEL),
                "FRIDAY_EMBEDDING_BATCH_SIZE": "999",
            })
        with self.assertRaises(ValueError):
            LocalTextEmbedder(MODEL, batch_size=True)

    def test_missing_or_unpinned_model_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            embedder = LocalTextEmbedder(temporary)
            with self.assertRaises(RuntimeError):
                embedder.encode(["private text"], kind="passage")
        embedder = LocalTextEmbedder(MODEL)
        with self.assertRaises(ValueError):
            embedder.encode([], kind="query")
        with self.assertRaises(ValueError):
            embedder.encode(["x" * 4_001], kind="query")

    @unittest.skipUnless(MODEL.is_dir(), "pinned embedding checkpoint not installed")
    def test_real_pinned_encoder_is_cpu_local_normalized_and_multilingual(self):
        old_hub, old_transformers = (
            os.environ.get("HF_HUB_OFFLINE"),
            os.environ.get("TRANSFORMERS_OFFLINE"))
        try:
            embedder = LocalTextEmbedder(MODEL, batch_size=4)
            passages = embedder.encode([
                "Keep progress visible with status updates.",
                "Use blue interface controls.",
            ], kind="passage")
            queries = embedder.encode([
                "¿Cómo debo mostrar las actualizaciones de progreso?",
                "मुझे प्रगति अपडेट कैसे दिखाने चाहिए?",
            ], kind="query")
        finally:
            if old_hub is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_hub
            if old_transformers is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = old_transformers
        self.assertEqual(passages.shape, (2, 384))
        self.assertEqual(queries.shape, (2, 384))
        self.assertTrue(np.allclose(np.linalg.norm(passages, axis=1), 1, atol=1e-5))
        scores = queries @ passages.T
        self.assertTrue(np.all(scores[:, 0] > scores[:, 1]))
        self.assertEqual(next(embedder._model.parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
