import unittest

import numpy as np

from friday_core.endpointing import UtteranceBuffer


class EndpointingTests(unittest.TestCase):
    def test_pre_roll_preserves_audio_before_vad_trigger(self):
        endpoint = UtteranceBuffer(
            1000, pre_roll_ms=300, post_roll_ms=200,
            silence_end_ms=400, barge_in_ms=200)
        for _ in range(3):
            endpoint.feed_listening(np.full(100, 0.25, dtype=np.float32), False)

        started, _ = endpoint.feed_listening(
            np.ones(100, dtype=np.float32), True)
        completed = None
        for _ in range(4):
            _, completed = endpoint.feed_listening(
                np.zeros(100, dtype=np.float32), False)

        self.assertTrue(started)
        self.assertIsNotNone(completed)
        np.testing.assert_allclose(completed[:300], 0.25)
        np.testing.assert_allclose(completed[300:400], 1.0)

    def test_post_roll_keeps_low_energy_word_tail(self):
        endpoint = UtteranceBuffer(
            1000, pre_roll_ms=0, post_roll_ms=200,
            silence_end_ms=400, barge_in_ms=200)
        endpoint.feed_listening(np.ones(100, dtype=np.float32), True)
        completed = None
        for _ in range(4):
            _, completed = endpoint.feed_listening(
                np.full(100, 0.1, dtype=np.float32), False)

        self.assertEqual(len(completed), 300)
        np.testing.assert_allclose(completed[-200:], 0.1)

    def test_barge_in_audio_becomes_start_of_next_utterance(self):
        endpoint = UtteranceBuffer(
            1000, pre_roll_ms=0, post_roll_ms=100,
            silence_end_ms=300, barge_in_ms=200)

        self.assertFalse(endpoint.feed_barge_in(
            np.full(100, 0.6, dtype=np.float32), True))
        self.assertTrue(endpoint.feed_barge_in(
            np.full(100, 0.7, dtype=np.float32), True))
        completed = None
        for _ in range(3):
            _, completed = endpoint.feed_listening(
                np.zeros(100, dtype=np.float32), False)

        np.testing.assert_allclose(completed[:100], 0.6)
        np.testing.assert_allclose(completed[100:200], 0.7)


if __name__ == "__main__":
    unittest.main()
