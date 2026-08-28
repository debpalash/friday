import unittest

import numpy as np

from friday_core.voice_transport import VoiceTransportSession


def session() -> VoiceTransportSession:
    return VoiceTransportSession.create(
        sample_rate=16_000, pre_roll_ms=350, post_roll_ms=250,
        silence_end_ms=700, barge_in_ms=220, max_utterance_s=30,
        playback_echo_tail_ms=650,
    )


class VoiceTransportSessionTests(unittest.TestCase):
    def test_playback_and_tail_drop_microphone_state_before_vad(self):
        state = session()
        state.vad_frames(np.ones(300, dtype=np.float32))
        state.playback_started()

        self.assertEqual(state.mode, "speak")
        self.assertEqual(state.vad_carry.size, 0)
        state.playback_ended(10.0)
        self.assertTrue(state.playback_blocks_input(10.649))
        self.assertFalse(state.playback_blocks_input(10.651))

    def test_vad_frame_carry_is_connection_local_and_lossless(self):
        state = session()
        first = np.arange(700, dtype=np.float32)
        frames = state.vad_frames(first)

        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(frames[0], first[:512])
        self.assertEqual(state.vad_carry.size, 188)

        second = np.arange(324, dtype=np.float32) + 700
        frames = state.vad_frames(second)
        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(
            frames[0], np.concatenate([first[512:], second]))
        self.assertEqual(state.vad_carry.size, 0)

    def test_interrupt_state_is_not_shared_between_sessions(self):
        first = session()
        second = session()
        first.interrupt.set()

        self.assertTrue(first.interrupt.is_set())
        self.assertFalse(second.interrupt.is_set())


if __name__ == "__main__":
    unittest.main()
