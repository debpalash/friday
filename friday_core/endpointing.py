"""Voice activity endpoint buffering for complete spoken utterances."""

from __future__ import annotations

import numpy as np


class PlaybackEchoGate:
    """Block microphone endpointing during playback and its acoustic tail."""

    def __init__(self, tail_ms: int = 650):
        if tail_ms < 0:
            raise ValueError("playback echo tail must not be negative")
        self.tail_seconds = tail_ms / 1000
        self._playing = False
        self._release_at = 0.0

    def start(self) -> None:
        self._playing = True

    def finish(self, now: float) -> None:
        self._playing = False
        self._release_at = max(self._release_at, now + self.tail_seconds)

    def blocks(self, now: float) -> bool:
        return self._playing or now < self._release_at


class UtteranceBuffer:
    """Preserve speech edges and carry barge-in audio into the next turn."""

    def __init__(self, sample_rate: int, *, pre_roll_ms: int = 350,
                 post_roll_ms: int = 250, silence_end_ms: int = 700,
                 barge_in_ms: int = 220, max_utterance_s: int = 30):
        self.sample_rate = sample_rate
        self.pre_roll_samples = sample_rate * pre_roll_ms // 1000
        self.post_roll_ms = post_roll_ms
        self.silence_end_ms = silence_end_ms
        self.barge_in_samples = sample_rate * barge_in_ms // 1000
        self.max_utterance_samples = sample_rate * max_utterance_s
        self.reset()

    def reset(self) -> None:
        self._pre: list[np.ndarray] = []
        self._speech: list[np.ndarray] = []
        self._barge: list[np.ndarray] = []
        self._pre_samples = 0
        self._speech_samples = 0
        self._barge_samples = 0
        self.spoke = False
        self.silent_ms = 0.0

    def _append_pre_roll(self, frame: np.ndarray) -> None:
        self._pre.append(frame)
        self._pre_samples += len(frame)
        while self._pre and self._pre_samples > self.pre_roll_samples:
            excess = self._pre_samples - self.pre_roll_samples
            if excess >= len(self._pre[0]):
                self._pre_samples -= len(self._pre.pop(0))
            else:
                self._pre[0] = self._pre[0][excess:]
                self._pre_samples -= excess

    def feed_listening(self, frame: np.ndarray,
                       is_speech: bool) -> tuple[bool, np.ndarray | None]:
        """Return (speech_started, completed_pcm)."""
        frame = np.asarray(frame, dtype=np.float32)
        started = False
        if is_speech:
            if not self.spoke:
                self.spoke = True
                started = True
                self._speech = [*self._pre, frame]
                self._speech_samples = self._pre_samples + len(frame)
                self._pre, self._pre_samples = [], 0
            else:
                self._speech.append(frame)
                self._speech_samples += len(frame)
            self.silent_ms = 0.0
        elif self.spoke:
            self._speech.append(frame)
            self._speech_samples += len(frame)
            self.silent_ms += len(frame) / self.sample_rate * 1000
            if self.silent_ms >= self.silence_end_ms:
                return started, self._finish()
        else:
            self._append_pre_roll(frame)

        if self.spoke and self._speech_samples >= self.max_utterance_samples:
            return started, self._finish(trim_silence=False)
        return started, None

    def feed_barge_in(self, frame: np.ndarray, is_speech: bool) -> bool:
        """Return true once sustained interruption speech should start a turn."""
        frame = np.asarray(frame, dtype=np.float32)
        if not is_speech:
            self._barge, self._barge_samples = [], 0
            return False
        self._barge.append(frame)
        self._barge_samples += len(frame)
        if self._barge_samples < self.barge_in_samples:
            return False
        self._speech = self._barge
        self._speech_samples = self._barge_samples
        self._barge, self._barge_samples = [], 0
        self._pre, self._pre_samples = [], 0
        self.spoke = True
        self.silent_ms = 0.0
        return True

    def _finish(self, *, trim_silence: bool = True) -> np.ndarray:
        pcm = np.concatenate(self._speech) if self._speech else np.zeros(
            0, dtype=np.float32)
        if trim_silence:
            removable_ms = max(0.0, self.silent_ms - self.post_roll_ms)
            trim = int(removable_ms / 1000 * self.sample_rate)
            if trim:
                pcm = pcm[:max(0, len(pcm) - trim)]
        self.reset()
        return pcm
