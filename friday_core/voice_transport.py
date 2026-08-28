"""Per-connection voice transport state and playback echo fencing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np

from .endpointing import PlaybackEchoGate, UtteranceBuffer


@dataclass
class VoiceTransportSession:
    """Mutable state owned by exactly one authorized WebSocket session."""

    utterance: UtteranceBuffer
    echo_gate: PlaybackEchoGate
    mode: str = "listen"
    interrupt: asyncio.Event = field(default_factory=asyncio.Event)
    active_speaker_task: asyncio.Task | None = None
    vad_carry: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32))
    frame_count: int = 0

    @classmethod
    def create(
        cls, *, sample_rate: int, pre_roll_ms: int, post_roll_ms: int,
        silence_end_ms: int, barge_in_ms: int, max_utterance_s: int,
        playback_echo_tail_ms: int,
    ) -> "VoiceTransportSession":
        return cls(
            utterance=UtteranceBuffer(
                sample_rate, pre_roll_ms=pre_roll_ms,
                post_roll_ms=post_roll_ms, silence_end_ms=silence_end_ms,
                barge_in_ms=barge_in_ms, max_utterance_s=max_utterance_s,
            ),
            echo_gate=PlaybackEchoGate(playback_echo_tail_ms),
        )

    def playback_started(self) -> None:
        self.echo_gate.start()
        self.mode = "speak"
        self.reset_audio_input()

    def playback_ended(self, now: float) -> None:
        self.echo_gate.finish(now)
        self.reset_audio_input()

    def reset_audio_input(self) -> None:
        self.utterance.reset()
        self.vad_carry = np.zeros(0, dtype=np.float32)

    def playback_blocks_input(self, now: float) -> bool:
        blocked = self.echo_gate.blocks(now)
        if blocked:
            self.reset_audio_input()
        return blocked

    def vad_frames(self, samples: np.ndarray, *, frame_size: int = 512) -> list[np.ndarray]:
        if frame_size < 1:
            raise ValueError("frame_size must be positive")
        self.vad_carry = np.concatenate([self.vad_carry, samples])
        count = len(self.vad_carry) // frame_size
        if not count:
            return []
        frames = list(self.vad_carry[: count * frame_size].reshape(
            count, frame_size))
        self.vad_carry = self.vad_carry[count * frame_size:]
        return frames

    def next_frame(self) -> int:
        self.frame_count += 1
        return self.frame_count
