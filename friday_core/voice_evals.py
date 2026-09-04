"""Artifact-backed, content-free qualification for Friday's voice path."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from friday_host import fs

from .endpointing import PlaybackEchoGate, UtteranceBuffer
from .graph import GraphStore, utc_now


MAX_VOICE_SUITE_BYTES = 128_000
_WORD = re.compile(r"[a-z0-9]+")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 3)


def _words(value: str) -> list[str]:
    return _WORD.findall(value.casefold())


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_word in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_word in enumerate(right, start=1):
            current.append(min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + (left_word != right_word),
            ))
        previous = current
    return previous[-1]


def _bounded_text(value: Any, field: str) -> str:
    if (not isinstance(value, str) or not 2 <= len(value) <= 240
            or any(ord(character) < 32 for character in value)):
        raise ValueError(f"voice {field} is invalid")
    return value


class VoiceEvalRunner:
    """Measure real local speech artifacts without retaining their content."""

    def __init__(
        self,
        graph: GraphStore,
        asr: Any,
        synthesizer: Any,
        *,
        sample_rate: int = 24_000,
    ):
        if not 8_000 <= sample_rate <= 96_000:
            raise ValueError("voice evaluation sample rate is invalid")
        self.graph = graph
        self.asr = asr
        self.synthesizer = synthesizer
        self.sample_rate = sample_rate
        self.asr_name = str(getattr(asr, "name", "unknown"))[:120]
        self.tts_name = str(
            getattr(synthesizer, "backend_name", "unknown"))[:120]
        self.voice_name = str(
            getattr(synthesizer, "voice_name", "unknown"))[:120]

    @staticmethod
    def _load_suite(path: str | Path) -> tuple[dict[str, Any], str]:
        try:
            descriptor = os.open(
                Path(path), os.O_RDONLY | fs.PRIVATE_OPEN_FLAGS)
        except OSError as exc:
            raise ValueError("voice suite is unavailable") from exc
        try:
            metadata = os.fstat(descriptor)
            if (not stat.S_ISREG(metadata.st_mode)
                    or not 2 <= metadata.st_size <= MAX_VOICE_SUITE_BYTES):
                raise ValueError("voice suite must be a bounded regular file")
            encoded = os.read(descriptor, MAX_VOICE_SUITE_BYTES + 1)
            if len(encoded) != metadata.st_size:
                raise ValueError("voice suite changed while being read")
        finally:
            os.close(descriptor)
        try:
            suite = json.loads(
                encoded.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite value: {value}")),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("voice suite is invalid JSON") from exc
        if (not isinstance(suite, dict)
                or set(suite) != {
                    "name", "version", "gates", "utterances", "reply",
                    "echo_tail_ms", "barge_in_ms", "repetitions",
                }
                or suite.get("name") != "friday-voice"
                or suite.get("version") != 1):
            raise ValueError("voice suite metadata is invalid")
        VoiceEvalRunner._validate_gates(suite["gates"])
        utterances = suite["utterances"]
        if (not isinstance(utterances, list) or not 5 <= len(utterances) <= 32
                or len(set(utterances)) != len(utterances)):
            raise ValueError("voice utterances are invalid")
        for utterance in utterances:
            _bounded_text(utterance, "utterance")
        if (isinstance(suite["repetitions"], bool)
                or not isinstance(suite["repetitions"], int)
                or not 2 <= suite["repetitions"] <= 5):
            raise ValueError("voice repetitions are invalid")
        _bounded_text(suite["reply"], "reply")
        for field, minimum, maximum in (
                ("echo_tail_ms", 100, 2_000),
                ("barge_in_ms", 100, 1_000)):
            value = suite[field]
            if (isinstance(value, bool) or not isinstance(value, int)
                    or not minimum <= value <= maximum):
                raise ValueError(f"voice {field} is invalid")
        return suite, hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _validate_gates(value: Any) -> None:
        expected = {
            "maximum_word_error_rate", "minimum_exact_rate",
            "maximum_asr_rtf_p95", "maximum_tts_rtf_p95",
            "maximum_first_audio_p95_ms", "minimum_rms",
            "maximum_clipped_ratio", "minimum_barge_capture_ratio",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("voice gates are invalid")
        for field in expected:
            item = value[field]
            if (isinstance(item, bool) or not isinstance(item, (int, float))
                    or not math.isfinite(float(item)) or float(item) < 0):
                raise ValueError("voice gate is invalid")
        if (not 0 <= value["maximum_word_error_rate"] <= 1
                or not 0 <= value["minimum_exact_rate"] <= 1
                or not 0 <= value["maximum_clipped_ratio"] <= 1
                or not 0 <= value["minimum_barge_capture_ratio"] <= 1
                or not 0 < value["minimum_rms"] <= 1):
            raise ValueError("voice quality gate is invalid")

    def _synthesize(self, text: str) -> tuple[np.ndarray, float]:
        started = time.perf_counter_ns()
        audio = np.asarray(
            self.synthesizer.synthesize(text), dtype=np.float32).reshape(-1)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        if (not audio.size or not np.all(np.isfinite(audio))
                or audio.size > self.sample_rate * 120):
            raise RuntimeError("voice synthesizer returned an invalid artifact")
        return audio, elapsed_ms

    def _transcribe_artifact(self, path: Path) -> tuple[str, float]:
        started = time.perf_counter_ns()
        if hasattr(self.asr, "transcribe_file"):
            result = self.asr.transcribe_file(path)
        else:
            audio, rate = sf.read(path, dtype="float32")
            result = self.asr.transcribe_samples(audio, int(rate))
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        return str(result).strip(), elapsed_ms

    def _quality_and_latency(
        self,
        suite: dict[str, Any],
        temporary: Path,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str], np.ndarray]:
        tts_ms: list[float] = []
        tts_rtf: list[float] = []
        asr_ms: list[float] = []
        asr_rtf: list[float] = []
        first_audio_ms: list[float] = []
        rms_values: list[float] = []
        clipped_ratios: list[float] = []
        exact = 0
        edits = 0
        reference_words = 0
        artifact_hashes: list[str] = []
        interruption_audio = np.zeros(0, dtype=np.float32)

        artifact_index = 0
        for repetition in range(suite["repetitions"]):
            for phrase_index, phrase in enumerate(suite["utterances"]):
                audio, synth_ms = self._synthesize(phrase)
                if artifact_index == 0:
                    interruption_audio = audio.copy()
                duration = len(audio) / self.sample_rate
                path = temporary / (
                    f"utterance-{repetition:02d}-{phrase_index:02d}.wav")
                sf.write(path, audio, self.sample_rate, subtype="PCM_16")
                os.chmod(path, 0o600)
                encoded = path.read_bytes()
                artifact_hashes.append(hashlib.sha256(encoded).hexdigest())
                transcript, recognize_ms = self._transcribe_artifact(path)
                reference = _words(phrase)
                observed = _words(transcript)
                distance = _edit_distance(reference, observed)
                edits += distance
                reference_words += len(reference)
                exact += int(distance == 0)
                rms_values.append(float(np.sqrt(np.mean(np.square(audio)))))
                clipped_ratios.append(float(np.mean(np.abs(audio) >= 0.999)))
                tts_ms.append(synth_ms)
                tts_rtf.append(synth_ms / max(duration * 1_000, 1e-9))
                asr_ms.append(recognize_ms)
                asr_rtf.append(recognize_ms / max(duration * 1_000, 1e-9))

                # Voice turn latency is measured from an already captured
                # speech artifact through ASR, deterministic local reply
                # selection, and synthesis of the first complete response.
                turn_started = time.perf_counter_ns()
                self._transcribe_artifact(path)
                reply_audio, _ = self._synthesize(suite["reply"])
                if not reply_audio.size:
                    raise RuntimeError("voice reply produced no audio")
                first_audio_ms.append(
                    (time.perf_counter_ns() - turn_started) / 1_000_000)
                artifact_index += 1

        artifact_count = len(suite["utterances"]) * suite["repetitions"]
        quality = {
            "unique_utterances": len(suite["utterances"]),
            "repetitions": suite["repetitions"],
            "utterances": artifact_count,
            "reference_words": reference_words,
            "word_errors": edits,
            "word_error_rate": edits / max(reference_words, 1),
            "exact_utterances": exact,
            "exact_rate": exact / artifact_count,
            "minimum_rms": round(min(rms_values), 6),
            "maximum_clipped_ratio": round(max(clipped_ratios), 8),
        }
        latency = {
            "asr_p50_ms": _percentile(asr_ms, 0.50),
            "asr_p95_ms": _percentile(asr_ms, 0.95),
            "asr_rtf_p50": _percentile(asr_rtf, 0.50),
            "asr_rtf_p95": _percentile(asr_rtf, 0.95),
            "tts_p50_ms": _percentile(tts_ms, 0.50),
            "tts_p95_ms": _percentile(tts_ms, 0.95),
            "tts_rtf_p50": _percentile(tts_rtf, 0.50),
            "tts_rtf_p95": _percentile(tts_rtf, 0.95),
            "first_audio_p50_ms": _percentile(first_audio_ms, 0.50),
            "first_audio_p95_ms": _percentile(first_audio_ms, 0.95),
            "language_stage": "deterministic_local_fixture",
        }
        return quality, latency, artifact_hashes, interruption_audio

    def _echo_and_interruption(
        self,
        audio: np.ndarray,
        suite: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        frame_samples = max(1, self.sample_rate // 50)
        frames = [audio[index:index + frame_samples]
                  for index in range(0, len(audio), frame_samples)]
        echo_gate = PlaybackEchoGate(suite["echo_tail_ms"])
        echo_gate.start()
        rejected = sum(1 for index, _frame in enumerate(frames)
                       if echo_gate.blocks(index * 0.02))
        playback_end = len(frames) * 0.02
        echo_gate.finish(playback_end)
        tail_probe = playback_end + suite["echo_tail_ms"] / 1_000 - 0.001
        tail_blocked = echo_gate.blocks(tail_probe)
        release_probe = playback_end + suite["echo_tail_ms"] / 1_000
        released = not echo_gate.blocks(release_probe)
        forwarded = len(frames) - rejected

        endpoint = UtteranceBuffer(
            self.sample_rate,
            pre_roll_ms=0,
            post_roll_ms=100,
            silence_end_ms=300,
            barge_in_ms=suite["barge_in_ms"],
        )
        triggered_at_ms: float | None = None
        speech_samples = 0
        for index, frame in enumerate(frames):
            speech_samples += len(frame)
            if endpoint.feed_barge_in(frame, True):
                triggered_at_ms = round(speech_samples / self.sample_rate * 1_000, 3)
                break
        completed = None
        if triggered_at_ms is not None:
            for _ in range(15):
                _, completed = endpoint.feed_listening(
                    np.zeros(frame_samples, dtype=np.float32), False)
                if completed is not None:
                    break
        retained = (min(speech_samples, len(completed))
                    if completed is not None else 0)
        capture_ratio = retained / max(speech_samples, 1)
        prefix_matches = bool(
            completed is not None
            and retained > 0
            and np.allclose(completed[:retained], audio[:retained], atol=1e-6))
        return ({
            "playback_frames": len(frames),
            "frames_rejected_before_vad": rejected,
            "frames_forwarded_to_vad": forwarded,
            "tail_blocked": tail_blocked,
            "released_after_tail": released,
        }, {
            "triggered": triggered_at_ms is not None,
            "triggered_at_ms": triggered_at_ms,
            "configured_barge_in_ms": suite["barge_in_ms"],
            "captured_speech_ratio": round(capture_ratio, 6),
            "captured_prefix_matches": prefix_matches,
        })

    def run(self, suite_path: str | Path) -> dict[str, Any]:
        suite, suite_sha256 = self._load_suite(suite_path)
        temporary_path: Path | None = None
        with tempfile.TemporaryDirectory(prefix="friday-voice-eval-") as value:
            temporary_path = Path(value)
            os.chmod(temporary_path, 0o700)
            quality, latency, artifact_hashes, interruption_audio = (
                self._quality_and_latency(suite, temporary_path))
            echo, interruption = self._echo_and_interruption(
                interruption_audio, suite)
            artifact_set_sha256 = hashlib.sha256(
                "".join(artifact_hashes).encode("ascii")).hexdigest()
            private_artifacts = all(
                path.stat().st_mode & 0o077 == 0
                for path in temporary_path.iterdir())
        cleanup_verified = bool(
            temporary_path is not None and not temporary_path.exists())

        gates = suite["gates"]
        checks = {
            "asr_quality": (
                quality["word_error_rate"]
                <= gates["maximum_word_error_rate"]
                and quality["exact_rate"] >= gates["minimum_exact_rate"]),
            "speech_signal": (
                quality["minimum_rms"] >= gates["minimum_rms"]
                and quality["maximum_clipped_ratio"]
                <= gates["maximum_clipped_ratio"]),
            "asr_latency": (
                latency["asr_rtf_p95"] <= gates["maximum_asr_rtf_p95"]),
            "speech_latency": (
                latency["tts_rtf_p95"] <= gates["maximum_tts_rtf_p95"]),
            "first_audio_latency": (
                latency["first_audio_p95_ms"]
                <= gates["maximum_first_audio_p95_ms"]),
            "echo_rejection": (
                echo["frames_forwarded_to_vad"] == 0
                and echo["tail_blocked"]
                and echo["released_after_tail"]),
            "interruption": (
                interruption["triggered"]
                and interruption["captured_prefix_matches"]
                and interruption["captured_speech_ratio"]
                >= gates["minimum_barge_capture_ratio"]),
            "artifact_privacy": private_artifacts and cleanup_verified,
        }
        body = {
            "suite": suite["name"],
            "version": suite["version"],
            "suite_sha256": suite_sha256,
            "artifact_set_sha256": artifact_set_sha256,
            "artifact_count": len(artifact_hashes),
            "artifacts_retained": 0,
            "asr": {"backend": self.asr_name, "device": str(
                getattr(self.asr, "device", "unknown"))[:40]},
            "tts": {"backend": self.tts_name, "voice": self.voice_name,
                    "sample_rate": self.sample_rate},
            "quality": quality,
            "latency": latency,
            "echo": echo,
            "interruption": interruption,
            "gates": gates,
            "checks": checks,
            "passed": all(checks.values()),
            "privacy": {
                "input_source": "disposable_local_synthesis",
                "microphone_content_used": False,
                "transcripts_persisted": False,
                "artifacts_private_while_live": private_artifacts,
                "artifact_cleanup_verified": cleanup_verified,
            },
            "ran_at": utc_now(),
        }
        run_id = self.graph.record_node(
            "voice_evaluation_run", body,
            actor="voice_eval_runner",
            event_type="evaluation.voice_completed",
        )
        return {"evaluation_run_id": run_id, **body}
