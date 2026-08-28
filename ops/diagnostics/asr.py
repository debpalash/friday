"""Manual Faster Whisper smoke test against generated TTS samples."""

import time
from faster_whisper import WhisperModel

t0 = time.time()
model = WhisperModel("small", device="cpu", compute_type="int8")
print(f"load: {time.time()-t0:.1f}s")

for f in ["/tmp/opencode/tts_happy.wav", "/tmp/opencode/tts_plain-designed.wav"]:
    t0 = time.time()
    segments, info = model.transcribe(f, beam_size=1)
    text = " ".join(s.text for s in segments)
    dt = time.time() - t0
    print(f"{f}: {dt:.2f}s | {info.language} | {text!r}")
