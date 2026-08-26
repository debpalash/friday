"""Friday — realtime voice assistant.

mic -> Silero VAD -> Parakeet TDT ASR -> Qwen3.8-27B (vLLM :18021, streamed)
     -> sentence-chunked OmniVoice TTS -> speakers, with barge-in.

Run:  venv/bin/python friday.py
Stop: Ctrl+C
"""
import asyncio
import os
import re
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from friday_core import load_asr
from friday_core.local_http import normalize_loopback_model_base_url

SAMPLE_RATE = 16000              # vad/asr rate
MIC_RATE = 48000                 # hw mic rate (PortAudio pipewire bridge is broken)
BLOCK = 1536                     # mic samples per block @48k -> 512 @16k (32 ms)


def find_mic() -> int | None:
    import os
    if os.environ.get("MIC_DEVICE"):
        return int(os.environ["MIC_DEVICE"])
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and "pipewire" not in d["name"].lower():
            try:
                s = sd.InputStream(device=i, samplerate=MIC_RATE, channels=1,
                                   dtype="float32", blocksize=BLOCK)
                s.start()
                s.stop()
                s.close()
                return i
            except Exception:
                continue
    return None
TTS_RATE = 24000
SPEECH_THRESHOLD = 0.5
BARGE_IN_BLOCKS = 6              # ~190 ms of speech interrupts playback
SILENCE_END_MS = 700
PRE_ROLL_S = 0.5
MAX_UTTERANCE_S = 30
HISTORY_TURNS = 12

SYSTEM_PROMPT = (
    "You are Friday, a personal AI assistant modeled after a witty, warm, capable "
    "voice assistant. You are spoken aloud, so reply in short conversational "
    "sentences without markdown, lists, or emoji. Keep answers brief unless asked "
    "for detail. You have a dry sense of humor."
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
LOCAL_BASE_URL = normalize_loopback_model_base_url(os.environ.get(
    "FRIDAY_LOCAL_BASE_URL", "http://127.0.0.1:18021/v1"))
LOCAL_MODEL = os.environ.get("FRIDAY_LOCAL_MODEL", "qwen3.8-27b")
TTS_DEVICE = os.environ.get("FRIDAY_TTS_DEVICE", "cuda")
LLM_REPO = Path(os.environ.get(
    "FRIDAY_LLM_REPO",
    os.environ.get("FRIDAY_QWEN_ROOT", "/home/pal/github/qwen38-27b-uncensored"),
)).expanduser().resolve()
KEY = os.environ.get("FRIDAY_LOCAL_API_KEY", "").strip()
if not KEY:
    key_file = Path(os.environ.get(
        "FRIDAY_LOCAL_API_KEY_FILE", str(LLM_REPO / "api_key.txt"))).expanduser()
    try:
        KEY = key_file.read_text().strip()
    except OSError:
        KEY = "friday-local"


def log(state, msg=""):
    end = "" if state == "listening" else "\n"
    print(f"\r\033[K[{state}] {msg}", flush=True, end=end)


class Friday:
    def __init__(self):
        from silero_vad import load_silero_vad
        from omnivoice.models.omnivoice import OmniVoice

        self.vad = load_silero_vad()
        print("loading asr...", flush=True)
        self.asr = load_asr(
            Path(__file__).parent / "models" /
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
        print(f"asr backend: {self.asr.name}", flush=True)
        print("loading tts...", flush=True)
        t0 = time.time()
        self.tts = OmniVoice.from_pretrained(
            "khaledmezdour/omnivoice-singing", device_map=TTS_DEVICE).eval()
        print(f"models loaded in {time.time()-t0:.1f}s")
        self.llm = AsyncOpenAI(
            base_url=LOCAL_BASE_URL, api_key=KEY,
            http_client=DefaultAsyncHttpxClient(
                trust_env=False, follow_redirects=False))
        self.instruct = "female, young adult, moderate pitch"

        self.mode = "listen"          # 'listen' | 'speak'
        self.interrupt = asyncio.Event()
        self._utterance_done = None   # asyncio.Event set when utterance complete
        self._speech_buf: list[np.ndarray] = []
        self._spoke = False
        self._silent_ms = 0.0
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ---------- audio ----------
    def speech_prob(self, mono: np.ndarray) -> float:
        x = mono.reshape(-1, 3).mean(axis=1)          # 48k -> 16k
        t = torch.from_numpy(x).float()
        with torch.no_grad():
            return self.vad(t, SAMPLE_RATE).item()

    def transcribe(self, pcm: np.ndarray) -> str:
        x = pcm.reshape(-1, 3).mean(axis=1)           # 48k -> 16k
        return self.asr.transcribe_samples(x, SAMPLE_RATE)

    def synth(self, text: str) -> np.ndarray:
        with torch.no_grad():
            out = self.tts.generate(text=text, language="English",
                                    instruct=self.instruct)
        a = out[0]
        return a.cpu().numpy() if hasattr(a, "cpu") else a

    # ---------- llm ----------
    async def respond(self, user_text: str, speak_q: asyncio.Queue):
        self.history.append({"role": "user", "content": user_text})
        msgs = [self.history[0]] + self.history[1:][-2 * HISTORY_TURNS:]
        stream = await self.llm.chat.completions.create(
            model=LOCAL_MODEL,
            messages=msgs,
            temperature=0.7, top_p=0.8, max_tokens=1024,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        buf, full = "", ""
        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                buf += delta
                full += delta
                parts = SENTENCE_SPLIT.split(buf)
                while len(parts) > 1:
                    s = parts.pop(0).strip()
                    if s:
                        await speak_q.put(s)
                    buf = parts[0]
            if buf.strip():
                await speak_q.put(buf.strip())
        finally:
            self.history.append({"role": "assistant", "content": full})
            await speak_q.put(None)

    # ---------- mic consumer ----------
    async def consume(self, block: np.ndarray):
        p = await asyncio.get_event_loop().run_in_executor(
            None, self.speech_prob, block[:, 0])
        mono = block[:, 0]

        if self.mode == "speak":
            if p > SPEECH_THRESHOLD:
                self._barge += 1
                if self._barge >= BARGE_IN_BLOCKS:
                    self.interrupt.set()
            else:
                self._barge = 0
            return

        # listen mode
        if p > SPEECH_THRESHOLD:
            if not self._spoke:
                self._spoke = True
                log("listening", "<speaking>")
            self._silent_ms = 0.0
            self._speech_buf.append(mono)
        elif self._spoke:
            self._silent_ms += BLOCK / MIC_RATE * 1000
            self._speech_buf.append(mono)
            if self._silent_ms >= SILENCE_END_MS:
                self._utterance_done.set()
        elif self._speech_buf:
            self._speech_buf.append(mono)
            total = sum(len(x) for x in self._speech_buf)
            if total > MIC_RATE * PRE_ROLL_S * 2:
                keep = int(MIC_RATE * PRE_ROLL_S / BLOCK)
                self._speech_buf = self._speech_buf[-keep:]
        if self._spoke and sum(len(x) for x in self._speech_buf) > MIC_RATE * MAX_UTTERANCE_S:
            self._utterance_done.set()

    async def listen_utterance(self) -> np.ndarray | None:
        self._speech_buf, self._spoke, self._silent_ms = [], False, 0.0
        self._utterance_done = asyncio.Event()
        log("listening", "...")
        await self._utterance_done.wait()
        if not self._spoke:
            return None
        pcm = np.concatenate(self._speech_buf)
        trim = int(self._silent_ms / 1000 * SAMPLE_RATE)
        return pcm[: max(0, len(pcm) - trim)]

    # ---------- speak ----------
    async def think_and_speak(self, text: str):
        speak_q: asyncio.Queue = asyncio.Queue()
        llm_task = asyncio.create_task(self.respond(text, speak_q))
        loop = asyncio.get_event_loop()
        self.interrupt.clear()
        self._barge = 0
        self.mode = "speak"
        player = sd.OutputStream(samplerate=TTS_RATE, channels=1, dtype="float32")
        player.start()
        interrupted = False
        try:
            while True:
                sentence = await speak_q.get()
                if sentence is None:
                    break
                log("friday", sentence)
                audio = await loop.run_in_executor(None, self.synth, sentence)
                if self.interrupt.is_set():
                    interrupted = True
                    break
                step = TTS_RATE // 5                     # 200 ms chunks
                for i in range(0, len(audio), step):
                    if self.interrupt.is_set():
                        interrupted = True
                        break
                    player.write(audio[i:i + step].astype(np.float32))
                if interrupted:
                    break
        except asyncio.CancelledError:
            interrupted = True
        finally:
            if interrupted:
                player.abort()
                llm_task.cancel()
                log("interrupted", "(barge-in)")
            else:
                player.stop()
            player.close()
            self.mode = "listen"

    # ---------- main ----------
    async def run(self):
        aio_q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        mic = find_mic()
        if mic is None:
            print("No microphone found. Plug one in (USB recommended) and "
                  "rerun, or set MIC_DEVICE=<index>.")
            return
        print(f"using mic device {mic}: {sd.query_devices(mic)['name']}")

        def cb(indata, frames, t, status):
            loop.call_soon_threadsafe(aio_q.put_nowait, indata.copy())

        with sd.InputStream(device=mic, samplerate=MIC_RATE, channels=1,
                            dtype="float32", blocksize=BLOCK, callback=cb):
            print("Friday ready. Speak. Ctrl+C to quit.\n")
            consumer = asyncio.create_task(self._consume_loop(aio_q))
            try:
                while True:
                    utt = await self.listen_utterance()
                    if utt is None or len(utt) < MIC_RATE * 0.3:
                        continue
                    text = await loop.run_in_executor(None, self.transcribe, utt)
                    if len(text) < 2:
                        continue
                    log("you", text)
                    await self.think_and_speak(text)
            finally:
                consumer.cancel()

    async def _consume_loop(self, aio_q: asyncio.Queue):
        while True:
            block = await aio_q.get()
            await self.consume(block)


async def _main():
    friday = Friday()
    try:
        await friday.run()
    finally:
        await friday.llm.close()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nbye")
