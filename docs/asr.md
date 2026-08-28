# Friday ASR

Friday uses NVIDIA Parakeet TDT 0.6B v3 through Sherpa-ONNX on CPU. Live
microphone audio reaches the recognizer as mono float32 PCM at 16 kHz. The
active backend is returned by `/api/status` and included in every ASR timing
line in the diagnostic panel.

Install the complete hash-locked Friday environment and exact ASR assets:

```bash
uv pip sync --python venv/bin/python --require-hashes requirements/runtime.lock
venv/bin/python ops/install_asr_model.py
```

The asset installer checks the release archive before extraction, rejects
traversal, links, and device entries, then checks every required output file.
The expected model directory is
`models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/` and must contain
`encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, and `tokens.txt`.

Environment controls:

- `FRIDAY_ASR=parakeet` selects the default backend.
- `FRIDAY_ASR_THREADS=4` sets CPU inference threads.
- `FRIDAY_ASR_FALLBACK=0` makes startup fail instead of falling back when the
  Parakeet assets cannot load.
- `FRIDAY_ASR=whisper` explicitly selects the old Faster-Whisper small backend.
