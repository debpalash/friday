# Friday ASR

Friday uses NVIDIA Parakeet TDT 0.6B v3 through Sherpa-ONNX on CPU. Live
microphone audio reaches the recognizer as mono float32 PCM at 16 kHz. The
active backend is returned by `/api/status` and included in every ASR timing
line in the diagnostic panel.

Install the runtime into Friday's environment:

```bash
uv pip install --python venv/bin/python -r requirements/asr.txt
```

Download and unpack the int8 model from Sherpa-ONNX's official model release:

```bash
curl --fail --location --output /tmp/friday-parakeet.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
mkdir -p models
tar -xjf /tmp/friday-parakeet.tar.bz2 -C models
```

The expected model directory is
`models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/` and must contain
`encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, and `tokens.txt`.

Environment controls:

- `FRIDAY_ASR=parakeet` selects the default backend.
- `FRIDAY_ASR_THREADS=4` sets CPU inference threads.
- `FRIDAY_ASR_FALLBACK=0` makes startup fail instead of falling back when the
  Parakeet assets cannot load.
- `FRIDAY_ASR=whisper` explicitly selects the old Faster-Whisper small backend.
