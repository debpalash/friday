# Third-party software and model provenance

Friday's source repository does not contain model weights. The installer
downloads exact upstream revisions into the user's private data directories.
Those components keep their own licenses and terms. Friday does not grant rights
to them.

This file covers the large or operationally important components. It is not a
replacement for the metadata shipped by every Python package in
`requirements/runtime.lock`.

## Language model runtime

| Component | Pin | Upstream license |
|---|---|---|
| [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) | `f238b9320a2ef1a48cfe47c4c2db3b0ef89d93b1` | Apache-2.0 |
| [ababaka/Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound](https://huggingface.co/ababaka/Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound) | `92600100b5c2b97bf1fd1745479c1e0f8007e008` | Apache-2.0 |
| [vLLM](https://github.com/vllm-project/vllm) | `0.27.1`, then patched by the pinned runtime | Apache-2.0 |

The selected Qwen checkpoint is an independently modified, abliterated, and
quantized derivative. Its outputs can be inaccurate, unsafe, or offensive.

## Speech

| Component | Pin | Upstream license or provenance |
|---|---|---|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | `1.13.3` | Apache-2.0 |
| [NVIDIA Parakeet-TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | INT8 ONNX conversion from the sherpa-onnx `asr-models` release, exact files SHA-256 pinned | CC BY 4.0 |
| [OmniVoice](https://github.com/k2-fsa/OmniVoice) | Python package `0.2.1` | Apache-2.0 |
| [khaledmezdour/omnivoice-singing](https://huggingface.co/khaledmezdour/omnivoice-singing) | `31927d2ac03a2a7259f4f5ca02329457d89cb353` | Apache-2.0 |
| [piper-tts](https://github.com/OHF-Voice/piper1-gpl) | `1.4.2` | GPL-3.0-or-later |
| [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) | `375a0fe641dea077c2a47b4e9a056d6da521eed3` | Repository metadata: MIT |
| `en_US-kristin-medium` voice | exact ONNX and config files SHA-256 pinned | Model card states the LibriVox recordings are public domain |
| [silero-vad](https://github.com/snakers4/silero-vad) | `6.2.1` | MIT |

Attribution for the ASR model: Parakeet-TDT 0.6B v3 by NVIDIA, exported and
quantized to INT8 ONNX by the sherpa-onnx project. Friday does not modify the
downloaded ONNX files.

The Piper runtime is GPL-3.0-or-later. Any distribution of Friday must be
reviewed for compatibility with that dependency and the license selected for
Friday itself.

Do not create or distribute a reference-based voice without the speaker's
permission and the rights required for the source recording and intended use.
Friday intentionally excludes user voice clips from version control.

## Memory and installer components

| Component | Pin | Upstream license |
|---|---|---|
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | MIT |
| [uv](https://github.com/astral-sh/uv) | `0.12.1`, standalone archive SHA-256 pinned | Apache-2.0 |

Release maintainers must recheck upstream model cards, package metadata, and
license files whenever a pin changes.
