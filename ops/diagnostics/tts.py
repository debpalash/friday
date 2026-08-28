"""Manual OmniVoice synthesis and real-time-factor check."""

import time

import torch
from omnivoice.models.omnivoice import OmniVoice

t0 = time.time()
model = OmniVoice.from_pretrained("khaledmezdour/omnivoice-singing", device_map="cuda").eval()
print(f"load: {time.time()-t0:.1f}s")

text = "Systems online. All checks passed. I am listening."
for label, kwargs in [
    ("plain-designed", dict(instruct="female, young adult, moderate pitch")),
    ("happy", dict(instruct="female, young adult, high pitch", text_prefix="[happy] ")),
]:
    t0 = time.time()
    audios = model.generate(
        text=kwargs.get("text_prefix", "") + text,
        language="English",
        guidance_scale=3.0 if kwargs.get("text_prefix") else 2.0,
        instruct=kwargs["instruct"],
    )
    dt = time.time() - t0
    audio = audios[0]
    if hasattr(audio, "cpu"): audio = audio.cpu().numpy()
    dur = len(audio) / model.sampling_rate
    print(f"{label}: {dt:.2f}s synth for {dur:.2f}s audio (RTF {dt/dur:.2f})")
    import soundfile as sf
    sf.write(f"/tmp/opencode/tts_{label}.wav", audio, model.sampling_rate)

print("VRAM allocated:", torch.cuda.max_memory_allocated() / 2**30, "GiB")
