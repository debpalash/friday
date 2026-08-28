#!/usr/bin/env python3
"""Atomically install Friday's exact pinned OmniVoice checkpoint."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.speech import (
    OMNIVOICE_ASSETS,
    OMNIVOICE_MODEL_DIRECTORY,
    OMNIVOICE_MODEL_ID,
    OMNIVOICE_MODEL_REVISION,
    pinned_omnivoice_model_path,
)


TARGET = REPO / "models" / OMNIVOICE_MODEL_DIRECTORY


def main() -> int:
    if TARGET.exists() or TARGET.is_symlink():
        pinned_omnivoice_model_path(REPO)
        print(f"verified existing OmniVoice model: {TARGET}")
        return 0

    TARGET.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{TARGET.name}-install-", dir=TARGET.parent))
    try:
        snapshot_download(
            repo_id=OMNIVOICE_MODEL_ID,
            revision=OMNIVOICE_MODEL_REVISION,
            local_dir=staging,
            allow_patterns=sorted(OMNIVOICE_ASSETS),
        )
        metadata = staging / ".cache"
        if metadata.exists():
            shutil.rmtree(metadata)
        for path in staging.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o644)
        manifest = staging / "FRIDAY_MODEL_PIN"
        manifest.write_text(
            f"model={OMNIVOICE_MODEL_ID}\n"
            f"revision={OMNIVOICE_MODEL_REVISION}\n",
            encoding="utf-8",
        )
        os.chmod(manifest, 0o644)
        os.replace(staging, TARGET)
        pinned_omnivoice_model_path(REPO)
        directory = os.open(TARGET.parent, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"installed OmniVoice model: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
