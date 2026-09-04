#!/usr/bin/env python3
"""Actionable preflight and installed-runtime diagnostics for Friday."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.speech import pinned_omnivoice_model_path


STATE = Path(os.environ.get("FRIDAY_STATE_DIR", str(REPO / "state"))).expanduser()
DEFAULT_INSTALL_ROOT = Path(
    os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
) / "friday"
QWEN = Path(
    os.environ.get("FRIDAY_LLM_REPO", str(DEFAULT_INSTALL_ROOT / "runtime" / "qwen"))
).expanduser()


def _check(name: str, passed: bool, detail: str, *, required: bool = True) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "required": required, "detail": detail}


def _command(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    return _check(f"command:{name}", path is not None, path or "not found")


def _gpu() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return _check("nvidia_gpu", False, "nvidia-smi is not installed")
    try:
        output = subprocess.run(
            [executable, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=10, check=True,
        ).stdout.strip()
        rows = [row.strip() for row in output.splitlines() if row.strip()]
        capacities = [int(row.rsplit(",", 1)[1].strip()) for row in rows]
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return _check("nvidia_gpu", False, f"GPU probe failed: {exc}")
    enough = bool(capacities) and max(capacities) >= 22 * 1024
    return _check(
        "nvidia_gpu", enough,
        "; ".join(rows) + ("" if enough else "; Friday's default 27B tier needs at least 22 GiB"),
    )


def _systemd() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _check("systemd_user", False, str(exc))
    return _check(
        "systemd_user", result.returncode == 0,
        "available" if result.returncode == 0 else result.stderr.strip()[-300:],
    )


def _health(port: int) -> dict[str, Any]:
    ca = STATE / "tls" / "friday-local-ca.crt"
    if not ca.is_file():
        return _check("friday_health", False, "local TLS identity has not been created")
    try:
        context = ssl.create_default_context(cafile=str(ca))
        with urllib.request.urlopen(
            f"https://127.0.0.1:{port}/healthz", context=context, timeout=5
        ) as response:
            body = response.read(128)
        passed = response.status == 200
        detail = body.decode("utf-8", "replace")
    except Exception as exc:
        passed, detail = False, str(exc)
    return _check("friday_health", passed, detail)


def _omnivoice() -> dict[str, Any]:
    try:
        model = pinned_omnivoice_model_path(REPO)
    except RuntimeError as exc:
        return _check("omnivoice_model", False, str(exc))
    return _check("omnivoice_model", True, str(model))


def run(*, expect_running: bool = False) -> dict[str, Any]:
    results = [
        _check(
            "platform",
            sys.platform.startswith("linux") and platform.machine() in {"x86_64", "amd64"},
            f"{sys.platform}/{platform.machine()}; supported: Linux x86_64",
        ),
        _systemd(),
        *(_command(name) for name in ("curl", "openssl", "bwrap", "systemctl")),
        _gpu(),
        _check("app_python", (REPO / "venv" / "bin" / "python").is_file(), str(REPO / "venv")),
        _check("asr_model", (REPO / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8" / "encoder.int8.onnx").is_file(), "pinned Parakeet int8 assets"),
        _check("piper_voice", any((REPO / "models").glob("piper-en_US-kristin-medium-*/en_US-kristin-medium.onnx")), "pinned Kristin voice"),
        _omnivoice(),
        _check("embedding_model", any((REPO / "models").glob("multilingual-e5-small-*/onnx/model.onnx")), "pinned multilingual embedding model (ONNX)", required=False),
        _check("vad_model", any((REPO / "models").glob("silero-vad-*/silero_vad.onnx")), "pinned Silero VAD model"),
        _check("qwen_root", (QWEN / "single-user" / "start_qwen.sh").is_file(), str(QWEN)),
        _check("qwen_runtime", (QWEN / "venv" / "bin" / "vllm").is_file(), "pinned vLLM runtime"),
        _check("qwen_model", (QWEN / "models" / "Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound" / "config.json").is_file(), "Friday's default local checkpoint"),
    ]
    if expect_running:
        try:
            port = int(os.environ.get("FRIDAY_PORT", "8500"))
        except ValueError:
            port = -1
        results.append(_health(port))
    failures = [item for item in results if item["required"] and not item["passed"]]
    return {
        "status": "ready" if not failures else "blocked",
        "release": REPO.name,
        "checks": results,
        "failures": len(failures),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--expect-running", action="store_true")
    args = parser.parse_args()
    report = run(expect_running=args.expect_running)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Friday doctor")
        for item in report["checks"]:
            marker = "OK" if item["passed"] else ("WARN" if not item["required"] else "FAIL")
            print(f"  {marker:4} {item['name']}: {item['detail']}")
        print(f"\n{report['status']}")
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
