#!/usr/bin/env python3
"""Actionable preflight and installed-runtime diagnostics for Friday."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_host.host import current_host
from friday_host.paths import default_install_root, venv_python
from friday_host.service import backend_for

HOST = current_host()
STATE = Path(os.environ.get("FRIDAY_STATE_DIR", str(REPO / "state"))).expanduser()
DEFAULT_INSTALL_ROOT = default_install_root()
QWEN = Path(
    os.environ.get("FRIDAY_LLM_REPO", str(DEFAULT_INSTALL_ROOT / "runtime" / "qwen"))
).expanduser()
RUNTIME_ROOT = Path(os.environ.get("FRIDAY_RUNTIME_ROOT", str(QWEN.parent))).expanduser()
ENGINE = os.environ.get("FRIDAY_LLM_ENGINE", "vllm" if HOST.is_linux else "auto").strip()


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


def _service_manager() -> dict[str, Any]:
    if HOST.is_linux:
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
    if HOST.is_macos:
        try:
            backend = backend_for(HOST)
            registered = backend.is_enabled()
        except (OSError, NotImplementedError) as exc:
            return _check("launchd_agent", False, str(exc))
        return _check("launchd_agent", registered,
                      "login agent registered" if registered
                      else "login agent is not registered; run the installer",
                      required=False)
    return _check("service_manager", False, "no service backend for this platform")


def _portable_engine() -> list[dict[str, Any]]:
    """Engine checks for the llama-server and MLX runtimes."""
    checks = []
    if ENGINE == "mlx_lm" or (ENGINE == "auto" and HOST.is_macos and HOST.arch == "aarch64"):
        checks.append(_check(
            "mlx_runtime", (RUNTIME_ROOT / "mlx" / "FRIDAY_ENGINE_PIN").is_file(),
            str(RUNTIME_ROOT / "mlx")))
    else:
        pins = list((RUNTIME_ROOT / "llama-server").glob("*/FRIDAY_ENGINE_PIN"))
        checks.append(_check("llama_server", bool(pins),
                             str(pins[0].parent) if pins else "no pinned llama-server build"))
    model_pins = list((REPO / "models").glob("qwen3-*/FRIDAY_MODEL_PIN"))
    checks.append(_check("local_model", bool(model_pins),
                         str(model_pins[0].parent) if model_pins
                         else "no pinned Qwen3 checkpoint under models/"))
    key_file = Path(os.environ.get("FRIDAY_LOCAL_API_KEY_FILE", str(STATE / "local-api-key")))
    checks.append(_check("local_api_key", key_file.is_file() or not key_file.parent.exists(),
                         "created on first start" if not key_file.is_file() else str(key_file),
                         required=False))
    return checks


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
    from friday_core.speech import pinned_omnivoice_model_path  # noqa: PLC0415

    try:
        model = pinned_omnivoice_model_path(REPO)
    except RuntimeError as exc:
        return _check("omnivoice_model", False, str(exc))
    return _check("omnivoice_model", True, str(model))


def run(*, expect_running: bool = False) -> dict[str, Any]:
    supported = (HOST.is_linux and HOST.arch == "x86_64") or (HOST.is_macos and HOST.arch == "aarch64")
    linux_vllm = HOST.is_linux and ENGINE in {"vllm", "auto"}
    results = [
        _check(
            "platform", supported,
            f"{HOST.os}/{HOST.arch}; supported: Linux x86_64, macOS Apple Silicon",
        ),
        _service_manager(),
        *(_command(name) for name in
          (("curl", "openssl", "bwrap", "systemctl") if HOST.is_linux else ("curl", "launchctl"))),
        *([_gpu()] if linux_vllm else []),
        _check("app_python", venv_python(REPO, HOST).is_file(), str(REPO / "venv")),
        _check("asr_model", (REPO / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8" / "encoder.int8.onnx").is_file(), "pinned Parakeet int8 assets"),
        _check("piper_voice", any((REPO / "models").glob("piper-en_US-kristin-medium-*/en_US-kristin-medium.onnx")), "pinned Kristin voice"),
        *([_omnivoice()] if linux_vllm else []),
        _check("embedding_model", any((REPO / "models").glob("multilingual-e5-small-*/onnx/model.onnx")), "pinned multilingual embedding model (ONNX)", required=False),
        _check("vad_model", any((REPO / "models").glob("silero-vad-*/silero_vad.onnx")), "pinned Silero VAD model"),
    ]
    if linux_vllm:
        results += [
            _check("qwen_root", (QWEN / "single-user" / "start_qwen.sh").is_file(), str(QWEN)),
            _check("qwen_runtime", (QWEN / "venv" / "bin" / "vllm").is_file(), "pinned vLLM runtime"),
            _check("qwen_model", (QWEN / "models" / "Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound" / "config.json").is_file(), "Friday's default local checkpoint"),
        ]
    else:
        results += _portable_engine()
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
