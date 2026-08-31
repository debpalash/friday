#!/usr/bin/env python3
"""Run realistic scenarios through Friday's installed WebSocket server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import ssl
import stat
import sys
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.live_assistant_evals import LiveAssistantEvalRunner
from friday_core.live_runtime import (
    read_live_runtime,
    runtime_environment,
)


def _private_ca_path(state_dir: Path) -> Path:
    path = state_dir / "tls" / "friday-local-ca.crt"
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError("Friday local CA is unavailable") from exc
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077 or not 256 <= metadata.st_size <= 64_000):
        raise RuntimeError("Friday local CA is not a private regular file")
    return path


class InstalledFridayClient:
    def __init__(self, *, state_dir: Path, port: int):
        if not 1 <= port <= 65_535:
            raise ValueError("Friday control-plane port is invalid")
        self.http_url = f"https://127.0.0.1:{port}"
        self.ws_url = (
            f"wss://127.0.0.1:{port}/ws?mode=text&context=ephemeral")
        self.origin = f"https://127.0.0.1:{port}"
        self.ssl_context = ssl.create_default_context(
            cafile=str(_private_ca_path(state_dir)))

    def get_json(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(self.http_url + path)
        with urllib.request.urlopen(
                request, context=self.ssl_context, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"Friday HTTP probe failed with {response.status}")
            body = response.read(1_000_001)
        if len(body) > 1_000_000:
            raise RuntimeError("Friday HTTP response exceeded the evaluator limit")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise RuntimeError("Friday HTTP response is not an object")
        return value

    async def run_case_async(self, case: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "installed Friday runtime is missing the WebSocket client") from exc
        observations = []
        async with websockets.connect(
            self.ws_url,
            ssl=self.ssl_context,
            origin=self.origin,
            subprotocols=["friday.v1"],
            proxy=None,
            open_timeout=15,
            max_size=2_000_000,
        ) as socket:
            # Reconnect status belongs to the session envelope, not the first
            # evaluated turn. Drain the bounded initial snapshot before send.
            for _ in range(128):
                try:
                    await asyncio.wait_for(socket.recv(), timeout=0.05)
                except TimeoutError:
                    break
            for turn in case["turns"]:
                before = await asyncio.to_thread(
                    self.get_json, "/api/progress?latest=true")
                before_cursor = int(before.get("latest") or 0)
                await socket.send(json.dumps({
                    "type": "text", "text": turn["prompt"],
                }, separators=(",", ":")))
                events: list[str] = []
                outputs: list[str] = []
                while True:
                    raw = await asyncio.wait_for(socket.recv(), timeout=120)
                    message = json.loads(raw)
                    if not isinstance(message, dict):
                        raise RuntimeError("Friday WebSocket event is not an object")
                    event_type = message.get("type")
                    if not isinstance(event_type, str):
                        raise RuntimeError("Friday WebSocket event has no type")
                    events.append(event_type)
                    if event_type == "friday":
                        text = message.get("text")
                        if isinstance(text, str) and text.strip():
                            outputs.append(text.strip())
                    if event_type == "done":
                        break
                    if len(events) > 128:
                        raise RuntimeError("Friday emitted too many events for one turn")
                after = await asyncio.to_thread(
                    self.get_json, f"/api/progress?since={before_cursor}")
                after_cursor = int(after.get("latest") or 0)
                observations.append({
                    "output": "\n".join(outputs),
                    "events": events,
                    "progress_cursor_advanced": after_cursor > before_cursor,
                })
        return observations

    def run_case(self, case: dict[str, Any]) -> list[dict[str, Any]]:
        return asyncio.run(self.run_case_async(case))


def _suite_path(name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}\.json", name):
        raise RuntimeError("live assistant suite name is invalid")
    path = (REPO / "evals" / name).resolve()
    if path.parent != (REPO / "evals").resolve():
        raise RuntimeError("live assistant suite must be under evals")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", default="live-assistant-v1.json",
        help="bounded suite filename under evals")
    args = parser.parse_args()
    runtime = read_live_runtime(REPO)
    environment = runtime_environment()
    if str(environment.get("FRIDAY_BIND_HOST", "127.0.0.1")) != "127.0.0.1":
        raise RuntimeError("live assistant evaluation requires loopback-only Friday")
    try:
        port = int(str(environment.get("FRIDAY_PORT", "8500")))
    except ValueError as exc:
        raise RuntimeError("Friday control-plane port is invalid") from exc
    client = InstalledFridayClient(state_dir=runtime.state_dir, port=port)
    health = client.get_json("/healthz")
    if health.get("ready") is not True:
        raise RuntimeError("Friday is not ready for live assistant evaluation")
    result = LiveAssistantEvalRunner(
        client.run_case,
        runtime_fingerprint=runtime.fingerprint,
    ).run(_suite_path(args.suite))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
