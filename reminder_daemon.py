#!/usr/bin/env python3
"""Standalone restart-safe reminder delivery process."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

from friday_core import GraphStore, ReminderService, ReminderWorker


REPO = Path(__file__).resolve().parent
STATE = Path(
    os.environ.get("FRIDAY_STATE_DIR", str(REPO / "state"))
).expanduser().resolve()


async def deliver(receipt: dict) -> None:
    await asyncio.to_thread(
        subprocess.run,
        ["notify-send", "Friday reminder", str(receipt.get("text") or "Reminder")],
        capture_output=True, timeout=10, check=True)


async def main() -> None:
    graph = GraphStore(STATE / "friday.db")
    worker = ReminderWorker(ReminderService(graph), deliver)
    await worker.start()
    try:
        await asyncio.Event().wait()
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
