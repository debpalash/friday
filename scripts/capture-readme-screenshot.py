#!/usr/bin/env python3
"""Capture the real Friday interface with a clearly synthetic demo session."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "frontend" / "index.html"
OUTPUT = ROOT / "assets" / "friday-interface.png"


def interface_html() -> str:
    html = SOURCE.read_text(encoding="utf-8")
    if not html.lstrip().lower().startswith("<!doctype html>"):
        raise RuntimeError("frontend/index.html is not an HTML document")
    return html


def main() -> int:
    html = interface_html()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser_path = (
            os.environ.get("FRIDAY_SCREENSHOT_BROWSER")
            or shutil.which("chromium")
            or shutil.which("google-chrome")
            or playwright.chromium.executable_path
        )
        browser = playwright.chromium.launch(
            executable_path=browser_path)
        page = browser.new_page(
            viewport={"width": 1440, "height": 960},
            device_scale_factor=1,
            color_scheme="dark",
        )
        page.set_content(html, wait_until="domcontentloaded")
        page.evaluate(
            """
            document.body.className = 'started idle';
            document.getElementById('log').replaceChildren();
            document.getElementById('status').textContent = 'Text';
            document.getElementById('activity').textContent = '';
            document.getElementById('modechip').textContent = 'Connected';
            document.getElementById('dot').classList.add('on');
            document.getElementById('sendbtn').disabled = false;
            add(
              'you',
              'Build a release rehearsal for Friday.',
              'you'
            );
            add(
              'fri',
              `## Release rehearsal

Use a clean, supported Linux workstation.

1. Install from the tagged asset and verify **SHA256SUMS**.
2. Complete one text turn and one voice turn.
3. Reject a state-changing action, then approve one bounded action.
4. Stop Friday and confirm the model leaves GPU memory.
5. Exercise update rollback, uninstall, and reinstall with preserved state.

### Ship only if

| Boundary | Required evidence |
| --- | --- |
| Installer | Exact tag and source digest |
| Actions | Approval and verification receipts |
| Privacy | No private data in logs or release files |
| Recovery | Previous release restores cleanly |`,
              'friday'
            );
            document.getElementById('log').scrollTop = 0;
            """
        )
        page.screenshot(path=str(OUTPUT), animations="disabled")
        browser.close()
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
