#!/usr/bin/env python3
"""Render the README guided tour from the real Friday interface.

Every turn is synthetic and scripted here. Nothing is read from a Friday
database, a model, a microphone, or the maintainer's machine. The frames show
the shipped frontend rendering the same message types the server emits:
user turns, progress, approval cards, task cards, news cards, and Markdown
replies.

Usage:
    venv/bin/python scripts/capture-readme-demo.py [--keep-frames]

Requires Playwright with Chromium and an ffmpeg binary on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "frontend" / "index.html"
OUTPUT = ROOT / "assets" / "friday-tour.gif"
VIEWPORT = {"width": 1440, "height": 960}
GIF_WIDTH = 1080
GIF_FPS = 12


class Recorder:
    """Collect screenshots with a display duration for each frame."""

    def __init__(self, page: Page, directory: Path) -> None:
        self.page = page
        self.directory = directory
        self.frames: list[tuple[Path, float]] = []

    def js(self, script: str, *args: object) -> None:
        self.page.evaluate(script, list(args))

    def hold(self, seconds: float) -> None:
        path = self.directory / f"frame-{len(self.frames):04d}.png"
        self.page.screenshot(path=str(path), animations="disabled")
        self.frames.append((path, seconds))

    def type_request(self, text: str) -> None:
        step = 3
        for end in range(step, len(text) + step, step):
            self.js("([v]) => { $('textinput').value = v; }", text[:end])
            self.hold(0.055)
        self.hold(0.5)
        self.js(
            """([v]) => {
              $('textinput').value = '';
              currentTaskId = null;
              add('you', v, 'you');
              setState('thinking'); setStatus('Thinking');
            }""",
            text,
        )
        self.hold(0.55)

    def activity(self, label: str, seconds: float = 0.6) -> None:
        self.js("([t]) => { $('activity').textContent = t; }", label)
        self.hold(seconds)

    def reply(self, markdown: str, seconds: float, task_id: str | None = None) -> None:
        self.js(
            """([md, taskId]) => {
              $('activity').textContent = '';
              const card = add('fri', md, 'friday');
              quickActions(card, taskId);
              setState('idle'); setStatus('Text');
            }""",
            markdown,
            task_id,
        )
        self.hold(seconds)


def interface_html() -> str:
    html = SOURCE.read_text(encoding="utf-8")
    if not html.lstrip().lower().startswith("<!doctype html>"):
        raise RuntimeError("frontend/index.html is not an HTML document")
    return html


def reset(page: Page) -> None:
    page.evaluate(
        """() => {
          document.body.className = 'started idle';
          $('log').replaceChildren();
          $('status').textContent = 'Text';
          $('activity').textContent = '';
          $('modechip').textContent = 'Connected';
          $('dot').classList.add('on');
          $('sendbtn').disabled = false;
        }"""
    )


def scene_reminder(rec: Recorder) -> None:
    rec.type_request("Remind me to call the dentist tomorrow at 9.")
    rec.activity("Saving reminder")
    rec.reply(
        "Reminder saved for **tomorrow at 09:00**: call the dentist.\n\n"
        "I will notify you here and on the desktop. Say *cancel the dentist "
        "reminder* to remove it.",
        2.0,
        "task_demo_reminder",
    )


def scene_news(rec: Recorder) -> None:
    rec.type_request("What's in the news this morning?")
    rec.activity("Fetching verified headlines")
    rec.js(
        """() => showNews({
          region: 'Morning',
          headlines: [
            {title: 'City council approves weekend transit expansion',
             source: 'Synthetic Wire', url: 'https://example.org/transit'},
            {title: 'Regional library extends evening hours through winter',
             source: 'Example Gazette', url: 'https://example.org/library'},
            {title: 'Open-source speech models close the gap on latency',
             source: 'Synthetic Tech', url: 'https://example.org/speech'},
            {title: 'Local farmers market moves indoors for the season',
             source: 'Example Gazette', url: 'https://example.org/market'}
          ]
        })"""
    )
    rec.hold(0.7)
    rec.reply(
        "Four headlines from the feed. Open any of them to read the story. "
        "I summarize a page only when you ask me to read it.",
        2.2,
    )


def scene_theme(rec: Recorder) -> None:
    rec.type_request("Switch the theme to Tokyo Night and turn on night light.")
    rec.activity("Checking Omarchy status")
    rec.js(
        """() => {
          currentTaskId = 'task_demo_theme';
          const card = add('taskcard approval',
            'Changing the desktop theme and night light alters your session. '
            + 'Friday will apply both through Omarchy and verify the result.');
          const label = document.createElement('div');
          label.className = 'tasklabel';
          label.textContent = 'Your approval is required';
          card.prepend(label);
          const preview = document.createElement('pre');
          preview.textContent = JSON.stringify({
            tool: 'machine_omarchy_set_theme',
            theme: 'tokyo-night',
            then: {tool: 'machine_omarchy_set_nightlight', enabled: true}
          }, null, 2);
          preview.style.whiteSpace = 'pre-wrap';
          preview.style.color = 'var(--dim)';
          card.appendChild(preview);
          const row = document.createElement('div');
          row.className = 'quickrow';
          for (const [text, kind] of [['approve', 'approve'], ['deny', 'deny']]) {
            const b = document.createElement('button');
            b.className = 'quick ' + kind; b.textContent = text; row.appendChild(b);
          }
          card.appendChild(row);
          window.demoApprovalRow = row;
          $('activity').textContent = 'Waiting for your approval';
        }"""
    )
    rec.hold(1.8)
    rec.js(
        """() => {
          for (const b of demoApprovalRow.querySelectorAll('button')) b.disabled = true;
          demoApprovalRow.querySelector('.approve').textContent = 'recorded';
          $('activity').textContent = 'Approval recorded';
        }"""
    )
    rec.hold(0.7)
    rec.js(
        """() => showProgress({seq: 1, task_id: 'task_demo_theme',
          state: 'running', label: 'Applying Omarchy theme',
          detail: 'tokyo-night'})"""
    )
    rec.hold(0.8)
    rec.js(
        """() => showProgress({seq: 2, task_id: 'task_demo_theme',
          state: 'completed', label: 'Theme and night light verified',
          detail: 'Omarchy status receipt'})"""
    )
    rec.hold(0.7)
    rec.reply(
        "Done. Omarchy reports **Tokyo Night** as the active theme and night "
        "light is on.\n\nBoth changes were verified by reading the status back "
        "after the change, not assumed from the command exit code.",
        2.4,
        "task_demo_theme",
    )


def scene_document(rec: Recorder) -> None:
    rec.type_request(
        "Read ~/Documents/lease.pdf and list the dates I need to remember.")
    rec.activity("Reading document: 12 pages", 0.9)
    rec.reply(
        "### Dates in lease.pdf\n\n"
        "| Date | What | Where |\n"
        "| --- | --- | --- |\n"
        "| 1 Oct 2026 | Lease begins | Section 1 |\n"
        "| 1 Aug 2027 | Last day to give notice | Section 9 |\n"
        "| 30 Sep 2027 | Lease ends | Section 1 |\n"
        "| 15th of each month | Rent due | Section 4 |\n\n"
        "Want reminders for the notice deadline and the monthly rent date?",
        2.8,
        "task_demo_document",
    )


def scene_voice(rec: Recorder) -> None:
    rec.js(
        """() => {
          setState('listening'); setStatus('Friday, …');
          $('modechip').textContent = 'Listening';
          $('activity').textContent = 'Voice mode: say "Friday, …"';
        }"""
    )
    rec.hold(1.2)
    rec.js(
        """() => {
          setState('hearing'); setStatus('Listening');
          $('activity').textContent = 'Hearing an addressed command';
        }"""
    )
    rec.hold(0.9)
    rec.js(
        """() => {
          currentTaskId = null;
          const card = add('you', "Friday, what's using my GPU right now?", 'you');
          const row = document.createElement('div'); row.className = 'quickrow';
          const b = document.createElement('button'); b.className = 'quick';
          b.textContent = 'edit transcript'; row.appendChild(b); card.appendChild(row);
          setState('thinking'); setStatus('Thinking');
          $('activity').textContent = 'Inspecting processes';
        }"""
    )
    rec.hold(0.9)
    rec.js(
        """() => {
          $('activity').textContent = '';
          const card = add('fri',
            "Three processes hold GPU memory:\\n\\n"
            + "| Process | VRAM | Note |\\n| --- | --- | --- |\\n"
            + "| friday-qwen (vLLM) | 21.4 GiB | The local model |\\n"
            + "| firefox | 0.6 GiB | Hardware video decode |\\n"
            + "| Hyprland | 0.2 GiB | Compositor |\\n\\n"
            + "Say *stop Friday* to unload the model when you need the card.",
            'friday');
          quickActions(card, 'task_demo_gpu');
          setState('speaking'); setStatus('Speaking');
          $('modechip').textContent = 'Speaking';
        }"""
    )
    rec.hold(2.4)
    rec.js(
        """() => {
          setState('listening'); setStatus('Friday, …');
          $('modechip').textContent = 'Listening';
        }"""
    )
    rec.hold(1.6)


def encode(frames: list[tuple[Path, float]], output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode the tour")
    listing = frames[0][0].parent / "frames.txt"
    lines = []
    for path, seconds in frames:
        lines.append(f"file '{path.name}'")
        lines.append(f"duration {seconds:.3f}")
    lines.append(f"file '{frames[-1][0].name}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    filters = (
        f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos,"
        "split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(listing), "-vf", filters, "-loop", "0", str(output),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep-frames", action="store_true",
                        help="print the frame directory instead of deleting it")
    args = parser.parse_args()

    html = interface_html()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="friday-tour-"))
    try:
        with sync_playwright() as playwright:
            browser_path = (
                os.environ.get("FRIDAY_SCREENSHOT_BROWSER")
                or shutil.which("chromium")
                or shutil.which("google-chrome")
                or playwright.chromium.executable_path
            )
            browser = playwright.chromium.launch(executable_path=browser_path)
            page = browser.new_page(
                viewport=VIEWPORT, device_scale_factor=1, color_scheme="dark")
            page.set_content(html, wait_until="domcontentloaded")
            reset(page)
            rec = Recorder(page, workdir)
            rec.hold(0.9)
            scene_reminder(rec)
            scene_news(rec)
            scene_theme(rec)
            scene_document(rec)
            scene_voice(rec)
            browser.close()
        encode(rec.frames, OUTPUT)
        total = sum(seconds for _, seconds in rec.frames)
        print(json.dumps({
            "output": str(OUTPUT.relative_to(ROOT)),
            "frames": len(rec.frames),
            "seconds": round(total, 1),
            "bytes": OUTPUT.stat().st_size,
        }))
    finally:
        if args.keep_frames:
            print(f"frames kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
