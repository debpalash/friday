"""Constrained web research and a dedicated visible Chromium profile."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .graph import utc_now
from .local_http import open_loopback_request
from .public_http import (normalize_public_http_url, request_public_http)


MAX_WEB_BYTES = 2_000_000
USER_AGENT = "FridayPersonalAssistant/1.0 (+local supervised operator)"


def validate_public_url(value: str) -> str:
    return normalize_public_http_url(value)


class _ReadableHTML(HTMLParser):
    SKIP = {"script", "style", "svg", "noscript", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.title_depth = 0
        self.title: list[str] = []
        self.text: list[str] = []
        self.links: list[dict[str, str]] = []
        self._href: str | None = None
        self._anchor: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self.skip_depth += 1
        if tag == "title":
            self.title_depth += 1
        if tag == "a" and not self.skip_depth:
            self._href = dict(attrs).get("href")
            self._anchor = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        if tag == "title" and self.title_depth:
            self.title_depth -= 1
        if tag == "a" and self._href:
            text = " ".join("".join(self._anchor).split())
            if text:
                self.links.append({"title": text, "url": self._href})
            self._href = None
            self._anchor = []

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self.title_depth:
            self.title.append(data)
        cleaned = " ".join(data.split())
        if cleaned:
            self.text.append(cleaned)
            if self._href:
                self._anchor.append(cleaned)


class _DuckDuckGoLiteResults(HTMLParser):
    """Extract result titles plus the adjacent snippets from DDG Lite."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._capture: str | None = None
        self._depth = 0
        self._buffer: list[str] = []
        self._href = ""

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
        value = dict(attrs).get("class") or ""
        return set(value.split())

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, str | None]]) -> None:
        if self._capture:
            self._depth += 1
            return
        classes = self._classes(attrs)
        if tag == "a" and "result-link" in classes:
            self._capture, self._depth = "title", 1
            self._href = dict(attrs).get("href") or ""
            self._buffer = []
        elif tag == "td" and "result-snippet" in classes:
            self._capture, self._depth = "snippet", 1
            self._buffer = []
        elif tag == "span" and "timestamp" in classes:
            self._capture, self._depth = "published_at", 1
            self._buffer = []

    def handle_endtag(self, _tag: str) -> None:
        if not self._capture:
            return
        self._depth -= 1
        if self._depth:
            return
        value = " ".join(" ".join(self._buffer).split())
        if self._capture == "title" and value and self._href:
            self.results.append({"title": value, "url": self._href})
        elif self.results and value:
            self.results[-1][self._capture] = value
        self._capture = None
        self._buffer = []
        self._href = ""

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._buffer.append(data)


def _fetch(url: str) -> tuple[str, bytes, str]:
    response = request_public_http(
        url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain",
        }, timeout_seconds=15.0, max_response_bytes=MAX_WEB_BYTES,
        allowed_content_types=frozenset({
            "text/html", "application/xhtml+xml", "text/plain"}),
        max_redirects=10)
    if not 200 <= response.status <= 299:
        raise RuntimeError(
            f"web server returned HTTP status {response.status}")
    return response.url, response.body, response.charset


class WebOperator:
    def __init__(self, profile_dir: str | Path, *,
                 chromium_path: str = "/usr/bin/chromium"):
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        info = self.profile_dir.lstat()
        if (not self.profile_dir.is_dir() or self.profile_dir.is_symlink()
                or info.st_uid != os.getuid()):
            raise RuntimeError("managed browser profile boundary is invalid")
        os.chmod(self.profile_dir, 0o700)
        self.chromium_path = chromium_path
        self.debug_port = 9223
        self._browser_process: subprocess.Popen | None = None
        self._managed_runtime_verifier: Callable[[], bool] | None = None

    def require_managed_runtime(self, verifier: Callable[[], bool]) -> None:
        """Disable direct spawning and bind all CDP use to managed evidence."""
        if not callable(verifier):
            raise TypeError("managed browser verifier must be callable")
        if self._managed_runtime_verifier is not None:
            raise RuntimeError("managed browser verifier is already configured")
        self._managed_runtime_verifier = verifier

    def _verify_managed_runtime(self) -> None:
        verifier = self._managed_runtime_verifier
        if verifier is not None and verifier() is not True:
            raise RuntimeError("managed Chromium runtime is not verified")

    def _debug_url(self, path: str = "/json/version") -> str:
        return f"http://127.0.0.1:{self.debug_port}{path}"

    def _probe_debug_endpoint(self) -> None:
        request = urllib.request.Request(self._debug_url())
        with open_loopback_request(request, timeout=1):
            pass

    def _ensure_browser(self) -> None:
        if self._managed_runtime_verifier is not None:
            self._verify_managed_runtime()
            try:
                self._probe_debug_endpoint()
            except Exception as exc:
                raise RuntimeError(
                    "managed Chromium control endpoint is unavailable") from exc
            self._verify_managed_runtime()
            return
        try:
            self._probe_debug_endpoint()
            return
        except Exception:
            pass
        self._browser_process = subprocess.Popen(
            [self.chromium_path, f"--user-data-dir={self.profile_dir}",
             f"--remote-debugging-port={self.debug_port}",
             "--remote-debugging-address=127.0.0.1", "--no-first-run",
             "--no-default-browser-check", "about:blank"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(30):
            try:
                self._probe_debug_endpoint()
                return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("managed Chromium did not expose its local control port")

    @staticmethod
    def _select_page(browser, page_url: str | None = None):
        pages = [page for context in browser.contexts for page in context.pages]
        if not pages:
            raise RuntimeError("managed Chromium has no open pages")
        if page_url:
            selected = next((page for page in reversed(pages)
                             if page_url in page.url), None)
            if selected is None:
                raise ValueError("the requested browser page is not open")
            return selected
        return pages[-1]

    @staticmethod
    def _public_page_url(page) -> str:
        try:
            return validate_public_url(str(page.url))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "browser page is outside the public web boundary") from exc

    def _controlled(self, operation):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "browser automation needs requirements-operator.txt") from exc
        self._ensure_browser()
        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(
                f"http://127.0.0.1:{self.debug_port}")
            self._verify_managed_runtime()
            result = operation(browser)
            self._verify_managed_runtime()
            return result

    def read(self, url: str, *, max_chars: int = 12000) -> dict[str, Any]:
        final_url, data, charset = _fetch(url)
        parser = _ReadableHTML()
        parser.feed(data.decode(charset, errors="replace"))
        text = re.sub(r"\s+", " ", " ".join(parser.text)).strip()[:max_chars]
        return {"url": final_url, "title": " ".join(parser.title).strip(),
                "text": text, "fetched_at": utc_now()}

    def search(self, query: str, *, limit: int = 5) -> dict[str, Any]:
        term = query.strip()
        if not term:
            raise ValueError("search query cannot be empty")
        # The Lite endpoint is stable HTML and does not require executing third-party
        # JavaScript; the result redirect is unwrapped before it becomes evidence.
        url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": term})
        final_url, data, charset = _fetch(url)
        document = data.decode(charset, errors="replace")
        lite = _DuckDuckGoLiteResults()
        lite.feed(document)
        raw_results = lite.results
        if not raw_results:
            # Small fixtures and a future markup fallback still get attributable
            # links, though without snippets.
            parser = _ReadableHTML()
            parser.feed(document)
            raw_results = parser.links
        results = []
        seen: set[str] = set()
        for link in raw_results:
            href = urllib.parse.urljoin(final_url, link["url"])
            parsed = urllib.parse.urlparse(href)
            if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
                target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
                href = urllib.parse.unquote(target) or href
            try:
                href = validate_public_url(href)
            except ValueError:
                continue
            if href in seen or "duckduckgo.com" in urllib.parse.urlparse(href).netloc:
                continue
            seen.add(href)
            item = {"title": link["title"][:300], "url": href,
                    "source": urllib.parse.urlparse(href).netloc}
            if link.get("snippet"):
                item["snippet"] = link["snippet"][:1000]
            if link.get("published_at"):
                item["published_at"] = link["published_at"][:100]
            results.append(item)
            if len(results) >= min(max(limit, 1), 10):
                break
        if not results:
            raise RuntimeError("web search returned no attributable results")
        return {"query": term, "provider": "DuckDuckGo", "results": results,
                "fetched_at": utc_now()}

    def open(self, url: str) -> dict[str, Any]:
        safe = validate_public_url(url)
        def operation(browser):
            if not browser.contexts:
                raise RuntimeError("managed Chromium has no browser context")
            page = browser.contexts[-1].new_page()
            try:
                response = page.goto(
                    safe, wait_until="domcontentloaded", timeout=15000)
                final_url = self._public_page_url(page)
                if response is None:
                    raise RuntimeError(
                        "managed Chromium navigation was not confirmed")
                return {
                    "url": final_url,
                    "title": page.title(),
                    "http_status": int(response.status),
                    "managed": True, "visible": True,
                    "opened_at": utc_now(),
                }
            except Exception:
                try:
                    page.close()
                except Exception:
                    pass
                raise
        return self._controlled(operation)

    def snapshot(self, page_url: str | None = None, *,
                 max_chars: int = 12000) -> dict[str, Any]:
        def operation(browser):
            page = self._select_page(browser, page_url)
            safe_url = self._public_page_url(page)
            return {"url": safe_url, "title": page.title(),
                    "text": page.locator("body").inner_text(timeout=5000)[:max_chars],
                    "observed_at": utc_now()}
        return self._controlled(operation)

    def click(self, selector: str, *, page_url: str | None = None) -> dict[str, Any]:
        if not selector.strip() or len(selector) > 500:
            raise ValueError("browser selector must be 1-500 characters")
        def operation(browser):
            page = self._select_page(browser, page_url)
            self._public_page_url(page)
            page.locator(selector).first.click(timeout=10000)
            page.wait_for_timeout(300)
            safe_url = self._public_page_url(page)
            return {"url": safe_url, "title": page.title(),
                    "selector": selector, "status": "clicked",
                    "observed_at": utc_now()}
        return self._controlled(operation)

    def type(self, selector: str, text: str, *, page_url: str | None = None,
             submit: bool = False) -> dict[str, Any]:
        if not selector.strip() or len(selector) > 500:
            raise ValueError("browser selector must be 1-500 characters")
        if len(text) > 10000:
            raise ValueError("browser input exceeds 10,000 characters")
        def operation(browser):
            page = self._select_page(browser, page_url)
            self._public_page_url(page)
            field = page.locator(selector).first
            field.fill(text, timeout=10000)
            if submit:
                field.press("Enter")
                page.wait_for_timeout(500)
            safe_url = self._public_page_url(page)
            return {"url": safe_url, "title": page.title(),
                    "selector": selector, "submitted": submit,
                    "characters_typed": len(text), "observed_at": utc_now()}
        return self._controlled(operation)


def format_search_result(receipt: dict[str, Any], *, limit: int = 3) -> str:
    items = receipt.get("results", [])[:limit]
    return " ".join(
        f"{index}. {item['title']}, from {item.get('source', 'the web')}."
        for index, item in enumerate(items, 1))
