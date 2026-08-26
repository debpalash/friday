"""Small, dependency-free live news client for Friday."""

from __future__ import annotations

import html
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from .public_http import request_public_http


GOOGLE_NEWS_RSS = "https://news.google.com/rss"
MAX_FEED_BYTES = 1_000_000
REGIONS = {
    "india": {"name": "India", "hl": "en-IN", "gl": "IN",
              "ceid": "IN:en", "search": "India"},
    "us": {"name": "United States", "hl": "en-US", "gl": "US",
           "ceid": "US:en", "search": "United States"},
    "usa": {"name": "United States", "hl": "en-US", "gl": "US",
            "ceid": "US:en", "search": "United States"},
    "united states": {"name": "United States", "hl": "en-US", "gl": "US",
                      "ceid": "US:en", "search": "United States"},
    "uk": {"name": "United Kingdom", "hl": "en-GB", "gl": "GB",
           "ceid": "GB:en", "search": "United Kingdom"},
    "united kingdom": {"name": "United Kingdom", "hl": "en-GB", "gl": "GB",
                       "ceid": "GB:en", "search": "United Kingdom"},
    "world": {"name": "World", "hl": "en-US", "gl": "US",
              "ceid": "US:en", "search": "world"},
    "global": {"name": "World", "hl": "en-US", "gl": "US",
               "ceid": "US:en", "search": "world"},
}


def _published(value: str) -> str:
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OverflowError):
        return value.strip()


def parse_news_feed(payload: bytes, *, limit: int = 5) -> list[dict[str, str]]:
    """Parse an RSS feed into bounded, source-attributed headline records."""
    root = ET.fromstring(payload)
    headlines: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in root.findall("./channel/item"):
        title = html.unescape((item.findtext("title") or "").strip())
        url = (item.findtext("link") or "").strip()
        source_node = item.find("source")
        source = ((source_node.text or "").strip()
                  if source_node is not None else "Unknown source")
        suffix = f" - {source}"
        if source != "Unknown source" and title.casefold().endswith(suffix.casefold()):
            title = title[:-len(suffix)].rstrip()
        if not title or not url or not url.startswith(("https://", "http://")):
            continue
        key = (title.casefold(), source.casefold())
        if key in seen:
            continue
        seen.add(key)
        headlines.append({
            "title": title,
            "source": source,
            "published_at": _published(item.findtext("pubDate") or ""),
            "url": url,
        })
        if len(headlines) >= min(max(int(limit), 1), 10):
            break
    if not headlines:
        raise ValueError("news feed returned no usable headlines")
    return headlines


def fetch_news(topic: str = "", limit: int = 5, region: str = "India", *,
               opener=None) -> dict:
    """Fetch current English-language headlines by region and optional topic."""
    count = min(max(int(limit), 1), 10)
    query = str(topic or "").strip()
    region_key = str(region or "India").strip().casefold()
    # Tool-call models often put a bare country in `topic`. Treat that as a
    # region switch, which also handles the observed {"topic": "US"} call.
    topic_key = query.casefold().strip(" .")
    if region_key == "india" and topic_key in REGIONS and topic_key != "india":
        region_key, query = topic_key, ""
    config = REGIONS.get(region_key)
    if config is None:
        supported = ", ".join(sorted({item["name"] for item in REGIONS.values()}))
        raise ValueError(f"unsupported news region {region!r}; supported: {supported}")
    search_terms = query
    if config["search"].casefold() not in search_terms.casefold():
        search_terms = (search_terms + " " + config["search"]).strip()
    search_terms = (search_terms + " when:1d").strip()
    endpoint = GOOGLE_NEWS_RSS + "/search?" + urllib.parse.urlencode({
        "q": search_terms,
        "hl": config["hl"],
        "gl": config["gl"],
        "ceid": config["ceid"],
    })
    if opener is None:
        response = request_public_http(
            endpoint,
            headers={
                "User-Agent": "Friday/1.0 (+local personal assistant)",
                "Accept": "application/rss+xml,application/xml,text/xml",
            }, timeout_seconds=10, max_response_bytes=MAX_FEED_BYTES,
            allowed_content_types=frozenset({
                "application/rss+xml", "application/xml", "text/xml"}),
            max_redirects=5)
        if not 200 <= response.status <= 299:
            raise RuntimeError(
                f"news provider returned HTTP status {response.status}")
        payload = response.body
    else:
        # Dependency injection is retained for deterministic offline tests;
        # the server never supplies an alternate opener.
        request = urllib.request.Request(
            endpoint,
            headers={"User-Agent": "Friday/1.0 (+local personal assistant)"},
        )
        with opener(request, timeout=10) as response:
            payload = response.read(MAX_FEED_BYTES + 1)
    if len(payload) > MAX_FEED_BYTES:
        raise ValueError("news feed exceeded the response-size limit")
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "region": config["name"],
        "topic": query or f"{config['name']} top stories",
        "headlines": parse_news_feed(payload, limit=count),
    }


def format_news_segments(receipt: dict, *, max_headlines: int = 3) -> list[str]:
    """Create fast TTS-sized sentences containing only receipt-backed claims."""
    region = str(receipt.get("region") or "current").strip()
    headlines = list(receipt.get("headlines") or [])[:max(1, max_headlines)]
    items = []
    for headline in headlines:
        title = str(headline.get("title") or "").strip().rstrip(". ")
        source = str(headline.get("source") or "Unknown source").strip()
        if title:
            items.append(f"{title}, from {source}.")
    if not items:
        raise ValueError("news receipt contained no speakable headlines")
    return [f"Here are today's top {region} stories.", *items]


def format_news_brief(receipt: dict, *, max_headlines: int = 3) -> str:
    """Return the same grounded news delivery as one text transcript."""
    return " ".join(format_news_segments(receipt, max_headlines=max_headlines))
