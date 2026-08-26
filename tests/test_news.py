import unittest
from unittest.mock import patch

import server
from friday_core.news import (MAX_FEED_BYTES, fetch_news, format_news_brief,
                              format_news_segments, parse_news_feed)
from friday_core.public_http import PublicHTTPResponse


FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item><title>First &amp; verified</title><link>https://example.com/1</link>
    <pubDate>Sat, 22 Aug 2026 04:00:00 GMT</pubDate><source>Source One</source></item>
  <item><title>Second story</title><link>https://example.com/2</link>
    <pubDate>Sat, 22 Aug 2026 03:00:00 GMT</pubDate><source>Source Two</source></item>
</channel></rss>"""


class _Response:
    def __init__(self, payload=FEED):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return self.payload


class NewsTests(unittest.TestCase):
    def test_production_fetch_uses_pinned_public_transport(self):
        response = PublicHTTPResponse(
            url="https://news.google.com/rss/search", status=200,
            content_type="application/rss+xml", charset="utf-8", body=FEED)
        with patch("friday_core.news.request_public_http",
                   return_value=response) as request:
            result = fetch_news("technology", 2, "India")

        self.assertEqual(len(result["headlines"]), 2)
        self.assertEqual(request.call_args.kwargs["max_redirects"], 5)
        self.assertEqual(
            request.call_args.kwargs["max_response_bytes"], MAX_FEED_BYTES)
        self.assertTrue(request.call_args.args[0].startswith(
            "https://news.google.com/rss/search?"))

    def test_feed_is_bounded_and_source_attributed(self):
        headlines = parse_news_feed(FEED, limit=1)

        self.assertEqual(len(headlines), 1)
        self.assertEqual(headlines[0]["title"], "First & verified")
        self.assertEqual(headlines[0]["source"], "Source One")
        self.assertEqual(headlines[0]["published_at"], "2026-08-22T04:00:00Z")

    def test_fetch_uses_india_locale_and_topic(self):
        requests = []

        def opener(request, timeout):
            requests.append((request.full_url, timeout))
            return _Response()

        result = fetch_news("technology India", 2, opener=opener)

        self.assertEqual(len(result["headlines"]), 2)
        self.assertIn("q=technology+India+when%3A1d", requests[0][0])
        self.assertIn("ceid=IN%3Aen", requests[0][0])
        self.assertEqual(requests[0][1], 10)

    def test_bare_us_topic_switches_region_and_locale(self):
        requests = []

        def opener(request, timeout):
            requests.append(request.full_url)
            return _Response()

        result = fetch_news("US", 2, opener=opener)

        self.assertEqual(result["region"], "United States")
        self.assertEqual(result["topic"], "United States top stories")
        self.assertIn("q=United+States+when%3A1d", requests[0])
        self.assertIn("ceid=US%3Aen", requests[0])
        self.assertNotIn("India", requests[0])

    def test_spoken_brief_contains_only_receipt_headlines_and_sources(self):
        receipt = {"region": "India", "headlines": [
            {"title": "First verified story", "source": "Source One"},
            {"title": "Second verified story", "source": "Source Two"},
        ]}

        brief = format_news_brief(receipt)

        self.assertIn("Here are today's top India stories", brief)
        self.assertIn("First verified story, from Source One", brief)
        self.assertIn("Second verified story, from Source Two", brief)
        segments = format_news_segments(receipt)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0], "Here are today's top India stories.")

    def test_oversized_feed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "response-size"):
            fetch_news(opener=lambda *_args, **_kwargs: _Response(
                b"x" * (MAX_FEED_BYTES + 1)))

    def test_builtin_returns_receipt_and_errors_are_explicit(self):
        with patch("server.fetch_news", return_value={"headlines": [{"title": "A"}]}):
            self.assertIn('"title": "A"', server.exec_tool("fetch_news", {}))
        with patch("server.fetch_news", side_effect=TimeoutError("offline")):
            self.assertEqual(server.exec_tool("fetch_news", {}),
                             "error: news fetch failed: offline")
