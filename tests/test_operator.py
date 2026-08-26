import tempfile
import socket
import threading
import unittest
import stat
from pathlib import Path
from unittest.mock import Mock, call, patch

from friday_core.operator import WebOperator, _fetch, validate_public_url
from friday_core.web_proxy import PublicNetworkDenied


class OperatorTests(unittest.TestCase):
    @staticmethod
    def _http_peer(response: bytes, requests: list[bytes]):
        client, server = socket.socketpair()

        def serve():
            try:
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = server.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                requests.append(data)
                server.sendall(response)
            finally:
                server.close()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        return client, thread

    def test_browser_profile_is_owner_private_and_rejects_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile"
            WebOperator(profile)
            self.assertEqual(stat.S_IMODE(profile.stat().st_mode), 0o700)
            target = root / "target"
            target.mkdir()
            alias = root / "alias"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "profile boundary"):
                WebOperator(alias)

    def test_managed_browser_never_attaches_or_spawns_without_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            operator = WebOperator(Path(tmp) / "profile")
            operator.require_managed_runtime(lambda: False)
            with (patch("friday_core.operator.open_loopback_request") as open_,
                  patch("friday_core.operator.subprocess.Popen") as spawn,
                  self.assertRaisesRegex(RuntimeError, "not verified")):
                operator._ensure_browser()
            open_.assert_not_called()
            spawn.assert_not_called()

    def test_managed_browser_endpoint_is_fenced_before_and_after_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            operator = WebOperator(Path(tmp) / "profile")
            proof = Mock(side_effect=[True, True])
            operator.require_managed_runtime(proof)
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            with (patch("friday_core.operator.open_loopback_request",
                        return_value=response),
                  patch("friday_core.operator.subprocess.Popen") as spawn):
                operator._ensure_browser()
            self.assertEqual(proof.call_count, 2)
            spawn.assert_not_called()

            changed = WebOperator(Path(tmp) / "changed-profile")
            changed.require_managed_runtime(
                Mock(side_effect=[True, False]))
            with (patch("friday_core.operator.open_loopback_request",
                        return_value=response),
                  self.assertRaisesRegex(RuntimeError, "not verified")):
                changed._ensure_browser()

    def test_managed_navigation_failure_never_falls_back_to_unowned_spawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            operator = WebOperator(Path(tmp) / "profile")
            operator.require_managed_runtime(lambda: True)
            with (patch("friday_core.operator.validate_public_url",
                        return_value="https://example.com"),
                  patch.object(operator, "_controlled",
                               side_effect=RuntimeError("navigation failed")),
                  patch("friday_core.operator.subprocess.Popen") as spawn,
                  self.assertRaisesRegex(RuntimeError, "navigation failed")):
                operator.open("https://example.com")
            spawn.assert_not_called()

    def test_managed_navigation_receipt_does_not_expose_profile_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "private-profile"
            operator = WebOperator(profile)
            operator.require_managed_runtime(lambda: True)
            receipt = {
                "url": "https://example.com/final", "title": "Example",
                "http_status": 200, "managed": True, "visible": True,
                "opened_at": "now",
            }
            with (patch("friday_core.operator.validate_public_url",
                        return_value="https://example.com"),
                  patch.object(operator, "_controlled", return_value=receipt),
                  patch("friday_core.operator.subprocess.Popen") as spawn):
                observed = operator.open("https://example.com")
            spawn.assert_not_called()
            self.assertTrue(observed["managed"])
            self.assertNotIn(str(profile), str(observed))

    def test_managed_navigation_confirms_public_final_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            operator = WebOperator(Path(tmp) / "profile")
            page = Mock(url="https://example.com/final")
            page.title.return_value = "Final page"
            response = Mock(status=204)
            page.goto.return_value = response
            context = Mock()
            context.new_page.return_value = page
            browser = Mock(contexts=[context])
            with (patch.object(
                      operator, "_controlled",
                      side_effect=lambda operation: operation(browser)),
                  patch("friday_core.operator.validate_public_url",
                        side_effect=["https://example.com/start",
                                     "https://example.com/final"])):
                receipt = operator.open("https://example.com/start")

            page.goto.assert_called_once_with(
                "https://example.com/start", wait_until="domcontentloaded",
                timeout=15000)
            self.assertEqual(receipt["url"], "https://example.com/final")
            self.assertEqual(receipt["http_status"], 204)
            self.assertEqual(receipt["title"], "Final page")

    def test_managed_navigation_rejects_private_final_page_and_closes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            operator = WebOperator(Path(tmp) / "profile")
            page = Mock(url="http://127.0.0.1/private")
            page.goto.return_value = Mock(status=200)
            context = Mock()
            context.new_page.return_value = page
            browser = Mock(contexts=[context])
            with (patch.object(
                      operator, "_controlled",
                      side_effect=lambda operation: operation(browser)),
                  patch("friday_core.operator.validate_public_url",
                        side_effect=["https://example.com/start",
                                     ValueError("private blocked")]),
                  self.assertRaisesRegex(RuntimeError, "public web boundary")):
                operator.open("https://example.com/start")
            page.close.assert_called_once()

    def test_private_network_urls_are_blocked(self):
        with self.assertRaisesRegex(ValueError, "local network"):
            validate_public_url("http://localhost/admin")

    def test_url_validation_uses_scheme_port_and_rejects_credentials(self):
        with patch("friday_core.public_http.resolve_public_endpoints") as resolve:
            self.assertEqual(
                validate_public_url("http://Example.COM/path?q=1"),
                "http://example.com/path?q=1")
        resolve.assert_called_once_with("example.com", 80)

        with self.assertRaisesRegex(ValueError, "credentials"):
            validate_public_url("https://user:secret@example.com/")

    def test_url_validation_rejects_control_characters_and_invalid_ports(self):
        for url in ("https://example.com/a\nb", "https://example.com:99999/"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_public_url(url)

    def test_fetch_connects_to_pinned_endpoint_and_sends_origin_form(self):
        requests = []
        client, peer = self._http_peer(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\n"
            b"Content-Length: 2\r\nConnection: close\r\n\r\nok",
            requests)
        with (patch("friday_core.public_http.normalize_public_http_url",
                    return_value="http://example.com/path?q=1"),
              patch("friday_core.public_http.connect_public_stream",
                    return_value=client) as connect):
            final, data, charset = _fetch("http://example.com/path?q=1")
        peer.join(timeout=2)

        self.assertEqual((final, data, charset),
                         ("http://example.com/path?q=1", b"ok", "utf-8"))
        connect.assert_called_once_with(
            "example.com", 80, timeout_seconds=15.0)
        self.assertTrue(requests[0].startswith(
            b"GET /path?q=1 HTTP/1.1\r\n"))
        self.assertIn(b"\r\nHost: example.com\r\n", requests[0])

    def test_fetch_revalidates_and_repins_every_redirect(self):
        requests = []
        first, first_peer = self._http_peer(
            b"HTTP/1.1 302 Found\r\nLocation: http://other.example/final\r\n"
            b"Content-Length: 0\r\nConnection: close\r\n\r\n", requests)
        second, second_peer = self._http_peer(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
            b"Content-Length: 5\r\nConnection: close\r\n\r\nfinal", requests)
        with (patch("friday_core.public_http.normalize_public_http_url",
                    side_effect=["http://example.com/start",
                                 "http://other.example/final"]),
              patch("friday_core.public_http.connect_public_stream",
                    side_effect=[first, second]) as connect):
            final, data, _charset = _fetch("http://example.com/start")
        first_peer.join(timeout=2)
        second_peer.join(timeout=2)

        self.assertEqual(final, "http://other.example/final")
        self.assertEqual(data, b"final")
        self.assertEqual(connect.call_args_list, [
            call("example.com", 80, timeout_seconds=15.0),
            call("other.example", 80, timeout_seconds=15.0),
        ])

    def test_fetch_rebinding_to_private_address_fails_before_http(self):
        with (patch("friday_core.public_http.normalize_public_http_url",
                    return_value="http://example.com/"),
              patch("friday_core.public_http.connect_public_stream",
                    side_effect=PublicNetworkDenied("private blocked")),
              patch("friday_core.operator.open_loopback_request") as legacy,
              self.assertRaises(PublicNetworkDenied)):
            _fetch("http://example.com/")
        legacy.assert_not_called()

    def test_read_extracts_visible_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            operator = WebOperator(Path(tmp) / "profile")
            html = b"<html><title>Page</title><script>bad()</script><p>Useful text.</p></html>"
            with patch("friday_core.operator._fetch",
                       return_value=("https://example.com", html, "utf-8")):
                receipt = operator.read("https://example.com")
        self.assertEqual(receipt["title"], "Page")
        self.assertIn("Useful text", receipt["text"])
        self.assertNotIn("bad()", receipt["text"])

    def test_search_returns_attributed_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            operator = WebOperator(Path(tmp) / "profile")
            html = b'<a href="https://example.com/a">First result</a>'
            with patch("friday_core.operator._fetch",
                       return_value=("https://duckduckgo.com/html", html, "utf-8")), \
                    patch("friday_core.operator.validate_public_url",
                          side_effect=lambda url: url):
                receipt = operator.search("test")
        self.assertEqual(receipt["results"][0]["source"], "example.com")

    def test_search_retains_snippets_for_grounded_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            operator = WebOperator(Path(tmp) / "profile")
            html = b"""
              <a class='result-link' href='https://example.com/story'>Current move</a>
              <td class='result-snippet'>The leader announced <b>new sanctions</b> today.</td>
              <span class='timestamp'>2026-08-22</span>
            """
            with patch("friday_core.operator._fetch",
                       return_value=("https://lite.duckduckgo.com/lite/", html,
                                     "utf-8")), \
                    patch("friday_core.operator.validate_public_url",
                          side_effect=lambda url: url):
                receipt = operator.search("current move")

        self.assertEqual(receipt["results"][0]["snippet"],
                         "The leader announced new sanctions today.")
        self.assertEqual(receipt["results"][0]["published_at"], "2026-08-22")


if __name__ == "__main__":
    unittest.main()
