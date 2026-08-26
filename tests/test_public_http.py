import socket
import threading
import unittest
from unittest.mock import Mock, call, patch

from friday_core.public_http import (
    PublicHTTPRedirectError, request_public_http,
)


def _peer(response: bytes, requests: list[bytes]):
    client, server = socket.socketpair()

    def serve():
        try:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = server.recv(4096)
                if not chunk:
                    break
                data += chunk
            header, _, remainder = data.partition(b"\r\n\r\n")
            length = 0
            for line in header.split(b"\r\n")[1:]:
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":", 1)[1].strip())
            while len(remainder) < length:
                remainder += server.recv(length - len(remainder))
            requests.append(header + b"\r\n\r\n" + remainder)
            server.sendall(response)
        finally:
            server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return client, thread


class PublicHTTPTests(unittest.TestCase):
    def test_credentialed_post_is_exact_and_never_follows_redirect(self):
        requests = []
        client, thread = _peer(
            b"HTTP/1.1 307 Temporary Redirect\r\n"
            b"Location: http://other.example/steal\r\n"
            b"Content-Length: 0\r\nConnection: close\r\n\r\n", requests)
        with (patch(
                  "friday_core.public_http.normalize_public_http_url",
                  return_value="http://provider.example/v1"),
              patch("friday_core.public_http.connect_public_stream",
                    return_value=client) as connect,
              self.assertRaises(PublicHTTPRedirectError)):
            request_public_http(
                "http://provider.example/v1", method="POST", body=b"{}",
                headers={"Authorization": "Bearer private",
                         "Content-Type": "application/json"},
                allow_redirects=True)
        thread.join(timeout=2)

        connect.assert_called_once_with(
            "provider.example", 80, timeout_seconds=15.0)
        self.assertIn(b"Authorization: Bearer private\r\n", requests[0])
        self.assertTrue(requests[0].endswith(b"\r\n\r\n{}"))

    def test_sensitive_get_redirect_is_not_forwarded(self):
        requests = []
        client, thread = _peer(
            b"HTTP/1.1 302 Found\r\nLocation: http://other.example/\r\n"
            b"Content-Length: 0\r\nConnection: close\r\n\r\n", requests)
        with (patch(
                  "friday_core.public_http.normalize_public_http_url",
                  return_value="http://provider.example/start") as normalize,
              patch("friday_core.public_http.connect_public_stream",
                    return_value=client),
              self.assertRaises(PublicHTTPRedirectError)):
            request_public_http(
                "http://provider.example/start",
                headers={"Cookie": "session=private"})
        thread.join(timeout=2)
        self.assertEqual(normalize.call_count, 1)
        self.assertIn(b"Cookie: session=private\r\n", requests[0])

    def test_public_get_revalidates_and_repins_redirect(self):
        requests = []
        first, first_thread = _peer(
            b"HTTP/1.1 302 Found\r\nLocation: /final\r\n"
            b"Content-Length: 0\r\nConnection: close\r\n\r\n", requests)
        second, second_thread = _peer(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\n"
            b"Content-Length: 2\r\nConnection: close\r\n\r\nok", requests)
        with (patch(
                  "friday_core.public_http.normalize_public_http_url",
                  side_effect=["http://example.com/start",
                               "http://example.com/final"]),
              patch("friday_core.public_http.connect_public_stream",
                    side_effect=[first, second]) as connect):
            response = request_public_http(
                "http://example.com/start",
                allowed_content_types=frozenset({"text/plain"}))
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

        self.assertEqual(response.url, "http://example.com/final")
        self.assertEqual(response.body, b"ok")
        self.assertEqual(connect.call_args_list, [
            call("example.com", 80, timeout_seconds=15.0),
            call("example.com", 80, timeout_seconds=15.0),
        ])

    def test_https_redirect_cannot_downgrade_to_plaintext(self):
        requests = []
        client, thread = _peer(
            b"HTTP/1.1 302 Found\r\nLocation: http://example.com/plain\r\n"
            b"Content-Length: 0\r\nConnection: close\r\n\r\n", requests)
        tls = Mock()
        tls.settimeout = client.settimeout
        tls.sendall = client.sendall
        tls.makefile = client.makefile
        tls.close = client.close
        with (patch(
                  "friday_core.public_http.normalize_public_http_url",
                  side_effect=["https://example.com/start",
                               "http://example.com/plain"]),
              patch("friday_core.public_http.connect_public_stream",
                    return_value=client),
              patch("friday_core.public_http.ssl.create_default_context")
              as context,
              self.assertRaisesRegex(
                  PublicHTTPRedirectError, "downgrade")):
            context.return_value.wrap_socket.return_value = tls
            request_public_http("https://example.com/start")
        thread.join(timeout=2)

    def test_invalid_headers_fail_before_dns_or_connection(self):
        for headers in (
                {"Host": "attacker"},
                {"X-Test": "safe\r\nAuthorization: stolen"},
                {"Transfer-Encoding": "chunked"}):
            with (self.subTest(headers=headers),
                  patch("friday_core.public_http.normalize_public_http_url")
                  as normalize,
                  patch("friday_core.public_http.connect_public_stream")
                  as connect,
                  self.assertRaises(ValueError)):
                request_public_http("https://example.com", headers=headers)
            normalize.assert_not_called()
            connect.assert_not_called()

    def test_announced_oversize_is_rejected_without_reading_body(self):
        requests = []
        client, thread = _peer(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: 1001\r\nConnection: close\r\n\r\n", requests)
        with (patch(
                  "friday_core.public_http.normalize_public_http_url",
                  return_value="http://example.com/"),
              patch("friday_core.public_http.connect_public_stream",
                    return_value=client),
              self.assertRaisesRegex(ValueError, "size limit")):
            request_public_http(
                "http://example.com", max_response_bytes=1000)
        thread.join(timeout=2)

    def test_duplicate_length_is_rejected_as_ambiguous(self):
        requests = []
        client, thread = _peer(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
            b"Content-Length: 2\r\nContent-Length: 2\r\n"
            b"Connection: close\r\n\r\nok", requests)
        with (patch(
                  "friday_core.public_http.normalize_public_http_url",
                  return_value="http://example.com/"),
              patch("friday_core.public_http.connect_public_stream",
                    return_value=client),
              self.assertRaisesRegex(ValueError, "ambiguous")):
            request_public_http("http://example.com")
        thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
