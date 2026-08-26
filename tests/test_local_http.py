import json
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock, patch

import server
from friday_core.local_http import (
    _RejectLoopbackRedirects, normalize_loopback_model_base_url,
    open_loopback_request,
)


class LocalHTTPTests(unittest.TestCase):
    def test_model_base_is_exact_numeric_loopback_v1(self):
        self.assertEqual(
            normalize_loopback_model_base_url(
                "http://127.0.0.1:18021/v1/"),
            "http://127.0.0.1:18021/v1")
        for value in (
                "http://localhost:18021/v1",
                "http://192.168.1.2:18021/v1",
                "https://127.0.0.1:18021/v1",
                "http://user:secret@127.0.0.1:18021/v1",
                "http://127.0.0.1:18021/other",
                "http://127.0.0.1:18021/v1?redirect=evil"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_loopback_model_base_url(value)

    def test_loopback_open_disables_proxies_and_redirects(self):
        request = urllib.request.Request(
            "http://127.0.0.1:18021/tokenize",
            headers={"Authorization": "Bearer private"})
        opener = Mock()
        response = Mock()
        opener.open.return_value = response
        with patch("friday_core.local_http.urllib.request.build_opener",
                   return_value=opener) as build:
            observed = open_loopback_request(request, timeout=5)

        self.assertIs(observed, response)
        self.assertIsInstance(
            build.call_args.args[0], urllib.request.ProxyHandler)
        self.assertEqual(build.call_args.args[0].proxies, {})
        self.assertIsInstance(
            build.call_args.args[1], _RejectLoopbackRedirects)
        opener.open.assert_called_once_with(request, timeout=5)

    def test_loopback_redirect_refuses_credential_forwarding(self):
        handler = _RejectLoopbackRedirects()
        request = urllib.request.Request(
            "http://127.0.0.1:18021/v1/chat/completions")
        with self.assertRaises(urllib.error.HTTPError):
            handler.redirect_request(
                request, None, 307, "redirect", {},
                "https://attacker.example/steal")

    def test_server_openai_client_disables_env_proxy_and_redirects(self):
        transport = Mock()
        client = Mock()
        with (patch.object(server, "DefaultAsyncHttpxClient",
                           return_value=transport) as create_transport,
              patch.object(server, "AsyncOpenAI",
                           return_value=client) as create_client):
            observed = server._new_local_llm_client()

        self.assertIs(observed, client)
        create_transport.assert_called_once_with(
            trust_env=False, follow_redirects=False)
        self.assertIs(create_client.call_args.kwargs["http_client"], transport)
        self.assertEqual(
            create_client.call_args.kwargs["base_url"], server.LOCAL_BASE_URL)

    def test_token_count_uses_proxy_free_authenticated_loopback_request(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps({"count": 123}).encode()
        with patch.object(server, "open_loopback_request",
                          return_value=response) as open_request:
            count = server.Friday._token_count_sync(
                [{"role": "user", "content": "hello"}], False)

        self.assertEqual(count, 123)
        request = open_request.call_args.args[0]
        self.assertEqual(request.full_url, server._tokenize_url())
        self.assertEqual(
            request.get_header("Authorization"), "Bearer " + server.KEY)
        self.assertEqual(open_request.call_args.kwargs["timeout"], 5)


if __name__ == "__main__":
    unittest.main()
