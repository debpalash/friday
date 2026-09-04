"""server.py counts prompt tokens through the engine's own endpoints."""

from __future__ import annotations

import json
import unittest
from unittest import mock

import server


class _Response:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class TokenCountTests(unittest.TestCase):
    def test_llama_server_renders_then_tokenizes(self) -> None:
        calls = []

        def fake_open(request, timeout):
            calls.append((request.full_url, json.loads(request.data)))
            if request.full_url.endswith("/apply-template"):
                return _Response({"prompt": "<rendered prompt>"})
            return _Response({"tokens": [1, 2, 3]})

        with mock.patch.object(server, "LLM_ENGINE", "llama_server"), \
                mock.patch.object(server, "open_loopback_request", fake_open), \
                mock.patch.object(server, "_tokenize_url",
                                  return_value="http://127.0.0.1:18021/tokenize"):
            count = server.Friday._token_count_sync(
                [{"role": "user", "content": "hi"}], False)
        self.assertEqual(count, 3)
        self.assertEqual(calls[0][0], "http://127.0.0.1:18021/apply-template")
        self.assertEqual(calls[0][1]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(calls[1][0], "http://127.0.0.1:18021/tokenize")
        self.assertEqual(calls[1][1], {"content": "<rendered prompt>", "add_special": False})

    def test_vllm_and_mlx_use_the_count_field(self) -> None:
        for engine in ("vllm", "mlx_lm"):
            calls = []

            def fake_open(request, timeout):
                calls.append(request.full_url)
                return _Response({"count": 9, "max_model_len": 4096})

            with mock.patch.object(server, "LLM_ENGINE", engine), \
                    mock.patch.object(server, "open_loopback_request", fake_open), \
                    mock.patch.object(server, "_tokenize_url",
                                      return_value="http://127.0.0.1:18021/tokenize"):
                count = server.Friday._token_count_sync(
                    [{"role": "user", "content": "hi"}], True)
            self.assertEqual(count, 9)
            self.assertEqual(calls, ["http://127.0.0.1:18021/tokenize"])


if __name__ == "__main__":
    unittest.main()
