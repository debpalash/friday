"""The MLX wrapper adds authentication and context reporting to the upstream server."""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from friday_core.mlx_server import build_handler


class FakeTokenizer:
    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True,
                            tokenize=True, **kwargs):
        assert kwargs.get("enable_thinking") is False
        return list(range(len(messages) * 3 + (2 if tools else 0)))

    def encode(self, prompt):
        return [7] * len(prompt.split())


class FakeProvider:
    tokenizer = FakeTokenizer()


class FakeUpstreamHandler(BaseHTTPRequestHandler):
    """Stands in for mlx_lm.server.APIHandler: it echoes the body it receives."""

    def __init__(self, model_provider, *args, **kwargs):
        self.model_provider = model_provider
        super().__init__(*args, **kwargs)

    def log_message(self, *_args):
        pass

    def do_GET(self):  # noqa: N802
        body = b'{"status": "ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        received = json.loads(self.rfile.read(length))
        body = json.dumps({"echo": received}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class WrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        handler = build_handler(
            FakeUpstreamHandler, served_model="qwen3-8b", context_tokens=16384,
            max_sequences=2, api_key="topsecret")
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(handler, FakeProvider()))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def _call(self, path, *, method="GET", body=None, key="topsecret"):
        headers = {"Content-Type": "application/json"}
        if key is not None:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            self.base + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read())
            exc.close()
            return exc.code, payload

    def test_health_is_public_and_everything_else_needs_the_key(self) -> None:
        self.assertEqual(self._call("/health", key=None)[0], 200)
        self.assertEqual(self._call("/v1/models", key=None)[0], 401)
        self.assertEqual(self._call("/v1/models", key="wrong")[0], 401)
        status, models = self._call("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual([item["id"] for item in models["data"]], ["qwen3-8b"])
        self.assertEqual(self._call("/nope")[0], 404)

    def test_tokenize_reports_the_context_ceiling(self) -> None:
        status, payload = self._call("/tokenize", method="POST",
                                     body={"model": "qwen3-8b", "prompt": "a b c"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["count"], 3)
        self.assertEqual(payload["max_model_len"], 16384)
        status, payload = self._call("/tokenize", method="POST", body={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
            "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(payload["count"], 5)
        self.assertEqual(self._call("/tokenize", method="POST",
                                    body={"model": "other", "prompt": "x"})[0], 404)
        self.assertEqual(self._call("/tokenize", method="POST", body={})[0], 400)

    def test_chat_completions_are_pinned_to_the_loaded_model(self) -> None:
        status, payload = self._call("/v1/chat/completions", method="POST", body={
            "model": "qwen3-8b", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32})
        self.assertEqual(status, 200)
        self.assertEqual(payload["echo"]["model"], "default_model")
        self.assertEqual(self._call("/v1/chat/completions", method="POST",
                                    body={"model": "gpt", "messages": []})[0], 404)
        self.assertEqual(self._call("/v1/chat/completions", method="POST",
                                    body={"messages": [], "max_tokens": 99999})[0], 400)
        self.assertEqual(self._call("/v1/chat/completions", method="POST",
                                    key=None, body={"messages": []})[0], 401)


if __name__ == "__main__":
    unittest.main()
