"""Friday's launcher for ``mlx_lm.server`` on Apple Silicon.

Upstream ``mlx_lm.server`` has no API key, no ``tool_choice``, and does not
report its context length. This wrapper runs the upstream handler with the
additions Friday's supervisor requires: a bearer credential on every route
except ``/health``, a ``/v1/models`` listing under the served alias, a
vLLM-shaped ``/tokenize`` that reports ``max_model_len``, and a hard context
cap. It runs inside the pinned MLX virtual environment and imports nothing
from ``friday_core`` beyond the standard library.
"""

from __future__ import annotations

import argparse
import hmac
import io
import json
import re
import sys
import threading
from functools import partial
from pathlib import Path

_SERVED = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")


def _load_key(path: str) -> str:
    value = Path(path).read_text(encoding="utf-8").strip()
    if not value:
        raise SystemExit("api key file is empty")
    return value


def build_handler(base_class, *, served_model: str, context_tokens: int,
                  max_sequences: int, api_key: str):
    """Subclass the upstream handler with Friday's authentication and probes."""

    expected = f"Bearer {api_key}".encode("ascii")
    semaphore = threading.BoundedSemaphore(max(1, max_sequences))

    class FridayAPIHandler(base_class):
        def _authorized(self) -> bool:
            supplied = str(self.headers.get("Authorization", "")).encode(
                "utf-8", "replace")
            return hmac.compare_digest(supplied, expected)

        def _reply(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _reject(self, status: int, message: str) -> None:
            self._reply(status, {"error": {"message": message, "type": "friday"}})

        def do_GET(self):  # noqa: N802 - http.server API
            if self.path == "/health":
                return super().do_GET()
            if not self._authorized():
                return self._reject(401, "authentication required")
            if self.path == "/v1/models":
                return self._reply(200, {"object": "list", "data": [
                    {"id": served_model, "object": "model", "owned_by": "friday"}]})
            return self._reject(404, "not found")

        def _read_body(self) -> dict | None:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 64 * 1024 * 1024:
                return None
            try:
                return json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError):
                return None

        def _tokenize(self, body: dict) -> None:
            tokenizer = self.model_provider.tokenizer
            if tokenizer is None:
                return self._reject(503, "model is still loading")
            messages = body.get("messages")
            if isinstance(messages, list):
                extra = body.get("chat_template_kwargs")
                kwargs = dict(extra) if isinstance(extra, dict) else {}
                tools = body.get("tools") if isinstance(body.get("tools"), list) else None
                tokens = tokenizer.apply_chat_template(
                    messages, tools=tools, add_generation_prompt=True,
                    tokenize=True, **kwargs)
            else:
                prompt = body.get("prompt")
                if not isinstance(prompt, str):
                    return self._reject(400, "prompt or messages required")
                tokens = tokenizer.encode(prompt)
            tokens = list(tokens)
            return self._reply(200, {"count": len(tokens), "tokens": tokens,
                                     "max_model_len": context_tokens})

        def do_POST(self):  # noqa: N802 - http.server API
            if not self._authorized():
                return self._reject(401, "authentication required")
            if self.path == "/tokenize":
                body = self._read_body()
                if body is None:
                    return self._reject(400, "invalid request body")
                if body.get("model") not in (None, served_model):
                    return self._reject(404, "unknown model")
                return self._tokenize(body)
            if self.path not in {"/v1/chat/completions", "/chat/completions",
                                 "/v1/completions"}:
                return self._reject(404, "not found")
            body = self._read_body()
            if body is None:
                return self._reject(400, "invalid request body")
            if body.get("model") not in (None, served_model):
                return self._reject(404, "unknown model")
            # Upstream resolves unknown model ids as paths or hub repos;
            # pin the request to the loaded default model instead.
            body["model"] = "default_model"
            requested = body.get("max_tokens")
            if isinstance(requested, int) and requested > context_tokens:
                return self._reject(400, "max_tokens exceeds the maximum context length")
            encoded = json.dumps(body).encode("utf-8")
            self.headers.replace_header("Content-Length", str(len(encoded)))
            self.rfile = io.BytesIO(encoded)
            if not semaphore.acquire(timeout=120):
                return self._reject(503, "all sequence slots are busy")
            try:
                return super().do_POST()
            finally:
                semaphore.release()

    return FridayAPIHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Friday MLX model server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18021)
    parser.add_argument("--api-key-file", required=True)
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        raise SystemExit("the MLX server binds to 127.0.0.1 only")
    if not _SERVED.fullmatch(args.served_model):
        raise SystemExit("served model name is not safe")
    if not 2048 <= args.context_tokens <= 1_000_000:
        raise SystemExit("context tokens are out of range")
    api_key = _load_key(args.api_key_file)

    from mlx_lm import server as upstream  # noqa: PLC0415

    namespace = argparse.Namespace(
        model=args.model, adapter_path=None, trust_remote_code=False,
        chat_template="", use_default_chat_template=False,
        chat_template_args={"enable_thinking": False}, draft_model=None,
        num_draft_tokens=3, log_level="INFO", max_tokens=min(4096, args.context_tokens),
        temp=0.0, top_p=1.0, top_k=0, min_p=0.0, allowed_origins=None,
        prompt_cache_size=8, prompt_cache_bytes=None, decode_concurrency=1,
        prompt_concurrency=1, prefill_step_size=2048, pipeline=False)
    provider = upstream.ModelProvider(namespace)
    provider.load(args.model)
    handler = build_handler(
        upstream.APIHandler, served_model=args.served_model,
        context_tokens=args.context_tokens, max_sequences=args.max_sequences,
        api_key=api_key)
    print(f"friday mlx server: {args.served_model} on {args.host}:{args.port}",
          file=sys.stderr, flush=True)
    upstream.run(args.host, args.port, provider, handler_class=handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
