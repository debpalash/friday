"""Exact loopback-only HTTP helpers for Friday's local model control plane."""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request


def normalize_loopback_model_base_url(value: str) -> str:
    if not isinstance(value, str) or any(
            ord(char) <= 0x20 or ord(char) == 0x7f for char in value):
        raise ValueError("local model URL is invalid")
    try:
        parsed = urllib.parse.urlsplit(value.strip().rstrip("/"))
        port = parsed.port
    except ValueError as exc:
        raise ValueError("local model URL is invalid") from exc
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or port is None or not 1 <= port <= 65535
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1"):
        raise ValueError(
            "local model URL must be exact 127.0.0.1 HTTP ending in /v1")
    return f"http://127.0.0.1:{port}/v1"


class _RejectLoopbackRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers,
                         new_url):
        raise urllib.error.HTTPError(
            request.full_url, code, "local credential redirect refused",
            headers, file_pointer)


def open_loopback_request(
    request: urllib.request.Request,
    *,
    timeout: float,
):
    if timeout <= 0 or timeout > 120:
        raise ValueError("local model timeout is invalid")
    try:
        target = urllib.parse.urlsplit(request.full_url)
        port = target.port
    except ValueError as exc:
        raise ValueError("local model request URL is invalid") from exc
    if (target.scheme != "http" or target.hostname != "127.0.0.1"
            or port is None or not 1 <= port <= 65535
            or target.username is not None or target.password is not None
            or target.fragment):
        raise ValueError("local model request must target exact loopback HTTP")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _RejectLoopbackRedirects())
    return opener.open(request, timeout=timeout)


__all__ = ["normalize_loopback_model_base_url", "open_loopback_request"]
