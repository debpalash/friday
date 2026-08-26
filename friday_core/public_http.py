"""Small connect-time-pinned HTTP transport for public built-in services."""

from __future__ import annotations

import http.client
import re
import socket
import ssl
import urllib.parse
from dataclasses import dataclass
from typing import Mapping

from .web_proxy import (PublicNetworkError, connect_public_stream,
                        normalize_public_host, resolve_public_endpoints)


_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,80}\Z")
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_FORBIDDEN_CALLER_HEADERS = frozenset({
    "accept-encoding", "connection", "content-length", "host",
    "proxy-authorization", "proxy-connection", "te", "trailer",
    "transfer-encoding", "upgrade",
})
_SENSITIVE_HEADERS = frozenset({"authorization", "cookie"})


class PublicHTTPError(RuntimeError):
    """Base class for bounded public HTTP transport errors."""


class PublicHTTPRedirectError(PublicHTTPError):
    """A redirect violated caller policy or its bounded chain."""


@dataclass(frozen=True)
class PublicHTTPResponse:
    url: str
    status: int
    content_type: str
    charset: str
    body: bytes


def normalize_public_http_url(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("URL must be a string")
    supplied = value.strip()
    if (not supplied or any(ord(char) <= 0x20 or ord(char) == 0x7f
                            for char in supplied)):
        raise ValueError("URL contains invalid characters")
    try:
        parsed = urllib.parse.urlsplit(supplied)
        hostname = parsed.hostname
        explicit_port = parsed.port
    except ValueError as exc:
        raise ValueError("URL authority is invalid") from exc
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("URL must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL credentials are not available to public clients")
    try:
        port = explicit_port or (443 if parsed.scheme == "https" else 80)
        host = normalize_public_host(hostname)
        resolve_public_endpoints(host, port)
    except (PublicNetworkError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    netloc = f"[{host}]" if ":" in host else host
    if explicit_port is not None:
        netloc += f":{port}"
    path = urllib.parse.quote(
        parsed.path, safe="/%:@!$&'()*+,;=-._~")
    query = urllib.parse.quote(
        parsed.query, safe="=&?/:;+,%@!$'()*-._~")
    fragment = urllib.parse.quote(
        parsed.fragment, safe="=&?/:;+,%@!$'()*-._~")
    return urllib.parse.urlunsplit((
        parsed.scheme, netloc, path, query, fragment))


def _caller_headers(
    headers: Mapping[str, str] | None,
) -> tuple[tuple[str, str], ...]:
    output: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_name, raw_value in (headers or {}).items():
        name = str(raw_name)
        value = str(raw_value)
        normalized = name.casefold()
        if (_HEADER_NAME.fullmatch(name) is None or normalized in seen
                or normalized in _FORBIDDEN_CALLER_HEADERS
                or any(ord(char) < 0x20 or ord(char) > 0x7e
                       for char in value)):
            raise ValueError("public HTTP header is invalid")
        seen.add(normalized)
        output.append((name, value))
    return tuple(output)


def _request_headers(
    caller_headers: tuple[tuple[str, str], ...],
    *,
    host: str,
    port: int,
    explicit_port: bool,
    body: bytes | None,
) -> tuple[tuple[str, str], ...]:
    output = list(caller_headers)
    host_header = f"[{host}]" if ":" in host else host
    if explicit_port:
        host_header += f":{port}"
    output.extend((
        ("Host", host_header),
        ("Accept-Encoding", "identity"),
        ("Connection", "close"),
    ))
    if body is not None:
        output.append(("Content-Length", str(len(body))))
    return tuple(output)


def request_public_http(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    body: bytes | None = None,
    timeout_seconds: float = 15.0,
    max_response_bytes: int = 2_000_000,
    allowed_content_types: frozenset[str] | None = None,
    max_redirects: int = 10,
    allow_redirects: bool = True,
    allow_https_downgrade: bool = False,
) -> PublicHTTPResponse:
    """Issue one bounded request whose DNS answer is pinned to the socket.

    Redirects are GET-only and are never followed when caller headers include
    credentials. Every admitted redirect is normalized, resolved, and pinned
    again before a new connection is made.
    """
    selected_method = str(method).upper()
    if selected_method not in {"GET", "POST"}:
        raise ValueError("public HTTP method is invalid")
    if body is not None and not isinstance(body, bytes):
        raise TypeError("public HTTP body must be bytes")
    if selected_method == "GET" and body is not None:
        raise ValueError("public HTTP GET cannot contain a body")
    if body is not None and len(body) > 2_000_000:
        raise ValueError("public HTTP request exceeded 2 MB")
    if (timeout_seconds <= 0 or timeout_seconds > 120
            or not 1 <= max_response_bytes <= 16_000_000
            or not 0 <= max_redirects <= 10):
        raise ValueError("public HTTP bounds are invalid")
    caller_headers = _caller_headers(headers)
    normalized_headers = {name.casefold() for name, _value in caller_headers}
    has_credentials = bool(normalized_headers & _SENSITIVE_HEADERS)
    current = normalize_public_http_url(url)
    visited: set[str] = set()
    for redirect_count in range(max_redirects + 1):
        if current in visited:
            raise PublicHTTPRedirectError("public HTTP redirect loop was blocked")
        visited.add(current)
        parsed = urllib.parse.urlsplit(current)
        host = str(parsed.hostname)
        explicit_port = parsed.port is not None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        raw = connect_public_stream(
            host, port, timeout_seconds=timeout_seconds)
        connection: socket.socket | ssl.SSLSocket = raw
        wire_response: http.client.HTTPResponse | None = None
        try:
            if parsed.scheme == "https":
                try:
                    connection = ssl.create_default_context().wrap_socket(
                        raw, server_hostname=host)
                except Exception:
                    raw.close()
                    raise
            connection.settimeout(timeout_seconds)
            target = parsed.path or "/"
            if parsed.query:
                target += "?" + parsed.query
            exact_headers = _request_headers(
                caller_headers, host=host, port=port,
                explicit_port=explicit_port, body=body)
            head = [f"{selected_method} {target} HTTP/1.1"]
            head.extend(f"{name}: {value}" for name, value in exact_headers)
            request = ("\r\n".join(head) + "\r\n\r\n").encode("ascii")
            connection.sendall(request)
            if body is not None:
                connection.sendall(body)
            wire_response = http.client.HTTPResponse(
                connection, method=selected_method)
            wire_response.begin()
            status = int(wire_response.status)
            if status in _REDIRECT_STATUSES:
                location = wire_response.headers.get("Location")
                wire_response.close()
                if (not allow_redirects or selected_method != "GET"
                        or has_credentials):
                    raise PublicHTTPRedirectError(
                        "public HTTP redirect was not authorized")
                if not location:
                    raise PublicHTTPRedirectError(
                        "public HTTP redirect omitted its destination")
                if redirect_count >= max_redirects:
                    raise PublicHTTPRedirectError(
                        "public HTTP redirect limit was exceeded")
                destination = normalize_public_http_url(
                    urllib.parse.urljoin(current, location))
                if (not allow_https_downgrade
                        and parsed.scheme == "https"
                        and urllib.parse.urlsplit(destination).scheme != "https"):
                    raise PublicHTTPRedirectError(
                        "public HTTP HTTPS downgrade was blocked")
                current = destination
                continue
            encodings = wire_response.headers.get_all(
                "Content-Encoding", failobj=[])
            if len(encodings) > 1:
                raise ValueError(
                    "public HTTP content encoding is ambiguous")
            content_encoding = str(
                encodings[0] if encodings else "identity"
            ).strip().casefold()
            if content_encoding not in {"", "identity"}:
                raise ValueError("encoded public HTTP content is not accepted")
            content_lengths = wire_response.headers.get_all(
                "Content-Length", failobj=[])
            if len(content_lengths) > 1:
                raise ValueError("public HTTP content length is ambiguous")
            if content_lengths:
                try:
                    announced = int(content_lengths[0])
                except ValueError as exc:
                    raise ValueError(
                        "public HTTP content length is invalid") from exc
                if announced < 0 or announced > max_response_bytes:
                    raise ValueError(
                        "public HTTP response exceeded its size limit")
            content_type = wire_response.headers.get_content_type()
            if (200 <= status <= 299 and allowed_content_types is not None
                    and content_type not in allowed_content_types):
                raise ValueError(
                    f"unsupported public HTTP content type: {content_type}")
            payload = wire_response.read(max_response_bytes + 1)
            if len(payload) > max_response_bytes:
                raise ValueError(
                    "public HTTP response exceeded its size limit")
            charset = wire_response.headers.get_content_charset() or "utf-8"
            wire_response.close()
            return PublicHTTPResponse(
                url=current, status=status, content_type=content_type,
                charset=charset, body=payload)
        finally:
            if wire_response is not None:
                wire_response.close()
            connection.close()
    raise PublicHTTPRedirectError("public HTTP redirect limit was exceeded")


__all__ = [
    "PublicHTTPError", "PublicHTTPRedirectError", "PublicHTTPResponse",
    "normalize_public_http_url", "request_public_http",
]
