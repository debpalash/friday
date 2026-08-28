"""Pure parsing and admission policy for Friday's local control plane."""

from __future__ import annotations

import hmac
import urllib.parse
from collections.abc import Mapping

from .controller_auth import ControllerAuthError, normalize_https_origin


def valid_host(value: str | None, allowed_hosts: frozenset[str]) -> bool:
    if not value:
        return False
    try:
        parsed = urllib.parse.urlsplit("//" + value)
    except ValueError:
        return False
    return (
        parsed.username is None
        and parsed.password is None
        and (parsed.hostname or "").lower() in allowed_hosts
    )


def valid_origin(value: str | None, allowed_origins: frozenset[str]) -> bool:
    """Allow non-browser clients to omit Origin; validate every supplied one."""
    return value is None or value.lower().rstrip("/") in allowed_origins


def valid_control_token(value: str | None, expected: str) -> bool:
    return bool(value) and hmac.compare_digest(value, expected)


def websocket_session_token(protocol_header: str | None) -> str | None:
    for protocol in (protocol_header or "").split(","):
        value = protocol.strip()
        if value.startswith("session."):
            return value.removeprefix("session.")
    return None


def controller_origin(headers: Mapping[str, str]) -> str:
    supplied = headers.get("origin")
    if supplied:
        return normalize_https_origin(supplied)
    host = headers.get("host")
    if not host:
        raise ControllerAuthError()
    return normalize_https_origin("https://" + host)


def bearer_session_token(value: str | None) -> str | None:
    if not value or not value.startswith("Bearer "):
        return None
    token = value.removeprefix("Bearer ")
    return token if token and not any(char.isspace() for char in token) else None
