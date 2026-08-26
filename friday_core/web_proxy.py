"""Public-only SOCKS5 egress for Friday's managed browser.

The proxy owns DNS resolution and connects to a numeric address from the exact
validated answer set.  Chromium therefore never gets a validation-to-connect
DNS race and cannot use a rebinding answer to reach local infrastructure.
"""

from __future__ import annotations

import ipaddress
import re
import select
import socket
import socketserver
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence


_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_MAX_DNS_ANSWERS = 32
_SOCKS_VERSION = 5
_SOCKS_CONNECT = 1
_SOCKS_NO_AUTH = 0
_SOCKS_NO_ACCEPTABLE_METHOD = 0xFF
_SOCKS_ADDRESS_IPV4 = 1
_SOCKS_ADDRESS_DOMAIN = 3
_SOCKS_ADDRESS_IPV6 = 4


class PublicNetworkError(RuntimeError):
    """Base class for stable fail-closed public-network failures."""


class PublicNetworkDenied(PublicNetworkError):
    """The requested host or one of its answers is not globally routable."""


class PublicNetworkUnavailable(PublicNetworkError):
    """The public destination could not be resolved or connected."""


@dataclass(frozen=True)
class PublicEndpoint:
    family: int
    socket_type: int
    protocol: int
    sockaddr: tuple
    address: str


Resolver = Callable[..., Sequence[tuple]]


def _public_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise PublicNetworkDenied("public network address is invalid") from exc
    if (not address.is_global or address.is_private or address.is_loopback
            or address.is_link_local or address.is_multicast
            or address.is_reserved or address.is_unspecified):
        raise PublicNetworkDenied(
            "private or special network addresses are blocked")
    return address


def normalize_public_host(value: str) -> str:
    """Return a canonical IP/IDNA host without performing DNS resolution."""
    if not isinstance(value, str):
        raise PublicNetworkDenied("public network host is invalid")
    raw = value.strip().rstrip(".").casefold()
    if (not raw or len(raw) > 253 or raw == "localhost"
            or raw.endswith(".local") or any(ord(char) <= 0x20 for char in raw)):
        raise PublicNetworkDenied("local network hosts are blocked")
    try:
        return _public_ip(raw).compressed
    except PublicNetworkDenied:
        # A value that looks like an IP literal must never fall through to DNS
        # where alternate notations have resolver-specific interpretations.
        try:
            ipaddress.ip_address(raw)
        except ValueError:
            pass
        else:
            raise
    try:
        ascii_host = raw.encode("idna").decode("ascii")
    except (UnicodeError, ValueError) as exc:
        raise PublicNetworkDenied("public network host is invalid") from exc
    labels = ascii_host.split(".")
    if (not labels or any(_HOST_LABEL.fullmatch(label) is None
                          for label in labels)):
        raise PublicNetworkDenied("public network host is invalid")
    return ascii_host


def resolve_public_endpoints(
    host: str,
    port: int,
    *,
    resolver: Resolver = socket.getaddrinfo,
) -> tuple[PublicEndpoint, ...]:
    """Resolve once and reject the complete answer set if any IP is unsafe."""
    if (isinstance(port, bool) or not isinstance(port, int)
            or not 1 <= port <= 65535):
        raise PublicNetworkDenied("public network port is invalid")
    canonical = normalize_public_host(host)
    try:
        literal = ipaddress.ip_address(canonical)
    except ValueError:
        try:
            answers = resolver(
                canonical, port, socket.AF_UNSPEC, socket.SOCK_STREAM,
                socket.IPPROTO_TCP)
        except (OSError, socket.gaierror) as exc:
            raise PublicNetworkUnavailable(
                "public network host could not be resolved") from exc
    else:
        family = socket.AF_INET6 if literal.version == 6 else socket.AF_INET
        sockaddr = ((literal.compressed, port, 0, 0) if literal.version == 6
                    else (literal.compressed, port))
        answers = ((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                    sockaddr),)
    if (not answers or len(answers) > _MAX_DNS_ANSWERS):
        raise PublicNetworkUnavailable("public network DNS answer is invalid")
    endpoints: list[PublicEndpoint] = []
    seen: set[tuple[int, str, int, int]] = set()
    for answer in answers:
        if not isinstance(answer, tuple) or len(answer) != 5:
            raise PublicNetworkUnavailable("public network DNS answer is invalid")
        family, socket_type, protocol, _canonical_name, sockaddr = answer
        if (family not in {socket.AF_INET, socket.AF_INET6}
                or socket_type != socket.SOCK_STREAM
                or protocol not in {0, socket.IPPROTO_TCP}
                or not isinstance(sockaddr, tuple)
                or len(sockaddr) < 2):
            raise PublicNetworkUnavailable("public network DNS answer is invalid")
        address = _public_ip(str(sockaddr[0])).compressed
        observed_port = int(sockaddr[1])
        scope_id = int(sockaddr[3]) if family == socket.AF_INET6 \
            and len(sockaddr) >= 4 else 0
        if observed_port != port:
            raise PublicNetworkUnavailable("public network DNS answer is invalid")
        key = (int(family), address, observed_port, scope_id)
        if key in seen:
            continue
        seen.add(key)
        pinned = ((address, port, 0, scope_id) if family == socket.AF_INET6
                  else (address, port))
        endpoints.append(PublicEndpoint(
            family=int(family), socket_type=socket.SOCK_STREAM,
            protocol=socket.IPPROTO_TCP, sockaddr=pinned, address=address))
    if not endpoints:
        raise PublicNetworkUnavailable("public network DNS answer is invalid")
    return tuple(endpoints)


def connect_public_stream(
    host: str,
    port: int,
    *,
    timeout_seconds: float = 10.0,
    resolver: Resolver = socket.getaddrinfo,
) -> socket.socket:
    """Connect only to a numeric address from one fully validated DNS set."""
    if timeout_seconds <= 0 or timeout_seconds > 60:
        raise ValueError("public network timeout is invalid")
    endpoints = resolve_public_endpoints(host, port, resolver=resolver)
    last_error: OSError | None = None
    for endpoint in endpoints:
        connection = socket.socket(
            endpoint.family, endpoint.socket_type, endpoint.protocol)
        try:
            connection.settimeout(timeout_seconds)
            connection.connect(endpoint.sockaddr)
            peer = connection.getpeername()
            if (not isinstance(peer, tuple) or not peer
                    or _public_ip(str(peer[0])).compressed != endpoint.address):
                raise PublicNetworkDenied(
                    "public network peer identity changed")
            return connection
        except PublicNetworkDenied:
            connection.close()
            raise
        except OSError as exc:
            last_error = exc
            connection.close()
    raise PublicNetworkUnavailable(
        "public network destination is unavailable") from last_error


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    if length < 0 or length > 65_536:
        raise PublicNetworkDenied("SOCKS request is invalid")
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise PublicNetworkUnavailable("SOCKS client disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_reply(connection: socket.socket, code: int) -> None:
    connection.sendall(bytes((_SOCKS_VERSION, code, 0,
                              _SOCKS_ADDRESS_IPV4, 0, 0, 0, 0, 0, 0)))


def _relay(left: socket.socket, right: socket.socket, *,
           lifetime_seconds: float, idle_seconds: float) -> None:
    started = last_activity = time.monotonic()
    left.settimeout(min(10.0, idle_seconds))
    right.settimeout(min(10.0, idle_seconds))
    while True:
        now = time.monotonic()
        if (now - started >= lifetime_seconds
                or now - last_activity >= idle_seconds):
            return
        readable, _, _ = select.select(
            (left, right), (), (), min(1.0, idle_seconds))
        for source in readable:
            destination = right if source is left else left
            try:
                data = source.recv(64 * 1024)
            except (OSError, socket.timeout):
                return
            if not data:
                return
            try:
                destination.sendall(data)
            except (OSError, socket.timeout):
                return
            last_activity = time.monotonic()


class _ThreadingProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True
    block_on_close = True
    request_queue_size = 128

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stopping = threading.Event()
        self._connections_lock = threading.Lock()
        self._connections: set[socket.socket] = set()

    def track(self, connection: socket.socket) -> None:
        with self._connections_lock:
            if self.stopping.is_set():
                raise PublicNetworkUnavailable("public web proxy is stopping")
            self._connections.add(connection)

    def untrack(self, connection: socket.socket) -> None:
        with self._connections_lock:
            self._connections.discard(connection)

    def close_tracked(self) -> None:
        with self._connections_lock:
            connections = tuple(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass


class _SocksHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        if not server.connection_slots.acquire(blocking=False):
            server.record("rejected")
            return
        server.record("active", 1)
        upstream: socket.socket | None = None
        request_ready = False
        try:
            server.track(self.request)
            self.request.settimeout(server.handshake_timeout_seconds)
            version, method_count = _receive_exact(self.request, 2)
            if (version != _SOCKS_VERSION or method_count < 1
                    or method_count > 32):
                raise PublicNetworkDenied("SOCKS negotiation is invalid")
            methods = _receive_exact(self.request, method_count)
            if _SOCKS_NO_AUTH not in methods:
                self.request.sendall(bytes(
                    (_SOCKS_VERSION, _SOCKS_NO_ACCEPTABLE_METHOD)))
                server.record("blocked")
                return
            self.request.sendall(bytes((_SOCKS_VERSION, _SOCKS_NO_AUTH)))
            version, command, reserved, address_type = _receive_exact(
                self.request, 4)
            request_ready = True
            if version != _SOCKS_VERSION or reserved != 0:
                raise PublicNetworkDenied("SOCKS request is invalid")
            if command != _SOCKS_CONNECT:
                _send_reply(self.request, 7)
                server.record("blocked")
                return
            if address_type == _SOCKS_ADDRESS_IPV4:
                host = socket.inet_ntop(socket.AF_INET,
                                        _receive_exact(self.request, 4))
            elif address_type == _SOCKS_ADDRESS_IPV6:
                host = socket.inet_ntop(socket.AF_INET6,
                                        _receive_exact(self.request, 16))
            elif address_type == _SOCKS_ADDRESS_DOMAIN:
                size = _receive_exact(self.request, 1)[0]
                if not 1 <= size <= 253:
                    raise PublicNetworkDenied("SOCKS host is invalid")
                try:
                    host = _receive_exact(self.request, size).decode("ascii")
                except UnicodeDecodeError as exc:
                    raise PublicNetworkDenied("SOCKS host is invalid") from exc
            else:
                _send_reply(self.request, 8)
                server.record("blocked")
                return
            port = struct.unpack("!H", _receive_exact(self.request, 2))[0]
            upstream = connect_public_stream(
                host, port, timeout_seconds=server.connect_timeout_seconds,
                resolver=server.resolver)
            server.track(upstream)
            _send_reply(self.request, 0)
            server.record("accepted")
            _relay(self.request, upstream,
                   lifetime_seconds=server.connection_lifetime_seconds,
                   idle_seconds=server.connection_idle_seconds)
        except PublicNetworkDenied:
            if request_ready:
                try:
                    _send_reply(self.request, 2)
                except OSError:
                    pass
            server.record("blocked")
        except PublicNetworkUnavailable:
            if request_ready:
                try:
                    _send_reply(self.request, 4)
                except OSError:
                    pass
            server.record("failed")
        except (OSError, ValueError, struct.error):
            if request_ready:
                try:
                    _send_reply(self.request, 1)
                except OSError:
                    pass
            server.record("failed")
        finally:
            if upstream is not None:
                server.untrack(upstream)
                upstream.close()
            server.untrack(self.request)
            server.record("active", -1)
            server.connection_slots.release()


class PublicWebProxy:
    """Bounded loopback SOCKS5 proxy with privacy-safe aggregate status."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 9224,
        resolver: Resolver = socket.getaddrinfo,
        maximum_connections: int = 128,
        handshake_timeout_seconds: float = 10.0,
        connect_timeout_seconds: float = 10.0,
        connection_idle_seconds: float = 60.0,
        connection_lifetime_seconds: float = 300.0,
    ):
        if host != "127.0.0.1" or not 0 <= port <= 65535:
            raise ValueError("public web proxy listener is invalid")
        if (not 1 <= maximum_connections <= 1024
                or min(handshake_timeout_seconds, connect_timeout_seconds,
                       connection_idle_seconds,
                       connection_lifetime_seconds) <= 0
                or connection_lifetime_seconds < connection_idle_seconds):
            raise ValueError("public web proxy bounds are invalid")
        self.host = host
        self.port = int(port)
        self.resolver = resolver
        self.maximum_connections = int(maximum_connections)
        self.handshake_timeout_seconds = float(handshake_timeout_seconds)
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.connection_idle_seconds = float(connection_idle_seconds)
        self.connection_lifetime_seconds = float(connection_lifetime_seconds)
        self._lock = threading.RLock()
        self._metrics_lock = threading.Lock()
        self._server: _ThreadingProxyServer | None = None
        self._thread: threading.Thread | None = None
        self._metrics = {
            "accepted": 0, "blocked": 0, "failed": 0,
            "rejected": 0, "active": 0,
        }

    @property
    def listener_port(self) -> int:
        with self._lock:
            if self._server is None:
                return self.port
            return int(self._server.server_address[1])

    def _record(self, name: str, delta: int = 1) -> None:
        with self._metrics_lock:
            self._metrics[name] += delta

    def start(self) -> None:
        with self._lock:
            if self._server is not None or self._thread is not None:
                raise RuntimeError("public web proxy is already started")
            try:
                server = _ThreadingProxyServer(
                    (self.host, self.port), _SocksHandler)
            except OSError as exc:
                raise RuntimeError(
                    "public web proxy listener is unavailable") from exc
            server.resolver = self.resolver
            server.connection_slots = threading.BoundedSemaphore(
                self.maximum_connections)
            server.handshake_timeout_seconds = self.handshake_timeout_seconds
            server.connect_timeout_seconds = self.connect_timeout_seconds
            server.connection_idle_seconds = self.connection_idle_seconds
            server.connection_lifetime_seconds = \
                self.connection_lifetime_seconds
            server.record = self._record
            thread = threading.Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": 0.1},
                name="friday-public-web-proxy", daemon=True)
            self._server = server
            self._thread = thread
            thread.start()
            if not thread.is_alive():
                server.server_close()
                self._server = None
                self._thread = None
                raise RuntimeError("public web proxy did not start")

    def healthy(self) -> bool:
        with self._lock:
            return bool(
                self._server is not None and self._thread is not None
                and self._thread.is_alive()
                and self._server.socket.fileno() >= 0)

    def status(self) -> dict[str, int | bool]:
        with self._metrics_lock:
            metrics = dict(self._metrics)
        return {"healthy": self.healthy(), **metrics}

    def stop(self) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            if server is None and thread is None:
                return
        if server is not None:
            server.stopping.set()
            server.shutdown()
            server.close_tracked()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError("public web proxy did not stop")
        with self._lock:
            if self._server is server and self._thread is thread:
                self._server = None
                self._thread = None


__all__ = [
    "PublicEndpoint", "PublicNetworkDenied", "PublicNetworkError",
    "PublicNetworkUnavailable", "PublicWebProxy", "connect_public_stream",
    "normalize_public_host", "resolve_public_endpoints",
]
