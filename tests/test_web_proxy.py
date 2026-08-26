import socket
import time
import unittest
from unittest import mock

from friday_core.web_proxy import (
    PublicNetworkDenied, PublicNetworkUnavailable, PublicWebProxy,
    connect_public_stream, resolve_public_endpoints,
)


def _answer(address: str, port: int):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sockaddr = ((address, port, 0, 0) if family == socket.AF_INET6
                else (address, port))
    return (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    chunks = []
    while sum(map(len, chunks)) < length:
        chunk = connection.recv(length - sum(map(len, chunks)))
        if not chunk:
            raise AssertionError("proxy disconnected before the expected reply")
        chunks.append(chunk)
    return b"".join(chunks)


class PublicEndpointPolicyTests(unittest.TestCase):
    def test_complete_dns_set_must_be_public(self):
        public = lambda *_args: [
            _answer("93.184.216.34", 443),
            _answer("2606:2800:220:1:248:1893:25c8:1946", 443),
        ]
        endpoints = resolve_public_endpoints(
            "example.com", 443, resolver=public)
        self.assertEqual(len(endpoints), 2)

        mixed = lambda *_args: [
            _answer("93.184.216.34", 443),
            _answer("127.0.0.1", 443),
        ]
        with self.assertRaises(PublicNetworkDenied):
            resolve_public_endpoints("example.com", 443, resolver=mixed)

    def test_private_literals_never_reach_dns(self):
        resolver = mock.Mock()
        for host in (
                "127.0.0.1", "10.1.2.3", "192.168.1.1", "169.254.1.1",
                "::1", "fc00::1", "fe80::1"):
            with self.subTest(host=host), self.assertRaises(
                    PublicNetworkDenied):
                resolve_public_endpoints(host, 443, resolver=resolver)
        resolver.assert_not_called()

    def test_connect_uses_one_dns_read_and_the_pinned_numeric_address(self):
        resolver = mock.Mock(return_value=[_answer("93.184.216.34", 443)])
        connection = mock.Mock()
        connection.getpeername.return_value = ("93.184.216.34", 443)
        with mock.patch("friday_core.web_proxy.socket.socket",
                        return_value=connection) as create:
            result = connect_public_stream(
                "example.com", 443, resolver=resolver)
        self.assertIs(result, connection)
        self.assertEqual(resolver.call_count, 1)
        connection.connect.assert_called_once_with(("93.184.216.34", 443))
        create.assert_called_once_with(
            socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)

    def test_peer_change_fails_closed(self):
        resolver = mock.Mock(return_value=[_answer("93.184.216.34", 443)])
        connection = mock.Mock()
        connection.getpeername.return_value = ("127.0.0.1", 443)
        with (mock.patch("friday_core.web_proxy.socket.socket",
                         return_value=connection),
              self.assertRaises(PublicNetworkDenied)):
            connect_public_stream("example.com", 443, resolver=resolver)
        connection.close.assert_called_once()

    def test_empty_oversized_and_malformed_dns_answers_fail_closed(self):
        for answers in ([], [_answer("93.184.216.34", 443)] * 33,
                        [(socket.AF_UNIX, socket.SOCK_STREAM, 0, "", ("x", 443))]):
            with self.subTest(size=len(answers)), self.assertRaises(
                    PublicNetworkUnavailable):
                resolve_public_endpoints(
                    "example.com", 443, resolver=lambda *_args, a=answers: a)


class PublicWebProxyTests(unittest.TestCase):
    @staticmethod
    def _connect(proxy: PublicWebProxy) -> socket.socket:
        client = socket.create_connection(
            ("127.0.0.1", proxy.listener_port), timeout=2)
        client.settimeout(2)
        client.sendall(b"\x05\x01\x00")
        if _receive_exact(client, 2) != b"\x05\x00":
            raise AssertionError("SOCKS no-auth negotiation failed")
        return client

    def test_domain_connect_is_tunneled_without_destination_disclosure(self):
        proxy = PublicWebProxy(
            port=0, connection_idle_seconds=2,
            connection_lifetime_seconds=3)
        proxy.start()
        upstream_proxy, upstream_server = socket.socketpair()
        upstream_server.settimeout(2)
        try:
            with mock.patch(
                    "friday_core.web_proxy.connect_public_stream",
                    return_value=upstream_proxy) as connect:
                client = self._connect(proxy)
                host = b"example.com"
                client.sendall(
                    b"\x05\x01\x00\x03" + bytes((len(host),)) + host
                    + b"\x01\xbb")
                self.assertEqual(_receive_exact(client, 10)[:2], b"\x05\x00")
                client.sendall(b"request-through-tunnel")
                self.assertEqual(
                    upstream_server.recv(64), b"request-through-tunnel")
                upstream_server.sendall(b"response-through-tunnel")
                self.assertEqual(
                    client.recv(64), b"response-through-tunnel")
                client.close()
                connect.assert_called_once_with(
                    "example.com", 443, timeout_seconds=10.0,
                    resolver=proxy.resolver)
            for _ in range(50):
                if proxy.status()["active"] == 0:
                    break
                time.sleep(0.01)
            status = proxy.status()
            self.assertEqual(status["accepted"], 1)
            self.assertNotIn("host", status)
            self.assertNotIn("destination", status)
        finally:
            upstream_server.close()
            proxy.stop()
        self.assertFalse(proxy.healthy())

    def test_loopback_and_udp_association_are_blocked(self):
        proxy = PublicWebProxy(port=0)
        proxy.start()
        try:
            client = self._connect(proxy)
            client.sendall(
                b"\x05\x01\x00\x01\x7f\x00\x00\x01\x21\x38")
            self.assertEqual(_receive_exact(client, 10)[1], 2)
            client.close()

            udp = self._connect(proxy)
            udp.sendall(
                b"\x05\x03\x00\x01\x08\x08\x08\x08\x00\x35")
            self.assertEqual(_receive_exact(udp, 10)[1], 7)
            udp.close()
        finally:
            proxy.stop()

    def test_listener_collision_fails_closed_and_stop_is_idempotent(self):
        owner = PublicWebProxy(port=0)
        owner.start()
        contender = PublicWebProxy(port=owner.listener_port)
        try:
            with self.assertRaisesRegex(RuntimeError, "listener"):
                contender.start()
        finally:
            contender.stop()
            owner.stop()
            owner.stop()

    def test_stop_closes_active_tunnels_without_waiting_for_idle_timeout(self):
        proxy = PublicWebProxy(
            port=0, connection_idle_seconds=60,
            connection_lifetime_seconds=300)
        proxy.start()
        upstream_proxy, upstream_server = socket.socketpair()
        upstream_server.settimeout(2)
        client = None
        try:
            with mock.patch(
                    "friday_core.web_proxy.connect_public_stream",
                    return_value=upstream_proxy):
                client = self._connect(proxy)
                host = b"example.com"
                client.sendall(
                    b"\x05\x01\x00\x03" + bytes((len(host),)) + host
                    + b"\x01\xbb")
                self.assertEqual(_receive_exact(client, 10)[1], 0)
                started = time.monotonic()
                proxy.stop()
                self.assertLess(time.monotonic() - started, 2.0)
        finally:
            if client is not None:
                client.close()
            upstream_server.close()
            proxy.stop()


if __name__ == "__main__":
    unittest.main()
