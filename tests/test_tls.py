import hashlib
import os
import ssl
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import supervisor
from friday_core.tls import (
    TLSBootstrapError,
    ensure_tls_material,
    normalize_tls_hosts,
)


class TLSBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_bootstrap_is_private_exact_and_idempotent(self) -> None:
        first = ensure_tls_material(
            self.root, ["192.168.1.158", "omarchy.local"])
        second = ensure_tls_material(
            self.root, ["omarchy.local", "192.168.1.158"])

        self.assertEqual(first, second)
        self.assertEqual(first.hosts, (
            "127.0.0.1", "192.168.1.158", "::1", "localhost",
            "omarchy.local",
        ))
        for path in (
            self.root / "tls", self.root / "tls" / first.generation,
        ):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
        for path in (
            first.certfile, first.keyfile, first.cafile,
            self.root / "tls" / "ca-key.pem",
            self.root / "tls" / "active.json",
            self.root / "tls-bootstrap.lock",
        ):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_host_change_renews_only_leaf_and_preserves_trust_anchor(self):
        first = ensure_tls_material(self.root, ["192.168.1.158"])
        ca_before = self._digest(first.cafile)
        first_certificate = self._digest(first.certfile)

        second = ensure_tls_material(
            self.root, ["192.168.1.158", "friday.local"])

        self.assertNotEqual(second.generation, first.generation)
        self.assertNotEqual(self._digest(second.certfile), first_certificate)
        self.assertEqual(self._digest(second.cafile), ca_before)
        self.assertTrue(first.certfile.exists())
        self.assertIn("friday.local", second.hosts)

    def test_tampered_material_fails_without_rotating_local_ca(self) -> None:
        material = ensure_tls_material(self.root, ["192.168.1.158"])
        ca_before = self._digest(material.cafile)
        active_before = (self.root / "tls" / "active.json").read_bytes()
        with material.keyfile.open("ab") as stream:
            stream.write(b"tamper")

        with self.assertRaisesRegex(TLSBootstrapError, "digest changed"):
            ensure_tls_material(self.root, ["192.168.1.158"])

        self.assertEqual(self._digest(material.cafile), ca_before)
        self.assertEqual(
            (self.root / "tls" / "active.json").read_bytes(), active_before)

    def test_symlinked_tls_root_is_rejected(self) -> None:
        target = self.root / "target"
        target.mkdir(mode=0o700)
        (self.root / "tls").symlink_to(target, target_is_directory=True)

        with self.assertRaisesRegex(TLSBootstrapError, "identity is unsafe"):
            ensure_tls_material(self.root, ["192.168.1.158"])

        self.assertEqual(list(target.iterdir()), [])

    def test_concurrent_bootstrap_publishes_one_active_generation(self) -> None:
        def bootstrap(_index: int):
            return ensure_tls_material(self.root, ["192.168.1.158"])

        with ThreadPoolExecutor(max_workers=8) as pool:
            materials = list(pool.map(bootstrap, range(16)))

        self.assertEqual(len({item.generation for item in materials}), 1)
        generations = [path for path in (self.root / "tls").iterdir()
                       if path.name.startswith("generation_")]
        self.assertEqual(len(generations), 1)
        self.assertEqual(
            len([path for path in self.root.iterdir()
                 if path.name.startswith(".tls-stage-")]),
            0,
        )

    def test_host_normalization_rejects_ambiguous_names(self) -> None:
        self.assertEqual(
            normalize_tls_hosts(["EXAMPLE.local", "2001:0db8::1"]),
            ("127.0.0.1", "2001:db8::1", "::1", "example.local",
             "localhost"),
        )
        for invalid in (
            "example.local.", "bad_host", "*.example.local",
            "example.local:8500", "exa%6dple.local", "éxample.local",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(
                    TLSBootstrapError):
                normalize_tls_hosts([invalid])

    def test_group_writable_state_root_fails_closed(self) -> None:
        self.root.chmod(0o770)
        with self.assertRaisesRegex(TLSBootstrapError, "identity is unsafe"):
            ensure_tls_material(self.root, ["192.168.1.158"])
        self.assertFalse((self.root / "tls").exists())

    def test_supervisor_health_completes_verified_loopback_tls_handshake(self):
        material = ensure_tls_material(self.root, ["192.168.1.158"])

        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200 if self.path == "/healthz" else 404)
                self.end_headers()

            def log_message(self, _format, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), HealthHandler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(material.certfile, material.keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"https://127.0.0.1:{server.server_port}/healthz"
        try:
            with mock.patch.multiple(
                    supervisor, STATE=self.root, FRIDAY_HEALTH_URL=url):
                self.assertTrue(supervisor.healthy(url, timeout=2))
                self.assertFalse(supervisor.healthy(
                    f"https://localhost:{server.server_port}/healthz",
                    timeout=2))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
