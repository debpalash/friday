"""Engine and model installers verify every byte before publishing."""

from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import httpx

from friday_core import engine_assets
from ops import install_llama_server, install_local_model


def _tarball(files: dict[str, bytes], *, root: str = "llama-b0-bin-test") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, body in files.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(body)
            info.mode = 0o755 if name.endswith("llama-server") else 0o644
            archive.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return buffer.getvalue()


def _binary(payload: bytes, *, platform="linux", arch="x86_64", backend="cpu",
            name="llama-b0-bin-test.tar.gz", executable="llama-server", extra=()):
    return engine_assets.EngineBinary(
        platform=platform, arch=arch, backend=backend, tag="b0", name=name,
        url=f"https://example.invalid/{name}", size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(), executable=executable, extra=extra)


def _transport(bodies: dict[str, bytes]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = bodies.get(request.url.path.rsplit("/", 1)[-1])
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, content=body)
    return httpx.MockTransport(handler)


class LlamaServerInstallerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "runtime"
        self.cache = Path(self.temporary.name) / "cache"
        self.script = b"#!/bin/sh\necho 'llama-server test build'\n"

    def test_archive_is_verified_extracted_and_published(self) -> None:
        payload = _tarball({"llama-server": self.script, "LICENSE": b"MIT\n"})
        binary = _binary(payload)
        client = httpx.Client(transport=_transport({binary.name: payload}))
        executable = install_llama_server.install(binary, self.root, cache=self.cache,
                                                  client=client)
        self.assertEqual(executable, self.root / "llama-server" / "b0-cpu" / "llama-server")
        self.assertEqual(executable.read_bytes(), self.script)
        pin = (self.root / "llama-server" / "b0-cpu" / "FRIDAY_ENGINE_PIN").read_text()
        self.assertIn(binary.sha256, pin)
        # Idempotent: a verified install is reused without re-downloading.
        again = install_llama_server.install(binary, self.root, cache=self.cache,
                                             client=httpx.Client(transport=_transport({})))
        self.assertEqual(again, executable)

    def test_digest_mismatch_leaves_nothing_behind(self) -> None:
        payload = _tarball({"llama-server": self.script})
        binary = _binary(payload)
        tampered = payload[:-1] + bytes([payload[-1] ^ 0xFF])
        client = httpx.Client(transport=_transport({binary.name: tampered}))
        with self.assertRaisesRegex(RuntimeError, "pin mismatch"):
            install_llama_server.install(binary, self.root, cache=self.cache, client=client)
        self.assertEqual(list((self.root / "llama-server").iterdir()), [])

    def test_link_members_and_traversal_are_rejected(self) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            link = tarfile.TarInfo("root/evil"); link.type = tarfile.SYMTYPE; link.linkname = "/etc/passwd"
            archive.addfile(link)
        payload = buffer.getvalue()
        binary = _binary(payload)
        with self.assertRaisesRegex(RuntimeError, "link or special"):
            install_llama_server.install(binary, self.root, cache=self.cache,
                                         client=httpx.Client(transport=_transport({binary.name: payload})))
        payload = _zip({"../escape": b"x", "root/llama-server.exe": b"MZ"})
        binary = _binary(payload, platform="windows", name="llama-b0-bin-win.zip",
                         executable="llama-server.exe")
        with self.assertRaisesRegex(RuntimeError, "unsafe archive member"):
            install_llama_server.install(binary, self.root, cache=self.cache,
                                         client=httpx.Client(transport=_transport({binary.name: payload})))

    def test_windows_zip_with_cudart_extra_is_merged(self) -> None:
        main = _zip({"llama-server.exe": b"MZ main", "ggml.dll": b"dll"})
        extra = _zip({"cudart64_13.dll": b"cuda"})
        extra_digest = hashlib.sha256(extra).hexdigest()
        binary = _binary(main, platform="windows", backend="cuda",
                         name="llama-b0-bin-win-cuda.zip", executable="llama-server.exe",
                         extra=(("cudart.zip", "https://example.invalid/cudart.zip",
                                 len(extra), extra_digest),))
        with mock.patch.object(install_llama_server.subprocess, "run") as run:
            run.return_value = mock.Mock(stdout="version b0", stderr="")
            executable = install_llama_server.install(
                binary, self.root, cache=self.cache,
                client=httpx.Client(transport=_transport({binary.name: main, "cudart.zip": extra})))
        self.assertTrue(executable.is_file())
        self.assertTrue((executable.parent / "cudart64_13.dll").is_file())
        self.assertTrue((executable.parent / "ggml.dll").is_file())


class ModelInstallerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model_root = Path(self.temporary.name) / "models"
        self.files = {"config.json": b'{"a": 1}', "model.safetensors": b"weights" * 100}
        self.asset = engine_assets.ModelAsset(
            key="qwen3-test-mlx-4bit", engine="mlx_lm", size_label="8b",
            repo="mlx-community/Test", revision="b" * 40, license="Apache-2.0",
            files=tuple((name, len(body), hashlib.sha256(body).hexdigest())
                        for name, body in self.files.items()),
            weights_bytes=700, kv_bytes_per_token=1, entry="")

    def test_files_are_verified_and_published_atomically(self) -> None:
        client = httpx.Client(transport=_transport(self.files))
        target = install_local_model.install(self.asset, self.model_root, client=client)
        self.assertEqual(target, self.model_root / f"qwen3-test-mlx-4bit-{'b' * 8}")
        for name, body in self.files.items():
            self.assertEqual((target / name).read_bytes(), body)
        self.assertIn("asset=qwen3-test-mlx-4bit", (target / "FRIDAY_MODEL_PIN").read_text())
        self.assertEqual(install_local_model.install(
            self.asset, self.model_root, client=httpx.Client(transport=_transport({}))), target)

    def test_wrong_bytes_are_refused_before_publication(self) -> None:
        bodies = dict(self.files); bodies["model.safetensors"] = b"other"
        client = httpx.Client(transport=_transport(bodies))
        with self.assertRaisesRegex(RuntimeError, "exceeded its pin|pin mismatch"):
            install_local_model.install(self.asset, self.model_root, client=client)
        self.assertEqual([p.name for p in self.model_root.iterdir()], [])

    def test_real_assets_have_download_urls_in_the_pinned_repositories(self) -> None:
        for asset in engine_assets.model_assets():
            self.assertTrue(asset.repo.startswith(("Qwen/", "mlx-community/")))
            self.assertTrue(all(not name.startswith("/") for name, _s, _d in asset.files))


if __name__ == "__main__":
    unittest.main()
