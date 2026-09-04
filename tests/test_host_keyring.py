"""The secret store speaks each host keyring's exact dialect and fails closed."""

from __future__ import annotations

import base64
import subprocess
import unittest
from unittest import mock

from friday_host import host_keyring as secret_store
from friday_host.host import HostPlatform
from friday_host.host_keyring import SecretStore, SecretStoreUnavailable

LINUX = HostPlatform(os="linux", arch="x86_64")
MACOS = HostPlatform(os="macos", arch="aarch64")
WINDOWS = HostPlatform(os="windows", arch="x86_64")
VALUE = base64.urlsafe_b64encode(bytes(range(64)))


class RecordingRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        returncode, stdout = self.responses.pop(0)
        return subprocess.CompletedProcess(command, returncode, stdout, b"")


class LinuxSecretServiceTests(unittest.TestCase):
    def test_lookup_uses_the_pre_port_attributes(self) -> None:
        runner = RecordingRunner([(0, VALUE + b"\n")])
        store = SecretStore(host=LINUX, runner=runner)
        self.assertEqual(store.get_or_create("corrected-audio", 64), bytes(range(64)))
        self.assertEqual(runner.calls[0][0], [
            "secret-tool", "lookup", "application", "friday",
            "purpose", "corrected-audio"])
        self.assertEqual(len(runner.calls), 1, "an existing key is never re-stored")

    def test_missing_secret_is_minted_and_stored_via_stdin(self) -> None:
        runner = RecordingRunner([(0, b""), (0, b"")])
        store = SecretStore(host=LINUX, runner=runner)
        key = store.get_or_create("corrected-audio", 64)
        self.assertEqual(len(key), 64)
        command, kwargs = runner.calls[1]
        self.assertEqual(command, [
            "secret-tool", "store", "--label=Friday corrected-audio",
            "application", "friday", "purpose", "corrected-audio"])
        self.assertEqual(base64.urlsafe_b64decode(kwargs["input"].strip()), key)

    def test_refused_store_fails_closed(self) -> None:
        runner = RecordingRunner([(0, b""), (1, b"")])
        with self.assertRaises(SecretStoreUnavailable):
            SecretStore(host=LINUX, runner=runner).get_or_create("corrected-audio", 64)

    def test_missing_secret_tool_fails_closed(self) -> None:
        def missing(command, **kwargs):
            raise FileNotFoundError(command[0])

        with self.assertRaises(SecretStoreUnavailable):
            SecretStore(host=LINUX, runner=missing).get("corrected-audio")

    def test_short_or_invalid_stored_values_are_rejected(self) -> None:
        for stored in (b"not*base64!", base64.urlsafe_b64encode(b"short")):
            runner = RecordingRunner([(0, stored)])
            with self.subTest(stored=stored), self.assertRaises(SecretStoreUnavailable):
                SecretStore(host=LINUX, runner=runner).get_or_create("corrected-audio", 64)

    def test_delete(self) -> None:
        runner = RecordingRunner([(0, b"")])
        self.assertTrue(SecretStore(host=LINUX, runner=runner).delete("corrected-audio"))
        self.assertEqual(runner.calls[0][0][:2], ["secret-tool", "clear"])


class MacKeychainTests(unittest.TestCase):
    def test_lookup_and_store_keep_the_secret_out_of_argv(self) -> None:
        runner = RecordingRunner([(44, b""), (0, b"")])
        store = SecretStore(host=MACOS, runner=runner)
        key = store.get_or_create("corrected-audio", 64)
        self.assertEqual(len(key), 64)
        self.assertEqual(runner.calls[0][0], [
            "security", "find-generic-password", "-s", "friday",
            "-a", "corrected-audio", "-w"])
        command, kwargs = runner.calls[1]
        self.assertEqual(command, ["security", "-i"])
        script = kwargs["input"].decode()
        self.assertTrue(script.startswith(
            "add-generic-password -U -s friday -a corrected-audio "))
        self.assertIn(base64.urlsafe_b64encode(key).decode(), script)

    def test_existing_item_is_returned(self) -> None:
        runner = RecordingRunner([(0, VALUE + b"\n")])
        self.assertEqual(
            SecretStore(host=MACOS, runner=runner).get_or_create("corrected-audio", 64),
            bytes(range(64)))
        self.assertEqual(len(runner.calls), 1)

    def test_refused_keychain_fails_closed(self) -> None:
        runner = RecordingRunner([(44, b""), (1, b"")])
        with self.assertRaises(SecretStoreUnavailable):
            SecretStore(host=MACOS, runner=runner).get_or_create("corrected-audio", 64)


class WindowsCredentialManagerTests(unittest.TestCase):
    def test_credential_manager_round_trip_through_the_ctypes_layer(self) -> None:
        vault = {}

        def fake_get(purpose):
            return vault.get(purpose)

        def fake_set(purpose, value):
            vault[purpose] = value

        with mock.patch.object(secret_store, "_windows_get", fake_get), \
                mock.patch.object(secret_store, "_windows_set", fake_set):
            store = SecretStore(host=WINDOWS)
            first = store.get_or_create("corrected-audio", 64)
            second = store.get_or_create("corrected-audio", 64)
        self.assertEqual(first, second)
        self.assertEqual(list(vault), ["corrected-audio"])

    def test_target_name_is_namespaced(self) -> None:
        self.assertEqual(secret_store._windows_target("corrected-audio"),
                         "friday/corrected-audio")


class ValidationTests(unittest.TestCase):
    def test_non_printable_values_are_rejected_before_any_command(self) -> None:
        runner = RecordingRunner([])
        with self.assertRaises(ValueError):
            SecretStore(host=LINUX, runner=runner).set("x", b"bad\nvalue")
        self.assertEqual(runner.calls, [])


if __name__ == "__main__":
    unittest.main()
