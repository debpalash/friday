"""The in-process ciphers reproduce what the openssl command line produced."""

from __future__ import annotations

import base64
import json
import secrets
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from friday_core.evidence import CorrectedAudioStore
from friday_core.local_cipher import aes256_ctr
from friday_core.step_payloads import StepPayloadCipher

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "crypto"
OPENSSL = shutil.which("openssl")


class Aes256CtrTests(unittest.TestCase):
    def test_round_trip_and_argument_validation(self) -> None:
        key, iv = secrets.token_bytes(32), secrets.token_bytes(16)
        data = secrets.token_bytes(1000)
        self.assertEqual(aes256_ctr(key, iv, aes256_ctr(key, iv, data)), data)
        self.assertEqual(aes256_ctr(key, iv, b""), b"")
        with self.assertRaises(ValueError):
            aes256_ctr(key[:16], iv, data)
        with self.assertRaises(ValueError):
            aes256_ctr(key, iv[:8], data)

    @unittest.skipUnless(OPENSSL, "environment: openssl is not installed")
    def test_matches_openssl_enc_including_counter_carry(self) -> None:
        key = secrets.token_bytes(32)
        # An IV ending in 0xff..ff forces the 128-bit counter to carry across
        # blocks, which is where CTR implementations disagree.
        iv = bytes.fromhex("0102030405060708ffffffffffffffff")
        data = secrets.token_bytes(4096 + 7)
        expected = subprocess.run(
            [OPENSSL, "enc", "-aes-256-ctr", "-K", key.hex(), "-iv", iv.hex()],
            input=data, capture_output=True, timeout=10, check=True).stdout
        self.assertEqual(aes256_ctr(key, iv, data), expected)


class StepPayloadFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(
            (FIXTURES / "step_payload_openssl_v1.json").read_text())
        self.key = base64.b64decode(self.fixture["key_b64"])
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cipher = StepPayloadCipher(
            Path(self.temporary.name) / "unused.key", key=self.key)

    def test_payload_sealed_by_openssl_still_opens(self) -> None:
        payload = json.dumps(self.fixture["payload"])
        self.assertEqual(
            self.cipher.open(payload, context=self.fixture["context"]),
            self.fixture["plaintext"])
        with self.assertRaisesRegex(RuntimeError, "authentication failed"):
            self.cipher.open(payload, context="another context")

    def test_resealing_with_the_recorded_iv_is_byte_identical(self) -> None:
        recorded = self.fixture["payload"]
        iv = base64.b64decode(recorded["iv"])
        resealed = json.loads(self.cipher._seal_with_iv(
            self.fixture["plaintext"], context=self.fixture["context"], iv=iv))
        self.assertEqual(resealed, recorded)

    def test_fresh_seal_round_trips_with_a_random_iv(self) -> None:
        first = self.cipher.seal({"a": 1}, context="c")
        second = self.cipher.seal({"a": 1}, context="c")
        self.assertNotEqual(json.loads(first)["iv"], json.loads(second)["iv"])
        self.assertEqual(self.cipher.open(first, context="c"), {"a": 1})


class CorrectedAudioFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(
            (FIXTURES / "corrected_audio_openssl_v1.json").read_text())
        self.key = base64.b64decode(self.fixture["key_b64"])
        self.pcm = base64.b64decode(self.fixture["pcm_b64"])

    def test_artifact_written_by_openssl_still_decrypts(self) -> None:
        self.assertEqual(
            CorrectedAudioStore.decrypt(self.fixture["artifact"], self.key),
            self.pcm)
        tampered = dict(self.fixture["artifact"])
        tampered["ciphertext"] = base64.b64encode(
            bytes(reversed(base64.b64decode(tampered["ciphertext"])))).decode()
        with self.assertRaisesRegex(ValueError, "authentication failed"):
            CorrectedAudioStore.decrypt(tampered, self.key)

    def test_storing_with_the_recorded_iv_reproduces_the_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CorrectedAudioStore(temporary, key_provider=lambda: self.key)
            recorded = self.fixture["artifact"]
            path = store.store(
                "again", self.pcm, recorded["metadata"],
                iv=base64.b64decode(recorded["iv"]))
            self.assertEqual(json.loads(Path(path).read_text()), recorded)
            fresh = json.loads(Path(store.store(
                "fresh", self.pcm, recorded["metadata"])).read_text())
            self.assertNotEqual(fresh["iv"], recorded["iv"])
            self.assertEqual(CorrectedAudioStore.decrypt(fresh, self.key), self.pcm)


if __name__ == "__main__":
    unittest.main()
