from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from friday_core.live_runtime import (
    read_live_runtime,
    read_local_model_credential,
    resolve_application_root,
    resolve_state_dir,
    runtime_environment,
)


FINGERPRINT = "a" * 64


class LiveRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.config = self.root / "config" / "friday"
        self.state = self.root / "state"
        self.install = self.root / "install"
        self.qwen = self.root / "qwen"
        self.repo = self.root / "repo"
        for path in (self.config, self.state, self.install, self.qwen, self.repo):
            path.mkdir(parents=True, exist_ok=True)
        current = self.install / "current"
        current.mkdir()
        self.key_file = self.qwen / "api_key.txt"
        self._private_write(self.key_file, "friday-eval-test-key-012345\n")
        self._private_write(
            self.config / "friday.env",
            "\n".join((
                f"FRIDAY_INSTALL_ROOT='{self.install}'",
                f"FRIDAY_STATE_DIR='{self.state}'",
                f"FRIDAY_LLM_REPO='{self.qwen}'",
            )) + "\n",
        )
        self._private_write(
            self.state / "runtime-resolved.json",
            json.dumps({
                "local_base_url": "http://127.0.0.1:18021/v1",
                "served_model": "qwen-test",
                "fingerprint": FINGERPRINT,
                "native_vision": {"enabled": False},
            }),
        )
        self.environment = {
            "HOME": str(self.home),
            "FRIDAY_CONFIG_ROOT": str(self.config),
        }

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _private_write(path: Path, body: str) -> None:
        path.write_text(body)
        path.chmod(0o600)

    def test_installed_config_resolves_state_application_and_credential(self):
        values = runtime_environment(self.environment)
        self.assertEqual(values["FRIDAY_STATE_DIR"], str(self.state))
        self.assertEqual(
            resolve_state_dir(self.repo, self.environment), self.state)
        self.assertEqual(
            resolve_application_root(self.repo, self.environment),
            self.install / "current",
        )
        runtime = read_live_runtime(self.repo, environment=self.environment)
        self.assertEqual(runtime.state_dir, self.state)
        self.assertEqual(runtime.base_url, "http://127.0.0.1:18021/v1")
        self.assertEqual(runtime.model, "qwen-test")
        self.assertEqual(runtime.fingerprint, FINGERPRINT)
        self.assertEqual(
            read_local_model_credential(self.repo, self.environment),
            "friday-eval-test-key-012345",
        )

    def test_environment_overrides_installed_configuration(self):
        alternate = self.root / "alternate"
        alternate.mkdir()
        environment = self.environment | {"FRIDAY_STATE_DIR": str(alternate)}
        self.assertEqual(resolve_state_dir(self.repo, environment), alternate)

    def test_configuration_is_parsed_as_data_not_executed(self):
        marker = self.root / "executed"
        self._private_write(
            self.config / "friday.env",
            f"FRIDAY_STATE_DIR=$(touch {marker})\n",
        )
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            runtime_environment(self.environment)
        self.assertFalse(marker.exists())

    def test_public_or_symlinked_private_files_are_rejected(self):
        config_file = self.config / "friday.env"
        config_file.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, "private runtime file is invalid"):
            runtime_environment(self.environment)
        config_file.unlink()
        external = self.root / "external.env"
        self._private_write(external, "FRIDAY_STATE_DIR='/tmp/state'\n")
        config_file.symlink_to(external)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            runtime_environment(self.environment)

    def test_manifest_rejects_non_loopback_and_non_finite_json(self):
        manifest = self.state / "runtime-resolved.json"
        value = json.loads(manifest.read_text())
        value["local_base_url"] = "http://0.0.0.0:18021/v1"
        self._private_write(manifest, json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, "loopback"):
            read_live_runtime(self.repo, environment=self.environment)
        self._private_write(manifest, '{"value": NaN}')
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            read_live_runtime(self.repo, environment=self.environment)

    def test_native_vision_requirement_is_exact(self):
        with self.assertRaisesRegex(RuntimeError, "requires an active"):
            read_live_runtime(
                self.repo, require_native_vision=True,
                environment=self.environment,
            )
        value = json.loads((self.state / "runtime-resolved.json").read_text())
        value["native_vision"] = {"enabled": True, "max_side": 1536}
        self._private_write(
            self.state / "runtime-resolved.json", json.dumps(value))
        runtime = read_live_runtime(
            self.repo, require_native_vision=True,
            environment=self.environment,
        )
        self.assertEqual(runtime.native_vision_max_side, 1536)

    def test_direct_credential_override_remains_private_in_repr_free_api(self):
        environment = self.environment | {
            "FRIDAY_LOCAL_API_KEY": "fixture",
        }
        self.assertEqual(
            read_local_model_credential(self.repo, environment),
            "fixture",
        )


if __name__ == "__main__":
    unittest.main()
