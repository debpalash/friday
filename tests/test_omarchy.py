from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from friday_core.builtin_tools import OMARCHY_TOOL_NAMES as CATALOG_TOOLS
from friday_core.omarchy import (
    OMARCHY_TOOL_NAMES,
    OmarchyBindingError,
    OmarchyDesktopBroker,
)


class FakeOmarchyBackend:
    def __init__(self, root: Path):
        self.capture_root = root / "Pictures" / "Friday"
        self.theme = "Gta6"
        self.font = "JetBrainsMono Nerd Font"
        self.nightlight = False
        self.idle = "allow_idle"
        self.brightness_value = 50
        self.locked_value = False
        self.fingerprint = "a" * 64
        self.calls: list[tuple] = []

    def version(self):
        return "4.0.1-1"

    def command_fingerprint(self, tool_name):
        if tool_name not in OMARCHY_TOOL_NAMES:
            raise ValueError("unsupported")
        return self.fingerprint

    def themes(self):
        return ["Gta6", "Tokyo Night"]

    def current_theme(self):
        return self.theme

    def set_theme(self, theme):
        self.calls.append(("theme", theme))
        self.theme = theme

    def fonts(self):
        return ["JetBrainsMono Nerd Font", "Adwaita Mono"]

    def current_font(self):
        return self.font

    def set_font(self, font):
        self.calls.append(("font", font))
        self.font = font

    def nightlight_enabled(self):
        return self.nightlight

    def toggle_nightlight(self):
        self.calls.append(("nightlight",))
        self.nightlight = not self.nightlight

    def idle_mode(self):
        return self.idle

    def set_idle(self, mode):
        self.calls.append(("idle", mode))
        self.idle = mode

    def brightness(self):
        return self.brightness_value

    def set_brightness(self, percent):
        self.calls.append(("brightness", percent))
        self.brightness_value = percent

    def locked(self):
        return self.locked_value

    def lock(self):
        self.calls.append(("lock",))
        self.locked_value = True

    def start_browser_installer(self, browser):
        self.calls.append(("browser_installer", browser))

    def screenshot(self):
        self.calls.append(("screenshot",))
        self.capture_root.mkdir(parents=True, mode=0o700)
        path = self.capture_root / "screenshot-test.png"
        content = b"\x89PNG\r\n\x1a\n" + b"verified pixels"
        path.write_bytes(content)
        return {
            "path": str(path), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }


class OmarchyDesktopBrokerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.backend = FakeOmarchyBackend(self.root)
        self.broker = OmarchyDesktopBroker(
            self.backend, sleeper=lambda _seconds: None,
            action_timeout_seconds=0.1)

    def tearDown(self):
        self.temporary.cleanup()

    def test_catalog_and_broker_tool_sets_cannot_drift(self):
        self.assertEqual(CATALOG_TOOLS, OMARCHY_TOOL_NAMES)

    def test_status_is_structured_and_authoritatively_verified(self):
        receipt = self.broker.status()

        self.assertEqual(receipt["platform"], "omarchy")
        self.assertEqual(receipt["theme"]["current"], "Gta6")
        self.assertEqual(receipt["idle_mode"], "allow_idle")
        self.assertTrue(self.broker.verify_receipt(
            "machine_omarchy_status", json.dumps(receipt), {}, None))
        forged = dict(receipt)
        forged["brightness_percent"] = 100
        self.assertFalse(self.broker.verify_receipt(
            "machine_omarchy_status", json.dumps(forged), {}, None))

    def test_theme_and_font_are_bound_to_exact_installed_labels(self):
        theme_args = {"theme": "Tokyo Night"}
        theme_binding = self.broker.binding_for_action(
            "machine_omarchy_set_theme", theme_args)
        theme_receipt = self.broker.execute(
            "machine_omarchy_set_theme", theme_args,
            expected_binding=theme_binding)

        self.assertEqual(theme_receipt["state"], "Tokyo Night")
        self.assertEqual(self.backend.calls, [("theme", "Tokyo Night")])
        self.assertTrue(self.broker.verify_receipt(
            "machine_omarchy_set_theme", theme_receipt, theme_args,
            "act_theme", expected_binding=theme_binding))
        replay = self.broker.execute(
            "machine_omarchy_set_theme", theme_args,
            expected_binding=theme_binding)
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(self.backend.calls, [("theme", "Tokyo Night")])

        with self.assertRaisesRegex(ValueError, "not installed"):
            self.broker.binding_for_action(
                "machine_omarchy_set_font", {"font": "Imaginary Mono"})
        for value in (" Tokyo Night", "Tokyo Night ", 7):
            with self.subTest(theme=value), self.assertRaises(ValueError):
                self.broker.binding_for_action(
                    "machine_omarchy_set_theme", {"theme": value})

    def test_nightlight_idle_brightness_and_lock_reach_exact_states(self):
        cases = (
            ("machine_omarchy_set_nightlight", {"enabled": True}, "enabled"),
            ("machine_omarchy_set_idle", {"mode": "stay_awake"}, "stay_awake"),
            ("machine_omarchy_set_brightness", {"percent": 65}, "65"),
            ("machine_omarchy_lock", {}, "locked"),
        )
        for tool_name, args, expected in cases:
            with self.subTest(tool_name=tool_name):
                binding = self.broker.binding_for_action(tool_name, args)
                receipt = self.broker.execute(
                    tool_name, args, expected_binding=binding)
                self.assertEqual(receipt["state"], expected)
                self.assertTrue(self.broker.verify_receipt(
                    tool_name, receipt, args, "act_exact",
                    expected_binding=binding))
                reconciled = self.broker.reconciliation_receipt(binding)
                self.assertIsNotNone(reconciled)
                self.assertTrue(reconciled["idempotent_replay"])

    def test_boolean_and_brightness_inputs_reject_loose_values(self):
        for tool_name, args in (
            ("machine_omarchy_set_nightlight", {"enabled": 1}),
            ("machine_omarchy_set_brightness", {"percent": True}),
            ("machine_omarchy_set_brightness", {"percent": 0}),
            ("machine_omarchy_lock", {"extra": "value"}),
        ):
            with self.subTest(tool_name=tool_name, args=args), self.assertRaises(
                    ValueError):
                self.broker.binding_for_action(tool_name, args)

    def test_command_identity_change_invalidates_approved_binding(self):
        args = {"theme": "Tokyo Night"}
        binding = self.broker.binding_for_action(
            "machine_omarchy_set_theme", args)
        self.backend.fingerprint = "b" * 64

        with self.assertRaises(OmarchyBindingError):
            self.broker.execute(
                "machine_omarchy_set_theme", args,
                expected_binding=binding)

    def test_screenshot_is_private_hashed_and_never_blindly_reconciled(self):
        tool_name = "machine_omarchy_take_screenshot"
        binding = self.broker.binding_for_action(tool_name, {})
        receipt = self.broker.execute(
            tool_name, {}, expected_binding=binding)

        self.assertEqual(receipt["state"], "captured")
        self.assertTrue(receipt["capture"]["path"].endswith(".png"))
        self.assertTrue(self.broker.verify_receipt(
            tool_name, receipt, {}, "act_capture",
            expected_binding=binding))
        self.assertIsNone(self.broker.reconciliation_receipt(binding))
        forged = json.loads(json.dumps(receipt))
        forged["capture"]["sha256"] = "0" * 64
        self.assertFalse(self.broker.verify_receipt(
            tool_name, forged, {}, "act_capture",
            expected_binding=binding))

    def test_firefox_installer_is_exact_approved_and_never_replayed(self):
        tool_name = "machine_omarchy_install_browser"
        args = {"browser": "firefox"}
        binding = self.broker.binding_for_action(tool_name, args)
        receipt = self.broker.execute(
            tool_name, args, expected_binding=binding)

        self.assertEqual(receipt["state"], "installer_started")
        self.assertEqual(self.backend.calls, [("browser_installer", "firefox")])
        self.assertTrue(self.broker.verify_receipt(
            tool_name, receipt, args, "act_installer",
            expected_binding=binding))
        self.assertIsNone(self.broker.reconciliation_receipt(binding))
        with self.assertRaises(ValueError):
            self.broker.binding_for_action(tool_name, {"browser": "chrome"})


if __name__ == "__main__":
    unittest.main()
