"""The server filters its tool catalog and reports platform limits."""

from __future__ import annotations

import unittest
from unittest import mock

import server
from friday_host.host import HostPlatform
from friday_host.platform_capabilities import (BROWSER_TOOLS, OMARCHY_TOOLS,
                                               PROCESS_TOOLS,
                                               compute_capabilities)

MACOS = HostPlatform(os="macos", arch="aarch64", session="aqua", has_launchd=True)
LINUX = HostPlatform(
    os="linux", arch="x86_64", session="wayland", has_systemd_user=True,
    has_systemd_run=True, has_bwrap=True, has_hyprland=True, has_omarchy=True,
    has_managed_chromium=True, has_dev_shm=True, has_nvidia_smi=True,
    has_pdftotext=True, has_tesseract=True, has_magick=True)


class ToolCatalogGatingTests(unittest.TestCase):
    def test_macos_catalog_omits_linux_only_tools_and_reports_them(self) -> None:
        platform = compute_capabilities(MACOS)
        with mock.patch.object(server, "HOST", MACOS), \
                mock.patch.object(server, "PLATFORM", platform):
            schema_names = {item["function"]["name"] for item in server.current_tool_schema()}
            names = server.available_tool_names()
            inventory = {item["name"]: item for item in server.capability_inventory()}
            self.assertFalse(server._desktop_expected())
        for group in (PROCESS_TOOLS, OMARCHY_TOOLS, BROWSER_TOOLS):
            self.assertTrue(group.isdisjoint(schema_names), group & schema_names)
            self.assertTrue(group.isdisjoint(names))
        for name in ("read_file", "create_reminder", "fetch_news", "read_web",
                     "machine_read_document", "recall_memory", "desktop_notify"):
            self.assertIn(name, schema_names)
            self.assertIn(name, names)
        launch = inventory["machine_launch_process"]
        self.assertEqual(launch["status"], "unsupported_on_platform")
        self.assertEqual(launch["reason"], "requires_linux_systemd_run_and_bwrap")
        self.assertEqual(inventory["read_file"]["status"], "active")
        self.assertNotIn("reason", inventory["read_file"])

    def test_linux_workstation_catalog_is_unchanged(self) -> None:
        platform = compute_capabilities(LINUX, accelerator="cuda")
        with mock.patch.object(server, "HOST", LINUX), \
                mock.patch.object(server, "PLATFORM", platform):
            names = server.available_tool_names()
            inventory = {item["name"]: item for item in server.capability_inventory()}
        self.assertTrue(PROCESS_TOOLS <= names)
        self.assertTrue(OMARCHY_TOOLS <= names)
        self.assertTrue(BROWSER_TOOLS <= names)
        self.assertNotEqual(inventory["machine_launch_process"]["status"],
                            "unsupported_on_platform")

    def test_status_platform_block_describes_the_host(self) -> None:
        platform = compute_capabilities(MACOS)
        with mock.patch.object(server, "HOST", MACOS), \
                mock.patch.object(server, "PLATFORM", platform):
            block = server._platform_status()
        self.assertEqual(block["os"], "macos")
        self.assertEqual(block["arch"], "aarch64")
        self.assertEqual(block["lock_id"], "macos-arm64")
        self.assertFalse(block["capabilities"]["desktop"])
        self.assertIn("machine_launch_process", block["capabilities"]["unavailable_tools"])


if __name__ == "__main__":
    unittest.main()
