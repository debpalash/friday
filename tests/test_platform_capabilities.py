"""Unsupported action classes are reported per platform, never hidden."""

from __future__ import annotations

import unittest

from friday_core import builtin_tools
from friday_host import platform_capabilities as caps
from friday_host.host import HostPlatform

LINUX_WORKSTATION = HostPlatform(
    os="linux", arch="x86_64", session="wayland", has_systemd_user=True,
    has_systemd_run=True, has_bwrap=True, has_hyprland=True, has_omarchy=True,
    has_managed_chromium=True, has_dev_shm=True, has_nvidia_smi=True,
    has_pdftotext=True, has_tesseract=True, has_magick=True)
LINUX_MINIMAL = HostPlatform(os="linux", arch="x86_64", session="x11",
                             has_systemd_user=True, has_dev_shm=True)
MACOS = HostPlatform(os="macos", arch="aarch64", session="aqua", has_launchd=True)
WINDOWS = HostPlatform(os="windows", arch="x86_64", session="win32",
                       has_task_scheduler=True, has_nvidia_smi=True)
HEADLESS = HostPlatform(os="linux", arch="x86_64", session="headless")


class ToolSetTests(unittest.TestCase):
    def test_tool_name_sets_match_the_application_catalog(self) -> None:
        self.assertEqual(caps.PROCESS_TOOLS, builtin_tools.PROCESS_TOOL_NAMES)
        self.assertEqual(caps.DESKTOP_TOOLS | caps.OMARCHY_TOOLS,
                         builtin_tools.DESKTOP_TOOL_NAMES)
        self.assertEqual(caps.OMARCHY_TOOLS, builtin_tools.OMARCHY_TOOL_NAMES)
        catalog = builtin_tools.BUILTIN_TOOL_NAMES
        for group in (caps.BROWSER_TOOLS, caps.OCR_TOOLS, caps.VISION_TOOLS,
                      caps.VOICE_PROFILE_TOOLS):
            self.assertTrue(group <= catalog, group - catalog)


class LinuxTests(unittest.TestCase):
    def test_full_workstation_supports_everything(self) -> None:
        result = caps.compute_capabilities(LINUX_WORKSTATION, accelerator="cuda")
        self.assertTrue(result.desktop and result.omarchy and result.managed_processes
                        and result.managed_browser and result.sandboxed_documents
                        and result.ocr and result.native_vision_host
                        and result.voice_profiles and result.notifications)
        self.assertEqual(dict(result.unavailable_tools), {})
        self.assertEqual(dict(result.reasons), {})
        self.assertEqual(result.document_formats[0], "pdf")

    def test_minimal_linux_reports_each_missing_dependency(self) -> None:
        result = caps.compute_capabilities(LINUX_MINIMAL)
        self.assertFalse(result.managed_processes)
        self.assertEqual(result.reasons["managed_processes"], caps.REASON_LINUX_SYSTEMD)
        self.assertFalse(result.desktop)
        self.assertEqual(result.reasons["desktop"], caps.REASON_LINUX_HYPRLAND)
        self.assertFalse(result.sandboxed_documents)
        self.assertNotIn("pdf", result.document_formats)
        self.assertEqual(result.unavailable_tools["machine_launch_process"],
                         caps.REASON_LINUX_SYSTEMD)
        self.assertEqual(result.unavailable_tools["create_voice_profile"],
                         caps.REASON_CUDA_VOICE)

    def test_disabled_desktop_mode_is_reported_as_configuration(self) -> None:
        result = caps.compute_capabilities(LINUX_WORKSTATION, desktop_mode="disabled",
                                           accelerator="cuda")
        self.assertFalse(result.desktop)
        self.assertEqual(result.reasons["desktop"], "disabled_by_configuration")
        self.assertEqual(result.unavailable_tools["machine_omarchy_set_theme"],
                         "disabled_by_configuration")
        self.assertTrue(result.managed_processes)

    def test_headless_session_reports_desktop_conveniences(self) -> None:
        result = caps.compute_capabilities(HEADLESS)
        self.assertFalse(result.notifications)
        self.assertEqual(result.reasons["notifications"], caps.REASON_HEADLESS)


class NonLinuxTests(unittest.TestCase):
    def test_macos_and_windows_keep_the_portable_core(self) -> None:
        for host in (MACOS, WINDOWS):
            with self.subTest(host=host.os):
                result = caps.compute_capabilities(host)
                self.assertFalse(result.desktop or result.omarchy
                                 or result.managed_processes or result.managed_browser
                                 or result.sandboxed_documents or result.ocr
                                 or result.native_vision_host or result.voice_profiles)
                self.assertTrue(result.notifications and result.clipboard
                                and result.open_local)
                unavailable = set(result.unavailable_tools)
                self.assertTrue(caps.PROCESS_TOOLS <= unavailable)
                self.assertTrue(caps.DESKTOP_TOOLS <= unavailable)
                self.assertTrue(caps.OMARCHY_TOOLS <= unavailable)
                self.assertTrue(caps.BROWSER_TOOLS <= unavailable)
                self.assertTrue(caps.OCR_TOOLS | caps.VISION_TOOLS <= unavailable)
                for name in ("read_file", "write_file", "create_reminder",
                             "fetch_news", "read_web", "machine_read_document",
                             "recall_memory", "desktop_notify", "clipboard_read"):
                    self.assertNotIn(name, unavailable)
                self.assertEqual(result.reasons["desktop"], caps.REASON_LINUX_HYPRLAND)

    def test_required_desktop_mode_fails_closed_off_linux(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsupported on macos"):
            caps.compute_capabilities(MACOS, desktop_mode="required")

    def test_status_payload_is_json_friendly(self) -> None:
        import json

        payload = caps.compute_capabilities(WINDOWS).to_status()
        json.dumps(payload)
        self.assertIn("unavailable_tools", payload)
        self.assertEqual(payload["document_formats"][0], "txt")

    def test_invalid_desktop_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            caps.compute_capabilities(MACOS, desktop_mode="sometimes")


if __name__ == "__main__":
    unittest.main()
