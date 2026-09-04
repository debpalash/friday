"""Host detection describes Linux, macOS, and Windows from injected probes."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from friday_host import host as host_module
from friday_host.host import HostPlatform, detect_host, normalize_arch, normalize_os


def _which(available: set[str]):
    return lambda name: f"/usr/bin/{name}" if name in available else None


def _paths(present: set[str]):
    return lambda path: str(path) in present


def _systemd(returncode: int):
    def run(command, **_kwargs):
        return subprocess.CompletedProcess(command, returncode, "", "")
    return run


class HostDetectionTests(unittest.TestCase):
    def test_linux_hyprland_workstation(self) -> None:
        present = {"/usr/share/omarchy/bin/omarchy", "/usr/bin/bwrap",
                   "/usr/lib/chromium/chromium", "/dev/shm", "/run/user/1000/hypr",
                   "/usr/bin/pdftotext", "/usr/bin/tesseract", "/usr/bin/magick"}
        detected = detect_host(
            {"WAYLAND_DISPLAY": "wayland-1"}, platform_name="linux",
            machine="x86_64", uid=1000,
            which=_which({"systemd-run", "nvidia-smi"}),
            exists=_paths(present), is_dir=_paths(present), is_file=_paths(present),
            read_text=lambda _path: "Linux version 6.1",
            run=_systemd(0))
        self.assertEqual(detected.os, "linux")
        self.assertEqual(detected.session, "wayland")
        self.assertTrue(detected.has_systemd_user)
        self.assertTrue(detected.has_hyprland)
        self.assertTrue(detected.has_omarchy)
        self.assertTrue(detected.has_bwrap)
        self.assertTrue(detected.has_managed_chromium)
        self.assertTrue(detected.has_nvidia_smi)
        self.assertTrue(detected.has_pdftotext and detected.has_tesseract
                        and detected.has_magick)
        self.assertFalse(detected.wsl)
        self.assertEqual(detected.lock_id, "linux-x86_64")
        self.assertEqual(detected.python_platform_tag, "x86_64-unknown-linux-gnu")

    def test_linux_headless_without_systemd_or_desktop(self) -> None:
        detected = detect_host(
            {}, platform_name="linux", machine="aarch64", uid=1000,
            which=_which(set()), exists=_paths(set()), is_dir=_paths(set()),
            is_file=_paths(set()),
            read_text=lambda _path: "Linux version 5.15.0-microsoft-standard-WSL2",
            run=_systemd(1))
        self.assertEqual(detected.session, "headless")
        self.assertTrue(detected.wsl)
        self.assertFalse(detected.has_systemd_user)
        self.assertFalse(detected.has_hyprland)
        self.assertEqual(detected.lock_id, "linux-arm64")
        self.assertEqual(detected.python_platform_tag, "aarch64-unknown-linux-gnu")

    def test_apple_silicon_mac(self) -> None:
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        detected = detect_host(
            {}, platform_name="darwin", machine="arm64", uid=501,
            which=_which({"launchctl"}), exists=_paths(set()),
            is_dir=_paths({"/dev/shm"}), is_file=_paths({"/usr/bin/bwrap"}),
            read_text=lambda _path: "", run=run)
        self.assertEqual(detected.os, "macos")
        self.assertEqual(detected.arch, "aarch64")
        self.assertEqual(detected.session, "aqua")
        self.assertTrue(detected.has_launchd)
        self.assertFalse(detected.has_systemd_user)
        self.assertFalse(detected.has_bwrap, "Linux-only probes never fire on macOS")
        self.assertFalse(detected.has_dev_shm)
        self.assertEqual(calls, [], "systemctl is never run off Linux")
        self.assertEqual(detected.lock_id, "macos-arm64")
        self.assertEqual(detected.python_platform_tag, "aarch64-apple-darwin")
        self.assertTrue(detected.is_posix)

    def test_windows_workstation(self) -> None:
        detected = detect_host(
            {}, platform_name="win32", machine="AMD64",
            which=_which({"schtasks", "nvidia-smi"}), exists=_paths(set()),
            is_dir=_paths(set()), is_file=_paths(set()),
            read_text=lambda _path: "", run=_systemd(0))
        self.assertEqual(detected.os, "windows")
        self.assertEqual(detected.arch, "x86_64")
        self.assertEqual(detected.session, "win32")
        self.assertTrue(detected.has_task_scheduler)
        self.assertTrue(detected.has_nvidia_smi)
        self.assertFalse(detected.has_systemd_user)
        self.assertFalse(detected.is_posix)
        self.assertEqual(detected.lock_id, "windows-x86_64")
        self.assertEqual(detected.python_platform_tag, "x86_64-pc-windows-msvc")

    def test_to_dict_is_json_friendly_and_names_the_lock(self) -> None:
        value = HostPlatform(os="macos", arch="x86_64").to_dict()
        self.assertEqual(value["lock_id"], "macos-x86_64")
        self.assertEqual(value["python_platform_tag"], "x86_64-apple-darwin")
        self.assertTrue(all(isinstance(item, (str, bool)) for item in value.values()))

    def test_unknown_os_or_arch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_os("plan9")
        with self.assertRaises(ValueError):
            normalize_arch("mips")
        with self.assertRaises(ValueError):
            HostPlatform(os="bsd", arch="x86_64")

    def test_normalization_aliases(self) -> None:
        self.assertEqual(normalize_os("Windows"), "windows")
        self.assertEqual(normalize_os("cygwin"), "windows")
        self.assertEqual(normalize_arch("arm64"), "aarch64")
        self.assertEqual(normalize_arch("x64"), "x86_64")

    def test_current_host_matches_this_machine_and_is_cached(self) -> None:
        host_module._reset_host_cache()
        try:
            first = host_module.current_host()
            second = host_module.current_host()
        finally:
            host_module._reset_host_cache()
        self.assertIs(first, second)
        self.assertIn(first.os, {"linux", "macos", "windows"})
        self.assertTrue(Path("/proc").exists() or not first.is_linux)


if __name__ == "__main__":
    unittest.main()
