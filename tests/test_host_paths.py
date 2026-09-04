"""Default locations follow each operating system and honour overrides."""

from __future__ import annotations

import unittest
from pathlib import Path

from friday_host import paths
from friday_host.host import HostPlatform

LINUX = HostPlatform(os="linux", arch="x86_64")
MACOS = HostPlatform(os="macos", arch="aarch64")
WINDOWS = HostPlatform(os="windows", arch="x86_64")


class LinuxPathTests(unittest.TestCase):
    def test_defaults_are_the_pre_port_xdg_locations(self) -> None:
        environment = {"HOME": "/home/tester"}
        self.assertEqual(paths.default_install_root(environment, LINUX),
                         Path("/home/tester/.local/share/friday"))
        self.assertEqual(paths.default_state_root(environment, LINUX),
                         Path("/home/tester/.local/state/friday"))
        self.assertEqual(paths.default_config_root(environment, LINUX),
                         Path("/home/tester/.config/friday"))
        self.assertEqual(paths.default_cache_root(environment, LINUX),
                         Path("/home/tester/.cache/friday"))
        self.assertEqual(paths.default_log_root(environment, LINUX),
                         Path("/home/tester/.local/state/friday/logs"))
        self.assertEqual(paths.default_bin_root(environment, LINUX),
                         Path("/home/tester/.local/bin"))
        self.assertEqual(paths.default_qwen_runtime(environment, LINUX),
                         Path("/home/tester/.local/share/friday/runtime/qwen"))

    def test_xdg_variables_are_honoured(self) -> None:
        environment = {"HOME": "/home/tester", "XDG_DATA_HOME": "/data",
                       "XDG_STATE_HOME": "/state", "XDG_CONFIG_HOME": "/cfg",
                       "XDG_CACHE_HOME": "/cache", "XDG_BIN_HOME": "/bin-home"}
        self.assertEqual(paths.default_install_root(environment, LINUX), Path("/data/friday"))
        self.assertEqual(paths.default_state_root(environment, LINUX), Path("/state/friday"))
        self.assertEqual(paths.default_config_root(environment, LINUX), Path("/cfg/friday"))
        self.assertEqual(paths.default_cache_root(environment, LINUX), Path("/cache/friday"))
        self.assertEqual(paths.default_bin_root(environment, LINUX), Path("/bin-home"))

    def test_runtime_paths_shim_delegates(self) -> None:
        from friday_core import runtime_paths

        self.assertEqual(runtime_paths.default_install_root(),
                         paths.default_install_root())
        self.assertEqual(runtime_paths.default_qwen_runtime(),
                         paths.default_qwen_runtime())


class MacPathTests(unittest.TestCase):
    def test_defaults_live_under_library(self) -> None:
        environment = {"HOME": "/Users/tester"}
        support = Path("/Users/tester/Library/Application Support/Friday")
        self.assertEqual(paths.default_install_root(environment, MACOS), support / "app")
        self.assertEqual(paths.default_state_root(environment, MACOS), support / "state")
        self.assertEqual(paths.default_config_root(environment, MACOS), support / "config")
        self.assertEqual(paths.default_cache_root(environment, MACOS),
                         Path("/Users/tester/Library/Caches/Friday"))
        self.assertEqual(paths.default_log_root(environment, MACOS),
                         Path("/Users/tester/Library/Logs/Friday"))
        self.assertEqual(paths.default_bin_root(environment, MACOS),
                         Path("/Users/tester/.local/bin"))

    def test_xdg_override_drives_the_linux_style_layout_for_tests(self) -> None:
        environment = {"HOME": "/Users/tester", "XDG_STATE_HOME": "/tmp/state"}
        self.assertEqual(paths.default_state_root(environment, MACOS),
                         Path("/tmp/state/friday"))
        self.assertEqual(paths.default_log_root(environment, MACOS),
                         Path("/tmp/state/friday/logs"))


class WindowsPathTests(unittest.TestCase):
    def test_defaults_live_under_localappdata(self) -> None:
        # Forward slashes: on the Linux test host these are PosixPaths, and
        # WindowsPath accepts them unchanged.
        environment = {"USERPROFILE": "C:/Users/tester",
                       "LOCALAPPDATA": "C:/Users/tester/AppData/Local"}
        root = Path("C:/Users/tester/AppData/Local/Friday")
        self.assertEqual(paths.default_install_root(environment, WINDOWS), root / "app")
        self.assertEqual(paths.default_state_root(environment, WINDOWS), root / "state")
        self.assertEqual(paths.default_config_root(environment, WINDOWS), root / "config")
        self.assertEqual(paths.default_cache_root(environment, WINDOWS), root / "cache")
        self.assertEqual(paths.default_log_root(environment, WINDOWS), root / "state" / "logs")
        self.assertEqual(paths.default_bin_root(environment, WINDOWS), root / "bin")

    def test_missing_localappdata_falls_back_to_the_profile(self) -> None:
        environment = {"USERPROFILE": "/Users/tester"}
        self.assertEqual(
            paths.default_install_root(environment, WINDOWS),
            Path("/Users/tester/AppData/Local/Friday/app"))

    def test_venv_interpreters(self) -> None:
        release = Path("/opt/friday/releases/1")
        self.assertEqual(paths.venv_python(release, LINUX), release / "venv/bin/python")
        self.assertEqual(paths.venv_pythonw(release, MACOS), release / "venv/bin/python")
        self.assertEqual(paths.venv_python(release, WINDOWS),
                         release / "venv" / "Scripts" / "python.exe")
        self.assertEqual(paths.venv_pythonw(release, WINDOWS),
                         release / "venv" / "Scripts" / "pythonw.exe")


class OverrideTests(unittest.TestCase):
    def test_explicit_friday_variables_win_on_every_platform(self) -> None:
        environment = {
            "HOME": "/home/tester", "FRIDAY_INSTALL_ROOT": "/srv/friday",
            "FRIDAY_STATE_ROOT": "/srv/state", "FRIDAY_CONFIG_ROOT": "/srv/cfg",
            "FRIDAY_CACHE_ROOT": "/srv/cache", "FRIDAY_LOG_ROOT": "/srv/logs",
            "FRIDAY_BIN_ROOT": "/srv/bin", "XDG_DATA_HOME": "/ignored",
        }
        for host in (LINUX, MACOS, WINDOWS):
            with self.subTest(host=host.os):
                self.assertEqual(paths.default_install_root(environment, host), Path("/srv/friday"))
                self.assertEqual(paths.default_state_root(environment, host), Path("/srv/state"))
                self.assertEqual(paths.default_config_root(environment, host), Path("/srv/cfg"))
                self.assertEqual(paths.default_cache_root(environment, host), Path("/srv/cache"))
                self.assertEqual(paths.default_log_root(environment, host), Path("/srv/logs"))
                self.assertEqual(paths.default_bin_root(environment, host), Path("/srv/bin"))
                self.assertEqual(paths.default_qwen_runtime(environment, host),
                                 Path("/srv/friday/runtime/qwen"))


if __name__ == "__main__":
    unittest.main()
