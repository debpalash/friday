"""Service backends issue exact systemctl and launchctl commands."""

from __future__ import annotations

import plistlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from friday_host.host import HostPlatform
from friday_host.service import (LaunchdBackend, ServiceSpec,
                                 SystemdUserBackend, backend_for)

REPO = Path(__file__).resolve().parents[1]


class RecordingRunner:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = dict(responses or {})

    def __call__(self, command, **kwargs):
        self.calls.append(list(command))
        key = " ".join(command[:3])
        returncode, stdout = self.responses.get(key, (0, ""))
        return subprocess.CompletedProcess(command, returncode, stdout, "")


def _spec(root: Path) -> ServiceSpec:
    return ServiceSpec(
        current_dir=root / "app" / "current", env_file=root / "config" / "friday.env",
        python=root / "app" / "current" / "venv" / "bin" / "python",
        supervisor=root / "app" / "current" / "supervisor.py",
        launcher=root / "app" / "current" / "ops" / "friday_launch.py",
        log_dir=root / "logs")


class SystemdBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.runner = RecordingRunner()
        self.backend = SystemdUserBackend(home=self.home, runner=self.runner)

    def test_install_renders_the_unit_and_enables_it(self) -> None:
        spec = _spec(self.home)
        self.backend.install(spec, REPO / "ops" / "friday.service.in")
        unit = self.home / ".config" / "systemd" / "user" / "friday.service"
        body = unit.read_text()
        self.assertIn(f"WorkingDirectory={spec.current_dir}", body)
        self.assertIn(f"EnvironmentFile={spec.env_file}", body)
        self.assertEqual(unit.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.runner.calls, [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "enable", "friday.service"]])
        self.assertEqual(self.backend.unit_paths(), (unit,))

    def test_lifecycle_commands_are_the_pre_port_ones(self) -> None:
        self.backend.start(); self.backend.stop(); self.backend.restart()
        self.backend.is_active(); self.backend.is_enabled(); self.backend.disable()
        self.assertEqual(self.runner.calls, [
            ["systemctl", "--user", "start", "friday.service"],
            ["systemctl", "--user", "stop", "friday.service"],
            ["systemctl", "--user", "restart", "friday.service"],
            ["systemctl", "--user", "is-active", "--quiet", "friday.service"],
            ["systemctl", "--user", "is-enabled", "--quiet", "friday.service"],
            ["systemctl", "--user", "disable", "friday.service"]])
        self.assertEqual(self.backend.log_command(follow=True),
                         ["journalctl", "--user", "-u", "friday.service", "-f"])
        self.assertEqual(self.backend.log_command(follow=False, lines=12),
                         ["journalctl", "--user", "-u", "friday.service", "-n", "12", "--no-pager"])

    def test_uninstall_removes_the_unit(self) -> None:
        self.backend.install(_spec(self.home), REPO / "ops" / "friday.service.in")
        self.runner.calls.clear()
        self.backend.uninstall()
        self.assertFalse(self.backend.unit_path.exists())
        self.assertEqual([c[2] for c in self.runner.calls], ["stop", "disable", "daemon-reload"])


class LaunchdBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.runner = RecordingRunner({"launchctl print gui/501/dev.palash.friday":
                                       (0, "state = running\n")})
        self.backend = LaunchdBackend(home=self.home, runner=self.runner, uid=501)

    def test_install_renders_a_valid_plist_and_bootstraps_it(self) -> None:
        spec = _spec(self.home)
        self.backend.install(spec, REPO / "ops" / "friday.launchd.plist.in")
        plist_path = self.home / "Library" / "LaunchAgents" / "dev.palash.friday.plist"
        plist = plistlib.loads(plist_path.read_bytes())
        self.assertEqual(plist["Label"], "dev.palash.friday")
        self.assertEqual(plist["ProgramArguments"][:2], [str(spec.python), str(spec.launcher)])
        self.assertIn(str(spec.env_file), plist["ProgramArguments"])
        self.assertEqual(plist["ProgramArguments"][-2:], [str(spec.supervisor), "watch"])
        self.assertTrue(plist["RunAtLoad"])
        self.assertEqual(plist["KeepAlive"], {"SuccessfulExit": False})
        self.assertEqual(plist["StandardOutPath"], str(spec.log_dir / "supervisor.log"))
        self.assertTrue(all(Path(item).is_absolute() for item in plist["ProgramArguments"]
                            if item.startswith("/")))
        self.assertNotIn("FRIDAY_LOCAL_API_KEY", str(plist))
        self.assertTrue(spec.log_dir.is_dir())
        self.assertEqual(self.runner.calls, [
            ["launchctl", "bootout", "gui/501/dev.palash.friday"],
            ["launchctl", "bootstrap", "gui/501", str(plist_path)],
            ["launchctl", "enable", "gui/501/dev.palash.friday"]])

    def test_lifecycle_commands(self) -> None:
        self.backend.install(_spec(self.home), REPO / "ops" / "friday.launchd.plist.in")
        self.runner.calls.clear()
        self.backend.start()
        self.assertEqual(self.runner.calls[-1], ["launchctl", "kickstart", "gui/501/dev.palash.friday"])
        self.runner.calls.clear()
        self.backend.restart()
        self.assertEqual(self.runner.calls[-1],
                         ["launchctl", "kickstart", "-k", "gui/501/dev.palash.friday"])
        self.assertTrue(self.backend.is_active())
        self.assertTrue(self.backend.is_enabled())
        self.runner.calls.clear()
        self.backend.stop()
        self.assertEqual(self.runner.calls, [["launchctl", "bootout", "gui/501/dev.palash.friday"]])
        self.assertEqual(self.backend.log_command(follow=True)[:2], ["tail", "-n"])
        self.assertTrue(self.backend.log_command(follow=True)[-1].endswith("supervisor.log"))

    def test_start_bootstraps_when_not_loaded(self) -> None:
        runner = RecordingRunner({"launchctl print gui/501/dev.palash.friday": (113, "")})
        backend = LaunchdBackend(home=self.home, runner=runner, uid=501)
        with self.assertRaisesRegex(RuntimeError, "not installed"):
            backend.start()
        backend.plist_path.parent.mkdir(parents=True)
        backend.plist_path.write_text("<plist/>")
        backend.start()
        self.assertEqual(runner.calls[-2][:2], ["launchctl", "bootstrap"])
        self.assertEqual(runner.calls[-1][:2], ["launchctl", "kickstart"])
        self.assertFalse(backend.is_active())
        self.assertEqual(backend.status_text(), "launch agent is not loaded")

    def test_uninstall_boots_out_and_removes_the_plist(self) -> None:
        self.backend.install(_spec(self.home), REPO / "ops" / "friday.launchd.plist.in")
        self.runner.calls.clear()
        self.backend.uninstall()
        self.assertFalse(self.backend.plist_path.exists())
        self.assertEqual(self.runner.calls, [["launchctl", "bootout", "gui/501/dev.palash.friday"]])


class SelectionTests(unittest.TestCase):
    def test_backend_for_host(self) -> None:
        self.assertIsInstance(backend_for(HostPlatform(os="linux", arch="x86_64")),
                              SystemdUserBackend)
        self.assertIsInstance(backend_for(HostPlatform(os="macos", arch="aarch64")),
                              LaunchdBackend)
        with self.assertRaises(NotImplementedError):
            backend_for(HostPlatform(os="windows", arch="x86_64"))


if __name__ == "__main__":
    unittest.main()
