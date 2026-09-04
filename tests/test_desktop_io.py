"""Desktop helpers emit exact per-platform commands and never leak text."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from friday_host import desktop_io
from friday_host.host import HostPlatform

LINUX = HostPlatform(os="linux", arch="x86_64")
MACOS = HostPlatform(os="macos", arch="aarch64")
WINDOWS = HostPlatform(os="windows", arch="x86_64")
SECRET = "the clipboard secret 12345"


class CommandTableTests(unittest.TestCase):
    def test_notification_commands(self) -> None:
        self.assertEqual(desktop_io.notification_command("Friday reminder", "Call", LINUX),
                         ["notify-send", "Friday reminder", "Call"])
        mac = desktop_io.notification_command("Friday reminder", 'Say "hi"\nnow', MACOS)
        self.assertEqual(mac[0], "osascript")
        self.assertEqual(mac[-2:], ["Friday reminder", 'Say "hi"\nnow'],
                         "text travels as AppleScript argv, never inside a script string")
        self.assertNotIn('Say "hi"', " ".join(mac[:-2]))
        windows = desktop_io.notification_command("Friday reminder", "Call", WINDOWS)
        self.assertEqual(windows[:2], ["powershell", "-NoProfile"])
        self.assertIn("ToastNotificationManager", windows[-1])
        self.assertNotIn("Call", windows[-1],
                         "text reaches PowerShell through the environment")

    def test_open_commands(self) -> None:
        target = Path("/tmp/report.pdf")
        self.assertEqual(desktop_io.open_command(target, LINUX), ["xdg-open", str(target)])
        self.assertEqual(desktop_io.open_command(target, MACOS), ["open", str(target)])
        self.assertEqual(desktop_io.open_command(target, WINDOWS),
                         ["cmd", "/c", "start", "", str(target)])

    def test_clipboard_commands(self) -> None:
        self.assertEqual(desktop_io.clipboard_read_command(LINUX), ["wl-paste", "--no-newline"])
        self.assertEqual(desktop_io.clipboard_write_command(LINUX), ["wl-copy"])
        self.assertEqual(desktop_io.clipboard_read_command(MACOS), ["pbpaste"])
        self.assertEqual(desktop_io.clipboard_write_command(MACOS), ["pbcopy"])
        self.assertEqual(desktop_io.clipboard_read_command(WINDOWS)[-1], "Get-Clipboard -Raw")
        self.assertEqual(desktop_io.clipboard_write_command(WINDOWS), ["clip.exe"])


class VerbTests(unittest.TestCase):
    def test_notify_runs_the_command_and_passes_windows_text_by_environment(self) -> None:
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, "", "")

        desktop_io.notify("Friday reminder", "Call", host=LINUX, runner=runner)
        self.assertEqual(calls[0][0], ["notify-send", "Friday reminder", "Call"])
        self.assertTrue(calls[0][1]["check"])
        desktop_io.notify("Friday reminder", "Call", host=WINDOWS, runner=runner)
        env = calls[1][1]["env"]
        self.assertEqual(env["FRIDAY_NOTIFY_TITLE"], "Friday reminder")
        self.assertEqual(env["FRIDAY_NOTIFY_BODY"], "Call")

    def test_failures_are_reported_without_the_text(self) -> None:
        def failing(command, **kwargs):
            raise subprocess.CalledProcessError(1, command, output=SECRET)

        with self.assertRaises(desktop_io.HostCommandError) as caught:
            desktop_io.notify("t", SECRET, host=LINUX, runner=failing)
        self.assertNotIn(SECRET, str(caught.exception))
        with self.assertRaises(desktop_io.HostCommandError) as caught:
            desktop_io.clipboard_write(SECRET, host=MACOS, runner=failing)
        self.assertNotIn(SECRET, str(caught.exception))
        with self.assertRaises(desktop_io.HostCommandError) as caught:
            desktop_io.clipboard_read(host=WINDOWS, runner=failing)
        self.assertNotIn(SECRET, str(caught.exception))

    def test_clipboard_round_trip_uses_stdin_and_bounds_output(self) -> None:
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, "x" * 5000, "")

        desktop_io.clipboard_write(SECRET, host=LINUX, runner=runner)
        self.assertEqual(calls[0][0], ["wl-copy"])
        self.assertEqual(calls[0][1]["input"], SECRET)
        self.assertEqual(len(desktop_io.clipboard_read(host=LINUX, runner=runner)), 4000)

    def test_open_path_detaches_the_child(self) -> None:
        calls = []

        def starter(command, **kwargs):
            calls.append((command, kwargs))

        desktop_io.open_path(Path("/tmp/x.txt"), host=LINUX, starter=starter)
        command, kwargs = calls[0]
        self.assertEqual(command, ["xdg-open", "/tmp/x.txt"])
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue(kwargs.get("start_new_session") or "creationflags" in kwargs)

        def failing(command, **kwargs):
            raise FileNotFoundError(command[0])

        with self.assertRaises(desktop_io.HostCommandError):
            desktop_io.open_path("/tmp/x", host=MACOS, starter=failing)


if __name__ == "__main__":
    unittest.main()
