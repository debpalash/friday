"""Desktop conveniences: notifications, clipboard, and opening local targets.

Each ``*_command`` function returns the exact argv for a host so tests can
assert it on any platform; the verbs take an injectable runner. User text is
never placed inside a shell string: every platform receives it as an argument
or through standard input, and macOS notifications read it from AppleScript
argv. Error messages never include the text.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from .host import HostPlatform, current_host
from .procs import detached_popen_kwargs, hidden_console_kwargs


class HostCommandError(RuntimeError):
    """A desktop helper command failed; the message carries no user text."""


Runner = Callable[..., subprocess.CompletedProcess]
Starter = Callable[..., object]

_NOTIFY_APPLESCRIPT = (
    "on run argv",
    "display notification (item 2 of argv) with title (item 1 of argv)",
    "end run",
)

_NOTIFY_POWERSHELL = (
    "$title = $env:FRIDAY_NOTIFY_TITLE; $body = $env:FRIDAY_NOTIFY_BODY; "
    "[Windows.UI.Notifications.ToastNotificationManager, "
    "Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; "
    "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, "
    "ContentType = WindowsRuntime] | Out-Null; "
    "$xml = New-Object Windows.Data.Xml.Dom.XmlDocument; "
    "$template = [Windows.UI.Notifications.ToastNotificationManager]::"
    "GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::"
    "ToastText02); "
    "$nodes = $template.GetElementsByTagName('text'); "
    "$nodes.Item(0).AppendChild($template.CreateTextNode($title)) | Out-Null; "
    "$nodes.Item(1).AppendChild($template.CreateTextNode($body)) | Out-Null; "
    "$toast = New-Object Windows.UI.Notifications.ToastNotification $template; "
    "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
    "'Friday').Show($toast)"
)

_POWERSHELL = ("powershell", "-NoProfile", "-NonInteractive",
               "-ExecutionPolicy", "Bypass", "-Command")


def notification_command(title: str, body: str,
                         host: HostPlatform | None = None) -> list[str]:
    host = host or current_host()
    if host.is_linux:
        return ["notify-send", title, body]
    if host.is_macos:
        command = ["osascript"]
        for line in _NOTIFY_APPLESCRIPT:
            command.extend(["-e", line])
        return [*command, title[:240], body[:240]]
    return [*_POWERSHELL, _NOTIFY_POWERSHELL]


def notify(title: str, body: str, *, host: HostPlatform | None = None,
           runner: Runner = subprocess.run, timeout: float = 10) -> None:
    host = host or current_host()
    command = notification_command(title, body, host)
    options: dict[str, object] = {
        "capture_output": True, "timeout": timeout, "check": True,
    }
    if host.is_windows:
        import os  # noqa: PLC0415

        options["env"] = {**os.environ, "FRIDAY_NOTIFY_TITLE": title[:240],
                          "FRIDAY_NOTIFY_BODY": body[:240]}
        options.update(hidden_console_kwargs())
    try:
        runner(command, **options)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostCommandError(
            f"desktop notification failed ({type(exc).__name__})") from exc


def open_command(target: str | Path,
                 host: HostPlatform | None = None) -> list[str]:
    host = host or current_host()
    value = str(target)
    if host.is_linux:
        return ["xdg-open", value]
    if host.is_macos:
        return ["open", value]
    return ["cmd", "/c", "start", "", value]


def open_path(target: str | Path, *, host: HostPlatform | None = None,
              starter: Starter = subprocess.Popen) -> None:
    host = host or current_host()
    try:
        starter(
            open_command(target, host), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            **detached_popen_kwargs())
    except OSError as exc:
        raise HostCommandError(
            f"local open failed ({type(exc).__name__})") from exc


def clipboard_read_command(host: HostPlatform | None = None) -> list[str]:
    host = host or current_host()
    if host.is_linux:
        return ["wl-paste", "--no-newline"]
    if host.is_macos:
        return ["pbpaste"]
    return [*_POWERSHELL, "Get-Clipboard -Raw"]


def clipboard_write_command(host: HostPlatform | None = None) -> list[str]:
    host = host or current_host()
    if host.is_linux:
        return ["wl-copy"]
    if host.is_macos:
        return ["pbcopy"]
    return ["clip.exe"]


def clipboard_read(*, host: HostPlatform | None = None,
                   runner: Runner = subprocess.run, timeout: float = 5,
                   limit: int = 4000) -> str:
    host = host or current_host()
    try:
        result = runner(
            clipboard_read_command(host), text=True, capture_output=True,
            timeout=timeout, check=True, **hidden_console_kwargs())
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostCommandError(
            f"clipboard read failed ({type(exc).__name__})") from exc
    return str(result.stdout)[:limit]


def clipboard_write(text: str, *, host: HostPlatform | None = None,
                    runner: Runner = subprocess.run, timeout: float = 5) -> None:
    host = host or current_host()
    try:
        runner(
            clipboard_write_command(host), input=text, text=True,
            capture_output=True, timeout=timeout, check=True,
            **hidden_console_kwargs())
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostCommandError(
            f"clipboard write failed ({type(exc).__name__})") from exc


__all__ = [
    "HostCommandError", "clipboard_read", "clipboard_read_command",
    "clipboard_write", "clipboard_write_command", "notification_command",
    "notify", "open_command", "open_path",
]
