"""Compose the host platform gate into the tool catalog and status reports.

``server.py`` computes one :class:`PlatformCapabilities` value at import and
passes it here; these helpers keep the composition root small and make the
filtering testable without the FastAPI application.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from friday_host.host import HostPlatform
from friday_host.platform_capabilities import PlatformCapabilities

UNSUPPORTED_STATUS = "unsupported_on_platform"


def unsupported_tools(platform: PlatformCapabilities | None) -> dict[str, str]:
    return dict(platform.unavailable_tools) if platform is not None else {}


def filter_tool_schema(
    schemas: Iterable[dict], *, unsupported: Mapping[str, str],
    remote_enabled: bool, vision_qualified: bool,
) -> list[dict]:
    """Drop tools this host cannot run or that lack a qualified backend."""
    kept = []
    for item in schemas:
        name = item["function"]["name"]
        if name in unsupported:
            continue
        if name == "remote_reason" and not remote_enabled:
            continue
        if name == "machine_understand_image" and not vision_qualified:
            continue
        kept.append(item)
    return kept


def inventory_entries(
    schemas: Iterable[dict], *, unsupported: Mapping[str, str],
    remote_enabled: bool, vision_qualified: bool,
) -> list[dict]:
    """Describe every built-in tool, including the ones the host refuses."""
    entries = []
    for item in schemas:
        name = item["function"]["name"]
        entry = {"name": name, "description": item["function"]["description"],
                 "kind": "builtin"}
        if name in unsupported:
            entry["status"] = UNSUPPORTED_STATUS
            entry["reason"] = unsupported[name]
        elif ((name == "remote_reason" and not remote_enabled)
              or (name == "machine_understand_image" and not vision_qualified)):
            entry["status"] = "unavailable"
        else:
            entry["status"] = "active"
        entries.append(entry)
    return entries


def platform_status(host: HostPlatform, platform: PlatformCapabilities) -> dict:
    return {
        "os": host.os, "arch": host.arch, "session": host.session,
        "lock_id": host.lock_id, "wsl": host.wsl,
        "capabilities": platform.to_status(),
    }


__all__ = [
    "UNSUPPORTED_STATUS", "filter_tool_schema", "inventory_entries",
    "platform_status", "unsupported_tools",
]
