"""Which of Friday's action classes this host can run.

Everything that depends on systemd, bubblewrap, Hyprland, Omarchy, or the
managed Chromium is Linux-only in this release. The gate never hides a
missing capability: unsupported tools are dropped from the model's tool
catalog and reported in ``/api/status`` with a reason code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .host import HostPlatform

# Tool names live in friday_core.builtin_tools; they are repeated here as
# plain strings so this stdlib package never imports the application core.
PROCESS_TOOLS = frozenset({
    "machine_launch_process", "machine_inspect_process",
    "machine_terminate_process", "machine_list_process_specs",
})
DESKTOP_TOOLS = frozenset({
    "machine_list_windows", "machine_focus_window", "machine_close_window",
})
OMARCHY_TOOLS = frozenset({
    "machine_omarchy_status", "machine_omarchy_set_theme",
    "machine_omarchy_set_font", "machine_omarchy_set_nightlight",
    "machine_omarchy_set_idle", "machine_omarchy_set_brightness",
    "machine_omarchy_take_screenshot", "machine_omarchy_lock",
    "machine_omarchy_install_browser",
})
BROWSER_TOOLS = frozenset({
    "browser_open", "browser_snapshot", "browser_click", "browser_type",
})
OCR_TOOLS = frozenset({"machine_ocr_image"})
VISION_TOOLS = frozenset({"machine_understand_image"})
VOICE_PROFILE_TOOLS = frozenset({"create_voice_profile"})

REASON_LINUX_SYSTEMD = "requires_linux_systemd_run_and_bwrap"
REASON_LINUX_HYPRLAND = "requires_linux_hyprland_session"
REASON_OMARCHY = "requires_omarchy"
REASON_MANAGED_CHROMIUM = "requires_linux_managed_chromium"
REASON_SANDBOX = "requires_linux_bwrap_sandbox"
REASON_CUDA_VOICE = "requires_cuda_omnivoice"
REASON_HEADLESS = "no_desktop_session"


@dataclass(frozen=True)
class PlatformCapabilities:
    desktop: bool
    omarchy: bool
    managed_processes: bool
    managed_browser: bool
    sandboxed_documents: bool
    ocr: bool
    native_vision_host: bool
    voice_profiles: bool
    notifications: bool
    clipboard: bool
    open_local: bool
    reasons: Mapping[str, str] = field(default_factory=dict)
    unavailable_tools: Mapping[str, str] = field(default_factory=dict)

    @property
    def document_formats(self) -> tuple[str, ...]:
        formats = ["txt", "md", "csv", "json", "docx", "odt", "epub", "pptx", "xlsx"]
        if self.sandboxed_documents:
            formats.insert(0, "pdf")
        return tuple(formats)

    def to_status(self) -> dict[str, object]:
        return {
            "desktop": self.desktop,
            "omarchy": self.omarchy,
            "managed_processes": self.managed_processes,
            "managed_browser": self.managed_browser,
            "sandboxed_documents": self.sandboxed_documents,
            "ocr": self.ocr,
            "native_vision_host": self.native_vision_host,
            "voice_profiles": self.voice_profiles,
            "notifications": self.notifications,
            "clipboard": self.clipboard,
            "open_local": self.open_local,
            "document_formats": list(self.document_formats),
            "reasons": dict(self.reasons),
            "unavailable_tools": dict(self.unavailable_tools),
        }


def compute_capabilities(host: HostPlatform, *, desktop_mode: str = "auto",
                         accelerator: str = "none") -> PlatformCapabilities:
    """Decide the action classes for ``host``; raises for impossible demands."""
    if desktop_mode not in {"auto", "required", "disabled"}:
        raise ValueError("desktop mode must be auto, required, or disabled")
    reasons: dict[str, str] = {}

    managed_processes = (host.is_linux and host.has_systemd_user
                         and host.has_systemd_run and host.has_bwrap)
    if not managed_processes:
        reasons["managed_processes"] = REASON_LINUX_SYSTEMD

    if desktop_mode == "required" and not host.is_linux:
        raise RuntimeError(
            f"desktop control is unsupported on {host.os}; set "
            "FRIDAY_DESKTOP_MODE=auto or disabled")
    desktop = (desktop_mode != "disabled" and host.is_linux
               and (desktop_mode == "required" or host.has_hyprland))
    if not desktop:
        reasons["desktop"] = (
            "disabled_by_configuration" if desktop_mode == "disabled"
            else REASON_LINUX_HYPRLAND)

    omarchy = desktop and host.has_omarchy
    if not omarchy:
        reasons["omarchy"] = reasons.get("desktop", REASON_OMARCHY)

    managed_browser = managed_processes and host.has_managed_chromium
    if not managed_browser:
        reasons["managed_browser"] = (
            REASON_MANAGED_CHROMIUM if managed_processes
            else REASON_LINUX_SYSTEMD)

    sandbox = host.is_linux and host.has_bwrap and host.has_dev_shm
    sandboxed_documents = sandbox and host.has_pdftotext
    if not sandboxed_documents:
        reasons["sandboxed_documents"] = REASON_SANDBOX
    ocr = sandbox and host.has_tesseract
    if not ocr:
        reasons["ocr"] = REASON_SANDBOX
    native_vision_host = sandbox and host.has_magick
    if not native_vision_host:
        reasons["native_vision_host"] = REASON_SANDBOX

    voice_profiles = accelerator == "cuda"
    if not voice_profiles:
        reasons["voice_profiles"] = REASON_CUDA_VOICE

    headless = host.session == "headless"
    if headless:
        reasons["notifications"] = REASON_HEADLESS
        reasons["clipboard"] = REASON_HEADLESS
        reasons["open_local"] = REASON_HEADLESS

    unavailable: dict[str, str] = {}
    if not managed_processes:
        for name in PROCESS_TOOLS:
            unavailable[name] = REASON_LINUX_SYSTEMD
    if not desktop:
        for name in DESKTOP_TOOLS:
            unavailable[name] = reasons["desktop"]
    if not omarchy:
        for name in OMARCHY_TOOLS:
            unavailable[name] = reasons["omarchy"]
    if not managed_browser:
        for name in BROWSER_TOOLS:
            unavailable[name] = reasons["managed_browser"]
    if not ocr:
        for name in OCR_TOOLS:
            unavailable[name] = REASON_SANDBOX
    if not native_vision_host:
        for name in VISION_TOOLS:
            unavailable[name] = REASON_SANDBOX
    if not voice_profiles:
        for name in VOICE_PROFILE_TOOLS:
            unavailable[name] = REASON_CUDA_VOICE

    return PlatformCapabilities(
        desktop=desktop, omarchy=omarchy,
        managed_processes=managed_processes, managed_browser=managed_browser,
        sandboxed_documents=sandboxed_documents, ocr=ocr,
        native_vision_host=native_vision_host, voice_profiles=voice_profiles,
        notifications=not headless, clipboard=not headless,
        open_local=not headless,
        reasons=MappingProxyType(reasons),
        unavailable_tools=MappingProxyType(unavailable),
    )


__all__ = [
    "BROWSER_TOOLS", "DESKTOP_TOOLS", "OCR_TOOLS", "OMARCHY_TOOLS",
    "PROCESS_TOOLS", "PlatformCapabilities", "VISION_TOOLS",
    "VOICE_PROFILE_TOOLS", "compute_capabilities",
]
