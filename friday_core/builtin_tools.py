"""Canonical declarations for Friday's built-in tools.

This module owns the static contract of every built-in tool. Runtime adapters
consume the catalog instead of maintaining parallel name-indexed declarations.
It deliberately has no dependencies on server composition or cognition models.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

OMARCHY_STATUS_TOOL = "machine_omarchy_status"
OMARCHY_ACTION_TOOLS = frozenset({
    "machine_omarchy_set_theme", "machine_omarchy_set_font",
    "machine_omarchy_set_nightlight", "machine_omarchy_set_idle",
    "machine_omarchy_set_brightness", "machine_omarchy_take_screenshot",
    "machine_omarchy_lock",
})
OMARCHY_TOOL_NAMES = frozenset({OMARCHY_STATUS_TOOL, *OMARCHY_ACTION_TOOLS})

BUILTIN_TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "fetch_news",
        "description": "Fetch current India news from a live external RSS feed. "
                       "Use this whenever the user asks for news, headlines, current "
                       "events, or what is happening; never substitute memory or files.",
        "parameters": {"type": "object",
                       "properties": {
                           "topic": {"type": "string",
                                     "description": "Optional subject, e.g. technology"},
                           "region": {"type": "string",
                                      "description": "India (default), US, UK, or World"},
                           "limit": {"type": "integer", "minimum": 1, "maximum": 10,
                                     "description": "Number of headlines; default 5"}},
                       "required": []}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files and folders in a directory of your project "
                       "(default: the project root).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string",
                                               "description": "relative dir, or '.' for root"}},
                       "required": []}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file from your project directory.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Propose an exact project file change for explicit user "
                       "approval, then stage it, run the verification suite, and "
                       "promote it with a recoverable checkpoint only after approval.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "restart",
        "description": "Restart your own server process to apply code changes. "
                       "Announce it to the user before calling.",
        "parameters": {"type": "object",
                       "properties": {"reason": {"type": "string"}},
                       "required": ["reason"]}}},
    {"type": "function", "function": {
        "name": "remember_preference",
        "description": "Store a durable user preference or correction. Use only when "
                       "the user explicitly states a lasting preference, never for guesses "
                       "or one-off task instructions.",
        "parameters": {"type": "object",
                       "properties": {
                           "key": {"type": "string",
                                   "description": "stable category, e.g. progress_style"},
                           "value": {"type": "string",
                                     "description": "the preference stated by the user"}},
                       "required": ["key", "value"]}}},
    {"type": "function", "function": {
        "name": "recall_memory",
        "description": "Search Friday's verified long-term memory.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "create_skill",
        "description": "Draft a reusable versioned skill from completed work. Drafts are "
                       "not active until deterministic tests validate them.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "instructions": {"type": "string"},
            "permissions": {"type": "array", "items": {"type": "string"}},
            "tests": {"type": "array", "items": {"type": "object"}}
        }, "required": ["name", "instructions", "permissions", "tests"]}}},
    {"type": "function", "function": {
        "name": "list_skills",
        "description": "List drafted, validated, active, and quarantined skills.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search_skill_catalog",
        "description": "Search Skills.sh for reusable procedural knowledge when a "
                       "request needs know-how that no active local skill provides. "
                       "Discovery is read-only and does not trust or install results.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10}
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "import_skill",
        "description": "Import one exact Skills.sh skill ID after user approval. "
                       "Friday pins its content hash, grants no new permissions, runs "
                       "local static checks, requires clean upstream security audits, "
                       "and quarantines anything that fails.",
        "parameters": {"type": "object", "properties": {
            "skill_id": {"type": "string",
                         "description": "Exact owner/repository/skill ID returned by search"}
        }, "required": ["skill_id"]}}},
]

BUILTIN_TOOL_SCHEMAS.extend([
    {"type": "function", "function": {
        "name": "create_capability",
        "description": "Create executable tool code as a versioned candidate. Static "
                       "policy checks and at least two executable tests must pass before "
                       "the tool becomes available.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "parameters": {"type": "object"},
            "code": {"type": "string", "description": "Python defining run(args)"},
            "permissions": {"type": "array", "items": {"type": "string",
                                                              "enum": ["network", "filesystem_read", "filesystem_write", "browser", "clipboard", "notifications", "scheduling", "process"]}},
            "tests": {"type": "array", "minItems": 2, "items": {"type": "object"}}
        }, "required": ["name", "description", "parameters", "code",
                         "permissions", "tests"]}}},
    {"type": "function", "function": {
        "name": "list_capabilities",
        "description": "List executable capability versions and lifecycle states.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "create_voice_profile",
        "description": "Create a candidate voice profile. A reference, when used, must "
                       "already be under persona/voices. The profile is not activated.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "instruct": {"type": "string"},
            "reference": {"type": "string"}
        }, "required": ["name", "instruct"]}}},
    {"type": "function", "function": {
        "name": "list_voices",
        "description": "Report the active synthesis backend, device, audible runtime "
                       "voice, separately stored active profile, activation support, "
                       "and available profile lifecycle states. Use this before every "
                       "claim about the current TTS or voice.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "set_voice",
        "description": "Synthesize a private test sample and activate the named voice "
                       "only if validation succeeds on OmniVoice; otherwise keep the "
                       "audible runtime voice and report the active backend.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "rollback_voice",
        "description": "Test and restore the previously active voice profile.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "upgrade_core",
        "description": "Run a sandboxed Pi maintenance agent against an isolated copy "
                       "of Friday's core. Existing tests cannot be changed; the candidate "
                       "and test output are preserved for explicit diff review and are "
                       "never automatically promoted.",
        "parameters": {"type": "object", "properties": {
            "objective": {"type": "string"}}, "required": ["objective"]}}},
    {"type": "function", "function": {
        "name": "list_core_upgrades",
        "description": "Inspect staged, rejected, and awaiting-review core candidates.",
        "parameters": {"type": "object", "properties": {}}}},
])

BUILTIN_TOOL_SCHEMAS.extend([
    {"type": "function", "function": {
        "name": "machine_omarchy_status",
        "description": "Inspect the installed Omarchy version, current and available "
                       "themes and fonts, night light, idle policy, display brightness, "
                       "and lock state through identity-pinned commands.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "machine_omarchy_set_theme",
        "description": "Apply one exact installed Omarchy theme after approval and "
                       "verify that it became current. Call machine_omarchy_status first.",
        "parameters": {"type": "object", "properties": {
            "theme": {"type": "string",
                      "description": "Exact theme label returned by status"}
        }, "required": ["theme"]}}},
    {"type": "function", "function": {
        "name": "machine_omarchy_set_font",
        "description": "Apply one exact installed Omarchy monospace font after "
                       "approval and verify that it became current. Call status first.",
        "parameters": {"type": "object", "properties": {
            "font": {"type": "string",
                     "description": "Exact font label returned by status"}
        }, "required": ["font"]}}},
    {"type": "function", "function": {
        "name": "machine_omarchy_set_nightlight",
        "description": "Set Omarchy night light to an explicit enabled or disabled "
                       "state after approval, then verify its reported state.",
        "parameters": {"type": "object", "properties": {
            "enabled": {"type": "boolean"}
        }, "required": ["enabled"]}}},
    {"type": "function", "function": {
        "name": "machine_omarchy_set_idle",
        "description": "Choose whether Omarchy may idle and lock or must stay awake. "
                       "This security-sensitive change requires exact approval.",
        "parameters": {"type": "object", "properties": {
            "mode": {"type": "string",
                     "enum": ["allow_idle", "stay_awake"]}
        }, "required": ["mode"]}}},
    {"type": "function", "function": {
        "name": "machine_omarchy_set_brightness",
        "description": "Set focused-display brightness to an exact percentage after "
                       "approval and verify the observed percentage.",
        "parameters": {"type": "object", "properties": {
            "percent": {"type": "integer", "minimum": 1, "maximum": 100}
        }, "required": ["percent"]}}},
    {"type": "function", "function": {
        "name": "machine_omarchy_take_screenshot",
        "description": "Capture the full desktop to Friday's private Pictures/Friday "
                       "directory after approval and return a verified PNG receipt.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "machine_omarchy_lock",
        "description": "Lock the Omarchy session after exact approval and verify the "
                       "session-lock state.",
        "parameters": {"type": "object", "properties": {}}}},
])

BUILTIN_TOOL_SCHEMAS.extend([
    {"type": "function", "function": {
        "name": "machine_list_windows",
        "description": "List identity-verified windows on the local Hyprland desktop. "
                       "Returns opaque window IDs, a bounded application label, workspace, and "
                       "layout state; never titles, compositor addresses, PIDs, or paths.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "machine_focus_window",
        "description": "Request exact approval to focus one opaque window from a fresh "
                       "machine_list_windows receipt. The target's desktop-session and "
                       "process identities are rebound before dispatch.",
        "parameters": {"type": "object", "properties": {
            "window_id": {"type": "string",
                          "description": "Opaque ID from machine_list_windows"}
        }, "required": ["window_id"]}}},
    {"type": "function", "function": {
        "name": "machine_close_window",
        "description": "Request exact approval to gracefully close one opaque window. "
                       "This never accepts a PID, title, class selector, signal, command, "
                       "or compositor address and verifies the exact window disappeared.",
        "parameters": {"type": "object", "properties": {
            "window_id": {"type": "string",
                          "description": "Opaque ID from machine_list_windows"}
        }, "required": ["window_id"]}}},
])

BUILTIN_TOOL_SCHEMAS.extend([
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the live public web and return attributable links. Use for "
                       "current or external information that is not specifically news.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10}
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_web",
        "description": "Read the visible text of a public HTTP or HTTPS page. Private "
                       "network addresses and oversized responses are blocked.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 500, "maximum": 20000}
        }, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "browser_open",
        "description": "Open a public URL in Friday's dedicated visible Chromium profile.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}
        }, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "browser_snapshot",
        "description": "Read the active managed-browser page after navigation or interaction.",
        "parameters": {"type": "object", "properties": {
            "page_url": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 500, "maximum": 20000}
        }}}},
    {"type": "function", "function": {
        "name": "browser_click",
        "description": "Click one CSS selector in the managed browser. This requires "
                       "an approval receipt because clicks may cause external effects.",
        "parameters": {"type": "object", "properties": {
            "selector": {"type": "string"}, "page_url": {"type": "string"}
        }, "required": ["selector"]}}},
    {"type": "function", "function": {
        "name": "browser_type",
        "description": "Fill one CSS selector in the managed browser; optionally press Enter. "
                       "Typed text is redacted from the graph and approval is required.",
        "parameters": {"type": "object", "properties": {
            "selector": {"type": "string"}, "text": {"type": "string"},
            "page_url": {"type": "string"}, "submit": {"type": "boolean"}
        }, "required": ["selector", "text"]}}},
    {"type": "function", "function": {
        "name": "clipboard_read",
        "description": "Read up to 4,000 characters from the local Wayland clipboard.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "clipboard_write",
        "description": "Replace the local Wayland clipboard with the supplied text.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}
        }, "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "desktop_notify",
        "description": "Show a local desktop notification.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "message": {"type": "string"}
        }, "required": ["title", "message"]}}},
    {"type": "function", "function": {
        "name": "open_local",
        "description": "Open a file from Friday's project in its default desktop app.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}
        }, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "create_reminder",
        "description": "Create a persistent reminder. due_at must be ISO-8601 and include "
                       "a timezone; use the user's Asia/Kolkata timezone unless told otherwise.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"},
            "due_at": {"type": "string"},
            "interval_seconds": {"type": "integer", "minimum": 60}
        }, "required": ["text", "due_at"]}}},
    {"type": "function", "function": {
        "name": "list_reminders",
        "description": "List scheduled, fired, or cancelled reminders.",
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["scheduled", "fired", "cancelled"]}
        }}}},
    {"type": "function", "function": {
        "name": "cancel_reminder",
        "description": "Cancel a reminder by its exact reminder_id.",
        "parameters": {"type": "object", "properties": {
            "reminder_id": {"type": "string"}
        }, "required": ["reminder_id"]}}},
    {"type": "function", "function": {
        "name": "remote_reason",
        "description": "Escalate one redacted reasoning prompt to the configured remote "
                       "OpenAI-compatible model. Always requires a separate approval "
                       "showing the payload preview. Unavailable until configured.",
        "parameters": {"type": "object", "properties": {
            "prompt": {"type": "string"}
        }, "required": ["prompt"]}}},
])

BUILTIN_TOOL_SCHEMAS.extend([
    {"type": "function", "function": {
        "name": "machine_grant_path",
        "description": "Request an encrypted, revocable grant for one exact existing "
                       "machine directory. The grant never bypasses per-action approval "
                       "for writes, and sensitive directories need explicit opt-in.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Exact absolute directory path"},
            "permissions": {"type": "array", "minItems": 1,
                            "uniqueItems": True, "items": {"type": "string",
                            "enum": ["inspect", "list", "read", "write"]}},
            "allow_sensitive": {"type": "boolean"},
            "expires_at": {"type": "string",
                           "description": "Optional timezone-aware ISO-8601 expiry"}
        }, "required": ["path", "permissions"]}}},
    {"type": "function", "function": {
        "name": "machine_list_grants",
        "description": "List redacted machine path grants and their permissions, "
                       "expiry, and lifecycle state.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "machine_revoke_grant",
        "description": "Revoke one exact machine path grant by grant_id.",
        "parameters": {"type": "object", "properties": {
            "grant_id": {"type": "string"}}, "required": ["grant_id"]}}},
    {"type": "function", "function": {
        "name": "machine_inspect_path",
        "description": "Inspect metadata for an exact absolute path covered by an "
                       "active inspect grant. Symlink aliases are rejected.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "machine_list_path",
        "description": "List an exact granted directory without following symlinks; "
                       "sensitive descendants are omitted unless explicitly granted.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500}
        }, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "machine_read_text",
        "description": "Read a bounded UTF-8 text file from an exact granted machine "
                       "path. Raw content remains private and encrypted at rest.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "max_bytes": {"type": "integer", "minimum": 1, "maximum": 256000}
        }, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "machine_read_document",
        "description": "Extract bounded text from an exact read-granted PDF, "
                       "DOCX, ODT, EPUB, PPTX, or XLSX file. Archives are parsed "
                       "without executing content and PDFs use a resource-limited "
                       "sandbox. Private text is encrypted at rest.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 1,
                          "maximum": 250000}
        }, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "machine_ocr_image",
        "description": "Recognize bounded English text in an exact read-granted "
                       "PNG or JPEG. The untrusted decoder runs in a networkless, "
                       "resource-limited sandbox. This is OCR only and must not be "
                       "presented as general visual understanding.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 1,
                          "maximum": 250000}
        }, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "machine_understand_image",
        "description": "Answer one question about an exact read-granted PNG or "
                       "JPEG using the active local native-vision model. The image "
                       "is decoded and canonicalized in a networkless resource-limited "
                       "sandbox first. This tool is available only on a profile whose "
                       "native-vision boot canaries passed.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "question": {"type": "string", "minLength": 1,
                         "maxLength": 2000}
        }, "required": ["path", "question"]}}},
    {"type": "function", "function": {
        "name": "machine_write_text",
        "description": "Request approval for one exact bounded text replacement on a "
                       "write-granted path. The write is atomic, verified, idempotent, "
                       "and creates an encrypted rollback checkpoint.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}
        }, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "machine_rollback_write",
        "description": "Request approval to restore a machine write from its exact "
                       "rollback operation ID. Refuses to overwrite later edits.",
        "parameters": {"type": "object", "properties": {
            "operation_id": {"type": "string"}},
            "required": ["operation_id"]}}},
])

BUILTIN_TOOL_SCHEMAS.extend([
    {"type": "function", "function": {
        "name": "machine_list_process_specs",
        "description": "List the trusted, immutable applications Friday is allowed "
                       "to manage. This never accepts executable paths or commands.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "machine_launch_process",
        "description": "Request exact approval to launch one trusted process spec in "
                       "an owned, resource-limited user-systemd boundary. Never accepts "
                       "a shell command, executable path, environment, unit, or PID.",
        "parameters": {"type": "object", "properties": {
            "spec_id": {"type": "string",
                        "description": "Opaque ID from machine_list_process_specs"},
            "parameter_values": {
                "type": "object",
                "description": "Only the bounded typed fields declared by that spec"}
        }, "required": ["spec_id"]}}},
    {"type": "function", "function": {
        "name": "machine_inspect_process",
        "description": "Inspect one Friday-owned process by opaque instance ID. "
                       "Returns aggregate lifecycle/resource state, never PID, command "
                       "line, environment, unit name, paths, or raw output.",
        "parameters": {"type": "object", "properties": {
            "instance_id": {"type": "string"}
        }, "required": ["instance_id"]}}},
    {"type": "function", "function": {
        "name": "machine_terminate_process",
        "description": "Request exact approval to terminate the owned cgroup for one "
                       "opaque Friday process instance. Raw PIDs, groups, and unit names "
                       "are never accepted.",
        "parameters": {"type": "object", "properties": {
            "instance_id": {"type": "string"}
        }, "required": ["instance_id"]}}},
])

BUILTIN_TOOL_NAMES = {item["function"]["name"] for item in BUILTIN_TOOL_SCHEMAS}
EXACT_STEP_APPROVAL_TOOLS = {
    "browser_click", "browser_type", "remote_reason", "upgrade_core",
    "import_skill", "write_file", "create_capability", "machine_grant_path",
    "machine_revoke_grant", "machine_write_text", "machine_rollback_write",
    "machine_launch_process", "machine_terminate_process",
    "machine_focus_window", "machine_close_window",
} | set(OMARCHY_ACTION_TOOLS)
BLOCKING_IO_TOOLS = {
    "fetch_news", "web_search", "read_web", "browser_open", "browser_snapshot",
    "browser_click", "browser_type", "clipboard_read", "clipboard_write",
    "desktop_notify", "open_local", "search_skill_catalog",
    "machine_read_document", "machine_ocr_image",
}
PROCESS_TOOL_NAMES = frozenset({
    "machine_list_process_specs", "machine_launch_process",
    "machine_inspect_process", "machine_terminate_process",
})
DESKTOP_TOOL_NAMES = frozenset({
    "machine_list_windows", "machine_focus_window", "machine_close_window",
}) | OMARCHY_TOOL_NAMES

TOOL_POLICY_DATA: dict[str, tuple[str, tuple[str, ...], bool]] = {
    "fetch_news": ("read_only", ("network",), False),
    "web_search": ("read_only", ("network",), False),
    "read_web": ("read_only", ("network",), False),
    "search_skill_catalog": ("read_only", ("network",), False),
    "import_skill": ("medium", ("network", "skill_write"), True),
    "browser_open": ("low", ("browser",), False),
    "browser_snapshot": ("read_only", ("browser",), False),
    "browser_click": ("medium", ("browser",), True),
    "browser_type": ("high", ("browser",), True),
    "clipboard_read": ("read_only", ("clipboard",), False),
    "clipboard_write": ("low", ("clipboard",), False),
    "desktop_notify": ("low", ("notifications",), False),
    "open_local": ("low", ("process", "filesystem_read"), False),
    "list_files": ("read_only", ("filesystem_read",), False),
    "read_file": ("read_only", ("filesystem_read",), False),
    "write_file": ("medium", ("filesystem_write",), True),
    "machine_grant_path": ("high", ("operator_grant",), True),
    "machine_list_grants": ("read_only", ("operator_grant",), False),
    "machine_revoke_grant": ("medium", ("operator_grant",), True),
    "machine_inspect_path": ("read_only", ("filesystem_read",), False),
    "machine_list_path": ("read_only", ("filesystem_read",), False),
    "machine_read_text": ("read_only", ("filesystem_read",), False),
    "machine_read_document": (
        "read_only", ("filesystem_read",), False),
    "machine_ocr_image": (
        "read_only", ("filesystem_read",), False),
    "machine_understand_image": (
        "read_only", ("filesystem_read",), False),
    "machine_write_text": ("high", ("filesystem_write",), True),
    "machine_rollback_write": ("high", ("filesystem_write",), True),
    "machine_list_process_specs": (
        "read_only", ("process",), False),
    "machine_launch_process": ("high", ("process",), True),
    "machine_inspect_process": (
        "read_only", ("process",), False),
    "machine_terminate_process": ("high", ("process",), True),
    "machine_list_windows": ("read_only", ("desktop",), False),
    "machine_focus_window": ("medium", ("desktop",), True),
    "machine_close_window": ("high", ("desktop",), True),
    "machine_omarchy_status": ("read_only", ("desktop",), False),
    "machine_omarchy_set_theme": ("medium", ("desktop",), True),
    "machine_omarchy_set_font": ("medium", ("desktop",), True),
    "machine_omarchy_set_nightlight": ("low", ("desktop",), True),
    "machine_omarchy_set_idle": ("high", ("desktop",), True),
    "machine_omarchy_set_brightness": ("low", ("desktop",), True),
    "machine_omarchy_take_screenshot": (
        "high", ("desktop", "filesystem_write"), True),
    "machine_omarchy_lock": ("high", ("desktop",), True),
    "create_reminder": ("low", ("scheduling", "notifications"), False),
    "list_reminders": ("read_only", ("scheduling",), False),
    "cancel_reminder": ("low", ("scheduling",), False),
    "set_voice": ("low", ("voice",), False),
    "rollback_voice": ("low", ("voice",), False),
    "upgrade_core": ("high", ("core_upgrade", "filesystem_write"), True),
    "create_capability": ("high",
                          ("dynamic_capability", "filesystem_write"), True),
    "remote_reason": ("high", ("remote_model", "network"), True),
    "restart": ("medium", ("process",), False),
}

RESOURCE_OVERRIDES: dict[str, dict[str, Any]] = {
    "browser_open": {"cpu_cores": 1.0, "ram_mib": 768, "network": True},
    "browser_snapshot": {"cpu_cores": 0.5, "ram_mib": 512},
    "browser_click": {"cpu_cores": 0.5, "ram_mib": 512},
    "browser_type": {"cpu_cores": 0.5, "ram_mib": 512},
    "fetch_news": {"ram_mib": 192, "network": True},
    "web_search": {"ram_mib": 192, "network": True},
    "read_web": {"ram_mib": 256, "network": True},
    "import_skill": {"cpu_cores": 1.0, "ram_mib": 512, "network": True},
    "create_capability": {"cpu_cores": 1.0, "ram_mib": 512,
                          "latency_class": "batch"},
    "upgrade_core": {"cpu_cores": 2.0, "ram_mib": 2048,
                     "latency_class": "batch"},
    "set_voice": {"cpu_cores": 1.0, "ram_mib": 2048,
                  "latency_class": "batch"},
    "machine_understand_image": {
        "cpu_cores": 1.0, "ram_mib": 768,
        "latency_class": "interactive"},
    # Process launch claims are replaced with the immutable curated spec claim
    # before a durable batch is staged.  This conservative fallback is used
    # only if a caller bypasses that binding step, in which case the broker
    # rejects the unbound launch before creating an external effect.
    "machine_launch_process": {
        "cpu_cores": 1.0, "ram_mib": 512, "latency_class": "interactive"},
    # Cleanup must remain dispatchable when ordinary capacity is saturated.
    # ResourceAdmissionController recognizes only this exact zero-resource
    # shape as a short-lived control-lane claim; workload transfer rejects it.
    "machine_terminate_process": {
        "cpu_cores": 0.0, "ram_mib": 0, "vram_mib": 0,
        "accelerator": "none", "network": False,
        "concurrency_slots": 1, "latency_class": "control"},
    "machine_inspect_process": {
        "cpu_cores": 0.0, "ram_mib": 0, "vram_mib": 0,
        "accelerator": "none", "network": False,
        "concurrency_slots": 1, "latency_class": "control"},
    "machine_list_windows": {"cpu_cores": 0.1, "ram_mib": 64},
    "machine_focus_window": {"cpu_cores": 0.1, "ram_mib": 64},
    "machine_close_window": {"cpu_cores": 0.1, "ram_mib": 64},
    "machine_omarchy_status": {"cpu_cores": 0.1, "ram_mib": 64},
    "machine_omarchy_set_theme": {"cpu_cores": 0.25, "ram_mib": 128},
    "machine_omarchy_set_font": {"cpu_cores": 0.25, "ram_mib": 128},
    "machine_omarchy_set_nightlight": {"cpu_cores": 0.1, "ram_mib": 64},
    "machine_omarchy_set_idle": {"cpu_cores": 0.1, "ram_mib": 64},
    "machine_omarchy_set_brightness": {"cpu_cores": 0.1, "ram_mib": 64},
    "machine_omarchy_take_screenshot": {"cpu_cores": 0.5, "ram_mib": 256},
    "machine_omarchy_lock": {"cpu_cores": 0.1, "ram_mib": 64},
}

TOOL_CRITERIA = {
    "fetch_news": ("fresh_sources", "Return current attributed headlines", "news_receipt"),
    "web_search": ("search_sources", "Return attributed search results", "source_receipt"),
    "read_web": ("page_read", "Return content from the requested page", "page_receipt"),
    "search_skill_catalog": ("skill_candidates", "Return attributable Skills.sh candidates", "skill_search_receipt"),
    "import_skill": ("skill_imported", "Activate only a pinned and security-validated skill", "skill_import_receipt"),
    "browser_open": ("browser_opened", "Open the requested URL in the managed profile", "browser_receipt"),
    "browser_snapshot": ("browser_observed", "Observe the active managed browser page", "browser_receipt"),
    "browser_click": ("browser_clicked", "Perform the approved browser click", "browser_receipt"),
    "browser_type": ("browser_typed", "Type into the approved browser field", "browser_receipt"),
    "write_file": ("deployment_passed", "Promote the requested file only after tests", "deployment_receipt"),
    "machine_grant_path": ("operator_grant_created", "Create only the exact approved encrypted path grant", "operator_grant_receipt"),
    "machine_revoke_grant": ("operator_grant_revoked", "Revoke the requested exact grant", "operator_grant_receipt"),
    "machine_inspect_path": ("machine_path_inspected", "Return verified metadata from a granted path", "machine_read_receipt"),
    "machine_list_path": ("machine_path_listed", "Return a bounded granted directory listing", "machine_read_receipt"),
    "machine_read_text": ("machine_text_read", "Return bounded text and its content hash", "machine_read_receipt"),
    "machine_read_document": ("machine_document_read", "Extract bounded text from an exact granted document and return source and text hashes", "machine_read_receipt"),
    "machine_ocr_image": ("machine_image_ocr", "Recognize bounded text from an exact granted PNG or JPEG and return source and text hashes", "machine_read_receipt"),
    "machine_understand_image": ("machine_image_understood", "Answer one bounded question about an exact granted, sandbox-sanitized PNG or JPEG using the profile-bound local vision model", "machine_read_receipt"),
    "machine_write_text": ("machine_text_written", "Atomically write, reread, and checkpoint the exact approved content", "machine_write_receipt"),
    "machine_rollback_write": ("machine_write_rolled_back", "Restore the checkpoint only when no later edit would be overwritten", "machine_rollback_receipt"),
    "machine_list_process_specs": ("process_specs_listed", "List only trusted immutable process specifications", "process_list_receipt"),
    "machine_launch_process": ("process_launched", "Launch exactly one approved curated process in an owned resource boundary", "process_launch_receipt"),
    "machine_inspect_process": ("process_inspected", "Inspect only a Friday-owned opaque process instance", "process_inspect_receipt"),
    "machine_terminate_process": ("process_terminated", "Terminate only the exact approved Friday-owned process boundary", "process_terminate_receipt"),
    "machine_list_windows": ("desktop_windows_listed", "List only identity-verified windows with opaque targets", "desktop_list_receipt"),
    "machine_focus_window": ("desktop_window_focused", "Focus only the exact approved window identity", "desktop_focus_receipt"),
    "machine_close_window": ("desktop_window_closed", "Gracefully close only the exact approved window identity", "desktop_close_receipt"),
    "machine_omarchy_status": ("omarchy_state_observed", "Inspect only the identity-pinned Omarchy desktop state", "omarchy_status_receipt"),
    "machine_omarchy_set_theme": ("omarchy_theme_set", "Apply and re-observe the exact approved installed theme", "omarchy_theme_receipt"),
    "machine_omarchy_set_font": ("omarchy_font_set", "Apply and re-observe the exact approved installed font", "omarchy_font_receipt"),
    "machine_omarchy_set_nightlight": ("omarchy_nightlight_set", "Set and re-observe the exact approved night-light state", "omarchy_nightlight_receipt"),
    "machine_omarchy_set_idle": ("omarchy_idle_set", "Set and re-observe the exact approved idle policy", "omarchy_idle_receipt"),
    "machine_omarchy_set_brightness": ("omarchy_brightness_set", "Set and re-observe the exact approved display brightness", "omarchy_brightness_receipt"),
    "machine_omarchy_take_screenshot": ("omarchy_screenshot_captured", "Create and hash one approved full-desktop PNG", "omarchy_screenshot_receipt"),
    "machine_omarchy_lock": ("omarchy_session_locked", "Request and re-observe the approved session lock", "omarchy_lock_receipt"),
    "create_reminder": ("reminder_saved", "Persist the requested reminder", "reminder_receipt"),
    "cancel_reminder": ("reminder_cancelled", "Cancel the requested reminder", "reminder_receipt"),
    "set_voice": ("voice_activated", "Activate the requested voice after synthesis", "voice_receipt"),
    "upgrade_core": ("upgrade_reviewable", "Prepare a tested core candidate for explicit review", "upgrade_receipt"),
}

PRIVATE_ARGUMENT_FIELDS = {
    "browser_open": {"url"},
    "browser_snapshot": {"page_url"},
    "browser_click": {"page_url"},
    "browser_type": {"page_url", "text"},
    "clipboard_write": {"text"},
    "desktop_notify": {"title", "message"},
    "remote_reason": {"prompt"},
    "write_file": {"content"},
    "machine_grant_path": {"path"},
    "machine_revoke_grant": {"grant_id"},
    "machine_inspect_path": {"path"},
    "machine_list_path": {"path"},
    "machine_read_text": {"path"},
    "machine_read_document": {"path"},
    "machine_ocr_image": {"path"},
    "machine_understand_image": {"path", "question"},
    "machine_write_text": {"path", "content"},
    "machine_rollback_write": {"operation_id"},
    "machine_launch_process": {"parameter_values"},
}

PRIVATE_PAYLOAD_TOOL_NAMES = frozenset({
    "clipboard_read", "clipboard_write", "read_file", "remote_reason",
})
PRIVATE_PAYLOAD_PREFIXES = ("browser_", "machine_")

PROTECTED_PROJECT_ROOTS = frozenset({
    ".git", "backups", "capabilities", "models", "state", "venv",
})
PROTECTED_PROJECT_FILES = frozenset({
    ".env", "friday.log", "server.log", "session.json", "supervisor.log",
})


@dataclass(frozen=True)
class BuiltinToolSpec:
    """Static contract and execution traits for one built-in tool."""

    name: str
    schema: Mapping[str, Any]
    risk: str | None
    permissions: tuple[str, ...]
    always_approve: bool | None
    resource_overrides: Mapping[str, Any]
    criterion: tuple[str, str, str] | None
    private_argument_fields: frozenset[str]
    private_payload: bool
    exact_step_approval: bool
    blocking_io: bool
    receipt_family: str | None


@dataclass(frozen=True)
class BuiltinToolAdapters:
    """Domain effectors required by the stateless built-in implementations."""

    repo: Path
    fetch_news: Callable[[str, int, str], Any]
    web: Any
    skill_source: Any
    reminders: Any
    run_process: Callable[..., Any]
    start_process: Callable[..., Any]


def _build_catalog() -> Mapping[str, BuiltinToolSpec]:
    schemas: dict[str, Mapping[str, Any]] = {}
    for schema in BUILTIN_TOOL_SCHEMAS:
        name = str(schema["function"]["name"])
        if name in schemas:
            raise RuntimeError(f"duplicate built-in tool schema: {name}")
        schemas[name] = MappingProxyType(schema)

    declared_names = set(schemas)
    metadata_names = (
        set(TOOL_POLICY_DATA) | set(RESOURCE_OVERRIDES) | set(TOOL_CRITERIA)
        | set(PRIVATE_ARGUMENT_FIELDS) | set(EXACT_STEP_APPROVAL_TOOLS)
        | set(BLOCKING_IO_TOOLS) | set(PROCESS_TOOL_NAMES)
        | set(DESKTOP_TOOL_NAMES) | set(PRIVATE_PAYLOAD_TOOL_NAMES)
    )
    unknown = metadata_names - declared_names
    if unknown:
        raise RuntimeError(
            f"built-in tool metadata has no schema: {sorted(unknown)}")
    invalid_risks = {
        name: policy[0] for name, policy in TOOL_POLICY_DATA.items()
        if policy[0] not in {"read_only", "low", "medium", "high"}
    }
    if invalid_risks:
        raise RuntimeError(f"invalid built-in tool risks: {invalid_risks}")
    unprotected = {
        name for name in EXACT_STEP_APPROVAL_TOOLS
        if not TOOL_POLICY_DATA.get(name, (None, (), False))[2]
    }
    if unprotected:
        raise RuntimeError(
            "exact-step approval tools must always require approval: "
            f"{sorted(unprotected)}")

    catalog: dict[str, BuiltinToolSpec] = {}
    for name, schema in schemas.items():
        policy = TOOL_POLICY_DATA.get(name)
        risk, permissions, always_approve = (
            policy if policy is not None else (None, (), None))
        receipt_family = (
            "process" if name in PROCESS_TOOL_NAMES
            else "desktop" if name in DESKTOP_TOOL_NAMES
            else None
        )
        catalog[name] = BuiltinToolSpec(
            name=name,
            schema=schema,
            risk=risk,
            permissions=tuple(permissions),
            always_approve=always_approve,
            resource_overrides=MappingProxyType(
                dict(RESOURCE_OVERRIDES.get(name, {}))),
            criterion=TOOL_CRITERIA.get(name),
            private_argument_fields=frozenset(
                PRIVATE_ARGUMENT_FIELDS.get(name, ())),
            private_payload=(
                name in PRIVATE_PAYLOAD_TOOL_NAMES
                or name.startswith(PRIVATE_PAYLOAD_PREFIXES)),
            exact_step_approval=name in EXACT_STEP_APPROVAL_TOOLS,
            blocking_io=name in BLOCKING_IO_TOOLS,
            receipt_family=receipt_family,
        )
    return MappingProxyType(catalog)


BUILTIN_TOOLS = _build_catalog()


class BuiltinToolRuntime:
    """Execute stateless built-ins through one narrow adapter seam."""

    @staticmethod
    def safe_project_path(repo: Path, path: str) -> Path | None:
        candidate = ((repo / path).resolve()
                     if not path.startswith("/") else Path(path).resolve())
        if repo not in candidate.parents and candidate != repo:
            return None
        try:
            relative = candidate.relative_to(repo)
        except ValueError:
            return None
        if (relative.parts
                and relative.parts[0] in PROTECTED_PROJECT_ROOTS):
            return None
        folded_name = candidate.name.casefold()
        if (candidate.name in PROTECTED_PROJECT_FILES
                or folded_name.startswith((".env.", "api_key"))
                or "secret" in folded_name
                or "token" in folded_name):
            return None
        return candidate

    def execute(self, name: str, args: dict[str, Any],
                adapters: BuiltinToolAdapters) -> str:
        """Execute one non-durable built-in and return its model-facing receipt."""
        if name == "fetch_news":
            try:
                result = adapters.fetch_news(
                    str(args.get("topic") or ""), int(args.get("limit") or 5),
                    str(args.get("region") or "India"))
                return json.dumps(result, ensure_ascii=False)
            except Exception as exc:
                return f"error: news fetch failed: {exc}"
        if name == "web_search":
            try:
                return json.dumps(adapters.web.search(
                    str(args.get("query") or ""),
                    limit=int(args.get("limit") or 5)), ensure_ascii=False)
            except Exception as exc:
                return f"error: web search failed: {exc}"
        if name == "search_skill_catalog":
            try:
                return json.dumps(adapters.skill_source.search(
                    str(args.get("query") or ""),
                    limit=int(args.get("limit") or 5)), ensure_ascii=False)
            except Exception as exc:
                return f"error: skill discovery failed: {exc}"
        if name == "read_web":
            try:
                return json.dumps(adapters.web.read(
                    str(args.get("url") or ""),
                    max_chars=int(args.get("max_chars") or 12000)),
                    ensure_ascii=False)
            except Exception as exc:
                return f"error: web read failed: {exc}"
        if name == "browser_open":
            try:
                return json.dumps(adapters.web.open(
                    str(args.get("url") or "")), ensure_ascii=False)
            except Exception as exc:
                return f"error: browser open failed: {exc}"
        if name == "browser_snapshot":
            try:
                return json.dumps(adapters.web.snapshot(
                    args.get("page_url"),
                    max_chars=int(args.get("max_chars") or 12000)),
                    ensure_ascii=False)
            except Exception as exc:
                return f"error: browser snapshot failed: {exc}"
        if name == "browser_click":
            try:
                return json.dumps(adapters.web.click(
                    str(args.get("selector") or ""),
                    page_url=args.get("page_url")), ensure_ascii=False)
            except Exception as exc:
                return f"error: browser click failed: {exc}"
        if name == "browser_type":
            try:
                return json.dumps(adapters.web.type(
                    str(args.get("selector") or ""),
                    str(args.get("text") or ""),
                    page_url=args.get("page_url"),
                    submit=bool(args.get("submit"))), ensure_ascii=False)
            except Exception as exc:
                return f"error: browser typing failed: {exc}"
        if name == "clipboard_read":
            try:
                value = adapters.run_process(
                    ["wl-paste", "--no-newline"], text=True,
                    capture_output=True, timeout=5, check=True).stdout[:4000]
                return json.dumps(
                    {"status": "ok", "text": value}, ensure_ascii=False)
            except Exception as exc:
                return f"error: clipboard read failed: {exc}"
        if name == "clipboard_write":
            try:
                value = str(args.get("text") or "")
                adapters.run_process(
                    ["wl-copy"], input=value, text=True, capture_output=True,
                    timeout=5, check=True)
                return json.dumps(
                    {"status": "ok", "characters": len(value)})
            except Exception as exc:
                return f"error: clipboard write failed: {exc}"
        if name == "desktop_notify":
            try:
                adapters.run_process(
                    ["notify-send", str(args.get("title") or "Friday"),
                     str(args.get("message") or "")], capture_output=True,
                    timeout=5, check=True)
                return json.dumps({"status": "ok", "delivered": True})
            except Exception as exc:
                return f"error: desktop notification failed: {exc}"
        if name == "open_local":
            path = self.safe_project_path(
                adapters.repo, str(args.get("path") or ""))
            if path is None or not path.exists():
                return "error: local target is unavailable (project paths only)"
            try:
                adapters.start_process(
                    ["xdg-open", str(path)], stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True)
                return json.dumps({
                    "status": "ok", "path": str(path.relative_to(adapters.repo)),
                })
            except Exception as exc:
                return f"error: local open failed: {exc}"
        if name == "create_reminder":
            try:
                return json.dumps(adapters.reminders.create(
                    str(args.get("text") or ""),
                    str(args.get("due_at") or ""),
                    interval_seconds=(int(args["interval_seconds"])
                                      if args.get("interval_seconds") else None)),
                    ensure_ascii=False)
            except Exception as exc:
                return f"error: reminder creation failed: {exc}"
        if name == "list_reminders":
            return json.dumps(
                adapters.reminders.list(status=args.get("status")),
                ensure_ascii=False)
        if name == "cancel_reminder":
            try:
                return json.dumps(adapters.reminders.cancel(
                    str(args.get("reminder_id") or "")), ensure_ascii=False)
            except Exception as exc:
                return f"error: reminder cancellation failed: {exc}"
        if name == "list_files":
            directory = self.safe_project_path(
                adapters.repo, args.get("path") or ".")
            if directory is None or not directory.is_dir():
                return (f"error: {args.get('path', '.')} is not a directory "
                        "in your project")
            entries = sorted(
                directory.iterdir(), key=lambda item: (
                    item.is_file(), item.name.lower()))
            visible = []
            for entry in entries[:100]:
                if (entry.name.startswith((".", "__"))
                        or entry.name == "venv"):
                    continue
                visible.append(
                    ("d " if entry.is_dir() else "f ") + entry.name)
            return "\n".join(visible) or "(empty)"
        if name == "read_file":
            path = self.safe_project_path(adapters.repo, args["path"])
            if path is None or not path.is_file():
                return f"error: {args['path']} not found (project dir only)"
            return path.read_text(errors="replace")[:20000]
        if name == "restart":
            reason = args.get("reason", "")
            adapters.start_process(
                [str(adapters.repo / "venv/bin/python"),
                 str(adapters.repo / "supervisor.py"),
                 "restart-friday", "--after", "1.5"], cwd=adapters.repo,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
            return f"restarting now: {reason}"
        return f"error: unknown tool {name}"


def builtin_tool(name: str) -> BuiltinToolSpec | None:
    return BUILTIN_TOOLS.get(name)


def private_argument_fields_for(name: str) -> frozenset[str]:
    spec = builtin_tool(name)
    return spec.private_argument_fields if spec else frozenset()


def tool_has_private_payload(name: str) -> bool:
    return (name in PRIVATE_PAYLOAD_TOOL_NAMES
            or name.startswith(PRIVATE_PAYLOAD_PREFIXES))
