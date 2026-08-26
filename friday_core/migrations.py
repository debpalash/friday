"""One-way imports from Friday's pre-graph persistence formats."""

from __future__ import annotations

import json
from pathlib import Path

from .graph import GraphStore


def migrate_session_json(graph: GraphStore, path: str | Path) -> int:
    """Import legacy chat history into journal-only graph nodes exactly once."""
    path = Path(path)
    if graph.count_nodes("legacy_session") or not path.is_file():
        return 0
    try:
        messages = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(messages, list):
        return 0

    session_id = graph.record_node(
        "legacy_session",
        {"source": path.name, "message_count": len(messages),
         "knowledge_layer": "journal_only"},
        actor="migration", event_type="migration.session_started")
    imported = 0
    kind_for_role = {
        "user": "utterance",
        "assistant": "assistant_message",
        "tool": "observation",
    }
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        role = str(message.get("role", "unknown"))
        kind = kind_for_role.get(role, "observation")
        node_id = graph.record_node(
            kind,
            {"source": path.name, "legacy_index": index, "role": role,
             "message": message, "knowledge_layer": "journal_only"},
            actor="migration", session_id=session_id,
            event_type="migration.message_imported")
        graph.record_edge(session_id, "contains", node_id, actor="migration")
        imported += 1
    return imported
