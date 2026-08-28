#!/usr/bin/env python3
"""Fail when Friday's composition boundaries regress into server.py."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server.py"
MAX_SERVER_LINES = 4_500
BOUNDARIES = {
    "transport": ROOT / "friday_core" / "transport.py",
    "controller": ROOT / "friday_core" / "controller_api.py",
    "conversation": ROOT / "friday_core" / "conversation_runtime.py",
    "speech": ROOT / "friday_core" / "speech.py",
    "voice_transport": ROOT / "friday_core" / "voice_transport.py",
    "task_orchestration": ROOT / "friday_core" / "task_orchestration.py",
    "frontend_loader": ROOT / "friday_core" / "frontend.py",
    "frontend_asset": ROOT / "frontend" / "index.html",
}


def inspect_architecture() -> dict[str, object]:
    source = SERVER.read_text(encoding="utf-8")
    lines = len(source.splitlines())
    missing = sorted(name for name, path in BOUNDARIES.items()
                     if not path.is_file() or path.stat().st_size == 0)
    imports = {
        "transport": "from friday_core.transport import",
        "controller": "from friday_core.controller_api import",
        "conversation": "from friday_core.conversation_runtime import",
        "voice_transport": "from friday_core.voice_transport import",
        "task_orchestration": "from friday_core.task_orchestration import",
        "frontend_loader": "from friday_core.frontend import",
    }
    missing_imports = sorted(name for name, marker in imports.items()
                             if marker not in source)
    embedded_frontend = 'HTML = """' in source or "HTML = '''" in source
    external_frontend = (
        'HTML = load_frontend(REPO / "frontend" / "index.html")' in source)
    return {
        "passed": (
            not missing and not missing_imports and not embedded_frontend
            and external_frontend and lines <= MAX_SERVER_LINES
        ),
        "server_lines": lines,
        "maximum_server_lines": MAX_SERVER_LINES,
        "missing_boundaries": missing,
        "missing_composition_imports": missing_imports,
        "embedded_frontend": embedded_frontend,
        "external_frontend": external_frontend,
    }


def main() -> int:
    result = inspect_architecture()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
