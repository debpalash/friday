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
    "conversation": ROOT / "friday_core" / "conversation_runtime.py",
    "speech": ROOT / "friday_core" / "speech.py",
    "voice_transport": ROOT / "friday_core" / "voice_transport.py",
    "task_orchestration": ROOT / "friday_core" / "task_orchestration.py",
    "frontend_loader": ROOT / "friday_core" / "frontend.py",
    "platform_gate": ROOT / "friday_core" / "platform_gate.py",
    "host_platform": ROOT / "friday_host" / "platform_capabilities.py",
    "frontend_asset": ROOT / "frontend" / "index.html",
}


POSIX_ONLY_MODULES = {"fcntl", "pwd", "grp", "termios", "resource", "msvcrt",
                      "winreg", "_winapi"}
PLATFORM_LAYER_EXEMPT = {ROOT / "friday_host" / "fs.py"}


def _posix_only_imports() -> list[str]:
    import ast

    offenders = []
    candidates = [ROOT / "server.py", ROOT / "supervisor.py", ROOT / "friday.py",
                  *sorted((ROOT / "friday_core").glob("*.py")),
                  *sorted((ROOT / "friday_host").glob("*.py")),
                  *sorted((ROOT / "ops").glob("*.py"))]
    for path in candidates:
        if path in PLATFORM_LAYER_EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in POSIX_ONLY_MODULES:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{name}")
    return offenders


def inspect_architecture() -> dict[str, object]:
    source = SERVER.read_text(encoding="utf-8")
    lines = len(source.splitlines())
    missing = sorted(name for name, path in BOUNDARIES.items()
                     if not path.is_file() or path.stat().st_size == 0)
    imports = {
        "transport": "from friday_core.transport import",
        "conversation": "from friday_core.conversation_runtime import",
        "voice_transport": "from friday_core.voice_transport import",
        "task_orchestration": "from friday_core.task_orchestration import",
        "frontend_loader": "from friday_core.frontend import",
        "platform_gate": "from friday_core.platform_gate import",
        "host_platform": "from friday_host.platform_capabilities import",
    }
    missing_imports = sorted(name for name, marker in imports.items()
                             if marker not in source)
    posix_only = sorted(_posix_only_imports())
    embedded_frontend = 'HTML = """' in source or "HTML = '''" in source
    external_frontend = (
        'HTML = load_frontend(REPO / "frontend" / "index.html")' in source)
    return {
        "passed": (
            not missing and not missing_imports and not embedded_frontend
            and external_frontend and lines <= MAX_SERVER_LINES
            and not posix_only
        ),
        "posix_only_imports_outside_platform_layer": posix_only,
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
