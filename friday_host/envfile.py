"""Read and write Friday's ``friday.env`` configuration file.

The file is a list of ``KEY='value'`` lines that both a POSIX shell (through
systemd ``EnvironmentFile=`` or ``source``) and this parser understand.
Single-quoted values may contain any character; an embedded quote is written
as ``'\\''``. Double-quoted and bare values are accepted when reading.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvFileError(ValueError):
    """The environment file is malformed."""


def parse_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            raise EnvFileError(f"line {number}: expected KEY=value")
        key, value = line.split("=", 1)
        key = key.strip()
        if not _KEY.match(key):
            raise EnvFileError(f"line {number}: invalid variable name")
        value = value.strip()
        if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
            inner = value[1:-1]
            if "'" in inner.replace("'\\''", ""):
                raise EnvFileError(f"line {number}: unbalanced quote")
            value = inner.replace("'\\''", "'")
        elif len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        elif value.startswith(("'", '"')):
            raise EnvFileError(f"line {number}: unterminated quote")
        values[key] = value
    return values


def read_env_file(path: Path) -> dict[str, str]:
    return parse_env_text(path.read_text(encoding="utf-8"))


def render_env_text(values: Mapping[str, str]) -> str:
    lines = []
    for key, value in values.items():
        if not _KEY.match(key):
            raise EnvFileError(f"invalid variable name: {key}")
        if "\n" in value or "\r" in value:
            raise EnvFileError(f"value for {key} contains a line break")
        lines.append(f"{key}='{value.replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'")
    return "\n".join(lines) + ("\n" if lines else "")


__all__ = ["EnvFileError", "parse_env_text", "read_env_file", "render_env_text"]
