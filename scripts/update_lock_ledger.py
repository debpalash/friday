#!/usr/bin/env python3
"""Refresh lock digests and package counts in the dependency-review ledger."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "compliance" / "dependency-review-v1.json"


def main() -> int:
    policy = json.loads(LEDGER.read_text(encoding="utf-8"))
    for name, entry in policy["locks"].items():
        path = ROOT / entry["path"]
        text = path.read_text(encoding="utf-8")
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry["packages"] = len(re.findall(r"^[A-Za-z0-9_.-]+==", text, re.M))
        print(f"{name}: {entry['packages']} packages")
    LEDGER.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
