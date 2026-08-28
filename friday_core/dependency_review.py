"""Exact lock, installed-distribution, and model license review."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


_LOCKED = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ;\\]+)")
_NORMALIZE = re.compile(r"[-_.]+")


def normalized_name(value: str) -> str:
    return _NORMALIZE.sub("-", value).lower()


def parse_lock(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _LOCKED.match(line)
        if not match:
            continue
        name = normalized_name(match.group(1))
        if name in packages:
            raise ValueError(f"duplicate locked package: {name}")
        packages[name] = match.group(2)
    if not packages:
        raise ValueError("dependency lock contains no exact packages")
    return packages


def _installed_distributions(python: Path) -> dict[str, dict[str, Any]]:
    probe = r'''
import json
import re
from importlib import metadata

result = {}
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    if not name:
        continue
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    classifiers = [
        value for value in distribution.metadata.get_all("Classifier") or []
        if value.startswith("License ::")
    ]
    license_files = sorted(
        str(value) for value in distribution.files or []
        if re.search(r"(^|/)(licen[sc]e|copying|notice|authors?)([^/]*$)",
                     str(value), re.I)
    )
    result[normalized] = {
        "name": name,
        "version": distribution.version,
        "license": distribution.metadata.get("License") or None,
        "license_expression": (
            distribution.metadata.get("License-Expression") or None),
        "license_classifiers": classifiers,
        "license_files": license_files,
    }
print(json.dumps(result, sort_keys=True))
'''
    completed = subprocess.run(
        [str(python), "-c", probe], text=True, capture_output=True,
        timeout=30, check=False,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    if completed.returncode != 0:
        raise RuntimeError("installed dependency metadata probe failed")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("installed dependency metadata probe was invalid")
    return value


def _review_environment(
    lock: Path, python: Path, manual: dict[str, str],
) -> dict[str, Any]:
    locked = parse_lock(lock)
    installed = _installed_distributions(python)
    packages = []
    mismatches = []
    missing_evidence = []
    for name, version in sorted(locked.items()):
        distribution = installed.get(name)
        key = f"{name}=={version}"
        if distribution is None:
            mismatches.append({"package": key, "installed": None})
            continue
        if distribution.get("version") != version:
            mismatches.append({
                "package": key, "installed": distribution.get("version")})
        automatic = (
            distribution.get("license_expression")
            or distribution.get("license")
            or (distribution.get("license_classifiers") or [None])[0]
            or ("bundled-license-file"
                if distribution.get("license_files") else None)
        )
        evidence = automatic or manual.get(key)
        if not evidence:
            missing_evidence.append(key)
        packages.append({
            "package": key,
            "license_evidence": str(evidence or "missing")[:500],
            "evidence_source": "installed_metadata" if automatic else "manual_review",
        })
    return {
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "locked_packages": len(locked),
        "matched_packages": len(locked) - len(mismatches),
        "mismatches": mismatches,
        "missing_license_evidence": missing_evidence,
        "packages": packages,
        "passed": not mismatches and not missing_evidence,
    }


def run_dependency_review(
    repo: Path, *, app_python: Path, qwen_python: Path,
) -> dict[str, Any]:
    policy_path = repo / "compliance" / "dependency-review-v1.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    manual = dict(policy["manual_license_evidence"])
    environments = {}
    for name, python in (("application", app_python),
                         ("qwen_runtime", qwen_python)):
        expected = policy["locks"][name]
        lock = repo / expected["path"]
        result = _review_environment(lock, python, manual)
        result["policy_sha256_matches"] = (
            result["lock_sha256"] == expected["sha256"])
        result["policy_package_count_matches"] = (
            result["locked_packages"] == expected["packages"])
        result["passed"] = bool(
            result["passed"] and result["policy_sha256_matches"]
            and result["policy_package_count_matches"])
        environments[name] = result
    models = policy.get("models_and_assets") or []
    models_complete = bool(models) and all(
        isinstance(item, dict)
        and all(str(item.get(key) or "").strip()
                for key in ("name", "pin", "license"))
        for item in models)
    passed = all(value["passed"] for value in environments.values()) \
        and models_complete and policy.get("engineering_review") == "complete"
    return {
        "review_version": policy["review_version"],
        "passed": passed,
        "environments": environments,
        "models_and_assets": models,
        "models_complete": models_complete,
        "copyleft_findings": policy.get("copyleft_findings") or [],
        "distribution_approval": policy["distribution_approval"],
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    }


def write_private_review(path: Path, result: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("dependency review target already exists")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".dependency-review-", dir=path.parent)
    try:
        os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        payload = (json.dumps(result, indent=2, sort_keys=True)
                   + "\n").encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
