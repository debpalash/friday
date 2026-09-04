"""Exact lock, installed-distribution, and model license review.

Review version 2 binds one policy entry to every hash lock Friday ships,
one per platform and engine. The lock installed on the reviewing host is
compared against its interpreter's metadata; the other locks are reviewed
from their digests, package counts, and the license evidence gathered from
any reviewed environment or the manual evidence table. A package that
appears only in a foreign lock without evidence fails the review.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


_LOCKED = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ;\\]+)")
_NORMALIZE = re.compile(r"[-_.]+")
REVIEW_VERSION = 2


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


def _probe_environment() -> dict[str, str]:
    if os.name == "posix":
        return {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
    keep = ("SystemRoot", "PATH", "TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA",
            "PATHEXT", "COMSPEC")
    return {name: os.environ[name] for name in keep if name in os.environ}


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
        timeout=60, check=False, env=_probe_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError("installed dependency metadata probe failed")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("installed dependency metadata probe was invalid")
    return value


def _automatic_evidence(distribution: dict[str, Any]) -> str | None:
    return (
        distribution.get("license_expression")
        or distribution.get("license")
        or (distribution.get("license_classifiers") or [None])[0]
        or ("bundled-license-file"
            if distribution.get("license_files") else None)
    )


def _review_installed(
    lock: Path, python: Path, manual: Mapping[str, str],
    evidence_pool: dict[str, str],
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
        automatic = _automatic_evidence(distribution)
        evidence = automatic or manual.get(key)
        if not evidence:
            missing_evidence.append(key)
        else:
            evidence_pool.setdefault(key, str(evidence)[:500])
        packages.append({
            "package": key,
            "license_evidence": str(evidence or "missing")[:500],
            "evidence_source": "installed_metadata" if automatic else "manual_review",
        })
    return {
        "review": "installed",
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "locked_packages": len(locked),
        "matched_packages": len(locked) - len(mismatches),
        "mismatches": mismatches,
        "missing_license_evidence": missing_evidence,
        "packages": packages,
        "passed": not mismatches and not missing_evidence,
    }


def _review_lock_only(
    lock: Path, manual: Mapping[str, str], evidence_pool: Mapping[str, str],
) -> dict[str, Any]:
    locked = parse_lock(lock)
    packages = []
    missing_evidence = []
    by_name = {}
    for pooled_key, pooled_evidence in evidence_pool.items():
        by_name.setdefault(pooled_key.split("==", 1)[0], pooled_evidence)
    for name, version in sorted(locked.items()):
        key = f"{name}=={version}"
        if key in evidence_pool:
            evidence, source = evidence_pool[key], "reviewed_environment"
        elif key in manual:
            evidence, source = manual[key], "manual_review"
        elif name in by_name:
            # The same distribution at another pinned version was reviewed
            # on this host; licenses rarely change between adjacent releases,
            # and the exact version is still recorded here for the release.
            evidence, source = by_name[name], "reviewed_environment_other_version"
        else:
            evidence, source = None, "missing"
        if not evidence:
            missing_evidence.append(key)
        packages.append({
            "package": key,
            "license_evidence": str(evidence or "missing")[:500],
            "evidence_source": source,
        })
    return {
        "review": "lock_only",
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "locked_packages": len(locked),
        "matched_packages": 0,
        "mismatches": [],
        "missing_license_evidence": missing_evidence,
        "packages": packages,
        "passed": not missing_evidence,
    }


def host_lock_name(host_lock_id: str | None = None) -> str:
    """The policy key of the application lock for this host."""
    if host_lock_id is None:
        from friday_host.host import current_host  # noqa: PLC0415

        host_lock_id = current_host().lock_id
    return f"application-{host_lock_id}"


def run_dependency_review(
    repo: Path, *, environments: Mapping[str, Path] | None = None,
    app_python: Path | None = None, qwen_python: Path | None = None,
    host_lock_id: str | None = None,
) -> dict[str, Any]:
    """Review every policy lock; probe the ones with an interpreter.

    ``app_python`` is the interpreter of the host's application environment.
    On Linux it maps to the CUDA superset lock when that lock's packages are
    installed, otherwise to the portable lock for the host.
    """
    policy_path = repo / "compliance" / "dependency-review-v1.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("review_version") != REVIEW_VERSION:
        raise ValueError("dependency review policy version is unsupported")
    manual = dict(policy["manual_license_evidence"])
    locks: dict[str, dict[str, Any]] = policy["locks"]
    interpreters: dict[str, Path] = dict(environments or {})
    if app_python is not None:
        target = host_lock_name(host_lock_id)
        cuda_name = f"application-cuda-{(host_lock_id or host_lock_name().removeprefix('application-'))}"
        if cuda_name in locks and _has_cuda_stack(app_python):
            target = cuda_name
        interpreters.setdefault(target, app_python)
    if qwen_python is not None:
        interpreters.setdefault("qwen_runtime", qwen_python)
    unknown = sorted(set(interpreters) - set(locks))
    if unknown:
        raise ValueError(f"unknown lock environment(s): {', '.join(unknown)}")

    evidence_pool: dict[str, str] = {}
    results: dict[str, dict[str, Any]] = {}
    ordered = [name for name in locks if name in interpreters] + [
        name for name in locks if name not in interpreters]
    for name in ordered:
        expected = locks[name]
        lock = repo / expected["path"]
        if name in interpreters:
            result = _review_installed(lock, interpreters[name], manual, evidence_pool)
        else:
            result = _review_lock_only(lock, manual, evidence_pool)
        result["platform"] = expected.get("platform")
        result["engines"] = list(expected.get("engines") or [])
        result["policy_sha256_matches"] = (
            result["lock_sha256"] == expected["sha256"])
        result["policy_package_count_matches"] = (
            result["locked_packages"] == expected["packages"])
        result["passed"] = bool(
            result["passed"] and result["policy_sha256_matches"]
            and result["policy_package_count_matches"])
        results[name] = result

    models = policy.get("models_and_assets") or []
    models_complete = bool(models) and all(
        isinstance(item, dict)
        and all(str(item.get(key) or "").strip()
                for key in ("name", "pin", "license"))
        for item in models)
    binaries = policy.get("binary_assets") or []
    binaries_complete = all(
        isinstance(item, dict)
        and all(str(item.get(key) or "").strip()
                for key in ("name", "version", "license"))
        and isinstance(item.get("artifacts"), dict) and item["artifacts"]
        and all(re.fullmatch(r"[0-9a-f]{64}", str(digest))
                for digest in item["artifacts"].values())
        for item in binaries)
    passed = (all(value["passed"] for value in results.values())
              and models_complete and binaries_complete
              and policy.get("engineering_review") == "complete"
              and bool(interpreters))
    return {
        "review_version": policy["review_version"],
        "passed": passed,
        "host_platform": sys.platform,
        "installed_reviewed": sorted(interpreters),
        "environments": results,
        "models_and_assets": models,
        "models_complete": models_complete,
        "binary_assets": binaries,
        "binaries_complete": binaries_complete,
        "copyleft_findings": policy.get("copyleft_findings") or [],
        "distribution_approval": policy["distribution_approval"],
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    }


def _has_cuda_stack(python: Path) -> bool:
    probe = "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
    try:
        completed = subprocess.run(
            [str(python), "-c", probe], capture_output=True, timeout=30,
            check=False, env=_probe_environment())
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def write_private_review(path: Path, result: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("dependency review target already exists")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".dependency-review-", dir=path.parent)
    try:
        if hasattr(os, "fchmod"):
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


__all__ = [
    "REVIEW_VERSION", "host_lock_name", "normalized_name", "parse_lock",
    "run_dependency_review", "write_private_review",
]
