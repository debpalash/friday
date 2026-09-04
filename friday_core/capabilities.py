"""Versioned, test-gated executable capability registry."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from friday_host import fs

from .graph import GraphStore, canonical_json, new_id, sha256_text, utc_now


ALLOWED_PERMISSIONS = {
    "network", "filesystem_read", "filesystem_write", "browser", "clipboard",
    "notifications", "scheduling", "process",
}
SAFE_IMPORTS = {"datetime", "html", "json", "math", "re", "statistics", "time", "xml"}
NETWORK_IMPORTS = {"http", "urllib"}
FILESYSTEM_IMPORTS = {"pathlib"}
PROCESS_IMPORTS = {"subprocess"}
BANNED_NAMES = {
    "__import__", "breakpoint", "compile", "eval", "exec", "getattr", "globals",
    "input", "locals", "setattr", "vars",
}
RUNNER = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("friday_capability", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    value = module.run(json.load(sys.stdin))
    print(json.dumps({"ok": True, "result": value}, ensure_ascii=False))
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
    raise SystemExit(2)
"""


class CapabilityManager:
    def __init__(self, graph: GraphStore, root: str | Path, *,
                 reserved_names: set[str] | None = None):
        self.graph = graph
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.reserved_names = reserved_names or set()

    @staticmethod
    def _private_directory(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)

    @staticmethod
    def _write_private(path: Path, content: str) -> None:
        path.write_text(content)
        os.chmod(path, 0o600)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())

    def _quarantine_orphan(self, path: Path) -> None:
        """Move an unpublished artifact aside instead of trusting or deleting it."""
        orphan_root = self.root / ".orphaned"
        self._private_directory(orphan_root)
        destination = orphan_root / (
            f"{path.parent.name}-{path.name}-{new_id('orphan')}")
        os.replace(path, destination)

    def _publish_artifacts(self, name: str, version: int, description: str,
                           parameters: dict[str, Any], code: str,
                           permissions: list[str], tests: list[dict[str, Any]],
                           handler_sha256: str) -> Path:
        """Publish one complete private version directory with a single rename."""
        capability_dir = self.root / name
        self._private_directory(capability_dir)
        version_dir = capability_dir / f"v{version}"
        if version_dir.exists() or version_dir.is_symlink():
            # MAX(version)+1 is chosen while the SQLite write lock is held, so an
            # existing directory at this exact target cannot belong to a committed
            # version. It is residue from a process death between rename and commit.
            self._quarantine_orphan(version_dir)
        staging = Path(tempfile.mkdtemp(
            prefix=f".v{version}.staging-", dir=capability_dir))
        os.chmod(staging, 0o700)
        try:
            self._write_private(staging / "handler.py", code)
            self._write_private(staging / "manifest.json", json.dumps({
                "name": name, "version": version, "description": description,
                "parameters": parameters, "permissions": permissions,
                "handler_sha256": handler_sha256,
            }, indent=2) + "\n")
            self._write_private(
                staging / "tests.json", json.dumps(tests, indent=2) + "\n")
            os.replace(staging, version_dir)
            # Persist the parent directory entry before the surrounding database
            # transaction commits its reference to this artifact.
            fs.fsync_directory(capability_dir)
            return version_dir
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    def _has_committed_version(self, name: str, version: int) -> bool:
        with self.graph._connect() as conn:
            row = conn.execute(
                """SELECT 1 FROM capability_versions v
                   JOIN capability_state s ON s.capability_id=v.capability_id
                   WHERE s.name=? AND v.version=?""", (name, version)).fetchone()
        return row is not None

    def _verify_artifacts(self, version: dict[str, Any]) -> Path:
        """Verify the executable artifact against authoritative code in SQLite."""
        path = self.root / version["name"] / f"v{version['version']}" / "handler.py"
        expected = sha256_text(str(version["code"]))
        try:
            actual = sha256_text(path.read_text())
        except OSError as exc:
            raise RuntimeError("capability artifact integrity check failed: "
                               "handler is unavailable") from exc
        if actual != expected:
            raise RuntimeError("capability artifact integrity check failed: "
                               "handler hash mismatch")
        manifest_path = path.parent / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError("capability artifact integrity check failed: "
                               "manifest is unavailable") from exc
        recorded = manifest.get("handler_sha256")
        if recorded is None:
            # Upgrade pre-integrity-boundary artifacts only after their handler
            # has matched the authoritative code already stored in SQLite.
            manifest["handler_sha256"] = expected
            replacement = path.parent / f".manifest-{new_id('upgrade')}.json"
            self._write_private(replacement, json.dumps(manifest, indent=2) + "\n")
            os.replace(replacement, manifest_path)
            recorded = expected
        if (recorded != expected or manifest.get("name") != version["name"]
                or int(manifest.get("version", -1)) != int(version["version"])):
            raise RuntimeError("capability artifact integrity check failed: "
                               "manifest metadata mismatch")
        return path

    @staticmethod
    def _slug(name: str) -> str:
        slug = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
        if not slug or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", slug):
            raise ValueError("capability name must become a 2-64 character tool name")
        return slug

    @staticmethod
    def _validate_parameters(parameters: dict[str, Any]) -> None:
        if parameters.get("type") != "object":
            raise ValueError("capability parameters must be a JSON object schema")
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("parameter properties must be an object")
        required = parameters.get("required", [])
        if not isinstance(required, list) or any(key not in properties for key in required):
            raise ValueError("required parameters must exist in properties")

    @staticmethod
    def _validate_code(code: str, permissions: list[str]) -> None:
        if len(code) > 20000:
            raise ValueError("capability code exceeds 20,000 characters")
        permission_set = set(permissions)
        unknown = permission_set - ALLOWED_PERMISSIONS
        if unknown:
            raise ValueError(f"unsupported capability permissions: {sorted(unknown)}")
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise ValueError(f"capability syntax error: {exc}") from exc
        # Permission-specific unions must be candidate-local.  Mutating the
        # module-level baseline would let one privileged candidate grant imports
        # to every capability validated later in this process.
        allowed_imports = set(SAFE_IMPORTS)
        if "network" in permission_set:
            allowed_imports |= NETWORK_IMPORTS
        if permission_set & {"filesystem_read", "filesystem_write"}:
            allowed_imports |= FILESYSTEM_IMPORTS
        if "process" in permission_set:
            allowed_imports |= PROCESS_IMPORTS
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        if "run" not in functions:
            raise ValueError("capability code must define run(args)")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
                if not roots <= allowed_imports:
                    raise ValueError(f"imports require unavailable permission: {sorted(roots - allowed_imports)}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root not in allowed_imports:
                    raise ValueError(f"import requires unavailable permission: {root}")
            elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
                raise ValueError(f"unsafe operation is not allowed: {node.id}")
            elif (isinstance(node, ast.Name) and node.id == "open"
                  and not permission_set & {"filesystem_read", "filesystem_write"}):
                raise ValueError("unsafe operation is not allowed: open requires a filesystem permission")
            elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ValueError("dunder attribute access is not allowed")

    @staticmethod
    def _validate_tests(tests: list[dict[str, Any]]) -> None:
        if len(tests) < 2:
            raise ValueError("a capability requires at least two executable tests")
        for test in tests:
            if not isinstance(test.get("args", {}), dict):
                raise ValueError("each capability test needs object args")
            if "expected" not in test and "expected_contains" not in test:
                raise ValueError("each test needs expected or expected_contains")

    def create_version(self, name: str, description: str,
                       parameters: dict[str, Any], code: str,
                       permissions: list[str], tests: list[dict[str, Any]], *,
                       source_node_ids: list[str], actor: str = "friday") -> str:
        slug = self._slug(name)
        if slug in self.reserved_names:
            raise ValueError("capability name collides with a built-in tool")
        if not description.strip():
            raise ValueError("capability requires a description")
        self._validate_parameters(parameters)
        self._validate_code(code, permissions)
        self._validate_tests(tests)
        if not source_node_ids or any(self.graph.get_node(node_id) is None
                                      for node_id in source_node_ids):
            raise ValueError("capability requires valid task provenance")
        now = utc_now()
        with self.graph.transaction() as conn:
            state = conn.execute("SELECT * FROM capability_state WHERE name=?",
                                 (slug,)).fetchone()
            if state:
                capability_id = state["capability_id"]
                version = int(conn.execute(
                    "SELECT COALESCE(MAX(version),0)+1 FROM capability_versions "
                    "WHERE capability_id=?", (capability_id,)).fetchone()[0])
            else:
                event_id, seq = self.graph.append_event(
                    conn, "capability.created", {"name": slug}, actor=actor)
                capability_id = self.graph.append_node(
                    conn, "capability", {"name": slug}, event_id=event_id,
                    node_id=new_id("capability"))
                conn.execute(
                    """INSERT INTO capability_state(capability_id,name,status,created_at,
                       updated_at,last_event_seq) VALUES (?,?,'drafted',?,?,?)""",
                    (capability_id, slug, now, now, seq))
                version = 1
            body = {"name": slug, "version": version, "description": description,
                    "parameters": parameters, "permissions": permissions, "tests": tests}
            event_id, seq = self.graph.append_event(
                conn, "capability.version_drafted", body, actor=actor)
            version_id = self.graph.append_node(
                conn, "capability_version", body, event_id=event_id,
                node_id=new_id("capv"))
            self.graph.append_edge(conn, capability_id, "contains", version_id,
                                   event_id=event_id)
            for source_id in source_node_ids:
                self.graph.append_edge(conn, version_id, "derived_from", source_id,
                                       event_id=event_id)
            conn.execute(
                """INSERT INTO capability_versions(version_id,capability_id,version,
                   description,parameters_json,code,permissions_json,tests_json,status,
                   created_at,last_event_seq) VALUES (?,?,?,?,?,?,?,?,'drafted',?,?)""",
                (version_id, capability_id, version, description,
                 canonical_json(parameters), code, canonical_json(permissions),
                 canonical_json(tests), now, seq))
            conn.execute("UPDATE capability_state SET status='drafted',updated_at=?,"
                         "last_event_seq=? WHERE capability_id=?",
                         (now, seq, capability_id))
            # Publish before committing the database reference. If publication
            # fails, this transaction rolls back. If SQLite commit itself then
            # fails, the next attempt quarantines the unreferenced directory.
            self._publish_artifacts(
                slug, version, description, parameters, code, permissions,
                tests, sha256_text(code))
        return version_id

    def _version(self, version_id: str) -> dict[str, Any]:
        with self.graph._connect() as conn:
            row = conn.execute(
                "SELECT v.*,s.name FROM capability_versions v JOIN capability_state s "
                "ON s.capability_id=v.capability_id WHERE v.version_id=?",
                (version_id,)).fetchone()
        if row is None:
            raise ValueError("capability version does not exist")
        return dict(row)

    @staticmethod
    def sandbox_status() -> tuple[bool, str | None]:
        """Return whether the required fail-closed Bubblewrap boundary can run."""
        if not shutil.which("bwrap"):
            return False, "Bubblewrap is not installed"
        command = [
            "bwrap", "--die-with-parent", "--new-session", "--unshare-all",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib", "/lib64", "--proc", "/proc",
            "--dev", "/dev", "--tmpfs", "/tmp", "/usr/bin/true",
        ]
        try:
            result = subprocess.run(
                command, text=True, capture_output=True, timeout=5,
                env={"PATH": "/usr/bin:/bin"})
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)[:500]
        if result.returncode:
            detail = (result.stderr or result.stdout or
                      "Bubblewrap preflight failed").strip()[-500:]
            return False, detail
        return True, None

    def _run(self, version: dict[str, Any], args: dict[str, Any], *,
             timeout: int = 15, evaluation: bool = False) -> Any:
        path = self._verify_artifacts(version)
        permissions = set(json.loads(version["permissions_json"]))
        evaluation_data = None
        if evaluation:
            evaluation_data = tempfile.TemporaryDirectory(
                prefix=f".{version['name']}-evaluation-", dir=self.root)
            data_dir = Path(evaluation_data.name)
        else:
            data_dir = self.root / version["name"] / "data"
            self._private_directory(data_dir)
        command = [
            "bwrap", "--die-with-parent", "--new-session", "--unshare-all",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib", "/lib64", "--proc", "/proc",
            "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind", str(path.parent),
            "/capability",
        ]
        if "network" in permissions:
            command.append("--share-net")
            for source in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf",
                           "/etc/ssl"):
                if Path(source).exists():
                    command.extend(["--ro-bind", source, source])
        if "filesystem_write" in permissions:
            command.extend(["--bind", str(data_dir), "/data"])
        elif "filesystem_read" in permissions:
            command.extend(["--ro-bind", str(data_dir), "/data"])
        else:
            command.extend(["--dir", "/data"])
        command.extend([
            "--chdir", "/capability", "/usr/bin/python", "-I", "-c", RUNNER,
            "/capability/handler.py",
        ])
        try:
            result = subprocess.run(
                command, cwd=path.parent, text=True, capture_output=True,
                input=canonical_json(args), timeout=timeout,
                env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"})
        finally:
            if evaluation_data is not None:
                evaluation_data.cleanup()
        if len(result.stdout) + len(result.stderr) > 128_000:
            raise RuntimeError("capability exceeded the output limit")
        lines = result.stdout.strip().splitlines()
        if not lines:
            raise RuntimeError((result.stderr or "capability returned no result")[-1000:])
        payload = json.loads(lines[-1])
        if result.returncode or not payload.get("ok"):
            raise RuntimeError(str(payload.get("error") or result.stderr)[-1000:])
        return payload.get("result")

    def evaluate_and_activate(self, version_id: str, *, actor: str = "verifier") -> bool:
        version = self._version(version_id)
        tests = json.loads(version["tests_json"])
        results = []
        sandbox_ready, sandbox_error = self.sandbox_status()
        if sandbox_ready:
            for test in tests:
                try:
                    actual = self._run(
                        version, test.get("args", {}), evaluation=True)
                    passed = (actual == test["expected"] if "expected" in test else
                              str(test["expected_contains"]) in str(actual))
                    results.append({"name": test.get("name", "test"), "passed": passed,
                                    "actual": actual})
                except Exception as exc:
                    results.append({"name": test.get("name", "test"), "passed": False,
                                    "error": str(exc)})
        else:
            results.append({"name": "sandbox_preflight", "passed": False,
                            "infrastructure_error": sandbox_error})
        passed = sandbox_ready and all(result["passed"] for result in results)
        with self.graph.transaction() as conn:
            body = {"version_id": version_id, "passed": passed, "results": results,
                    "infrastructure_ready": sandbox_ready}
            event_id, seq = self.graph.append_event(
                conn, "capability.evaluated", body, actor=actor)
            evaluation_id = self.graph.append_node(conn, "evaluation", body,
                                                   event_id=event_id)
            self.graph.append_edge(conn, version_id, "verified_by", evaluation_id,
                                   event_id=event_id)
            # A broken/missing sandbox says nothing about candidate quality. Keep
            # it drafted so a later healthy verifier can retry it.
            status = ("active" if passed else
                      "quarantined" if sandbox_ready else "drafted")
            conn.execute("UPDATE capability_versions SET status=?,last_event_seq=? "
                         "WHERE version_id=?", (status, seq, version_id))
            state = conn.execute(
                "SELECT active_version_id FROM capability_state WHERE capability_id=?",
                (version["capability_id"],)).fetchone()
            projected_status = (
                "active" if passed or (state and state["active_version_id"])
                else status)
            conn.execute(
                """UPDATE capability_state SET status=?,active_version_id=CASE WHEN ?
                   THEN ? ELSE active_version_id END,updated_at=?,last_event_seq=?
                   WHERE capability_id=?""",
                (projected_status, passed, version_id, utc_now(), seq,
                 version["capability_id"]))
            if passed:
                self.graph.append_edge(conn, version["capability_id"], "activated_as",
                                       version_id, event_id=event_id)
        return passed

    def tool_schemas(self) -> list[dict[str, Any]]:
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT s.name,v.description,v.parameters_json FROM capability_state s
                   JOIN capability_versions v ON v.version_id=s.active_version_id
                   WHERE s.status='active' ORDER BY s.name""").fetchall()
        return [{"type": "function", "function": {
            "name": row["name"], "description": row["description"],
            "parameters": json.loads(row["parameters_json"])}} for row in rows]

    def version_status(self, version_id: str) -> str:
        return str(self._version(version_id)["status"])

    def active_metadata(self, name: str) -> dict[str, Any] | None:
        """Return the immutable executor identity for an active dynamic tool."""
        with self.graph._connect() as conn:
            row = conn.execute(
                """SELECT v.version_id,v.version,v.code,v.permissions_json,s.name
                   FROM capability_state s JOIN capability_versions v
                     ON v.version_id=s.active_version_id
                   WHERE s.name=? AND s.status='active'""", (name,)).fetchone()
        if row is None:
            return None
        return {
            "kind": "capability",
            "name": str(row["name"]),
            "version_id": str(row["version_id"]),
            "version": int(row["version"]),
            "code_sha256": sha256_text(str(row["code"])),
            "permissions": json.loads(row["permissions_json"]),
        }

    def execute_version(self, version_id: str, args: dict[str, Any], *,
                        expected_name: str, expected_version: int,
                        expected_code_sha256: str,
                        expected_permissions: list[str]) -> Any:
        """Execute the exact tested version recorded in a durable step."""
        version = self._version(version_id)
        if version["name"] != expected_name:
            raise RuntimeError("capability executor binding name does not match")
        if int(version["version"]) != int(expected_version):
            raise RuntimeError("capability executor binding version does not match")
        actual_hash = sha256_text(str(version["code"]))
        if actual_hash != expected_code_sha256:
            raise RuntimeError("capability executor binding hash does not match")
        actual_permissions = json.loads(version["permissions_json"])
        if actual_permissions != expected_permissions:
            raise RuntimeError(
                "capability executor binding permissions do not match")
        if version["status"] != "active":
            raise RuntimeError("bound capability version is no longer active")
        parameters = json.loads(version["parameters_json"])
        missing = [key for key in parameters.get("required", []) if key not in args]
        if missing:
            raise ValueError(f"missing capability arguments: {missing}")
        return self._run(version, args)

    def execute(self, name: str, args: dict[str, Any]) -> Any:
        with self.graph._connect() as conn:
            row = conn.execute(
                """SELECT v.*,s.name FROM capability_state s JOIN capability_versions v
                   ON v.version_id=s.active_version_id
                   WHERE s.name=? AND s.status='active'""", (name,)).fetchone()
        if row is None:
            raise ValueError("active capability does not exist")
        version = dict(row)
        return self.execute_version(
            str(version["version_id"]), args,
            expected_name=str(version["name"]),
            expected_version=int(version["version"]),
            expected_code_sha256=sha256_text(str(version["code"])),
            expected_permissions=json.loads(version["permissions_json"]))

    def list(self) -> list[dict[str, Any]]:
        with self.graph._connect() as conn:
            rows = conn.execute("SELECT * FROM capability_state ORDER BY name").fetchall()
        return [dict(row) for row in rows]

    def active_names(self) -> set[str]:
        return {item["name"] for item in self.list() if item["status"] == "active"}
