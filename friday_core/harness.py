"""Sandboxed Pi maintenance worker for review-gated core-upgrade candidates."""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

from .deployments import DeploymentManager
from .graph import GraphStore, canonical_json, new_id, utc_now
from .tasks import TaskService


ROOT_FILES = {"server.py", "supervisor.py", "system_prompt.md", "friday.py"}
ROOT_DIRS = {"friday_core", "docs", "ops", "tests"}
IGNORED_NAMES = {"__pycache__", ".pytest_cache"}


class CoreUpgradeHarness:
    def __init__(self, graph: GraphStore, tasks: TaskService,
                 deployer: DeploymentManager, repo: str | Path, *,
                 pi_binary: str | Path | None = None,
                 api_key: str = "local-friday",
                 local_base_url: str | None = None,
                 local_model: str | None = None,
                 context_tokens: int | None = None):
        self.graph = graph
        self.tasks = tasks
        self.deployer = deployer
        self.repo = Path(repo).resolve()
        self.root = self.repo / "state" / "upgrades"
        self.root.mkdir(parents=True, exist_ok=True)
        self.pi_binary = Path(pi_binary or os.environ.get(
            "FRIDAY_PI_BIN", "/home/pal/.local/share/mise/installs/pi/latest/pi/pi"))
        self.api_key = api_key
        self.local_base_url = (local_base_url or os.environ.get(
            "FRIDAY_LOCAL_BASE_URL", "http://127.0.0.1:18021/v1")).rstrip("/")
        self.local_model = local_model or os.environ.get(
            "FRIDAY_LOCAL_MODEL", "qwen3.8-27b")
        self.context_tokens = context_tokens or int(os.environ.get(
            "FRIDAY_MODEL_CONTEXT_TOKENS", "8192"))
        parsed = urllib.parse.urlparse(self.local_base_url)
        if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
            raise ValueError("maintenance model URL must be an explicit local HTTP URL")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("maintenance model URL must use loopback")
        self.local_host, self.local_port = parsed.hostname, parsed.port

    @staticmethod
    def _hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _allowed(relative: Path) -> bool:
        if any(part in IGNORED_NAMES for part in relative.parts):
            return False
        if len(relative.parts) == 1:
            return relative.name in ROOT_FILES
        return relative.parts[0] in ROOT_DIRS

    def _copy_source(self, workspace: Path) -> dict[str, str]:
        before: dict[str, str] = {}
        for root_file in ROOT_FILES:
            source = self.repo / root_file
            if source.is_file():
                target = workspace / root_file
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                before[root_file] = self._hash(source)
        for root_dir in ROOT_DIRS:
            source_dir = self.repo / root_dir
            if not source_dir.is_dir():
                continue
            for source in source_dir.rglob("*"):
                if not source.is_file():
                    continue
                relative = source.relative_to(self.repo)
                if not self._allowed(relative):
                    continue
                target = workspace / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                before[str(relative)] = self._hash(source)
        return before

    def _write_pi_config(self, config: Path) -> None:
        config.mkdir(parents=True, exist_ok=True)
        provider = {
            # Pi sees the inner end of a Unix-socket bridge, never the host network.
            "baseUrl": "http://127.0.0.1:18021/v1",
            "api": "openai-completions", "apiKey": self.api_key,
            "compat": {"thinkingFormat": "qwen-chat-template",
                       "supportsDeveloperRole": True,
                       "supportsReasoningEffort": False,
                       "supportsUsageInStreaming": True,
                       "maxTokensField": "max_tokens"},
            "models": [{"id": self.local_model, "name": "Friday maintenance model",
                        "reasoning": False, "input": ["text"],
                        "contextWindow": self.context_tokens, "maxTokens": 2048,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0,
                                 "cacheWrite": 0}}],
        }
        (config / "models.json").write_text(json.dumps(
            {"providers": {"friday-local": provider}}, indent=2) + "\n")
        (config / "settings.json").write_text(json.dumps(
            {"defaultProvider": "friday-local", "defaultModel": self.local_model,
             "enableInstallTelemetry": False}, indent=2) + "\n")

    def _new_upgrade(self, task_id: str, objective: str,
                     workspace: Path) -> str:
        body = {"task_id": task_id, "objective": objective,
                "backend": "pi", "workspace": str(workspace), "status": "staging"}
        with self.graph.transaction() as conn:
            event_id, seq = self.graph.append_event(
                conn, "core_upgrade.created", body, actor="harness", task_id=task_id)
            upgrade_id = self.graph.append_node(
                conn, "core_upgrade", body, event_id=event_id,
                node_id=new_id("upgrade"))
            self.graph.append_edge(conn, task_id, "attempts", upgrade_id,
                                   event_id=event_id)
            now = utc_now()
            conn.execute(
                """INSERT INTO core_upgrade_state(upgrade_id,task_id,objective,backend,
                   workspace_path,status,created_at,updated_at,last_event_seq)
                   VALUES (?,?,?,'pi',?,'staging',?,?,?)""",
                (upgrade_id, task_id, objective, str(workspace), now, now, seq))
        return upgrade_id

    def _status(self, upgrade_id: str, task_id: str, status: str, *,
                changed: list[str] | None = None, deployment_id: str | None = None,
                error: str | None = None) -> None:
        body = {"upgrade_id": upgrade_id, "status": status,
                "changed": changed or [], "deployment_id": deployment_id,
                "error": error}
        with self.graph.transaction() as conn:
            _event_id, seq = self.graph.append_event(
                conn, "core_upgrade.status", body, actor="harness", task_id=task_id)
            conn.execute(
                """UPDATE core_upgrade_state SET status=?,changed_json=?,deployment_id=?,
                   last_error=?,updated_at=?,last_event_seq=? WHERE upgrade_id=?""",
                (status, canonical_json(changed or []), deployment_id, error,
                 utc_now(), seq, upgrade_id))

    def _pi_command(self, workspace: Path, config: Path,
                    objective: str) -> tuple[list[str], Path]:
        if not self.pi_binary.is_file() or not shutil.which("bwrap"):
            raise RuntimeError("Pi or Bubblewrap is unavailable")
        prompt = (
            "You are Friday's untrusted maintenance worker. Modify only /workspace to "
            "complete the objective. Inspect before editing. Do not merely describe work. "
            "Do not weaken or delete existing tests. Add focused tests when appropriate. "
            "You cannot deploy or restart anything; the trusted parent will independently "
            f"verify your staged files. Objective: {objective}")
        pi_args = [
            "/opt/pi-dist/pi", "--mode", "json", "--print", "--provider",
            "friday-local", "--model", self.local_model, "--thinking", "off",
            "--no-session", "--no-extensions", "--no-skills", "--no-context-files",
            "--no-approve", "--tools", "read,edit,write,grep,find,ls,bash", prompt,
        ]
        bridge_socket = config / "qwen.sock"
        inner = (
            "socat TCP-LISTEN:18021,bind=127.0.0.1,fork,reuseaddr "
            "UNIX-CONNECT:/config/qwen.sock & bridge=$!; "
            '"$@"; status=$?; kill "$bridge" 2>/dev/null; exit "$status"')
        command = [
            "bwrap", "--die-with-parent", "--unshare-all",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--dir", "/opt", "--ro-bind", str(self.pi_binary.parent), "/opt/pi-dist",
            "--dir", "/config", "--bind", str(config), "/config",
            "--dir", "/workspace", "--bind", str(workspace), "/workspace",
            "--dir", "/home", "--dir", "/home/friday",
            "--setenv", "HOME", "/home/friday",
            "--setenv", "PI_CODING_AGENT_DIR", "/config",
            "--setenv", "PI_OFFLINE", "1", "--chdir", "/workspace",
            "/bin/sh", "-c", inner, "friday-pi", *pi_args,
        ]
        return command, bridge_socket

    def _run_pi(self, command: list[str], task_id: str, *, bridge_socket: Path,
                timeout: int = 600) -> None:
        bridge_socket.unlink(missing_ok=True)
        bridge = subprocess.Popen(
            ["socat", f"UNIX-LISTEN:{bridge_socket},fork,mode=600",
             f"TCP:{self.local_host}:{self.local_port}"], stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True)
        for _ in range(50):
            if bridge_socket.exists():
                break
            if bridge.poll() is not None:
                raise RuntimeError("Qwen bridge failed: " + bridge.stderr.read()[-1000:])
            time.sleep(0.02)
        if not bridge_socket.exists():
            bridge.terminate()
            raise RuntimeError("Qwen bridge socket was not created")
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, bufsize=1)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout
        final_error = None
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    process.kill()
                    raise TimeoutError("Pi maintenance job exceeded ten minutes")
                for key, _ in selector.select(timeout=1):
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event_type = event.get("type")
                    if event_type == "tool_execution_start":
                        tool = event.get("toolName") or event.get("tool", "tool")
                        self.tasks.publish(task_id, "harness", "running",
                                           f"Pi started {tool}",
                                           "Executing inside the isolated staging workspace.")
                    elif event_type == "tool_execution_end":
                        tool = event.get("toolName") or event.get("tool", "tool")
                        self.tasks.publish(task_id, "harness", "updated",
                                           f"Pi completed {tool}",
                                           "The result remains staged and untrusted.")
                    elif event_type == "message_end":
                        message = event.get("message", {})
                        if message.get("stopReason") == "error":
                            final_error = message.get("errorMessage")
            stderr = process.stderr.read()[-4000:]
            if process.returncode or final_error:
                raise RuntimeError(final_error or stderr or "Pi maintenance worker failed")
        finally:
            if process.poll() is None:
                process.terminate()
            bridge.terminate()
            try:
                bridge.wait(timeout=2)
            except subprocess.TimeoutExpired:
                bridge.kill()
            bridge_socket.unlink(missing_ok=True)

    def _collect_changes(self, workspace: Path,
                         before: dict[str, str]) -> dict[str, str]:
        staged: dict[str, Path] = {}
        unexpected: list[str] = []
        for path in workspace.rglob("*"):
            relative = path.relative_to(workspace)
            if path.is_symlink():
                unexpected.append(f"{relative} (symlink)")
                continue
            if not path.is_file():
                continue
            if any(part in IGNORED_NAMES for part in relative.parts):
                # Generated caches are not promotion inputs and must not be
                # allowed to affect the subsequent verifier either.  In
                # particular, an unchecked-hash .pyc could otherwise shadow a
                # protected source file while remaining absent from `changes`.
                path.unlink()
                continue
            if not self._allowed(relative):
                unexpected.append(str(relative))
                continue
            staged[str(relative)] = path
        if unexpected:
            raise RuntimeError(
                f"core worker created files outside the promotion bundle: {sorted(unexpected)}")
        deleted = sorted(set(before) - set(staged))
        if deleted:
            raise RuntimeError(f"core worker may not delete files: {deleted}")
        changed = {relative: path.read_text(errors="strict")
                   for relative, path in staged.items()
                   if relative not in before or self._hash(path) != before[relative]}
        for relative in changed:
            if relative.startswith("tests/") and relative in before:
                raise RuntimeError(f"core worker may not alter existing test: {relative}")
            if len(changed[relative]) > 300000:
                raise RuntimeError(f"staged file is too large: {relative}")
        if not changed:
            raise RuntimeError("Pi completed without producing a code change")
        return changed

    def _verify_staged(self, workspace: Path, job_dir: Path) -> str:
        """Run generated code/tests without host network, home, or live-tree access."""
        python_root = Path(sys.base_prefix).resolve()
        site_packages = (Path(sys.prefix) / "lib" /
                         f"python{sys.version_info.major}.{sys.version_info.minor}" /
                         "site-packages").resolve()
        command = [
            "bwrap", "--die-with-parent", "--unshare-all",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--dir", "/python", "--ro-bind", str(python_root), "/python",
            "--dir", "/site", "--ro-bind", str(site_packages), "/site",
            "--dir", "/workspace", "--bind", str(workspace), "/workspace",
            "--dir", "/home", "--dir", "/home/friday",
            "--setenv", "HOME", "/tmp", "--setenv", "PYTHONPATH", "/site",
            "--setenv", "PYTHONNOUSERSITE", "1", "--setenv", "FRIDAY_VOICE_CLONE", "0",
            "--setenv", "FRIDAY_LOCAL_BASE_URL", self.local_base_url,
            "--setenv", "FRIDAY_LOCAL_MODEL", self.local_model,
            "--setenv", "FRIDAY_MODEL_CONTEXT_TOKENS", str(self.context_tokens),
            "--chdir", "/workspace", f"/python/bin/{Path(sys.executable).resolve().name}",
            "-B", "-m", "unittest", "discover", "-s", "tests", "-v",
        ]
        result = subprocess.run(command, text=True, capture_output=True, timeout=180)
        output = result.stdout + result.stderr
        if result.returncode:
            raise RuntimeError("sandboxed acceptance tests failed:\n" + output[-5000:])
        return output

    def upgrade(self, objective: str, *, task_id: str) -> dict[str, Any]:
        if not objective.strip():
            raise ValueError("core upgrade objective is empty")
        job_dir = self.root / new_id("job")
        workspace, config = job_dir / "workspace", job_dir / "pi-config"
        workspace.mkdir(parents=True)
        before = self._copy_source(workspace)
        self._write_pi_config(config)
        upgrade_id = self._new_upgrade(task_id, objective, workspace)
        try:
            self.tasks.publish(task_id, "harness", "running",
                               "Pi maintenance worker started",
                               "Core files are writable only in an isolated staging workspace.")
            self._status(upgrade_id, task_id, "agent_running")
            command, bridge_socket = self._pi_command(workspace, config, objective)
            self._run_pi(command, task_id, bridge_socket=bridge_socket)
            changes = self._collect_changes(workspace, before)
            changed_paths = sorted(changes)
            self._status(upgrade_id, task_id, "verifying", changed=changed_paths)
            self.tasks.publish(task_id, "harness", "verifying",
                               "Running independent acceptance tests",
                               f"Staged files: {', '.join(changed_paths)}")
            verification_output = self._verify_staged(workspace, job_dir)
            # Candidate code participates in the test process and is therefore
            # capable of terminating or manipulating that process.  Sandbox
            # isolation protects the host, but a zero exit status is not enough
            # authority to modify the live assistant.  Preserve the complete
            # candidate for a separate, diff-specific human review; never pass
            # an untrusted run through DeploymentManager as `preverified`.
            self._status(
                upgrade_id, task_id, "awaiting_review", changed=changed_paths)
            self.tasks.publish(
                task_id, "harness", "awaiting_review",
                "Core candidate is ready for explicit review",
                "Tests were informative only; no live files were changed.")
            return {"upgrade_id": upgrade_id, "changed": changed_paths,
                    "workspace": str(workspace),
                    "verification_output": verification_output[-4000:],
                    "status": "awaiting_review"}
        except Exception as exc:
            self._status(upgrade_id, task_id, "rejected", error=str(exc)[:2000])
            raise

    def list(self) -> list[dict[str, Any]]:
        with self.graph._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM core_upgrade_state ORDER BY updated_at DESC").fetchall()
        return [dict(row) | {"changed": json.loads(row["changed_json"])} for row in rows]
