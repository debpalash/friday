"""Transactional, test-gated local file deployment with rollback."""

from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .graph import GraphStore, new_id, sha256_text, utc_now


class DeploymentManager:
    PROTECTED_ROOTS = {
        ".git", "backups", "capabilities", "models", "state", "venv",
    }
    PROTECTED_FILES = {
        ".env", "friday.log", "server.log", "session.json", "supervisor.log",
    }

    def __init__(self, graph: GraphStore, repo: str | Path,
                 test_command: list[str] | None = None):
        self.graph = graph
        self.repo = Path(repo).resolve()
        self.checkpoints = self.repo / "state" / "checkpoints"
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.test_command = test_command or [
            sys.executable, "-m", "unittest", "discover", "-s", "tests"]

    def _target(self, relative_path: str) -> Path:
        raw = Path(relative_path)
        if raw.is_absolute() or not raw.parts:
            raise ValueError("deployment target must be a project-relative path")
        target = (self.repo / relative_path).resolve()
        if self.repo not in target.parents or target == self.repo:
            raise ValueError("deployment target must be inside the project")
        normalized = target.relative_to(self.repo)
        if normalized.parts[0] in self.PROTECTED_ROOTS:
            raise ValueError("deployment target is protected runtime state")
        name = target.name.casefold()
        if (target.name in self.PROTECTED_FILES
                or name.startswith((".env.", "api_key"))
                or "secret" in name or "token" in name):
            raise ValueError("deployment target is protected runtime state")
        return target

    @staticmethod
    def _protect_existing_test(relative_path: str, target: Path) -> None:
        parts = Path(relative_path).parts
        if parts and parts[0] == "tests" and target.is_file():
            raise ValueError(
                "automated deployment may not modify an existing acceptance test")

    def stage_write(self, relative_path: str, content: str, *,
                    task_id: str | None = None) -> dict[str, Any]:
        target = self._target(relative_path)
        self._protect_existing_test(relative_path, target)
        deployment_id = new_id("deploy")
        checkpoint_dir = self.checkpoints / deployment_id
        checkpoint_dir.mkdir(parents=True)
        checkpoint = checkpoint_dir / "before"
        existed = target.is_file()
        before = target.read_text(errors="replace") if existed else ""
        if existed:
            shutil.copy2(target, checkpoint)
        (checkpoint_dir / "metadata").write_text(
            f"path={relative_path}\nexisted={int(existed)}\n")
        stage = target.with_name(f".{target.name}.{deployment_id}.stage")
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_text(content)
        status, output = "testing", ""
        swapped = False
        try:
            if target.suffix == ".py":
                compile(content, str(target), "exec")
            os.replace(stage, target)
            swapped = True
            result = subprocess.run(self.test_command, cwd=self.repo, text=True,
                                    capture_output=True, timeout=120)
            output = (result.stdout + result.stderr)[-12000:]
            if result.returncode:
                status = "rejected"
                raise RuntimeError(f"verification failed:\n{output}")
            status = "promoted"
        except Exception:
            status = "rejected"
            if swapped:
                if existed:
                    shutil.copy2(checkpoint, target)
                elif target.exists():
                    target.unlink()
            raise
        finally:
            if stage.exists():
                stage.unlink()
            body = {"deployment_id": deployment_id, "target": relative_path,
                    "before_sha256": sha256_text(before) if existed else None,
                    "after_sha256": sha256_text(content), "status": status,
                    "test_output": output[-4000:]}
            with self.graph.transaction() as conn:
                event_id, seq = self.graph.append_event(
                    conn, "deployment.finished", body, actor="deployer",
                    task_id=task_id)
                self.graph.append_node(conn, "checkpoint",
                                       {"path": str(checkpoint), "existed": existed},
                                       event_id=event_id)
                node_id = self.graph.append_node(
                    conn, "deployment", body, event_id=event_id,
                    node_id=deployment_id)
                now = utc_now()
                conn.execute(
                    """INSERT INTO deployment_state(deployment_id,target_path,
                       checkpoint_path,before_sha256,after_sha256,status,test_output,
                       created_at,updated_at,last_event_seq)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (node_id, relative_path, str(checkpoint) if existed else None,
                     body["before_sha256"], body["after_sha256"], status,
                     body["test_output"], now, now, seq))
        return body

    def stage_bundle(self, changes: dict[str, str], *,
                     task_id: str | None = None, preverified: bool = False,
                     verification_output: str = "") -> dict[str, Any]:
        """Promote a verified multi-file core change as one rollback unit."""
        if not changes:
            raise ValueError("deployment bundle is empty")
        targets = {path: self._target(path) for path in changes}
        for relative, target in targets.items():
            self._protect_existing_test(relative, target)
        deployment_id = new_id("deploy")
        checkpoint_dir = self.checkpoints / deployment_id
        checkpoint_dir.mkdir(parents=True)
        metadata: dict[str, Any] = {"files": {}}
        stages: dict[str, Path] = {}
        before_hashes: dict[str, str | None] = {}
        swapped: list[str] = []
        status, output = "testing", ""
        try:
            for relative, target in targets.items():
                content = changes[relative]
                if target.suffix == ".py":
                    compile(content, str(target), "exec")
                existed = target.is_file()
                before = target.read_text(errors="replace") if existed else ""
                before_hashes[relative] = sha256_text(before) if existed else None
                metadata["files"][relative] = {"existed": existed}
                if existed:
                    saved = checkpoint_dir / "files" / relative
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, saved)
                stage = checkpoint_dir / "staged" / relative
                stage.parent.mkdir(parents=True, exist_ok=True)
                stage.write_text(content)
                stages[relative] = stage
            (checkpoint_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n")
            for relative, target in targets.items():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stages[relative], target)
                swapped.append(relative)
            if preverified:
                output = verification_output[-12000:]
            else:
                result = subprocess.run(self.test_command, cwd=self.repo, text=True,
                                        capture_output=True, timeout=180)
                output = (result.stdout + result.stderr)[-12000:]
                if result.returncode:
                    raise RuntimeError(f"bundle verification failed:\n{output}")
            status = "promoted"
        except Exception:
            status = "rejected"
            for relative in reversed(swapped):
                target = targets[relative]
                saved = checkpoint_dir / "files" / relative
                if metadata["files"][relative]["existed"]:
                    shutil.copy2(saved, target)
                elif target.exists():
                    target.unlink()
            raise
        finally:
            body = {"deployment_id": deployment_id,
                    "target": sorted(changes), "before_sha256": before_hashes,
                    "after_sha256": {path: sha256_text(content)
                                     for path, content in changes.items()},
                    "status": status, "test_output": output[-4000:]}
            with self.graph.transaction() as conn:
                event_id, seq = self.graph.append_event(
                    conn, "deployment.bundle_finished", body, actor="deployer",
                    task_id=task_id)
                self.graph.append_node(
                    conn, "checkpoint", {"path": str(checkpoint_dir),
                                         "files": sorted(changes)}, event_id=event_id)
                node_id = self.graph.append_node(
                    conn, "deployment", body, event_id=event_id,
                    node_id=deployment_id)
                now = utc_now()
                conn.execute(
                    """INSERT INTO deployment_state(deployment_id,target_path,
                       checkpoint_path,before_sha256,after_sha256,status,test_output,
                       created_at,updated_at,last_event_seq)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (node_id, "bundle:" + canonical_bundle(sorted(changes)),
                     str(checkpoint_dir), sha256_text(str(before_hashes)),
                     sha256_text(str(body["after_sha256"])), status,
                     body["test_output"], now, now, seq))
        return body

    def rollback(self, deployment_id: str) -> None:
        with self.graph._connect() as conn:
            row = conn.execute("SELECT * FROM deployment_state WHERE deployment_id=?",
                               (deployment_id,)).fetchone()
        if row is None or row["status"] != "promoted":
            raise ValueError("deployment is not rollbackable")
        if row["target_path"].startswith("bundle:"):
            checkpoint_dir = Path(row["checkpoint_path"])
            metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
            for relative, item in metadata["files"].items():
                target = self._target(relative)
                saved = checkpoint_dir / "files" / relative
                if item["existed"]:
                    shutil.copy2(saved, target)
                elif target.exists():
                    target.unlink()
            self._record_rollback(deployment_id)
            return
        target = self._target(row["target_path"])
        checkpoint = Path(row["checkpoint_path"]) if row["checkpoint_path"] else None
        if checkpoint and checkpoint.is_file():
            shutil.copy2(checkpoint, target)
        elif target.exists():
            target.unlink()
        self._record_rollback(deployment_id)

    def _record_rollback(self, deployment_id: str) -> None:
        with self.graph.transaction() as conn:
            event_id, seq = self.graph.append_event(
                conn, "deployment.rolled_back", {"deployment_id": deployment_id},
                actor="deployer")
            conn.execute("UPDATE deployment_state SET status='rolled_back',updated_at=?,"
                         "last_event_seq=? WHERE deployment_id=?",
                         (utc_now(), seq, deployment_id))


def canonical_bundle(paths: list[str]) -> str:
    return json.dumps(paths, separators=(",", ":"))
