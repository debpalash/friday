"""Persistent, test-gated voice profile lifecycle."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .graph import GraphStore, canonical_json, new_id, utc_now


AUDIO_SUFFIXES = {".mp3", ".wav", ".flac"}


class VoiceManager:
    def __init__(self, graph: GraphStore, root: str | Path):
        self.graph = graph
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slug(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if not slug:
            raise ValueError("voice profile name has no usable characters")
        return slug

    def _record_profile(self, name: str, kind: str, config: dict[str, Any], *,
                        status: str, actor: str,
                        source_node_ids: list[str] | None = None) -> str:
        now = utc_now()
        with self.graph.transaction() as conn:
            existing = conn.execute("SELECT voice_id FROM voice_profiles WHERE name=?",
                                    (name,)).fetchone()
            if existing:
                return str(existing["voice_id"])
            body = {"name": name, "kind": kind, "config": config, "status": status}
            event_id, seq = self.graph.append_event(
                conn, "voice.profile_created", body, actor=actor)
            voice_id = self.graph.append_node(
                conn, "voice_profile", body, event_id=event_id,
                node_id=new_id("voice"))
            for source_id in source_node_ids or []:
                self.graph.append_edge(conn, voice_id, "derived_from", source_id,
                                       event_id=event_id)
            conn.execute(
                """INSERT INTO voice_profiles(voice_id,name,kind,config_json,status,
                   created_at,updated_at,last_event_seq) VALUES (?,?,?,?,?,?,?,?)""",
                (voice_id, name, kind, canonical_json(config), status, now, now, seq))
            return voice_id

    def discover(self) -> None:
        base_id = self._record_profile(
            "base", "instruction",
            {"instruct": "female, young adult, moderate pitch"},
            status="validated", actor="system")
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            clips = [path for path in sorted(folder.iterdir())
                     if path.suffix.lower() in AUDIO_SUFFIXES]
            manifest = folder / "profile.json"
            config = {}
            if manifest.is_file():
                try:
                    config = json.loads(manifest.read_text())
                except (OSError, json.JSONDecodeError):
                    config = {}
            if clips:
                config = {"reference": str(clips[0].relative_to(self.root.parent.parent)),
                          "instruct": config.get("instruct", "")}
                self._record_profile(self._slug(folder.name), "clone", config,
                                     status="candidate", actor="system")
        with self.graph.transaction() as conn:
            runtime = conn.execute("SELECT 1 FROM voice_runtime WHERE singleton=1").fetchone()
            if runtime is None:
                event_id, seq = self.graph.append_event(
                    conn, "voice.runtime_initialized", {"voice_id": base_id},
                    actor="system")
                conn.execute(
                    """INSERT INTO voice_runtime(singleton,active_voice_id,
                       previous_voice_id,updated_at,last_event_seq)
                       VALUES (1,?,NULL,?,?)""", (base_id, utc_now(), seq))
                conn.execute("UPDATE voice_profiles SET status='active',updated_at=?,"
                             "last_event_seq=? WHERE voice_id=?",
                             (utc_now(), seq, base_id))

    def create(self, name: str, instruct: str, *, reference: str | None = None,
               source_node_ids: list[str], actor: str = "friday") -> str:
        slug = self._slug(name)
        with self.graph._connect() as conn:
            if conn.execute("SELECT 1 FROM voice_profiles WHERE name=?",
                            (slug,)).fetchone():
                raise ValueError("voice profile name already exists")
        if not instruct.strip() and not reference:
            raise ValueError("voice profile needs an instruction or reference audio")
        if not source_node_ids or any(self.graph.get_node(node_id) is None
                                      for node_id in source_node_ids):
            raise ValueError("voice profile requires valid task provenance")
        kind = "instruction"
        config: dict[str, Any] = {"instruct": instruct.strip()}
        if reference:
            target = (self.root.parent.parent / reference).resolve()
            if self.root not in target.parents or target.suffix.lower() not in AUDIO_SUFFIXES:
                raise ValueError("voice reference must be an audio file under persona/voices")
            if not target.is_file():
                raise ValueError("voice reference does not exist")
            self._probe_audio(target)
            kind = "clone"
            config["reference"] = str(target.relative_to(self.root.parent.parent))
        voice_id = self._record_profile(
            slug, kind, config, status="candidate", actor=actor,
            source_node_ids=source_node_ids)
        profile_dir = self.root / slug
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "profile.json").write_text(json.dumps(config, indent=2) + "\n")
        return voice_id

    @staticmethod
    def _probe_audio(path: Path) -> None:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            text=True, capture_output=True, timeout=10)
        if result.returncode:
            raise ValueError(f"invalid voice reference: {result.stderr[-300:]}")
        duration = float(result.stdout.strip())
        if not 0.5 <= duration <= 60:
            raise ValueError("voice reference duration must be 0.5-60 seconds")

    def get(self, name: str) -> dict[str, Any]:
        with self.graph._connect() as conn:
            row = conn.execute("SELECT * FROM voice_profiles WHERE name=?",
                               (self._slug(name),)).fetchone()
        if row is None:
            raise ValueError("voice profile does not exist")
        return dict(row) | {"config": json.loads(row["config_json"])}

    def active(self) -> dict[str, Any]:
        with self.graph._connect() as conn:
            row = conn.execute(
                """SELECT p.* FROM voice_runtime r JOIN voice_profiles p
                   ON p.voice_id=r.active_voice_id WHERE r.singleton=1""").fetchone()
        if row is None:
            raise RuntimeError("voice runtime is not initialized")
        return dict(row) | {"config": json.loads(row["config_json"])}

    def activate(self, name: str, verification: dict[str, Any], *,
                 actor: str = "verifier") -> dict[str, Any]:
        profile = self.get(name)
        if verification.get("passed") is not True:
            raise ValueError("voice verification did not pass")
        with self.graph.transaction() as conn:
            runtime = conn.execute("SELECT * FROM voice_runtime WHERE singleton=1").fetchone()
            previous = runtime["active_voice_id"]
            body = {"voice_id": profile["voice_id"], "previous": previous,
                    "verification": verification}
            event_id, seq = self.graph.append_event(
                conn, "voice.activated", body, actor=actor)
            evaluation_id = self.graph.append_node(
                conn, "evaluation", verification, event_id=event_id)
            self.graph.append_edge(conn, profile["voice_id"], "verified_by",
                                   evaluation_id, event_id=event_id)
            conn.execute("UPDATE voice_profiles SET status='validated',updated_at=?,"
                         "last_event_seq=? WHERE voice_id=? AND status='active'",
                         (utc_now(), seq, previous))
            conn.execute("UPDATE voice_profiles SET status='active',updated_at=?,"
                         "last_event_seq=? WHERE voice_id=?",
                         (utc_now(), seq, profile["voice_id"]))
            conn.execute(
                """UPDATE voice_runtime SET active_voice_id=?,previous_voice_id=?,
                   updated_at=?,last_event_seq=? WHERE singleton=1""",
                (profile["voice_id"], previous, utc_now(), seq))
            self.graph.append_edge(conn, profile["voice_id"], "activated_as",
                                   evaluation_id, event_id=event_id)
        return self.get(name)

    def previous(self) -> dict[str, Any] | None:
        with self.graph._connect() as conn:
            row = conn.execute(
                """SELECT p.* FROM voice_runtime r JOIN voice_profiles p
                   ON p.voice_id=r.previous_voice_id WHERE r.singleton=1""").fetchone()
        return (dict(row) | {"config": json.loads(row["config_json"])}) if row else None

    def list(self) -> list[dict[str, Any]]:
        with self.graph._connect() as conn:
            rows = conn.execute("SELECT * FROM voice_profiles ORDER BY name").fetchall()
        return [dict(row) | {"config": json.loads(row["config_json"])} for row in rows]
