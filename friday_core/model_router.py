"""Local-first model routing with explicit, redacted remote disclosures."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from typing import Any

from .graph import GraphStore, canonical_json, new_id, utc_now
from .public_http import (normalize_public_http_url, request_public_http)


SECRET_PATTERN = re.compile(
    r"(?i)(api[_ -]?key|authorization|password|secret|token)(\s*[:=]\s*)([^\s,;]+)")


class ModelRouter:
    def __init__(self, graph: GraphStore, *, local_base_url: str,
                 local_model: str, remote_base_url: str | None = None,
                 remote_model: str | None = None):
        self.graph = graph
        self.local_base_url = local_base_url
        self.local_model = local_model
        self.remote_base_url = remote_base_url or os.environ.get("FRIDAY_REMOTE_BASE_URL")
        self.remote_model = remote_model or os.environ.get("FRIDAY_REMOTE_MODEL")

    @property
    def remote_enabled(self) -> bool:
        return bool(self.remote_base_url and self.remote_model
                    and os.environ.get("FRIDAY_REMOTE_API_KEY"))

    @staticmethod
    def redact(value: Any) -> tuple[Any, list[str]]:
        redactions: list[str] = []

        def walk(item: Any, path: str) -> Any:
            if isinstance(item, dict):
                output = {}
                for key, child in item.items():
                    child_path = f"{path}.{key}" if path else str(key)
                    if re.search(r"(?i)(password|secret|token|api.?key|authorization)",
                                 str(key)):
                        output[key] = "[REDACTED]"
                        redactions.append(child_path)
                    else:
                        output[key] = walk(child, child_path)
                return output
            if isinstance(item, list):
                return [walk(child, f"{path}[{index}]")
                        for index, child in enumerate(item)]
            if isinstance(item, str):
                replaced, count = SECRET_PATTERN.subn(r"\1\2[REDACTED]", item)
                if count:
                    redactions.append(path or "text")
                return replaced
            return item

        return walk(value, ""), redactions

    def disclosure(self, payload: Any, *, task_id: str | None = None,
                   approved: bool = False) -> dict[str, Any]:
        if not self.remote_enabled:
            raise RuntimeError("remote reasoning is not configured")
        redacted, paths = self.redact(payload)
        encoded = canonical_json(redacted)
        body = {"task_id": task_id, "provider": self.remote_base_url,
                "model": self.remote_model,
                "payload_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "redactions": paths, "approved": approved}
        with self.graph.transaction() as conn:
            event_id, seq = self.graph.append_event(
                conn, "model.disclosure_prepared", body, actor="model_router",
                task_id=task_id)
            disclosure_id = self.graph.append_node(
                conn, "disclosure", body, event_id=event_id,
                node_id=new_id("disclosure"))
            conn.execute(
                """INSERT INTO model_disclosures
                   (disclosure_id,task_id,provider,model,payload_sha256,
                    redaction_json,approved,created_at,last_event_seq)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (disclosure_id, task_id, self.remote_base_url, self.remote_model,
                 body["payload_sha256"], json.dumps(paths), int(approved),
                 utc_now(), seq))
        return {"disclosure_id": disclosure_id, "payload_preview": redacted, **body}

    def status(self) -> dict[str, Any]:
        return {"default": "local", "local_model": self.local_model,
                "local_base_url": self.local_base_url,
                "remote_enabled": self.remote_enabled,
                "remote_model": self.remote_model if self.remote_enabled else None}

    def complete(self, prompt: str, *, task_id: str,
                 max_tokens: int = 600) -> dict[str, Any]:
        if not self.remote_enabled:
            raise RuntimeError("remote reasoning is not configured")
        try:
            remote_base = normalize_public_http_url(
                str(self.remote_base_url))
        except ValueError as exc:
            raise RuntimeError(
                "remote reasoning requires a public HTTPS endpoint") from exc
        parsed = urllib.parse.urlsplit(remote_base)
        if parsed.scheme != "https" or parsed.query or parsed.fragment:
            raise RuntimeError(
                "remote reasoning requires a public HTTPS endpoint")
        payload = {"model": self.remote_model,
                   "messages": [{"role": "system", "content": (
                       "Solve the supplied task. Do not claim actions or private context "
                       "that are not present in the payload.")},
                                {"role": "user", "content": prompt}],
                   "max_tokens": min(max(max_tokens, 1), 1200)}
        redacted, _paths = self.redact(payload)
        disclosure = self.disclosure(redacted, task_id=task_id, approved=True)
        endpoint = remote_base.rstrip("/") + "/chat/completions"
        response = request_public_http(
            endpoint, method="POST", body=canonical_json(redacted).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization":
                    "Bearer " + os.environ["FRIDAY_REMOTE_API_KEY"],
            }, timeout_seconds=60, max_response_bytes=2_000_000,
            allowed_content_types=frozenset({
                "application/json", "application/problem+json"}),
            max_redirects=0, allow_redirects=False)
        if not 200 <= response.status <= 299:
            raise RuntimeError(
                f"remote model returned HTTP status {response.status}")
        raw = response.body
        result = json.loads(raw)
        try:
            text = str(result["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("remote model returned an invalid response") from exc
        observation = {"disclosure_id": disclosure["disclosure_id"],
                       "model": self.remote_model, "text": text,
                       "received_at": utc_now()}
        observation_id = self.graph.record_node(
            "observation", observation, actor="model_router", task_id=task_id,
            event_type="model.remote_completed",
            links=[("derived_from", disclosure["disclosure_id"])])
        return {"model": self.remote_model, "text": text,
                "disclosure_id": disclosure["disclosure_id"],
                "observation_id": observation_id}
