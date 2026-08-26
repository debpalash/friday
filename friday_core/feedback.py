"""User feedback and transcript correction records with durable provenance."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from .controller_auth import ControllerAuthService, ControllerPrincipal
from .graph import GraphStore, canonical_json, new_id, sha256_text, utc_now
from .tasks import redact_tool_arguments


FEEDBACK_KINDS = {"correct", "wrong", "edit", "undo", "problem"}
_FEEDBACK_STOPWORDS = {
    "about", "after", "again", "could", "from", "have", "into", "just",
    "that", "their", "then", "this", "what", "when", "where", "which",
    "with", "would", "your",
}


def _terms(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.casefold())
            if len(token) >= 3 and token not in _FEEDBACK_STOPWORDS}


class FeedbackService:
    def __init__(self, graph: GraphStore):
        self.graph = graph

    def record(self, kind: str, *, task_id: str | None = None,
               turn_id: str | None = None, comment: str | None = None,
               supersedes_id: str | None = None, actor: str = "user") -> dict[str, Any]:
        if kind not in FEEDBACK_KINDS:
            raise ValueError(f"feedback kind must be one of {sorted(FEEDBACK_KINDS)}")
        if not task_id and not turn_id:
            raise ValueError("feedback requires a task_id or turn_id")
        if task_id and self.graph.get_node(task_id) is None:
            raise ValueError("task does not exist")
        if kind in {"edit", "problem"} and not str(comment or "").strip():
            raise ValueError(f"{kind} feedback requires a comment")
        if kind == "undo" and not supersedes_id:
            raise ValueError("undo feedback requires a prior feedback record")
        now = utc_now()
        body = {"kind": kind, "task_id": task_id, "turn_id": turn_id,
                "comment": str(comment or "").strip(),
                "supersedes_id": supersedes_id}
        with self.graph.transaction() as conn:
            if supersedes_id:
                prior = conn.execute(
                    "SELECT * FROM feedback_state WHERE feedback_id=?",
                    (supersedes_id,)).fetchone()
                if prior is None:
                    raise ValueError("superseded feedback does not exist")
            event_id, seq = self.graph.append_event(
                conn, "feedback.recorded", body, actor=actor, task_id=task_id,
                turn_id=turn_id)
            feedback_id = self.graph.append_node(
                conn, "feedback", body, event_id=event_id,
                node_id=new_id("feedback"))
            if task_id:
                self.graph.append_edge(conn, feedback_id, "evaluates", task_id,
                                       event_id=event_id)
            if supersedes_id:
                self.graph.append_edge(conn, feedback_id, "supersedes", supersedes_id,
                                       event_id=event_id)
                conn.execute(
                    "UPDATE feedback_state SET lifecycle='superseded' WHERE feedback_id=?",
                    (supersedes_id,))
            conn.execute(
                """INSERT INTO feedback_state
                   (feedback_id,task_id,turn_id,kind,comment,lifecycle,supersedes_id,
                    created_at,last_event_seq) VALUES (?,?,?,?,?,'active',?,?,?)""",
                (feedback_id, task_id, turn_id, kind, body["comment"], supersedes_id,
                 now, seq))
            if task_id and kind in {"wrong", "problem"}:
                disputed = {"status": "failed",
                            "summary": "User disputed the reported outcome",
                            "evidence": [feedback_id],
                            "missing": ["corrected outcome"]}
                conn.execute(
                    """UPDATE task_state SET verification_status='failed',
                       verification_json=?,updated_at=?,last_event_seq=?
                       WHERE task_id=?""",
                    (canonical_json(disputed), now, seq, task_id))
        return {"feedback_id": feedback_id, **body, "lifecycle": "active",
                "created_at": now}

    def correct_transcript(self, utterance_id: str, corrected_text: str, *,
                           audio_artifact: str | None = None,
                           actor: str = "user") -> dict[str, Any]:
        source = self.graph.get_node(utterance_id)
        if source is None or source["kind"] != "utterance":
            raise ValueError("correction requires an existing utterance")
        corrected = corrected_text.strip()
        if not corrected:
            raise ValueError("corrected transcript cannot be empty")
        original = str(source["body"].get("text") or "")
        if corrected == original.strip():
            raise ValueError("corrected transcript is unchanged")
        now = utc_now()
        body = {"utterance_id": utterance_id, "original_text": original,
                "corrected_text": corrected, "audio_artifact": audio_artifact}
        with self.graph.transaction() as conn:
            event_id, seq = self.graph.append_event(
                conn, "transcript.corrected", body, actor=actor)
            correction_id = self.graph.append_node(
                conn, "correction", body, event_id=event_id,
                node_id=new_id("correction"))
            self.graph.append_edge(conn, correction_id, "corrects", utterance_id,
                                   event_id=event_id)
            conn.execute(
                """INSERT INTO transcript_corrections
                   (correction_id,utterance_id,original_text,corrected_text,
                    audio_artifact,created_at,last_event_seq)
                   VALUES (?,?,?,?,?,?,?)""",
                (correction_id, utterance_id, original, corrected,
                 audio_artifact, now, seq))
        return {"correction_id": correction_id, **body, "created_at": now}

    def list(self, *, task_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM feedback_state"
        params: list[Any] = []
        if task_id:
            sql += " WHERE task_id=?"
            params.append(task_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.graph._connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def relevant_context(self, query: str, *, limit: int = 3
                         ) -> list[dict[str, Any]]:
        """Return actionable user comments from lexically similar prior tasks."""
        query_terms = _terms(query)
        if not query_terms:
            return []
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT f.feedback_id,f.kind,f.comment,f.created_at,t.objective
                   FROM feedback_state f JOIN task_state t ON t.task_id=f.task_id
                   WHERE f.lifecycle='active' AND f.kind IN ('wrong','problem','edit')
                     AND TRIM(f.comment)<>''
                   ORDER BY f.created_at DESC LIMIT 100""").fetchall()
        ranked = []
        for row in rows:
            objective_terms = _terms(str(row["objective"] or ""))
            comment_terms = _terms(str(row["comment"] or ""))
            score = len(query_terms & (objective_terms | comment_terms))
            if score:
                ranked.append((score, dict(row)))
        ranked.sort(key=lambda item: (item[0], item[1]["created_at"]), reverse=True)
        return [item for _score, item in ranked[:max(1, limit)]]

    def apply_transcript_corrections(self, text: str) -> tuple[str, list[str]]:
        """Apply only unambiguous exact corrections to future ASR utterances."""
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT correction_id,original_text,corrected_text
                   FROM transcript_corrections ORDER BY created_at""").fetchall()
        grouped: dict[str, list[Any]] = {}
        for row in rows:
            grouped.setdefault(row["original_text"].strip().casefold(), []).append(row)
        candidates = grouped.get(text.strip().casefold(), [])
        if not candidates:
            return text, []
        targets = {row["corrected_text"].strip().casefold() for row in candidates}
        if len(targets) != 1:
            return text, []
        latest = candidates[-1]
        return latest["corrected_text"], [row["correction_id"] for row in candidates]


class ApprovalService:
    def __init__(
        self, graph: GraphStore,
        controller_auth: ControllerAuthService | None = None, *,
        require_controller_decisions: bool = False,
        clock: Callable[[], datetime] | None = None,
        request_ttl_seconds: int = 300,
        authorization_ttl_seconds: int = 120,
    ):
        if not 30 <= request_ttl_seconds <= 900:
            raise ValueError("approval request TTL is invalid")
        if not 15 <= authorization_ttl_seconds <= 300:
            raise ValueError("approval authorization TTL is invalid")
        self.graph = graph
        self.controller_auth = controller_auth
        self.require_controller_decisions = bool(
            require_controller_decisions)
        self._clock = clock or (
            controller_auth.current_time if controller_auth is not None
            else lambda: datetime.now(UTC))
        self.request_ttl_seconds = request_ttl_seconds
        self.authorization_ttl_seconds = authorization_ttl_seconds

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise RuntimeError("approval clock must be timezone-aware")
        return value.astimezone(UTC)

    @staticmethod
    def _iso(value: datetime) -> str:
        return value.astimezone(UTC).isoformat(
            timespec="microseconds").replace("+00:00", "Z")

    def retire_unbound_legacy_requests(self) -> dict[str, Any]:
        """Cancel pre-controller approvals that can never gain authority.

        Older Friday versions created approvals without a durable step or a
        controller request. Strict controller mode must neither present those
        rows as actionable nor silently reinterpret them as signed authority.
        Every retirement is journaled and the operation is idempotent.
        """
        if not self.require_controller_decisions:
            return {"retired": 0, "task_ids": []}
        now = self._iso(self._now())
        task_ids: set[str] = set()
        retired = 0
        with self.graph.transaction() as conn:
            rows = conn.execute(
                """SELECT a.approval_id,a.task_id
                     FROM approval_state a
                    WHERE a.status='pending' AND a.step_id IS NULL
                      AND NOT EXISTS (
                          SELECT 1 FROM controller_approval_requests r
                           WHERE r.approval_id=a.approval_id)
                    ORDER BY a.created_at,a.approval_id"""
            ).fetchall()
            for row in rows:
                approval_id = str(row["approval_id"])
                task_id = str(row["task_id"])
                body = {
                    "approval_id": approval_id,
                    "task_id": task_id,
                    "status": "cancelled",
                    "reason_code": "legacy_unbound_controller_authority",
                }
                _, seq = self.graph.append_event(
                    conn, "approval.cancelled", body,
                    actor="controller_auth_migration", task_id=task_id)
                changed = conn.execute(
                    """UPDATE approval_state
                          SET status='cancelled',decided_at=?,last_event_seq=?
                        WHERE approval_id=? AND status='pending'
                          AND step_id IS NULL""",
                    (now, seq, approval_id),
                ).rowcount
                if changed != 1:
                    raise RuntimeError(
                        "legacy approval retirement lost its fence")
                task_ids.add(task_id)
                retired += 1
        return {"retired": retired, "task_ids": sorted(task_ids)}

    def _expire_controller_request(
        self, approval_id: str, *, now_value: datetime,
    ) -> bool:
        """Durably close one elapsed request without cancelling its step.

        The durable step intentionally remains ``pending`` and keeps waiting
        for approval. A caller can therefore create a fresh, newly signed
        request instead of either resurrecting the expired proof or leaving
        the whole batch permanently wedged.
        """
        now = self._iso(now_value)
        with self.graph.transaction() as conn:
            row = conn.execute(
                """SELECT a.task_id,a.status,r.expires_at
                     FROM approval_state a
                     JOIN controller_approval_requests r
                       ON r.approval_id=a.approval_id
                    WHERE a.approval_id=?""",
                (approval_id,),
            ).fetchone()
            if (row is None or row["status"] != "pending"
                    or row["expires_at"] > now):
                return False
            body = {
                "approval_id": approval_id,
                "task_id": str(row["task_id"]),
                "status": "expired",
            }
            _, seq = self.graph.append_event(
                conn, "approval.expired", body, actor="controller_auth",
                task_id=str(row["task_id"]))
            changed = conn.execute(
                """UPDATE approval_state
                      SET status='expired',decided_at=?,last_event_seq=?
                    WHERE approval_id=? AND status='pending'""",
                (now, seq, approval_id),
            ).rowcount
            if changed != 1:
                raise PermissionError("approval expiration lost its fence")
            return True

    def request(self, task_id: str, tool_name: str, args: dict[str, Any],
                reason: str, *, step_id: str | None = None,
                actor: str = "policy",
                controller_principal: ControllerPrincipal | None = None,
                ) -> dict[str, Any]:
        if self.graph.get_node(task_id) is None:
            raise ValueError("approval requires an existing task")
        args_sha256 = sha256_text(canonical_json(args))
        recorded_args = redact_tool_arguments(tool_name, args)
        recorded_args["_args_sha256"] = args_sha256
        body = {"task_id": task_id, "tool_name": tool_name,
                "args": recorded_args, "reason": reason, "status": "pending",
                "step_id": step_id}
        now_value = self._now()
        now = self._iso(now_value)
        with self.graph.transaction() as conn:
            task_authority = conn.execute(
                "SELECT * FROM controller_task_authorities WHERE task_id=?",
                (task_id,),
            ).fetchone()
            controller_row = None
            if (task_authority is not None
                    or self.require_controller_decisions):
                if (controller_principal is None
                        or self.controller_auth is None):
                    raise PermissionError(
                        "approval request requires controller authority")
                controller_row = (
                    self.controller_auth.require_principal_in_transaction(
                        conn, controller_principal, now_value=now_value))
                if (task_authority is None
                        or task_authority["controller_id"] !=
                            controller_principal.controller_id
                        or task_authority["controller_key_sha256"] !=
                            controller_principal.public_key_sha256
                        or int(task_authority["controller_epoch"]) !=
                            controller_principal.controller_epoch):
                    raise PermissionError(
                        "approval request controller does not own the task")
                if step_id is None:
                    raise ValueError(
                        "controller approval requires a durable step")
            if step_id:
                step = conn.execute(
                    """SELECT task_id,tool_name,args_sha256,approval_status
                       FROM task_steps WHERE step_id=?""", (step_id,)).fetchone()
                if (step is None or step["task_id"] != task_id
                        or step["tool_name"] != tool_name
                        or step["args_sha256"] != args_sha256):
                    raise ValueError(
                        "approval does not match the exact durable step")
                if step["approval_status"] != "pending":
                    raise ValueError("durable step is not awaiting approval")
            event_id, seq = self.graph.append_event(
                conn, "approval.requested", body, actor=actor, task_id=task_id)
            approval_id = self.graph.append_node(
                conn, "approval", body, event_id=event_id,
                node_id=new_id("approval"))
            self.graph.append_edge(conn, task_id, "requires", approval_id,
                                   event_id=event_id)
            conn.execute(
                """INSERT INTO approval_state
                   (approval_id,task_id,step_id,tool_name,args_json,reason,status,
                    created_at,last_event_seq)
                   VALUES (?,?,?,?,?,?,'pending',?,?)""",
                (approval_id, task_id, step_id, tool_name,
                 canonical_json(recorded_args), reason, now, seq))
            request_expires_at = None
            if controller_row is not None:
                assert controller_principal is not None
                absolute = datetime.fromisoformat(
                    controller_principal.absolute_expires_at.replace(
                        "Z", "+00:00"))
                request_expires = min(
                    now_value + timedelta(seconds=self.request_ttl_seconds),
                    absolute,
                )
                if request_expires <= now_value:
                    raise PermissionError("controller session is expired")
                request_expires_at = self._iso(request_expires)
                conn.execute(
                    """INSERT INTO controller_approval_requests
                       (approval_id,task_id,step_id,tool_name,args_sha256,
                        controller_id,controller_key_sha256,controller_epoch,
                        request_session_id,session_absolute_expires_at,
                        transport_binding_sha256,origin_sha256,requested_at,
                        expires_at,requested_event_seq)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (approval_id, task_id, step_id, tool_name, args_sha256,
                     controller_principal.controller_id,
                     controller_principal.public_key_sha256,
                     controller_principal.controller_epoch,
                     controller_principal.session_id,
                     controller_principal.absolute_expires_at,
                     controller_principal.transport_binding_sha256,
                     controller_principal.origin_sha256,
                     now, request_expires_at, seq),
                )
            if step_id:
                conn.execute(
                    """UPDATE task_steps SET approval_id=?,updated_at=?,
                       last_event_seq=? WHERE step_id=?""",
                    (approval_id, now, seq, step_id))
        response = {"approval_id": approval_id, **body, "created_at": now}
        if request_expires_at is not None:
            response["expires_at"] = request_expires_at
        return response

    @staticmethod
    def _decision_payload(
        request: Any, approved: bool, principal: ControllerPrincipal,
    ) -> str:
        return canonical_json({
            "approval_id": str(request["approval_id"]),
            "args_sha256": str(request["args_sha256"]),
            "controller_epoch": principal.controller_epoch,
            "controller_id": principal.controller_id,
            "controller_key_sha256": principal.public_key_sha256,
            "decision": "approved" if approved else "denied",
            "decision_session_id": principal.session_id,
            "kind": "friday.controller_approval_decision",
            "origin_sha256": principal.origin_sha256,
            "request_expires_at": str(request["expires_at"]),
            "schema_version": 1,
            "session_absolute_expires_at": principal.absolute_expires_at,
            "step_id": str(request["step_id"]),
            "task_id": str(request["task_id"]),
            "tool_name": str(request["tool_name"]),
            "transport_binding_sha256":
                principal.transport_binding_sha256,
        })

    def prepare_decision(
        self, approval_id: str, approved: bool,
        principal: ControllerPrincipal,
    ) -> dict[str, Any]:
        if self.controller_auth is None:
            raise RuntimeError("controller authentication service is unavailable")
        now_value = self._now()
        now = self._iso(now_value)
        # Observe request expiry before principal validation. This lets an
        # elapsed request become durable even when the associated controller
        # session expires at the same boundary.
        with self.graph._connect() as conn:
            observed = conn.execute(
                """SELECT r.expires_at,a.status AS approval_status
                     FROM controller_approval_requests r
                     JOIN approval_state a ON a.approval_id=r.approval_id
                    WHERE r.approval_id=?""",
                (approval_id,),
            ).fetchone()
        if (observed is not None and observed["approval_status"] == "pending"
                and observed["expires_at"] <= now):
            self._expire_controller_request(
                approval_id, now_value=now_value)
            raise PermissionError("controller approval request is stale")
        with self.graph._connect() as conn:
            self.controller_auth.require_principal_in_transaction(
                conn, principal, now_value=now_value)
            request = conn.execute(
                """SELECT r.*,a.status AS approval_status
                     FROM controller_approval_requests r
                     JOIN approval_state a ON a.approval_id=r.approval_id
                    WHERE r.approval_id=?""",
                (approval_id,),
            ).fetchone()
        if (request is None or request["approval_status"] != "pending"
                or request["expires_at"] <= now
                or request["controller_id"] != principal.controller_id
                or request["controller_key_sha256"] !=
                    principal.public_key_sha256
                or int(request["controller_epoch"]) !=
                    principal.controller_epoch):
            raise PermissionError("controller approval request is stale")
        payload = self._decision_payload(request, approved, principal)
        return {
            "approval_id": approval_id,
            "decision": "approved" if approved else "denied",
            "proof_payload": payload,
            "request_expires_at": str(request["expires_at"]),
        }

    def decide(
        self, approval_id: str, approved: bool, *, actor: str = "user",
        controller_principal: ControllerPrincipal | None = None,
        proof_payload: str | None = None,
        signature_b64url: str | None = None,
    ) -> dict[str, Any]:
        status = "approved" if approved else "denied"
        with self.graph._connect() as conn:
            has_controller_request = conn.execute(
                "SELECT 1 FROM controller_approval_requests WHERE approval_id=?",
                (approval_id,),
            ).fetchone() is not None
        if has_controller_request or self.require_controller_decisions:
            if (controller_principal is None or self.controller_auth is None
                    or not isinstance(proof_payload, str)
                    or not isinstance(signature_b64url, str)):
                raise PermissionError(
                    "approval decision requires a signed controller proof")
            prepared = self.prepare_decision(
                approval_id, approved, controller_principal)
            if prepared["proof_payload"] != proof_payload:
                raise PermissionError("controller approval proof is not exact")
            if not self.controller_auth.verify_principal_signature(
                    controller_principal, proof_payload, signature_b64url):
                raise PermissionError("controller approval signature is invalid")
            actor = controller_principal.controller_id
        now_value = self._now()
        now = self._iso(now_value)
        with self.graph.transaction() as conn:
            row = conn.execute("SELECT * FROM approval_state WHERE approval_id=?",
                               (approval_id,)).fetchone()
            if row is None:
                raise ValueError("approval does not exist")
            if row["status"] != "pending":
                raise ValueError("approval is already decided")
            request = conn.execute(
                "SELECT * FROM controller_approval_requests WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
            controller_row = None
            authorization_expires_at = None
            if request is not None:
                if (controller_principal is None or self.controller_auth is None
                        or proof_payload is None
                        or signature_b64url is None):
                    raise PermissionError(
                        "approval decision requires controller authority")
                controller_row = (
                    self.controller_auth.require_principal_in_transaction(
                        conn, controller_principal, now_value=now_value))
                expected = self._decision_payload(
                    request, approved, controller_principal)
                if (expected != proof_payload or request["expires_at"] <= now
                        or request["controller_id"] !=
                            controller_principal.controller_id
                        or request["controller_key_sha256"] !=
                            controller_principal.public_key_sha256
                        or int(request["controller_epoch"]) !=
                            controller_principal.controller_epoch):
                    raise PermissionError(
                        "controller approval request is stale")
                if approved:
                    absolute = datetime.fromisoformat(
                        controller_principal.absolute_expires_at.replace(
                            "Z", "+00:00"))
                    idle = datetime.fromisoformat(
                        str(controller_row["idle_expires_at"]).replace(
                            "Z", "+00:00"))
                    authorization_expires_at = self._iso(min(
                        now_value + timedelta(
                            seconds=self.authorization_ttl_seconds),
                        absolute, idle,
                    ))
                    if authorization_expires_at <= now:
                        raise PermissionError("controller session is expired")
            body = {"approval_id": approval_id, "status": status,
                    "task_id": row["task_id"]}
            if controller_principal is not None and request is not None:
                body |= {
                    "controller_id": controller_principal.controller_id,
                    "controller_session_id": controller_principal.session_id,
                    "proof_payload_sha256": sha256_text(proof_payload or ""),
                }
            event_id, seq = self.graph.append_event(
                conn, "approval.decided", body, actor=actor,
                task_id=row["task_id"])
            conn.execute(
                """UPDATE approval_state SET status=?,decided_at=?,last_event_seq=?
                   WHERE approval_id=?""", (status, now, seq, approval_id))
            decision_id = None
            if request is not None:
                assert controller_principal is not None
                assert proof_payload is not None
                assert signature_b64url is not None
                decision_body = {
                    "approval_id": approval_id,
                    "decision": status,
                    "controller_id": controller_principal.controller_id,
                    "proof_payload_sha256": sha256_text(proof_payload),
                    "signature_sha256": sha256_text(signature_b64url),
                }
                decision_id = self.graph.append_node(
                    conn, "controller_approval_decision", decision_body,
                    event_id=event_id, node_id=new_id("approval_decision"))
                conn.execute(
                    """INSERT INTO controller_approval_decisions
                       (decision_id,approval_id,task_id,step_id,tool_name,
                        args_sha256,controller_id,controller_key_sha256,
                        controller_epoch,decision_session_id,
                        session_absolute_expires_at,
                        transport_binding_sha256,origin_sha256,decision,
                        proof_payload_json,proof_payload_sha256,
                        signature_b64url,signature_sha256,decided_at,
                        authorization_expires_at,decided_event_seq)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (decision_id, approval_id, request["task_id"],
                     request["step_id"], request["tool_name"],
                     request["args_sha256"],
                     controller_principal.controller_id,
                     controller_principal.public_key_sha256,
                     controller_principal.controller_epoch,
                     controller_principal.session_id,
                     controller_principal.absolute_expires_at,
                     controller_principal.transport_binding_sha256,
                     controller_principal.origin_sha256,
                     status, proof_payload, sha256_text(proof_payload),
                     signature_b64url, sha256_text(signature_b64url),
                     now, authorization_expires_at, seq),
                )
            batch_id = None
            if row["step_id"]:
                step = conn.execute(
                    """SELECT * FROM task_steps WHERE step_id=? AND approval_id=?""",
                    (row["step_id"], approval_id)).fetchone()
                if step is None or step["approval_status"] != "pending":
                    raise PermissionError(
                        "approval is not bound to an awaiting durable step")
                batch_id = str(step["batch_id"])
                conn.execute(
                    """UPDATE task_steps SET approval_status=?,status=?,updated_at=?,
                       last_event_seq=? WHERE step_id=?""",
                    (status, "pending" if approved else "cancelled", now,
                     seq, row["step_id"]))
                if approved:
                    pending = int(conn.execute(
                        """SELECT COUNT(*) FROM task_steps WHERE batch_id=?
                           AND approval_status='pending'""",
                        (batch_id,)).fetchone()[0])
                    batch_status = "queued" if pending == 0 else "waiting_approval"
                else:
                    batch_status = "cancelled"
                    # A denied action makes every dependent action impossible.
                    # Invalidate both sides of every still-pending approval in
                    # that suffix so a stale UI decision cannot resurrect the
                    # cancelled batch after a restart.
                    cancelled_at = now
                    conn.execute(
                        """UPDATE task_steps SET status='cancelled',
                           approval_status=CASE
                               WHEN approval_status='pending' THEN 'cancelled'
                               ELSE approval_status END,
                           updated_at=?,last_event_seq=?
                           WHERE batch_id=? AND ordinal>=? AND status IN
                           ('pending','waiting_approval')""",
                        (cancelled_at, seq, batch_id, int(step["ordinal"])))
                    conn.execute(
                        """UPDATE approval_state SET status='cancelled',
                           decided_at=?,last_event_seq=?
                           WHERE status='pending' AND step_id IN
                               (SELECT step_id FROM task_steps
                                WHERE batch_id=? AND ordinal>=?)""",
                        (cancelled_at, seq, batch_id, int(step["ordinal"])))
                conn.execute(
                    """UPDATE task_step_batches SET status=?,updated_at=?,
                       last_event_seq=? WHERE batch_id=?""",
                    (batch_status, now, seq, batch_id))
        response = body | ({"step_id": row["step_id"], "batch_id": batch_id}
                           if row["step_id"] else {})
        if decision_id is not None:
            response["decision_id"] = decision_id
            response["authorization_expires_at"] = (
                authorization_expires_at)
        return response

    def is_approved(self, task_id: str, tool_name: str,
                    args: dict[str, Any]) -> bool:
        expected = sha256_text(canonical_json(args))
        with self.graph._connect() as conn:
            # A paired controller authorizes one exact durable step. Reusing
            # the legacy task/tool/argument lookup here would let a later step
            # inherit an earlier signature without creating or consuming a
            # controller approval decision.
            if conn.execute(
                    "SELECT 1 FROM controller_task_authorities WHERE task_id=?",
                    (task_id,)).fetchone() is not None:
                return False
            rows = conn.execute(
                """SELECT args_json FROM approval_state
                   WHERE task_id=? AND tool_name=? AND status='approved'""",
                (task_id, tool_name)).fetchall()
        return any(__import__("json").loads(row["args_json"]).get("_args_sha256")
                   == expected for row in rows)

    def list(self, *, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM approval_state"
        params: list[Any] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY created_at DESC"
        with self.graph._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) | {"args": __import__("json").loads(row["args_json"])}
                for row in rows]
