"""Durable task, action receipt, recovery, and progress projections."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .admission import ResourceAdmissionController, ResourceSnapshot
from .cognition import ResourceClaim, TaskContract, TaskPlan, VerificationResult
from .controller_auth import ControllerAuthService, ControllerPrincipal
from .graph import GraphStore, canonical_json, new_id, sha256_text, utc_now
from .step_payloads import StepPayloadCipher


TERMINAL = {"completed", "failed", "cancelled"}
ALLOWED_TRANSITIONS = {
    "created": {"interpreting", "cancelled"},
    "interpreting": {"planned", "waiting_input", "failed", "cancelled"},
    "planned": {"running", "waiting_input", "cancelled"},
    "running": {"verifying", "replanning", "waiting_input", "failed", "cancelled"},
    "verifying": {"completed", "replanning", "waiting_input", "failed", "cancelled"},
    "replanning": {"running", "waiting_input", "failed", "cancelled"},
    "waiting_input": {"running", "failed", "cancelled"},
    "recovering": {"running", "waiting_input", "failed", "cancelled"},
}

_PRIVATE_ARGUMENT_FIELDS = {
    "browser_open": {"url"},
    "browser_snapshot": {"page_url"},
    "browser_click": {"page_url"},
    "browser_type": {"page_url", "text"},
    "clipboard_write": {"text"},
    "desktop_notify": {"title", "message"},
    "remote_reason": {"prompt"},
    "write_file": {"content"},
    "machine_grant_path": {"path"},
    "machine_revoke_grant": {"grant_id"},
    "machine_inspect_path": {"path"},
    "machine_list_path": {"path"},
    "machine_read_text": {"path"},
    "machine_read_document": {"path"},
    "machine_ocr_image": {"path"},
    "machine_understand_image": {"path", "question"},
    "machine_write_text": {"path", "content"},
    "machine_rollback_write": {"operation_id"},
    "machine_launch_process": {"parameter_values"},
}
_SECRET_FIELD_FRAGMENTS = (
    "password", "secret", "token", "api_key", "authorization",
)
_SAFE_RESULT_STATUSES = {
    "ok", "succeeded", "success", "completed", "failed", "error",
    "cancelled", "pending", "active", "revoked", "expired",
    "prepared", "starting", "running", "stop_requested", "stopping",
    "terminated", "exited", "reconciling", "reconcile_required",
}


def tool_has_private_payload(tool_name: str) -> bool:
    """Return whether raw arguments/results must remain ephemeral."""
    return (tool_name.startswith("browser_")
            or tool_name.startswith("machine_")
            or tool_name in {
                "clipboard_read", "clipboard_write", "read_file",
                "remote_reason",
            })


def tool_arguments_are_private(tool_name: str) -> bool:
    """Return whether a tool call's raw arguments must stay out of sessions."""
    return bool(_PRIVATE_ARGUMENT_FIELDS.get(tool_name))


def _value_sha256(value: Any) -> str:
    if isinstance(value, str):
        return sha256_text(value)
    try:
        return sha256_text(canonical_json(value))
    except (TypeError, ValueError):
        return sha256_text(str(value))


def _redact_argument_value(value: Any,
                           private_fields: set[str]) -> Any:
    if isinstance(value, list):
        return [_redact_argument_value(item, private_fields) for item in value]
    if not isinstance(value, dict):
        return value
    sanitized: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for raw_field, item in value.items():
        field = str(raw_field)
        normalized = field.casefold()
        private = (normalized in private_fields
                   or any(fragment in normalized
                          for fragment in _SECRET_FIELD_FRAGMENTS))
        if private:
            sanitized[field] = "[REDACTED]"
            metadata[f"{field}_sha256"] = _value_sha256(item)
            if isinstance(item, str):
                metadata[f"{field}_characters"] = len(item)
        else:
            sanitized[field] = _redact_argument_value(item, private_fields)
    # Generated metadata wins over model-supplied fields with the same names.
    sanitized.update(metadata)
    return sanitized


def redact_tool_arguments(tool_name: str,
                          args: dict[str, Any]) -> dict[str, Any]:
    """Create a persistence-safe argument record without mutating live args."""
    return _redact_argument_value(
        args, _PRIVATE_ARGUMENT_FIELDS.get(tool_name, set()))


def _decoded_result(result: Any) -> Any:
    if not isinstance(result, str):
        return result
    try:
        return json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return result


def redact_tool_result(tool_name: str, result: Any) -> Any:
    """Replace private results with hashes and verification-safe metadata.

    The return value is for durable storage only. Callers retain and continue to
    use the original result in memory.
    """
    if not tool_has_private_payload(tool_name):
        return result
    value = _decoded_result(result)
    result_text = result if isinstance(result, str) else canonical_json(result)
    summary: dict[str, Any] = {
        "_redacted": True,
        "result_sha256": sha256_text(result_text),
        "result_characters": len(result_text),
    }
    if isinstance(value, dict):
        status = str(value.get("status") or "").casefold()
        if status in _SAFE_RESULT_STATUSES:
            summary["status"] = status
        for field in ("characters", "bytes", "count"):
            item = value.get(field)
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                summary[field] = item
        for field in ("text", "content", "response"):
            item = value.get(field)
            if isinstance(item, str):
                summary[f"{field}_characters"] = len(item)
                summary[f"{field}_sha256"] = sha256_text(item)
        if value.get("url") is not None:
            summary["url"] = "[REDACTED]"
            summary["url_sha256"] = _value_sha256(value["url"])
        for field in ("grant_id", "rollback_operation_id", "operation_id",
                      "spec_id", "instance_id"):
            item = value.get(field)
            if (isinstance(item, str)
                    and re.fullmatch(r"[A-Za-z0-9_.:-]{8,200}", item)):
                summary[field] = item
    elif isinstance(value, list):
        summary["count"] = len(value)
    # Most executors return JSON strings. Preserve that shape so idempotent
    # replay remains compatible with the server's string-oriented tool flow.
    return canonical_json(summary) if isinstance(result, str) else summary


def _redact_effects(tool_name: str,
                    effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tool_has_private_payload(tool_name):
        return effects
    sanitized: list[dict[str, Any]] = []
    for effect in effects:
        safe: dict[str, Any] = {
            "_redacted": True,
            "effect_sha256": _value_sha256(effect),
        }
        for field in ("kind", "action", "tool", "status"):
            value = effect.get(field)
            if (isinstance(value, str) and 0 < len(value) <= 80
                    and all(char.isalnum() or char in "._:-" for char in value)):
                safe[field] = value
        for field in ("verified", "count"):
            value = effect.get(field)
            if isinstance(value, (bool, int)):
                safe[field] = value
        if effect.get("url") is not None:
            safe["url"] = "[REDACTED]"
            safe["url_sha256"] = _value_sha256(effect["url"])
        sanitized.append(safe)
    return sanitized


def _redact_verification(tool_name: str,
                         verification: dict[str, Any] | None
                         ) -> dict[str, Any] | None:
    if verification is None or not tool_has_private_payload(tool_name):
        return verification
    status = str(verification.get("status") or "uncertain").casefold()
    if status not in {"passed", "failed", "uncertain",
                      "user_confirmation_required"}:
        status = "uncertain"
    evidence = list(verification.get("evidence") or [])
    missing = list(verification.get("missing") or [])
    return {
        "status": status,
        "summary": f"{tool_name} verification {status}",
        "evidence_count": len(evidence),
        "evidence_sha256": [_value_sha256(item) for item in evidence],
        "missing_count": len(missing),
        "effects": _redact_effects(
            tool_name, list(verification.get("effects") or [])),
        "_redacted": True,
    }


def tool_result_log_summary(tool_name: str, result: Any) -> str:
    """Return a bounded log line that never contains a private tool result."""
    if not tool_has_private_payload(tool_name):
        return str(result)[:120]
    result_text = result if isinstance(result, str) else canonical_json(result)
    return ("[REDACTED private result; "
            f"sha256={sha256_text(result_text)[:16]}; "
            f"characters={len(result_text)}]")


@dataclass(frozen=True)
class ActionHandle:
    action_id: str
    idempotency_key: str
    replayed: bool = False
    prior_result: Any = None


@dataclass(frozen=True, repr=False)
class ClaimedStep:
    """One fenced durable dispatch.

    Exact arguments deliberately do not appear in ``repr``.  They may be used
    only by the executor and must never be serialized into logs or progress.
    """

    step_id: str
    batch_id: str
    task_id: str
    round_index: int
    ordinal: int
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    idempotency_key: str
    idempotency_class: str
    recovery_policy: str
    risk: str
    approval_status: str
    action_id: str
    attempt_id: str
    attempt_number: int
    lease_id: str
    worker_id: str
    verifier: str
    executor_binding: dict[str, Any]
    resource_claims: dict[str, Any]
    context: dict[str, Any]
    resource_lease_id: str | None = None

    def __repr__(self) -> str:
        return (f"ClaimedStep(step_id={self.step_id!r}, "
                f"tool_name={self.tool_name!r}, "
                f"attempt_number={self.attempt_number})")


@dataclass(frozen=True, repr=False)
class ReconciliationCandidate:
    """Exact private state for one quarantined external action.

    The model and API receive only :meth:`TaskService.list_reconciliations`.
    Exact arguments and executor bindings stay in this non-serializable,
    deliberately redacted-from-repr object for a server-owned evidence probe.
    """

    step_id: str
    batch_id: str
    task_id: str
    ordinal: int
    tool_name: str
    args: dict[str, Any]
    args_sha256: str
    idempotency_key: str
    action_id: str
    attempt_id: str
    executor_binding: dict[str, Any]
    executor_binding_sha256: str

    def __repr__(self) -> str:
        return (f"ReconciliationCandidate(step_id={self.step_id!r}, "
                f"tool_name={self.tool_name!r})")


class TaskService:
    def __init__(self, graph: GraphStore,
                 step_cipher: StepPayloadCipher | None = None,
                 admission: ResourceAdmissionController | None = None,
                 controller_auth: ControllerAuthService | None = None,
                 require_controller_authority: bool = False):
        self.graph = graph
        self.admission = admission
        self.controller_auth = controller_auth
        self.require_controller_authority = bool(
            require_controller_authority)
        self.admission_sensor_error: str | None = None
        self.admission_sensor_checked_at: str | None = None
        self._step_cipher = step_cipher or StepPayloadCipher(
            graph.path.with_name("step-payload.key"))

    @staticmethod
    def _progress_payload(task_id: str, phase: str, state: str, label: str,
                          detail: str | None = None) -> dict[str, Any]:
        payload = {"type": "progress", "task_id": task_id, "phase": phase,
                   "state": state, "label": label, "occurred_at": utc_now()}
        if detail:
            payload["detail"] = detail
        return payload

    def _append_progress(self, conn, event_id: str, task_id: str,
                         payload: dict[str, Any]) -> dict[str, Any]:
        cur = conn.execute(
            """INSERT INTO progress_outbox(event_id, task_id, payload_json, occurred_at)
               VALUES (?, ?, ?, ?)""",
            (event_id, task_id, canonical_json(payload), payload["occurred_at"]),
        )
        return payload | {"seq": int(cur.lastrowid)}

    def create(self, objective: str, completion_contract: dict[str, Any], *,
               session_id: str | None = None, turn_id: str | None = None,
               actor: str = "friday",
               controller_principal: ControllerPrincipal | None = None,
               ) -> tuple[str, dict[str, Any]]:
        if (controller_principal is not None
                and self.controller_auth is None):
            raise RuntimeError("controller authentication service is unavailable")
        if (self.require_controller_authority and session_id is not None
                and controller_principal is None):
            raise PermissionError(
                "interactive task requires controller authority")
        now = utc_now()
        contract_version = int(completion_contract.get("version", 0))
        if contract_version:
            contract = TaskContract.model_validate(completion_contract)
            completion_contract = contract.model_dump(mode="json")
            intent_type = contract.intent_type.value
            risk = contract.risk.value
        else:
            intent_type = "action"
            risk = "low"
        body = {"objective": objective,
                "completion_contract": completion_contract, "status": "created"}
        with self.graph.transaction() as conn:
            event_id, seq = self.graph.append_event(
                conn, "task.created", body, actor=actor, session_id=session_id,
                turn_id=turn_id)
            task_id = self.graph.append_node(conn, "task", body, event_id=event_id,
                                             node_id=new_id("task"))
            conn.execute(
                """INSERT INTO task_state
                   (task_id, objective, completion_contract_json, contract_version,
                    intent_type, risk, status, created_at, updated_at, last_event_seq)
                   VALUES (?, ?, ?, ?, ?, ?, 'created', ?, ?, ?)""",
                (task_id, objective, canonical_json(completion_contract),
                 contract_version, intent_type, risk, now, now, seq),
            )
            if controller_principal is not None:
                assert self.controller_auth is not None
                authority_time = self.controller_auth.current_time()
                self.controller_auth.require_principal_in_transaction(
                    conn, controller_principal, now_value=authority_time)
                bound_at = authority_time.isoformat(
                    timespec="microseconds").replace("+00:00", "Z")
                authority_body = {
                    "task_id": task_id,
                    "controller_id": controller_principal.controller_id,
                    "session_id": controller_principal.session_id,
                    "controller_epoch": controller_principal.controller_epoch,
                }
                _, authority_seq = self.graph.append_event(
                    conn, "task.controller_authority_bound",
                    authority_body,
                    actor=controller_principal.controller_id,
                    session_id=session_id, turn_id=turn_id,
                    task_id=task_id)
                conn.execute(
                    """INSERT INTO controller_task_authorities
                       (task_id,controller_id,controller_key_sha256,
                        controller_epoch,session_id,
                        session_absolute_expires_at,
                        transport_binding_sha256,origin_sha256,bound_at,
                        bound_event_seq)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (task_id, controller_principal.controller_id,
                     controller_principal.public_key_sha256,
                     controller_principal.controller_epoch,
                     controller_principal.session_id,
                     controller_principal.absolute_expires_at,
                     controller_principal.transport_binding_sha256,
                     controller_principal.origin_sha256,
                     bound_at, authority_seq),
                )
                conn.execute(
                    "UPDATE task_state SET last_event_seq=? WHERE task_id=?",
                    (authority_seq, task_id),
                )
            progress = self._append_progress(
                conn, event_id, task_id,
                self._progress_payload(task_id, "task", "created",
                                       "Task created", objective[:180]),
            )
        return task_id, progress

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self.graph._connect() as conn:
            row = conn.execute("SELECT * FROM task_state WHERE task_id=?",
                               (task_id,)).fetchone()
        if row is None:
            return None
        return dict(row) | {
            "completion_contract": json.loads(row["completion_contract_json"]),
            "plan": json.loads(row["plan_json"]),
            "verification": (json.loads(row["verification_json"])
                             if row["verification_json"] else None),
            "cancellation_requested": bool(row["cancellation_requested"]),
        }

    def transition(self, task_id: str, new_status: str, *,
                   expected_status: str | None = None, label: str | None = None,
                   detail: str | None = None, error: str | None = None,
                   actor: str = "friday") -> dict[str, Any]:
        with self.graph.transaction() as conn:
            row = conn.execute("SELECT * FROM task_state WHERE task_id=?",
                               (task_id,)).fetchone()
            if row is None:
                raise ValueError("task does not exist")
            old = row["status"]
            if expected_status and old != expected_status:
                raise ValueError(f"expected task state {expected_status}, found {old}")
            if new_status not in ALLOWED_TRANSITIONS.get(old, set()):
                raise ValueError(f"invalid task transition: {old} -> {new_status}")
            if (new_status == "completed" and int(row["contract_version"] or 0) >= 1
                    and row["verification_status"] != "passed"):
                raise ValueError("a versioned task requires passed verification")
            if new_status == "completed":
                incomplete_steps = int(conn.execute(
                    """SELECT COUNT(*) FROM task_steps WHERE task_id=?
                       AND status<>'succeeded'""", (task_id,)).fetchone()[0])
                if incomplete_steps:
                    raise ValueError(
                        "task cannot complete before every durable step succeeds")
            payload = {"task_id": task_id, "from": old, "to": new_status}
            if error:
                payload["error"] = error
            event_id, seq = self.graph.append_event(
                conn, "task.transitioned", payload, actor=actor, task_id=task_id)
            now = utc_now()
            conn.execute(
                """UPDATE task_state SET status=?, last_error=?, updated_at=?,
                   last_event_seq=?, lease_id=CASE WHEN ? IN ('completed','failed',
                   'cancelled','waiting_input') THEN NULL ELSE lease_id END,
                   lease_expires_at=CASE WHEN ? IN ('completed','failed','cancelled',
                   'waiting_input') THEN NULL ELSE lease_expires_at END
                   WHERE task_id=?""",
                (new_status, error, now, seq, new_status, new_status, task_id),
            )
            progress = self._append_progress(
                conn, event_id, task_id,
                self._progress_payload(task_id, "task", new_status,
                                       label or new_status.replace("_", " ").title(), detail),
            )
        return progress

    def set_plan(self, task_id: str,
                 steps: list[str] | TaskPlan | list[dict[str, Any]], *,
                 actor: str = "friday") -> dict[str, Any]:
        if isinstance(steps, TaskPlan):
            encoded_plan: Any = steps.model_dump(mode="json")
            plan_steps = encoded_plan["steps"]
        elif steps and isinstance(steps[0], dict):
            encoded_plan = TaskPlan.model_validate(
                {"steps": steps}).model_dump(mode="json")
            plan_steps = encoded_plan["steps"]
        else:
            if not steps or any(not str(step).strip() for step in steps):
                raise ValueError("plan must contain non-empty steps")
            encoded_plan = [str(step) for step in steps]
            plan_steps = encoded_plan
        with self.graph.transaction() as conn:
            row = conn.execute("SELECT status FROM task_state WHERE task_id=?",
                               (task_id,)).fetchone()
            if row is None:
                raise ValueError("task does not exist")
            payload = {"task_id": task_id, "plan": encoded_plan}
            event_id, seq = self.graph.append_event(
                conn, "task.plan_set", payload, actor=actor, task_id=task_id)
            plan_id = self.graph.append_node(conn, "plan", payload,
                                             event_id=event_id)
            self.graph.append_edge(conn, task_id, "contains", plan_id,
                                   event_id=event_id)
            conn.execute(
                """UPDATE task_state SET plan_json=?, active_step=?, updated_at=?,
                   last_event_seq=? WHERE task_id=?""",
                (canonical_json(encoded_plan),
                 (plan_steps[0].get("description")
                  if isinstance(plan_steps[0], dict) else plan_steps[0]),
                 utc_now(), seq, task_id),
            )
            return self._append_progress(
                conn, event_id, task_id,
                self._progress_payload(task_id, "plan", "updated",
                                       f"Plan ready: {len(plan_steps)} step(s)"),
            )

    @staticmethod
    def _step_payload_context(task_id: str, batch_id: str, step_id: str,
                              tool_name: str, args_sha256: str) -> str:
        return "\0".join((task_id, batch_id, step_id, tool_name, args_sha256))

    @staticmethod
    def _safe_step(row: Any) -> dict[str, Any]:
        return {
            "step_id": row["step_id"],
            "task_id": row["task_id"],
            "batch_id": row["batch_id"],
            "round_index": int(row["round_index"]),
            "ordinal": int(row["ordinal"]),
            "tool_call_id": row["tool_call_id"],
            "tool_name": row["tool_name"],
            "args": json.loads(row["args_redacted_json"]),
            "args_sha256": row["args_sha256"],
            "idempotency_key": row["idempotency_key"],
            "idempotency_class": row["idempotency_class"],
            "recovery_policy": row["recovery_policy"],
            "status": row["status"],
            "depends_on": json.loads(row["depends_on_json"]),
            "verifier": row["verifier"],
            "executor_binding": json.loads(row["executor_binding_json"]),
            "resource_claims": json.loads(row["resource_claims_json"]),
            "resource_lease_id": row["resource_lease_id"],
            "admission_state": row["admission_state"],
            "admission_reason": row["admission_reason"],
            "admission_checked_at": row["admission_checked_at"],
            "next_admission_at": row["next_admission_at"],
            "risk": row["risk"],
            "approval_status": row["approval_status"],
            "approval_id": row["approval_id"],
            "action_id": row["action_id"],
            "attempt_count": int(row["attempt_count"]),
            "max_attempts": int(row["max_attempts"]),
            "last_error": row["last_error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _open_step_args(self, row: Any) -> dict[str, Any]:
        context = self._step_payload_context(
            row["task_id"], row["batch_id"], row["step_id"],
            row["tool_name"], row["args_sha256"])
        args = self._step_cipher.open(row["args_ciphertext"], context=context)
        if sha256_text(canonical_json(args)) != row["args_sha256"]:
            raise RuntimeError("durable step argument hash does not match")
        return args

    @staticmethod
    def _batch_identity(task_id: str, round_index: int,
                        calls: list[dict[str, Any]]) -> str:
        fingerprint = canonical_json({
            "task_id": task_id,
            "round_index": round_index,
            "calls": [{
                "id": item["tool_call_id"],
                "tool": item["tool_name"],
                "args_sha256": item["args_sha256"],
                "executor_binding": item["executor_binding"],
                "resource_claims": item["resource_claims"],
            } for item in calls],
        })
        return "batch_" + hashlib.sha256(fingerprint.encode()).hexdigest()[:32]

    @classmethod
    def _assert_batch_identity(cls, batch: Any, rows: list[Any]) -> None:
        calls = [{
            "tool_call_id": str(row["tool_call_id"]),
            "tool_name": str(row["tool_name"]),
            "args_sha256": str(row["args_sha256"]),
            "executor_binding": json.loads(row["executor_binding_json"]),
            "resource_claims": json.loads(row["resource_claims_json"]),
        } for row in rows]
        expected = cls._batch_identity(
            str(batch["task_id"]), int(batch["round_index"]), calls)
        if expected != str(batch["batch_id"]):
            raise RuntimeError(
                "durable step batch fingerprint integrity check failed")

    def stage_step_batch(
        self,
        task_id: str,
        calls: list[dict[str, Any]],
        *,
        round_index: int,
        context: dict[str, Any] | None = None,
        actor: str = "friday",
    ) -> tuple[str, list[dict[str, Any]]]:
        """Persist an ordered model tool-call batch before any dispatch.

        ``calls`` may contain exact arguments, but events, nodes, progress, and
        query APIs receive redacted previews only.  The exact object is sealed
        with context-bound authenticated encryption for deterministic restart.
        Re-staging the byte-identical round is idempotent.
        """
        if not calls:
            raise ValueError("a durable step batch cannot be empty")
        if round_index < 0:
            raise ValueError("round_index must be non-negative")
        state = self.get(task_id)
        if state is None:
            raise ValueError("task does not exist")
        if state["status"] in TERMINAL:
            raise ValueError("cannot stage steps for a terminal task")

        allowed_context = {}
        for key in ("session_id", "turn_id", "utterance_id"):
            value = (context or {}).get(key)
            if value is not None:
                allowed_context[key] = str(value)

        normalized: list[dict[str, Any]] = []
        call_ids: set[str] = set()
        for index, call in enumerate(calls, 1):
            tool_call_id = str(call.get("tool_call_id") or call.get("id") or "")
            tool_name = str(call.get("tool_name") or call.get("name") or "")
            args = call.get("args", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid arguments for {tool_name}") from exc
            if not tool_call_id or not tool_name or not isinstance(args, dict):
                raise ValueError(f"durable call {index} is malformed")
            if tool_call_id in call_ids:
                raise ValueError("tool call IDs must be unique within a batch")
            call_ids.add(tool_call_id)
            risk = str(call.get("risk") or "low")
            approval_status = str(call.get("approval_status") or "not_required")
            if approval_status not in {"not_required", "pending", "approved"}:
                raise ValueError("invalid durable step approval status")
            idempotency_class = str(call.get("idempotency_class") or
                                    ("read_only" if risk == "read_only"
                                     else "non_repeatable"))
            if idempotency_class not in {
                "read_only", "idempotent", "reconcilable", "non_repeatable"
            }:
                raise ValueError("invalid durable step idempotency class")
            recovery_policy = str(call.get("recovery_policy") or
                                  ("retry" if idempotency_class in {
                                      "read_only", "idempotent"
                                  } else "reconcile"))
            if recovery_policy not in {"retry", "reconcile"}:
                raise ValueError("invalid durable step recovery policy")
            args_json = canonical_json(args)
            executor_binding = call.get("executor_binding") or {}
            resource_claims = call.get("resource_claims") or {}
            if not isinstance(executor_binding, dict):
                raise ValueError("durable executor binding must be an object")
            if not isinstance(resource_claims, dict):
                raise ValueError("durable resource claims must be an object")
            resource_claims = ResourceClaim.model_validate(
                resource_claims).model_dump(mode="json")
            if len(canonical_json(executor_binding)) > 4096:
                raise ValueError("durable executor binding is too large")
            if len(canonical_json(resource_claims)) > 4096:
                raise ValueError("durable resource claims are too large")
            normalized.append({
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "args": args,
                "args_sha256": sha256_text(args_json),
                "args_redacted": redact_tool_arguments(tool_name, args),
                "risk": risk,
                "approval_status": approval_status,
                "approval_id": call.get("approval_id"),
                "idempotency_class": idempotency_class,
                "recovery_policy": recovery_policy,
                "verifier": str(call.get("verifier") or "successful_receipt"),
                "executor_binding": executor_binding,
                "resource_claims": resource_claims,
                "max_attempts": int(call.get("max_attempts") or
                                    (3 if recovery_policy == "retry" else 1)),
            })

        batch_id = self._batch_identity(task_id, round_index, normalized)
        prepared: list[dict[str, Any]] = []
        prior_step_id: str | None = None
        for index, item in enumerate(normalized, 1):
            step_source = "\0".join((task_id, str(round_index), batch_id,
                                      item["tool_call_id"], str(index)))
            step_id = "step_" + hashlib.sha256(step_source.encode()).hexdigest()[:32]
            idempotency_source = "\0".join(
                (task_id, str(round_index), step_id, item["tool_name"],
                 item["args_sha256"]))
            idempotency_key = "act_" + hashlib.sha256(
                idempotency_source.encode()).hexdigest()
            payload_context = self._step_payload_context(
                task_id, batch_id, step_id, item["tool_name"],
                item["args_sha256"])
            prepared.append(item | {
                "step_id": step_id,
                "idempotency_key": idempotency_key,
                "depends_on": [prior_step_id] if prior_step_id else [],
                "args_ciphertext": self._step_cipher.seal(
                    item["args"], context=payload_context),
            })
            prior_step_id = step_id

        batch_status = ("waiting_approval"
                        if any(item["approval_status"] == "pending"
                               for item in prepared) else "queued")
        now = utc_now()
        with self.graph.transaction() as conn:
            prior = conn.execute(
                "SELECT * FROM task_step_batches "
                "WHERE task_id=? AND round_index=?", (task_id, round_index)
            ).fetchone()
            if prior is not None:
                if prior["batch_id"] != batch_id:
                    raise RuntimeError(
                        "a different durable batch already owns this task round")
                rows = conn.execute(
                    "SELECT * FROM task_steps WHERE batch_id=? ORDER BY ordinal",
                    (batch_id,)).fetchall()
                self._assert_batch_identity(prior, list(rows))
                return batch_id, [self._safe_step(row) for row in rows]

            start_ordinal = int(conn.execute(
                "SELECT COALESCE(MAX(ordinal),0) FROM task_steps WHERE task_id=?",
                (task_id,)).fetchone()[0])
            batch_body = {
                "batch_id": batch_id, "task_id": task_id,
                "round_index": round_index, "status": batch_status,
                "call_count": len(prepared), "context": allowed_context,
            }
            batch_event_id, batch_seq = self.graph.append_event(
                conn, "step_batch.staged", batch_body, actor=actor,
                task_id=task_id, idempotency_key=f"stage:{batch_id}")
            self.graph.append_node(conn, "step_batch", batch_body,
                                   event_id=batch_event_id, node_id=batch_id)
            self.graph.append_edge(conn, task_id, "contains", batch_id,
                                   event_id=batch_event_id)
            conn.execute(
                """INSERT INTO task_step_batches
                   (batch_id,task_id,round_index,status,context_json,created_at,
                    updated_at,last_event_seq) VALUES (?,?,?,?,?,?,?,?)""",
                (batch_id, task_id, round_index, batch_status,
                 canonical_json(allowed_context), now, now, batch_seq))

            for index, item in enumerate(prepared, 1):
                ordinal = start_ordinal + index
                step_status = ("waiting_approval"
                               if item["approval_status"] == "pending"
                               else "pending")
                body = {
                    "step_id": item["step_id"], "batch_id": batch_id,
                    "task_id": task_id, "round_index": round_index,
                    "ordinal": ordinal, "tool_call_id": item["tool_call_id"],
                    "tool_name": item["tool_name"],
                    "args": item["args_redacted"],
                    "args_sha256": item["args_sha256"],
                    "idempotency_key": item["idempotency_key"],
                    "idempotency_class": item["idempotency_class"],
                    "recovery_policy": item["recovery_policy"],
                    "status": step_status, "depends_on": item["depends_on"],
                    "verifier": item["verifier"], "risk": item["risk"],
                    "executor_binding": item["executor_binding"],
                    "resource_claims": item["resource_claims"],
                    "approval_status": item["approval_status"],
                }
                event_id, seq = self.graph.append_event(
                    conn, "step.selected", body, actor=actor, task_id=task_id,
                    idempotency_key=f"select:{item['step_id']}")
                self.graph.append_node(conn, "step", body, event_id=event_id,
                                       node_id=item["step_id"])
                self.graph.append_edge(conn, batch_id, "contains", item["step_id"],
                                       event_id=event_id)
                for dependency in item["depends_on"]:
                    self.graph.append_edge(conn, item["step_id"], "depends_on",
                                           dependency, event_id=event_id)
                conn.execute(
                    """INSERT INTO task_steps
                       (step_id,task_id,batch_id,round_index,ordinal,tool_call_id,
                        tool_name,args_ciphertext,args_redacted_json,args_sha256,
                        context_json,idempotency_key,idempotency_class,
                        recovery_policy,status,depends_on_json,verifier,risk,
                        executor_binding_json,resource_claims_json,
                        approval_status,approval_id,max_attempts,created_at,
                        updated_at,last_event_seq)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (item["step_id"], task_id, batch_id, round_index, ordinal,
                     item["tool_call_id"], item["tool_name"],
                     item["args_ciphertext"], canonical_json(item["args_redacted"]),
                     item["args_sha256"], canonical_json(allowed_context),
                     item["idempotency_key"], item["idempotency_class"],
                     item["recovery_policy"], step_status,
                     canonical_json(item["depends_on"]), item["verifier"],
                     item["risk"], canonical_json(item["executor_binding"]),
                     canonical_json(item["resource_claims"]),
                     item["approval_status"], item["approval_id"],
                     item["max_attempts"], now, now, seq))
            conn.execute(
                """UPDATE task_state SET active_step=?,updated_at=?,last_event_seq=?
                   WHERE task_id=?""",
                (prepared[0]["step_id"], now, seq, task_id))

        return batch_id, self.list_steps(batch_id=batch_id)

    def list_steps(self, *, task_id: str | None = None,
                   batch_id: str | None = None) -> list[dict[str, Any]]:
        if bool(task_id) == bool(batch_id):
            raise ValueError("provide exactly one of task_id or batch_id")
        field, value = (("task_id", task_id) if task_id else
                        ("batch_id", batch_id))
        with self.graph._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM task_steps WHERE {field}=? ORDER BY ordinal",
                (value,)).fetchall()
        return [self._safe_step(row) for row in rows]

    def step_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self.graph._connect() as conn:
            row = conn.execute(
                "SELECT * FROM task_step_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
        if row is None:
            return None
        return dict(row) | {"context": json.loads(row["context_json"]),
                            "steps": self.list_steps(batch_id=batch_id)}

    def list_reconciliations(self) -> list[dict[str, Any]]:
        """Return a privacy-safe queue of quarantined external actions."""
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT s.step_id,s.task_id,s.batch_id,s.tool_name,
                          s.args_redacted_json,s.risk,s.approval_status,
                          s.created_at,s.updated_at,r.status AS receipt_status
                   FROM task_steps s
                   JOIN task_step_batches b ON b.batch_id=s.batch_id
                   JOIN action_receipts r ON r.step_id=s.step_id
                   WHERE s.status='reconcile_required'
                     AND b.status='reconcile_required'
                     AND r.status='outcome_unknown'
                   ORDER BY s.updated_at,s.ordinal"""
            ).fetchall()
        return [{
            "step_id": row["step_id"],
            "task_id": row["task_id"],
            "batch_id": row["batch_id"],
            "tool_name": row["tool_name"],
            "args": json.loads(row["args_redacted_json"]),
            "risk": row["risk"],
            "approval_status": row["approval_status"],
            "status": "reconcile_required",
            "receipt_status": row["receipt_status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        } for row in rows]

    def reconciliation_candidate(
        self, step_id: str,
    ) -> ReconciliationCandidate:
        """Open exact quarantined state for a server-owned read-only probe."""
        with self.graph._connect() as conn:
            step = conn.execute(
                "SELECT * FROM task_steps WHERE step_id=?", (step_id,)
            ).fetchone()
            if step is None:
                raise ValueError("durable step does not exist")
            batch = conn.execute(
                "SELECT * FROM task_step_batches WHERE batch_id=?",
                (step["batch_id"],)).fetchone()
            if (step["status"] != "reconcile_required" or batch is None
                    or batch["status"] != "reconcile_required"):
                raise ValueError("durable step is not awaiting reconciliation")
            rows = conn.execute(
                "SELECT * FROM task_steps WHERE batch_id=? ORDER BY ordinal",
                (step["batch_id"],)).fetchall()
            self._assert_batch_identity(batch, list(rows))
            receipt = conn.execute(
                "SELECT * FROM action_receipts WHERE step_id=?", (step_id,)
            ).fetchone()
            attempt = conn.execute(
                """SELECT * FROM action_attempts WHERE step_id=?
                   ORDER BY attempt_number DESC LIMIT 1""", (step_id,)
            ).fetchone()
            if (receipt is None or receipt["status"] != "outcome_unknown"
                    or receipt["idempotency_key"] != step["idempotency_key"]
                    or receipt["action_id"] != step["action_id"]
                    or attempt is None or attempt["status"] != "abandoned"):
                raise RuntimeError(
                    "durable reconciliation evidence is inconsistent")
            args = self._open_step_args(step)
            binding = json.loads(step["executor_binding_json"])
        return ReconciliationCandidate(
            step_id=str(step["step_id"]), batch_id=str(step["batch_id"]),
            task_id=str(step["task_id"]), ordinal=int(step["ordinal"]),
            tool_name=str(step["tool_name"]), args=args,
            args_sha256=str(step["args_sha256"]),
            idempotency_key=str(step["idempotency_key"]),
            action_id=str(step["action_id"]),
            attempt_id=str(attempt["attempt_id"]), executor_binding=binding,
            executor_binding_sha256=sha256_text(canonical_json(binding)),
        )

    def resolve_reconciliation(
        self,
        candidate: ReconciliationCandidate,
        result: Any,
        *,
        succeeded: bool,
        verification: VerificationResult | dict[str, Any],
        effects: list[dict[str, Any]] | None = None,
        reason_code: str = "authoritative_reconciliation",
        actor: str = "reconciler",
    ) -> dict[str, Any]:
        """CAS one unknown receipt into a proven success or explicit failure.

        Callers cannot use this as a retry mechanism.  The original attempt
        stays abandoned and no executor is invoked; only independently obtained
        evidence may settle the quarantined logical action.
        """
        if not isinstance(candidate, ReconciliationCandidate):
            raise TypeError("an exact reconciliation candidate is required")
        if isinstance(verification, VerificationResult):
            verification = verification.model_dump(mode="json")
        if not isinstance(verification, dict):
            raise TypeError("reconciliation verification is required")
        verification_status = str(verification.get("status") or "")
        if succeeded and verification_status != "passed":
            raise ValueError("successful reconciliation requires passed evidence")
        if not succeeded and verification_status != "failed":
            raise ValueError("failed reconciliation requires explicit failed evidence")
        if effects is None:
            effects = list(verification.get("effects") or [])
        reason = str(reason_code or "").strip().lower()
        if re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{0,79}", reason) is None:
            raise ValueError("reconciliation reason code is invalid")
        if sha256_text(canonical_json(candidate.args)) != candidate.args_sha256:
            raise RuntimeError("reconciliation argument binding changed")
        if (sha256_text(canonical_json(candidate.executor_binding))
                != candidate.executor_binding_sha256):
            raise RuntimeError("reconciliation executor binding changed")

        now = utc_now()
        with self.graph.transaction() as conn:
            step = conn.execute(
                "SELECT * FROM task_steps WHERE step_id=?",
                (candidate.step_id,)).fetchone()
            if step is None:
                raise ValueError("durable step does not exist")
            batch = conn.execute(
                "SELECT * FROM task_step_batches WHERE batch_id=?",
                (candidate.batch_id,)).fetchone()
            if batch is None:
                raise RuntimeError("durable step batch does not exist")
            rows = conn.execute(
                "SELECT * FROM task_steps WHERE batch_id=? ORDER BY ordinal",
                (candidate.batch_id,)).fetchall()
            self._assert_batch_identity(batch, list(rows))
            binding = json.loads(step["executor_binding_json"])
            if (step["status"] != "reconcile_required"
                    or batch["status"] != "reconcile_required"
                    or step["batch_id"] != candidate.batch_id
                    or step["task_id"] != candidate.task_id
                    or batch["task_id"] != candidate.task_id
                    or int(step["ordinal"]) != candidate.ordinal
                    or step["tool_name"] != candidate.tool_name
                    or step["args_sha256"] != candidate.args_sha256
                    or step["idempotency_key"] != candidate.idempotency_key
                    or step["action_id"] != candidate.action_id
                    or sha256_text(canonical_json(binding))
                        != candidate.executor_binding_sha256):
                raise PermissionError("reconciliation candidate is stale")
            receipt = conn.execute(
                "SELECT * FROM action_receipts WHERE step_id=?",
                (candidate.step_id,)).fetchone()
            attempt = conn.execute(
                """SELECT * FROM action_attempts WHERE step_id=?
                   ORDER BY attempt_number DESC LIMIT 1""",
                (candidate.step_id,)).fetchone()
            if (receipt is None or receipt["status"] != "outcome_unknown"
                    or receipt["idempotency_key"] != candidate.idempotency_key
                    or receipt["action_id"] != candidate.action_id
                    or attempt is None
                    or attempt["attempt_id"] != candidate.attempt_id
                    or attempt["status"] != "abandoned"):
                raise PermissionError("reconciliation evidence is stale")

            recorded_result = redact_tool_result(candidate.tool_name, result)
            recorded_effects = _redact_effects(
                candidate.tool_name, effects or [])
            recorded_verification = _redact_verification(
                candidate.tool_name, verification)
            resolution = "succeeded" if succeeded else "failed"
            action_body = {
                "action_id": candidate.action_id,
                "step_id": candidate.step_id,
                "attempt_id": candidate.attempt_id,
                "resolution": resolution,
                "reason": reason,
                "executor_binding_sha256": candidate.executor_binding_sha256,
                "result": recorded_result,
                "verification": recorded_verification,
            }
            action_event_id, _ = self.graph.append_event(
                conn, "action.reconciled", action_body, actor=actor,
                task_id=candidate.task_id)
            observation_id = self.graph.append_node(
                conn, "observation", action_body, event_id=action_event_id)
            self.graph.append_edge(
                conn, candidate.action_id, "reconciled_by", observation_id,
                event_id=action_event_id)
            changed = conn.execute(
                """UPDATE action_receipts SET status=?,observation_id=?,
                   result_json=?,effects_json=?,verification_json=?,updated_at=?
                   WHERE step_id=? AND idempotency_key=?
                     AND status='outcome_unknown'""",
                (resolution, observation_id, canonical_json(recorded_result),
                 canonical_json(recorded_effects),
                 canonical_json(recorded_verification), now,
                 candidate.step_id, candidate.idempotency_key)).rowcount
            if changed != 1:
                raise PermissionError("reconciliation receipt lost its fence")

            step_event_type = (
                "step.reconciled" if succeeded
                else "step.reconciliation_failed")
            step_body = {
                "step_id": candidate.step_id,
                "batch_id": candidate.batch_id,
                "action_id": candidate.action_id,
                "attempt_id": candidate.attempt_id,
                "status": resolution,
                "reason": reason,
            }
            step_event_id, step_seq = self.graph.append_event(
                conn, step_event_type, step_body, actor=actor,
                task_id=candidate.task_id)
            changed = conn.execute(
                """UPDATE task_steps SET status=?,last_error=?,updated_at=?,
                   last_event_seq=? WHERE step_id=?
                     AND status='reconcile_required'""",
                (resolution, None if succeeded else reason, now, step_seq,
                 candidate.step_id)).rowcount
            if changed != 1:
                raise PermissionError("reconciliation step lost its fence")

            task = conn.execute(
                "SELECT status,cancellation_requested FROM task_state WHERE task_id=?",
                (candidate.task_id,)).fetchone()
            cancelled = bool(task and task["cancellation_requested"])
            if cancelled:
                batch_status = "cancelled"
                conn.execute(
                    """UPDATE task_steps SET status='cancelled',last_error=?,
                       updated_at=?,last_event_seq=? WHERE batch_id=?
                       AND status IN ('pending','waiting_approval')""",
                    ("task cancellation requested", now, step_seq,
                     candidate.batch_id))
            elif succeeded:
                statuses = [str(item[0]) for item in conn.execute(
                    "SELECT status FROM task_steps WHERE batch_id=? ORDER BY ordinal",
                    (candidate.batch_id,)).fetchall()]
                if statuses and all(item == "succeeded" for item in statuses):
                    batch_status = "succeeded"
                elif "waiting_approval" in statuses:
                    batch_status = "waiting_approval"
                else:
                    batch_status = "queued"
            else:
                batch_status = "failed"
                conn.execute(
                    """UPDATE task_steps SET status='skipped',last_error=?,
                       updated_at=?,last_event_seq=? WHERE batch_id=? AND ordinal>?
                       AND status IN ('pending','waiting_approval')""",
                    (f"blocked by reconciled failure {candidate.step_id}",
                     now, step_seq, candidate.batch_id, candidate.ordinal))
            conn.execute(
                """UPDATE task_step_batches SET status=?,updated_at=?,
                   last_event_seq=? WHERE batch_id=?""",
                (batch_status, now, step_seq, candidate.batch_id))
            next_row = conn.execute(
                """SELECT step_id FROM task_steps WHERE task_id=?
                   AND status IN ('pending','waiting_approval','running',
                                  'reconcile_required')
                   ORDER BY ordinal LIMIT 1""", (candidate.task_id,)
            ).fetchone()
            conn.execute(
                """UPDATE task_state SET active_step=?,updated_at=?,
                   last_event_seq=? WHERE task_id=?""",
                (next_row["step_id"] if next_row else None, now, step_seq,
                 candidate.task_id))
            progress = self._append_progress(
                conn, step_event_id, candidate.task_id,
                self._progress_payload(
                    candidate.task_id, "reconciliation", resolution,
                    (f"Reconciled {candidate.tool_name}"
                     if succeeded else
                     f"Closed uncertain {candidate.tool_name} as failed"),
                    ("Authoritative postcondition evidence was recorded."
                     if succeeded else
                     "No success was inferred from the uncertain attempt.")))

        return {
            "step_id": candidate.step_id,
            "task_id": candidate.task_id,
            "batch_id": candidate.batch_id,
            "status": resolution,
            "batch_status": batch_status,
            "resolved": True,
            "progress": progress,
        }

    def acknowledge_unknown_reconciliation(
        self,
        candidate: ReconciliationCandidate,
        *,
        reason_code: str = "operator_abandoned_unknown",
        actor: str = "user",
    ) -> dict[str, Any]:
        """Stop waiting without falsely converting uncertainty into failure.

        The task/batch may end, but the action receipt intentionally remains
        ``outcome_unknown`` forever unless authoritative evidence later settles
        it.  This records an operator workflow decision, not an observation.
        """
        if not isinstance(candidate, ReconciliationCandidate):
            raise TypeError("an exact reconciliation candidate is required")
        reason = str(reason_code or "").strip().lower()
        if re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{0,79}", reason) is None:
            raise ValueError("reconciliation reason code is invalid")
        if (sha256_text(canonical_json(candidate.args))
                != candidate.args_sha256
                or sha256_text(canonical_json(candidate.executor_binding))
                != candidate.executor_binding_sha256):
            raise RuntimeError("reconciliation candidate binding changed")
        now = utc_now()
        with self.graph.transaction() as conn:
            step = conn.execute(
                "SELECT * FROM task_steps WHERE step_id=?",
                (candidate.step_id,)).fetchone()
            batch = conn.execute(
                "SELECT * FROM task_step_batches WHERE batch_id=?",
                (candidate.batch_id,)).fetchone()
            if step is None or batch is None:
                raise ValueError("durable reconciliation does not exist")
            rows = conn.execute(
                "SELECT * FROM task_steps WHERE batch_id=? ORDER BY ordinal",
                (candidate.batch_id,)).fetchall()
            self._assert_batch_identity(batch, list(rows))
            binding = json.loads(step["executor_binding_json"])
            receipt = conn.execute(
                "SELECT * FROM action_receipts WHERE step_id=?",
                (candidate.step_id,)).fetchone()
            attempt = conn.execute(
                """SELECT * FROM action_attempts WHERE step_id=?
                   ORDER BY attempt_number DESC LIMIT 1""",
                (candidate.step_id,)).fetchone()
            if (step["status"] != "reconcile_required"
                    or batch["status"] != "reconcile_required"
                    or step["batch_id"] != candidate.batch_id
                    or step["task_id"] != candidate.task_id
                    or batch["task_id"] != candidate.task_id
                    or int(step["ordinal"]) != candidate.ordinal
                    or step["tool_name"] != candidate.tool_name
                    or step["args_sha256"] != candidate.args_sha256
                    or step["idempotency_key"] != candidate.idempotency_key
                    or step["action_id"] != candidate.action_id
                    or sha256_text(canonical_json(binding))
                        != candidate.executor_binding_sha256
                    or receipt is None
                    or receipt["status"] != "outcome_unknown"
                    or receipt["task_id"] != candidate.task_id
                    or receipt["action_id"] != candidate.action_id
                    or receipt["tool_name"] != candidate.tool_name
                    or receipt["args_sha256"] != candidate.args_sha256
                    or receipt["idempotency_key"] != candidate.idempotency_key
                    or attempt is None
                    or attempt["attempt_id"] != candidate.attempt_id
                    or attempt["step_id"] != candidate.step_id
                    or attempt["idempotency_key"] != candidate.idempotency_key
                    or attempt["status"] != "abandoned"):
                raise PermissionError("reconciliation candidate is stale")

            body = {
                "action_id": candidate.action_id,
                "step_id": candidate.step_id,
                "attempt_id": candidate.attempt_id,
                "status": "outcome_unknown",
                "workflow": "abandoned_unknown",
                "reason": reason,
                "executor_binding_sha256": candidate.executor_binding_sha256,
            }
            event_id, seq = self.graph.append_event(
                conn, "action.outcome_unknown_acknowledged", body,
                actor=actor, task_id=candidate.task_id)
            verification = {
                "status": "uncertain",
                "summary": (f"{candidate.tool_name} outcome remains unknown; "
                            "operator stopped reconciliation"),
                "evidence": [],
                "missing": ["authoritative postcondition evidence"],
                "effects": [],
                "acknowledged": True,
                "_redacted": True,
            }
            changed = conn.execute(
                """UPDATE action_receipts SET verification_json=?,updated_at=?
                   WHERE step_id=? AND idempotency_key=?
                     AND status='outcome_unknown'""",
                (canonical_json(verification), now, candidate.step_id,
                 candidate.idempotency_key)).rowcount
            if changed != 1:
                raise PermissionError("reconciliation receipt lost its fence")
            changed = conn.execute(
                """UPDATE task_steps SET status='abandoned_unknown',
                   last_error=?,updated_at=?,last_event_seq=? WHERE step_id=?
                     AND status='reconcile_required'""",
                (reason, now, seq, candidate.step_id)).rowcount
            if changed != 1:
                raise PermissionError("reconciliation step lost its fence")
            conn.execute(
                """UPDATE task_steps SET status='skipped',last_error=?,
                   updated_at=?,last_event_seq=? WHERE batch_id=? AND ordinal>?
                   AND status IN ('pending','waiting_approval')""",
                (f"blocked by unknown outcome {candidate.step_id}", now, seq,
                 candidate.batch_id, candidate.ordinal))
            task = conn.execute(
                "SELECT cancellation_requested FROM task_state WHERE task_id=?",
                (candidate.task_id,)).fetchone()
            batch_status = (
                "cancelled" if task and task["cancellation_requested"]
                else "failed")
            changed = conn.execute(
                """UPDATE task_step_batches SET status=?,updated_at=?,
                   last_event_seq=? WHERE batch_id=? AND task_id=?
                     AND status='reconcile_required'""",
                (batch_status, now, seq, candidate.batch_id,
                 candidate.task_id)).rowcount
            if changed != 1:
                raise PermissionError("reconciliation batch lost its fence")
            changed = conn.execute(
                """UPDATE task_state SET active_step=NULL,updated_at=?,
                   last_event_seq=? WHERE task_id=?""",
                (now, seq, candidate.task_id)).rowcount
            if changed != 1:
                raise PermissionError("reconciliation task lost its fence")
            progress = self._append_progress(
                conn, event_id, candidate.task_id,
                self._progress_payload(
                    candidate.task_id, "reconciliation", "abandoned_unknown",
                    f"Stopped reconciliation for {candidate.tool_name}",
                    "The external effect remains explicitly outcome_unknown."))
        return {
            "step_id": candidate.step_id,
            "task_id": candidate.task_id,
            "batch_id": candidate.batch_id,
            "status": "abandoned_unknown",
            "receipt_status": "outcome_unknown",
            "batch_status": batch_status,
            "resolved": True,
            "progress": progress,
        }

    def pending_step_batches(self) -> list[str]:
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT b.batch_id FROM task_step_batches b
                   JOIN task_state t ON t.task_id=b.task_id
                   WHERE b.status IN ('queued','running','succeeded','failed',
                                      'cancelled')
                     AND t.status NOT IN ('completed','failed','cancelled')
                   ORDER BY b.created_at"""
            ).fetchall()
        return [str(row[0]) for row in rows]

    def bind_step_approval(self, step_id: str, approval_id: str,
                           *, actor: str = "policy") -> dict[str, Any]:
        with self.graph.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM task_steps WHERE step_id=?", (step_id,)
            ).fetchone()
            if row is None:
                raise ValueError("durable step does not exist")
            if row["approval_status"] != "pending":
                raise ValueError("durable step is not awaiting approval")
            body = {"step_id": step_id, "approval_id": approval_id,
                    "status": "pending"}
            event_id, seq = self.graph.append_event(
                conn, "step.approval_bound", body, actor=actor,
                task_id=row["task_id"])
            conn.execute(
                """UPDATE task_steps SET approval_id=?,updated_at=?,last_event_seq=?
                   WHERE step_id=?""", (approval_id, utc_now(), seq, step_id))
            return self._append_progress(
                conn, event_id, row["task_id"], self._progress_payload(
                    row["task_id"], "approval", "waiting",
                    f"Approval required for {row['tool_name']}"))

    def resolve_step_approval(self, step_id: str, approved: bool, *,
                              approval_id: str | None = None,
                              actor: str = "user") -> dict[str, Any]:
        decision = "approved" if approved else "denied"
        with self.graph.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM task_steps WHERE step_id=?", (step_id,)
            ).fetchone()
            if row is None:
                raise ValueError("durable step does not exist")
            if row["approval_status"] != "pending":
                raise ValueError("durable step approval is already resolved")
            if approval_id and row["approval_id"] != approval_id:
                raise PermissionError("approval is not bound to this durable step")
            body = {"step_id": step_id, "approval_id": row["approval_id"],
                    "status": decision}
            event_id, seq = self.graph.append_event(
                conn, "step.approval_resolved", body, actor=actor,
                task_id=row["task_id"])
            new_status = "pending" if approved else "cancelled"
            conn.execute(
                """UPDATE task_steps SET approval_status=?,status=?,updated_at=?,
                   last_event_seq=? WHERE step_id=?""",
                (decision, new_status, utc_now(), seq, step_id))
            siblings = conn.execute(
                "SELECT status,approval_status FROM task_steps WHERE batch_id=?",
                (row["batch_id"],)).fetchall()
            if not approved:
                batch_status = "cancelled"
                conn.execute(
                    """UPDATE task_steps SET status='cancelled',updated_at=?,
                       last_event_seq=? WHERE batch_id=? AND status IN
                       ('pending','waiting_approval')""",
                    (utc_now(), seq, row["batch_id"]))
            elif all(item["approval_status"] in {"approved", "not_required"}
                     for item in siblings):
                batch_status = "queued"
            else:
                batch_status = "waiting_approval"
            conn.execute(
                """UPDATE task_step_batches SET status=?,updated_at=?,last_event_seq=?
                   WHERE batch_id=?""",
                (batch_status, utc_now(), seq, row["batch_id"]))
            return self._append_progress(
                conn, event_id, row["task_id"], self._progress_payload(
                    row["task_id"], "approval", decision,
                    f"{row['tool_name']} {decision}"))

    def _first_effect_authority(
        self, conn: Any, candidate: Any, *, committed_at: str,
    ) -> dict[str, Any] | None:
        authority = conn.execute(
            "SELECT * FROM controller_task_authorities WHERE task_id=?",
            (candidate["task_id"],),
        ).fetchone()
        context = json.loads(candidate["context_json"])
        interactive = bool(context.get("session_id"))
        if authority is None:
            if self.require_controller_authority and interactive:
                raise PermissionError(
                    "interactive effect lacks controller authority")
            return None
        if candidate["approval_status"] == "approved":
            if not candidate["approval_id"]:
                raise PermissionError(
                    "approved effect lacks an exact approval identity")
            row = conn.execute(
                """SELECT d.*,s.status AS session_status,
                          s.idle_expires_at,i.status AS controller_status,
                          i.auth_epoch AS current_epoch
                     FROM controller_approval_decisions d
                     JOIN controller_sessions s
                       ON s.session_id=d.decision_session_id
                     JOIN controller_identities i
                       ON i.controller_id=d.controller_id
                    WHERE d.approval_id=? AND d.task_id=? AND d.step_id=?
                      AND d.tool_name=? AND d.args_sha256=?
                      AND d.controller_id=?
                      AND d.controller_key_sha256=?
                      AND d.controller_epoch=? AND d.decision='approved'""",
                (candidate["approval_id"], candidate["task_id"],
                 candidate["step_id"], candidate["tool_name"],
                 candidate["args_sha256"], authority["controller_id"],
                 authority["controller_key_sha256"],
                 int(authority["controller_epoch"])),
            ).fetchone()
            if (row is None or row["session_status"] != "active"
                    or row["controller_status"] != "active"
                    or int(row["current_epoch"]) != int(row["controller_epoch"])
                    or row["idle_expires_at"] <= committed_at
                    or row["session_absolute_expires_at"] <= committed_at
                    or row["authorization_expires_at"] is None
                    or row["authorization_expires_at"] <= committed_at):
                raise PermissionError(
                    "approved effect controller authority is stale")
            return {
                "controller_id": row["controller_id"],
                "controller_key_sha256": row["controller_key_sha256"],
                "controller_epoch": int(row["controller_epoch"]),
                "authorizing_session_id": row["decision_session_id"],
                "session_absolute_expires_at":
                    row["session_absolute_expires_at"],
                "transport_binding_sha256":
                    row["transport_binding_sha256"],
                "origin_sha256": row["origin_sha256"],
                "approval_id": row["approval_id"],
                "decision_id": row["decision_id"],
            }
        if candidate["approval_status"] != "not_required":
            raise PermissionError("effect approval state is not dispatchable")
        row = conn.execute(
            """SELECT s.*,i.status AS controller_status,
                      i.auth_epoch AS current_epoch
                 FROM controller_sessions s
                 JOIN controller_identities i
                   ON i.controller_id=s.controller_id
                WHERE s.session_id=? AND s.controller_id=?
                  AND s.controller_key_sha256=? AND s.controller_epoch=?
                  AND s.absolute_expires_at=?
                  AND s.transport_binding_sha256=? AND s.origin_sha256=?""",
            (authority["session_id"], authority["controller_id"],
             authority["controller_key_sha256"],
             int(authority["controller_epoch"]),
             authority["session_absolute_expires_at"],
             authority["transport_binding_sha256"],
             authority["origin_sha256"]),
        ).fetchone()
        if (row is None or row["status"] != "active"
                or row["controller_status"] != "active"
                or int(row["current_epoch"]) != int(row["controller_epoch"])
                or row["idle_expires_at"] <= committed_at
                or row["absolute_expires_at"] <= committed_at):
            raise PermissionError("effect controller authority is stale")
        return {
            "controller_id": authority["controller_id"],
            "controller_key_sha256": authority["controller_key_sha256"],
            "controller_epoch": int(authority["controller_epoch"]),
            "authorizing_session_id": authority["session_id"],
            "session_absolute_expires_at":
                authority["session_absolute_expires_at"],
            "transport_binding_sha256":
                authority["transport_binding_sha256"],
            "origin_sha256": authority["origin_sha256"],
            "approval_id": None,
            "decision_id": None,
        }

    def claim_next_step(self, batch_id: str, worker_id: str, *,
                        lease_seconds: int = 300,
                        actor: str = "worker") -> ClaimedStep | None:
        """Atomically fence and journal the next dependency-ready dispatch."""
        if not worker_id.strip():
            raise ValueError("worker_id is required")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        admission_snapshot: ResourceSnapshot | None = None
        if self.admission is not None:
            try:
                admission_snapshot = self.admission.get_snapshot()
                self.admission_sensor_error = None
            except Exception as exc:
                # Fail closed before action.started while leaving the durable
                # step retryable. Health/status expose the sensor failure.
                self.admission_sensor_error = str(exc)[:500]
                admission_snapshot = ResourceSnapshot(
                    available_cpu_millis=0, available_ram_mib=0,
                    available_network_slots=0,
                    available_accelerator_vram_mib={},
                    captured_at=datetime.now(UTC))
        # Timestamp the claim after live sampling so even a slow hardware probe
        # cannot make a freshly captured sample appear future-dated.
        instant = datetime.now(UTC)
        expires = (instant + timedelta(seconds=lease_seconds)).isoformat(
            timespec="microseconds").replace("+00:00", "Z")
        now = utc_now()
        authority_now = (
            self.controller_auth.current_time().isoformat(
                timespec="microseconds").replace("+00:00", "Z")
            if self.controller_auth is not None else now)
        if self.admission is not None:
            self.admission_sensor_checked_at = now
        claim_data: dict[str, Any] | None = None
        with self.graph.transaction() as conn:
            batch = conn.execute(
                "SELECT * FROM task_step_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise ValueError("durable step batch does not exist")
            if batch["status"] not in {"queued", "running"}:
                return None
            task = conn.execute(
                "SELECT status,cancellation_requested FROM task_state WHERE task_id=?",
                (batch["task_id"],)).fetchone()
            if (task is None or task["status"] in TERMINAL
                    or bool(task["cancellation_requested"])):
                return None
            rows = conn.execute(
                "SELECT * FROM task_steps WHERE batch_id=? ORDER BY ordinal",
                (batch_id,)).fetchall()
            self._assert_batch_identity(batch, list(rows))
            statuses = {str(row["step_id"]): str(row["status"]) for row in rows}
            candidate = None
            for row in rows:
                status = str(row["status"])
                if status == "succeeded":
                    continue
                if status != "pending":
                    return None
                dependencies = json.loads(row["depends_on_json"])
                if any(statuses.get(dep) != "succeeded" for dep in dependencies):
                    return None
                candidate = row
                break
            if candidate is None:
                if rows and all(row["status"] == "succeeded" for row in rows):
                    conn.execute(
                        "UPDATE task_step_batches SET status='succeeded',updated_at=? "
                        "WHERE batch_id=?", (now, batch_id))
                return None

            # Decryption and hash verification happen before the dispatch marker
            # commits.  A missing/tampered payload therefore cannot become an
            # ambiguous external action.
            args = self._open_step_args(candidate)
            receipt = conn.execute(
                "SELECT * FROM action_receipts WHERE idempotency_key=?",
                (candidate["idempotency_key"],)).fetchone()
            if receipt is not None and receipt["status"] == "succeeded":
                body = {"step_id": candidate["step_id"],
                        "action_id": receipt["action_id"],
                        "status": "succeeded", "reason": "receipt_reconciled"}
                event_id, seq = self.graph.append_event(
                    conn, "step.reconciled", body, actor=actor,
                    task_id=candidate["task_id"])
                conn.execute(
                    """UPDATE task_steps SET status='succeeded',action_id=?,
                       lease_id=NULL,worker_id=NULL,lease_expires_at=NULL,
                       updated_at=?,last_event_seq=? WHERE step_id=?""",
                    (receipt["action_id"], now, seq, candidate["step_id"]))
                return None
            if receipt is not None and receipt["status"] != "running":
                raise RuntimeError("durable action receipt is not dispatchable")

            first_effect_authority = None
            if receipt is None:
                first_effect_authority = self._first_effect_authority(
                    conn, candidate, committed_at=authority_now)
            else:
                task_authority = conn.execute(
                    "SELECT 1 FROM controller_task_authorities WHERE task_id=?",
                    (candidate["task_id"],),
                ).fetchone()
                context = json.loads(candidate["context_json"])
                if task_authority is not None or (
                        self.require_controller_authority
                        and context.get("session_id")):
                    prior_use = conn.execute(
                        """SELECT 1 FROM controller_effect_uses
                           WHERE action_id=? AND task_id=? AND step_id=?
                             AND idempotency_key=? AND tool_name=?
                             AND args_sha256=?""",
                        (receipt["action_id"], candidate["task_id"],
                         candidate["step_id"], candidate["idempotency_key"],
                         candidate["tool_name"], candidate["args_sha256"]),
                    ).fetchone()
                    if prior_use is None:
                        raise PermissionError(
                            "effect recovery lacks committed controller use")

            attempt_number = int(candidate["attempt_count"]) + 1
            if attempt_number > int(candidate["max_attempts"]):
                raise RuntimeError("durable step retry budget is exhausted")
            lease_id = f"step_lease_{uuid.uuid4().hex}"
            attempt_id = "attempt_" + hashlib.sha256(
                f"{candidate['step_id']}\0{attempt_number}".encode()
            ).hexdigest()[:32]
            resource_lease_id: str | None = None
            if self.admission is not None:
                resource_claim = ResourceClaim.model_validate_json(
                    candidate["resource_claims_json"])
                decision = self.admission.acquire_in_transaction(
                    conn, resource_claim, step_id=str(candidate["step_id"]),
                    attempt_id=attempt_id, worker_id=worker_id,
                    snapshot=admission_snapshot, now=instant,
                    lease_ttl_seconds=lease_seconds)
                if not decision.admitted:
                    next_check = (
                        instant + timedelta(seconds=1)
                    ).isoformat(timespec="microseconds").replace("+00:00", "Z")
                    state = ("deferred" if decision.retryable else "rejected")
                    if (candidate["admission_state"] != state
                            or candidate["admission_reason"] != decision.reason):
                        body = {
                            "step_id": candidate["step_id"],
                            "batch_id": batch_id, "status": state,
                            "reason": decision.reason,
                            "deficits": decision.deficits,
                        }
                        event_id, decision_seq = self.graph.append_event(
                            conn, f"step.admission_{state}", body,
                            actor="resource_admission",
                            task_id=candidate["task_id"])
                        self._append_progress(
                            conn, event_id, candidate["task_id"],
                            self._progress_payload(
                                candidate["task_id"], "admission", state,
                                ("Waiting for safe machine capacity"
                                 if decision.retryable else
                                 "Step exceeds this runtime profile"),
                                decision.reason))
                    else:
                        decision_seq = int(candidate["last_event_seq"])
                    if decision.retryable:
                        conn.execute(
                            """UPDATE task_steps SET admission_state='deferred',
                               admission_reason=?,admission_checked_at=?,
                               next_admission_at=?,updated_at=?,last_event_seq=?
                               WHERE step_id=? AND status='pending'""",
                            (decision.reason, now, next_check, now, decision_seq,
                             candidate["step_id"]))
                    else:
                        conn.execute(
                            """UPDATE task_steps SET status='failed',
                               admission_state='rejected',admission_reason=?,
                               admission_checked_at=?,next_admission_at=NULL,
                               last_error=?,updated_at=?,last_event_seq=?
                               WHERE step_id=? AND status='pending'""",
                            (decision.reason, now, decision.reason, now,
                             decision_seq, candidate["step_id"]))
                        conn.execute(
                            """UPDATE task_steps SET status='skipped',last_error=?,
                               updated_at=?,last_event_seq=? WHERE batch_id=?
                               AND ordinal>? AND status IN
                               ('pending','waiting_approval')""",
                            (f"blocked by rejected step {candidate['step_id']}",
                             now, decision_seq, batch_id,
                             int(candidate["ordinal"])))
                        conn.execute(
                            """UPDATE task_step_batches SET status='failed',
                               updated_at=?,last_event_seq=? WHERE batch_id=?""",
                            (now, decision_seq, batch_id))
                    return None
                resource_lease_id = decision.lease_id
            recorded_args = json.loads(candidate["args_redacted_json"])
            if receipt is None:
                action_body = {
                    "task_id": candidate["task_id"],
                    "step_id": candidate["step_id"],
                    "tool": candidate["tool_name"], "args": recorded_args,
                    "ordinal": int(candidate["ordinal"]),
                    "idempotency_key": candidate["idempotency_key"],
                    "risk": candidate["risk"],
                    "approval_status": candidate["approval_status"],
                }
                if first_effect_authority is not None:
                    action_body["controller_id"] = (
                        first_effect_authority["controller_id"])
                    action_body["controller_session_id"] = (
                        first_effect_authority["authorizing_session_id"])
                    action_body["approval_decision_id"] = (
                        first_effect_authority["decision_id"])
                event_id, seq = self.graph.append_event(
                    conn, "action.started", action_body, actor=actor,
                    task_id=candidate["task_id"],
                    idempotency_key=candidate["idempotency_key"])
                action_id = self.graph.append_node(
                    conn, "action", action_body, event_id=event_id)
                self.graph.append_edge(conn, candidate["task_id"], "attempts",
                                       action_id, event_id=event_id)
                self.graph.append_edge(conn, candidate["step_id"], "dispatches",
                                       action_id, event_id=event_id)
                conn.execute(
                    """INSERT INTO action_receipts
                       (idempotency_key,task_id,step_id,action_id,tool_name,
                        args_sha256,status,risk,approval_status,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,'running',?,?,?,?)""",
                    (candidate["idempotency_key"], candidate["task_id"],
                     candidate["step_id"], action_id, candidate["tool_name"],
                     candidate["args_sha256"], candidate["risk"],
                     candidate["approval_status"], now, now))
                if first_effect_authority is not None:
                    conn.execute(
                        """INSERT INTO controller_effect_uses
                           (action_id,task_id,step_id,idempotency_key,tool_name,
                            args_sha256,controller_id,
                            controller_key_sha256,controller_epoch,
                            authorizing_session_id,
                            session_absolute_expires_at,
                            transport_binding_sha256,origin_sha256,
                            approval_id,decision_id,committed_at,
                            committed_event_seq)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (action_id, candidate["task_id"],
                         candidate["step_id"], candidate["idempotency_key"],
                         candidate["tool_name"], candidate["args_sha256"],
                         first_effect_authority["controller_id"],
                         first_effect_authority["controller_key_sha256"],
                         first_effect_authority["controller_epoch"],
                         first_effect_authority["authorizing_session_id"],
                         first_effect_authority[
                             "session_absolute_expires_at"],
                         first_effect_authority[
                             "transport_binding_sha256"],
                         first_effect_authority["origin_sha256"],
                         first_effect_authority["approval_id"],
                         first_effect_authority["decision_id"],
                         authority_now, seq),
                    )
            else:
                action_id = str(receipt["action_id"])
                retry_body = {
                    "task_id": candidate["task_id"],
                    "step_id": candidate["step_id"], "action_id": action_id,
                    "tool": candidate["tool_name"], "args": recorded_args,
                    "attempt_number": attempt_number,
                    "reason": "safe restart retry",
                }
                event_id, seq = self.graph.append_event(
                    conn, "action.retry_started", retry_body, actor=actor,
                    task_id=candidate["task_id"])

            attempt_body = {
                "attempt_id": attempt_id, "step_id": candidate["step_id"],
                "action_id": action_id, "attempt_number": attempt_number,
                "lease_id": lease_id, "worker_id": worker_id,
                "resource_lease_id": resource_lease_id,
            }
            self.graph.append_node(conn, "action_attempt", attempt_body,
                                   event_id=event_id, node_id=attempt_id)
            self.graph.append_edge(conn, action_id, "has_attempt", attempt_id,
                                   event_id=event_id)
            conn.execute(
                """INSERT INTO action_attempts
                   (attempt_id,idempotency_key,step_id,attempt_number,lease_id,
                    worker_id,status,started_at,last_event_seq)
                   VALUES (?,?,?,?,?,?,'running',?,?)""",
                (attempt_id, candidate["idempotency_key"], candidate["step_id"],
                 attempt_number, lease_id, worker_id, now, seq))
            changed = conn.execute(
                """UPDATE task_steps SET status='running',action_id=?,lease_id=?,
                   worker_id=?,lease_expires_at=?,attempt_count=?,last_error=NULL,
                   resource_lease_id=?,admission_state=?,admission_reason=?,
                   admission_checked_at=?,next_admission_at=NULL,
                   updated_at=?,last_event_seq=?
                   WHERE step_id=? AND status='pending' AND lease_id IS NULL""",
                (action_id, lease_id, worker_id, expires, attempt_number,
                 resource_lease_id,
                 "admitted" if self.admission is not None else "not_required",
                 ("capacity_reserved" if self.admission is not None else None),
                 now if self.admission is not None else None,
                 now, seq, candidate["step_id"])).rowcount
            if changed != 1:
                raise RuntimeError("durable step claim lost its fence")
            conn.execute(
                """UPDATE task_step_batches SET status='running',updated_at=?,
                   last_event_seq=? WHERE batch_id=?""",
                (now, seq, batch_id))
            conn.execute(
                """UPDATE task_state SET active_step=?,updated_at=?,last_event_seq=?
                   WHERE task_id=?""",
                (candidate["step_id"], now, seq, candidate["task_id"]))
            progress = self._progress_payload(
                candidate["task_id"], "act", "started",
                *self._describe_action(candidate["tool_name"], recorded_args))
            self._append_progress(conn, event_id, candidate["task_id"], progress)
            claim_data = {
                "step_id": candidate["step_id"], "batch_id": batch_id,
                "task_id": candidate["task_id"],
                "round_index": int(candidate["round_index"]),
                "ordinal": int(candidate["ordinal"]),
                "tool_call_id": candidate["tool_call_id"],
                "tool_name": candidate["tool_name"], "args": args,
                "idempotency_key": candidate["idempotency_key"],
                "idempotency_class": candidate["idempotency_class"],
                "recovery_policy": candidate["recovery_policy"],
                "risk": candidate["risk"],
                "approval_status": candidate["approval_status"],
                "action_id": action_id, "attempt_id": attempt_id,
                "attempt_number": attempt_number, "lease_id": lease_id,
                "worker_id": worker_id, "verifier": candidate["verifier"],
                "executor_binding": json.loads(
                    candidate["executor_binding_json"]),
                "resource_claims": json.loads(
                    candidate["resource_claims_json"]),
                "context": json.loads(candidate["context_json"]),
                "resource_lease_id": resource_lease_id,
            }
        return ClaimedStep(**claim_data) if claim_data else None

    def heartbeat_step(self, claim: ClaimedStep, *,
                       lease_seconds: int = 300) -> bool:
        expires = (datetime.now(UTC) + timedelta(seconds=lease_seconds)).isoformat(
            timespec="microseconds").replace("+00:00", "Z")
        with self.graph.transaction() as conn:
            if self.admission is not None and claim.resource_lease_id:
                renewed = self.admission.heartbeat_in_transaction(
                    conn, claim.resource_lease_id, claim.attempt_id,
                    worker_id=claim.worker_id,
                    lease_ttl_seconds=lease_seconds)
                if (not renewed
                        and not self.admission
                        .is_step_lease_safely_discharged_in_transaction(
                            conn, claim.resource_lease_id, claim.attempt_id,
                            worker_id=claim.worker_id)):
                    return False
            changed = conn.execute(
                """UPDATE task_steps SET lease_expires_at=?,updated_at=?
                   WHERE step_id=? AND status='running' AND lease_id=?
                     AND worker_id=?""",
                (expires, utc_now(), claim.step_id, claim.lease_id,
                 claim.worker_id)).rowcount
        return changed == 1

    def finish_step(self, claim: ClaimedStep, result: Any, *, succeeded: bool,
                    verification: VerificationResult | dict[str, Any] | None = None,
                    effects: list[dict[str, Any]] | None = None,
                    actor: str = "worker") -> dict[str, Any]:
        """Atomically finish the attempt, logical receipt, and durable step."""
        if isinstance(verification, VerificationResult):
            verification = verification.model_dump(mode="json")
        if verification and effects is None:
            effects = list(verification.get("effects", []))
        verification_passed = (verification is None
                               or str(verification.get("status")) == "passed")
        step_succeeded = bool(succeeded and verification_passed)
        now = utc_now()
        with self.graph.transaction() as conn:
            step = conn.execute(
                "SELECT * FROM task_steps WHERE step_id=?", (claim.step_id,)
            ).fetchone()
            if step is None:
                raise ValueError("durable step does not exist")
            if (step["status"] != "running" or step["lease_id"] != claim.lease_id
                    or step["worker_id"] != claim.worker_id
                    or step["action_id"] != claim.action_id):
                raise PermissionError("durable step lease is stale")
            attempt = conn.execute(
                "SELECT * FROM action_attempts WHERE attempt_id=?",
                (claim.attempt_id,)).fetchone()
            receipt = conn.execute(
                "SELECT * FROM action_receipts WHERE idempotency_key=?",
                (claim.idempotency_key,)).fetchone()
            if (attempt is None or attempt["status"] != "running"
                    or attempt["lease_id"] != claim.lease_id):
                raise PermissionError("durable action attempt is stale")
            if receipt is None or receipt["status"] != "running":
                raise ValueError("durable action receipt is already finished")
            if self.admission is not None and claim.resource_lease_id:
                released = self.admission.release_in_transaction(
                    conn, claim.resource_lease_id, claim.attempt_id,
                    worker_id=claim.worker_id,
                    reason=("step_succeeded" if step_succeeded
                            else "step_failed"))
                if (not released
                        and not self.admission
                        .is_step_lease_safely_discharged_in_transaction(
                            conn, claim.resource_lease_id, claim.attempt_id,
                            worker_id=claim.worker_id)):
                    raise PermissionError("durable resource lease is stale")

            recorded_result = redact_tool_result(claim.tool_name, result)
            recorded_effects = _redact_effects(claim.tool_name, effects or [])
            recorded_verification = _redact_verification(
                claim.tool_name, verification)
            action_body = {
                "action_id": claim.action_id, "step_id": claim.step_id,
                "attempt_id": claim.attempt_id, "succeeded": step_succeeded,
                "result": recorded_result,
            }
            action_event_id, _ = self.graph.append_event(
                conn, "action.finished", action_body, actor=actor,
                task_id=claim.task_id)
            observation_id = self.graph.append_node(
                conn, "observation", action_body, event_id=action_event_id)
            self.graph.append_edge(conn, claim.action_id, "produced",
                                   observation_id, event_id=action_event_id)
            receipt_status = "succeeded" if step_succeeded else "failed"
            conn.execute(
                """UPDATE action_receipts SET status=?,observation_id=?,
                   result_json=?,effects_json=?,verification_json=?,updated_at=?
                   WHERE idempotency_key=?""",
                (receipt_status, observation_id, canonical_json(recorded_result),
                 canonical_json(recorded_effects),
                 (canonical_json(recorded_verification)
                  if recorded_verification else None), now,
                 claim.idempotency_key))
            error = None if step_succeeded else (
                str((verification or {}).get("summary") or result)[:1000])
            step_status = "succeeded" if step_succeeded else "failed"
            step_body = {
                "step_id": claim.step_id, "batch_id": claim.batch_id,
                "action_id": claim.action_id, "attempt_id": claim.attempt_id,
                "status": step_status, "attempt_number": claim.attempt_number,
            }
            step_event_id, step_seq = self.graph.append_event(
                conn, f"step.{step_status}", step_body, actor=actor,
                task_id=claim.task_id)
            conn.execute(
                """UPDATE action_attempts SET status=?,finished_at=?,last_error=?,
                   last_event_seq=? WHERE attempt_id=? AND lease_id=?""",
                (receipt_status, now, error, step_seq, claim.attempt_id,
                 claim.lease_id))
            conn.execute(
                """UPDATE task_steps SET status=?,lease_id=NULL,worker_id=NULL,
                   lease_expires_at=NULL,resource_lease_id=NULL,
                   admission_state='released',last_error=?,updated_at=?,
                   last_event_seq=?
                   WHERE step_id=? AND lease_id=?""",
                (step_status, error, now, step_seq, claim.step_id,
                 claim.lease_id))
            task_row = conn.execute(
                "SELECT status,cancellation_requested FROM task_state WHERE task_id=?",
                (claim.task_id,)).fetchone()
            cancellation_requested = bool(
                task_row and task_row["cancellation_requested"])
            if cancellation_requested:
                batch_status = "cancelled"
                conn.execute(
                    """UPDATE task_steps SET status='cancelled',last_error=?,
                       updated_at=?,last_event_seq=? WHERE batch_id=?
                       AND status IN ('pending','waiting_approval')""",
                    ("task cancellation requested", now, step_seq,
                     claim.batch_id))
            elif step_succeeded:
                remaining = int(conn.execute(
                    """SELECT COUNT(*) FROM task_steps WHERE batch_id=?
                       AND status<>'succeeded'""", (claim.batch_id,)).fetchone()[0])
                batch_status = "succeeded" if remaining == 0 else "running"
            else:
                batch_status = "failed"
                conn.execute(
                    """UPDATE task_steps SET status='skipped',last_error=?,
                       updated_at=?,last_event_seq=? WHERE batch_id=? AND ordinal>?
                       AND status IN ('pending','waiting_approval')""",
                    (f"blocked by failed step {claim.step_id}", now, step_seq,
                     claim.batch_id, claim.ordinal))
            conn.execute(
                """UPDATE task_step_batches SET status=?,updated_at=?,last_event_seq=?
                   WHERE batch_id=?""",
                (batch_status, now, step_seq, claim.batch_id))
            next_row = conn.execute(
                """SELECT step_id FROM task_steps WHERE task_id=?
                   AND status IN ('pending','waiting_approval','running')
                   ORDER BY ordinal LIMIT 1""", (claim.task_id,)).fetchone()
            conn.execute(
                """UPDATE task_state SET active_step=?,updated_at=?,last_event_seq=?
                   WHERE task_id=?""",
                (next_row["step_id"] if next_row else None, now, step_seq,
                 claim.task_id))
            if (cancellation_requested and task_row
                    and task_row["status"] not in TERMINAL):
                cancel_body = {"task_id": claim.task_id,
                               "from": task_row["status"], "to": "cancelled",
                               "reason": "in-flight action settled"}
                cancel_event_id, cancel_seq = self.graph.append_event(
                    conn, "task.transitioned", cancel_body, actor=actor,
                    task_id=claim.task_id)
                conn.execute(
                    """UPDATE task_state SET status='cancelled',active_step=NULL,
                       lease_id=NULL,lease_expires_at=NULL,updated_at=?,
                       last_event_seq=? WHERE task_id=?""",
                    (now, cancel_seq, claim.task_id))
                self._append_progress(
                    conn, cancel_event_id, claim.task_id,
                    self._progress_payload(
                        claim.task_id, "task", "cancelled", "Task cancelled",
                        "The in-flight receipt was recorded; no successor ran."))
            label = ("Completed " if step_succeeded else "Failed ") + claim.tool_name
            if tool_has_private_payload(claim.tool_name):
                detail = ("Private receipt recorded as verification metadata; "
                          f"result hash {_value_sha256(result)[:16]}.")
            elif isinstance(result, str) and result.startswith("error:"):
                detail = result[:240]
            elif isinstance(result, str):
                detail = f"Receipt recorded; result size {len(result)} characters."
            else:
                detail = "Receipt recorded."
            return self._append_progress(
                conn, step_event_id, claim.task_id,
                self._progress_payload(claim.task_id, "act", step_status,
                                       label, detail))

    def mark_step_outcome_unknown(
        self,
        claim: ClaimedStep,
        *,
        reason_code: str,
        actor: str = "worker",
    ) -> dict[str, Any]:
        """Quarantine a live attempt that may have crossed an effect boundary."""
        reason = str(reason_code or "").strip().lower()
        if re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{0,79}", reason) is None:
            reason = "external_action_outcome_unknown"
        # An explicit post-dispatch uncertainty signal always wins over a stale
        # or overly optimistic retry policy.  Quarantining is safer than either
        # replaying the effect or stranding a live lease because metadata was
        # misclassified.
        now = utc_now()
        with self.graph.transaction() as conn:
            step = conn.execute(
                "SELECT * FROM task_steps WHERE step_id=?", (claim.step_id,)
            ).fetchone()
            attempt = conn.execute(
                "SELECT * FROM action_attempts WHERE attempt_id=?",
                (claim.attempt_id,)).fetchone()
            receipt = conn.execute(
                "SELECT * FROM action_receipts WHERE idempotency_key=?",
                (claim.idempotency_key,)).fetchone()
            if (step is None or step["status"] != "running"
                    or step["lease_id"] != claim.lease_id
                    or step["worker_id"] != claim.worker_id
                    or step["action_id"] != claim.action_id
                    or attempt is None or attempt["status"] != "running"
                    or attempt["lease_id"] != claim.lease_id
                    or receipt is None or receipt["status"] != "running"
                    or receipt["step_id"] != claim.step_id):
                raise PermissionError("durable step lease is stale")
            if self.admission is not None and claim.resource_lease_id:
                fenced = self.admission.fence_interrupted_in_transaction(
                    conn, claim.resource_lease_id, claim.attempt_id,
                    worker_id=claim.worker_id, now=now)
                if (not fenced
                        and not self.admission
                        .is_step_lease_safely_discharged_in_transaction(
                            conn, claim.resource_lease_id, claim.attempt_id,
                            worker_id=claim.worker_id)):
                    raise PermissionError("durable resource lease is stale")

            body = {
                "step_id": claim.step_id,
                "batch_id": claim.batch_id,
                "action_id": claim.action_id,
                "attempt_id": claim.attempt_id,
                "from": "running",
                "to": "reconcile_required",
                "reason": reason,
            }
            event_id, seq = self.graph.append_event(
                conn, "step.reconciliation_required", body, actor=actor,
                task_id=claim.task_id)
            conn.execute(
                """UPDATE action_attempts SET status='abandoned',finished_at=?,
                   last_error=?,last_event_seq=? WHERE attempt_id=?
                     AND status='running' AND lease_id=?""",
                (now, reason, seq, claim.attempt_id, claim.lease_id))
            conn.execute(
                """UPDATE action_receipts SET status='outcome_unknown',
                   verification_json=?,updated_at=? WHERE idempotency_key=?
                     AND step_id=? AND status='running'""",
                (canonical_json({
                    "status": "uncertain",
                    "summary": (f"{claim.tool_name} outcome requires "
                                "authoritative reconciliation"),
                    "evidence": [],
                    "missing": ["authoritative postcondition evidence"],
                    "effects": [],
                    "_redacted": True,
                }), now, claim.idempotency_key, claim.step_id))
            changed = conn.execute(
                """UPDATE task_steps SET status='reconcile_required',
                   lease_id=NULL,worker_id=NULL,lease_expires_at=NULL,
                   resource_lease_id=NULL,admission_state='recovered',
                   last_error=?,updated_at=?,last_event_seq=? WHERE step_id=?
                     AND status='running' AND lease_id=? AND worker_id=?""",
                (reason, now, seq, claim.step_id, claim.lease_id,
                 claim.worker_id)).rowcount
            if changed != 1:
                raise PermissionError("durable step lease lost its fence")
            conn.execute(
                """UPDATE task_step_batches SET status='reconcile_required',
                   updated_at=?,last_event_seq=? WHERE batch_id=?""",
                (now, seq, claim.batch_id))
            conn.execute(
                """UPDATE task_state SET active_step=?,updated_at=?,
                   last_event_seq=? WHERE task_id=?""",
                (claim.step_id, now, seq, claim.task_id))
            task = conn.execute(
                """SELECT status,cancellation_requested FROM task_state
                   WHERE task_id=?""", (claim.task_id,)).fetchone()
            if (task is not None and task["cancellation_requested"]
                    and task["status"] not in TERMINAL):
                cancel_body = {
                    "task_id": claim.task_id,
                    "from": task["status"],
                    "to": "cancelled",
                    "reason": "in-flight action outcome quarantined",
                }
                cancel_event_id, cancel_seq = self.graph.append_event(
                    conn, "task.transitioned", cancel_body, actor=actor,
                    task_id=claim.task_id)
                changed = conn.execute(
                    """UPDATE task_state SET status='cancelled',active_step=NULL,
                       lease_id=NULL,lease_expires_at=NULL,updated_at=?,
                       last_event_seq=? WHERE task_id=?
                         AND cancellation_requested=1""",
                    (now, cancel_seq, claim.task_id)).rowcount
                if changed != 1:
                    raise PermissionError("cancelled task lost its fence")
                return self._append_progress(
                    conn, cancel_event_id, claim.task_id,
                    self._progress_payload(
                        claim.task_id, "task", "cancelled", "Task cancelled",
                        "The uncertain external effect remains visible for "
                        "passive reconciliation."))
            return self._append_progress(
                conn, event_id, claim.task_id,
                self._progress_payload(
                    claim.task_id, "reconciliation", "reconcile_required",
                    f"Checking uncertain outcome for {claim.tool_name}",
                    "The action was not repeated or recorded as failed."))

    def recover_inflight_steps(self, *, force: bool = False,
                               dead_worker_id: str | None = None,
                               force_reconcile_step_id: str | None = None,
                               actor: str = "supervisor") -> dict[str, list[str]]:
        """Fence interrupted dispatches and apply deterministic recovery policy."""
        recovered: dict[str, list[str]] = {"retry": [], "reconcile": []}
        now = utc_now()
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT step_id FROM task_steps
                   WHERE status IN ('leased','running') ORDER BY ordinal"""
            ).fetchall()
        for item in rows:
            with self.graph.transaction() as conn:
                step = conn.execute(
                    "SELECT * FROM task_steps WHERE step_id=?", (item["step_id"],)
                ).fetchone()
                if step is None or step["status"] not in {"leased", "running"}:
                    continue
                if dead_worker_id and step["worker_id"] != dead_worker_id:
                    continue
                if (force_reconcile_step_id is not None
                        and step["step_id"] != force_reconcile_step_id):
                    continue
                expired = bool(step["lease_expires_at"]
                               and step["lease_expires_at"] <= now)
                if not force and not expired:
                    continue
                attempt = conn.execute(
                    """SELECT * FROM action_attempts WHERE step_id=?
                       ORDER BY attempt_number DESC LIMIT 1""",
                    (step["step_id"],)).fetchone()
                if (self.admission is not None and step["resource_lease_id"]
                        and attempt is not None):
                    fenced = self.admission.fence_interrupted_in_transaction(
                        conn, str(step["resource_lease_id"]),
                        str(attempt["attempt_id"]),
                        worker_id=str(attempt["worker_id"]), now=now)
                    if not fenced:
                        lease = conn.execute(
                            "SELECT status FROM resource_leases WHERE lease_id=?",
                            (step["resource_lease_id"],)).fetchone()
                        if lease is not None and lease["status"] == "active":
                            raise RuntimeError(
                                "interrupted resource lease could not be fenced")
                forced_unknown = bool(
                    force_reconcile_step_id == step["step_id"])
                safely_retry = (not forced_unknown
                                and step["recovery_policy"] == "retry"
                                and int(step["attempt_count"])
                                < int(step["max_attempts"]))
                new_status = "pending" if safely_retry else "reconcile_required"
                event_type = ("step.retry_scheduled" if safely_retry
                              else "step.reconciliation_required")
                body = {
                    "step_id": step["step_id"], "batch_id": step["batch_id"],
                    "from": step["status"], "to": new_status,
                    "attempt_count": int(step["attempt_count"]),
                    "reason": ("external action outcome was reported unknown"
                               if forced_unknown
                               else "worker process interrupted"),
                }
                event_id, seq = self.graph.append_event(
                    conn, event_type, body, actor=actor, task_id=step["task_id"])
                if attempt is not None and attempt["status"] == "running":
                    conn.execute(
                        """UPDATE action_attempts SET status='abandoned',
                           finished_at=?,last_error=?,
                           last_event_seq=? WHERE attempt_id=?""",
                        (now,
                         ("external action outcome was reported unknown"
                          if forced_unknown else "worker process interrupted"),
                         seq, attempt["attempt_id"]))
                conn.execute(
                    """UPDATE task_steps SET status=?,lease_id=NULL,worker_id=NULL,
                       lease_expires_at=NULL,resource_lease_id=NULL,
                       admission_state='recovered',last_error=?,updated_at=?,
                       last_event_seq=?
                       WHERE step_id=?""",
                    (new_status,
                     ("external action outcome was reported unknown"
                      if forced_unknown else "worker process interrupted"),
                     now, seq,
                     step["step_id"]))
                batch_status = "queued" if safely_retry else "reconcile_required"
                conn.execute(
                    """UPDATE task_step_batches SET status=?,updated_at=?,
                       last_event_seq=? WHERE batch_id=?""",
                    (batch_status, now, seq, step["batch_id"]))
                if not safely_retry:
                    conn.execute(
                        """UPDATE action_receipts SET status='outcome_unknown',
                           updated_at=? WHERE step_id=? AND status='running'""",
                        (now, step["step_id"]))
                self._append_progress(
                    conn, event_id, step["task_id"], self._progress_payload(
                        step["task_id"], "recovery",
                        "retrying" if safely_retry else "reconcile_required",
                        (f"Retrying interrupted read-only step {step['tool_name']}"
                         if safely_retry else
                         f"Checking uncertain outcome for {step['tool_name']}")))
                if forced_unknown:
                    task = conn.execute(
                        """SELECT status,cancellation_requested FROM task_state
                           WHERE task_id=?""", (step["task_id"],)).fetchone()
                    if (task is not None and task["cancellation_requested"]
                            and task["status"] not in TERMINAL):
                        cancel_body = {
                            "task_id": step["task_id"],
                            "from": task["status"], "to": "cancelled",
                            "reason": "in-flight action outcome quarantined",
                        }
                        cancel_event_id, cancel_seq = self.graph.append_event(
                            conn, "task.transitioned", cancel_body, actor=actor,
                            task_id=step["task_id"])
                        changed = conn.execute(
                            """UPDATE task_state SET status='cancelled',
                               active_step=NULL,lease_id=NULL,
                               lease_expires_at=NULL,updated_at=?,last_event_seq=?
                               WHERE task_id=? AND cancellation_requested=1""",
                            (now, cancel_seq, step["task_id"])).rowcount
                        if changed != 1:
                            raise PermissionError(
                                "cancelled task lost its recovery fence")
                        self._append_progress(
                            conn, cancel_event_id, step["task_id"],
                            self._progress_payload(
                                step["task_id"], "task", "cancelled",
                                "Task cancelled",
                                "The uncertain external effect remains visible "
                                "for passive reconciliation."))
                recovered["retry" if safely_retry else "reconcile"].append(
                    str(step["step_id"]))
        return recovered

    def acquire_lease(self, task_id: str, *, minutes: int = 5) -> str:
        lease_id = f"lease_{uuid.uuid4().hex}"
        expires = (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat().replace(
            "+00:00", "Z")
        with self.graph.transaction() as conn:
            row = conn.execute("SELECT status FROM task_state WHERE task_id=?",
                               (task_id,)).fetchone()
            if row is None or row["status"] in TERMINAL:
                raise ValueError("task is not leasable")
            conn.execute(
                "UPDATE task_state SET lease_id=?, lease_expires_at=?, updated_at=? "
                "WHERE task_id=?", (lease_id, expires, utc_now(), task_id))
        return lease_id

    def begin_action(self, task_id: str, tool_name: str, args: dict[str, Any], *,
                     ordinal: int, risk: str = "low",
                     approval_status: str = "not_required",
                     actor: str = "friday") -> tuple[ActionHandle,
                                                       dict[str, Any] | None]:
        args_json = canonical_json(args)
        key_source = f"{task_id}\0{ordinal}\0{tool_name}\0{args_json}"
        key = "act_" + hashlib.sha256(key_source.encode()).hexdigest()
        with self.graph.transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM action_receipts WHERE idempotency_key=?", (key,)
            ).fetchone()
            if existing and existing["status"] == "succeeded":
                return (ActionHandle(existing["action_id"], key, True,
                                     json.loads(existing["result_json"])), None)
            if existing:
                raise RuntimeError("prior action outcome is not safely replayable")
            recorded_args = redact_tool_arguments(tool_name, args)
            body = {"task_id": task_id, "tool": tool_name, "args": recorded_args,
                    "ordinal": ordinal, "idempotency_key": key,
                    "risk": risk, "approval_status": approval_status}
            event_id, _ = self.graph.append_event(
                conn, "action.started", body, actor=actor, task_id=task_id,
                idempotency_key=key)
            action_id = self.graph.append_node(conn, "action", body,
                                               event_id=event_id)
            self.graph.append_edge(conn, task_id, "attempts", action_id,
                                   event_id=event_id)
            now = utc_now()
            conn.execute(
                """INSERT INTO action_receipts
                   (idempotency_key, task_id, action_id, tool_name, args_sha256,
                   status, risk, approval_status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)""",
                (key, task_id, action_id, tool_name, sha256_text(args_json), risk,
                 approval_status, now, now),
            )
            progress = self._append_progress(
                conn, event_id, task_id,
                self._progress_payload(task_id, "act", "started",
                                       *self._describe_action(tool_name,
                                                              recorded_args)),
            )
        return ActionHandle(action_id, key), progress

    @staticmethod
    def _describe_action(tool_name: str, args: dict[str, Any]) -> tuple[str, str]:
        """Return safe, human-readable progress without leaking file content."""
        path = str(args.get("path") or ".")[:160]
        descriptions = {
            "fetch_news": ("Fetching current news",
                           f"Live {str(args.get('region') or args.get('topic') or 'India')} headlines; topic: {str(args.get('topic') or 'top stories')[:120]}"),
            "web_search": ("Searching the live web",
                           f"Query: {str(args.get('query') or '')[:160]}"),
            "read_web": ("Reading a web page",
                         f"URL: {str(args.get('url') or '')[:180]}"),
            "browser_open": ("Opening the managed browser",
                             f"URL: {str(args.get('url') or '')[:180]}"),
            "browser_snapshot": ("Reading the managed browser",
                                 "Capturing the active page's visible text."),
            "browser_click": ("Clicking in the managed browser",
                              f"Selector: {str(args.get('selector') or '')[:160]}"),
            "browser_type": ("Typing in the managed browser",
                             f"Selector: {str(args.get('selector') or '')[:160]}; text is redacted."),
            "clipboard_read": ("Reading the clipboard",
                               "Reading up to 4,000 local text characters."),
            "clipboard_write": ("Updating the clipboard",
                                "Replacing local clipboard text."),
            "desktop_notify": ("Sending a desktop notification",
                               f"Title: {str(args.get('title') or '')[:120]}"),
            "open_local": ("Opening a local project file", f"Path: {path}"),
            "create_reminder": ("Creating a persistent reminder",
                                f"Due: {str(args.get('due_at') or '')[:120]}"),
            "list_reminders": ("Listing reminders",
                               f"Status: {str(args.get('status') or 'all')[:80]}"),
            "cancel_reminder": ("Cancelling a reminder",
                                f"Reminder: {str(args.get('reminder_id') or '')[:120]}"),
            "list_files": (f"Listing {path}",
                           "Reading the project directory; no files are being changed."),
            "read_file": (f"Reading {path}",
                          "Inspecting this project file; no files are being changed."),
            "write_file": (f"Testing a change to {path}",
                           "Applying exact user-approved content through verification and a recoverable checkpoint."),
            "machine_grant_path": ("Creating a machine path grant",
                                   "Recording one exact approved directory and permission set in encrypted storage."),
            "machine_list_grants": ("Listing machine path grants",
                                    "Reading redacted grant scopes and lifecycle state."),
            "machine_revoke_grant": ("Revoking a machine path grant",
                                     "Removing previously granted local authority."),
            "machine_inspect_path": ("Inspecting a granted machine path",
                                     "Reading metadata without following symbolic links."),
            "machine_list_path": ("Listing a granted machine directory",
                                  "Reading a bounded directory listing; sensitive descendants remain hidden."),
            "machine_read_text": ("Reading a granted machine file",
                                  "Reading bounded private text through the filesystem broker."),
            "machine_read_document": ("Reading a granted machine document",
                                      "Extracting bounded private text without executing embedded document content."),
            "machine_ocr_image": ("Reading text from a granted machine image",
                                  "Running bounded private OCR without granting the decoder ambient filesystem or network access."),
            "machine_understand_image": (
                "Understanding a granted machine image",
                "Sanitizing one private raster in a networkless boundary and asking the verified local vision model one bounded question."),
            "machine_write_text": ("Applying an approved machine file change",
                                   "Writing atomically, verifying the content hash, and recording an encrypted rollback checkpoint."),
            "machine_rollback_write": ("Rolling back a machine file change",
                                       "Restoring the encrypted checkpoint only if no later edit would be overwritten."),
            "restart": ("Restarting Friday",
                        str(args.get("reason") or "Applying a verified change")[:180]),
            "recall_memory": ("Searching verified memory",
                              f"Query: {str(args.get('query') or '')[:160]}"),
            "remember_preference": ("Recording an explicit preference",
                                    f"Preference key: {str(args.get('key') or '')[:120]}"),
            "create_skill": ("Drafting a skill version",
                             f"Skill: {str(args.get('name') or '')[:120]}; it will not activate without evaluation."),
            "list_skills": ("Inspecting skill lifecycle",
                            "Reading drafted, validated, active, and quarantined versions."),
            "create_capability": ("Building a capability candidate",
                                  "Static policy checks and executable tests run before activation."),
            "list_capabilities": ("Inspecting executable capabilities",
                                  "Reading active, drafted, and quarantined tool versions."),
            "create_voice_profile": ("Creating a voice candidate",
                                     "Registering a profile; it remains inactive until synthesis passes."),
            "list_voices": ("Inspecting voice profiles",
                            "Reading active, candidate, and validated voice configurations."),
            "set_voice": ("Testing a voice before activation",
                          f"Voice: {str(args.get('name') or '')[:120]}"),
            "rollback_voice": ("Testing the previous voice for rollback",
                               "The current voice remains active unless synthesis succeeds."),
            "upgrade_core": ("Starting a sandboxed core-upgrade agent",
                             "Pi can edit only an isolated copy; its candidate will remain awaiting explicit review."),
            "list_core_upgrades": ("Inspecting core-upgrade jobs",
                                   "Reading staged, rejected, and awaiting-review maintenance receipts."),
        }
        return descriptions.get(
            tool_name,
            (f"Running {tool_name}", "Executing a recorded, receipt-backed action."))

    def finish_action(self, handle: ActionHandle, result: Any, *, succeeded: bool,
                      verification: VerificationResult | dict[str, Any] | None = None,
                      effects: list[dict[str, Any]] | None = None,
                      actor: str = "friday") -> dict[str, Any]:
        if isinstance(verification, VerificationResult):
            verification = verification.model_dump(mode="json")
        if verification and effects is None:
            effects = list(verification.get("effects", []))
        with self.graph.transaction() as conn:
            receipt = conn.execute(
                "SELECT * FROM action_receipts WHERE idempotency_key=?",
                (handle.idempotency_key,),
            ).fetchone()
            if receipt is None:
                raise ValueError("action receipt does not exist")
            if receipt["status"] != "running":
                raise ValueError("action is already finished")
            tool_name = str(receipt["tool_name"])
            recorded_result = redact_tool_result(tool_name, result)
            recorded_effects = _redact_effects(tool_name, effects or [])
            recorded_verification = _redact_verification(tool_name, verification)
            body = {"action_id": handle.action_id, "succeeded": succeeded,
                    "result": recorded_result}
            event_id, _ = self.graph.append_event(
                conn, "action.finished", body, actor=actor,
                task_id=receipt["task_id"])
            observation_id = self.graph.append_node(conn, "observation", body,
                                                    event_id=event_id)
            self.graph.append_edge(conn, handle.action_id, "produced", observation_id,
                                   event_id=event_id)
            conn.execute(
                """UPDATE action_receipts SET status=?, observation_id=?, result_json=?,
                   effects_json=?, verification_json=?, updated_at=?
                   WHERE idempotency_key=?""",
                ("succeeded" if succeeded else "failed", observation_id,
                 canonical_json(recorded_result), canonical_json(recorded_effects),
                 (canonical_json(recorded_verification)
                  if recorded_verification else None),
                 utc_now(), handle.idempotency_key),
            )
            label = ("Completed " if succeeded else "Failed ") + tool_name
            if tool_has_private_payload(tool_name):
                detail = ("Private receipt recorded as verification metadata; "
                          f"result hash {_value_sha256(result)[:16]}.")
            elif isinstance(result, str) and result.startswith("error:"):
                detail = result[:240]
            elif isinstance(result, str):
                detail = f"Receipt recorded; result size {len(result)} characters."
            else:
                detail = "Receipt recorded."
            return self._append_progress(
                conn, event_id, receipt["task_id"],
                self._progress_payload(receipt["task_id"], "act",
                                       "succeeded" if succeeded else "failed", label,
                                       detail),
            )

    def record_verification(self, task_id: str,
                            result: VerificationResult | dict[str, Any], *,
                            actor: str = "verifier") -> dict[str, Any]:
        if isinstance(result, VerificationResult):
            result = result.model_dump(mode="json")
        status = str(result.get("status", "uncertain"))
        if status not in {"passed", "failed", "uncertain",
                          "user_confirmation_required"}:
            raise ValueError("invalid verification status")
        summary = str(result.get("summary") or "Verification produced no summary")
        evidence = list(result.get("evidence", []))
        missing = list(result.get("missing", []))
        with self.graph.transaction() as conn:
            if conn.execute("SELECT 1 FROM task_state WHERE task_id=?",
                            (task_id,)).fetchone() is None:
                raise ValueError("task does not exist")
            body = {"task_id": task_id, **result}
            event_id, seq = self.graph.append_event(
                conn, "task.verified", body, actor=actor, task_id=task_id)
            verification_id = self.graph.append_node(
                conn, "evaluation", body, event_id=event_id,
                node_id=new_id("verification"))
            self.graph.append_edge(conn, task_id, "verified_by", verification_id,
                                   event_id=event_id)
            conn.execute(
                """INSERT INTO task_verifications
                   (verification_id,task_id,status,summary,evidence_json,missing_json,
                    created_at,last_event_seq) VALUES (?,?,?,?,?,?,?,?)""",
                (verification_id, task_id, status, summary, canonical_json(evidence),
                 canonical_json(missing), utc_now(), seq))
            conn.execute(
                """UPDATE task_state SET verification_status=?,verification_json=?,
                   updated_at=?,last_event_seq=? WHERE task_id=?""",
                (status, canonical_json(result), utc_now(), seq, task_id))
            return self._append_progress(
                conn, event_id, task_id,
                self._progress_payload(
                    task_id, "verification", status,
                    ({
                        "passed": "Outcome verified",
                        "failed": "Verification failed",
                        "uncertain": "Outcome remains uncertain",
                        "user_confirmation_required": (
                            "User confirmation required"),
                    })[status],
                    summary[:240]),
            )

    def request_cancel(self, task_id: str, *, actor: str = "user") -> dict[str, Any]:
        state = self.get(task_id)
        if state is None:
            raise ValueError("task does not exist")
        if state["status"] in TERMINAL:
            return {"task_id": task_id, "status": state["status"],
                    "already_terminal": True}
        with self.graph.transaction() as conn:
            body = {"task_id": task_id, "requested": True}
            event_id, seq = self.graph.append_event(
                conn, "task.cancellation_requested", body,
                actor=actor, task_id=task_id)
            conn.execute(
                """UPDATE task_state SET cancellation_requested=1,updated_at=?,
                   last_event_seq=? WHERE task_id=?""",
                (utc_now(), seq, task_id))
            conn.execute(
                """UPDATE task_steps SET status='cancelled',last_error=?,
                   updated_at=?,last_event_seq=? WHERE task_id=?
                   AND status IN ('pending','waiting_approval')""",
                ("task cancellation requested", utc_now(), seq, task_id))
            conn.execute(
                """UPDATE task_step_batches SET status='cancelled',updated_at=?,
                   last_event_seq=? WHERE task_id=? AND NOT EXISTS
                   (SELECT 1 FROM task_steps s WHERE s.batch_id=task_step_batches.batch_id
                    AND s.status IN ('running','reconcile_required'))""",
                (utc_now(), seq, task_id))
            running = int(conn.execute(
                "SELECT COUNT(*) FROM task_steps WHERE task_id=? AND status='running'",
                (task_id,)).fetchone()[0])
            progress = self._append_progress(
                conn, event_id, task_id,
                self._progress_payload(task_id, "task", "cancelling",
                                       "Cancellation requested",
                                       "No additional action will start."))
        if running:
            return progress | {"status": "cancelling"}
        return self.transition(task_id, "cancelled", label="Task cancelled")

    def is_cancelled(self, task_id: str) -> bool:
        state = self.get(task_id)
        return bool(state and (state["status"] == "cancelled"
                               or state["cancellation_requested"]))

    def recover_interrupted(self) -> list[dict[str, Any]]:
        self.recover_inflight_steps(force=True)
        recovered: list[dict[str, Any]] = []
        with self.graph._connect() as conn:
            candidates = [tuple(row) for row in conn.execute(
                """SELECT t.task_id,EXISTS(
                         SELECT 1 FROM task_steps s
                         WHERE s.task_id=t.task_id
                           AND s.status='reconcile_required')
                   FROM task_state t
                   WHERE t.status IN ('running','verifying','replanning')"""
            ).fetchall()]
        for task_id, needs_reconciliation in candidates:
            with self.graph.transaction() as conn:
                row = conn.execute("SELECT status FROM task_state WHERE task_id=?",
                                   (task_id,)).fetchone()
                if row is None or row["status"] not in {
                    "running", "verifying", "replanning"
                }:
                    continue
                destination = (
                    "waiting_input" if needs_reconciliation else "recovering")
                event_type = (
                    "task.reconciliation_required" if needs_reconciliation
                    else "task.recovered")
                reason = (
                    "uncertain external action requires reconciliation"
                    if needs_reconciliation else "process restart")
                payload = {"task_id": task_id, "from": row["status"],
                           "to": destination, "reason": reason}
                event_id, seq = self.graph.append_event(
                    conn, event_type, payload, actor="supervisor",
                    task_id=task_id)
                conn.execute(
                    """UPDATE task_state SET status=?, lease_id=NULL,
                       lease_expires_at=NULL, updated_at=?, last_event_seq=?
                       WHERE task_id=?""",
                    (destination, utc_now(), seq, task_id))
                recovered.append(self._append_progress(
                    conn, event_id, task_id,
                    self._progress_payload(
                        task_id, "recovery", destination,
                        ("Outcome reconciliation required"
                         if needs_reconciliation else
                         "Recovering interrupted task"),
                        ("A consequential action was not replayed."
                         if needs_reconciliation else None)),
                ))
        return recovered

    def progress_since(self, seq: int = 0, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT seq, payload_json FROM progress_outbox WHERE seq > ?
                   ORDER BY seq LIMIT ?""", (seq, limit)
            ).fetchall()
        return [json.loads(row["payload_json"]) | {"seq": row["seq"]} for row in rows]

    def latest_progress_sequence(self) -> int:
        with self.graph._connect() as conn:
            return int(conn.execute(
                "SELECT COALESCE(MAX(seq),0) FROM progress_outbox").fetchone()[0])

    def action_history(self, task_id: str) -> list[dict[str, Any]]:
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT idempotency_key,step_id,action_id,tool_name,status,result_json,
                          effects_json,verification_json,risk,approval_status,
                          observation_id,created_at,updated_at
                   FROM action_receipts WHERE task_id=? ORDER BY created_at""",
                (task_id,)).fetchall()
        return [dict(row) | {
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "effects": json.loads(row["effects_json"] or "[]"),
            "verification": (json.loads(row["verification_json"])
                             if row["verification_json"] else None),
        } for row in rows]

    def publish(self, task_id: str, phase: str, state: str, label: str,
                detail: str | None = None, *, actor: str = "worker") -> dict[str, Any]:
        payload = self._progress_payload(task_id, phase, state, label, detail)
        with self.graph.transaction() as conn:
            event_id, _ = self.graph.append_event(
                conn, "task.progress", payload, actor=actor, task_id=task_id)
            return self._append_progress(conn, event_id, task_id, payload)

    def nonterminal(self) -> list[dict[str, Any]]:
        with self.graph._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM task_state
                   WHERE status NOT IN ('completed','failed','cancelled')
                   ORDER BY updated_at"""
            ).fetchall()
        return [dict(row) for row in rows]
