"""Typed contracts, plans, policy decisions, and deterministic verification."""

from __future__ import annotations

import json
import hashlib
import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .builtin_tools import (
    DESKTOP_TOOL_NAMES,
    PROCESS_TOOL_NAMES,
    RESOURCE_OVERRIDES,
    TOOL_CRITERIA,
    TOOL_POLICY_DATA,
)


class IntentType(StrEnum):
    CONVERSATION = "conversation"
    QUESTION = "question"
    ACTION = "action"
    CORRECTION = "correction"
    CANCELLATION = "cancellation"
    SCHEDULED = "scheduled"


class RiskClass(StrEnum):
    READ_ONLY = "read_only"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerificationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    USER_CONFIRMATION_REQUIRED = "user_confirmation_required"


class SuccessCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterion_id: str
    description: str
    verifier: str
    required: bool = True


class TaskContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    objective: str = Field(min_length=1, max_length=4000)
    intent_type: IntentType = IntentType.ACTION
    success_criteria: list[SuccessCriterion]
    required_tools: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    risk: RiskClass = RiskClass.LOW
    freshness_seconds: int | None = Field(default=None, ge=1)
    needs_user_confirmation: bool = False


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str
    description: str
    tool_name: str | None = None
    expected_observation: str
    verifier: str
    depends_on: list[str] = Field(default_factory=list)


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    steps: list[PlanStep]


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: VerificationStatus
    summary: str
    evidence: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    effects: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed: bool
    approval_required: bool
    risk: RiskClass
    permissions: list[str]
    reason: str


class ResourceClaim(BaseModel):
    """Admission inputs persisted with every executable step."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cpu_cores: float = Field(default=0.25, ge=0.0, le=256.0)
    ram_mib: int = Field(default=128, ge=0, le=1_048_576)
    vram_mib: int = Field(default=0, ge=0, le=1_048_576)
    accelerator: str = Field(default="none", pattern=r"^(?:none|cuda(?::[0-9]+)?)$")
    network: bool = False
    concurrency_slots: int = Field(default=1, ge=1, le=64)
    latency_class: str = Field(
        default="interactive",
        pattern=r"^(?:interactive|background|batch|control)$")

    @model_validator(mode="after")
    def validate_accelerator_capacity(self) -> "ResourceClaim":
        if self.vram_mib and self.accelerator == "none":
            raise ValueError("nonzero VRAM requires an explicit accelerator")
        return self


TOOL_POLICIES: dict[str, tuple[RiskClass, tuple[str, ...], bool]] = {
    name: (RiskClass(risk), permissions, always_approve)
    for name, (risk, permissions, always_approve) in TOOL_POLICY_DATA.items()
}

_PERMISSION_RISK = {
    "filesystem_read": RiskClass.MEDIUM,
    "network": RiskClass.HIGH,
    "browser": RiskClass.HIGH,
    "clipboard": RiskClass.HIGH,
    "filesystem_write": RiskClass.HIGH,
    "process": RiskClass.HIGH,
    "desktop": RiskClass.HIGH,
    "notifications": RiskClass.MEDIUM,
    "scheduling": RiskClass.MEDIUM,
}

_RESOURCE_OVERRIDES = RESOURCE_OVERRIDES


def resource_claim_for(tool_name: str, *,
                       permissions: list[str] | tuple[str, ...] = ()) -> ResourceClaim:
    values = dict(_RESOURCE_OVERRIDES.get(tool_name, {}))
    permission_set = set(permissions)
    if "network" in permission_set:
        values["network"] = True
    if ("process" in permission_set
            and values.get("latency_class") != "control"):
        values["cpu_cores"] = max(float(values.get("cpu_cores", 0.25)), 1.0)
        values["ram_mib"] = max(int(values.get("ram_mib", 128)), 512)
    return ResourceClaim(**values)


ACTION_WORDS = re.compile(
    r"\b(?:add|build|change|check|create|delete|edit|fetch|find|inspect|open|"
    r"read|remind|remove|restart|search|set|show|switch|update|upgrade|write)\b",
    re.IGNORECASE,
)


class IntentInterpreter:
    """Conservative deterministic routing around the model's tool proposal."""

    def interpret(self, text: str, proposed_tools: list[str] | None = None) -> IntentType:
        lowered = text.strip().lower()
        if re.search(r"\b(?:cancel|stop|never mind|nevermind)\b", lowered):
            return IntentType.CANCELLATION
        if re.search(r"\b(?:that transcript is wrong|i said|correct that)\b", lowered):
            return IntentType.CORRECTION
        if proposed_tools or ACTION_WORDS.search(text):
            return IntentType.ACTION
        if "?" in text or re.match(r"^(?:what|who|why|when|where|how|can)\b", lowered):
            return IntentType.QUESTION
        return IntentType.CONVERSATION


class ContractBuilder:
    _TOOL_CRITERIA = TOOL_CRITERIA

    def build(self, objective: str, tool_names: list[str], *,
              dynamic_permissions: dict[str, list[str]] | None = None
              ) -> TaskContract:
        policies = []
        for name in tool_names:
            declared = (dynamic_permissions or {}).get(name)
            if declared is None:
                policies.append(
                    TOOL_POLICIES.get(name, (RiskClass.LOW, (), False)))
                continue
            risk_order = list(RiskClass)
            risk = max(
                (_PERMISSION_RISK.get(item, RiskClass.MEDIUM)
                 for item in declared),
                key=risk_order.index, default=RiskClass.MEDIUM)
            policies.append((risk, tuple(sorted(set(declared))), True))
        risk_order = list(RiskClass)
        risk = max((item[0] for item in policies),
                   key=risk_order.index, default=RiskClass.LOW)
        permissions = sorted({permission for _, values, _ in policies
                              for permission in values})
        criteria = []
        for index, name in enumerate(dict.fromkeys(tool_names), 1):
            cid, description, verifier = self._TOOL_CRITERIA.get(
                name, (f"action_{index}", f"Complete {name} successfully",
                       "successful_receipt"))
            criteria.append(SuccessCriterion(
                criterion_id=cid, description=description, verifier=verifier))
        return TaskContract(
            objective=objective,
            intent_type=IntentType.ACTION,
            success_criteria=criteria,
            required_tools=list(dict.fromkeys(tool_names)),
            permissions=permissions,
            risk=risk,
            freshness_seconds=900 if "fetch_news" in tool_names else None,
            needs_user_confirmation=any(item[2] for item in policies),
        )


class Planner:
    def build(self, calls: list[dict[str, Any]], contract: TaskContract) -> TaskPlan:
        steps: list[PlanStep] = []
        for index, call in enumerate(calls, 1):
            name = str(call["name"])
            verifier = ContractBuilder._TOOL_CRITERIA.get(
                name, ("", "", "successful_receipt"))[2]
            steps.append(PlanStep(
                step_id=f"step_{index}", description=f"Run {name}", tool_name=name,
                expected_observation=f"A structured {name} receipt",
                verifier=verifier,
                depends_on=[f"step_{index - 1}"] if index > 1 else [],
            ))
        steps.append(PlanStep(
            step_id=f"step_{len(steps) + 1}",
            description="Verify the task contract",
            expected_observation="Independent verifier passes every required criterion",
            verifier="task_contract",
            depends_on=[steps[-1].step_id] if steps else [],
        ))
        return TaskPlan(steps=steps)


class PolicyEngine:
    def decide(self, tool_name: str, *, explicitly_requested: bool = True,
               dynamic_permissions: list[str] | None = None,
               executor_identity: str | None = None) -> PolicyDecision:
        if dynamic_permissions is not None:
            permissions = sorted(set(dynamic_permissions))
            risk_order = list(RiskClass)
            risk = max(
                (_PERMISSION_RISK.get(item, RiskClass.MEDIUM)
                 for item in permissions),
                key=risk_order.index, default=RiskClass.MEDIUM)
            identity = executor_identity or tool_name
            return PolicyDecision(
                allowed=True, approval_required=True, risk=risk,
                permissions=permissions,
                reason=("exact approval required for dynamic capability "
                        f"{identity} with permissions {permissions}"),
            )
        risk, permissions, always_approve = TOOL_POLICIES.get(
            tool_name, (RiskClass.MEDIUM, ("dynamic_capability",), True))
        approval = always_approve and not explicitly_requested
        return PolicyDecision(
            allowed=True, approval_required=approval, risk=risk,
            permissions=list(permissions),
            reason=("explicit user request authorizes this staged action"
                    if always_approve and explicitly_requested
                    else "approval required for a consequential action"
                    if approval else "allowed by supervised local policy"),
        )


class OutcomeVerifier:
    """Validate receipts independently of the model that selected the action."""

    _PROCESS_RECEIPT_TOOLS = PROCESS_TOOL_NAMES
    _DESKTOP_RECEIPT_TOOLS = DESKTOP_TOOL_NAMES

    def __init__(self, *, process_receipt_verifier: Callable[
            [str, Any, dict[str, Any] | None, str | None], bool] | None = None,
            desktop_receipt_verifier: Callable[
                [str, Any, dict[str, Any] | None, str | None], bool]
            | None = None):
        # Process receipts require a live, authoritative broker query.  Keeping
        # that dependency injected avoids a cognition -> process-broker import
        # cycle and, importantly, makes standalone/default verification fail
        # closed instead of accepting plausible-looking JSON.
        self._process_receipt_verifier = process_receipt_verifier
        self._desktop_receipt_verifier = desktop_receipt_verifier

    @staticmethod
    def _json_result(result: Any) -> Any:
        if not isinstance(result, str):
            return result
        try:
            return json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return result

    @staticmethod
    def _machine_path(value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            return None
        return str(Path(os.path.abspath(path)))

    @staticmethod
    def _sha256_bytes(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def verify_action(self, tool_name: str, result: Any, *, succeeded: bool,
                      args: dict[str, Any] | None = None,
                      idempotency_key: str | None = None) -> VerificationResult:
        value = self._json_result(result)
        if not succeeded or (isinstance(result, str) and result.startswith("error:")):
            return VerificationResult(
                status=VerificationStatus.FAILED,
                summary=f"{tool_name} reported failure", missing=["successful receipt"])

        evidence: list[str] = []
        effects: list[dict[str, Any]] = []
        valid = True
        if tool_name in self._PROCESS_RECEIPT_TOOLS:
            try:
                valid = bool(
                    self._process_receipt_verifier
                    and self._process_receipt_verifier(
                        tool_name, result, args, idempotency_key))
            except Exception:
                valid = False
            effects = ([{"kind": tool_name, "verified": True}]
                       if valid else [])
        elif tool_name in self._DESKTOP_RECEIPT_TOOLS:
            try:
                valid = bool(
                    self._desktop_receipt_verifier
                    and self._desktop_receipt_verifier(
                        tool_name, result, args, idempotency_key))
            except Exception:
                valid = False
            effects = ([{"kind": tool_name, "verified": True}]
                       if valid else [])
        elif tool_name == "fetch_news":
            headlines = value.get("headlines", []) if isinstance(value, dict) else []
            valid = bool(headlines) and all(item.get("title") and item.get("source")
                                            and item.get("url") for item in headlines)
            evidence = [str(item.get("url")) for item in headlines if item.get("url")]
            effects = [{"kind": "web_observation", "count": len(headlines)}]
        elif tool_name == "web_search":
            results = value.get("results", []) if isinstance(value, dict) else []
            valid = bool(results) and all(item.get("title") and item.get("url")
                                          for item in results)
            evidence = [str(item.get("url")) for item in results if item.get("url")]
            effects = [{"kind": "web_observation", "count": len(results)}]
        elif tool_name == "read_web":
            valid = (isinstance(value, dict) and bool(value.get("url"))
                     and bool(str(value.get("text", "")).strip()))
            evidence = [str(value.get("url"))] if isinstance(value, dict) else []
        elif tool_name == "search_skill_catalog":
            results = value.get("results", []) if isinstance(value, dict) else []
            valid = bool(results) and all(item.get("id") and item.get("url")
                                          for item in results)
            evidence = [str(item.get("url")) for item in results if item.get("url")]
            effects = [{"kind": "skill_discovery", "count": len(results)}]
        elif tool_name == "import_skill":
            valid = (isinstance(value, dict) and value.get("status") == "active"
                     and bool(value.get("version_id")) and bool(value.get("hash"))
                     and value.get("permissions_granted") == [])
            evidence = ([str(value.get("external_id"))]
                        if isinstance(value, dict) and value.get("external_id") else [])
            effects = ([{"kind": "skill_activation",
                         "version_id": value.get("version_id"),
                         "hash": value.get("hash")}]
                       if valid else [])
        elif tool_name == "write_file":
            valid = isinstance(result, str) and "after tests; deployment " in result
            effects = [{"kind": "filesystem_write", "verified": valid}]
        elif tool_name == "machine_grant_path":
            approved_target = self._machine_path((args or {}).get("path"))
            approved_permissions = sorted(set(
                (args or {}).get("permissions") or []))
            approved_expiry = (str(args.get("expires_at"))
                               if args and args.get("expires_at") else None)
            target = value.get("target") if isinstance(value, dict) else None
            valid = (isinstance(value, dict) and args is not None
                     and value.get("status") == "active"
                     and str(value.get("grant_id") or "").startswith("grant_")
                     and target == approved_target
                     and value.get("target_sha256") == (
                         self._sha256_bytes(target) if isinstance(target, str)
                         else None)
                     and value.get("permissions") == approved_permissions
                     and bool(value.get("allow_sensitive")) == bool(
                         args.get("allow_sensitive", False))
                     and value.get("expires_at") == approved_expiry)
            effects = ([{"kind": "operator_grant_created",
                         "grant_id": value.get("grant_id")}]
                       if valid else [])
        elif tool_name == "machine_list_grants":
            valid = (isinstance(value, list)
                     and all(isinstance(item, dict)
                             and bool(item.get("grant_id"))
                             and isinstance(item.get("permissions"), list)
                             for item in value))
            effects = ([{"kind": "operator_grants_observed",
                         "count": len(value)}] if valid else [])
        elif tool_name == "machine_revoke_grant":
            valid = (isinstance(value, dict) and args is not None
                     and value.get("grant_id") == args.get("grant_id")
                     and value.get("status") == "revoked")
            effects = ([{"kind": "operator_grant_revoked",
                         "grant_id": value.get("grant_id")}]
                       if valid else [])
        elif tool_name in {"machine_inspect_path", "machine_list_path",
                           "machine_read_text", "machine_read_document",
                           "machine_ocr_image", "machine_understand_image"}:
            approved_path = self._machine_path((args or {}).get("path"))
            common = (isinstance(value, dict) and args is not None
                      and value.get("status") == "ok"
                      and value.get("verified") is True
                      and str(value.get("grant_id") or "").startswith("grant_")
                      and value.get("path") == approved_path)
            if tool_name == "machine_inspect_path":
                valid = common and value.get("kind") in {
                    "file", "directory", "symlink", "special"}
            elif tool_name == "machine_list_path":
                valid = common and isinstance(value.get("entries"), list)
            elif tool_name == "machine_read_text":
                valid = (common and isinstance(value.get("text"), str)
                         and value.get("bytes") == len(
                             value.get("text").encode("utf-8"))
                         and value.get("sha256") == self._sha256_bytes(
                             value.get("text")))
            elif tool_name == "machine_read_document":
                text = value.get("text") if isinstance(value, dict) else None
                source_hash = (value.get("source_sha256")
                               if isinstance(value, dict) else None)
                valid = (
                    common and value.get("format") in {
                        "pdf", "docx", "odt", "epub", "pptx", "xlsx"}
                    and value.get("extractor") in {
                        "sandboxed-poppler", "bounded-archive-xml"}
                    and isinstance(text, str) and bool(text)
                    and isinstance(value.get("characters"), int)
                    and not isinstance(value.get("characters"), bool)
                    and value.get("characters") == len(text)
                    and value.get("text_sha256") == self._sha256_bytes(text)
                    and isinstance(value.get("source_bytes"), int)
                    and not isinstance(value.get("source_bytes"), bool)
                    and 1 <= value.get("source_bytes") <= 64 * 1024 * 1024
                    and isinstance(source_hash, str)
                    and re.fullmatch(r"[0-9a-f]{64}", source_hash) is not None
                    and isinstance(value.get("truncated"), bool)
                    and ("pages" not in value
                         or (value.get("format") == "pdf"
                             and isinstance(value.get("pages"), int)
                             and not isinstance(value.get("pages"), bool)
                             and value.get("pages") >= 1)))
            elif tool_name == "machine_ocr_image":
                text = value.get("text") if isinstance(value, dict) else None
                source_hash = (value.get("source_sha256")
                               if isinstance(value, dict) else None)
                width = value.get("width") if isinstance(value, dict) else None
                height = value.get("height") if isinstance(value, dict) else None
                valid = (
                    common and value.get("format") in {"png", "jpeg"}
                    and value.get("extractor") == "sandboxed-tesseract"
                    and value.get("language") == "eng"
                    and value.get("limitations") == "ocr_only"
                    and isinstance(width, int) and not isinstance(width, bool)
                    and isinstance(height, int) and not isinstance(height, bool)
                    and 1 <= width <= 20_000 and 1 <= height <= 20_000
                    and isinstance(value.get("pixels"), int)
                    and not isinstance(value.get("pixels"), bool)
                    and value.get("pixels") == width * height
                    and value.get("pixels") <= 40_000_000
                    and isinstance(text, str)
                    and isinstance(value.get("characters"), int)
                    and not isinstance(value.get("characters"), bool)
                    and value.get("characters") == len(text)
                    and value.get("text_sha256") == self._sha256_bytes(text)
                    and value.get("text_detected") is bool(text)
                    and isinstance(value.get("source_bytes"), int)
                    and not isinstance(value.get("source_bytes"), bool)
                    and 1 <= value.get("source_bytes") <= 32 * 1024 * 1024
                    and isinstance(source_hash, str)
                    and re.fullmatch(r"[0-9a-f]{64}", source_hash) is not None
                    and isinstance(value.get("truncated"), bool))
            else:
                answer = value.get("answer") if isinstance(value, dict) else None
                question = (args or {}).get("question")
                width = value.get("width") if isinstance(value, dict) else None
                height = value.get("height") if isinstance(value, dict) else None
                source_width = (value.get("source_width")
                                if isinstance(value, dict) else None)
                source_height = (value.get("source_height")
                                 if isinstance(value, dict) else None)
                valid = (
                    common and value.get("format") == "png"
                    and value.get("source_format") in {"png", "jpeg"}
                    and value.get("sanitizer") == "sandboxed-imagemagick"
                    and value.get("limitations")
                    == "single_image_question_answering"
                    and isinstance(question, str) and 1 <= len(question) <= 2_000
                    and value.get("question_sha256")
                    == self._sha256_bytes(question)
                    and isinstance(answer, str) and 1 <= len(answer) <= 4_000
                    and value.get("answer_characters") == len(answer)
                    and value.get("answer_sha256") == self._sha256_bytes(answer)
                    and isinstance(width, int) and not isinstance(width, bool)
                    and isinstance(height, int) and not isinstance(height, bool)
                    and isinstance(value.get("max_side"), int)
                    and not isinstance(value.get("max_side"), bool)
                    and 256 <= value.get("max_side") <= 4_096
                    and 1 <= width <= value.get("max_side")
                    and 1 <= height <= value.get("max_side")
                    and value.get("pixels") == width * height
                    and value.get("pixels") <= value.get("max_side") ** 2
                    and isinstance(source_width, int)
                    and not isinstance(source_width, bool)
                    and isinstance(source_height, int)
                    and not isinstance(source_height, bool)
                    and 1 <= source_width <= 20_000
                    and 1 <= source_height <= 20_000
                    and value.get("source_pixels")
                    == source_width * source_height
                    and value.get("source_pixels") <= 40_000_000
                    and isinstance(value.get("source_bytes"), int)
                    and not isinstance(value.get("source_bytes"), bool)
                    and 1 <= value.get("source_bytes") <= 16 * 1024 * 1024
                    and isinstance(value.get("image_bytes"), int)
                    and not isinstance(value.get("image_bytes"), bool)
                    and 1 <= value.get("image_bytes") <= 16 * 1024 * 1024
                    and all(re.fullmatch(r"[0-9a-f]{64}", str(
                        value.get(field) or "")) is not None for field in (
                            "source_sha256", "image_sha256",
                            "runtime_fingerprint"))
                    and isinstance(value.get("model"), str)
                    and 1 <= len(value.get("model")) <= 160)
            effects = ([{"kind": tool_name, "verified": True}]
                       if valid else [])
        elif tool_name == "machine_write_text":
            approved_path = self._machine_path((args or {}).get("path"))
            approved_content = (args or {}).get("content")
            valid = (isinstance(value, dict) and args is not None
                     and isinstance(approved_content, str)
                     and value.get("status") == "ok"
                     and value.get("verified") is True
                     and str(value.get("grant_id") or "").startswith("grant_")
                     and value.get("path") == approved_path
                     and value.get("bytes") == len(approved_content.encode("utf-8"))
                     and value.get("after_sha256") == self._sha256_bytes(
                         approved_content)
                     and bool(idempotency_key)
                     and value.get("rollback_operation_id") == idempotency_key)
            effects = ([{"kind": "machine_filesystem_write", "verified": True,
                         "operation_id": value.get("rollback_operation_id")}]
                       if valid else [])
        elif tool_name == "machine_rollback_write":
            expected_operation = (args or {}).get("operation_id")
            restored_hash = (value.get("restored_sha256")
                             if isinstance(value, dict) else None)
            restored_absence = (value.get("restored_absence")
                                if isinstance(value, dict) else None)
            valid = (isinstance(value, dict) and args is not None
                     and value.get("status") == "ok"
                     and value.get("verified") is True
                     and str(value.get("grant_id") or "").startswith("grant_")
                     and self._machine_path(value.get("path")) == value.get("path")
                     and bool(expected_operation)
                     and value.get("operation_id") == expected_operation
                     and ((restored_absence is True and restored_hash is None)
                          or (restored_absence is False
                              and bool(re.fullmatch(
                                  r"[0-9a-f]{64}",
                                  str(restored_hash or ""))))))
            effects = ([{"kind": "machine_filesystem_rollback",
                         "verified": True,
                         "operation_id": value.get("operation_id")}]
                       if valid else [])
        elif tool_name == "create_reminder":
            valid = isinstance(value, dict) and bool(value.get("reminder_id"))
            effects = [{"kind": "reminder_created",
                        "id": value.get("reminder_id")}] if valid else []
        elif tool_name == "cancel_reminder":
            valid = isinstance(value, dict) and value.get("status") == "cancelled"
            effects = [{"kind": "reminder_cancelled",
                        "id": value.get("reminder_id")}] if valid else []
        elif tool_name == "list_reminders":
            valid = (isinstance(value, list)
                     and all(isinstance(item, dict)
                             and bool(item.get("reminder_id"))
                             and bool(item.get("text"))
                             and bool(item.get("due_at"))
                             and item.get("status") in {
                                 "scheduled", "fired", "cancelled"}
                             for item in value))
            evidence = ([str(item["reminder_id"]) for item in value]
                        if valid else [])
            effects = ([{"kind": "reminders_observed", "count": len(value)}]
                       if valid else [])
        elif tool_name in {"browser_open", "browser_snapshot", "browser_click",
                           "browser_type"}:
            valid = isinstance(value, dict) and bool(value.get("url"))
            effects = [{"kind": "browser_action", "action": tool_name,
                        "url": value.get("url")}] if valid else []
        elif tool_name in {"clipboard_read", "clipboard_write", "desktop_notify",
                           "open_local"}:
            valid = isinstance(value, dict) and value.get("status") == "ok"
            effects = [{"kind": tool_name, "verified": valid}]
        elif tool_name == "set_voice":
            valid = isinstance(result, str) and result.startswith("activated voice ")
            effects = [{"kind": "voice_activation", "verified": valid}]
        elif tool_name == "upgrade_core":
            valid = (isinstance(value, dict)
                     and value.get("status") == "awaiting_review"
                     and bool(value.get("changed")))
            effects = [{"kind": "core_upgrade", "status": value.get("status")}] if valid else []
        else:
            valid = result not in (None, "", "[]", "{}", "(empty)")
            effects = [{"kind": "tool_result", "tool": tool_name}]

        return VerificationResult(
            status=(VerificationStatus.PASSED if valid else VerificationStatus.FAILED),
            summary=(f"{tool_name} receipt satisfies its verifier" if valid
                     else f"{tool_name} receipt lacks required evidence"),
            evidence=evidence,
            missing=[] if valid else ["required structured evidence"],
            effects=effects,
        )

    def verify_task(self, contract: TaskContract,
                    actions: list[dict[str, Any]]) -> VerificationResult:
        passed_tools = {action["tool_name"] for action in actions
                        if action["status"] == "succeeded"
                        and (action.get("verification") or {}).get("status") == "passed"}
        missing_tools = [name for name in contract.required_tools
                         if name not in passed_tools]
        failed = [action["tool_name"] for action in actions
                  if action["status"] == "failed"
                  or (action.get("verification") or {}).get("status") == "failed"]
        unknown = [action["tool_name"] for action in actions
                   if action["status"] == "outcome_unknown"
                   or (action.get("verification") or {}).get("status") in {
                       "uncertain", "user_confirmation_required"}]
        evidence = [item for action in actions
                    for item in (action.get("verification") or {}).get("evidence", [])]
        if failed:
            missing = [*(f"verified receipt for {name}" for name in missing_tools),
                       *(f"successful outcome for {name}" for name in failed)]
            return VerificationResult(
                status=VerificationStatus.FAILED,
                summary="The task contract is not satisfied",
                evidence=evidence, missing=list(dict.fromkeys(missing)))
        if unknown:
            missing = [
                *(f"authoritative outcome for {name}" for name in unknown),
                *(f"verified receipt for {name}" for name in missing_tools),
            ]
            return VerificationResult(
                status=VerificationStatus.UNCERTAIN,
                summary=("The task contains an external action whose outcome "
                         "has not been authoritatively reconciled"),
                evidence=evidence, missing=list(dict.fromkeys(missing)))
        if missing_tools:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                summary="The task contract is not satisfied",
                evidence=evidence,
                missing=[f"verified receipt for {name}"
                         for name in missing_tools])
        # Consequential actions cannot acquire a receipt until PolicyEngine has
        # produced an approval decision. Completion therefore verifies the receipt;
        # it never asks for a second approval after the action already ran.
        return VerificationResult(
            status=VerificationStatus.PASSED,
            summary="Every required action has an independently verified receipt",
            evidence=evidence,
        )
