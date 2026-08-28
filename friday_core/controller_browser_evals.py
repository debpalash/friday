"""Signed-controller and managed-browser workflow qualification."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .controller_auth import ControllerAuthError, ControllerAuthService
from .feedback import ApprovalService
from .graph import GraphStore, utc_now
from .operator import WebOperator
from .tasks import TaskService


MAX_CONTROLLER_BROWSER_SUITE_BYTES = 64_000


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _der_signature_to_raw(value: bytes) -> bytes:
    if len(value) < 8 or value[0] != 0x30:
        raise RuntimeError("controller signature is malformed")
    offset = 2
    if value[1] & 0x80:
        count = value[1] & 0x7F
        offset = 2 + count
    integers: list[bytes] = []
    for _ in range(2):
        if offset + 2 > len(value) or value[offset] != 0x02:
            raise RuntimeError("controller signature is malformed")
        length = value[offset + 1]
        offset += 2
        integer = value[offset:offset + length]
        offset += length
        integer = integer.lstrip(b"\x00")
        if len(integer) > 32:
            raise RuntimeError("controller signature is malformed")
        integers.append(integer.rjust(32, b"\x00"))
    if offset != len(value):
        raise RuntimeError("controller signature is malformed")
    return b"".join(integers)


class _ControllerKey:
    def __init__(self, root: Path):
        self.path = root / "controller-private.pem"
        subprocess.run(
            ["/usr/bin/openssl", "genpkey", "-algorithm", "EC", "-pkeyopt",
             "ec_paramgen_curve:P-256", "-out", str(self.path)],
            check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=5,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        os.chmod(self.path, 0o600)
        public_der = subprocess.run(
            ["/usr/bin/openssl", "pkey", "-in", str(self.path), "-pubout",
             "-outform", "DER"],
            check=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=5,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        ).stdout
        point = public_der[-65:]
        if len(point) != 65 or point[0] != 4:
            raise RuntimeError("controller public key is malformed")
        self.jwk = {
            "kty": "EC", "crv": "P-256",
            "x": _b64url(point[1:33]), "y": _b64url(point[33:]),
        }

    def sign(self, payload: str) -> str:
        signature = subprocess.run(
            ["/usr/bin/openssl", "dgst", "-sha256", "-sign", str(self.path)],
            input=payload.encode("utf-8"), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        ).stdout
        return _b64url(_der_signature_to_raw(signature))


class _FixtureLocator:
    def __init__(self, page: "_FixturePage", selector: str):
        self.page = page
        self.selector = selector
        self.first = self

    def fill(self, text: str, *, timeout: int) -> None:
        if timeout != 10_000:
            raise RuntimeError("browser fill timeout changed")
        self.page.fills.append((self.selector, text))

    def press(self, key: str) -> None:
        self.page.presses.append(key)


class _FixturePage:
    def __init__(self, url: str):
        self.url = url
        self.fills: list[tuple[str, str]] = []
        self.presses: list[str] = []
        self.waits: list[int] = []

    def locator(self, selector: str) -> _FixtureLocator:
        return _FixtureLocator(self, selector)

    def wait_for_timeout(self, milliseconds: int) -> None:
        self.waits.append(milliseconds)

    @staticmethod
    def title() -> str:
        return "Controller browser evaluation"


class _FixtureContext:
    def __init__(self, page: _FixturePage):
        self.pages = [page]


class _FixtureBrowser:
    def __init__(self, page: _FixturePage):
        self.contexts = [_FixtureContext(page)]


class _ManagedFixtureOperator(WebOperator):
    """Exercise WebOperator operations behind an exact managed-runtime gate."""

    def __init__(self, profile: Path, browser: _FixtureBrowser):
        super().__init__(profile)
        self.browser = browser

    def _controlled(self, operation):
        self._verify_managed_runtime()
        result = operation(self.browser)
        self._verify_managed_runtime()
        return result


class ControllerBrowserEvalRunner:
    def __init__(self, graph: GraphStore):
        self.graph = graph

    @staticmethod
    def _load_suite(path: str | Path) -> tuple[dict[str, Any], str]:
        try:
            descriptor = os.open(
                Path(path), os.O_RDONLY | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise ValueError("controller-browser suite is unavailable") from exc
        try:
            metadata = os.fstat(descriptor)
            if (not stat.S_ISREG(metadata.st_mode)
                    or not 2 <= metadata.st_size
                    <= MAX_CONTROLLER_BROWSER_SUITE_BYTES):
                raise ValueError(
                    "controller-browser suite must be a bounded regular file")
            encoded = os.read(
                descriptor, MAX_CONTROLLER_BROWSER_SUITE_BYTES + 1)
            if len(encoded) != metadata.st_size:
                raise ValueError("controller-browser suite changed while read")
        finally:
            os.close(descriptor)
        try:
            suite = json.loads(
                encoded.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite value: {value}")),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("controller-browser suite is invalid JSON") from exc
        expected = {
            "name", "version", "origin", "transport_binding_sha256",
            "page_url", "selector", "input", "controller_label",
        }
        if (not isinstance(suite, dict) or set(suite) != expected
                or suite.get("name") != "friday-controller-browser"
                or suite.get("version") != 1
                or not re.fullmatch(r"[0-9a-f]{64}", str(
                    suite.get("transport_binding_sha256", "")))):
            raise ValueError("controller-browser suite metadata is invalid")
        for field, maximum in (
                ("origin", 200), ("page_url", 500), ("selector", 200),
                ("input", 1_000), ("controller_label", 80)):
            value = suite[field]
            if (not isinstance(value, str) or not 1 <= len(value) <= maximum
                    or any(ord(character) < 32 for character in value)):
                raise ValueError(f"controller-browser {field} is invalid")
        if not suite["origin"].startswith("https://"):
            raise ValueError("controller-browser origin is invalid")
        if not suite["page_url"].startswith("https://"):
            raise ValueError("controller-browser page URL is invalid")
        return suite, hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _pair(
        auth: ControllerAuthService,
        key: _ControllerKey,
        suite: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        pairing = auth.create_pairing(suite["transport_binding_sha256"])
        prepared = auth.prepare_pairing(
            pairing["pairing_token"], suite["controller_label"], key.jwk,
            origin=suite["origin"],
            transport_binding_sha256=suite["transport_binding_sha256"])
        paired = auth.complete_pairing(
            pairing["pairing_token"], suite["controller_label"], key.jwk,
            key.sign(prepared["proof_payload"]), origin=suite["origin"],
            transport_binding_sha256=suite["transport_binding_sha256"])
        return pairing, paired

    @staticmethod
    def _task(
        tasks: TaskService,
        principal,
        *,
        objective: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any]]:
        task_id, _ = tasks.create(
            objective, {"version": 0, "evidence": "signed exact approval"},
            session_id="controller-browser-session",
            turn_id="controller-browser-turn",
            controller_principal=principal)
        batch_id, steps = tasks.stage_step_batch(
            task_id, [{
                "tool_call_id": f"call-{tool_name}",
                "tool_name": tool_name,
                "args": args,
                "risk": "high",
                "approval_status": "pending",
                "idempotency_class": "non_repeatable",
                "recovery_policy": "reconcile",
                "verifier": "browser_receipt",
            }], round_index=0,
            context={"session_id": "controller-browser-session",
                     "turn_id": "controller-browser-turn"})
        return task_id, batch_id, steps[0]

    def _run_workflow(
        self,
        suite: dict[str, Any],
        root: Path,
        graph: GraphStore,
    ) -> dict[str, Any]:
        key = _ControllerKey(root)
        state = root / "auth-state"
        auth = ControllerAuthService(graph, state)
        pairing, paired = self._pair(auth, key, suite)
        first_principal = auth.authenticate_session(
            paired["session_token"], origin=suite["origin"],
            transport_binding_sha256=suite["transport_binding_sha256"])

        # Reconstruct the service from disk and prove returning-controller
        # possession with a fresh signed challenge.
        auth = ControllerAuthService(graph, state)
        challenge = auth.create_session_challenge(
            paired["controller_id"], origin=suite["origin"],
            transport_binding_sha256=suite["transport_binding_sha256"])
        reconnected = auth.complete_session(
            challenge["challenge_id"], challenge["challenge"],
            challenge["proof_payload"], key.sign(challenge["proof_payload"]))
        principal = auth.authenticate_session(
            reconnected["session_token"], origin=suite["origin"],
            transport_binding_sha256=suite["transport_binding_sha256"])

        tasks = TaskService(
            graph, controller_auth=auth, require_controller_authority=True)
        approvals = ApprovalService(
            graph, auth, require_controller_decisions=True)

        rejected_args = {
            "selector": "#cancel", "page_url": suite["page_url"]}
        rejected_task, rejected_batch, rejected_step = self._task(
            tasks, principal, objective="Reject one exact browser action",
            tool_name="browser_click", args=rejected_args)
        rejected_request = approvals.request(
            rejected_task, "browser_click", rejected_args,
            "reject exact browser action", step_id=rejected_step["step_id"],
            controller_principal=principal)
        rejection_proof = approvals.prepare_decision(
            rejected_request["approval_id"], False, principal)
        rejected = approvals.decide(
            rejected_request["approval_id"], False,
            controller_principal=principal,
            proof_payload=rejection_proof["proof_payload"],
            signature_b64url=key.sign(rejection_proof["proof_payload"]))

        exact_args = {
            "selector": suite["selector"], "text": suite["input"],
            "page_url": suite["page_url"], "submit": True,
        }
        exact_task, exact_batch, exact_step = self._task(
            tasks, principal, objective="Execute one exact browser action",
            tool_name="browser_type", args=exact_args)
        exact_request = approvals.request(
            exact_task, "browser_type", exact_args,
            "approve exact browser input", step_id=exact_step["step_id"],
            controller_principal=principal)
        exact_proof = approvals.prepare_decision(
            exact_request["approval_id"], True, principal)
        approved = approvals.decide(
            exact_request["approval_id"], True,
            controller_principal=principal,
            proof_payload=exact_proof["proof_payload"],
            signature_b64url=key.sign(exact_proof["proof_payload"]))

        claim = tasks.claim_next_step(exact_batch, "controller-browser-worker")
        if claim is None:
            raise RuntimeError("approved browser action was not dispatched")
        runtime_identity = hashlib.sha256(
            b"friday-managed-browser-evaluation-runtime").hexdigest()
        active_runtime_identity = runtime_identity
        runtime_checks = 0

        def verify_runtime() -> bool:
            nonlocal runtime_checks
            runtime_checks += 1
            return active_runtime_identity == runtime_identity

        page = _FixturePage(suite["page_url"])
        operator = _ManagedFixtureOperator(
            root / "browser-profile", _FixtureBrowser(page))
        operator.require_managed_runtime(verify_runtime)
        receipt = operator.type(**claim.args)
        verified = bool(
            receipt.get("submitted") is True
            and receipt.get("characters_typed") == len(suite["input"])
            and page.fills == [(suite["selector"], suite["input"])]
            and page.presses == ["Enter"]
            and runtime_checks == 2)
        tasks.finish_step(
            claim, receipt, succeeded=verified,
            verification={
                "status": "passed" if verified else "failed",
                "summary": "managed browser receipt verified",
                "evidence": [
                    hashlib.sha256(suite["input"].encode()).hexdigest(),
                    runtime_identity,
                ],
                "missing": [] if verified else ["browser postcondition"],
                "effects": [{"kind": "browser_input", "verified": verified}],
            })
        replay = tasks.claim_next_step(exact_batch, "controller-browser-worker")

        with graph._connect() as connection:
            rejection_effects = int(connection.execute(
                "SELECT COUNT(*) FROM controller_effect_uses WHERE task_id=?",
                (rejected_task,)).fetchone()[0])
            exact_effects = int(connection.execute(
                "SELECT COUNT(*) FROM controller_effect_uses WHERE task_id=?",
                (exact_task,)).fetchone()[0])
            dump = "\n".join(connection.iterdump())

        auth.revoke_controller(principal, principal.controller_id)
        revoked_session = False
        try:
            auth.authenticate_session(
                reconnected["session_token"], origin=suite["origin"],
                transport_binding_sha256=suite["transport_binding_sha256"])
        except ControllerAuthError:
            revoked_session = True
        inventory = auth.list_controllers()
        checks = {
            "paired": first_principal.controller_id == paired["controller_id"],
            "reconnected": (
                principal.controller_id == first_principal.controller_id
                and principal.session_id != first_principal.session_id),
            "rejection_recorded": (
                rejected["status"] == "denied"
                and tasks.step_batch(rejected_batch)["status"] == "cancelled"
                and rejection_effects == 0),
            "exact_approval_consumed_once": (
                approved["status"] == "approved" and exact_effects == 1
                and replay is None
                and not approvals.is_approved(
                    exact_task, "browser_type", exact_args)),
            "managed_browser_verified": (
                verified and runtime_checks == 2
                and tasks.step_batch(exact_batch)["status"] == "succeeded"),
            "controller_revoked": (
                revoked_session and len(inventory) == 1
                and inventory[0]["status"] == "revoked"
                and inventory[0]["active_sessions"] == 0),
            "private_input_absent": suite["input"] not in dump,
            "bearer_secrets_absent": all(
                secret not in dump for secret in (
                    pairing["pairing_token"].split(".")[2],
                    pairing["pairing_token"].split(".")[3],
                    paired["session_token"].split(".")[2],
                    reconnected["session_token"].split(".")[2],
                )),
        }
        return {
            "checks": checks,
            "pairing": {"controllers": 1, "sessions_issued": 2},
            "approval": {
                "rejected_actions_executed": rejection_effects,
                "approved_effect_uses": exact_effects,
                "approval_reusable": replay is not None,
            },
            "browser": {
                "transport": "deterministic_cdp_fixture",
                "managed_runtime_checks": runtime_checks,
                "mutations": len(page.fills),
                "submitted": bool(page.presses),
                "input_sha256": hashlib.sha256(
                    suite["input"].encode()).hexdigest(),
            },
            "revocation": {
                "controller_status": inventory[0]["status"],
                "active_sessions": inventory[0]["active_sessions"],
            },
            "passed": all(checks.values()),
        }

    def run(self, suite_path: str | Path) -> dict[str, Any]:
        suite, suite_sha256 = self._load_suite(suite_path)
        root_path: Path | None = None
        with tempfile.TemporaryDirectory(
                prefix="friday-controller-browser-") as value:
            root_path = Path(value)
            os.chmod(root_path, 0o700)
            result = self._run_workflow(
                suite, root_path, GraphStore(root_path / "workflow.db"))
        cleanup_verified = bool(root_path is not None and not root_path.exists())
        checks = dict(result["checks"])
        checks["fixture_cleanup"] = cleanup_verified
        body = {
            "suite": suite["name"],
            "version": suite["version"],
            "suite_sha256": suite_sha256,
            "pairing": result["pairing"],
            "approval": result["approval"],
            "browser": result["browser"],
            "revocation": result["revocation"],
            "checks": checks,
            "passed": result["passed"] and cleanup_verified,
            "privacy": {
                "fixture_content_persisted": False,
                "controller_private_key_persisted": False,
                "bearer_secrets_persisted": False,
                "cleanup_verified": cleanup_verified,
            },
            "ran_at": utc_now(),
        }
        run_id = self.graph.record_node(
            "controller_browser_evaluation_run", body,
            actor="controller_browser_eval_runner",
            event_type="evaluation.controller_browser_completed")
        return {"evaluation_run_id": run_id, **body}
