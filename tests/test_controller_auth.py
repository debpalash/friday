import base64
import json
import sqlite3
import stat
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from friday_core.controller_auth import (
    ControllerAuthError,
    ControllerAuthService,
    ControllerPrincipal,
    normalize_https_origin,
    normalize_public_jwk,
    verify_p256_signature,
)
from friday_core.feedback import ApprovalService
from friday_core.db_migrations import (
    LATEST_SCHEMA_VERSION,
    MIGRATIONS,
    apply_schema_migrations,
)
from friday_core.graph import GraphStore
from friday_core.tasks import TaskService


OPENSSL = "/usr/bin/openssl"
ORIGIN = "https://192.168.1.158:8500"
BINDING = "a" * 64


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _der_length(value: bytes, offset: int) -> tuple[int, int]:
    first = value[offset]
    if first < 0x80:
        return first, offset + 1
    width = first & 0x7F
    if width < 1 or width > 2 or offset + 1 + width > len(value):
        raise ValueError("invalid DER length")
    start = offset + 1
    return int.from_bytes(value[start:start + width], "big"), start + width


def _der_signature_to_raw(value: bytes) -> bytes:
    if not value or value[0] != 0x30:
        raise ValueError("invalid DER signature")
    sequence_length, offset = _der_length(value, 1)
    if offset + sequence_length != len(value):
        raise ValueError("invalid DER signature length")
    integers = []
    for _ in range(2):
        if offset >= len(value) or value[offset] != 0x02:
            raise ValueError("invalid DER signature integer")
        integer_length, offset = _der_length(value, offset + 1)
        integer = value[offset:offset + integer_length]
        offset += integer_length
        if not integer or (integer[0] == 0 and len(integer) > 1):
            integer = integer.lstrip(b"\x00")
        if len(integer) > 32:
            raise ValueError("invalid DER signature integer width")
        integers.append(integer.rjust(32, b"\x00"))
    if offset != len(value):
        raise ValueError("trailing DER signature data")
    return b"".join(integers)


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 24, 8, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


class P256TestKey:
    def __init__(self, root: Path, name: str) -> None:
        self.path = root / f"{name}.pem"
        subprocess.run(
            [OPENSSL, "genpkey", "-algorithm", "EC", "-pkeyopt",
             "ec_paramgen_curve:P-256", "-out", str(self.path)],
            check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=5,
        )
        public_der = subprocess.run(
            [OPENSSL, "pkey", "-in", str(self.path), "-pubout",
             "-outform", "DER"],
            check=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=5,
        ).stdout
        point = public_der[-65:]
        if len(point) != 65 or point[0] != 4:
            raise RuntimeError("unexpected P-256 public key encoding")
        self.jwk = {
            "kty": "EC", "crv": "P-256",
            "x": _b64url(point[1:33]), "y": _b64url(point[33:]),
        }

    def sign(self, payload: str) -> str:
        signature_der = subprocess.run(
            [OPENSSL, "dgst", "-sha256", "-sign", str(self.path)],
            input=payload.encode("utf-8"), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5,
        ).stdout
        return _b64url(_der_signature_to_raw(signature_der))


class ControllerAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph = GraphStore(self.root / "friday.db")
        self.clock = MutableClock()
        self.service = ControllerAuthService(
            self.graph, self.root / "state", key_provider=lambda: b"k" * 32,
            clock=self.clock, pairing_ttl_seconds=300,
            challenge_ttl_seconds=60, idle_session_ttl_seconds=60,
            absolute_session_ttl_seconds=120,
        )
        self.key = P256TestKey(self.root, "controller")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _pair(
        self, *, service: ControllerAuthService | None = None,
        key: P256TestKey | None = None, label: str = "Friday controller",
    ) -> tuple[dict, dict]:
        service = service or self.service
        key = key or self.key
        pairing = service.create_pairing(BINDING)
        prepared = service.prepare_pairing(
            pairing["pairing_token"], label, key.jwk, origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        result = service.complete_pairing(
            pairing["pairing_token"], label, key.jwk,
            key.sign(prepared["proof_payload"]), origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        return pairing, result

    def _new_session(
        self, controller_id: str, key: P256TestKey | None = None,
    ) -> dict:
        key = key or self.key
        challenge = self.service.create_session_challenge(
            controller_id, origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        return self.service.complete_session(
            challenge["challenge_id"], challenge["challenge"],
            challenge["proof_payload"], key.sign(challenge["proof_payload"]),
        )

    def _authorized_task(
        self, paired: dict, *, approval_status: str, tool_name: str = "write_file",
        args: dict | None = None, idempotency_class: str = "non_repeatable",
    ) -> tuple[TaskService, str, str, list[dict], ControllerPrincipal]:
        principal = self.service.authenticate_session(
            paired["session_token"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        tasks = TaskService(
            self.graph, controller_auth=self.service,
            require_controller_authority=True,
        )
        task_id, _ = tasks.create(
            "Execute only with exact controller authority",
            {"version": 0, "evidence": "controller authority chain"},
            session_id="graph_session_controller", turn_id="turn_controller",
            controller_principal=principal,
        )
        exact_args = args or {"path": "authorized.txt", "content": "exact"}
        batch_id, steps = tasks.stage_step_batch(
            task_id,
            [{
                "tool_call_id": "controller-call",
                "tool_name": tool_name,
                "args": exact_args,
                "risk": "high" if approval_status == "pending" else "read_only",
                "approval_status": approval_status,
                "idempotency_class": idempotency_class,
                "recovery_policy": (
                    "retry" if idempotency_class in {"read_only", "idempotent"}
                    else "reconcile"),
            }],
            round_index=0,
            context={"session_id": "graph_session_controller",
                     "turn_id": "turn_controller"},
        )
        return tasks, task_id, batch_id, steps, principal

    def test_pair_authenticate_issue_new_session_and_restart(self) -> None:
        state = self.root / "durable-state"
        first = ControllerAuthService(
            self.graph, state, clock=self.clock,
            idle_session_ttl_seconds=60, absolute_session_ttl_seconds=120,
        )
        pairing = first.create_pairing(BINDING)
        prepared = first.prepare_pairing(
            pairing["pairing_token"], "Laptop", self.key.jwk,
            origin=ORIGIN, transport_binding_sha256=BINDING,
        )
        paired = first.complete_pairing(
            pairing["pairing_token"], "Laptop", self.key.jwk,
            self.key.sign(prepared["proof_payload"]), origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        restarted = ControllerAuthService(
            self.graph, state, clock=self.clock,
            idle_session_ttl_seconds=60, absolute_session_ttl_seconds=120,
        )
        principal = restarted.authenticate_session(
            paired["session_token"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        self.assertEqual(principal.controller_id, paired["controller_id"])

        challenge = restarted.create_session_challenge(
            principal.controller_id, origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        issued = restarted.complete_session(
            challenge["challenge_id"], challenge["challenge"],
            challenge["proof_payload"],
            self.key.sign(challenge["proof_payload"]),
        )
        self.assertNotEqual(issued["session_id"], principal.session_id)
        restarted.authenticate_session(
            issued["session_token"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )

    def test_raw_pairing_and_session_secrets_never_reach_sqlite(self) -> None:
        pairing, paired = self._pair()
        secrets_to_find = [
            pairing["pairing_token"].split(".")[2],
            pairing["pairing_token"].split(".")[3],
            paired["session_token"].split(".")[2],
        ]
        with self.graph._connect() as conn:
            dump = "\n".join(conn.iterdump())
        for secret in secrets_to_find:
            self.assertNotIn(secret, dump)

    def test_bad_pairing_code_consumes_attempts_then_cancels(self) -> None:
        pairing = self.service.create_pairing(BINDING)
        parts = pairing["pairing_token"].split(".")
        parts[2] = "z" * len(parts[2])
        invalid = ".".join(parts)
        for _ in range(5):
            with self.assertRaises(ControllerAuthError):
                self.service.prepare_pairing(
                    invalid, "Laptop", self.key.jwk, origin=ORIGIN,
                    transport_binding_sha256=BINDING,
                )
        with self.graph._connect() as conn:
            row = conn.execute(
                "SELECT status,attempts_remaining FROM controller_pairings"
            ).fetchone()
        self.assertEqual(tuple(row), ("cancelled", 0))

    def test_expired_pairing_is_durable_without_burning_attempt(self) -> None:
        pairing = self.service.create_pairing(BINDING)
        self.clock.advance(300)
        with self.assertRaises(ControllerAuthError):
            self.service.prepare_pairing(
                pairing["pairing_token"], "Laptop", self.key.jwk,
                origin=ORIGIN, transport_binding_sha256=BINDING,
            )
        with self.graph._connect() as conn:
            row = conn.execute(
                "SELECT status,attempts_remaining FROM controller_pairings"
            ).fetchone()
        self.assertEqual(tuple(row), ("expired", 5))

    def test_wrong_signature_and_binding_fail_closed(self) -> None:
        wrong_key = P256TestKey(self.root, "wrong")
        pairing = self.service.create_pairing(BINDING)
        prepared = self.service.prepare_pairing(
            pairing["pairing_token"], "Laptop", self.key.jwk,
            origin=ORIGIN, transport_binding_sha256=BINDING,
        )
        with self.assertRaisesRegex(
                ControllerAuthError, "controller_proof_invalid"):
            self.service.complete_pairing(
                pairing["pairing_token"], "Laptop", self.key.jwk,
                wrong_key.sign(prepared["proof_payload"]), origin=ORIGIN,
                transport_binding_sha256=BINDING,
            )
        with self.assertRaises(ControllerAuthError):
            self.service.prepare_pairing(
                pairing["pairing_token"], "Laptop", self.key.jwk,
                origin=ORIGIN, transport_binding_sha256="b" * 64,
            )
        with self.graph._connect() as conn:
            attempts = conn.execute(
                "SELECT attempts_remaining FROM controller_pairings"
            ).fetchone()[0]
        self.assertEqual(attempts, 3)

    def test_pairing_is_single_use_under_concurrent_completion(self) -> None:
        pairing = self.service.create_pairing(BINDING)
        prepared = self.service.prepare_pairing(
            pairing["pairing_token"], "Laptop", self.key.jwk,
            origin=ORIGIN, transport_binding_sha256=BINDING,
        )
        signature = self.key.sign(prepared["proof_payload"])

        def complete() -> bool:
            try:
                self.service.complete_pairing(
                    pairing["pairing_token"], "Laptop", self.key.jwk,
                    signature, origin=ORIGIN,
                    transport_binding_sha256=BINDING,
                )
            except ControllerAuthError:
                return False
            return True

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _: complete(), range(2)))
        self.assertEqual(sorted(outcomes), [False, True])
        with self.graph._connect() as conn:
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM controller_identities").fetchone()[0],
                1,
            )
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM controller_sessions").fetchone()[0],
                1,
            )

    def test_session_proof_rejects_non_object_noncanonical_and_wrong_signature(self):
        _, paired = self._pair()
        wrong_key = P256TestKey(self.root, "session-wrong")
        challenge = self.service.create_session_challenge(
            paired["controller_id"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        for malformed in ("[]", "{ \"schema_version\": 1 }"):
            with self.assertRaisesRegex(
                    ControllerAuthError, "controller_proof_invalid"):
                self.service.complete_session(
                    challenge["challenge_id"], challenge["challenge"],
                    malformed, "x" * 86,
                )
        with self.assertRaisesRegex(
                ControllerAuthError, "controller_proof_invalid"):
            self.service.complete_session(
                challenge["challenge_id"], challenge["challenge"],
                challenge["proof_payload"],
                wrong_key.sign(challenge["proof_payload"]),
            )
        with self.graph._connect() as conn:
            attempts = conn.execute(
                "SELECT attempts_remaining FROM controller_session_challenges "
                "WHERE challenge_id=?", (challenge["challenge_id"],),
            ).fetchone()[0]
        self.assertEqual(attempts, 2)

    def test_expired_session_challenge_is_durable_without_burning_attempt(self):
        _, paired = self._pair()
        challenge = self.service.create_session_challenge(
            paired["controller_id"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        signature = self.key.sign(challenge["proof_payload"])
        self.clock.advance(60)
        with self.assertRaises(ControllerAuthError):
            self.service.complete_session(
                challenge["challenge_id"], challenge["challenge"],
                challenge["proof_payload"], signature,
            )
        with self.graph._connect() as conn:
            row = conn.execute(
                "SELECT status,attempts_remaining "
                "FROM controller_session_challenges WHERE challenge_id=?",
                (challenge["challenge_id"],),
            ).fetchone()
        self.assertEqual(tuple(row), ("expired", 5))

    def test_session_token_origin_and_transport_are_exactly_bound(self) -> None:
        _, paired = self._pair()
        with self.assertRaises(ControllerAuthError):
            self.service.authenticate_session(
                paired["session_token"] + "x", origin=ORIGIN,
                transport_binding_sha256=BINDING,
            )
        with self.assertRaises(ControllerAuthError):
            self.service.authenticate_session(
                paired["session_token"], origin="https://localhost:8500",
                transport_binding_sha256=BINDING,
            )
        with self.assertRaises(ControllerAuthError):
            self.service.authenticate_session(
                paired["session_token"], origin=ORIGIN,
                transport_binding_sha256="b" * 64,
            )
        self.service.authenticate_session(
            paired["session_token"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )

    def test_idle_refresh_is_sliding_but_absolute_expiry_is_hard(self) -> None:
        _, paired = self._pair()
        self.clock.advance(30)
        principal = self.service.authenticate_session(
            paired["session_token"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        self.assertEqual(
            principal.idle_expires_at, "2026-08-24T08:01:30.000000Z")
        self.clock.advance(59)
        principal = self.service.authenticate_session(
            paired["session_token"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        self.assertEqual(
            principal.idle_expires_at, "2026-08-24T08:02:00.000000Z")
        self.clock.advance(31)
        with self.assertRaises(ControllerAuthError):
            self.service.authenticate_session(
                paired["session_token"], origin=ORIGIN,
                transport_binding_sha256=BINDING,
            )
        with self.graph._connect() as conn:
            status = conn.execute(
                "SELECT status FROM controller_sessions WHERE session_id=?",
                (paired["session_id"],),
            ).fetchone()[0]
        self.assertEqual(status, "expired")

    def test_revocation_fences_session_and_controller_atomically(self) -> None:
        _, paired = self._pair()
        second = self._new_session(paired["controller_id"])
        principal = self.service.authenticate_session(
            paired["session_token"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        self.service.revoke_session(principal, second["session_id"])
        with self.assertRaises(ControllerAuthError):
            self.service.authenticate_session(
                second["session_token"], origin=ORIGIN,
                transport_binding_sha256=BINDING,
            )
        self.service.revoke_controller(principal, principal.controller_id)
        with self.assertRaises(ControllerAuthError):
            self.service.revalidate_principal(principal)
        with self.graph._connect() as conn:
            controller = conn.execute(
                "SELECT status,auth_epoch FROM controller_identities"
            ).fetchone()
            active_sessions = conn.execute(
                "SELECT COUNT(*) FROM controller_sessions WHERE status='active'"
            ).fetchone()[0]
        self.assertEqual(tuple(controller), ("revoked", 2))
        self.assertEqual(active_sessions, 0)

    def test_sqlite_rejects_identity_and_terminal_evidence_tampering(self):
        pairing, paired = self._pair()
        challenge = self.service.create_session_challenge(
            paired["controller_id"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        issued = self.service.complete_session(
            challenge["challenge_id"], challenge["challenge"],
            challenge["proof_payload"],
            self.key.sign(challenge["proof_payload"]),
        )
        principal = self.service.authenticate_session(
            issued["session_token"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        self.service.revoke_session(principal)

        with self.graph._connect() as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE controller_identities SET auth_epoch=auth_epoch+1 "
                    "WHERE controller_id=?", (paired["controller_id"],))
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE controller_pairings SET proof_signature_sha256=? "
                    "WHERE pairing_id=?",
                    ("f" * 64, pairing["pairing_id"]),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE controller_session_challenges "
                    "SET consumed_at='2026-08-24T08:00:01Z' "
                    "WHERE challenge_id=?", (challenge["challenge_id"],))
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE controller_sessions "
                    "SET ended_at='2026-08-24T08:01:59Z' WHERE session_id=?",
                    (issued["session_id"],),
                )

    def test_controller_cannot_revoke_another_controllers_session(self) -> None:
        _, first = self._pair(label="First")
        second_key = P256TestKey(self.root, "second-controller")
        _, second = self._pair(key=second_key, label="Second")
        principal = self.service.authenticate_session(
            first["session_token"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        with self.assertRaises(ControllerAuthError):
            self.service.revoke_session(principal, second["session_id"])
        self.service.authenticate_session(
            second["session_token"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )

    def test_revocation_before_first_effect_use_blocks_dispatch(self) -> None:
        _, paired = self._pair()
        tasks, _task_id, batch_id, _steps, principal = self._authorized_task(
            paired, approval_status="not_required", tool_name="list_files",
            args={"path": "/tmp"}, idempotency_class="read_only",
        )
        self.service.revoke_controller(principal, principal.controller_id)
        with self.assertRaisesRegex(PermissionError, "authority is stale"):
            tasks.claim_next_step(batch_id, "controller-worker")
        self.assertEqual(self.graph.count("action_receipts"), 0)
        self.assertEqual(self.graph.count("controller_effect_uses"), 0)

    def test_signed_approval_is_consumed_by_one_exact_first_effect(self) -> None:
        _, paired = self._pair()
        tasks, task_id, batch_id, steps, principal = self._authorized_task(
            paired, approval_status="pending")
        args = {"path": "authorized.txt", "content": "exact"}
        approvals = ApprovalService(
            self.graph, self.service, require_controller_decisions=True,
            clock=self.clock, request_ttl_seconds=120,
            authorization_ttl_seconds=60,
        )
        requested = approvals.request(
            task_id, "write_file", args, "exact filesystem change",
            step_id=steps[0]["step_id"], controller_principal=principal,
        )
        prepared = approvals.prepare_decision(
            requested["approval_id"], True, principal)
        decision = approvals.decide(
            requested["approval_id"], True,
            controller_principal=principal,
            proof_payload=prepared["proof_payload"],
            signature_b64url=self.key.sign(prepared["proof_payload"]),
        )
        claim = tasks.claim_next_step(batch_id, "controller-worker")
        self.assertIsNotNone(claim)
        with self.graph._connect() as conn:
            use = conn.execute(
                """SELECT approval_id,decision_id,task_id,step_id,action_id,
                          controller_id,authorizing_session_id
                     FROM controller_effect_uses"""
            ).fetchone()
        self.assertEqual(use["approval_id"], requested["approval_id"])
        self.assertEqual(use["decision_id"], decision["decision_id"])
        self.assertEqual(use["task_id"], task_id)
        self.assertEqual(use["step_id"], claim.step_id)
        self.assertEqual(use["action_id"], claim.action_id)
        self.assertEqual(use["controller_id"], principal.controller_id)
        self.assertEqual(use["authorizing_session_id"], principal.session_id)
        self.assertFalse(approvals.is_approved(task_id, "write_file", args))

    def test_wrong_approval_signature_leaves_no_decision_or_projection(self):
        _, paired = self._pair()
        _tasks, task_id, _batch_id, steps, principal = self._authorized_task(
            paired, approval_status="pending")
        args = {"path": "authorized.txt", "content": "exact"}
        approvals = ApprovalService(
            self.graph, self.service, require_controller_decisions=True,
            clock=self.clock, request_ttl_seconds=120,
            authorization_ttl_seconds=60,
        )
        requested = approvals.request(
            task_id, "write_file", args, "exact filesystem change",
            step_id=steps[0]["step_id"], controller_principal=principal,
        )
        prepared = approvals.prepare_decision(
            requested["approval_id"], True, principal)
        wrong_key = P256TestKey(self.root, "approval-wrong")
        with self.assertRaisesRegex(PermissionError, "signature is invalid"):
            approvals.decide(
                requested["approval_id"], True,
                controller_principal=principal,
                proof_payload=prepared["proof_payload"],
                signature_b64url=wrong_key.sign(prepared["proof_payload"]),
            )
        with self.graph._connect() as conn:
            status = conn.execute(
                "SELECT status FROM approval_state WHERE approval_id=?",
                (requested["approval_id"],),
            ).fetchone()[0]
            decisions = conn.execute(
                "SELECT COUNT(*) FROM controller_approval_decisions"
            ).fetchone()[0]
        self.assertEqual(status, "pending")
        self.assertEqual(decisions, 0)

    def test_cross_controller_cannot_request_or_decide_task_approval(self):
        _, owner = self._pair(label="Owner")
        _tasks, task_id, _batch_id, steps, owner_principal = (
            self._authorized_task(owner, approval_status="pending"))
        second_key = P256TestKey(self.root, "approval-outsider")
        _, outsider = self._pair(key=second_key, label="Outsider")
        outsider_principal = self.service.authenticate_session(
            outsider["session_token"], origin=ORIGIN,
            transport_binding_sha256=BINDING,
        )
        args = {"path": "authorized.txt", "content": "exact"}
        approvals = ApprovalService(
            self.graph, self.service, require_controller_decisions=True,
            clock=self.clock,
        )
        with self.assertRaisesRegex(PermissionError, "does not own the task"):
            approvals.request(
                task_id, "write_file", args, "exact filesystem change",
                step_id=steps[0]["step_id"],
                controller_principal=outsider_principal,
            )
        requested = approvals.request(
            task_id, "write_file", args, "exact filesystem change",
            step_id=steps[0]["step_id"],
            controller_principal=owner_principal,
        )
        with self.assertRaisesRegex(PermissionError, "request is stale"):
            approvals.prepare_decision(
                requested["approval_id"], True, outsider_principal)
        with self.graph._connect() as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM controller_approval_requests"
            ).fetchone()[0], 1)
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM controller_approval_decisions"
            ).fetchone()[0], 0)

    def test_revocation_after_signed_approval_blocks_first_effect(self) -> None:
        _, paired = self._pair()
        tasks, task_id, batch_id, steps, principal = self._authorized_task(
            paired, approval_status="pending")
        args = {"path": "authorized.txt", "content": "exact"}
        approvals = ApprovalService(
            self.graph, self.service, require_controller_decisions=True,
            clock=self.clock,
        )
        requested = approvals.request(
            task_id, "write_file", args, "exact filesystem change",
            step_id=steps[0]["step_id"], controller_principal=principal,
        )
        prepared = approvals.prepare_decision(
            requested["approval_id"], True, principal)
        approvals.decide(
            requested["approval_id"], True,
            controller_principal=principal,
            proof_payload=prepared["proof_payload"],
            signature_b64url=self.key.sign(prepared["proof_payload"]),
        )
        self.service.revoke_controller(principal, principal.controller_id)
        with self.assertRaisesRegex(PermissionError, "authority is stale"):
            tasks.claim_next_step(batch_id, "controller-worker")
        self.assertEqual(self.graph.count("action_receipts"), 0)
        self.assertEqual(self.graph.count("controller_effect_uses"), 0)

    def test_request_expiry_is_durable_and_step_can_be_reauthorized(self):
        _, paired = self._pair()
        _tasks, task_id, _batch_id, steps, principal = self._authorized_task(
            paired, approval_status="pending")
        args = {"path": "authorized.txt", "content": "exact"}
        approvals = ApprovalService(
            self.graph, self.service, require_controller_decisions=True,
            clock=self.clock, request_ttl_seconds=30,
            authorization_ttl_seconds=15,
        )
        first = approvals.request(
            task_id, "write_file", args, "first request",
            step_id=steps[0]["step_id"], controller_principal=principal,
        )
        self.clock.advance(30)
        with self.assertRaisesRegex(PermissionError, "request is stale"):
            approvals.prepare_decision(first["approval_id"], True, principal)
        with self.graph._connect() as conn:
            expired = conn.execute(
                "SELECT status,decided_at FROM approval_state WHERE approval_id=?",
                (first["approval_id"],),
            ).fetchone()
            step = conn.execute(
                "SELECT status,approval_status FROM task_steps WHERE step_id=?",
                (steps[0]["step_id"],),
            ).fetchone()
        self.assertEqual(expired["status"], "expired")
        self.assertIsNotNone(expired["decided_at"])
        self.assertEqual(tuple(step), ("waiting_approval", "pending"))
        renewed = approvals.request(
            task_id, "write_file", args, "renewed request",
            step_id=steps[0]["step_id"], controller_principal=principal,
        )
        self.assertNotEqual(renewed["approval_id"], first["approval_id"])

    def test_authorization_expiry_blocks_first_effect_at_exact_boundary(self):
        _, paired = self._pair()
        tasks, task_id, batch_id, steps, principal = self._authorized_task(
            paired, approval_status="pending")
        args = {"path": "authorized.txt", "content": "exact"}
        approvals = ApprovalService(
            self.graph, self.service, require_controller_decisions=True,
            clock=self.clock, request_ttl_seconds=60,
            authorization_ttl_seconds=15,
        )
        requested = approvals.request(
            task_id, "write_file", args, "short authorization",
            step_id=steps[0]["step_id"], controller_principal=principal,
        )
        prepared = approvals.prepare_decision(
            requested["approval_id"], True, principal)
        approvals.decide(
            requested["approval_id"], True,
            controller_principal=principal,
            proof_payload=prepared["proof_payload"],
            signature_b64url=self.key.sign(prepared["proof_payload"]),
        )
        self.clock.advance(15)
        with self.assertRaisesRegex(PermissionError, "authority is stale"):
            tasks.claim_next_step(batch_id, "controller-worker")
        self.assertEqual(self.graph.count("action_receipts"), 0)
        self.assertEqual(self.graph.count("controller_effect_uses"), 0)

    def test_signed_decision_replay_cannot_append_a_second_decision(self):
        _, paired = self._pair()
        _tasks, task_id, _batch_id, steps, principal = self._authorized_task(
            paired, approval_status="pending")
        args = {"path": "authorized.txt", "content": "exact"}
        approvals = ApprovalService(
            self.graph, self.service, require_controller_decisions=True,
            clock=self.clock,
        )
        requested = approvals.request(
            task_id, "write_file", args, "exact filesystem change",
            step_id=steps[0]["step_id"], controller_principal=principal,
        )
        prepared = approvals.prepare_decision(
            requested["approval_id"], True, principal)
        signature = self.key.sign(prepared["proof_payload"])
        approvals.decide(
            requested["approval_id"], True,
            controller_principal=principal,
            proof_payload=prepared["proof_payload"],
            signature_b64url=signature,
        )
        with self.assertRaises(PermissionError):
            approvals.decide(
                requested["approval_id"], True,
                controller_principal=principal,
                proof_payload=prepared["proof_payload"],
                signature_b64url=signature,
            )
        self.assertEqual(self.graph.count("controller_approval_decisions"), 1)

    def test_strict_interactive_task_without_controller_is_atomic(self):
        tasks = TaskService(
            self.graph, controller_auth=self.service,
            require_controller_authority=True,
        )
        with self.assertRaisesRegex(PermissionError, "requires controller"):
            tasks.create(
                "unauthorized interactive task", {"version": 0},
                session_id="browser-session", turn_id="browser-turn",
            )
        self.assertEqual(self.graph.count("task_state"), 0)
        with self.graph._connect() as conn:
            task_events = conn.execute(
                "SELECT COUNT(*) FROM graph_events WHERE event_type='task.created'"
            ).fetchone()[0]
        self.assertEqual(task_events, 0)

    def test_approval_proof_and_database_never_persist_private_arguments(self):
        _, paired = self._pair()
        secret = "private-controller-content-5f10da91b66f"
        args = {"path": "private.txt", "content": secret}
        _tasks, task_id, _batch_id, steps, principal = self._authorized_task(
            paired, approval_status="pending", args=args)
        approvals = ApprovalService(
            self.graph, self.service, require_controller_decisions=True,
            clock=self.clock,
        )
        requested = approvals.request(
            task_id, "write_file", args, "private filesystem change",
            step_id=steps[0]["step_id"], controller_principal=principal,
        )
        prepared = approvals.prepare_decision(
            requested["approval_id"], True, principal)
        self.assertNotIn(secret, prepared["proof_payload"])
        approvals.decide(
            requested["approval_id"], True,
            controller_principal=principal,
            proof_payload=prepared["proof_payload"],
            signature_b64url=self.key.sign(prepared["proof_payload"]),
        )
        with self.graph._connect() as conn:
            dump = "\n".join(conn.iterdump())
        self.assertNotIn(secret, dump)

    def test_database_trigger_rejects_effect_committed_at_session_expiry(self):
        _, paired = self._pair()
        _tasks, task_id, _batch_id, steps, principal = self._authorized_task(
            paired, approval_status="not_required", tool_name="list_files",
            args={"path": "/tmp"}, idempotency_class="read_only",
        )
        with self.assertRaisesRegex(
                sqlite3.IntegrityError, "controller effect authority is stale"):
            with self.graph.transaction() as conn:
                step = conn.execute(
                    "SELECT * FROM task_steps WHERE step_id=?",
                    (steps[0]["step_id"],),
                ).fetchone()
                authority = conn.execute(
                    "SELECT * FROM controller_task_authorities WHERE task_id=?",
                    (task_id,),
                ).fetchone()
                event_id, seq = self.graph.append_event(
                    conn, "action.started", {"forged": True},
                    actor="adversarial-test", task_id=task_id,
                )
                action_id = self.graph.append_node(
                    conn, "action", {"forged": True}, event_id=event_id)
                conn.execute(
                    """INSERT INTO action_receipts
                       (idempotency_key,task_id,step_id,action_id,tool_name,
                        args_sha256,status,risk,approval_status,created_at,
                        updated_at)
                       VALUES (?,?,?,?,?,?,'running','read_only',
                               'not_required',?,?)""",
                    (step["idempotency_key"], task_id, step["step_id"],
                     action_id, step["tool_name"], step["args_sha256"],
                     principal.absolute_expires_at,
                     principal.absolute_expires_at),
                )
                conn.execute(
                    """INSERT INTO controller_effect_uses
                       (action_id,task_id,step_id,idempotency_key,tool_name,
                        args_sha256,controller_id,controller_key_sha256,
                        controller_epoch,authorizing_session_id,
                        session_absolute_expires_at,transport_binding_sha256,
                        origin_sha256,approval_id,decision_id,committed_at,
                        committed_event_seq)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,?,?)""",
                    (action_id, task_id, step["step_id"],
                     step["idempotency_key"], step["tool_name"],
                     step["args_sha256"], authority["controller_id"],
                     authority["controller_key_sha256"],
                     authority["controller_epoch"], authority["session_id"],
                     authority["session_absolute_expires_at"],
                     authority["transport_binding_sha256"],
                     authority["origin_sha256"],
                     principal.absolute_expires_at, seq),
                )
        self.assertEqual(self.graph.count("action_receipts"), 0)
        self.assertEqual(self.graph.count("controller_effect_uses"), 0)

    def test_committed_first_use_survives_revocation_for_safe_retry(self) -> None:
        _, paired = self._pair()
        tasks, _task_id, batch_id, _steps, principal = self._authorized_task(
            paired, approval_status="not_required", tool_name="list_files",
            args={"path": "/tmp"}, idempotency_class="read_only",
        )
        first = tasks.claim_next_step(batch_id, "worker-before-revocation")
        self.assertIsNotNone(first)
        self.service.revoke_controller(principal, principal.controller_id)
        recovered = tasks.recover_inflight_steps(force=True)
        self.assertEqual(recovered["retry"], [first.step_id])
        retry = tasks.claim_next_step(batch_id, "worker-after-revocation")
        self.assertIsNotNone(retry)
        self.assertEqual(retry.action_id, first.action_id)
        self.assertEqual(retry.attempt_number, 2)
        self.assertEqual(self.graph.count("controller_effect_uses"), 1)

    def test_key_file_is_private_rehardened_and_symlinks_are_rejected(self):
        state = self.root / "key-state"
        ControllerAuthService(self.graph, state, clock=self.clock)
        key_path = state / "controller-auth.key"
        self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)
        key_path.chmod(0o644)
        ControllerAuthService(self.graph, state, clock=self.clock)
        self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)

        unsafe = self.root / "unsafe-state"
        unsafe.mkdir()
        target = self.root / "target-key"
        target.write_bytes(b"q" * 32)
        (unsafe / "controller-auth.key").symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, "key identity is invalid"):
            ControllerAuthService(self.graph, unsafe, clock=self.clock)

    def test_openssl_timeout_has_stable_fail_closed_error(self) -> None:
        payload = "controller proof"
        signature = self.key.sign(payload)
        with mock.patch(
            "friday_core.controller_auth.subprocess.run",
            side_effect=subprocess.TimeoutExpired([OPENSSL], 3),
        ):
            with self.assertRaisesRegex(
                    RuntimeError, "controller proof verifier failed"):
                verify_p256_signature(self.key.jwk, payload.encode(), signature)

    def test_origin_and_jwk_canonicalization_reject_ambiguity(self) -> None:
        self.assertEqual(
            normalize_https_origin("https://EXAMPLE.com:443/"),
            "https://example.com",
        )
        self.assertEqual(
            normalize_https_origin("https://[2001:0db8::1]:8500"),
            "https://[2001:db8::1]:8500",
        )
        for invalid in (
            "http://example.com", "https://example.com/path",
            "https://example.com./", "https://user@example.com",
            "https://exa%6dple.com", "https://-example.com",
        ):
            with self.assertRaises(ControllerAuthError, msg=invalid):
                normalize_https_origin(invalid)
        malformed = dict(self.key.jwk, d="private-material")
        with self.assertRaisesRegex(ControllerAuthError, "controller_key_invalid"):
            normalize_public_jwk(malformed)


class ControllerAuthoritySchemaTests(unittest.TestCase):
    def test_v11_upgrade_is_empty_idempotent_and_integrity_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "friday-v11.db"
            schema = Path(__file__).parents[1] / "friday_core" / "schema.sql"
            with sqlite3.connect(path) as conn:
                conn.execute("PRAGMA foreign_keys=ON")
                conn.executescript(schema.read_text())
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations "
                    "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL "
                    "DEFAULT CURRENT_TIMESTAMP)"
                )
                for version, migration in MIGRATIONS:
                    if version > 11:
                        break
                    migration(conn)
                    conn.execute(
                        "INSERT INTO schema_migrations(version) VALUES (?)",
                        (version,),
                    )
                conn.execute("PRAGMA user_version=11")
                conn.commit()
                self.assertEqual(
                    apply_schema_migrations(conn), LATEST_SCHEMA_VERSION)
                self.assertEqual(
                    apply_schema_migrations(conn), LATEST_SCHEMA_VERSION)
                conn.commit()
                versions = [row[0] for row in conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version")]
                counts = {
                    table: conn.execute(
                        f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in (
                        "controller_identities", "controller_pairings",
                        "controller_session_challenges", "controller_sessions",
                        "controller_task_authorities",
                        "controller_approval_requests",
                        "controller_approval_decisions",
                        "controller_effect_uses",
                    )
                }
                quick = conn.execute("PRAGMA quick_check").fetchone()[0]
                foreign_keys = conn.execute(
                    "PRAGMA foreign_key_check").fetchall()
            self.assertEqual(versions, list(range(1, 15)))
            self.assertEqual(set(counts.values()), {0})
            self.assertEqual(quick, "ok")
            self.assertEqual(foreign_keys, [])

    def test_fresh_schema_has_exact_controller_tables_indexes_and_triggers(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            with graph._connect() as conn:
                tables = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name LIKE 'controller_%'")}
                indexes = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name LIKE 'controller_%'")}
                triggers = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger' "
                    "AND name LIKE 'controller_%'")}
                pairing_fks = conn.execute(
                    "PRAGMA foreign_key_list(controller_pairings)"
                ).fetchall()
            self.assertEqual(tables, {
                "controller_identities", "controller_pairings",
                "controller_session_challenges", "controller_sessions",
                "controller_task_authorities",
                "controller_approval_requests",
                "controller_approval_decisions",
                "controller_effect_uses",
            })
            self.assertTrue({
                "controller_identities_auth_identity",
                "controller_pairings_pending",
                "controller_session_challenges_pending",
                "controller_sessions_active",
                "controller_sessions_exact_identity",
                "controller_sessions_authority_identity",
                "controller_approval_requests_expiry",
                "controller_approval_decisions_expiry",
                "controller_effect_uses_controller",
            }.issubset(indexes))
            self.assertTrue({
                "controller_identities_identity_immutable",
                "controller_identities_state_monotonic",
                "controller_pairings_identity_immutable",
                "controller_pairings_state_monotonic",
                "controller_challenges_identity_immutable",
                "controller_challenges_state_monotonic",
                "controller_sessions_identity_immutable",
                "controller_sessions_state_monotonic",
                "controller_task_authority_valid",
                "controller_approval_request_valid",
                "controller_approval_decision_valid",
                "controller_effect_use_valid",
            }.issubset(triggers))
            composite_groups = {}
            for row in pairing_fks:
                composite_groups.setdefault(int(row[0]), []).append(
                    (str(row[3]), str(row[2]), str(row[4])))
            self.assertIn(
                [
                    ("controller_id", "controller_identities", "controller_id"),
                    ("proposed_key_sha256", "controller_identities",
                     "public_key_sha256"),
                ],
                list(composite_groups.values()),
            )
