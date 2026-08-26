import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.responses import JSONResponse
from starlette.requests import Request

import server
from friday_core.controller_auth import ControllerAuthService
from friday_core.feedback import ApprovalService
from friday_core.graph import GraphStore
from friday_core.tasks import TaskService
from tests.test_controller_auth import BINDING, ORIGIN, P256TestKey


def _request(path: str, *, authorization: str | None = None,
             origin: str = ORIGIN, bootstrap: str | None = None) -> Request:
    headers = [(b"host", b"192.168.1.158:8500"),
               (b"origin", origin.encode())]
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))
    if bootstrap is not None:
        headers.append((b"x-friday-token", bootstrap.encode()))
    return Request({
        "type": "http", "http_version": "1.1", "method": "GET",
        "scheme": "https", "path": path, "raw_path": path.encode(),
        "query_string": b"", "headers": headers,
        "server": ("192.168.1.158", 8500),
        "client": ("192.168.1.175", 41000),
    })


class ControllerServerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph = GraphStore(self.root / "friday.db")
        self.auth = ControllerAuthService(
            self.graph, self.root / "controller-auth",
            key_provider=lambda: b"k" * 32)
        self.key = P256TestKey(self.root, "browser")
        self.tls = SimpleNamespace(transport_binding_sha256=BINDING)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    async def _pair_through_endpoints(self) -> dict:
        request = SimpleNamespace(
            headers={"origin": ORIGIN, "host": "192.168.1.158:8500"})
        with mock.patch.multiple(
                server, CONTROLLER_AUTH=self.auth, TLS_MATERIAL=self.tls):
            pairing = await server.api_create_controller_pairing()
            prepared = await server.api_prepare_controller_pairing(request, {
                "pairing_token": pairing["pairing_token"],
                "label": "Browser controller",
                "public_jwk": self.key.jwk,
            })
            completed = await server.api_complete_controller_pairing(request, {
                "pairing_token": pairing["pairing_token"],
                "label": "Browser controller",
                "public_jwk": self.key.jwk,
                "signature_b64url": self.key.sign(
                    prepared["proof_payload"]),
            })
        return completed

    async def test_pairing_endpoints_issue_controller_session_and_safe_me(self):
        completed = await self._pair_through_endpoints()
        principal = self.auth.authenticate_session(
            completed["session_token"], origin=ORIGIN,
            transport_binding_sha256=BINDING)
        request = SimpleNamespace(
            state=SimpleNamespace(controller_principal=principal))
        identity = await server.api_controller_identity(request)

        self.assertEqual(identity["controller_id"], completed["controller_id"])
        self.assertEqual(identity["session_id"], completed["session_id"])
        self.assertNotIn("session_token", identity)
        self.assertNotIn("transport_binding_sha256", identity)

    async def test_returning_controller_challenge_and_session_revocation(self):
        completed = await self._pair_through_endpoints()
        request = SimpleNamespace(
            headers={"origin": ORIGIN, "host": "192.168.1.158:8500"})
        with mock.patch.multiple(
                server, CONTROLLER_AUTH=self.auth, TLS_MATERIAL=self.tls):
            challenge = await server.api_controller_session_challenge(
                request, {"controller_id": completed["controller_id"]})
            resumed = await server.api_complete_controller_session({
                "challenge_id": challenge["challenge_id"],
                "challenge": challenge["challenge"],
                "proof_payload": challenge["proof_payload"],
                "signature_b64url": self.key.sign(
                    challenge["proof_payload"]),
            })
            principal = self.auth.authenticate_session(
                resumed["session_token"], origin=ORIGIN,
                transport_binding_sha256=BINDING)
            revoke_request = SimpleNamespace(
                state=SimpleNamespace(controller_principal=principal))
            revoked = await server.api_revoke_controller_session(
                revoke_request)

        self.assertEqual(revoked["status"], "revoked")
        self.assertEqual(revoked["session_id"], resumed["session_id"])
        with self.assertRaises(PermissionError):
            self.auth.authenticate_session(
                resumed["session_token"], origin=ORIGIN,
                transport_binding_sha256=BINDING)

    async def test_controller_inventory_is_safe_and_revocation_is_immediate(self):
        completed = await self._pair_through_endpoints()
        principal = self.auth.authenticate_session(
            completed["session_token"], origin=ORIGIN,
            transport_binding_sha256=BINDING)
        request = SimpleNamespace(
            state=SimpleNamespace(controller_principal=principal))
        with mock.patch.multiple(server, CONTROLLER_AUTH=self.auth):
            inventory = await server.api_controller_list(request)
            revoked = await server.api_revoke_controller(
                completed["controller_id"], request)

        self.assertEqual(len(inventory["controllers"]), 1)
        item = inventory["controllers"][0]
        self.assertEqual(item["controller_id"], completed["controller_id"])
        self.assertNotIn("public_jwk_json", item)
        self.assertNotIn("transport_binding_sha256", item)
        self.assertNotIn("token_digest", item)
        self.assertEqual(revoked["status"], "revoked")
        with self.assertRaises(PermissionError):
            self.auth.authenticate_session(
                completed["session_token"], origin=ORIGIN,
                transport_binding_sha256=BINDING)

    def test_startup_retires_unsigned_legacy_approval_and_orphan_task(self):
        tasks = TaskService(self.graph)
        task_id, _ = tasks.create(
            "Legacy request cannot become paired authority",
            {"version": 0, "evidence": "retirement journal"})
        tasks.transition(task_id, "interpreting")
        tasks.transition(task_id, "waiting_input")
        legacy = ApprovalService(self.graph)
        approval = legacy.request(
            task_id, "write_file", {"path": "legacy.txt"},
            "unsigned legacy request")
        strict = ApprovalService(
            self.graph, self.auth, require_controller_decisions=True)

        with mock.patch.multiple(server, APPROVALS=strict, TASKS=tasks):
            first = server._retire_legacy_controller_authority()
            second = server._retire_legacy_controller_authority()

        self.assertEqual(first, {
            "retired_approvals": 1, "cancelled_tasks": 1})
        self.assertEqual(second, {
            "retired_approvals": 0, "cancelled_tasks": 0})
        self.assertEqual(tasks.get(task_id)["status"], "cancelled")
        self.assertEqual(
            strict.list(status="cancelled")[0]["approval_id"],
            approval["approval_id"])
        event_types = [
            event["event_type"]
            for event in self.graph.events_since(0, limit=100)]
        self.assertIn("approval.cancelled", event_types)
        self.assertIn("task.transitioned", event_types)

    async def test_middleware_requires_bearer_session_for_operational_api(self):
        completed = await self._pair_through_endpoints()
        observed = {}

        async def accepted(request):
            observed["principal"] = request.state.controller_principal
            return JSONResponse({"accepted": True})

        with mock.patch.multiple(
                server, CONTROLLER_AUTH=self.auth, TLS_MATERIAL=self.tls,
                ALLOWED_HOSTS=frozenset({"192.168.1.158"}),
                ALLOWED_ORIGINS=frozenset({ORIGIN.lower()})):
            denied = await server.protect_control_plane(
                _request("/api/status"), accepted)
            allowed = await server.protect_control_plane(
                _request(
                    "/api/status",
                    authorization="Bearer " + completed["session_token"]),
                accepted)

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(
            observed["principal"].controller_id, completed["controller_id"])
        self.assertEqual(allowed.headers["cache-control"], "no-store")

    async def test_bootstrap_token_is_accepted_only_on_pairing_start(self):
        called = []

        async def accepted(_request):
            called.append(True)
            return JSONResponse({"accepted": True})

        with mock.patch.multiple(
                server, ALLOWED_HOSTS=frozenset({"192.168.1.158"}),
                ALLOWED_ORIGINS=frozenset({ORIGIN.lower()})):
            pairing = await server.protect_control_plane(
                _request(
                    "/api/controllers/pairings",
                    bootstrap=server.CONTROL_TOKEN),
                accepted)
            operational = await server.protect_control_plane(
                _request("/api/status", bootstrap=server.CONTROL_TOKEN),
                accepted)

        self.assertEqual(pairing.status_code, 200)
        self.assertEqual(operational.status_code, 401)
        self.assertEqual(called, [True])

    async def test_cross_origin_controller_session_fails_before_route(self):
        completed = await self._pair_through_endpoints()

        async def must_not_run(_request):
            self.fail("cross-origin route was reached")

        with mock.patch.multiple(
                server, CONTROLLER_AUTH=self.auth, TLS_MATERIAL=self.tls,
                ALLOWED_HOSTS=frozenset({"192.168.1.158"}),
                ALLOWED_ORIGINS=frozenset({ORIGIN.lower()})):
            response = await server.protect_control_plane(
                _request(
                    "/api/status", origin="https://attacker.example",
                    authorization="Bearer " + completed["session_token"]),
                must_not_run)

        self.assertEqual(response.status_code, 403)

    async def test_pairing_endpoint_rejects_extra_fields_before_auth_service(self):
        request = SimpleNamespace(
            headers={"origin": ORIGIN, "host": "192.168.1.158:8500"})
        body = {
            "pairing_token": "not-used", "label": "Browser",
            "public_jwk": self.key.jwk, "private_key": "must-never-arrive",
        }
        with self.assertRaises(server.HTTPException) as raised, \
             mock.patch.object(self.auth, "prepare_pairing") as prepare, \
             mock.patch.multiple(
                 server, CONTROLLER_AUTH=self.auth, TLS_MATERIAL=self.tls):
            await server.api_prepare_controller_pairing(request, body)

        self.assertEqual(raised.exception.status_code, 400)
        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
