import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.responses import JSONResponse
from starlette.requests import Request

import server
from friday_core.feedback import ApprovalService
from friday_core.graph import GraphStore
from friday_core.tasks import TaskService


ORIGIN = "https://127.0.0.1:8500"


def _request(path: str, *, host: str = "127.0.0.1:8500",
             origin: str | None = ORIGIN,
             authorization: str | None = None,
             bootstrap: str | None = None) -> Request:
    headers = [(b"host", host.encode())]
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))
    if bootstrap is not None:
        headers.append((b"x-friday-token", bootstrap.encode()))
    return Request({
        "type": "http", "http_version": "1.1", "method": "GET",
        "scheme": "https", "path": path, "raw_path": path.encode(),
        "query_string": b"", "headers": headers,
        "server": ("127.0.0.1", 8500),
        "client": ("127.0.0.1", 41000),
    })


class LocalControlPlaneIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_operational_api_needs_no_token_or_bearer(self):
        called = []

        async def accepted(request):
            called.append(request.url.path)
            return JSONResponse({"accepted": True})

        response = await server.protect_control_plane(
            _request("/api/status"), accepted)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(called, ["/api/status"])
        self.assertEqual(response.headers["cache-control"], "no-store")

    async def test_old_auth_headers_are_ignored_on_loopback(self):
        async def accepted(_request):
            return JSONResponse({"accepted": True})

        response = await server.protect_control_plane(
            _request(
                "/api/status", authorization="Bearer obsolete",
                bootstrap="obsolete"),
            accepted)

        self.assertEqual(response.status_code, 200)

    async def test_foreign_host_and_origin_fail_before_route(self):
        async def must_not_run(_request):
            self.fail("foreign request reached the application route")

        wrong_host = await server.protect_control_plane(
            _request("/api/status", host="attacker.example"), must_not_run)
        wrong_origin = await server.protect_control_plane(
            _request("/api/status", origin="https://attacker.example"),
            must_not_run)

        self.assertEqual(wrong_host.status_code, 403)
        self.assertEqual(wrong_origin.status_code, 403)

    def test_runtime_is_fixed_to_loopback(self):
        self.assertIn(server.BIND_HOST, server.LOOPBACK_HOSTS)
        self.assertEqual(
            server.ALLOWED_HOSTS,
            frozenset({"localhost", "127.0.0.1", "::1"}),
        )

    def test_pairing_and_controller_routes_are_absent(self):
        paths = {route.path for route in server.app.routes}

        self.assertFalse(any(path.startswith("/api/controllers")
                             for path in paths))
        self.assertNotIn("/api/approvals/{approval_id}/prepare", paths)

    def test_upgrade_removes_retired_auth_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            controller_state = state / "controller-auth"
            controller_state.mkdir()
            (state / "control-token").write_text("obsolete")
            (controller_state / "controller-auth.key").write_bytes(b"obsolete")

            with mock.patch.object(server, "STATE_DIR", state):
                server._remove_retired_auth_artifacts()

            self.assertFalse((state / "control-token").exists())
            self.assertFalse(controller_state.exists())

    async def test_local_approval_uses_one_exact_boolean(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            approvals = ApprovalService(graph)
            task_id, _ = tasks.create(
                "Approve one local action", {"version": 0})
            approval = approvals.request(
                task_id, "write_file", {"path": "approved.txt"},
                "local user decision")
            with mock.patch.multiple(
                    server, APPROVALS=approvals, TASKS=tasks, WORKER=None):
                decision = await server.api_decide_approval(
                    approval["approval_id"], {"approved": True})

        self.assertEqual(decision["status"], "approved")

    async def test_local_approval_rejects_extra_or_loose_fields(self):
        for body in (
                {"approved": "true"},
                {"approved": True, "signature_b64url": "obsolete"}):
            with self.subTest(body=body), self.assertRaises(
                    server.HTTPException) as raised:
                await server.api_decide_approval("approval_missing", body)
            self.assertEqual(raised.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
