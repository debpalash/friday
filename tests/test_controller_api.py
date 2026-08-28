import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from friday_core.controller_api import ControllerAPI, ControllerAPIError
from friday_core.controller_auth import ControllerAuthError


class ControllerAPITests(unittest.TestCase):
    def setUp(self):
        self.auth = Mock()
        self.api = ControllerAPI(self.auth, "a" * 64)

    def test_pairing_boundary_rejects_extra_fields_before_auth(self):
        with self.assertRaises(ControllerAPIError) as raised:
            self.api.prepare_pairing(
                {"origin": "https://localhost:8500"},
                {"pairing_token": "p", "label": "local",
                 "public_jwk": {}, "extra": True},
            )

        self.assertEqual(raised.exception.status_code, 400)
        self.auth.prepare_pairing.assert_not_called()

    def test_pairing_boundary_passes_canonical_origin_and_binding(self):
        self.auth.prepare_pairing.return_value = {"challenge": "ok"}

        result = self.api.prepare_pairing(
            {"origin": "https://LOCALHOST:8500/"},
            {"pairing_token": "p", "label": "local", "public_jwk": {}},
        )

        self.assertEqual(result, {"challenge": "ok"})
        self.assertEqual(
            self.auth.prepare_pairing.call_args.kwargs,
            {"origin": "https://localhost:8500",
             "transport_binding_sha256": "a" * 64},
        )

    def test_auth_failures_are_coarse_at_the_api_boundary(self):
        self.auth.complete_session.side_effect = ControllerAuthError("private")

        with self.assertRaises(ControllerAPIError) as raised:
            self.api.complete_session({
                "challenge_id": "id", "challenge": "c", "proof_payload": {},
                "signature_b64url": "s",
            })

        self.assertEqual((raised.exception.status_code, raised.exception.detail),
                         (401, "controller proof was rejected"))

    def test_identity_and_revocation_results_are_bounded(self):
        principal = SimpleNamespace(
            controller_id="controller", session_id="session",
            public_key_sha256="key", controller_epoch=3,
            idle_expires_at="idle", absolute_expires_at="absolute",
        )

        self.assertEqual(self.api.identity(principal)["controller_epoch"], 3)
        self.assertEqual(
            self.api.revoke_controller(principal, "target"),
            {"status": "revoked", "controller_id": "target"},
        )
        self.assertEqual(
            self.api.revoke_session(principal),
            {"status": "revoked", "session_id": "session"},
        )
        self.auth.revoke_controller.assert_called_once_with(principal, "target")
        self.auth.revoke_session.assert_called_once_with(principal)


if __name__ == "__main__":
    unittest.main()
