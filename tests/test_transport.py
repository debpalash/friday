import unittest

from friday_core.controller_auth import ControllerAuthError
from friday_core.transport import (
    bearer_session_token,
    controller_origin,
    valid_control_token,
    valid_host,
    valid_origin,
    websocket_session_token,
)


class ControlPlaneTransportTests(unittest.TestCase):
    def test_host_and_origin_admission_is_exact(self):
        hosts = frozenset({"localhost", "127.0.0.1"})
        origins = frozenset({"https://localhost:8500"})

        self.assertTrue(valid_host("localhost:8500", hosts))
        self.assertTrue(valid_host("127.0.0.1", hosts))
        self.assertFalse(valid_host("user@localhost:8500", hosts))
        self.assertFalse(valid_host("attacker.invalid", hosts))
        self.assertFalse(valid_host(None, hosts))
        self.assertTrue(valid_origin(None, origins))
        self.assertTrue(valid_origin("https://LOCALHOST:8500/", origins))
        self.assertFalse(valid_origin("https://localhost.evil:8500", origins))

    def test_bootstrap_comparison_and_session_parsing_fail_closed(self):
        self.assertTrue(valid_control_token("secret", "secret"))
        self.assertFalse(valid_control_token("secreu", "secret"))
        self.assertFalse(valid_control_token(None, "secret"))
        self.assertEqual(
            websocket_session_token("friday.v1, session.bound-token"),
            "bound-token",
        )
        self.assertIsNone(websocket_session_token("friday.v1"))
        self.assertEqual(bearer_session_token("Bearer bound-token"),
                         "bound-token")
        for value in (None, "Basic token", "Bearer ", "Bearer two tokens"):
            with self.subTest(value=value):
                self.assertIsNone(bearer_session_token(value))

    def test_controller_origin_requires_canonical_https_origin(self):
        self.assertEqual(
            controller_origin({"origin": "https://LOCALHOST:8500/"}),
            "https://localhost:8500",
        )
        self.assertEqual(
            controller_origin({"host": "localhost:8500"}),
            "https://localhost:8500",
        )
        with self.assertRaises(ControllerAuthError):
            controller_origin({})
        with self.assertRaises(ControllerAuthError):
            controller_origin({"origin": "http://localhost:8500"})


if __name__ == "__main__":
    unittest.main()
