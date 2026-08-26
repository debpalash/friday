import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from friday_core import GraphStore, ModelRouter, TaskService
from friday_core.public_http import PublicHTTPResponse


class ModelRouterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.graph = GraphStore(Path(self.tmp.name) / "friday.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_remote_is_disabled_without_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            router = ModelRouter(
                self.graph, local_base_url="http://local", local_model="local")
            self.assertFalse(router.status()["remote_enabled"])
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                router.disclosure({"text": "hello"})

    def test_disclosure_redacts_secrets_and_records_hash(self):
        with patch.dict(os.environ, {"FRIDAY_REMOTE_API_KEY": "configured"}, clear=True):
            router = ModelRouter(
                self.graph, local_base_url="http://local", local_model="local",
                remote_base_url="https://remote.example/v1", remote_model="strong")
            result = router.disclosure(
                {"password": "private", "text": "token=abc task"})
        self.assertEqual(result["payload_preview"]["password"], "[REDACTED]")
        self.assertIn("[REDACTED]", result["payload_preview"]["text"])
        self.assertEqual(self.graph.count("model_disclosures"), 1)

    def test_approved_remote_completion_records_observation(self):
        task_id, _ = TaskService(self.graph).create(
            "reason", {"evidence": "legacy"})
        with patch.dict(os.environ, {"FRIDAY_REMOTE_API_KEY": "configured"}, clear=True), \
                patch("friday_core.model_router.normalize_public_http_url",
                      return_value="https://remote.example/v1"), \
                patch("friday_core.model_router.request_public_http",
                      return_value=PublicHTTPResponse(
                          url="https://remote.example/v1/chat/completions",
                          status=200, content_type="application/json",
                          charset="utf-8", body=json.dumps({
                              "choices": [{"message": {
                                  "content": "grounded answer"}}]
                          }).encode())) as request:
            router = ModelRouter(
                self.graph, local_base_url="http://local", local_model="local",
                remote_base_url="https://remote.example/v1", remote_model="strong")
            result = router.complete("password=private solve this", task_id=task_id)
        self.assertEqual(result["text"], "grounded answer")
        observation = self.graph.get_node(result["observation_id"])
        self.assertEqual(observation["kind"], "observation")
        self.assertFalse(request.call_args.kwargs["allow_redirects"])
        self.assertEqual(request.call_args.kwargs["max_redirects"], 0)
        self.assertEqual(request.call_args.kwargs["method"], "POST")

    def test_remote_completion_rejects_local_or_insecure_provider(self):
        task_id, _ = TaskService(self.graph).create(
            "reason", {"evidence": "legacy"})
        with patch.dict(
                os.environ, {"FRIDAY_REMOTE_API_KEY": "configured"}, clear=True):
            router = ModelRouter(
                self.graph, local_base_url="http://local",
                local_model="local",
                remote_base_url="http://127.0.0.1:9000/v1",
                remote_model="strong")
            with self.assertRaisesRegex(RuntimeError, "public HTTPS"):
                router.complete("solve", task_id=task_id)

            router = ModelRouter(
                self.graph, local_base_url="http://local",
                local_model="local",
                remote_base_url="http://remote.example/v1",
                remote_model="strong")
            with (patch(
                      "friday_core.model_router.normalize_public_http_url",
                      return_value="http://remote.example/v1"),
                  self.assertRaisesRegex(RuntimeError, "public HTTPS")):
                router.complete("solve", task_id=task_id)


if __name__ == "__main__":
    unittest.main()
