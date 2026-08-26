import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from friday_core import GraphStore, TaskService


class TaskServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.graph = GraphStore(Path(self.tmp.name) / "friday.db")
        self.tasks = TaskService(self.graph)

    def tearDown(self):
        self.tmp.cleanup()

    def make_running_task(self):
        task_id, _ = self.tasks.create(
            "Inspect voice files", {"evidence": "tool receipts and final verification"})
        self.tasks.transition(task_id, "interpreting")
        self.tasks.set_plan(task_id, ["List files", "Verify result"])
        self.tasks.transition(task_id, "planned")
        self.tasks.transition(task_id, "running")
        return task_id

    def test_task_lifecycle_and_progress_are_durable(self):
        task_id = self.make_running_task()
        self.tasks.transition(task_id, "verifying")
        self.tasks.transition(task_id, "completed")

        task = self.tasks.get(task_id)
        progress = self.tasks.progress_since()
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["plan"], ["List files", "Verify result"])
        self.assertEqual(progress[-1]["state"], "completed")
        self.assertGreaterEqual(len(progress), 7)

    def test_invalid_transition_is_rejected(self):
        task_id, _ = self.tasks.create("x", {"evidence": "y"})
        with self.assertRaises(ValueError):
            self.tasks.transition(task_id, "completed")

    def test_successful_action_is_idempotent(self):
        task_id = self.make_running_task()
        handle, progress = self.tasks.begin_action(
            task_id, "list_files", {"path": "."}, ordinal=1)
        self.assertIsNotNone(progress)
        self.tasks.finish_action(handle, "f server.py", succeeded=True)

        replay, replay_progress = self.tasks.begin_action(
            task_id, "list_files", {"path": "."}, ordinal=1)
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.prior_result, "f server.py")
        self.assertIsNone(replay_progress)
        self.assertEqual(self.graph.count("action_receipts"), 1)

    def test_private_tool_payloads_are_hash_only_in_durable_graph(self):
        task_id = self.make_running_task()
        cases = [
            ("clipboard_write", {"text": "clipboard-write-argument-secret"},
             {"status": "ok", "text": "clipboard-write-result-secret"}),
            ("clipboard_read", {},
             {"status": "ok", "text": "clipboard-read-result-secret"}),
            ("browser_type", {
                "selector": "#password", "text": "browser-typed-secret",
                "page_url": "https://example.test/?token=browser-url-secret",
             }, {"status": "ok", "url": "https://example.test/private",
                 "text": "browser-result-secret"}),
            ("remote_reason", {"prompt": "remote-prompt-secret"},
             {"status": "ok", "response": "remote-result-secret"}),
            ("read_file", {"path": "private.txt"},
             "file-read-result-secret"),
            ("machine_launch_process", {
                "spec_id": "spec_calculator_0001",
                "parameter_values": {
                    "document_alias": "process-parameter-secret"},
             }, {"status": "running",
                 "spec_id": "spec_calculator_0001",
                 "instance_id": "process_instance_0001",
                 "private_output": "process-result-secret"}),
        ]
        raw_results: dict[str, str] = {}
        for ordinal, (tool_name, args, value) in enumerate(cases, start=1):
            handle, _ = self.tasks.begin_action(
                task_id, tool_name, args, ordinal=ordinal)
            raw_result = json.dumps(value)
            raw_results[tool_name] = raw_result
            self.tasks.finish_action(handle, raw_result, succeeded=True,
                                     verification={
                                         "status": "passed",
                                         "summary": (
                                             value.get("text")
                                             or value.get("response")
                                             or "verified"
                                             if isinstance(value, dict)
                                             else "verified"),
                                         "evidence": (
                                             list(value.values())
                                             if isinstance(value, dict)
                                             else [value]),
                                         "effects": [{
                                             "kind": "tool_result",
                                             **(value if isinstance(value, dict)
                                                else {"content": value}),
                                         }],
                                     })

        with self.graph._connect() as conn:
            durable_dump = "\n".join(conn.iterdump())
            started = [json.loads(row["payload_json"])
                       for row in conn.execute(
                           "SELECT payload_json FROM graph_events "
                           "WHERE event_type='action.started'")]
            receipts = {row["tool_name"]: dict(row) for row in conn.execute(
                "SELECT tool_name,result_json FROM action_receipts")}

        secrets = {
            "clipboard-write-argument-secret",
            "clipboard-write-result-secret",
            "clipboard-read-result-secret",
            "browser-typed-secret",
            "browser-url-secret",
            "browser-result-secret",
            "remote-prompt-secret",
            "remote-result-secret",
            "file-read-result-secret",
            "process-parameter-secret",
            "process-result-secret",
        }
        for secret in secrets:
            self.assertNotIn(secret, durable_dump)

        clipboard_start = next(
            event for event in started if event["tool"] == "clipboard_write")
        self.assertEqual(clipboard_start["args"]["text"], "[REDACTED]")
        self.assertEqual(
            clipboard_start["args"]["text_sha256"],
            hashlib.sha256(b"clipboard-write-argument-secret").hexdigest())
        for tool_name, raw_result in raw_results.items():
            persisted_string = json.loads(receipts[tool_name]["result_json"])
            persisted = json.loads(persisted_string)
            self.assertTrue(persisted["_redacted"])
            self.assertEqual(
                persisted["result_sha256"],
                hashlib.sha256(raw_result.encode()).hexdigest())

    def test_file_content_and_notification_text_are_not_journaled(self):
        task_id = self.make_running_task()
        cases = [
            ("write_file", {"path": "generated.py",
                            "content": "write-content-secret"}),
            ("desktop_notify", {"title": "notification-title-secret",
                                "message": "notification-message-secret"}),
        ]
        for ordinal, (tool_name, args) in enumerate(cases, start=1):
            handle, _ = self.tasks.begin_action(
                task_id, tool_name, args, ordinal=ordinal)
            self.tasks.finish_action(
                handle, '{"status":"ok"}', succeeded=True)

        with self.graph._connect() as conn:
            durable_dump = "\n".join(conn.iterdump())
            started = [json.loads(row[0]) for row in conn.execute(
                "SELECT payload_json FROM graph_events "
                "WHERE event_type='action.started'")]

        self.assertNotIn("write-content-secret", durable_dump)
        self.assertNotIn("notification-title-secret", durable_dump)
        self.assertNotIn("notification-message-secret", durable_dump)
        write_args = next(
            item["args"] for item in started if item["tool"] == "write_file")
        self.assertEqual(write_args["content"], "[REDACTED]")

    def test_interrupted_task_enters_recovery_once(self):
        task_id = self.make_running_task()

        recovered = self.tasks.recover_interrupted()
        recovered_again = self.tasks.recover_interrupted()

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered_again, [])
        self.assertEqual(self.tasks.get(task_id)["status"], "recovering")


if __name__ == "__main__":
    unittest.main()
