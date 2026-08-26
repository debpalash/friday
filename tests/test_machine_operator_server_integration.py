import asyncio
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import server
from friday_core import (ApprovalService, ClaimedStep, GraphStore,
                         MachineOperator, NativeVisionInput, OperatorGrantService,
                         ReflectionService, TaskService)


class _NoCapabilities:
    def active_metadata(self, _name):
        return None

    def active_names(self):
        return set()

    def tool_schemas(self):
        return []


def _claim(tool_name, args, *, task_id="task_machine_test",
           idempotency_key="act_machine_test_0001"):
    return ClaimedStep(
        step_id=f"step_{tool_name}", batch_id="batch_machine_test",
        task_id=task_id, round_index=0, ordinal=1,
        tool_call_id=f"call_{tool_name}", tool_name=tool_name,
        args=dict(args), idempotency_key=idempotency_key,
        idempotency_class=("read_only" if tool_name in {
            "machine_inspect_path", "machine_list_path", "machine_read_text",
            "machine_read_document", "machine_ocr_image",
            "machine_understand_image"
        } else "idempotent"),
        recovery_policy="retry", risk=(
            "read_only" if tool_name in {
                "machine_inspect_path", "machine_list_path", "machine_read_text",
                "machine_read_document", "machine_ocr_image",
                "machine_understand_image"
            } else "high"),
        approval_status=("not_required" if tool_name in {
            "machine_inspect_path", "machine_list_path", "machine_read_text",
            "machine_read_document", "machine_ocr_image",
            "machine_understand_image"
        } else "approved"),
        action_id=f"action_{tool_name}", attempt_id=f"attempt_{tool_name}",
        attempt_number=1, lease_id=f"lease_{tool_name}",
        worker_id="worker_machine_test", verifier="successful_receipt",
        executor_binding={}, resource_claims={}, context={},
    )


class MachineOperatorServerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_vision_step_sends_only_sanitized_ephemeral_image(self):
        calls = []
        prepared = NativeVisionInput(
            grant_id="grant_vision_server_0001", path="/tmp/private-scene.png",
            encoded=b"\x89PNG\r\n\x1a\nSANITIZED",
            provenance={
                "format": "png", "source_format": "jpeg",
                "sanitizer": "sandboxed-imagemagick",
                "limitations": "single_image_question_answering",
                "width": 512, "height": 256, "pixels": 131_072,
                "source_width": 1200, "source_height": 600,
                "source_pixels": 720_000, "source_bytes": 50_000,
                "source_sha256": "a" * 64, "image_bytes": 17,
                "image_sha256": "b" * 64,
            })

        class RecordingBroker:
            @staticmethod
            def native_vision_image(path, *, max_side):
                calls.append((path, max_side))
                return prepared

        class Completions:
            async def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="The red square."))])

        friday = server.Friday.__new__(server.Friday)
        friday.llm = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions()))
        question = "Which shape is farther left?"
        claim = _claim("machine_understand_image", {
            "path": prepared.path, "question": question})
        with patch.multiple(
                server, MACHINE_OPERATOR=RecordingBroker(),
                NATIVE_VISION_ENABLED=True, NATIVE_VISION_MAX_SIDE=512,
                RUNTIME_FINGERPRINT="c" * 64, LOCAL_MODEL="vision-model"), \
             patch.object(
                 server, "_native_vision_qualified", return_value=True):
            outcome = await friday.execute_claimed_step(claim)

        self.assertEqual(calls[0], (prepared.path, 512))
        request = calls[1]
        image_url = request["messages"][1]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        receipt = json.loads(outcome.result)
        self.assertNotIn("image_url", receipt)
        self.assertNotIn("encoded", receipt)
        self.assertEqual(receipt["answer"], "The red square.")
        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.verification["status"], "passed")

    async def test_execute_claimed_step_runs_bounded_image_ocr(self):
        calls = []
        text = "VISIBLE IMAGE 42"

        class RecordingBroker:
            @staticmethod
            def ocr_image(path, *, max_chars):
                calls.append((path, max_chars))
                return {
                    "status": "ok", "verified": True,
                    "grant_id": "grant_image_server_0001", "path": path,
                    "format": "png", "extractor": "sandboxed-tesseract",
                    "language": "eng", "limitations": "ocr_only",
                    "width": 800, "height": 200, "pixels": 160_000,
                    "text": text, "characters": len(text),
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "text_detected": True, "source_bytes": 2048,
                    "source_sha256": "a" * 64, "truncated": False,
                }

        friday = server.Friday.__new__(server.Friday)
        claim = _claim(
            "machine_ocr_image",
            {"path": "/tmp/visible-image.png", "max_chars": 12_345},
            idempotency_key="act_machine_image_0001")
        with patch.object(server, "MACHINE_OPERATOR", RecordingBroker()):
            outcome = await friday.execute_claimed_step(claim)
        self.assertEqual(calls, [("/tmp/visible-image.png", 12_345)])
        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.verification["status"], "passed")
        self.assertEqual(json.loads(outcome.result)["text"], text)

    async def test_grant_and_write_always_get_fresh_exact_ephemeral_approval(self):
        grant_args = {
            "path": "/private/server-integration-grant-41d8c7",
            "permissions": ["read", "write"],
            "allow_sensitive": False,
            "expires_at": "2035-01-02T03:04:05Z",
        }
        write_args = {
            "path": "/private/server-integration-grant-41d8c7/config.txt",
            "content": "exact ephemeral replacement 792ead\n",
        }

        with tempfile.TemporaryDirectory() as temporary:
            graph = GraphStore(Path(temporary) / "friday.db")
            tasks = TaskService(graph)
            approvals = ApprovalService(graph)
            contract = server.CONTRACTS.build(
                "Perform the exact approved machine operations.",
                ["machine_grant_path", "machine_write_text"],
            )
            task_id, _ = tasks.create(
                "Perform the exact approved machine operations.",
                contract.model_dump(mode="json"),
            )
            tasks.transition(task_id, "interpreting")
            tasks.transition(task_id, "planned")
            tasks.transition(task_id, "running")

            # Same-task, same-name, same-argument approvals are intentionally
            # insufficient: grants and writes require a fresh step-bound choice.
            for tool_name, args in (
                    ("machine_grant_path", grant_args),
                    ("machine_write_text", write_args)):
                stale = approvals.request(
                    task_id, tool_name, args, "legacy unbound approval")
                approvals.decide(stale["approval_id"], True)

            friday = server.Friday.__new__(server.Friday)
            friday.history = [{"role": "system", "content": "test"}]
            friday.save_session = lambda: None

            async def fake_stream(_messages, _speak_q, use_tools=True,
                                  required_tool=None):
                return "", [
                    {"id": "call_grant", "name": "machine_grant_path",
                     "args": json.dumps(grant_args)},
                    {"id": "call_write", "name": "machine_write_text",
                     "args": json.dumps(write_args)},
                ]

            friday._stream_once = fake_stream
            queue = asyncio.Queue()
            progress = []
            no_memory = SimpleNamespace(retrieve=lambda *_args, **_kwargs: [])
            no_feedback = SimpleNamespace(
                relevant_context=lambda *_args, **_kwargs: [])
            no_skills = SimpleNamespace(
                relevant_context=lambda *_args, **_kwargs: [])

            with patch.multiple(
                server,
                TASKS=tasks,
                APPROVALS=approvals,
                CAPABILITIES=_NoCapabilities(),
                REFLECTION=ReflectionService(graph),
                MEMORY=no_memory,
                FEEDBACK=no_feedback,
                SKILLS=no_skills,
                WORKER=None,
            ):
                await friday.respond(
                    "Perform both exact machine operations.", queue,
                    existing_task_id=task_id,
                    progress_sink=lambda event: _collect(progress, event),
                )

            exact_events = {
                item["tool_name"]: item for item in progress
                if item.get("type") == "approval_required"
            }
            steps = {item["tool_name"]: item
                     for item in tasks.list_steps(task_id=task_id)}
            pending = {item["tool_name"]: item
                       for item in approvals.list(status="pending")}
            with graph._connect() as conn:
                durable_dump = "\n".join(conn.iterdump())

            self.assertEqual(set(exact_events), {
                "machine_grant_path", "machine_write_text"})
            self.assertEqual(exact_events["machine_grant_path"]["args"],
                             grant_args | {"_args_sha256":
                                           steps["machine_grant_path"][
                                               "args_sha256"]})
            self.assertEqual(exact_events["machine_write_text"]["args"],
                             write_args | {"_args_sha256":
                                           steps["machine_write_text"][
                                               "args_sha256"]})
            self.assertEqual(tasks.get(task_id)["status"], "waiting_input")
            for tool_name in ("machine_grant_path", "machine_write_text"):
                self.assertEqual(steps[tool_name]["approval_status"], "pending")
                self.assertEqual(steps[tool_name]["status"], "waiting_approval")
                self.assertEqual(pending[tool_name]["step_id"],
                                 steps[tool_name]["step_id"])
                self.assertEqual(pending[tool_name]["args"]["path"],
                                 "[REDACTED]")
            self.assertEqual(pending["machine_write_text"]["args"]["content"],
                             "[REDACTED]")
            for secret in (grant_args["path"], write_args["path"],
                           write_args["content"]):
                self.assertNotIn(secret, durable_dump)
            self.assertEqual(len(approvals.list(status="approved")), 2)
            self.assertEqual(await queue.get(),
                             "I need your approval before I can do that.")
            self.assertIsNone(await queue.get())

    async def test_machine_write_uses_claim_idempotency_key_as_operation_id(self):
        calls = []

        class RecordingBroker:
            def write_text(self, path, content, *, operation_id):
                calls.append((path, content, operation_id))
                return {
                    "status": "ok", "verified": True,
                    "grant_id": "grant_recording_0001", "path": path,
                    "bytes": len(content.encode()),
                    "before_sha256": None,
                    "after_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "rollback_operation_id": operation_id,
                    "idempotent_replay": False,
                }

        friday = server.Friday.__new__(server.Friday)
        claim = _claim(
            "machine_write_text",
            {"path": "/tmp/recorded-machine-path", "content": "replacement\n"},
            idempotency_key="act_durable_machine_write_42",
        )
        with patch.object(server, "MACHINE_OPERATOR", RecordingBroker()):
            outcome = await friday.execute_claimed_step(claim)

        self.assertEqual(calls, [(
            "/tmp/recorded-machine-path", "replacement\n",
            "act_durable_machine_write_42")])
        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.verification["status"], "passed")
        self.assertEqual(json.loads(outcome.result)["rollback_operation_id"],
                         claim.idempotency_key)

    async def test_execute_claimed_step_runs_grant_read_write_and_rollback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            home = root / "home"
            state = root / "state"
            scope = project / "scope"
            project.mkdir()
            home.mkdir()
            state.mkdir()
            scope.mkdir()
            target = scope / "settings.txt"
            target.write_text("before\n")
            document = scope / "brief.docx"
            archive_bytes = io.BytesIO()
            with zipfile.ZipFile(
                    archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    "[Content_Types].xml",
                    '<Types><Override ContentType="application/vnd.'
                    'openxmlformats-officedocument.wordprocessingml.'
                    'document.main+xml"/></Types>')
                archive.writestr(
                    "word/document.xml",
                    '<w:document xmlns:w="w"><w:p><w:r><w:t>private brief'
                    '</w:t></w:r></w:p></w:document>')
            document.write_bytes(archive_bytes.getvalue())
            graph = GraphStore(state / "friday.db")
            tasks = TaskService(graph)
            task_id, _ = tasks.create("machine integration", {})
            grants = OperatorGrantService(
                graph, project, home=home, state_root=state)
            broker = MachineOperator(grants, state_root=state)
            friday = server.Friday.__new__(server.Friday)

            grant_claim = _claim(
                "machine_grant_path",
                {"path": str(scope),
                 "permissions": ["inspect", "list", "read", "write"],
                 "allow_sensitive": False},
                task_id=task_id,
                idempotency_key="act_machine_grant_0001",
            )
            read_claim = _claim(
                "machine_read_text", {"path": str(target)}, task_id=task_id,
                idempotency_key="act_machine_read_0001")
            document_claim = _claim(
                "machine_read_document",
                {"path": str(document), "max_chars": 10_000},
                task_id=task_id,
                idempotency_key="act_machine_document_0001")
            write_claim = _claim(
                "machine_write_text",
                {"path": str(target), "content": "after\n"}, task_id=task_id,
                idempotency_key="act_machine_write_0001")
            rollback_claim = _claim(
                "machine_rollback_write",
                {"operation_id": write_claim.idempotency_key}, task_id=task_id,
                idempotency_key="act_machine_rollback_0001")

            with patch.multiple(
                    server, OPERATOR_GRANTS=grants, MACHINE_OPERATOR=broker):
                granted = await friday.execute_claimed_step(grant_claim)
                read = await friday.execute_claimed_step(read_claim)
                document_read = await friday.execute_claimed_step(document_claim)
                written = await friday.execute_claimed_step(write_claim)
                self.assertEqual(target.read_text(), "after\n")
                rolled_back = await friday.execute_claimed_step(rollback_claim)

            for outcome in (granted, read, document_read, written, rolled_back):
                self.assertTrue(outcome.succeeded)
                self.assertEqual(outcome.verification["status"], "passed")
            self.assertEqual(json.loads(read.result)["text"], "before\n")
            self.assertEqual(
                json.loads(document_read.result)["text"], "private brief")
            self.assertEqual(
                json.loads(written.result)["rollback_operation_id"],
                write_claim.idempotency_key)
            self.assertEqual(
                json.loads(rolled_back.result)["operation_id"],
                write_claim.idempotency_key)
            self.assertEqual(target.read_text(), "before\n")

    def test_machine_verifier_rejects_generic_and_semantically_forged_receipts(self):
        tools = (
            "machine_grant_path", "machine_read_text",
            "machine_write_text", "machine_rollback_write",
        )
        for tool_name in tools:
            with self.subTest(tool_name=tool_name, kind="generic"):
                generic = server.OUTCOMES.verify_action(
                    tool_name, "done", succeeded=True)
                self.assertFalse(generic.passed)

        forged = {
            "machine_grant_path": ({
                "status": "active", "grant_id": "grant_forged_0001",
                "target": "/tmp/exact-grant", "target_sha256": "0" * 64,
                "permissions": ["write"], "allow_sensitive": False,
                "expires_at": None,
            }, {"path": "/tmp/exact-grant", "permissions": ["write"],
                "allow_sensitive": False}),
            "machine_read_text": ({
                "status": "ok", "verified": True,
                "grant_id": "grant_forged_0001", "path": "/tmp/exact.txt",
                "text": "actual text", "bytes": len("actual text".encode()),
                "sha256": "0" * 64,
            }, {"path": "/tmp/exact.txt", "max_bytes": 64_000}),
            "machine_write_text": ({
                "status": "ok", "verified": True,
                "grant_id": "grant_forged_0001", "path": "/tmp/exact.txt",
                "bytes": len("approved text".encode()),
                "after_sha256": "0" * 64,
                "rollback_operation_id": "forged_operation_0001",
            }, {"path": "/tmp/exact.txt", "content": "approved text"}),
            "machine_rollback_write": ({
                "status": "ok", "verified": True,
                "grant_id": "grant_forged_0001", "path": "/tmp/exact.txt",
                "operation_id": "forged_operation_0001",
                "restored_absence": True, "restored_sha256": "0" * 64,
            }, {"operation_id": "forged_operation_0001"}),
        }
        for tool_name, (receipt, args) in forged.items():
            with self.subTest(tool_name=tool_name, kind="forged"):
                check = server.OUTCOMES.verify_action(
                    tool_name, receipt, succeeded=True, args=args,
                    idempotency_key="forged_operation_0001")
                self.assertFalse(check.passed)


async def _collect(target, event):
    target.append(event)


if __name__ == "__main__":
    unittest.main()
