import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont

import server
from friday_core.graph import GraphStore
from friday_core.images import extract_image_text, prepare_native_vision_image
from friday_core.machine import MachineOperator, OperatorGrantService
from friday_core.tasks import redact_tool_arguments, redact_tool_result


FONT = Path("/usr/share/fonts/liberation/LiberationSans-Regular.ttf")


def _render(path: Path, text: str, *, image_format: str) -> None:
    image = Image.new("RGB", (1200, 260), "white")
    if text:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(FONT), 72)
        draw.text((40, 70), text, fill="black", font=font)
    image.save(path, format=image_format)


class ImageExtractionTests(unittest.TestCase):
    def _extract(self, path: Path, *, max_chars=80_000):
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            return extract_image_text(fd, path.name, max_chars=max_chars)
        finally:
            os.close(fd)

    def _prepare_vision(self, path: Path, *, max_side=512):
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            return prepare_native_vision_image(
                fd, path.name, max_side=max_side)
        finally:
            os.close(fd)

    def test_real_native_vision_sanitizer_is_networkless_bounded_and_canonical(self):
        with tempfile.TemporaryDirectory() as temporary:
            for suffix, image_format in (("png", "PNG"), ("jpg", "JPEG")):
                with self.subTest(suffix=suffix):
                    path = Path(temporary) / f"scene.{suffix}"
                    _render(path, "UNTRUSTED IMAGE TEXT", image_format=image_format)
                    source = path.read_bytes()
                    encoded, receipt = self._prepare_vision(path)
                    sanitized = Image.open(io.BytesIO(encoded))
                    sanitized.verify()
                    sanitized = Image.open(io.BytesIO(encoded))
                    self.assertEqual(sanitized.format, "PNG")
                    self.assertEqual(sanitized.size, (512, 111))
                    self.assertEqual(receipt["format"], "png")
                    self.assertEqual(
                        receipt["source_format"],
                        "png" if suffix == "png" else "jpeg")
                    self.assertEqual(
                        receipt["sanitizer"], "sandboxed-imagemagick")
                    self.assertEqual(receipt["limitations"],
                                     "single_image_question_answering")
                    self.assertEqual(receipt["source_width"], 1200)
                    self.assertEqual(receipt["source_height"], 260)
                    self.assertEqual(receipt["width"], 512)
                    self.assertEqual(receipt["height"], 111)
                    self.assertEqual(receipt["image_bytes"], len(encoded))
                    self.assertEqual(
                        receipt["image_sha256"], hashlib.sha256(encoded).hexdigest())
                    self.assertEqual(
                        receipt["source_sha256"], hashlib.sha256(source).hexdigest())

    @unittest.skipUnless(FONT.is_file(), "stable OCR test font is unavailable")
    def test_real_sandboxed_png_and_jpeg_ocr_returns_bound_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            for suffix, image_format in (("png", "PNG"), ("jpg", "JPEG")):
                with self.subTest(suffix=suffix):
                    path = Path(temporary) / f"canary.{suffix}"
                    _render(path, "ORBIT KEY 7294", image_format=image_format)
                    payload = path.read_bytes()
                    result = self._extract(path)
                    self.assertEqual(result["format"], "png" if suffix == "png" else "jpeg")
                    self.assertEqual(result["extractor"], "sandboxed-tesseract")
                    self.assertEqual(result["language"], "eng")
                    self.assertEqual(result["limitations"], "ocr_only")
                    self.assertEqual(result["width"], 1200)
                    self.assertEqual(result["height"], 260)
                    self.assertEqual(result["pixels"], 312_000)
                    self.assertEqual(result["text"], "ORBIT KEY 7294")
                    self.assertTrue(result["text_detected"])
                    self.assertEqual(
                        result["text_sha256"],
                        hashlib.sha256(b"ORBIT KEY 7294").hexdigest())
                    self.assertEqual(
                        result["source_sha256"], hashlib.sha256(payload).hexdigest())

    @unittest.skipUnless(FONT.is_file(), "stable OCR test font is unavailable")
    def test_blank_image_is_valid_evidence_of_no_detected_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "blank.png"
            _render(path, "", image_format="PNG")
            result = self._extract(path)
            self.assertEqual(result["text"], "")
            self.assertEqual(result["characters"], 0)
            self.assertFalse(result["text_detected"])
            self.assertEqual(
                result["text_sha256"], hashlib.sha256(b"").hexdigest())

    def test_pixel_bomb_and_malformed_or_mismatched_headers_fail_predecode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bomb = root / "bomb.png"
            bomb.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                + (100_000).to_bytes(4, "big") + (100_000).to_bytes(4, "big")
                + b"\x08\x02\x00\x00\x00" + b"\x00\x00\x00\x00")
            wrong = root / "wrong.png"
            wrong.write_bytes(b"not a png image")
            broken = root / "broken.jpg"
            broken.write_bytes(b"\xff\xd8\xff\xd9")
            unsupported = root / "image.webp"
            unsupported.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
            with patch("friday_core.images._sandboxed_tesseract") as decoder:
                for path in (bomb, wrong, broken, unsupported):
                    with self.subTest(path=path.name), self.assertRaises(ValueError):
                        self._extract(path)
                decoder.assert_not_called()

    @unittest.skipUnless(FONT.is_file(), "stable OCR test font is unavailable")
    def test_character_limit_hashes_only_the_returned_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "limit.png"
            _render(path, "ALPHA BRAVO CHARLIE", image_format="PNG")
            result = self._extract(path, max_chars=5)
            self.assertEqual(result["text"], "ALPHA")
            self.assertTrue(result["truncated"])
            self.assertEqual(result["text_sha256"], hashlib.sha256(b"ALPHA").hexdigest())


class ImageOperatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.project = root / "project"
        self.home = root / "home"
        self.state = root / "state"
        self.project.mkdir()
        self.home.mkdir()
        self.state.mkdir()
        graph = GraphStore(self.state / "friday.db")
        self.grants = OperatorGrantService(
            graph, self.project, home=self.home, state_root=self.state)
        self.operator = MachineOperator(self.grants, state_root=self.state)

    def tearDown(self):
        self.temporary.cleanup()

    @unittest.skipUnless(FONT.is_file(), "stable OCR test font is unavailable")
    def test_ocr_requires_read_grant_rejects_symlink_and_detects_mutation(self):
        scope = self.project / "images"
        scope.mkdir()
        target = scope / "receipt.png"
        _render(target, "SAFE IMAGE 314", image_format="PNG")
        alias = scope / "alias.png"
        alias.symlink_to(target)
        with self.assertRaises(PermissionError):
            self.operator.ocr_image(target)
        grant = self.grants.grant_path(scope, ["read"])
        receipt = self.operator.ocr_image(target)
        self.assertEqual(receipt["grant_id"], grant["grant_id"])
        self.assertEqual(receipt["path"], str(target.resolve()))
        self.assertEqual(receipt["text"], "SAFE IMAGE 314")
        with self.assertRaises(PermissionError):
            self.operator.ocr_image(alias)

        original = extract_image_text

        def mutate(fd, filename, *, max_chars):
            result = original(fd, filename, max_chars=max_chars)
            payload = target.read_bytes()
            target.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
            return result

        with patch("friday_core.machine.extract_image_text", side_effect=mutate):
            with self.assertRaises(RuntimeError):
                self.operator.ocr_image(target)

    def test_schema_verifier_and_private_redaction_cover_ocr(self):
        schema = {item["function"]["name"]: item for item in server.TOOL_SCHEMA}
        self.assertIn("machine_ocr_image", schema)
        self.assertEqual(
            redact_tool_arguments("machine_ocr_image", {"path": "/private/a.png"})["path"],
            "[REDACTED]")
        redacted = redact_tool_result(
            "machine_ocr_image", {"status": "ok", "text": "private OCR"})
        self.assertTrue(redacted["_redacted"])
        self.assertNotIn("private OCR", str(redacted))

        text = "VISIBLE 42"
        receipt = {
            "status": "ok", "verified": True,
            "grant_id": "grant_image_0001", "path": "/tmp/a.png",
            "format": "png", "extractor": "sandboxed-tesseract",
            "language": "eng", "limitations": "ocr_only",
            "width": 100, "height": 50, "pixels": 5_000,
            "text": text, "characters": len(text),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text_detected": True, "source_bytes": 500,
            "source_sha256": "a" * 64, "truncated": False,
        }
        valid = server.OUTCOMES.verify_action(
            "machine_ocr_image", receipt, succeeded=True,
            args={"path": "/tmp/a.png"})
        self.assertTrue(valid.passed)
        for field, forged_value in (
                ("pixels", 4_999), ("text_sha256", "0" * 64),
                ("limitations", "general_vision"), ("text_detected", False),
                ("characters", True)):
            forged = dict(receipt)
            forged[field] = forged_value
            with self.subTest(field=field):
                check = server.OUTCOMES.verify_action(
                    "machine_ocr_image", forged, succeeded=True,
                    args={"path": "/tmp/a.png"})
                self.assertFalse(check.passed)

    def test_native_vision_snapshot_requires_grant_and_never_reprs_bytes(self):
        scope = self.project / "vision"
        scope.mkdir()
        target = scope / "scene.png"
        _render(target, "PRIVATE SCENE", image_format="PNG")
        with self.assertRaises(PermissionError):
            self.operator.native_vision_image(target, max_side=512)
        grant = self.grants.grant_path(scope, ["read"])
        prepared = self.operator.native_vision_image(target, max_side=512)
        self.assertEqual(prepared.grant_id, grant["grant_id"])
        self.assertEqual(prepared.path, str(target.resolve()))
        self.assertTrue(prepared.encoded.startswith(b"\x89PNG"))
        self.assertNotIn("encoded", repr(prepared))
        self.assertNotIn(str(target), repr(prepared))

    def test_native_vision_schema_gating_redaction_and_receipt_verifier(self):
        all_schema = {
            item["function"]["name"]: item for item in server.TOOL_SCHEMA}
        active_schema = {
            item["function"]["name"] for item in server.current_tool_schema()}
        self.assertIn("machine_understand_image", all_schema)
        self.assertNotIn("machine_understand_image", active_schema)
        with patch.object(
                server, "_native_vision_qualified", return_value=True):
            self.assertIn("machine_understand_image", {
                item["function"]["name"]
                for item in server.current_tool_schema()})
        redacted = redact_tool_arguments("machine_understand_image", {
            "path": "/private/scene.png", "question": "What is private?"})
        self.assertEqual(redacted["path"], "[REDACTED]")
        self.assertEqual(redacted["question"], "[REDACTED]")

        question = "Which shape is left?"
        answer = "The red square."
        receipt = {
            "status": "ok", "verified": True,
            "grant_id": "grant_vision_0001", "path": "/tmp/scene.png",
            "format": "png", "source_format": "jpeg",
            "sanitizer": "sandboxed-imagemagick",
            "limitations": "single_image_question_answering",
            "width": 512, "height": 256, "pixels": 131_072,
            "max_side": 512, "source_width": 1200, "source_height": 600,
            "source_pixels": 720_000, "source_bytes": 50_000,
            "source_sha256": "a" * 64, "image_bytes": 20_000,
            "image_sha256": "b" * 64,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "answer": answer, "answer_characters": len(answer),
            "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "model": "vision-model", "runtime_fingerprint": "c" * 64,
        }
        valid = server.OUTCOMES.verify_action(
            "machine_understand_image", receipt, succeeded=True,
            args={"path": "/tmp/scene.png", "question": question})
        self.assertTrue(valid.passed)
        for field, forged in (
                ("answer_sha256", "0" * 64), ("pixels", 1),
                ("question_sha256", "0" * 64),
                ("limitations", "general_vision"), ("max_side", 128)):
            bad = dict(receipt)
            bad[field] = forged
            with self.subTest(field=field):
                self.assertFalse(server.OUTCOMES.verify_action(
                    "machine_understand_image", bad, succeeded=True,
                    args={"path": "/tmp/scene.png",
                          "question": question}).passed)
