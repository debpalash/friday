import hashlib
import io
import os
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import server
from friday_core.documents import _pdf_page_count, extract_document
from friday_core.graph import GraphStore
from friday_core.machine import MachineOperator, OperatorGrantService
from friday_core.tasks import redact_tool_arguments, redact_tool_result


def _zip(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return output.getvalue()


FORMATS = {
    "docx": _zip([
        ("[Content_Types].xml",
         b'<Types><Override ContentType="application/vnd.openxmlformats-'
         b'officedocument.wordprocessingml.document.main+xml"/></Types>'),
        ("word/document.xml",
         b'<w:document xmlns:w="w"><w:body><w:p><w:r><w:t>Hello DOCX'
         b'</w:t></w:r></w:p></w:body></w:document>'),
    ]),
    "odt": _zip([
        ("mimetype", b"application/vnd.oasis.opendocument.text"),
        ("content.xml",
         b'<office:document xmlns:office="o" xmlns:text="t"><text:p>Hello '
         b'<text:span>ODT</text:span></text:p></office:document>'),
    ]),
    "pptx": _zip([
        ("[Content_Types].xml",
         b'<Types><Override ContentType="application/vnd.openxmlformats-'
         b'officedocument.presentationml.presentation.main+xml"/></Types>'),
        ("ppt/slides/slide1.xml",
         b'<p:sld xmlns:p="p" xmlns:a="a"><a:p><a:r><a:t>Hello PPTX'
         b'</a:t></a:r></a:p></p:sld>'),
    ]),
    "xlsx": _zip([
        ("[Content_Types].xml",
         b'<Types><Override ContentType="application/vnd.openxmlformats-'
         b'officedocument.spreadsheetml.sheet.main+xml"/></Types>'),
        ("xl/sharedStrings.xml",
         b'<sst xmlns="x"><si><t>Hello XLSX</t></si></sst>'),
        ("xl/worksheets/sheet1.xml",
         b'<worksheet xmlns="x"><sheetData><row><c r="A1" t="s"><v>0</v>'
         b'</c><c r="B1"><f>1+1</f><v>2</v></c></row></sheetData>'
         b'</worksheet>'),
    ]),
    "epub": _zip([
        ("mimetype", b"application/epub+zip"),
        ("META-INF/container.xml",
         b'<container><rootfiles><rootfile full-path="OPS/package.opf"/>'
         b'</rootfiles></container>'),
        ("OPS/package.opf",
         b'<package><manifest><item id="chapter" href="chapter.xhtml" '
         b'media-type="application/xhtml+xml"/></manifest><spine><itemref '
         b'idref="chapter"/></spine></package>'),
        ("OPS/chapter.xhtml",
         b'<html><body><h1>Hello EPUB</h1><script>not extracted</script>'
         b'<p>Chapter text</p></body></html>'),
    ]),
}


class DocumentExtractionTests(unittest.TestCase):
    def _extract(self, suffix, payload, *, max_chars=80_000):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / f"sample.{suffix}"
            path.write_bytes(payload)
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            try:
                return extract_document(fd, path.name, max_chars=max_chars)
            finally:
                os.close(fd)

    def test_supported_archive_formats_return_bounded_hashed_text(self):
        expected = {
            "docx": "Hello DOCX", "odt": "Hello ODT",
            "pptx": "Slide 1\nHello PPTX",
            "xlsx": "Sheet 1\nA1: Hello XLSX\nB1: 2 [formula: 1+1]",
            "epub": "Hello EPUB\n\nChapter text",
        }
        for suffix, payload in FORMATS.items():
            with self.subTest(suffix=suffix):
                result = self._extract(suffix, payload)
                self.assertEqual(result["format"], suffix)
                self.assertEqual(result["extractor"], "bounded-archive-xml")
                self.assertEqual(result["text"], expected[suffix])
                self.assertEqual(result["characters"], len(expected[suffix]))
                self.assertEqual(
                    result["text_sha256"],
                    hashlib.sha256(expected[suffix].encode()).hexdigest())
                self.assertEqual(result["source_bytes"], len(payload))
                self.assertEqual(
                    result["source_sha256"], hashlib.sha256(payload).hexdigest())
                self.assertFalse(result["truncated"])

    def test_character_limit_reports_truncation_and_hashes_returned_text(self):
        result = self._extract("docx", FORMATS["docx"], max_chars=5)
        self.assertEqual(result["text"], "Hello")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["text_sha256"], hashlib.sha256(b"Hello").hexdigest())

    def test_pdf_page_count_handles_poppler_terminal_form_feed(self):
        self.assertEqual(_pdf_page_count("page one\f"), 1)
        self.assertEqual(_pdf_page_count("page one\fpage two\f\n"), 2)
        self.assertEqual(_pdf_page_count("page one\fpage two"), 2)
        self.assertEqual(_pdf_page_count(""), 1)

    def test_archive_path_duplicates_entities_and_signatures_are_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            duplicate = _zip([
                ("[Content_Types].xml", b"x"),
                ("[Content_Types].xml", b"y")])
        malicious = {
            "traversal": _zip([
                ("[Content_Types].xml", b"x"), ("../escape.xml", b"x")]),
            "duplicate": duplicate,
            "entity": _zip([
                ("[Content_Types].xml",
                 b'<Types><Override ContentType="wordprocessingml.document.main+xml"/>'
                 b'</Types>'),
                ("word/document.xml",
                 b'<!DOCTYPE x [<!ENTITY leak SYSTEM "file:///etc/passwd">]>'
                 b'<document><p><t>&leak;</t></p></document>'),
            ]),
        }
        for label, payload in malicious.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                self._extract("docx", payload)
        with self.assertRaises(ValueError):
            self._extract("docx", b"not a zip")
        with self.assertRaises(ValueError):
            self._extract("pdf", b"not a pdf")

    def test_epub_content_cannot_escape_its_package_root(self):
        payload = _zip([
            ("mimetype", b"application/epub+zip"),
            ("META-INF/container.xml",
             b'<container><rootfile full-path="book/package.opf"/></container>'),
            ("book/package.opf",
             b'<package><manifest><item id="x" href="../../escape.xhtml" '
             b'media-type="application/xhtml+xml"/></manifest><spine>'
             b'<itemref idref="x"/></spine></package>'),
        ])
        with self.assertRaises(ValueError):
            self._extract("epub", payload)

    def test_high_ratio_archive_member_is_rejected_before_xml_parsing(self):
        payload = _zip([
            ("[Content_Types].xml", b"A" * 1_100_000),
            ("word/document.xml", b"<document><p><t>x</t></p></document>"),
        ])
        with self.assertRaisesRegex(ValueError, "expansion ratio"):
            self._extract("docx", payload)


class DocumentOperatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name).resolve()
        self.project = root / "project"
        self.state = root / "state"
        self.project.mkdir()
        self.state.mkdir()
        graph = GraphStore(self.state / "friday.db")
        self.grants = OperatorGrantService(
            graph, self.project, home=root / "home", state_root=self.state)
        self.operator = MachineOperator(self.grants, state_root=self.state)

    def tearDown(self):
        self.temporary.cleanup()

    def test_document_requires_exact_read_grant_and_returns_verified_receipt(self):
        scope = self.project / "documents"
        scope.mkdir()
        target = scope / "sample.docx"
        target.write_bytes(FORMATS["docx"])
        grant = self.grants.grant_path(scope, ["read"])

        receipt = self.operator.read_document(target)

        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["grant_id"], grant["grant_id"])
        self.assertEqual(receipt["path"], str(target.resolve()))
        self.assertEqual(receipt["text"], "Hello DOCX")

    def test_document_symlinks_and_source_mutation_are_rejected(self):
        scope = self.project / "documents"
        scope.mkdir()
        target = scope / "sample.docx"
        target.write_bytes(FORMATS["docx"])
        alias = scope / "alias.docx"
        alias.symlink_to(target)
        self.grants.grant_path(scope, ["read"])
        with self.assertRaises(PermissionError):
            self.operator.read_document(alias)

        original = extract_document

        def mutate(fd, filename, *, max_chars):
            result = original(fd, filename, max_chars=max_chars)
            target.write_bytes(b"X" + FORMATS["docx"][1:])
            return result

        with patch("friday_core.machine.extract_document", side_effect=mutate):
            with self.assertRaises(RuntimeError):
                self.operator.read_document(target)

    def test_schema_verifier_and_private_redaction_cover_documents(self):
        schema = {item["function"]["name"]: item for item in server.TOOL_SCHEMA}
        self.assertIn("machine_read_document", schema)
        self.assertEqual(
            redact_tool_arguments("machine_read_document", {"path": "/secret/a.docx"})["path"],
            "[REDACTED]")
        redacted = redact_tool_result(
            "machine_read_document", {"status": "ok", "text": "private"})
        self.assertTrue(redacted["_redacted"])
        self.assertNotIn("private", str(redacted))

        receipt = {
            "status": "ok", "verified": True,
            "grant_id": "grant_document_0001", "path": "/tmp/a.docx",
            "format": "docx", "extractor": "bounded-archive-xml",
            "text": "Hello", "characters": 5,
            "text_sha256": hashlib.sha256(b"Hello").hexdigest(),
            "source_bytes": 100, "source_sha256": "a" * 64,
            "truncated": False,
        }
        valid = server.OUTCOMES.verify_action(
            "machine_read_document", receipt, succeeded=True,
            args={"path": "/tmp/a.docx"})
        self.assertTrue(valid.passed)
        for field, value in (("text_sha256", "0" * 64), ("characters", True)):
            forged_receipt = dict(receipt)
            forged_receipt[field] = value
            forged = server.OUTCOMES.verify_action(
                "machine_read_document", forged_receipt, succeeded=True,
                args={"path": "/tmp/a.docx"})
            self.assertFalse(forged.passed)
