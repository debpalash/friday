"""Bounded, non-executing text extraction for granted local documents."""

from __future__ import annotations

import hashlib
import os
import posixpath
import re
import signal
import stat
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Any


MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4096
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024
MAX_XML_BYTES = 8 * 1024 * 1024
MAX_EXTRACTED_BYTES = 1_000_000
MAX_DOCUMENT_CHARS = 250_000
MAX_EPUB_SPINE_ITEMS = 256
SUPPORTED_DOCUMENT_FORMATS = frozenset({
    "pdf", "docx", "odt", "epub", "pptx", "xlsx",
})
_SAFE_ZIP_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
_SLIDE = re.compile(r"ppt/slides/slide([1-9][0-9]*)\.xml\Z")
_SHEET = re.compile(r"xl/worksheets/sheet([1-9][0-9]*)\.xml\Z")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def document_source_sha256(fd: int, size: int) -> str:
    if (isinstance(size, bool) or not isinstance(size, int)
            or not 0 <= size <= MAX_DOCUMENT_BYTES):
        raise ValueError("document exceeds the source-size limit")
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(65_536, size - offset), offset)
        if not chunk:
            raise RuntimeError("document became shorter while being hashed")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(fd, 1, size):
        raise RuntimeError("document grew while being hashed")
    return digest.hexdigest()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml(payload: bytes) -> ET.Element:
    if len(payload) > MAX_XML_BYTES:
        raise ValueError("document XML exceeds the member-size limit")
    # Reject alternate-width encodings before declaration scanning so a UTF-16
    # DTD cannot hide its entity declarations between NUL bytes.
    if b"\x00" in payload:
        raise ValueError("document XML must use a byte-compatible UTF-8 encoding")
    if re.search(br"<!\s*(?:doctype|entity)\b", payload, re.IGNORECASE):
        raise ValueError("document XML declarations are not accepted")
    try:
        return ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError("document contains malformed XML") from exc


def _walk_text(node: ET.Element, output: list[str], *,
               text_tags: frozenset[str], paragraph_tags: frozenset[str]) -> None:
    local = _local_name(node.tag)
    if local in text_tags and node.text:
        output.append(node.text)
    elif local in {"tab"}:
        output.append("\t")
    elif local in {"br", "line-break"}:
        output.append("\n")
    for child in node:
        _walk_text(child, output, text_tags=text_tags,
                   paragraph_tags=paragraph_tags)
        if child.tail and local in text_tags:
            output.append(child.tail)
    if local in paragraph_tags:
        output.append("\n")


def _xml_text(payload: bytes, *, text_tags: frozenset[str],
              paragraph_tags: frozenset[str]) -> str:
    output: list[str] = []
    _walk_text(_xml(payload), output, text_tags=text_tags,
               paragraph_tags=paragraph_tags)
    return "".join(output)


class _HTMLText(HTMLParser):
    _SKIP = frozenset({"script", "style", "svg", "template", "noscript"})
    _BLOCK = frozenset({
        "address", "article", "aside", "blockquote", "br", "div", "footer",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main",
        "nav", "ol", "p", "pre", "section", "table", "tr", "ul",
    })

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.output: list[str] = []

    def handle_starttag(self, tag: str, _attrs) -> None:
        lowered = tag.casefold()
        if lowered in self._SKIP:
            self.skip_depth += 1
        elif not self.skip_depth and lowered in self._BLOCK:
            self.output.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in self._SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif not self.skip_depth and lowered in self._BLOCK:
            self.output.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.output.append(data)


def _normalize_text(value: str, max_chars: int) -> tuple[str, bool]:
    value = _CONTROL.sub(" ", value.replace("\r\n", "\n").replace("\r", "\n"))
    lines: list[str] = []
    blank = False
    for raw in value.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if line:
            lines.append(line)
            blank = False
        elif lines and not blank:
            lines.append("")
            blank = True
    normalized = "\n".join(lines).strip()
    if not normalized:
        raise ValueError("document contains no extractable text")
    truncated = len(normalized) > max_chars
    return normalized[:max_chars], truncated


class _Archive:
    def __init__(self, fd: int):
        self._stream = os.fdopen(os.dup(fd), "rb")
        try:
            self.zip = zipfile.ZipFile(self._stream)
            self.members = self._validate(self.zip.infolist())
        except Exception:
            self._stream.close()
            raise

    @staticmethod
    def _validate(infos: list[zipfile.ZipInfo]) -> dict[str, zipfile.ZipInfo]:
        if not 1 <= len(infos) <= MAX_ARCHIVE_MEMBERS:
            raise ValueError("document archive member count is invalid")
        members: dict[str, zipfile.ZipInfo] = {}
        total = 0
        for info in infos:
            name = info.filename
            path = PurePosixPath(name)
            if (not name or len(name) > 512 or "\\" in name or "\x00" in name
                    or name.startswith("/") or any(
                        part in {"", ".", ".."} for part in path.parts)
                    or name in members):
                raise ValueError("document archive contains an unsafe member name")
            if info.flag_bits & 0x1:
                raise ValueError("encrypted document archive members are unsupported")
            if info.compress_type not in _SAFE_ZIP_COMPRESSION:
                raise ValueError("document archive compression is unsupported")
            if (info.file_size < 0 or info.compress_size < 0
                    or info.file_size > MAX_ARCHIVE_MEMBER_BYTES):
                raise ValueError("document archive member exceeds the size limit")
            total += info.file_size
            if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ValueError("document archive exceeds the expansion limit")
            if (info.file_size > 1_000_000
                    and info.file_size > max(info.compress_size, 1) * 1000):
                raise ValueError("document archive has an unsafe expansion ratio")
            members[name] = info
        return members

    def read(self, name: str, *, maximum: int = MAX_XML_BYTES) -> bytes:
        info = self.members.get(name)
        if info is None or info.is_dir():
            raise ValueError(f"document archive is missing {name}")
        if info.file_size > maximum:
            raise ValueError("document archive member exceeds the extraction limit")
        try:
            with self.zip.open(info) as stream:
                payload = stream.read(maximum + 1)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise ValueError("document archive member failed integrity checks") from exc
        if len(payload) != info.file_size or len(payload) > maximum:
            raise ValueError("document archive member size is inconsistent")
        return payload

    def close(self) -> None:
        try:
            self.zip.close()
        finally:
            self._stream.close()

    def __enter__(self) -> "_Archive":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _docx(archive: _Archive) -> str:
    content_types = archive.read("[Content_Types].xml")
    if b"wordprocessingml.document.main+xml" not in content_types:
        raise ValueError("archive is not a DOCX document")
    names = ["word/document.xml"] + sorted(
        name for name in archive.members
        if re.fullmatch(r"word/(?:header|footer)[1-9][0-9]*\.xml", name))
    return "\n".join(_xml_text(
        archive.read(name), text_tags=frozenset({"t"}),
        paragraph_tags=frozenset({"p", "tr"})) for name in names)


def _odt(archive: _Archive) -> str:
    if archive.read("mimetype", maximum=256).decode(
            "ascii", errors="strict") != "application/vnd.oasis.opendocument.text":
        raise ValueError("archive is not an ODT document")
    return _xml_text(
        archive.read("content.xml"),
        text_tags=frozenset({"p", "h", "span"}),
        paragraph_tags=frozenset({"p", "h", "list-item"}))


def _pptx(archive: _Archive) -> str:
    content_types = archive.read("[Content_Types].xml")
    if b"presentationml.presentation.main+xml" not in content_types:
        raise ValueError("archive is not a PPTX document")
    slides = sorted(
        ((int(match.group(1)), name) for name in archive.members
         if (match := _SLIDE.fullmatch(name))), key=lambda item: item[0])
    if not slides or len(slides) > 10_000:
        raise ValueError("presentation slide inventory is invalid")
    return "\n\n".join(
        f"Slide {number}\n" + _xml_text(
            archive.read(name), text_tags=frozenset({"t"}),
            paragraph_tags=frozenset({"p"}))
        for number, name in slides)


def _xlsx(archive: _Archive) -> str:
    content_types = archive.read("[Content_Types].xml")
    if b"spreadsheetml.sheet.main+xml" not in content_types:
        raise ValueError("archive is not an XLSX document")
    shared: list[str] = []
    if "xl/sharedStrings.xml" in archive.members:
        root = _xml(archive.read("xl/sharedStrings.xml"))
        for item in root.iter():
            if _local_name(item.tag) == "si":
                shared.append("".join(
                    str(node.text or "") for node in item.iter()
                    if _local_name(node.tag) == "t"))
    sheets = sorted(
        ((int(match.group(1)), name) for name in archive.members
         if (match := _SHEET.fullmatch(name))), key=lambda item: item[0])
    if not sheets or len(sheets) > 10_000:
        raise ValueError("workbook sheet inventory is invalid")
    output: list[str] = []
    for number, name in sheets:
        output.append(f"Sheet {number}")
        root = _xml(archive.read(name))
        for cell in (node for node in root.iter()
                     if _local_name(node.tag) == "c"):
            reference = str(cell.attrib.get("r") or "?")[:32]
            cell_type = cell.attrib.get("t")
            value_node = next((node for node in cell
                               if _local_name(node.tag) == "v"), None)
            formula_node = next((node for node in cell
                                 if _local_name(node.tag) == "f"), None)
            if cell_type == "inlineStr":
                value = "".join(str(node.text or "") for node in cell.iter()
                                if _local_name(node.tag) == "t")
            else:
                value = str(value_node.text or "") if value_node is not None else ""
                if cell_type == "s" and value:
                    try:
                        value = shared[int(value)]
                    except (ValueError, IndexError):
                        raise ValueError("workbook shared-string index is invalid")
                elif cell_type == "b" and value in {"0", "1"}:
                    value = "TRUE" if value == "1" else "FALSE"
            formula = (str(formula_node.text or "").strip()
                       if formula_node is not None else "")
            if value or formula:
                output.append(
                    f"{reference}: {value}" +
                    (f" [formula: {formula}]" if formula else ""))
        output.append("")
    return "\n".join(output)


def _safe_archive_path(base: str, href: str) -> str:
    if not href or "\\" in href or "\x00" in href:
        raise ValueError("EPUB contains an unsafe content path")
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(base), href))
    if joined.startswith("../") or joined.startswith("/") or joined == "..":
        raise ValueError("EPUB content escapes its archive root")
    return joined


def _epub(archive: _Archive) -> str:
    if archive.read("mimetype", maximum=256).decode(
            "ascii", errors="strict") != "application/epub+zip":
        raise ValueError("archive is not an EPUB document")
    container = _xml(archive.read("META-INF/container.xml"))
    rootfiles = [str(node.attrib.get("full-path") or "")
                 for node in container.iter()
                 if _local_name(node.tag) == "rootfile"]
    if len(rootfiles) != 1:
        raise ValueError("EPUB package inventory is ambiguous")
    package_path = _safe_archive_path("", rootfiles[0])
    package = _xml(archive.read(package_path))
    manifest: dict[str, tuple[str, str]] = {}
    for node in package.iter():
        if _local_name(node.tag) != "item":
            continue
        item_id = str(node.attrib.get("id") or "")
        href = str(node.attrib.get("href") or "")
        media_type = str(node.attrib.get("media-type") or "")
        if not item_id or item_id in manifest:
            raise ValueError("EPUB manifest identifiers are invalid")
        manifest[item_id] = (_safe_archive_path(package_path, href), media_type)
    spine = [str(node.attrib.get("idref") or "") for node in package.iter()
             if _local_name(node.tag) == "itemref"]
    if not 1 <= len(spine) <= MAX_EPUB_SPINE_ITEMS:
        raise ValueError("EPUB spine is missing or too large")
    output: list[str] = []
    for item_id in spine:
        item = manifest.get(item_id)
        if item is None or item[1] not in {
                "application/xhtml+xml", "text/html"}:
            raise ValueError("EPUB spine references unsupported content")
        payload = archive.read(item[0], maximum=MAX_XML_BYTES)
        if b"<!doctype" in payload.lower() or b"<!entity" in payload.lower():
            raise ValueError("EPUB declarations are not accepted")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("EPUB content must be UTF-8") from exc
        parser = _HTMLText()
        parser.feed(text)
        parser.close()
        output.append("".join(parser.output))
    return "\n\n".join(output)


def _pdf_page_count(text: str) -> int:
    breaks = text.count("\f")
    # Poppler normally terminates every page, including the last, with a form
    # feed. If a version omits the final delimiter, the remaining text is one
    # additional page.
    terminal = text.rstrip(" \t\r\n").endswith("\f")
    return max(1, breaks if terminal else breaks + 1)


def _pdf(fd: int) -> tuple[str, int]:
    if not os.path.isfile("/usr/bin/pdftotext"):
        raise RuntimeError("the pinned PDF extractor is unavailable")
    if not os.path.isfile("/usr/bin/bwrap"):
        raise RuntimeError("the PDF sandbox is unavailable")
    # A named private file in tmpfs gives Bubblewrap a bindable output without
    # ever placing extracted plaintext on persistent storage. Only this exact
    # file is writable from the sandbox; the surrounding host directory is not.
    if not os.path.isdir("/dev/shm"):
        raise RuntimeError("the private in-memory PDF output filesystem is unavailable")
    with tempfile.TemporaryDirectory(
            prefix="friday-pdf-", dir="/dev/shm") as output_directory:
        os.chmod(output_directory, 0o700)
        output_path = os.path.join(output_directory, "output.txt")
        output_fd = os.open(
            output_path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0), 0o600)
        error_file = tempfile.TemporaryFile(dir="/dev/shm")
        command = [
            "/usr/bin/bwrap", "--die-with-parent", "--new-session",
            "--unshare-all", "--ro-bind", "/usr", "/usr", "--symlink",
            "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64", "--proc",
            "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind-try",
            "/etc/fonts", "/etc/fonts", "--clearenv", "--setenv", "HOME",
            "/tmp", "--setenv", "PATH", "/usr/bin:/bin", "--ro-bind-fd",
            str(fd), "/input.pdf", "--bind-fd", str(output_fd), "/output.txt",
            "--chdir", "/tmp", "/usr/bin/prlimit",
            f"--fsize={MAX_EXTRACTED_BYTES}", "--cpu=20", "--as=1073741824",
            "--nofile=64", "--nproc=32", "--", "/usr/bin/pdftotext", "-enc",
            "UTF-8", "-layout", "/input.pdf", "/output.txt",
        ]
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=error_file, pass_fds=(fd, output_fd),
                start_new_session=True, env={"PATH": "/usr/bin:/bin"})
            try:
                returncode = process.wait(timeout=25)
            except subprocess.TimeoutExpired as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                finally:
                    process.wait()
                raise RuntimeError("PDF extraction exceeded its time limit") from exc
            size = os.fstat(output_fd).st_size
            if returncode or size > MAX_EXTRACTED_BYTES:
                error_file.seek(0)
                detail = error_file.read(4096).decode(
                    "utf-8", errors="replace")
                raise ValueError(
                    (detail.strip() or "PDF extraction failed")[-500:])
            payload = os.pread(output_fd, size, 0)
        finally:
            error_file.close()
            os.close(output_fd)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("PDF extractor returned invalid UTF-8") from exc
    return text, _pdf_page_count(text)


def extract_document(fd: int, filename: str, *, max_chars: int = 80_000
                     ) -> dict[str, Any]:
    if (isinstance(max_chars, bool) or not isinstance(max_chars, int)
            or not 1 <= max_chars <= MAX_DOCUMENT_CHARS):
        raise ValueError(
            f"document character limit must be between 1 and {MAX_DOCUMENT_CHARS}")
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("only regular document files can be read")
    if not 1 <= info.st_size <= MAX_DOCUMENT_BYTES:
        raise ValueError("document source size is outside the safety limit")
    format_name = PurePosixPath(filename).suffix.casefold().lstrip(".")
    if format_name not in SUPPORTED_DOCUMENT_FORMATS:
        raise ValueError("unsupported document format")
    source_hash = document_source_sha256(fd, int(info.st_size))
    signature = os.pread(fd, 8, 0)
    pages: int | None = None
    if format_name == "pdf":
        if not signature.startswith(b"%PDF-"):
            raise ValueError("file extension and PDF signature disagree")
        raw_text, pages = _pdf(fd)
        extractor = "sandboxed-poppler"
    else:
        if not signature.startswith(b"PK\x03\x04"):
            raise ValueError("file extension and document archive disagree")
        try:
            with _Archive(fd) as archive:
                raw_text = {
                    "docx": _docx,
                    "odt": _odt,
                    "epub": _epub,
                    "pptx": _pptx,
                    "xlsx": _xlsx,
                }[format_name](archive)
        except zipfile.BadZipFile as exc:
            raise ValueError("document archive is invalid") from exc
        extractor = "bounded-archive-xml"
    text, truncated = _normalize_text(raw_text, max_chars)
    result: dict[str, Any] = {
        "format": format_name,
        "extractor": extractor,
        "text": text,
        "characters": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_bytes": int(info.st_size),
        "source_sha256": source_hash,
        "truncated": truncated,
    }
    if pages is not None:
        result["pages"] = pages
    return result
