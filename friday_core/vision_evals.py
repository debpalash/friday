"""Artifact-backed, exact-graded native-vision evaluation.

Every case renders a deterministic text-free scene, passes it through Friday's
real networkless image sanitizer, and gives only the canonical PNG to the
profile-bound local model. Raw model answers are hashed rather than journaled.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

from .graph import GraphStore, canonical_json, utc_now
from .images import prepare_native_vision_image


MAX_VISION_SUITE_BYTES = 128_000
MAX_VISION_CASES = 32
MAX_VISION_OUTPUT_CHARS = 256
_SCENES = {
    "left_color_shape": (
        "Which colored shape is farther left? Reply with exactly two words.",
        "red square"),
    "count_green_triangles": (
        "How many green triangles are visible? Reply with only the number.", "3"),
    "triangle_color_binding": (
        "What color is the triangle? Reply with only the color.", "yellow"),
    "shape_inside_frame": (
        "Which colored shape is inside the black frame? Reply with exactly two words.",
        "purple circle"),
    "larger_colored_circle": (
        "Which colored circle is larger? Reply with exactly two words.",
        "red circle"),
}


def _normalized(value: str) -> str:
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _render_scene(name: str) -> bytes:
    if name not in _SCENES:
        raise ValueError("native-vision scene is not allowlisted")
    image = Image.new("RGB", (640, 400), "white")
    draw = ImageDraw.Draw(image)
    if name == "left_color_shape":
        draw.rectangle((55, 125, 205, 275), fill="#e52521")
        draw.ellipse((405, 125, 555, 275), fill="#1e5ad7")
    elif name == "count_green_triangles":
        for center_x, center_y in ((120, 115), (320, 115), (220, 285)):
            draw.polygon(((center_x, center_y - 65),
                          (center_x - 65, center_y + 55),
                          (center_x + 65, center_y + 55)), fill="#159447")
        draw.ellipse((455, 55, 555, 155), fill="#ec7f18")
        draw.ellipse((455, 245, 555, 345), fill="#ec7f18")
    elif name == "triangle_color_binding":
        draw.polygon(((120, 55), (45, 205), (195, 205)), fill="#f2c112")
        draw.rectangle((260, 90, 410, 240), fill="#7b3fbb")
        draw.ellipse((470, 90, 620, 240), fill="#19a7ae")
    elif name == "shape_inside_frame":
        draw.rectangle((65, 55, 390, 345), outline="black", width=18)
        draw.ellipse((165, 130, 290, 255), fill="#7b3fbb")
        draw.polygon(((520, 115), (445, 285), (595, 285)), fill="#ec7f18")
    else:
        draw.ellipse((65, 145, 175, 255), fill="#1e5ad7")
        draw.ellipse((330, 60, 610, 340), fill="#e52521")
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    return stream.getvalue()


def has_qualified_native_vision_score(
        graph: GraphStore, *, model: str, runtime_fingerprint: str,
        max_side: int) -> bool:
    """Verify one append-only exact-fingerprint five-scene pass."""
    if (not isinstance(model, str) or not 1 <= len(model) <= 160
            or re.fullmatch(r"[0-9a-f]{64}", runtime_fingerprint) is None
            or isinstance(max_side, bool) or not isinstance(max_side, int)
            or not 256 <= max_side <= 4096):
        return False
    try:
        with graph._connect() as conn:
            rows = conn.execute(
                "SELECT body_json,body_sha256 FROM nodes "
                "WHERE kind='native_vision_evaluation_run' "
                "ORDER BY rowid DESC LIMIT 16").fetchall()
        for row in rows:
            encoded = str(row["body_json"])
            if not 2 <= len(encoded.encode("utf-8")) <= 256_000:
                continue
            body = json.loads(encoded)
            if (hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
                    != row["body_sha256"]):
                continue
            results = body.get("results") if isinstance(body, dict) else None
            if (body.get("suite") != "friday-native-vision"
                    or body.get("version") != 1
                    or body.get("runtime_fingerprint") != runtime_fingerprint
                    or body.get("model") != model
                    or body.get("max_side") != max_side
                    or body.get("passed") != len(_SCENES)
                    or body.get("total") != len(_SCENES)
                    or body.get("pass_rate") != 1.0
                    or not isinstance(results, list)
                    or len(results) != len(_SCENES)
                    or {item.get("scene") for item in results
                        if isinstance(item, dict)} != set(_SCENES)
                    or not all(
                        isinstance(item, dict) and item.get("passed") is True
                        and item.get("sanitizer") == "sandboxed-imagemagick"
                        and all(re.fullmatch(r"[0-9a-f]{64}", str(
                            item.get(field) or "")) is not None for field in (
                                "source_sha256", "image_sha256",
                                "output_sha256"))
                        for item in results)):
                continue
            return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return False


class NativeVisionEvalRunner:
    def __init__(self, graph: GraphStore,
                 complete: Callable[[str, bytes], str], *, model: str,
                 runtime_fingerprint: str, max_side: int):
        if not callable(complete):
            raise TypeError("native-vision evaluator requires a completion callback")
        if not isinstance(model, str) or not 1 <= len(model) <= 160:
            raise ValueError("native-vision evaluator model identity is invalid")
        if (not isinstance(runtime_fingerprint, str)
                or re.fullmatch(r"[0-9a-f]{64}", runtime_fingerprint) is None):
            raise ValueError("native-vision runtime fingerprint is invalid")
        if (isinstance(max_side, bool) or not isinstance(max_side, int)
                or not 256 <= max_side <= 4096):
            raise ValueError("native-vision evaluation side limit is invalid")
        self.graph = graph
        self.complete = complete
        self.model = model
        self.runtime_fingerprint = runtime_fingerprint
        self.max_side = max_side

    @staticmethod
    def _load_suite(suite_path: str | Path) -> dict[str, Any]:
        try:
            descriptor = os.open(
                Path(suite_path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if (not stat.S_ISREG(metadata.st_mode)
                        or not 2 <= metadata.st_size <= MAX_VISION_SUITE_BYTES):
                    raise ValueError(
                        "native-vision suite must be a bounded regular file")
                encoded = stream.read(MAX_VISION_SUITE_BYTES + 1)
        except OSError as exc:
            raise ValueError(
                "native-vision suite must be a bounded regular file") from exc
        if len(encoded) != metadata.st_size:
            raise ValueError("native-vision suite changed while being read")

        def reject_constant(_value: str):
            raise ValueError("native-vision suite contains a non-finite number")

        suite = json.loads(encoded.decode("utf-8"), parse_constant=reject_constant)
        if not isinstance(suite, dict):
            raise ValueError("native-vision suite must be an object")
        name, version = suite.get("name"), suite.get("version")
        coverage, cases = suite.get("coverage", []), suite.get("cases")
        if (not isinstance(name, str) or not 1 <= len(name) <= 128
                or isinstance(version, bool) or not isinstance(version, int)
                or not 1 <= version <= 1_000_000
                or not isinstance(coverage, list) or len(coverage) > 32
                or any(not isinstance(item, str) or not 1 <= len(item) <= 80
                       for item in coverage)
                or len(set(coverage)) != len(coverage)
                or not isinstance(cases, list)
                or not 1 <= len(cases) <= MAX_VISION_CASES):
            raise ValueError("native-vision suite metadata is invalid")
        seen: set[str] = set()
        for case in cases:
            if not isinstance(case, dict) or set(case) != {
                    "name", "scene", "question", "expected_answer"}:
                raise ValueError("native-vision case metadata is invalid")
            case_name, scene = case["name"], case["scene"]
            canonical = _SCENES.get(scene)
            if (not isinstance(case_name, str) or not 1 <= len(case_name) <= 160
                    or case_name in seen or canonical is None
                    or (case["question"], case["expected_answer"]) != canonical):
                raise ValueError("native-vision case metadata is invalid")
            seen.add(case_name)
        return suite

    def _artifact(self, scene: str) -> tuple[bytes, dict[str, Any]]:
        with tempfile.TemporaryDirectory(
                prefix="friday-native-vision-artifact-") as temporary:
            path = Path(temporary) / "scene.png"
            path.write_bytes(_render_scene(scene))
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            try:
                return prepare_native_vision_image(
                    descriptor, path.name, max_side=self.max_side)
            finally:
                os.close(descriptor)

    def run(self, suite_path: str | Path) -> dict[str, Any]:
        suite = self._load_suite(suite_path)
        results: list[dict[str, Any]] = []
        for case in suite["cases"]:
            try:
                image, provenance = self._artifact(case["scene"])
                raw = self.complete(case["question"], image)
                if not isinstance(raw, str):
                    raise TypeError("completion callback returned a non-string")
                bounded = len(raw) <= MAX_VISION_OUTPUT_CHARS
                exact = bounded and _normalized(raw) == _normalized(
                    case["expected_answer"])
                grade = {
                    "output_sha256": hashlib.sha256(
                        raw.encode("utf-8")).hexdigest(),
                    "output_characters": len(raw),
                    "bounded_output": bounded,
                    "answer_exact": exact,
                }
                failure = None
            except Exception as exc:
                provenance = {}
                grade = {
                    "output_sha256": hashlib.sha256(b"").hexdigest(),
                    "output_characters": 0,
                    "bounded_output": False,
                    "answer_exact": False,
                }
                failure = type(exc).__name__
            result = {
                "name": case["name"], "scene": case["scene"],
                "passed": bool(grade["bounded_output"]
                               and grade["answer_exact"]),
                **provenance, **grade,
            }
            if failure:
                result["failure"] = failure
            results.append(result)
        passed = sum(int(item["passed"]) for item in results)
        body = {
            "suite": suite["name"], "version": suite["version"],
            "model": self.model,
            "runtime_fingerprint": self.runtime_fingerprint,
            "max_side": self.max_side,
            "coverage": list(suite.get("coverage", [])),
            "passed": passed, "total": len(results),
            "pass_rate": passed / len(results), "results": results,
            "ran_at": utc_now(),
        }
        if not math.isfinite(body["pass_rate"]):
            raise RuntimeError("native-vision evaluation score is non-finite")
        run_id = self.graph.record_node(
            "native_vision_evaluation_run", body,
            actor="native_vision_eval_runner",
            event_type="evaluation.native_vision_completed")
        return {"evaluation_run_id": run_id, **body}
