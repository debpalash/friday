import tempfile
import unittest
from pathlib import Path

from friday_core.frontend import FrontendAssetError, load_frontend


class FrontendAssetTests(unittest.TestCase):
    def test_loads_bounded_utf8_html(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "index.html"
            path.write_text("<!doctype html>\n<title>Friday</title>\n",
                            encoding="utf-8")

            self.assertEqual(load_frontend(path), path.read_text())

    def test_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "real.html"
            target.write_text("<!doctype html>", encoding="utf-8")
            link = Path(root) / "index.html"
            link.symlink_to(target)

            with self.assertRaises(FrontendAssetError):
                load_frontend(link)

    def test_rejects_oversized_invalid_and_non_html_assets(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "index.html"
            cases = (
                (b"<!doctype html>too large", 8),
                (b"\xff\xfe", 8),
                (b"plain text", 32),
            )
            for payload, limit in cases:
                with self.subTest(payload=payload):
                    path.write_bytes(payload)
                    with self.assertRaises(FrontendAssetError):
                        load_frontend(path, max_bytes=limit)

    def test_rejects_missing_asset(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FrontendAssetError):
                load_frontend(Path(root) / "missing.html")


if __name__ == "__main__":
    unittest.main()
