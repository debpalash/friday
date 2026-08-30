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

    def test_reconnect_fast_forwards_progress_without_replaying_old_tasks(self):
        frontend = (
            Path(__file__).parents[1] / "frontend" / "index.html"
        ).read_text()

        self.assertIn("let progressSeq=0,progressInitialized=false;", frontend)
        self.assertIn("localStorage.removeItem('friday-progress-seq');", frontend)
        self.assertNotIn("localStorage.getItem('friday-progress-seq')", frontend)
        self.assertIn("progressInitialized=false;void pollProgress();", frontend)
        self.assertIn("/api/progress?latest=true", frontend)
        self.assertIn(
            "progressSeq=Math.max(progressSeq,Number(cursor.latest)||0);",
            frontend,
        )
        self.assertIn(
            "const taskEvent=Boolean(m.seq&&m.task_id&&String(m.task_id)"
            ".startsWith('task_'));",
            frontend,
        )
        self.assertIn("if(taskEvent)showTaskCard(m,detail);", frontend)

    def test_progress_diagnostics_use_the_recorded_event_time(self):
        frontend = (
            Path(__file__).parents[1] / "frontend" / "index.html"
        ).read_text()

        self.assertIn("function dlog(t,occurredAt)", frontend)
        self.assertIn("d.textContent+=stamp+'  '+t+'\\n';", frontend)
        self.assertNotIn("d.textContent+=stamp+'  '+t+'\\\\n';", frontend)
        self.assertIn(
            "dlog((m.phase||'task')+' '+m.state+': '+m.label+detail,m.occurred_at);",
            frontend,
        )


if __name__ == "__main__":
    unittest.main()
