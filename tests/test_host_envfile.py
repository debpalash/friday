"""friday.env parsing and rendering round-trip every value."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from friday_host import envfile


class EnvFileTests(unittest.TestCase):
    def test_round_trip_preserves_quotes_spaces_and_hashes(self) -> None:
        values = {
            "FRIDAY_INSTALL_ROOT": "/home/t/.local/share/friday",
            "FRIDAY_OWNER_NAME": "Pal O'Brien # not a comment",
            "FRIDAY_BIND_HOST": "127.0.0.1",
            "EMPTY": "",
        }
        text = envfile.render_env_text(values)
        self.assertIn("FRIDAY_OWNER_NAME='Pal O'\\''Brien # not a comment'", text)
        self.assertEqual(envfile.parse_env_text(text), values)

    def test_reads_shell_style_variants(self) -> None:
        text = (
            "# comment\n"
            "\n"
            "export FRIDAY_PORT=8500\n"
            'FRIDAY_QUOTED="a \\"b\\" c"\n'
            "FRIDAY_SINGLE='x'\n"
        )
        self.assertEqual(envfile.parse_env_text(text), {
            "FRIDAY_PORT": "8500", "FRIDAY_QUOTED": 'a "b" c', "FRIDAY_SINGLE": "x",
        })

    def test_rejects_malformed_lines(self) -> None:
        for bad in ("NOEQUALS", "1BAD=x", "FRIDAY='open", "FRIDAY='a'b'"):
            with self.subTest(bad=bad), self.assertRaises(envfile.EnvFileError):
                envfile.parse_env_text(bad)
        with self.assertRaises(envfile.EnvFileError):
            envfile.render_env_text({"BAD KEY": "x"})
        with self.assertRaises(envfile.EnvFileError):
            envfile.render_env_text({"KEY": "line\nbreak"})

    def test_read_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "friday.env"
            path.write_text("FRIDAY_PORT='8500'\n", encoding="utf-8")
            self.assertEqual(envfile.read_env_file(path), {"FRIDAY_PORT": "8500"})


if __name__ == "__main__":
    unittest.main()
