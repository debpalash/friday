import hashlib
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from friday_core.installer_rehearsal import InstallerRehearsalRunner


REPO = Path(__file__).resolve().parents[1]


class InstallerRehearsalTests(unittest.TestCase):
    def test_current_source_archive_completes_clean_lifecycle_without_external_hosts(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.tar.gz"
            tracked = subprocess.run(
                ["git", "ls-files", "-z"], cwd=REPO, check=True,
                capture_output=True, timeout=20).stdout.split(b"\0")
            with tarfile.open(archive, "w:gz") as bundle:
                for encoded in tracked:
                    if not encoded:
                        continue
                    relative = Path(encoded.decode("utf-8"))
                    bundle.add(
                        REPO / relative,
                        arcname=str(Path("friday-candidate") / relative),
                        recursive=False,
                    )
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()

            result = InstallerRehearsalRunner(REPO).run(archive, digest)

        self.assertTrue(result["passed"], result)
        self.assertEqual(result["external_contacted_hosts"], [])
        self.assertTrue(result["checks"]["failed_update_rolled_back"])
        self.assertTrue(result["checks"]["reinstall_preserved_state"])
        self.assertGreater(result["archive_bytes"], 0)
        self.assertGreater(result["disk"]["application_bytes"], 0)
        self.assertTrue(result["privacy"]["fixture_cleanup_verified"])

    def test_wrong_archive_digest_is_rejected_before_fixture_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.tar.gz"
            archive.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "identity"):
                InstallerRehearsalRunner(REPO).run(archive, "0" * 64)


if __name__ == "__main__":
    unittest.main()
