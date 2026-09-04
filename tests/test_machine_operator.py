import hashlib
import os
import socket
import stat
import tempfile
import unittest

from tests.platform_markers import linux_only
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from friday_core.graph import GraphStore
from friday_core.machine import MachineOperator, OperatorGrantService


class MachineOperatorTests(unittest.TestCase):
    PERMISSIONS = ["inspect", "list", "read", "write"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.project = self.root / "project"
        self.home = self.root / "home"
        self.state = self.root / "state"
        self.project.mkdir()
        self.home.mkdir()
        self.state.mkdir()
        self.graph = GraphStore(self.state / "friday.db")
        self.grants = OperatorGrantService(
            self.graph,
            self.project,
            home=self.home,
            state_root=self.state,
        )
        self.operator = MachineOperator(self.grants, state_root=self.state)

    def tearDown(self):
        self.tmp.cleanup()

    def grant(self, path: Path, permissions=None, **kwargs):
        return self.grants.grant_path(
            path,
            list(permissions or self.PERMISSIONS),
            **kwargs,
        )

    def database_dump(self) -> str:
        with self.graph._connect() as conn:
            return "\n".join(conn.iterdump())

    def private_state_snapshot(self):
        snapshot = {}
        for path in self.state.rglob("*"):
            if not path.is_file() or path.name.startswith("friday.db"):
                continue
            relative = str(path.relative_to(self.state))
            snapshot[relative] = {
                "mode": stat.S_IMODE(path.stat().st_mode),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        return snapshot

    def test_grant_target_is_encrypted_and_grant_key_is_private(self):
        target = self.home / "private-scope-4f1bd89a"
        target.mkdir()

        grant = self.grant(target, ["read"])

        self.assertNotIn(str(target.resolve()), self.database_dump())
        redacted = {item["grant_id"]: item
                    for item in self.grants.list_grants()}
        self.assertNotEqual(redacted[grant["grant_id"]]["target"],
                            str(target.resolve()))
        revealed = {item["grant_id"]: item
                    for item in self.grants.list_grants(reveal_targets=True)}
        self.assertEqual(revealed[grant["grant_id"]]["target"],
                         str(target.resolve()))
        key_files = list(self.state.rglob("*.key"))
        self.assertTrue(key_files, "creating a grant must create a local key")
        for key_file in key_files:
            self.assertEqual(stat.S_IMODE(key_file.stat().st_mode), 0o600)

    def test_directory_grant_is_contained_and_permissions_are_enforced(self):
        scope = self.project / "scope"
        scope.mkdir()
        child = scope / "inside.txt"
        child.write_text("inside\n")
        sibling = self.project / "outside.txt"
        sibling.write_text("outside\n")
        self.grant(scope, ["inspect", "list", "read"])

        inspection = self.operator.inspect(child)
        listing = self.operator.list_path(scope)
        reading = self.operator.read_text(child)

        self.assertEqual(inspection["kind"], "file")
        self.assertEqual({item["name"] for item in listing["entries"]},
                         {"inside.txt"})
        self.assertEqual(reading["text"], "inside\n")
        with self.assertRaises(PermissionError):
            self.operator.write_text(
                child, "unauthorized\n", operation_id="op-denied-0001")
        with self.assertRaises(PermissionError):
            self.operator.read_text(sibling)
        with self.assertRaises(PermissionError):
            self.operator.read_text(str(scope / ".." / "outside.txt"))
        self.assertEqual(child.read_text(), "inside\n")

    def test_expired_and_revoked_grants_do_not_authorize(self):
        scope = self.project / "leased"
        scope.mkdir()
        target = scope / "note.txt"
        target.write_text("temporary authority\n")
        future = datetime.now(UTC) + timedelta(hours=1)
        expiring = self.grant(
            scope,
            ["read"],
            expires_at=future.isoformat().replace("+00:00", "Z"),
        )
        self.assertEqual(self.operator.read_text(target)["text"],
                         "temporary authority\n")

        expired_now = future + timedelta(seconds=1)

        class ExpiredClock(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return expired_now.replace(tzinfo=None)
                return expired_now.astimezone(tz)

        with patch("friday_core.machine.datetime", ExpiredClock):
            with self.assertRaises(PermissionError):
                self.operator.read_text(target)
            listed = {item["grant_id"]: item
                      for item in self.grants.list_grants()}
            self.assertEqual(listed[expiring["grant_id"]]["status"],
                             "expired")
        self.grants.revoke(expiring["grant_id"])

        durable = self.grant(scope, ["read"])
        self.assertEqual(self.operator.read_text(target)["text"],
                         "temporary authority\n")
        revoked = self.grants.revoke(durable["grant_id"])
        self.assertEqual(revoked["status"], "revoked")
        with self.assertRaises(PermissionError):
            self.operator.read_text(target)
        listed = {item["grant_id"]: item
                  for item in self.grants.list_grants()}
        self.assertEqual(listed[durable["grant_id"]]["status"], "revoked")

    def test_symlink_paths_are_rejected_even_when_they_resolve_inside_grant(self):
        scope = self.project / "links"
        scope.mkdir()
        target = scope / "real.txt"
        target.write_text("real content\n")
        alias = scope / "alias.txt"
        alias.symlink_to(target)
        outside = self.project / "outside-link-target.txt"
        outside.write_text("outside content\n")
        outside_alias = scope / "outside-alias.txt"
        outside_alias.symlink_to(outside)
        self.grant(scope)

        for path in (alias, outside_alias):
            with self.subTest(path=path), self.assertRaises(PermissionError):
                self.operator.read_text(path)
        with self.assertRaises(PermissionError):
            self.operator.write_text(
                alias, "replacement\n", operation_id="op-symlink-0001")
        self.assertTrue(alias.is_symlink())
        self.assertEqual(target.read_text(), "real content\n")
        self.assertEqual(outside.read_text(), "outside content\n")

    @linux_only
    def test_special_files_are_rejected_without_being_replaced(self):
        scope = self.project / "special"
        scope.mkdir()
        socket_path = scope / "service.sock"
        local_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(local_socket.close)
        local_socket.bind(str(socket_path))
        self.grant(scope)

        self.assertEqual(self.operator.inspect(socket_path)["kind"], "special")
        with self.assertRaises((ValueError, OSError)):
            self.operator.read_text(socket_path)
        with self.assertRaises(ValueError):
            self.operator.write_text(
                socket_path, "replacement", operation_id="op-special-0001")
        self.assertTrue(stat.S_ISSOCK(socket_path.lstat().st_mode))

    def test_write_is_an_atomic_verified_replacement(self):
        scope = self.project / "atomic"
        scope.mkdir()
        target = scope / "settings.ini"
        target.write_text("version=1\n")
        before_inode = target.stat().st_ino
        self.grant(scope)
        content = "version=2\nfeature=true\n"

        with patch("friday_core.machine.os.replace",
                   wraps=os.replace) as replace:
            receipt = self.operator.write_text(
                target, content, operation_id="op-atomic-0001")

        self.assertTrue(replace.called, "writes must commit with os.replace")
        self.assertNotEqual(target.stat().st_ino, before_inode)
        self.assertEqual(target.read_text(), content)
        self.assertEqual(receipt["status"], "ok")
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["after_sha256"],
                         hashlib.sha256(content.encode()).hexdigest())
        self.assertEqual(receipt["rollback_operation_id"], "op-atomic-0001")
        self.assertFalse(any(
            item.name.startswith(".friday-stage-")
            for item in scope.iterdir()
        ))

    def test_retry_reuses_one_deterministic_private_backup(self):
        scope = self.project / "idempotent"
        scope.mkdir()
        target = scope / "document.txt"
        target.write_text("original-private-payload\n")
        self.grant(scope)
        operation_id = "op-retry-0001"
        baseline = self.private_state_snapshot()

        first = self.operator.write_text(
            target, "replacement\n", operation_id=operation_id)
        first_snapshot = self.private_state_snapshot()
        second = self.operator.write_text(
            target, "replacement\n", operation_id=operation_id)
        second_snapshot = self.private_state_snapshot()

        new_files = set(first_snapshot) - set(baseline)
        self.assertTrue(new_files, "a reversible write needs a checkpoint")
        self.assertEqual(second_snapshot, first_snapshot)
        self.assertEqual(first["rollback_operation_id"], operation_id)
        self.assertEqual(second["rollback_operation_id"], operation_id)
        for item in first_snapshot.values():
            self.assertEqual(item["mode"], 0o600)
        with self.assertRaises(RuntimeError):
            self.operator.write_text(
                target, "different replay\n", operation_id=operation_id)
        self.assertEqual(target.read_text(), "replacement\n")

        rollback = self.operator.rollback(operation_id)

        self.assertTrue(rollback["verified"])
        self.assertEqual(target.read_text(), "original-private-payload\n")

    def test_rollback_refuses_to_overwrite_a_later_edit(self):
        scope = self.project / "rollback-conflict"
        scope.mkdir()
        target = scope / "document.txt"
        target.write_text("before\n")
        self.grant(scope)
        operation_id = "op-conflict-0001"
        self.operator.write_text(
            target, "friday edit\n", operation_id=operation_id)
        target.write_text("later human edit\n")

        with self.assertRaises(RuntimeError):
            self.operator.rollback(operation_id)

        self.assertEqual(target.read_text(), "later human edit\n")

    def test_write_replay_refuses_a_later_edit_and_preserves_it(self):
        scope = self.project / "replay-conflict"
        scope.mkdir()
        target = scope / "document.txt"
        target.write_text("before\n")
        self.grant(scope)
        operation_id = "op-replay-conflict-0001"
        self.operator.write_text(
            target, "friday edit\n", operation_id=operation_id)
        target.write_text("later human edit\n")
        before_replay = self.private_state_snapshot()

        with self.assertRaises(RuntimeError):
            self.operator.write_text(
                target, "friday edit\n", operation_id=operation_id)

        self.assertEqual(target.read_text(), "later human edit\n")
        self.assertEqual(self.private_state_snapshot(), before_replay)

    def test_write_replay_after_rollback_preserves_the_restored_original(self):
        scope = self.project / "replay-after-rollback"
        scope.mkdir()
        target = scope / "document.txt"
        target.write_text("original\n")
        self.grant(scope)
        operation_id = "op-replay-rolled-back-0001"
        self.operator.write_text(
            target, "friday edit\n", operation_id=operation_id)
        self.operator.rollback(operation_id)
        restored_inode = target.stat().st_ino
        restored_state = self.private_state_snapshot()

        with self.assertRaises(RuntimeError):
            self.operator.write_text(
                target, "friday edit\n", operation_id=operation_id)

        self.assertEqual(target.read_text(), "original\n")
        self.assertEqual(target.stat().st_ino, restored_inode)
        self.assertEqual(self.private_state_snapshot(), restored_state)

    def test_operation_journal_is_encrypted_private_and_rollback_is_idempotent(self):
        scope = self.project / "private-journal"
        scope.mkdir()
        target = scope / "document.txt"
        original = "unique-original-journal-secret-02d6828b\n"
        replacement = "unique-replacement-journal-secret-cb270bf4\n"
        target.write_text(original)
        self.grant(scope)
        operation_id = "op-private-journal-0001"
        baseline = self.private_state_snapshot()
        self.operator.write_text(
            target, replacement, operation_id=operation_id)

        journals = list(self.operator.backup_root.glob("*.enc"))
        self.assertEqual(len(journals), 1)
        journal = journals[0]
        self.assertEqual(stat.S_IMODE(journal.stat().st_mode), 0o600)
        encrypted = journal.read_bytes()
        for plaintext in (
                original.encode(), replacement.encode(),
                str(target.resolve()).encode(), operation_id.encode()):
            self.assertNotIn(plaintext, encrypted)
        after_write = self.private_state_snapshot()
        self.assertGreater(len(after_write), len(baseline))
        for item in after_write.values():
            self.assertEqual(item["mode"], 0o600)

        first = self.operator.rollback(operation_id)
        restored_inode = target.stat().st_ino
        after_first_rollback = self.private_state_snapshot()
        second = self.operator.rollback(operation_id)

        self.assertFalse(first["already_restored"])
        self.assertTrue(second["already_restored"])
        self.assertTrue(second["verified"])
        self.assertEqual(target.read_text(), original)
        self.assertEqual(target.stat().st_ino, restored_inode)
        self.assertEqual(self.private_state_snapshot(), after_first_rollback)
        encrypted_after_rollback = journal.read_bytes()
        for plaintext in (
                original.encode(), replacement.encode(),
                str(target.resolve()).encode(), operation_id.encode()):
            self.assertNotIn(plaintext, encrypted_after_rollback)

    def test_sensitive_descendants_need_an_explicit_sensitive_grant(self):
        ssh = self.home / ".ssh"
        ssh.mkdir()
        private_key = ssh / "id_ed25519"
        private_key.write_text("not-a-real-private-key\n")
        ordinary = self.home / "notes.txt"
        ordinary.write_text("ordinary\n")
        self.grant(self.home, ["inspect", "list", "read"])

        listing = self.operator.list_path(self.home)
        self.assertEqual(self.operator.read_text(ordinary)["text"],
                         "ordinary\n")
        self.assertNotIn(".ssh",
                         {entry["name"] for entry in listing["entries"]})
        self.assertEqual(listing["omitted_sensitive"], 1)
        with self.assertRaises(PermissionError):
            self.operator.inspect(ssh)
        with self.assertRaises(PermissionError):
            self.operator.read_text(private_key)

        self.grant(
            ssh,
            ["inspect", "read"],
            allow_sensitive=True,
        )

        self.assertEqual(self.operator.inspect(private_key)["kind"], "file")
        self.assertEqual(self.operator.read_text(private_key)["text"],
                         "not-a-real-private-key\n")


if __name__ == "__main__":
    unittest.main()
