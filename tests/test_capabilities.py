import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path

from friday_core import CapabilityManager, GraphStore, TaskService


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.graph = GraphStore(root / "friday.db")
        self.tasks = TaskService(self.graph)
        self.manager = CapabilityManager(
            self.graph, root / "capabilities", reserved_names={"read_file"})
        self.task_id, _ = self.tasks.create("add doubling tool", {"evidence": "tests"})

    def tearDown(self):
        self.tmp.cleanup()

    def require_sandbox(self):
        ready, detail = self.manager.sandbox_status()
        if not ready:
            self.skipTest(f"Bubblewrap sandbox unavailable: {detail}")

    def test_tested_capability_becomes_dynamic_tool(self):
        self.require_sandbox()
        version = self.manager.create_version(
            "double_value", "Double an integer.",
            {"type": "object", "properties": {"value": {"type": "integer"}},
             "required": ["value"]},
            "def run(args):\n    return args['value'] * 2\n", [],
            [{"name": "two", "args": {"value": 2}, "expected": 4},
             {"name": "negative", "args": {"value": -3}, "expected": -6}],
            source_node_ids=[self.task_id])

        self.assertEqual(self.manager.tool_schemas(), [])
        self.assertTrue(self.manager.evaluate_and_activate(version))
        self.assertEqual(self.manager.execute("double_value", {"value": 5}), 10)
        self.assertEqual(self.manager.tool_schemas()[0]["function"]["name"],
                         "double_value")

    def test_failed_tests_quarantine_capability(self):
        self.require_sandbox()
        version = self.manager.create_version(
            "wrong_math", "A deliberately failing candidate.",
            {"type": "object", "properties": {}},
            "def run(args):\n    return 1\n", [],
            [{"args": {}, "expected": 2}, {"args": {}, "expected": 3}],
            source_node_ids=[self.task_id])

        self.assertFalse(self.manager.evaluate_and_activate(version))
        self.assertEqual(self.manager.list()[0]["status"], "quarantined")
        with self.assertRaises(ValueError):
            self.manager.execute("wrong_math", {})

    def test_unsafe_code_and_builtin_collision_are_rejected(self):
        parameters = {"type": "object", "properties": {}}
        tests = [{"args": {}, "expected": 1}, {"args": {}, "expected": 1}]
        with self.assertRaisesRegex(ValueError, "collides"):
            self.manager.create_version(
                "read_file", "collision", parameters,
                "def run(args):\n    return 1\n", [], tests,
                source_node_ids=[self.task_id])
        with self.assertRaisesRegex(ValueError, "unsafe operation"):
            self.manager.create_version(
                "unsafe_tool", "unsafe", parameters,
                "def run(args):\n    return open('/etc/passwd').read()\n", [], tests,
                source_node_ids=[self.task_id])

    def test_one_candidates_permissions_do_not_leak_into_the_next(self):
        self.manager._validate_code(
            "import subprocess\ndef run(args):\n    return 1\n", ["process"])

        with self.assertRaisesRegex(ValueError, "unavailable permission"):
            self.manager._validate_code(
                "import subprocess\ndef run(args):\n    return 1\n", [])

    def test_sandbox_outage_leaves_candidate_retryable(self):
        version = self.manager.create_version(
            "retry_later", "A candidate awaiting sandbox infrastructure.",
            {"type": "object", "properties": {}},
            "def run(args):\n    return 1\n", [],
            [{"args": {}, "expected": 1}, {"args": {}, "expected": 1}],
            source_node_ids=[self.task_id])
        self.manager.sandbox_status = lambda: (False, "namespace unavailable")

        self.assertFalse(self.manager.evaluate_and_activate(version))
        self.assertEqual(self.manager.list()[0]["status"], "drafted")
        self.assertEqual(self.manager.version_status(version), "drafted")

    def test_artifacts_are_private_atomic_and_tampering_fails_closed(self):
        code = "def run(args):\n    return 7\n"
        version_id = self.manager.create_version(
            "sealed_tool", "Integrity checked.",
            {"type": "object", "properties": {}}, code, [],
            [{"args": {}, "expected": 7}, {"args": {}, "expected": 7}],
            source_node_ids=[self.task_id])
        version = self.manager._version(version_id)
        directory = self.manager.root / "sealed_tool" / "v1"
        manifest = json.loads((directory / "manifest.json").read_text())

        self.assertEqual(
            manifest["handler_sha256"], hashlib.sha256(code.encode()).hexdigest())
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        self.assertEqual(
            stat.S_IMODE((directory / "handler.py").stat().st_mode), 0o600)
        self.assertEqual(
            self.manager._verify_artifacts(version), directory / "handler.py")

        (directory / "handler.py").write_text(
            "def run(args):\n    return 'tampered'\n")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            self.manager._verify_artifacts(version)

    def test_orphaned_publish_is_quarantined_before_version_commit(self):
        orphan = self.manager.root / "recoverable" / "v1"
        orphan.mkdir(parents=True)
        (orphan / "handler.py").write_text("uncommitted")

        version_id = self.manager.create_version(
            "recoverable", "Recover after a publish crash.",
            {"type": "object", "properties": {}},
            "def run(args):\n    return 1\n", [],
            [{"args": {}, "expected": 1}, {"args": {}, "expected": 1}],
            source_node_ids=[self.task_id])

        self.assertEqual(self.manager._version(version_id)["version"], 1)
        self.assertTrue((self.manager.root / "recoverable" / "v1" /
                         "manifest.json").is_file())
        self.assertEqual(
            len(list((self.manager.root / ".orphaned").iterdir())), 1)

    def test_evaluation_uses_ephemeral_data_not_live_capability_state(self):
        self.require_sandbox()
        version = self.manager.create_version(
            "isolated_data", "Prove evaluation data isolation.",
            {"type": "object", "properties": {}},
            "from pathlib import Path\n"
            "def run(args):\n"
            "    marker = Path('/data/live-marker')\n"
            "    existed = marker.exists()\n"
            "    marker.write_text('candidate-test')\n"
            "    return existed\n",
            ["filesystem_write"],
            [{"args": {}, "expected": False},
             {"args": {}, "expected": False}],
            source_node_ids=[self.task_id])
        live = self.manager.root / "isolated_data" / "data"
        live.mkdir()
        (live / "live-marker").write_text("production")

        self.assertTrue(self.manager.evaluate_and_activate(version))
        self.assertEqual((live / "live-marker").read_text(), "production")
        self.assertTrue(self.manager.execute("isolated_data", {}))

    def test_failed_new_version_keeps_prior_active_version(self):
        self.require_sandbox()
        parameters = {"type": "object", "properties": {}}
        v1 = self.manager.create_version(
            "stable_version", "Stable v1.", parameters,
            "def run(args):\n    return 1\n", [],
            [{"args": {}, "expected": 1}, {"args": {}, "expected": 1}],
            source_node_ids=[self.task_id])
        self.assertTrue(self.manager.evaluate_and_activate(v1))
        v2 = self.manager.create_version(
            "stable_version", "Broken v2.", parameters,
            "def run(args):\n    return 2\n", [],
            [{"args": {}, "expected": 3}, {"args": {}, "expected": 3}],
            source_node_ids=[self.task_id])

        self.assertFalse(self.manager.evaluate_and_activate(v2))
        self.assertEqual(self.manager.execute("stable_version", {}), 1)
        self.assertEqual(self.manager.list()[0]["status"], "active")

    def test_active_metadata_binds_execution_to_exact_code_hash(self):
        self.require_sandbox()
        version = self.manager.create_version(
            "bound_executor", "Bound executor.",
            {"type": "object", "properties": {}},
            "def run(args):\n    return 9\n", ["network"],
            [{"args": {}, "expected": 9}, {"args": {}, "expected": 9}],
            source_node_ids=[self.task_id])
        self.assertTrue(self.manager.evaluate_and_activate(version))
        metadata = self.manager.active_metadata("bound_executor")

        self.assertEqual(metadata["version_id"], version)
        self.assertEqual(metadata["permissions"], ["network"])
        self.assertEqual(self.manager.execute_version(
            version, {}, expected_name=metadata["name"],
            expected_version=metadata["version"],
            expected_code_sha256=metadata["code_sha256"],
            expected_permissions=metadata["permissions"]), 9)
        with self.assertRaisesRegex(RuntimeError, "binding hash"):
            self.manager.execute_version(
                version, {}, expected_name=metadata["name"],
                expected_version=metadata["version"],
                expected_code_sha256="0" * 64,
                expected_permissions=metadata["permissions"])


if __name__ == "__main__":
    unittest.main()
