import sys
import tempfile
import unittest
from pathlib import Path

from friday_core import DeploymentManager, GraphStore


class DeploymentTests(unittest.TestCase):
    def test_promote_and_rollback_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            target = root / "module.py"
            target.write_text("VALUE = 1\n")
            graph = GraphStore(root / "state/friday.db")
            deploy = DeploymentManager(
                graph, root, [sys.executable, "-c", "raise SystemExit(0)"])

            result = deploy.stage_write("module.py", "VALUE = 2\n")
            self.assertEqual(result["status"], "promoted")
            self.assertEqual(target.read_text(), "VALUE = 2\n")
            deploy.rollback(result["deployment_id"])
            self.assertEqual(target.read_text(), "VALUE = 1\n")

    def test_failed_verification_does_not_change_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "module.py"
            target.write_text("VALUE = 1\n")
            graph = GraphStore(root / "state/friday.db")
            deploy = DeploymentManager(
                graph, root, [sys.executable, "-c", "raise SystemExit(7)"])

            with self.assertRaises(RuntimeError):
                deploy.stage_write("module.py", "VALUE = 2\n")

            self.assertEqual(target.read_text(), "VALUE = 1\n")
            with graph._connect() as conn:
                status = conn.execute(
                    "SELECT status FROM deployment_state").fetchone()[0]
            self.assertEqual(status, "rejected")

    def test_syntax_error_is_rejected_before_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "module.py"
            target.write_text("VALUE = 1\n")
            graph = GraphStore(root / "state/friday.db")
            deploy = DeploymentManager(
                graph, root, [sys.executable, "-c", "raise SystemExit(0)"])
            with self.assertRaises(SyntaxError):
                deploy.stage_write("module.py", "def broken(:\n")
            self.assertEqual(target.read_text(), "VALUE = 1\n")

    def test_existing_acceptance_test_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            target = root / "tests/test_guard.py"
            target.write_text("EXPECTED = True\n")
            graph = GraphStore(root / "state/friday.db")
            deploy = DeploymentManager(
                graph, root, [sys.executable, "-c", "raise SystemExit(0)"])

            with self.assertRaisesRegex(ValueError, "may not modify"):
                deploy.stage_write("tests/test_guard.py", "EXPECTED = False\n")

            self.assertEqual(target.read_text(), "EXPECTED = True\n")

    def test_runtime_authority_and_generated_executables_are_not_deployable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = GraphStore(root / "state/friday.db")
            deploy = DeploymentManager(
                graph, root, [sys.executable, "-c", "raise SystemExit(0)"])

            for target in (
                    "state/control-token", "session.json", ".env.local",
                    "capabilities/tool/v1/handler.py", "venv/bin/python"):
                with self.subTest(target=target), self.assertRaisesRegex(
                        ValueError, "protected runtime state"):
                    deploy.stage_write(target, "untrusted replacement\n")

    def test_bundle_promotes_and_rolls_back_as_one_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("VALUE = 1\n")
            (root / "b.py").write_text("VALUE = 2\n")
            graph = GraphStore(root / "state/friday.db")
            deploy = DeploymentManager(
                graph, root, [sys.executable, "-c", "raise SystemExit(0)"])

            result = deploy.stage_bundle(
                {"a.py": "VALUE = 3\n", "b.py": "VALUE = 4\n"})
            self.assertEqual((root / "a.py").read_text(), "VALUE = 3\n")
            self.assertEqual((root / "b.py").read_text(), "VALUE = 4\n")

            deploy.rollback(result["deployment_id"])
            self.assertEqual((root / "a.py").read_text(), "VALUE = 1\n")
            self.assertEqual((root / "b.py").read_text(), "VALUE = 2\n")

    def test_failed_bundle_restores_every_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("VALUE = 1\n")
            (root / "b.py").write_text("VALUE = 2\n")
            graph = GraphStore(root / "state/friday.db")
            deploy = DeploymentManager(
                graph, root, [sys.executable, "-c", "raise SystemExit(9)"])

            with self.assertRaises(RuntimeError):
                deploy.stage_bundle(
                    {"a.py": "VALUE = 3\n", "b.py": "VALUE = 4\n"})

            self.assertEqual((root / "a.py").read_text(), "VALUE = 1\n")
            self.assertEqual((root / "b.py").read_text(), "VALUE = 2\n")


if __name__ == "__main__":
    unittest.main()
