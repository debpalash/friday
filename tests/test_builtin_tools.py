import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from friday_core.builtin_tools import (
    BLOCKING_IO_TOOLS,
    BUILTIN_TOOL_NAMES,
    BUILTIN_TOOL_SCHEMAS,
    BUILTIN_TOOLS,
    DESKTOP_TOOL_NAMES,
    EXACT_STEP_APPROVAL_TOOLS,
    PROCESS_TOOL_NAMES,
    RESOURCE_OVERRIDES,
    TOOL_CRITERIA,
    TOOL_POLICY_DATA,
    BuiltinToolAdapters,
    BuiltinToolRuntime,
    builtin_tool,
)
from friday_core.cognition import (
    ContractBuilder,
    RiskClass,
    TOOL_POLICIES,
    resource_claim_for,
)
from friday_core.tasks import (
    tool_arguments_are_private,
    tool_has_private_payload,
)


class BuiltinToolCatalogTests(unittest.TestCase):
    def test_every_schema_has_one_catalog_entry(self):
        schema_names = [
            item["function"]["name"] for item in BUILTIN_TOOL_SCHEMAS
        ]

        self.assertEqual(len(schema_names), len(set(schema_names)))
        self.assertEqual(set(schema_names), BUILTIN_TOOL_NAMES)
        self.assertEqual(set(schema_names), set(BUILTIN_TOOLS))
        for name in schema_names:
            self.assertIsNotNone(builtin_tool(name))
            self.assertEqual(builtin_tool(name).schema["function"]["name"], name)

    def test_execution_categories_are_catalog_traits(self):
        self.assertEqual(
            BLOCKING_IO_TOOLS,
            {name for name, spec in BUILTIN_TOOLS.items() if spec.blocking_io},
        )
        self.assertEqual(
            EXACT_STEP_APPROVAL_TOOLS,
            {name for name, spec in BUILTIN_TOOLS.items()
             if spec.exact_step_approval},
        )
        self.assertTrue(all(
            BUILTIN_TOOLS[name].always_approve
            for name in EXACT_STEP_APPROVAL_TOOLS
        ))
        self.assertEqual(
            PROCESS_TOOL_NAMES,
            frozenset(name for name, spec in BUILTIN_TOOLS.items()
                      if spec.receipt_family == "process"),
        )
        self.assertEqual(
            DESKTOP_TOOL_NAMES,
            frozenset(name for name, spec in BUILTIN_TOOLS.items()
                      if spec.receipt_family == "desktop"),
        )

    def test_cognition_uses_catalog_policy_and_criteria(self):
        self.assertIs(ContractBuilder._TOOL_CRITERIA, TOOL_CRITERIA)
        self.assertEqual(set(TOOL_POLICIES), set(TOOL_POLICY_DATA))
        for name, (risk, permissions, always_approve) in TOOL_POLICY_DATA.items():
            self.assertEqual(
                TOOL_POLICIES[name],
                (RiskClass(risk), permissions, always_approve),
            )
            spec = BUILTIN_TOOLS[name]
            self.assertEqual(spec.risk, risk)
            self.assertEqual(spec.permissions, permissions)
            self.assertEqual(spec.always_approve, always_approve)

    def test_resource_claims_use_catalog_overrides(self):
        for name, overrides in RESOURCE_OVERRIDES.items():
            claim = resource_claim_for(name).model_dump()
            for field, expected in overrides.items():
                self.assertEqual(claim[field], expected)

    def test_privacy_rules_use_catalog_traits(self):
        for name, spec in BUILTIN_TOOLS.items():
            self.assertEqual(tool_has_private_payload(name), spec.private_payload)
            self.assertEqual(
                tool_arguments_are_private(name),
                bool(spec.private_argument_fields),
            )
        # Prefix-based privacy remains fail-safe for future machine adapters.
        self.assertTrue(tool_has_private_payload("machine_future_adapter"))
        self.assertTrue(tool_has_private_payload("browser_future_adapter"))
        self.assertFalse(tool_has_private_payload("ordinary_dynamic_tool"))


class BuiltinToolRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name).resolve()
        self.fetch_news = Mock(return_value={"headlines": [{"title": "Current"}]})
        self.web = SimpleNamespace(
            search=Mock(return_value={"results": [{"url": "https://example.com"}]}),
            read=Mock(return_value={"url": "https://example.com", "text": "page"}),
            open=Mock(return_value={"status": "ok"}),
            snapshot=Mock(return_value={"status": "ok", "text": "page"}),
            click=Mock(return_value={"status": "ok"}),
            type=Mock(return_value={"status": "ok"}),
        )
        self.skill_source = SimpleNamespace(
            search=Mock(return_value={"results": [{"id": "owner/repo/skill"}]}))
        self.reminders = SimpleNamespace(
            create=Mock(return_value={"reminder_id": "reminder_1"}),
            list=Mock(return_value=[{"reminder_id": "reminder_1"}]),
            cancel=Mock(return_value={"status": "cancelled"}),
        )
        self.run_process = Mock(return_value=SimpleNamespace(stdout="clipboard"))
        self.start_process = Mock()
        self.adapters = BuiltinToolAdapters(
            repo=self.repo,
            fetch_news=self.fetch_news,
            web=self.web,
            skill_source=self.skill_source,
            reminders=self.reminders,
            run_process=self.run_process,
            start_process=self.start_process,
        )
        self.runtime = BuiltinToolRuntime()

    def tearDown(self):
        self.temp.cleanup()

    def test_network_and_schedule_tools_use_injected_adapters(self):
        news = json.loads(self.runtime.execute(
            "fetch_news", {"topic": "tech", "limit": 2, "region": "India"},
            self.adapters))
        search = json.loads(self.runtime.execute(
            "web_search", {"query": "Friday", "limit": 3}, self.adapters))
        reminder = json.loads(self.runtime.execute(
            "create_reminder", {
                "text": "test", "due_at": "2026-08-28T09:00:00+05:30",
            }, self.adapters))

        self.assertEqual(news["headlines"][0]["title"], "Current")
        self.assertEqual(search["results"][0]["url"], "https://example.com")
        self.assertEqual(reminder["reminder_id"], "reminder_1")
        self.fetch_news.assert_called_once_with("tech", 2, "India")
        self.web.search.assert_called_once_with("Friday", limit=3)
        self.reminders.create.assert_called_once_with(
            "test", "2026-08-28T09:00:00+05:30", interval_seconds=None)

    def test_project_file_tools_share_the_hardened_path_adapter(self):
        (self.repo / "visible.txt").write_text("hello")
        (self.repo / ".env").write_text("do not read")
        (self.repo / "venv").mkdir()
        (self.repo / "venv" / "hidden.txt").write_text("do not read")

        self.assertEqual(
            self.runtime.execute("read_file", {"path": "visible.txt"},
                                 self.adapters),
            "hello",
        )
        self.assertTrue(self.runtime.execute(
            "read_file", {"path": ".env"}, self.adapters).startswith("error:"))
        self.assertTrue(self.runtime.execute(
            "read_file", {"path": "venv/hidden.txt"},
            self.adapters).startswith("error:"))
        self.assertIsNone(self.runtime.safe_project_path(
            self.repo, str(self.repo.parent / "outside.txt")))

    def test_desktop_effects_use_injected_process_adapters(self):
        clipboard = json.loads(self.runtime.execute(
            "clipboard_read", {}, self.adapters))
        notification = json.loads(self.runtime.execute(
            "desktop_notify", {"title": "Friday", "message": "done"},
            self.adapters))

        self.assertEqual(clipboard, {"status": "ok", "text": "clipboard"})
        self.assertEqual(notification, {"status": "ok", "delivered": True})
        self.assertEqual(self.run_process.call_count, 2)

    def test_adapter_failure_becomes_a_tool_error_receipt(self):
        self.web.search.side_effect = RuntimeError("offline")

        result = self.runtime.execute(
            "web_search", {"query": "Friday"}, self.adapters)

        self.assertEqual(result, "error: web search failed: offline")
        self.assertEqual(
            self.runtime.execute("not_a_tool", {}, self.adapters),
            "error: unknown tool not_a_tool",
        )


if __name__ == "__main__":
    unittest.main()
