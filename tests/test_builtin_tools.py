import unittest

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


if __name__ == "__main__":
    unittest.main()
