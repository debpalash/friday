import re
import unittest

from friday_core.conversation_runtime import (
    bounded_response_fallback,
    canonical_chat_turn,
    completion_integrity_issue,
    compile_chat_messages,
    compile_fast_chat_messages,
    conversation_history_scope,
    drop_repeated_echo_messages,
    response_contract_issue,
)


class ConversationRuntimeTests(unittest.TestCase):
    def test_bounded_response_fallback_keeps_meaningful_prefix(self):
        self.assertEqual(
            bounded_response_fallback("one two three four five", 3),
            "one two three.")
        self.assertEqual(bounded_response_fallback("Already short.", 3),
                         "Already short.")

    def test_isolated_history_scope_restores_owner_and_keeps_updates(self):
        owner = type("Owner", (), {})()
        persistent = [{"role": "system", "content": "persistent"}]
        isolated = [{"role": "system", "content": "isolated"}]
        owner.history = persistent

        with conversation_history_scope(owner, isolated):
            self.assertIs(owner.history, isolated)
            owner.history.append({"role": "user", "content": "test"})

        self.assertIs(owner.history, persistent)
        self.assertEqual(isolated[-1]["content"], "test")

    def test_response_contract_rejects_thin_complex_answer(self):
        self.assertEqual(
            response_contract_issue("Unknown.", "Tell me the exact value", 20),
            "thin_answer")
        self.assertIsNone(response_contract_issue(
            "I do not have a recorded measurement for that time.",
            "Tell me the exact value", 20))

    def test_historical_exact_unknown_requires_an_evidence_basis(self):
        prompt = "Tell me the exact CPU temperature at 2 AM last Tuesday."
        self.assertEqual(
            response_contract_issue("I don't know the exact value.", prompt, 30),
            "missing_basis")
        self.assertIsNone(response_contract_issue(
            "I don't know because no measurement was recorded for that time.",
            prompt, 30))

    def test_completion_integrity_rejects_only_broken_responses(self):
        self.assertEqual(completion_integrity_issue(""), "empty")
        self.assertEqual(completion_integrity_issue("I"), "fragment")
        self.assertEqual(
            completion_integrity_issue("Partial", finish_reason="length"),
            "token_limit")
        self.assertEqual(
            completion_integrity_issue("```python\nprint('x')"),
            "unclosed_code_fence")
        self.assertIsNone(completion_integrity_issue("No."))
        self.assertIsNone(completion_integrity_issue("Cobalt"))

    def test_canonical_turn_requires_complete_tool_receipts(self):
        turn = [
            {"role": "user", "content": "inspect"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call", "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"x\"}"},
            }]},
            {"role": "tool", "tool_call_id": "call", "content": "ok"},
            {"role": "assistant", "content": "Done."},
        ]
        self.assertEqual(
            canonical_chat_turn(turn, redacted_tool_receipt="redacted")[-1],
            {"role": "assistant", "content": "Done."},
        )
        self.assertIsNone(canonical_chat_turn(
            turn[:-2], redacted_tool_receipt="redacted"))

    def test_sustained_echo_loop_is_removed_without_dropping_normal_turns(self):
        messages = [{"role": "system", "content": "prompt"}]
        for _ in range(4):
            messages.extend((
                {"role": "user", "content": "Okay."},
                {"role": "assistant", "content": "Okay."},
            ))
        messages.extend((
            {"role": "user", "content": "Continue"},
            {"role": "assistant", "content": "Ready."},
        ))

        cleaned = drop_repeated_echo_messages(messages)

        self.assertEqual([message["content"] for message in cleaned],
                         ["prompt", "Continue", "Ready."])

    def test_compiler_keeps_current_receipt_but_drops_failed_prior_turn(self):
        history = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "bad"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "a", "type": "function",
                "function": {"name": "x", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "a", "content": "error: failed"},
            {"role": "assistant", "content": "No."},
            {"role": "user", "content": "current"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "b", "type": "function",
                "function": {"name": "x", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "b", "content": "verified"},
        ]

        result = compile_chat_messages(
            history, base_prompt="prompt", local_time="now (zone).",
            context_sections=[], history_turns=24,
            redacted_tool_receipt="redacted", synthetic_fallbacks=set(),
            stale_capability_denial=re.compile(r"never"),
            ungrounded_action_claim=re.compile(r"never"),
        )

        self.assertNotIn("bad", [message.get("content") for message in result])
        self.assertEqual(result[-1]["content"], "verified")

    def test_fast_compiler_drops_fragment_and_repeated_short_reply_runs(self):
        history = [{"role": "system", "content": "prompt"}]
        history.extend((
            {"role": "user", "content": "sdef"},
            {"role": "assistant", "content": "I"},
        ))
        for prompt in ("hi", "hih", "olas"):
            history.extend((
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Hey"},
            ))
        history.append({"role": "user", "content": "Where were we?"})

        result = compile_fast_chat_messages(
            history, system_prompt="fast", history_turns=6,
            context_chars=8_000, redacted_tool_receipt="redacted")

        self.assertEqual(result, [
            {"role": "system", "content": "fast"},
            {"role": "user", "content": "Where were we?"},
        ])

    def test_fast_compiler_keeps_user_context_from_vacuous_reply_run(self):
        history = [{"role": "system", "content": "prompt"}]
        for prompt in ("Use SQLite", "Keep it local", "No cloud"):
            history.extend((
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Okay"},
            ))
        history.append({"role": "user", "content": "What did I decide?"})

        result = compile_fast_chat_messages(
            history, system_prompt="fast", history_turns=6,
            context_chars=8_000, redacted_tool_receipt="redacted")

        self.assertEqual(result, [
            {"role": "system", "content": "fast"},
            {"role": "user", "content": "Use SQLite"},
            {"role": "user", "content": "Keep it local"},
            {"role": "user", "content": "No cloud"},
            {"role": "user", "content": "What did I decide?"},
        ])


if __name__ == "__main__":
    unittest.main()
