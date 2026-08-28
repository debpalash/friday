import re
import unittest

from friday_core.conversation_runtime import (
    canonical_chat_turn,
    compile_chat_messages,
    drop_repeated_echo_messages,
)


class ConversationRuntimeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
