import unittest

from friday_core.conversation import (FAST_CONVERSATION_TEMPERATURE,
                                      FAST_CONVERSATION_TOP_P,
                                      fast_system_prompt, format_runtime_answer,
                                      runtime_topics, safe_for_fast_conversation)


class ConversationRoutingTests(unittest.TestCase):
    def test_runtime_topics_cover_each_live_identity_dimension(self):
        cases = {
            "What model are you running?": ("model",),
            "Which ASR are you using?": ("asr",),
            "What TTS are you using right now?": ("tts",),
            "Are you Piper or OmniVoice?": ("tts",),
            "What device are you running on?": ("device",),
            "What's under the hood?": ("model", "asr", "tts", "device"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(runtime_topics(text), expected)

    def test_runtime_topics_do_not_capture_general_model_questions(self):
        self.assertEqual(runtime_topics("What model should I use for OCR?"), ())
        self.assertEqual(runtime_topics("What is ASR?"), ())
        self.assertEqual(
            runtime_topics("Which voice should I use for a documentary?"), ())

    def test_fast_path_is_conservative_about_tools_and_durable_state(self):
        for text in (
                "Hey Friday.", "Tell me a compiler joke.",
                "Explain recursion in one sentence.", "That was funny."):
            with self.subTest(text=text):
                self.assertTrue(safe_for_fast_conversation(text))
        for text in (
                "What is in README.md?", "What's my git email?",
                "Remember that I prefer terse answers.",
                "What is the latest Python release?", "Where were we?",
                "Who is the current prime minister?", "What time is it?",
                "What can you do?", "Do not use Markdown."):
            with self.subTest(text=text):
                self.assertFalse(safe_for_fast_conversation(text))
        self.assertFalse(safe_for_fast_conversation(
            "Explain recursion.", action_request=True))

    def test_fast_prompt_forbids_workflow_and_runtime_guessing(self):
        voice = fast_system_prompt(owner_name="Pal", display_mode=False)
        text = fast_system_prompt(owner_name="Pal", display_mode=True)

        self.assertIn("one short natural sentence", voice)
        self.assertIn("Do not use Markdown", voice)
        self.assertIn("complete answer", text)
        self.assertIn("under 120 words and six sentences", text)
        self.assertIn("do not guess", voice)
        self.assertEqual(FAST_CONVERSATION_TEMPERATURE, 0.0)
        self.assertEqual(FAST_CONVERSATION_TOP_P, 1.0)

    def test_runtime_answer_preserves_exact_backend_names_and_runtime_voice(self):
        receipt = {
            "llm": {"model": "qwen3.8-27b", "devices": ["cuda:0"]},
            "asr": {"backend": "parakeet-tdt-0.6b-v3-int8", "device": "cpu"},
            "tts": {
                "backend": "piper", "device": "cpu",
                "runtime_voice": "kristin", "stored_active_profile": "scarlet",
                "stored_profile_is_runtime_active": False,
                "runtime_change_required": (
                    "restart Friday with a compatible OmniVoice runtime profile"),
            },
        }

        answer = format_runtime_answer(receipt, ("model", "asr", "tts"))

        self.assertIn("qwen3.8-27b", answer)
        self.assertIn("CUDA device 0", answer)
        self.assertIn("parakeet-tdt-0.6b-v3-int8", answer)
        self.assertIn("Piper", answer)
        self.assertIn("kristin", answer)
        self.assertIn("scarlet is stored but is not the audible voice", answer)
        self.assertIn("OmniVoice", answer)


if __name__ == "__main__":
    unittest.main()
