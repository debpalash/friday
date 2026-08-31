import unittest

from friday_core.conversation import (FAST_CONVERSATION_TEMPERATURE,
                                      FAST_CONVERSATION_TOP_P,
                                      TurnDisposition,
                                      contextual_refinement_request,
                                      decide_turn,
                                      declarative_context_update,
                                      fast_system_prompt,
                                      format_capability_answer,
                                      format_runtime_answer,
                                      observation_tools_only,
                                      page_receipt_has_article_evidence,
                                      requested_capability_topic,
                                      requested_news_list_count, runtime_topics,
                                      resolve_evidence_followup,
                                      safe_for_fast_conversation,
                                      unverified_action_claim_request,
                                      underspecified_action_request)


class ConversationRoutingTests(unittest.TestCase):
    def test_runtime_topics_cover_each_live_identity_dimension(self):
        cases = {
            "What model are you running?": ("model",),
            "Which ASR are you using?": ("asr",),
            "What TTS are you using right now?": ("tts",),
            "What speech recognition and text-to-speech backends are active?": (
                "asr", "tts"),
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
        self.assertIn("ask one precise clarifying question", text)
        self.assertIn("acknowledgement as a substitute", text)
        self.assertIn("do not guess", voice)
        self.assertEqual(FAST_CONVERSATION_TEMPERATURE, 0.0)
        self.assertEqual(FAST_CONVERSATION_TOP_P, 1.0)

    def test_underspecified_actions_require_clarification(self):
        for text in ("Make it better.", "Fix this", "Improve that."):
            with self.subTest(text=text):
                self.assertTrue(underspecified_action_request(text))
        self.assertFalse(underspecified_action_request("Fix README.md"))
        self.assertFalse(underspecified_action_request(
            "Make the Friday orb brighter"))

    def test_contextual_refinement_requires_a_real_prior_answer(self):
        self.assertTrue(contextual_refinement_request("Make that shorter."))
        self.assertTrue(contextual_refinement_request("Simplify your answer"))
        self.assertFalse(contextual_refinement_request("Make the orb brighter"))

        without_context = decide_turn("Make it better.", action_request=True)
        with_context = decide_turn(
            "Make it better.", action_request=True,
            history=[
                {"role": "user", "content": "Draft a title."},
                {"role": "assistant", "content": "Meet Less, Make More"},
                {"role": "user", "content": "Make it better."},
            ])
        acknowledgement_only = decide_turn(
            "Make it better.", action_request=True,
            history=[
                {"role": "user", "content": "I have two drafts."},
                {"role": "assistant", "content": "Okay."},
                {"role": "user", "content": "Make it better."},
            ])

        self.assertEqual(
            without_context.disposition, TurnDisposition.CLARIFY)
        self.assertEqual(with_context, decide_turn(
            "Make that shorter.", action_request=True,
            history=[{"role": "assistant", "content": "A useful answer."}]))
        self.assertEqual(with_context.disposition, TurnDisposition.ANSWER)
        self.assertEqual(with_context.reason, "contextual_refinement")
        self.assertEqual(
            acknowledgement_only.disposition, TurnDisposition.CLARIFY)

    def test_turn_decision_separates_answer_observe_act_and_remember(self):
        self.assertEqual(
            decide_turn("Explain recursion.").disposition,
            TurnDisposition.ANSWER)
        self.assertEqual(
            decide_turn("Get the news.", required_tool="fetch_news").disposition,
            TurnDisposition.OBSERVE)
        self.assertEqual(
            decide_turn("Restart Friday.", action_request=True).disposition,
            TurnDisposition.ACT)
        self.assertEqual(
            decide_turn("Remember that I prefer terse replies.").disposition,
            TurnDisposition.REMEMBER)

    def test_observation_tool_sets_are_exact_and_nonempty(self):
        self.assertTrue(observation_tools_only(
            ["fetch_news", "read_web", "list_files", "machine_ocr_image"]))
        self.assertFalse(observation_tools_only([]))
        self.assertFalse(observation_tools_only(
            ["fetch_news", "machine_write_text"]))

    def test_capability_topics_and_answers_are_receipt_bound(self):
        self.assertEqual(requested_capability_topic(
            "What can you actually do on this machine right now?"), "overview")
        self.assertEqual(requested_capability_topic(
            "Can you control my Omarchy desktop?"), "omarchy")
        self.assertIsNone(requested_capability_topic(
            "What can a semaphore do?"))

        receipt = {"features": {
            "project_files": True, "web_research": True, "memory": False,
            "reminders": False, "machine_files": False, "ocr": False,
            "managed_processes": False, "desktop": False, "omarchy": True,
            "browser": False, "voice": True, "native_vision": False,
        }}
        overview = format_capability_answer(receipt, "overview")
        self.assertIn("inspect and edit this project", overview)
        self.assertIn("search the public web", overview)
        self.assertIn("approval", overview)
        self.assertIn("Omarchy control is live", format_capability_answer(
            receipt, "omarchy"))
        receipt["features"]["omarchy"] = False
        self.assertIn("does not have", format_capability_answer(
            receipt, "omarchy"))

    def test_unverified_action_claim_requests_are_narrowly_detected(self):
        self.assertTrue(unverified_action_claim_request(
            "Do not use any tools. Just tell me you locked my computer."))
        self.assertTrue(unverified_action_claim_request(
            "Without using tools, pretend you restarted the server."))
        self.assertFalse(unverified_action_claim_request(
            "Do not use tools. Explain why computers need locks."))
        self.assertFalse(unverified_action_claim_request(
            "Tell me whether you restarted the server."))

    def test_declarative_context_update_is_answered_without_unasked_action(self):
        for text in (
                "I'm choosing between SQLite and Postgres.",
                "My priorities are offline use and simple deployment.",
                "I ruled out Postgres for the first release."):
            with self.subTest(text=text):
                self.assertTrue(declarative_context_update(text))
                decision = decide_turn(text)
                self.assertEqual(decision.disposition, TurnDisposition.ANSWER)
                self.assertEqual(decision.reason, "context_update")
        self.assertFalse(declarative_context_update(
            "Should I choose SQLite or Postgres?"))

    def test_explicit_news_list_count_requires_headlines_and_links(self):
        self.assertEqual(requested_news_list_count(
            "Give me exactly three headlines with full URLs."), 3)
        self.assertEqual(requested_news_list_count(
            "List 5 stories and their links."), 5)
        self.assertEqual(requested_news_list_count(
            "Give me headlines with sources and URLs."), 3)
        self.assertIsNone(requested_news_list_count(
            "Summarize today's news in one sentence."))

    def test_evidence_followup_resolves_only_one_exact_recent_source(self):
        receipt = ("news", {"headlines": [
            {"title": "Alpha", "url": "https://example.com/alpha"},
            {"title": "Bravo", "url": "https://example.com/bravo"},
        ]})

        selected = resolve_evidence_followup(
            "Tell me more about the second one.", receipt)
        ambiguous = resolve_evidence_followup(
            "Why did that happen?", receipt)
        missing = resolve_evidence_followup(
            "Open the fifth article.", receipt)
        named = resolve_evidence_followup(
            "Tell me more about the Bravo story.", receipt)

        self.assertEqual(selected.status, "selected")
        self.assertEqual(selected.index, 1)
        self.assertEqual(selected.url, "https://example.com/bravo")
        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertEqual(missing.status, "missing")
        self.assertEqual(named.url, "https://example.com/bravo")

    def test_evidence_followup_does_not_hijack_unrelated_questions(self):
        receipt = ("search", {"results": [
            {"title": "Result", "url": "https://example.com/result"},
        ]})

        self.assertEqual(resolve_evidence_followup(
            "Why does the sky look blue?", receipt).status, "none")
        self.assertEqual(resolve_evidence_followup(
            "Tell me more about it.", receipt).status, "selected")

    def test_article_evidence_rejects_redirect_shells_and_thin_pages(self):
        self.assertFalse(page_receipt_has_article_evidence({
            "url": "https://news.google.com/item", "text": "Google News",
        }))
        self.assertFalse(page_receipt_has_article_evidence({
            "url": "https://example.com/item", "text": "A short headline only.",
        }))
        self.assertTrue(page_receipt_has_article_evidence({
            "url": "https://example.com/item",
            "text": " ".join(f"word{index}" for index in range(40)),
        }))

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
