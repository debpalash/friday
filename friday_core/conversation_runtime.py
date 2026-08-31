"""Pure conversation-history validation and prompt compilation."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from typing import Pattern


Message = dict
Turn = list[Message]

_FRAGMENT_ONLY = re.compile(
    r"^(?:i|i\s+am|i['’]m|it|this|that|the|a|an|and|but|or|so|because|to)[.!?]*$",
    re.IGNORECASE,
)


@contextmanager
def conversation_history_scope(owner, history: list[dict] | None):
    """Temporarily bind one isolated conversation under the caller's lock."""
    persistent = owner.history
    if history is not None:
        owner.history = history
    try:
        yield
    finally:
        if history is not None:
            history[:] = owner.history
            owner.history = persistent


def completion_integrity_issue(
    text: str, *, finish_reason: str | None = None,
) -> str | None:
    """Return the reason a model response must not be shown to the user."""
    if finish_reason == "length":
        return "token_limit"
    value = str(text or "").strip()
    if not value:
        return "empty"
    if _FRAGMENT_ONLY.fullmatch(value):
        return "fragment"
    if value.count("```") % 2:
        return "unclosed_code_fence"
    return None


def response_contract_issue(
        text: str, user_text: str, max_words: int | None) -> str | None:
    """Reject overlong or contextually useless short answers."""
    words = re.findall(r"\b[\w'-]+\b", text)
    if max_words is not None and len(words) > max_words:
        return "word_limit"
    if (len(words) <= 3 and re.search(
            r"\b(?:explain|why|based|evidence|verified|plan|exact|tell me|"
            r"recommend|compare)\b", user_text, re.IGNORECASE)):
        return "thin_answer"
    historical_exact = (
        re.search(r"\bexact\b", user_text, re.IGNORECASE)
        and re.search(
            r"\b(?:at\s+\d|last|yesterday|ago|previous)\b",
            user_text, re.IGNORECASE))
    admits_unknown = re.search(
        r"\b(?:cannot|can't|do not know|don't know|unknown|unavailable)\b",
        text, re.IGNORECASE)
    names_basis = re.search(
        r"\b(?:evidence|record|data|measurement|receipt|source|log)\w*\b",
        text, re.IGNORECASE)
    if historical_exact and admits_unknown and not names_basis:
        return "missing_basis"
    return None


def bounded_response_fallback(text: str, max_words: int) -> str:
    """Keep a meaningful bounded prefix when a repair still ignores its limit."""
    matches = list(re.finditer(r"\b[\w'-]+\b", str(text or "")))
    if len(matches) <= max_words:
        return str(text or "").strip()
    prefix = str(text)[:matches[max_words - 1].end()].rstrip(" ,;:-")
    return prefix if prefix.endswith((".", "!", "?")) else prefix + "."


def grounded_project_messages(
        messages: list[dict], user_text: str, receipts: list[str]) -> list[dict]:
    """Present private project receipts as transient context, not tool protocol."""
    if not receipts or not messages:
        return messages
    system = dict(messages[0])
    system["content"] = (
        str(system.get("content") or "")
        + "\n\nCurrent verified project evidence:\n"
        + "\n\n".join(receipts))
    return [system, {"role": "user", "content": user_text}]


def canonical_chat_turn(
    turn: Sequence[Message], *, redacted_tool_receipt: str,
) -> Turn | None:
    """Return one model-safe user turn, or omit the complete damaged turn."""
    if (not turn or turn[0].get("role") != "user"
            or not isinstance(turn[0].get("content"), str)):
        return None
    canonical = [{"role": "user", "content": turn[0]["content"]}]
    pending_calls: set[str] = set()
    seen_calls: set[str] = set()
    for message in turn[1:]:
        role = message.get("role")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if calls:
                if pending_calls or not isinstance(calls, list):
                    return None
                normalized_calls = []
                for call in calls:
                    if not isinstance(call, dict):
                        return None
                    function = call.get("function")
                    call_id = call.get("id")
                    if (not isinstance(function, dict)
                            or not isinstance(call_id, str) or not call_id
                            or call_id in seen_calls):
                        return None
                    name = function.get("name")
                    arguments = function.get("arguments")
                    if (not isinstance(name, str) or not name
                            or not isinstance(arguments, str)):
                        return None
                    try:
                        parsed_arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        return None
                    if (not isinstance(parsed_arguments, dict)
                            or parsed_arguments.get("_FRIDAY_REDACTED") is True):
                        return None
                    pending_calls.add(call_id)
                    seen_calls.add(call_id)
                    normalized_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                parsed_arguments, ensure_ascii=False,
                                separators=(",", ":")),
                        },
                    })
                content = message.get("content")
                if content is not None and not isinstance(content, str):
                    return None
                canonical.append({
                    "role": "assistant", "content": content,
                    "tool_calls": normalized_calls,
                })
            else:
                if pending_calls or not isinstance(message.get("content"), str):
                    return None
                canonical.append({
                    "role": "assistant", "content": message["content"]})
        elif role == "tool":
            call_id = message.get("tool_call_id")
            content = message.get("content")
            if (not isinstance(call_id, str) or call_id not in pending_calls
                    or not isinstance(content, str)
                    or content == redacted_tool_receipt):
                return None
            pending_calls.remove(call_id)
            canonical.append({
                "role": "tool", "tool_call_id": call_id,
                "content": content,
            })
        else:
            return None
    return None if pending_calls else canonical


def echo_turn_signature(turn: Sequence[Message]) -> str | None:
    if (len(turn) != 2 or turn[0].get("role") != "user"
            or turn[1].get("role") != "assistant"
            or turn[1].get("tool_calls")):
        return None
    user = turn[0].get("content")
    assistant = turn[1].get("content")
    if not isinstance(user, str) or not isinstance(assistant, str):
        return None
    normalized_user = re.sub(r"[^\w]+", " ", user.casefold()).strip()
    normalized_assistant = re.sub(
        r"[^\w]+", " ", assistant.casefold()).strip()
    if (not normalized_user or len(normalized_user) > 80
            or normalized_user != normalized_assistant):
        return None
    return normalized_user


def drop_repeated_echo_turns(turns: Sequence[Turn]) -> list[Turn]:
    """Remove only sustained identical user/assistant feedback cycles."""
    kept: list[Turn] = []
    index = 0
    while index < len(turns):
        signature = echo_turn_signature(turns[index])
        end = index + 1
        if signature is not None:
            while (end < len(turns)
                   and echo_turn_signature(turns[end]) == signature):
                end += 1
        if signature is None or end - index < 3:
            kept.extend(turns[index:end])
        index = end
    return kept


def short_reply_turn_signature(turn: Sequence[Message]) -> str | None:
    """Identify short repeated assistant replies that carry no durable context."""
    if (len(turn) != 2 or turn[0].get("role") != "user"
            or turn[1].get("role") != "assistant"
            or turn[1].get("tool_calls")):
        return None
    user = turn[0].get("content")
    assistant = turn[1].get("content")
    if not isinstance(user, str) or not isinstance(assistant, str):
        return None
    normalized = re.sub(r"[^\w]+", " ", assistant.casefold()).strip()
    if (not normalized or len(user.strip()) > 32 or len(normalized) > 24
            or len(normalized.split()) > 4):
        return None
    return normalized


def _short_user_turn_carries_context(turn: Sequence[Message]) -> bool:
    """Keep compact user facts even when the model answered them vacuously."""
    if not turn or turn[0].get("role") != "user":
        return False
    content = turn[0].get("content")
    if not isinstance(content, str):
        return False
    tokens = re.findall(r"[\w'-]+", content, re.UNICODE)
    return len(tokens) >= 2 or any(
        len(token) >= 7 or any(character.isdigit() for character in token)
        for token in tokens
    )


def drop_repeated_short_reply_turns(turns: Sequence[Turn]) -> list[Turn]:
    """Remove vacuous reply runs without deleting meaningful user context."""
    kept: list[Turn] = []
    index = 0
    while index < len(turns):
        signature = short_reply_turn_signature(turns[index])
        end = index + 1
        if signature is not None:
            while (end < len(turns)
                   and short_reply_turn_signature(turns[end]) == signature):
                end += 1
        if signature is None or end - index < 3:
            kept.extend(turns[index:end])
        else:
            kept.extend(
                [turn[0]] for turn in turns[index:end]
                if _short_user_turn_carries_context(turn)
            )
        index = end
    return kept


def group_user_turns(messages: Iterable[Message]) -> list[Turn]:
    turns: list[Turn] = []
    for message in messages:
        if message.get("role") == "user":
            turns.append([message])
        elif turns:
            turns[-1].append(message)
    return turns


def drop_repeated_echo_messages(messages: Sequence[Message]) -> list[Message]:
    prefix: list[Message] = []
    turns: list[Turn] = []
    for message in messages:
        if message.get("role") == "user":
            turns.append([message])
        elif turns:
            turns[-1].append(message)
        else:
            prefix.append(message)
    turns = drop_repeated_echo_turns(turns)
    return prefix + [message for turn in turns for message in turn]


def compile_chat_messages(
    history: Sequence[Message], *, base_prompt: str,
    local_time: str, context_sections: Sequence[str], history_turns: int,
    redacted_tool_receipt: str, synthetic_fallbacks: set[str],
    stale_capability_denial: Pattern[str], ungrounded_action_claim: Pattern[str],
) -> list[Message]:
    prompt = base_prompt + "\n\nCurrent local time: " + local_time
    sections = [section.strip() for section in context_sections
                if section and section.strip()]
    if sections:
        prompt += "\n\nRuntime context:\n\n" + "\n\n".join(sections)
    conversation = [message for message in history[1:]
                    if message.get("role") != "system"]
    raw_turns = group_user_turns(conversation)
    cleaned_turns: list[Turn] = []
    for index, raw_turn in enumerate(raw_turns):
        turn = canonical_chat_turn(
            raw_turn, redacted_tool_receipt=redacted_tool_receipt)
        if turn is None:
            continue
        if any(message.get("role") == "assistant"
               and (message.get("content") in synthetic_fallbacks
                    or stale_capability_denial.search(
                        str(message.get("content") or ""))
                    or ungrounded_action_claim.search(
                        str(message.get("content") or "")))
               for message in turn):
            continue
        if any(message.get("role") == "tool"
               and str(message.get("content") or "").startswith("error:")
               for message in turn):
            continue
        has_tools = any(message.get("role") == "tool"
                        or message.get("tool_calls") for message in turn)
        final = turn[-1]
        complete = (final.get("role") == "assistant"
                    and not final.get("tool_calls")
                    and completion_integrity_issue(
                        str(final.get("content") or "")) is None)
        current_user_only = index == len(raw_turns) - 1 and len(turn) == 1
        current_tool_receipt = (
            index == len(raw_turns) - 1 and has_tools
            and final.get("role") == "tool")
        if (not has_tools and (complete or current_user_only)) or (
                has_tools and (complete or current_tool_receipt)):
            cleaned_turns.append(turn)
    cleaned_turns = drop_repeated_short_reply_turns(
        drop_repeated_echo_turns(cleaned_turns))
    cleaned = [message for turn in cleaned_turns[-history_turns:]
               for message in turn]
    return [{"role": "system", "content": prompt}] + cleaned


def compile_fast_chat_messages(
    history: Sequence[Message], *, system_prompt: str,
    history_turns: int, context_chars: int, redacted_tool_receipt: str,
) -> list[Message]:
    conversation = [message for message in history[1:]
                    if message.get("role") != "system"]
    raw_turns = group_user_turns(conversation)
    plain_turns: list[Turn] = []
    for index, raw_turn in enumerate(raw_turns):
        turn = canonical_chat_turn(
            raw_turn, redacted_tool_receipt=redacted_tool_receipt)
        if turn is None or any(
                message.get("role") == "tool" or message.get("tool_calls")
                for message in turn):
            continue
        final = turn[-1]
        complete = (final.get("role") == "assistant"
                    and completion_integrity_issue(
                        str(final.get("content") or "")) is None)
        current_user_only = index == len(raw_turns) - 1 and len(turn) == 1
        if complete or current_user_only:
            plain_turns.append(turn)
    plain_turns = drop_repeated_short_reply_turns(
        drop_repeated_echo_turns(plain_turns))
    selected: list[Turn] = []
    used_chars = 0
    for turn in reversed(plain_turns[-history_turns:]):
        turn_chars = sum(len(str(message.get("content") or ""))
                         for message in turn)
        if selected and used_chars + turn_chars > context_chars:
            break
        selected.append(turn)
        used_chars += turn_chars
    selected.reverse()
    return ([{"role": "system", "content": system_prompt}]
            + [message for turn in selected for message in turn])


def is_action_request(messages: Sequence[Message], pattern: Pattern[str]) -> bool:
    latest_user = next(
        (str(message.get("content") or "") for message in reversed(messages)
         if message.get("role") == "user"), "")
    return bool(pattern.search(latest_user))


def latest_user_only(messages: Sequence[Message]) -> list[Message]:
    latest_user = next(
        (message for message in reversed(messages[1:])
         if message.get("role") == "user"), None)
    return [messages[0]] + ([latest_user] if latest_user else [])
