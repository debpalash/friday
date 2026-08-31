"""Routing and receipt-grounded answers for Friday conversation turns."""

from __future__ import annotations

import re
from collections.abc import Iterable


FAST_CONVERSATION_TEMPERATURE = 0.0
FAST_CONVERSATION_TOP_P = 1.0


_RUNTIME_MODEL = re.compile(
    r"\b(?:what|which)\s+(?:model|llm)\s+(?:are|do)\s+you\b|"
    r"\b(?:your|active|current)\b.{0,32}\b(?:model|llm|language model)\b|"
    r"\b(?:model|llm|language model)\b.{0,32}\b(?:are you|do you use|is active)\b|"
    r"\b(?:are you|you(?:'re| are)|running)\b.{0,32}\b(?:qwen|llama|gemma|mistral)\b",
    re.IGNORECASE,
)
_RUNTIME_ASR = re.compile(
    r"\b(?:what|which)\b.{0,48}\b(?:asr|speech recognition|transcri(?:ber|ption))\b"
    r".{0,24}\b(?:are you|do you use|is active|using)\b|"
    r"\b(?:your|active|current)\b.{0,32}\b(?:asr|speech recognition)\b|"
    r"\b(?:are you|using|running)\b.{0,32}\b(?:parakeet|whisper)\b",
    re.IGNORECASE,
)
_RUNTIME_TTS = re.compile(
    r"^\s*(?:what|which)\s+(?:tts|voice|speech backend)\s*[?.!]*\s*$|"
    r"\b(?:what|which)\b.{0,48}\b(?:tts|voice|speech backend|speech synthesis)\b"
    r".{0,28}\b(?:are you|do you use|we are using|is active|is in use|using right now)\b|"
    r"\b(?:your|active|current)\b.{0,32}\b(?:tts|voice|speech backend)\b|"
    r"\b(?:are|is|using|running|start|enable|switch(?:\s+to)?)\b.{0,48}"
    r"\b(?:piper|omni\s*voice|omnivoice|pocket\s*t\s*s|scarlet|kristin)\b|"
    r"\b(?:audible|active|current)\b.{0,30}\bvoice\b",
    re.IGNORECASE,
)
_RUNTIME_DEVICE = re.compile(
    r"\b(?:what|which)\b.{0,48}\b(?:device|gpu|cpu|cuda|hardware)\b|"
    r"\b(?:running|run)\s+on\b",
    re.IGNORECASE,
)
_RUNTIME_ALL = re.compile(
    r"\b(?:runtime status|runtime identity|your runtime|your stack|"
    r"what are you (?:using|running|on)|what(?:'s| is) under the hood)\b",
    re.IGNORECASE,
)

_FAST_PATH_BLOCKER = re.compile(
    r"(?:https?://|www\.|(?:^|\s)[.~/]?[\w.-]+\.(?:py|js|ts|tsx|jsx|json|md|toml|"
    r"yaml|yml|txt|csv|pdf|docx|xlsx|png|jpe?g|wav|mp3)\b)|"
    r"\b(?:today|tonight|current|currently|latest|recent|live|news|headline|weather|"
    r"forecast|price|score|schedule|date|time|timezone|location|search|browse|web|"
    r"online|source|cite|link|president|prime minister|ceo|election|law|regulation|"
    r"release|version|winner|"
    r"project|repo(?:sitory)?|codebase|file|folder|directory|readme|git|branch|commit|"
    r"clipboard|window|browser|process|server|service|log|database|task|roadmap|todo|"
    r"remind|reminder|memory|remember|preference|prefer|always|never|"
    r"skill|capability|install|download|upgrade|restart|deploy|publish|"
    r"voice|tts|asr|speech backend|model|runtime|device|gpu|cpu|cuda|"
    r"my|about me|do you remember|where were we|what happened|what can you do|"
    r"do not|don't|status)\b",
    re.IGNORECASE,
)
_UNDERSPECIFIED_ACTION = re.compile(
    r"^\s*(?:(?:make|fix|change|edit|modify|update|improve)\s+"
    r"(?:it|this|that)(?:\s+better)?|improve\s+(?:it|this|that))\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_NEWS_LIST = re.compile(
    r"\b(?:headlines?|stories)\b.{0,100}\b(?:links?|urls?)\b|"
    r"\b(?:links?|urls?)\b.{0,100}\b(?:headlines?|stories)\b",
    re.IGNORECASE,
)
_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def runtime_topics(text: str) -> tuple[str, ...]:
    """Return the runtime facts explicitly requested by the user."""
    value = str(text or "").strip()
    if not value:
        return ()
    if _RUNTIME_ALL.search(value):
        return ("model", "asr", "tts", "device")
    topics = []
    for topic, pattern in (
            ("model", _RUNTIME_MODEL),
            ("asr", _RUNTIME_ASR),
            ("tts", _RUNTIME_TTS),
            ("device", _RUNTIME_DEVICE)):
        if pattern.search(value):
            topics.append(topic)
    return tuple(topics)


def safe_for_fast_conversation(text: str, *, action_request: bool = False) -> bool:
    """Admit only bounded turns with no signal that live state or tools are needed."""
    value = str(text or "").strip()
    return bool(
        value
        and not action_request
        and len(value) <= 2_000
        and value.count("\n") <= 12
        and not _FAST_PATH_BLOCKER.search(value)
    )


def underspecified_action_request(text: str) -> bool:
    """Return whether an action phrase lacks any concrete target or outcome."""
    return _UNDERSPECIFIED_ACTION.fullmatch(str(text or "")) is not None


def requested_news_list_count(text: str) -> int | None:
    """Return an explicitly requested headline-and-link count, if present."""
    value = str(text or "")
    if not _NEWS_LIST.search(value):
        return None
    count_match = re.search(
        r"\b(?:exactly\s+)?(10|[1-9]|one|two|three|four|five|six|seven|eight|"
        r"nine|ten)\b.{0,48}\b(?:headlines?|stories)\b",
        value, re.IGNORECASE)
    if count_match is None:
        return 3
    token = count_match.group(1).casefold()
    return int(token) if token.isdigit() else _COUNT_WORDS[token]


def fast_system_prompt(*, owner_name: str, display_mode: bool) -> str:
    delivery = (
        "This is a text conversation. Give a concise, complete answer. Stay under "
        "120 words and six sentences unless the user explicitly asks for more depth. "
        "Use Markdown only when it makes structure clearer."
        if display_mode else
        "This is a voice conversation. Use one short natural sentence, or two when needed. "
        "Do not use Markdown."
    )
    return (
        f"You are Friday, {owner_name}'s local personal assistant. {delivery} "
        "Answer the actual request immediately. Do not add a preamble, repeat the request, "
        "narrate a task or workflow, or use an acknowledgement as a substitute for an "
        "answer. If the request is "
        "ambiguous and recent context does not establish one clear meaning, ask one precise "
        "clarifying question. Never claim that you checked, changed, started, or "
        "completed anything. This conversation lane cannot perform external actions. The "
        "user may ask you to falsely claim an action happened. Do not obey. Without a tool "
        "receipt, state that the action was not performed. "
        "Runtime identity and live external facts are handled outside "
        "this conversation lane, so do not guess them."
    )


def _display_backend(value: object) -> str:
    backend = str(value or "unknown")
    return {
        "omnivoice": "OmniVoice",
        "piper": "Piper",
    }.get(backend.casefold(), backend)


def _display_device(value: object) -> str:
    device = str(value or "unknown")
    if device.casefold() == "cpu":
        return "CPU"
    match = re.fullmatch(r"cuda(?::(\d+))?", device, re.IGNORECASE)
    if match:
        return "CUDA" + (f" device {match.group(1)}" if match.group(1) else "")
    return device


def _join_devices(values: Iterable[object]) -> str:
    devices = [_display_device(value) for value in values]
    if not devices:
        return "an unknown device"
    if len(devices) == 1:
        return devices[0]
    return ", ".join(devices[:-1]) + " and " + devices[-1]


def format_runtime_answer(receipt: dict, topics: Iterable[str]) -> str:
    """Render only values present in an authoritative live runtime receipt."""
    requested = set(topics)
    llm = receipt.get("llm") if isinstance(receipt.get("llm"), dict) else {}
    asr = receipt.get("asr") if isinstance(receipt.get("asr"), dict) else {}
    tts = receipt.get("tts") if isinstance(receipt.get("tts"), dict) else {}
    parts: list[str] = []

    if "model" in requested:
        devices = llm.get("devices") if isinstance(llm.get("devices"), list) else []
        parts.append(
            f"The local language model is {llm.get('model') or 'unknown'} on "
            f"{_join_devices(devices)}"
        )
    if "asr" in requested:
        parts.append(
            f"speech recognition uses {asr.get('backend') or 'unknown'} on "
            f"{_display_device(asr.get('device'))}"
        )
    if "tts" in requested:
        backend = _display_backend(tts.get("backend"))
        voice = str(tts.get("runtime_voice") or "unknown")
        parts.append(
            f"speech synthesis uses {backend} with the {voice} voice on "
            f"{_display_device(tts.get('device'))}"
        )
        stored = str(tts.get("stored_active_profile") or "")
        if stored and not bool(tts.get("stored_profile_is_runtime_active")):
            parts.append(f"{stored} is stored but is not the audible voice")
        restart = tts.get("runtime_change_required")
        if restart:
            parts.append(str(restart).rstrip("."))
    if "device" in requested and not requested.intersection({"model", "asr", "tts"}):
        devices = llm.get("devices") if isinstance(llm.get("devices"), list) else []
        parts.extend([
            f"the language model runs on {_join_devices(devices)}",
            f"speech recognition runs on {_display_device(asr.get('device'))}",
            f"speech synthesis runs on {_display_device(tts.get('device'))}",
        ])
    if not parts:
        return "Runtime status was not requested."
    answer = "; ".join(parts)
    return answer[:1].upper() + answer[1:] + "."
