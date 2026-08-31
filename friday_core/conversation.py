"""Routing and receipt-grounded answers for Friday conversation turns."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


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
_CONTEXTUAL_REFINEMENT = re.compile(
    r"^\s*(?:(?:make|rewrite|rephrase|revise|say|explain|summari[sz]e)\s+"
    r"(?:it|this|that|the answer|your answer|that answer)"
    r"(?:\s+(?:more\s+)?(?:brief|casual|clear|concise|direct|formal|friendly|"
    r"natural|polished|precise|simple|specific|technical|terse|warm|"
    r"beginner[- ]friendly|shorter|longer|better))?"
    r"|(?:shorten|expand|simplify|clarify)\s+(?:it|this|that|the answer|"
    r"your answer|that answer))\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_MEMORY_REQUEST = re.compile(
    r"\b(?:remember|don't forget|do not forget|from now on|"
    r"i (?:prefer|always want)|my preference is)\b",
    re.IGNORECASE,
)
_DECLARATIVE_CONTEXT = re.compile(
    r"^\s*(?:i(?:'m| am| have|'ve| chose| choose| decided| ruled out|"
    r"picked| plan to| intend to)\b|my\s+(?:goal|goals|priority|priorities|"
    r"constraint|constraints|plan|plans|choice|decision)\b|"
    r"we(?:'re| are| have|'ve| chose| decided| ruled out| plan to)\b)",
    re.IGNORECASE,
)
_OBSERVATION_TOOLS = frozenset({
    "fetch_news", "list_voices", "machine_inspect_process",
    "machine_list_process_specs", "machine_list_windows", "read_web",
    "search_skill_catalog", "web_search",
})
_EMPTY_REFERENT_REPLIES = frozenset({
    "fine", "got it", "hey", "hello", "okay", "ok", "ready", "sure",
})
_EVIDENCE_DETAIL_FOLLOWUP = re.compile(
    r"\b(?:tell me more|more about|details?|open|read|what (?:does|did)\b.{0,32}"
    r"\b(?:say|report)|what caused|why did|how did|according to)\b",
    re.IGNORECASE,
)
_EVIDENCE_REFERENT = re.compile(
    r"\b(?:it|that|this|one|story|article|headline|result|source)\b",
    re.IGNORECASE,
)
_ORDINALS = {
    "first": 0, "1st": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4,
    "sixth": 5, "6th": 5,
    "seventh": 6, "7th": 6,
    "eighth": 7, "8th": 7,
    "ninth": 8, "9th": 8,
    "tenth": 9, "10th": 9,
}
_EVIDENCE_QUERY_STOPWORDS = frozenset({
    "about", "according", "article", "details", "first", "fourth", "headline",
    "more", "ninth", "open", "read", "report", "result", "second", "seventh",
    "sixth", "source", "story", "tell", "tenth", "third", "what", "when",
    "where", "which", "with",
})
_NEWS_LIST = re.compile(
    r"\b(?:headlines?|stories)\b.{0,100}\b(?:links?|urls?)\b|"
    r"\b(?:links?|urls?)\b.{0,100}\b(?:headlines?|stories)\b",
    re.IGNORECASE,
)
_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


class TurnDisposition(str, Enum):
    """The single conversational outcome Friday should pursue for one turn."""

    ANSWER = "answer"
    CLARIFY = "clarify"
    OBSERVE = "observe"
    ACT = "act"
    REMEMBER = "remember"


@dataclass(frozen=True)
class TurnDecision:
    disposition: TurnDisposition
    reason: str


@dataclass(frozen=True)
class EvidenceFollowup:
    status: str
    source_kind: str | None = None
    index: int | None = None
    title: str | None = None
    url: str | None = None


def contextual_refinement_request(text: str) -> bool:
    """Return whether text asks to transform Friday's immediately prior answer."""
    return _CONTEXTUAL_REFINEMENT.fullmatch(str(text or "")) is not None


def declarative_context_update(text: str) -> bool:
    """Return whether the user is supplying context without making a request."""
    value = str(text or "").strip()
    return bool(value and "?" not in value and _DECLARATIVE_CONTEXT.search(value))


def has_conversational_referent(history: Iterable[dict]) -> bool:
    """Return whether a recent, complete assistant answer can resolve a pronoun."""
    for message in reversed(list(history)):
        role = message.get("role")
        if role == "user":
            continue
        if role != "assistant":
            continue
        if message.get("tool_calls"):
            return False
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return False
        normalized = re.sub(r"[^\w]+", " ", content.casefold()).strip()
        return normalized not in _EMPTY_REFERENT_REPLIES
    return False


def decide_turn(
    text: str, *, history: Iterable[dict] = (), action_request: bool = False,
    required_tool: str | None = None,
) -> TurnDecision:
    """Choose one bounded response contract before generation or tool planning."""
    if contextual_refinement_request(text) and has_conversational_referent(history):
        return TurnDecision(TurnDisposition.ANSWER, "contextual_refinement")
    if underspecified_action_request(text):
        return TurnDecision(TurnDisposition.CLARIFY, "missing_target")
    if _MEMORY_REQUEST.search(str(text or "")):
        return TurnDecision(TurnDisposition.REMEMBER, "explicit_memory_request")
    if required_tool in _OBSERVATION_TOOLS:
        return TurnDecision(TurnDisposition.OBSERVE, "live_evidence_required")
    if required_tool is not None or action_request:
        return TurnDecision(TurnDisposition.ACT, "external_action_requested")
    if declarative_context_update(text):
        return TurnDecision(TurnDisposition.ANSWER, "context_update")
    return TurnDecision(TurnDisposition.ANSWER, "direct_response")


def observation_tools_only(tool_names: Iterable[str]) -> bool:
    """Return whether a nonempty tool set is entirely read-only observation."""
    names = tuple(str(name) for name in tool_names)
    return bool(names) and all(name in _OBSERVATION_TOOLS for name in names)


def resolve_evidence_followup(
    text: str, recent_receipt: tuple[str, dict] | None,
) -> EvidenceFollowup:
    """Resolve an article-level follow-up to one exact recent source."""
    value = str(text or "").strip()
    if (recent_receipt is None or not value
            or _EVIDENCE_DETAIL_FOLLOWUP.search(value) is None):
        return EvidenceFollowup("none")
    source_kind, receipt = recent_receipt
    key = "headlines" if source_kind == "news" else "results"
    raw_items = receipt.get(key) if isinstance(receipt, dict) else None
    items = raw_items if isinstance(raw_items, list) else []
    usable = [item for item in items if isinstance(item, dict)
              and str(item.get("url") or "").startswith(("https://", "http://"))]
    ordinal = next(
        (index for token, index in _ORDINALS.items()
         if re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", value,
                      re.IGNORECASE)),
        None,
    )
    numeric = re.search(
        r"\b(?:story|article|headline|result|source|number|#)\s*(10|[1-9])\b",
        value, re.IGNORECASE)
    if ordinal is None and numeric is not None:
        ordinal = int(numeric.group(1)) - 1
    if ordinal is not None:
        if ordinal >= len(usable):
            return EvidenceFollowup("missing", source_kind=source_kind,
                                    index=ordinal)
        selected = usable[ordinal]
        return EvidenceFollowup(
            "selected", source_kind=source_kind, index=ordinal,
            title=str(selected.get("title") or "Source"),
            url=str(selected["url"]))
    query_terms = {
        token for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) >= 4 and token not in _EVIDENCE_QUERY_STOPWORDS
    }
    scored = []
    for index, item in enumerate(usable):
        source_text = " ".join((
            str(item.get("title") or ""), str(item.get("source") or "")))
        source_terms = set(re.findall(r"[a-z0-9]+", source_text.casefold()))
        score = len(query_terms.intersection(source_terms))
        if score:
            scored.append((score, index, item))
    if scored:
        best_score = max(score for score, _index, _item in scored)
        best = [entry for entry in scored if entry[0] == best_score]
        if len(best) == 1:
            _score, index, selected = best[0]
            return EvidenceFollowup(
                "selected", source_kind=source_kind, index=index,
                title=str(selected.get("title") or "Source"),
                url=str(selected["url"]))
    if len(usable) == 1 and (
            _EVIDENCE_REFERENT.search(value) or "tell me more" in value.casefold()):
        selected = usable[0]
        return EvidenceFollowup(
            "selected", source_kind=source_kind, index=0,
            title=str(selected.get("title") or "Source"),
            url=str(selected["url"]))
    if len(usable) > 1 and (
            _EVIDENCE_REFERENT.search(value) or "tell me more" in value.casefold()):
        return EvidenceFollowup("ambiguous", source_kind=source_kind)
    return EvidenceFollowup("none")


def page_receipt_has_article_evidence(receipt: dict, *, min_words: int = 40) -> bool:
    """Return whether a page receipt contains enough body text for article detail."""
    if not isinstance(receipt, dict) or not str(receipt.get("url") or "").startswith(
            ("https://", "http://")):
        return False
    text = re.sub(r"\s+", " ", str(receipt.get("text") or "")).strip()
    if text.casefold() in {"google news", "news", "redirecting", "just a moment"}:
        return False
    return len(re.findall(r"\b[\w'-]+\b", text)) >= max(1, int(min_words))


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
        "answer. When the user gives a declarative context update with no question or "
        "imperative, respond in at most 12 words. Acknowledge or retain it, but do not give "
        "unasked advice, analysis, or options. If a follow-up refers to one clear "
        "recent target, use that target. If two or more recent targets are plausible, ask "
        "which one before requesting content or proposing a change. If the request is "
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
