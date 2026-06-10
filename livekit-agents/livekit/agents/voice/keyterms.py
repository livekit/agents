from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import TYPE_CHECKING

from typing_extensions import TypedDict

from .. import llm as llm_module
from ..llm import LLM, ChatContext, FunctionToolCall, function_tool
from ..llm.utils import parse_function_arguments
from ..log import logger
from ..stt import STT
from ..utils import aio
from .events import ConversationItemAddedEvent

if TYPE_CHECKING:
    from .agent_session import AgentSession


DEFAULT_MAX_KEYTERMS = 50
DEFAULT_TURN_INTERVAL = 1
# A pending term not confirmed within this many passes is dropped (confirmed terms persist).
DEFAULT_PENDING_TTL = 3


class KeytermDetectionOptions(TypedDict, total=False):
    """Configuration for automatic keyterm detection.

    Lives under the ``detection`` key of :class:`KeytermsOptions`. Absent or
    ``{"enabled": False}`` keeps the analyzer off.
    """

    enabled: bool
    """Whether to run the background analyzer. Defaults to ``False``."""
    llm: LLM | None
    """LLM used for extraction. Defaults to the active agent activity's LLM."""
    turn_interval: int
    """Run a pass once per N user turns. Defaults to ``1``."""
    max_keyterms: int
    """Cap on the confirmed (applied) auto keyterms. Defaults to ``50``."""
    instructions: str | None
    """Override the built-in extraction prompt."""


class KeytermOptions(TypedDict, total=False):
    """Keyterm configuration for an :class:`AgentSession`.

    Can be passed as a plain dict::

        AgentSession(
            keyterm_options={
                "terms": ["LiveKit", "Acme Corp"],
                "detection": {"enabled": True, "turn_interval": 1},
            },
        )
    """

    terms: list[str]
    """User-defined keyterms. Never modified by auto-detection."""
    detection: KeytermDetectionOptions
    """Automatic keyterm detection configuration."""


_KEYTERM_DETECTION_DEFAULTS: KeytermDetectionOptions = {
    "enabled": False,
    "llm": None,
    "turn_interval": DEFAULT_TURN_INTERVAL,
    "max_keyterms": DEFAULT_MAX_KEYTERMS,
    "instructions": None,
}


def _resolve_detection(config: KeytermDetectionOptions | None) -> KeytermDetectionOptions:
    """Return a fully-defaulted detection config (``enabled`` defaults to False)."""
    if config is None:
        return KeytermDetectionOptions(**_KEYTERM_DETECTION_DEFAULTS)
    return KeytermDetectionOptions(**{**_KEYTERM_DETECTION_DEFAULTS, **config})


class KeytermManager:
    """Tracks keyterms and pushes the confirmed set to the active STT.

    User terms are always applied. Auto-detected terms apply only once confirmed; a pending
    term is tracked (and fed back to the detector) but never biases recognition.
    """

    def __init__(
        self,
        *,
        user_keyterms: list[str] | None = None,
        max_keyterms: int = DEFAULT_MAX_KEYTERMS,
        pending_ttl: int = DEFAULT_PENDING_TTL,
    ) -> None:
        self._user_terms = list(dict.fromkeys(user_keyterms or []))
        self._auto_terms: list[str] = []  # confirmed terms, insertion order (oldest first)
        self._pending_terms: dict[str, int] = {}  # term -> pass it was added (for TTL)

        self._max_keyterms = max(0, max_keyterms)
        self._pending_ttl = max(0, pending_ttl)
        self._tick = 0  # detection-pass counter
        self._stt: STT | None = None

    @property
    def user_keyterms(self) -> list[str]:
        return list(self._user_terms)

    @property
    def auto_entries(self) -> list[tuple[str, bool]]:
        """All auto-detected terms with their confirmed flag (confirmed first, then pending)."""
        return [(t, True) for t in self._auto_terms] + [(t, False) for t in self._pending_terms]

    @property
    def keyterms(self) -> list[str]:
        """The effective list applied to the STT: user terms + confirmed auto terms."""
        return list(dict.fromkeys([*self._user_terms, *self._auto_terms]))

    @property
    def max_keyterms(self) -> int:
        return self._max_keyterms

    def attach_stt(self, stt: STT | None) -> None:
        """Bind the STT that keyterms are pushed to and push the current set."""
        self._stt = stt
        if stt is not None:
            stt.update_keyterms(self.keyterms)

    def set_user_keyterms(self, terms: list[str]) -> None:
        self._user_terms = list(dict.fromkeys(terms))
        if self._stt is not None:
            self._stt.update_keyterms(self.keyterms)

    def update(
        self,
        pending: list[str] | None = None,
        confirm: list[str] | None = None,
        remove: list[str] | None = None,
    ) -> bool:
        """Apply one detector pass: drop ``remove``, track ``pending``, apply ``confirm``.

        A term's state only moves forward; a pending term not confirmed within ``pending_ttl``
        passes ages out. Returns whether the applied (confirmed) set changed.
        """
        before = self.keyterms
        self._tick += 1
        logger.debug(
            f"KeytermManager: pending={pending}, confirm={confirm}, "
            f"remove={remove}, current={self.keyterms}"
        )
        for term in remove or []:
            self._pending_terms.pop(term, None)
            if term in self._auto_terms:
                self._auto_terms.remove(term)

        for term in pending or []:
            # track a new candidate; ignore user terms and ones already known
            if not term or term in self._user_terms:
                continue
            if term not in self._auto_terms and term not in self._pending_terms:
                self._pending_terms[term] = self._tick

        for term in confirm or []:
            if term and term not in self._user_terms:
                self._pending_terms.pop(term, None)  # promote out of pending
                if term not in self._auto_terms:
                    self._auto_terms.append(term)

        self._evict_stale_pending()
        while len(self._auto_terms) > self._max_keyterms:
            self._auto_terms.pop(0)  # FIFO eviction of the oldest confirmed term

        if self.keyterms != before and self._stt is not None:
            self._stt.update_keyterms(self.keyterms)
        return self.keyterms != before

    def _evict_stale_pending(self) -> None:
        """Drop pending terms not confirmed within ``pending_ttl`` passes of being added."""
        if not self._pending_ttl:
            return
        stale = [
            t for t, since in self._pending_terms.items() if self._tick - since >= self._pending_ttl
        ]
        for term in stale:
            del self._pending_terms[term]


_DEFAULT_INSTRUCTIONS = """\
You keep a small set of STT keyterms that bias a speech recognizer toward the correct \
spelling of distinctive words — names, places, companies, products, technical terms. You run \
repeatedly during a live call; each turn you see the recent transcript plus the keyterms \
already applied and the candidates seen so far, and you adjust the set.

CRITICAL: a keyterm you apply BIASES the recognizer toward that spelling. If you apply a \
WRONG spelling, the recognizer keeps producing that wrong spelling — recognition gets worse, \
not better, and the user cannot correct it. So only ever apply a spelling you are confident \
is CORRECT. Missing a keyterm is harmless; applying a wrong one is damaging. When unsure, do \
not apply.

Trust the inputs for different things:
- USER lines are raw speech-to-text: words may be wrong, and the recognizer repeats the SAME \
error every time, so a word recurring in USER lines is NOT proof it is correct.
- ASSISTANT lines are the agent's own writing, so they SPELL real words correctly — but the \
agent only knows what the (possibly wrong) transcript told it, so it can echo a misheard word \
straight back. The assistant gives you correct SPELLING, never correct MEANING.

What makes a term correct is the USER settling on it, so read the user's intent across the \
recent turns. A term needs follow-up before you trust it — its first appearance alone is never \
enough, however certain the spelling looks:
- JUST APPEARED — shows up only in the latest turn, with nothing after it yet: keep it \
`pending`, even a clean spelling. You have not seen the user carry on past it.
- CORRECTING — says "no", "I mean…", spells it out, or the word keeps coming out different: \
the current spelling is wrong. Do not apply it, and `remove` any garbled variant already \
applied. The right spelling is the letters the user spelled, or the clean form the assistant \
offered that the user then accepted.
- ACCEPTING — a later turn bears the term out (the user moved on without correcting it, \
repeated it, or built on it): it is settled. `confirm` it, spelled the way the assistant \
writes it. The follow-up may be several turns back in the transcript you are given, not only \
in a prior call.

Call `record_keyterms` once with three lists:
- `confirm`: a distinctive term the user has SETTLED on (corroborated after its first \
mention), in its correct spelling (spell it as the assistant does). Never confirm a word just \
because it recurs in USER lines or because the assistant echoed it — the user must have \
accepted it.
- `pending`: a distinctive term the user seems to mean but has not settled on yet (just \
appeared, still being corrected, or brand new) — track it, don't apply. (Pending terms drop \
automatically if never confirmed, so you don't need to repeat or clean them up.)
- `remove`: a wrong spelling that must stop biasing the recognizer — a misrecognition the \
user corrected, or one the assistant has since spelled differently.

Do NOT record ordinary language: fillers ("yes", "okay", "thanks"), greetings, common words, \
or generic phrases (e.g. "voice agent", "the meeting"). When in doubt, leave it out. Never \
repeat a term already applied unless you are removing it."""

_MAX_TRANSCRIPT_MESSAGES = 12


@function_tool(name="record_keyterms")
async def _record_keyterms(pending: list[str], confirm: list[str], remove: list[str]) -> None:
    """Update the STT keyterms based on the latest transcript.

    Args:
        pending: Distinctive terms seen but not yet corroborated — tracked, not applied. Put a
            term here on its first appearance (no follow-up yet), even if the spelling looks
            certain; wait for the conversation to bear it out before applying.
        confirm: Terms to apply, in their correct spelling — only once the conversation has
            corroborated the term AFTER its first mention (the user moved on without correcting
            it, repeated it, spelled it, or accepted a read-back). Do not confirm a term whose
            only appearance is the latest turn.
        remove: Wrong spellings to drop so they stop biasing the recognizer — a USER-side
            misrecognition the assistant has since spelled correctly, or one the user fixed.
    """
    # only used to elicit a structured tool call; never executed
    ...


class KeytermAnalyzer:
    """Background analyzer that proposes keyterms during a conversation.

    The detection itself (:meth:`detect`) is independent of any session, so it can
    be evaluated/benchmarked directly. :meth:`start` binds the analyzer to a session
    for its lifetime: it triggers on user turns (the freshest point for new terms,
    confirmations, and corrections), throttled by ``turn_interval``, and applies the
    result to the session's :class:`KeytermManager`.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        turn_interval: int = DEFAULT_TURN_INTERVAL,
        instructions: str | None = None,
    ) -> None:
        self._llm = llm
        self._turn_interval = max(1, turn_interval)
        self._instructions = instructions or _DEFAULT_INSTRUCTIONS
        self._turn_count = 0
        self._task: asyncio.Task[None] | None = None
        self._session: AgentSession | None = None

    async def detect(
        self,
        chat_ctx: ChatContext,
        *,
        current_keyterms: list[tuple[str, bool]] | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Run one extraction pass over the conversation via a forced function call.

        ``chat_ctx`` is the conversation history and ``current_keyterms`` the manager's
        auto entries (``(term, confirmed)``). Returns ``(pending, confirm, remove)``.
        Does not require a session.
        """
        current = current_keyterms or []
        user_msg = _format_input(chat_ctx, current)
        if user_msg is None:  # no transcript yet — nothing to detect
            return [], [], []
        req_ctx = ChatContext.empty()
        req_ctx.add_message(role="system", content=self._instructions)
        req_ctx.add_message(role="user", content=user_msg)
        response = await self._llm.chat(
            chat_ctx=req_ctx, tools=[_record_keyterms], tool_choice="required"
        ).collect()
        result = _parse_tool_call(response.tool_calls)
        _debug_dump(current, user_msg, response.tool_calls, result)
        return result

    def start(self, session: AgentSession) -> None:
        if self._session is not None:
            return
        self._session = session
        session.on("conversation_item_added", self._on_item_added)

    async def aclose(self) -> None:
        if self._session is None:
            return
        self._session.off("conversation_item_added", self._on_item_added)
        self._session = None
        if self._task is not None and not self._task.done():
            await aio.cancel_and_wait(self._task)
        self._task = None

    def _on_item_added(self, ev: ConversationItemAddedEvent) -> None:
        session = self._session
        if session is None:
            return
        # trigger on user turns: the freshest point for new terms and corrections; the prior
        # assistant turn is still in the snapshot for its clean spelling
        if ev.item.type != "message" or ev.item.role != "user":
            return

        self._turn_count += 1
        if self._turn_count % self._turn_interval != 0:
            return

        # single-flight: skip while a pass is still running
        if self._task is not None and not self._task.done():
            return

        # snapshot the transcript now so the pass isn't affected by later turns
        chat_ctx = session.history.copy(
            exclude_config_update=True,
            exclude_function_call=True,
            exclude_handoff=True,
            exclude_empty_message=True,
        )
        self._task = asyncio.create_task(self._run(session._keyterm_manager, chat_ctx))

    async def _run(self, manager: KeytermManager, chat_ctx: ChatContext) -> None:
        try:
            pending, confirm, remove = await self.detect(
                chat_ctx, current_keyterms=manager.auto_entries
            )
            if pending or confirm or remove:
                manager.update(pending, confirm, remove)
        except asyncio.CancelledError:
            raise
        except Exception:
            # sandboxed: a failed pass must never affect the session
            logger.exception("keyterm detection pass failed")


def _format_input(chat_ctx: ChatContext, current_keyterms: list[tuple[str, bool]]) -> str | None:
    """Render the detector's user message: recent transcript + current keyterms.

    Returns ``None`` when the transcript holds no user/assistant text yet.
    """
    # walk newest-first and stop once we have enough, then restore chronological order
    turns: list[str] = []
    for item in reversed(chat_ctx.items):
        if not isinstance(item, llm_module.ChatMessage) or item.role not in ("user", "assistant"):
            continue
        if text := item.text_content:
            # keep the message's line structure but drop blank lines, so the blank line
            # between turns is the only blank line and reliably marks a turn boundary
            body = "\n".join(line for line in text.splitlines() if line.strip())
            turns.append(f"{item.role.upper()}: {body}")
            if len(turns) >= _MAX_TRANSCRIPT_MESSAGES:
                break
    if not turns:
        return None
    turns.reverse()

    sections = [
        "## Transcript (USER = raw STT, may be wrong; ASSISTANT = correct spelling)\n"
        + "\n\n".join(turns)  # blank line between turns
    ]
    applied = [term for term, ok in current_keyterms if ok]
    candidates = [term for term, ok in current_keyterms if not ok]
    if applied:
        sections.append("## Applied keyterms (biasing the recognizer now)\n" + ", ".join(applied))
    if candidates:
        sections.append("## Candidate keyterms (seen, not yet applied)\n" + ", ".join(candidates))
    sections.append("Update the keyterms from the latest turns, then call `record_keyterms` once.")
    return "\n\n".join(sections)  # blank line + ## heading between sections


def _parse_tool_call(
    tool_calls: list[FunctionToolCall],
) -> tuple[list[str], list[str], list[str]]:
    """Parse the `record_keyterms` tool call into (pending, confirm, remove)."""
    fnc = next((c for c in tool_calls if c.name == _record_keyterms.info.name), None)
    if fnc is None:
        return [], [], []
    try:
        data = parse_function_arguments(fnc.arguments)
    except ValueError:
        return [], [], []

    def _terms(key: str) -> list[str]:
        return [t for t in data.get(key, []) if isinstance(t, str) and t.strip()]

    return _terms("pending"), _terms("confirm"), _terms("remove")


# Set LK_KEYTERMS_DEBUG_FILE to a path to append the LLM input/output of every detect pass
# there (for live debugging). Unset -> no-op, no overhead beyond an env lookup.
_DEBUG_FILE_ENV = "LK_KEYTERMS_DEBUG_FILE"


def _debug_dump(
    current_keyterms: list[tuple[str, bool]],
    user_msg: str,
    tool_calls: list[FunctionToolCall],
    result: tuple[list[str], list[str], list[str]],
) -> None:
    path = os.environ.get(_DEBUG_FILE_ENV)
    if not path:
        return
    pending, confirm, remove = result
    current = ", ".join(f"{t} ({'confirmed' if ok else 'pending'})" for t, ok in current_keyterms)
    raw = next(
        (c.arguments for c in tool_calls if c.name == _record_keyterms.info.name), "<no call>"
    )
    block = (
        f"==== detect @ {datetime.now().isoformat(timespec='seconds')} ====\n"
        f"--- current keyterms ---\n{current or '(none)'}\n"
        f"--- input ---\n{user_msg}\n"
        f"--- output ---\n"
        f"pending: {pending}\nconfirm: {confirm}\nremove: {remove}\n"
        f"raw: {raw}\n\n"
    )
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)
    except Exception:
        logger.exception("failed to write keyterm debug record")
