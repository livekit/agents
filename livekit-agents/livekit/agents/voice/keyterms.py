from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field
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

# Confirmation status of an auto-detected keyterm. Only confirmed terms
# (implicit/explicit) are applied to the STT — an unconfirmed term has only been
# seen once and might be a misrecognition, so it is tracked but not yet applied.
KeytermConfirmation = Literal["unconfirmed", "implicit", "explicit"]
_CONFIRMED: tuple[KeytermConfirmation, ...] = ("implicit", "explicit")
_CONFIRMATION_VALUES: tuple[KeytermConfirmation, ...] = ("unconfirmed", "implicit", "explicit")


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
    """Cap on the auto-detected keyterm set. Defaults to ``50``."""
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
    """Holds keyterms and pushes the confirmed set to the active STT.

    Keeps user-defined terms (immutable by auto-detection, always applied) and
    auto-detected terms, each carrying a confirmation status. Only confirmed auto
    terms (``implicit``/``explicit``) are applied to the STT — an unconfirmed term
    has been seen but not yet corroborated, so it is tracked (and fed back to the
    detector) without biasing recognition toward a possible misrecognition.
    """

    def __init__(
        self,
        *,
        user_keyterms: list[str] | None = None,
        max_keyterms: int = DEFAULT_MAX_KEYTERMS,
    ) -> None:
        self._user = list(dict.fromkeys(user_keyterms or []))
        # term -> status, kept in insertion order (oldest first for eviction)
        self._auto: dict[str, KeytermConfirmation] = {}
        self._max_keyterms = max(0, max_keyterms)
        self._stt: STT | None = None

    @property
    def user_keyterms(self) -> list[str]:
        return list(self._user)

    @property
    def auto_entries(self) -> list[tuple[str, KeytermConfirmation]]:
        """All auto-detected terms with their confirmation status (incl. unconfirmed)."""
        return list(self._auto.items())

    @property
    def keyterms(self) -> list[str]:
        """The effective list applied to the STT: user terms + confirmed auto terms."""
        confirmed = [t for t, status in self._auto.items() if status in _CONFIRMED]
        return list(dict.fromkeys([*self._user, *confirmed]))

    @property
    def max_keyterms(self) -> int:
        return self._max_keyterms

    def attach_stt(self, stt: STT | None) -> None:
        """Bind the STT that keyterms are pushed to and push the current set."""
        self._stt = stt
        if stt is not None:
            stt.update_keyterms(self.keyterms)

    def set_user_keyterms(self, terms: list[str]) -> None:
        self._user = list(dict.fromkeys(terms))
        if self._stt is not None:
            self._stt.update_keyterms(self.keyterms)

    def apply_detection(self, upsert: list[tuple[str, KeytermConfirmation]]) -> bool:
        """Apply detector output: add or update auto terms with their status.

        Pushes to the STT only when the *confirmed* (applied) set changes, so status
        churn among unconfirmed terms doesn't trigger reconnects. Returns whether the
        applied set changed.
        """
        before = self.keyterms
        for term, status in upsert:
            # user terms take precedence; reassigning a key keeps its FIFO position
            if term and term not in self._user:
                self._auto[term] = status
        while len(self._auto) > self._max_keyterms:
            del self._auto[next(iter(self._auto))]  # FIFO eviction; user terms untouched

        if self.keyterms != before and self._stt is not None:
            self._stt.update_keyterms(self.keyterms)
        return self.keyterms != before


_DEFAULT_INSTRUCTIONS = """\
You maintain a list of STT keyterms — words and short phrases a speech recognizer is \
likely to mishear — used to bias recognition toward them. Review the current keyterms \
(with their confirmation status) and the recent transcript, then call `record_keyterms` \
exactly once to update the list.

INCLUDE terms that are specific and easily misheard:
- People's names, first and last, especially uncommon ones.
- Street, neighborhood, place, and city names — even when they contain ordinary words \
like "Street" or "Park" (e.g. "Elm Street", "Tewkesbury").
- Company, product, brand, and venue names.
- Specialized jargon, medication or part names, model numbers, and UNCOMMON acronyms.

DO NOT INCLUDE (never record these at any confirmation status, even if the user spoke, \
repeated, emphasized, or corrected them):
- Common words, generic phrases, or full sentences.
- Acronyms spoken as individual letters (e.g. SSO, VPN, MFA, API, FAQ, ID, USB, PDF) — \
a recognizer transcribes letter sequences reliably, so they are never worth biasing, \
even when they are central to the conversation. Only include an acronym if it is \
pronounced as a word and is domain-specific.
- Ordinary conversational vocabulary, including ordinary words a user stresses while \
correcting a misunderstanding (e.g. "afternoon" vs "morning").

The confirmation status below applies ONLY to terms that already pass the inclusion \
rules above — it is never a reason to record an excluded term.

The transcript is raw speech-to-text output and may contain recognition errors, so each \
keyterm carries a confirmation status reflecting how sure you are of its spelling — only \
confirmed terms are actually applied to recognition:
- "unconfirmed": the term has only appeared in the latest user message and has not yet \
been corroborated. It might be a misrecognition; report it, but do not yet trust it.
- "implicit": the user has continued for a turn or more without correcting it (e.g. the \
agent repeated it back and the user moved on), so the spelling appears reliable.
- "explicit": the user spelled it out, repeated it, confirmed it after a read-back, or \
it is the user's own correction of an earlier term.

First apply the include/exclude rules to decide whether a term qualifies at all; only \
qualifying terms are ever recorded. For each qualifying term, record it with its current \
status — including ones you are not yet sure of, marked "unconfirmed", so you can confirm \
them later instead of dropping them — and re-report an existing keyterm whenever its \
status should change (upgrade "unconfirmed" to "implicit"/"explicit" as evidence \
accumulates). When the user corrects a term, record the corrected spelling as "explicit" \
and leave the earlier misheard spelling "unconfirmed" so it is never applied."""

_MAX_TRANSCRIPT_MESSAGES = 12


class _KeytermObservation(BaseModel):
    term: str = Field(description="The keyterm spelling.")
    confirmation: KeytermConfirmation = Field(
        description="How confirmed the spelling is; upgrade it as evidence accumulates."
    )


@function_tool(name="record_keyterms")
async def _record_keyterms(keyterms: list[_KeytermObservation]) -> None:
    """Record the STT keyterms observed so far, each with its confirmation status.

    Args:
        keyterms: Every keyterm noticed, to add or update with its confirmation status.
    """
    # only used to elicit a structured tool call; never executed
    ...


class KeytermAnalyzer:
    """Background analyzer that proposes keyterms during a conversation.

    The detection itself (:meth:`detect`) is independent of any session, so it can
    be evaluated/benchmarked directly. :meth:`start` binds the analyzer to a session
    for its lifetime: it triggers on completed user turns (the freshest point for new
    terms, confirmations, and corrections), throttled by ``turn_interval``, and
    applies the result to the session's :class:`KeytermManager`.
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
        *,
        transcript: str,
        current_keyterms: list[tuple[str, KeytermConfirmation]] | None = None,
    ) -> list[tuple[str, KeytermConfirmation]]:
        """Run one extraction pass via a forced function call.

        Returns the observed keyterms as ``(term, confirmation)``. Does not require
        a session.
        """
        chat_ctx = ChatContext.empty()
        chat_ctx.add_message(role="system", content=self._instructions)
        chat_ctx.add_message(role="user", content=_format_input(transcript, current_keyterms or []))
        response = await self._llm.chat(
            chat_ctx=chat_ctx, tools=[_record_keyterms], tool_choice="required"
        ).collect()
        return _parse_tool_call(response.tool_calls)

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
        # trigger on user turns: the user's latest utterance is where new terms,
        # confirmations, and corrections land, making it the most informative point
        # to analyze (and we react right when the user corrects, not a turn late)
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
            transcript = _recent_transcript(chat_ctx)
            if not transcript:
                return
            upsert = await self.detect(transcript=transcript, current_keyterms=manager.auto_entries)
            if upsert:
                manager.apply_detection(upsert)
        except asyncio.CancelledError:
            raise
        except Exception:
            # sandboxed: a failed pass must never affect the session
            logger.exception("keyterm detection pass failed")


def _format_input(transcript: str, current_keyterms: list[tuple[str, KeytermConfirmation]]) -> str:
    parts = [f"Conversation so far:\n{transcript}"]
    if current_keyterms:
        rendered = ", ".join(f"{term} [{status}]" for term, status in current_keyterms)
        parts.append(f"Current keyterms with status: {rendered}")
    else:
        parts.append("No keyterms recorded yet.")
    return "\n\n".join(parts)


def _recent_transcript(chat_ctx: ChatContext) -> str:
    lines: list[str] = []
    for item in chat_ctx.items:
        if not isinstance(item, llm_module.ChatMessage) or item.role not in ("user", "assistant"):
            continue
        if text := item.text_content:
            lines.append(f"{item.role}: {text.strip()}")
    return "\n".join(lines[-_MAX_TRANSCRIPT_MESSAGES:])


def _parse_tool_call(
    tool_calls: list[FunctionToolCall],
) -> list[tuple[str, KeytermConfirmation]]:
    """Parse the `record_keyterms` tool call into a list of (term, confirmation)."""
    call = next((c for c in tool_calls if c.name == _record_keyterms.info.name), None)
    if call is None:
        return []
    try:
        data = parse_function_arguments(call.arguments)
    except ValueError:
        return []

    keyterms: list[tuple[str, KeytermConfirmation]] = []
    for entry in data.get("keyterms", []):
        if not isinstance(entry, dict) or not isinstance(term := entry.get("term"), str):
            continue
        if not term.strip():
            continue
        status = entry.get("confirmation")
        confirmation: KeytermConfirmation = (
            status if status in _CONFIRMATION_VALUES else "unconfirmed"
        )
        keyterms.append((term, confirmation))

    return keyterms
