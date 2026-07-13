from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Literal

from typing_extensions import TypedDict

from livekit import rtc

from .. import llm as llm_module, utils
from ..llm import LLM, ChatContext, FunctionToolCall, function_tool
from ..llm.utils import parse_function_arguments
from ..log import logger
from ..stt.stt import STT
from ..utils import aio

if TYPE_CHECKING:
    from ..metrics import LLMMetrics
    from .agent_session import AgentSession
    from .events import ConversationItemAddedEvent


class KeytermsOptions(TypedDict, total=False):
    """Keyterm biasing for STTs that accept a term list.

    Can be passed as a plain dict::

        AgentSession(
            keyterms_options={
                "keyterms": ["LiveKit", "Acme Corp"],
                "keyterm_detection": {"enabled": True, "turn_interval": 1},
            },
        )
    """

    keyterms: list[str]
    """Static keyterms applied wherever the STT accepts a term list; never touched by detection."""
    keyterm_detection: KeytermDetectionOptions
    """LLM-based keyterm extraction, for STTs that accept a term list."""


class KeytermDetectionOptions(TypedDict, total=False):
    """Configuration for automatic keyterm detection.

    Lives under the ``keyterm_detection`` key of :class:`KeytermsOptions`. Absent or
    ``{"enabled": False}`` keeps detection off.
    """

    enabled: bool
    """Whether to run the background detector. Defaults to ``False``."""
    llm: LLM | str | None
    """LLM used for extraction. An ``LLM`` instance, or a model string (e.g.
    ``"google/gemini-3.5-flash"``) resolved via the inference gateway. Defaults to a
    built-in detection model; the agent's own LLM is not used."""
    turn_interval: int
    """Run a pass once per N user turns. Defaults to ``1``."""
    max_keyterms: int | None
    """Cap on the confirmed (applied) detected keyterms if provided. Defaults to ``None``."""
    instructions: str | None
    """Override the built-in extraction prompt."""
    timeout: float
    """Seconds a single detection pass may run before it is dropped (no keyterm change).
    Defaults to ``10.0``. Raise it if a slow detection ``llm`` needs longer."""


# bound a single pass so a stuck LLM call can't hold the single-flight guard forever and
# stall detection for the rest of the call; a timed-out pass simply makes no change
_DETECTION_TIMEOUT = 10.0

_KEYTERM_DETECTION_DEFAULTS: KeytermDetectionOptions = {
    "enabled": False,
    "llm": None,
    "turn_interval": 1,
    "max_keyterms": None,
    "instructions": None,
    "timeout": _DETECTION_TIMEOUT,
}

_PENDING_TTL = 3  # a pending term not confirmed within this many passes is dropped
_MAX_TRANSCRIPT_MESSAGES = 12

# default model for keyterm extraction when ``keyterm_detection.llm`` is not set
_DEFAULT_DETECTION_MODEL = "google/gemma-4-31b-it"

# set LK_KEYTERMS_DEBUG=1 to log the input/output of every detection pass
lk_keyterms_debug = int(os.getenv("LK_KEYTERMS_DEBUG", 0))


def _resolve_detection(config: KeytermDetectionOptions | None) -> KeytermDetectionOptions:
    """Return a fully-defaulted keyterm-detection config (``enabled`` defaults to False)."""
    return KeytermDetectionOptions(**{**_KEYTERM_DETECTION_DEFAULTS, **(config or {})})


def _resolve_keyterms_options(config: KeytermsOptions | None) -> KeytermsOptions:
    """Return a fully-defaulted keyterms config."""
    config = config or {}
    return KeytermsOptions(
        keyterms=list(config.get("keyterms", [])),
        keyterm_detection=_resolve_detection(config.get("keyterm_detection")),
    )


def _resolve_detection_llm(configured: LLM | str | None) -> LLM | None:
    """Resolve the configured detection ``llm``: an ``LLM`` instance is used directly; a
    model string (or the default model when unset) is created via the inference gateway."""
    if isinstance(configured, LLM):
        return configured
    model = configured if isinstance(configured, str) else _DEFAULT_DETECTION_MODEL
    try:
        from ..inference import LLM as InferenceLLM

        return InferenceLLM.from_model_string(model)
    except Exception:  # noqa: BLE001 — never let detection setup break the session
        logger.warning("keyterm detection: could not create detection LLM %r; skipping", model)
        return None


_DEFAULT_KEYTERM_INSTRUCTIONS = """\
You maintain STT keyterms that bias a recognizer toward the correct spelling of distinctive \
words (names, places, companies, products, technical terms). Each turn, adjust them with one \
`record_keyterms` call.

A WRONG spelling biases the recognizer for the rest of the call with no recovery, so precision \
beats coverage: apply only a spelling you can CORROBORATE, and when unsure change nothing.

USER lines are raw STT — often wrong, and the same error recurs, so repetition is NOT proof a \
spelling is right. ASSISTANT lines are the agent's own writing: trust the agent's confident use \
of its OWN names (brands, staff, locations) and confirm those promptly — but an assistant merely \
echoing the user's sounds, or hedging about a spelling, does NOT corroborate.

CONFIRM a pending term only when corroborated by one of:
  1. a letter-by-letter spell-out the assistant then accepts WITHOUT reservation — confirm \
exactly those letters, appending nothing;
  2. the assistant's own confident use of that exact distinctive spelling;
  3. an explicit user correction ("no, not X — it's Y").
Recurrence alone never confirms.

HEDGE RULE: if after a spell-out or name read-back the assistant signals the letters may be off \
("for now", "with that caveat", "may have that slightly off", "did I catch that?", "to be \
confirmed", "I don't want to guess", "double-check"), the spelling is unreliable — keep the term \
PENDING and never confirm it, EVEN IF the user replies "yes". Only a cleanly accepted spell-out \
confirms.

Never apply: a user-line word that sounds like a known term (it's that term misheard); a \
distinctive name glued to an ordinary word ("Blue Haven Hotel" — keep the bare name pending); an \
odd phrase only the user says and the assistant never adopts; a fragment left by an interruption; \
ordinary words or fillers.

Report only CHANGES; never re-list an applied term.
  - `pending`: a distinctive term seen but not yet corroborated;
  - `confirm`: a pending term that just met the bar above;
  - `remove`: only a spelling the user just corrected away. Applied terms are otherwise sticky.
If nothing meets the bar this turn, change nothing."""


@function_tool(name="record_keyterms")
async def _record_keyterms(pending: list[str], confirm: list[str], remove: list[str]) -> None:
    """Update the STT keyterms based on the latest transcript.

    Args:
        pending: Distinctive terms seen but not yet trusted — tracked, not applied.
        confirm: Pending terms the transcript has now corroborated — applied.
        remove: Only a spelling the user corrected away; applied terms are otherwise sticky.
    """
    # only used to elicit a structured tool call; never executed
    ...


class KeytermDetector(rtc.EventEmitter[Literal["metrics_collected"]]):
    """Maintains the STT keyterm set and, when enabled, auto-detects keyterms during a call.

    Owned by the :class:`AgentSession` so keyterm state survives agent handoffs. Each agent
    activity binds it to that activity's STT via :meth:`start` and releases it via
    :meth:`aclose`. When detection is on, an LLM extracts distinctive spellings from the
    conversation; only confirmed terms are pushed to the STT, while pending terms are tracked
    (and fed back to the detector) without biasing recognition.
    """

    def __init__(
        self,
        *,
        static_keyterms: list[str] | None = None,
        options: KeytermDetectionOptions | None = None,
    ) -> None:
        super().__init__()
        options = _resolve_detection(options)
        self._detection = options
        self._max_keyterms = options["max_keyterms"]
        self._turn_interval = max(1, options["turn_interval"])
        self._instructions = options["instructions"] or _DEFAULT_KEYTERM_INSTRUCTIONS
        self._detection_timeout = options["timeout"]

        self._static_terms = list(dict.fromkeys(static_keyterms or []))
        self._detected_terms: list[str] = []  # confirmed terms, oldest first (for eviction)
        self._pending_terms: dict[str, int] = {}  # term -> pass it was added (for TTL)
        self._tick = 0  # detection-pass counter

        # bound per agent activity (see start/aclose)
        self._stt: STT | None = None
        self._llm: LLM | None = options["llm"] if isinstance(options["llm"], LLM) else None
        self._session: AgentSession | None = None
        self._turn_count = 0
        self._detect_task: asyncio.Task[None] | None = None

    @property
    def keyterms(self) -> list[str]:
        """The effective list applied to the STT: static terms + confirmed detected terms."""
        return list(dict.fromkeys([*self._static_terms, *self._detected_terms]))

    @property
    def static_keyterms(self) -> list[str]:
        return list(self._static_terms)

    def set_static_keyterms(self, terms: list[str]) -> None:
        self._static_terms = list(dict.fromkeys(terms))
        if self._stt is not None:
            self._stt._update_session_keyterms(self.keyterms)

    def start(self, session: AgentSession, stt: STT) -> None:
        """Bind this activity's STT (always) and start detection (if enabled)."""
        # static keyterms must reach the recognizer even with detection disabled
        if stt is not self._stt:
            self._stt = stt
            if self.keyterms:
                self._stt._update_session_keyterms(self.keyterms)

        if not self._detection["enabled"]:
            return

        # don't waste LLM detection passes when no STT can consume the keyterms
        if not stt.capabilities.keyterms:
            logger.warning(
                "keyterm detection is enabled but the STT does not support keyterms; "
                "skipping detection",
                extra={"stt": stt.label if stt is not None else None},
            )
            return

        detect_llm = _resolve_detection_llm(self._detection["llm"])
        if detect_llm is None:
            logger.warning(
                "keyterm detection is enabled but no detection LLM is available; skipping"
            )
            return

        self._llm = detect_llm
        detect_llm.on("metrics_collected", self._forward_metrics)
        self._session = session
        self._turn_count = 0
        session.on("conversation_item_added", self._on_conversation_item_added)

    async def aclose(self) -> None:
        """Stop detection for the current activity; keyterm state is kept."""
        if self._llm is not None:
            self._llm.off("metrics_collected", self._forward_metrics)
        if self._session is not None:
            self._session.off("conversation_item_added", self._on_conversation_item_added)
            self._session = None
        if self._detect_task is not None:
            await aio.cancel_and_wait(self._detect_task)
            self._detect_task = None

    def _forward_metrics(self, ev: LLMMetrics) -> None:
        self.emit("metrics_collected", ev)

    def _on_conversation_item_added(self, ev: ConversationItemAddedEvent) -> None:
        if (session := self._session) is None:
            return

        item = ev.item
        # keyterm detection triggers on non-empty user turns
        if (
            not isinstance(item, llm_module.ChatMessage)
            or item.role != "user"
            or not item.text_content
        ):
            return

        self._turn_count += 1
        if self._turn_count % self._turn_interval != 0:
            return

        # single-flight: skip while a pass is still running
        if self._detect_task is not None and not self._detect_task.done():
            return

        # snapshot the transcript now so the pass isn't affected by later turns
        self._detect_task = asyncio.create_task(self._run_once(self._snapshot(session)))

    @staticmethod
    def _snapshot(session: AgentSession) -> ChatContext:
        return session.history.copy(
            exclude_config_update=True,
            exclude_function_call=True,
            exclude_handoff=True,
            exclude_empty_message=True,
        )

    @utils.log_exceptions(logger=logger)
    async def _run_once(self, chat_ctx: ChatContext) -> None:
        if not isinstance(self._llm, LLM):
            return

        # show static terms as applied too, or the LLM keeps re-proposing them
        current = (
            [(t, True) for t in self._static_terms]
            + [(t, True) for t in self._detected_terms]
            + [(t, False) for t in self._pending_terms]
        )
        pending, confirm, remove = await _detect_keyterms(
            llm=self._llm,
            chat_ctx=chat_ctx,
            current_keyterms=current,
            instructions=self._instructions,
            timeout=self._detection_timeout,
        )

        before = self.keyterms
        self._tick += 1

        # update the keyterm state
        for term in remove:
            self._pending_terms.pop(term, None)
            if term in self._detected_terms:
                self._detected_terms.remove(term)

        for term in pending:
            # track a new candidate; ignore static terms and ones already known
            if term and term not in self._static_terms:
                if term not in self._detected_terms and term not in self._pending_terms:
                    self._pending_terms[term] = self._tick

        for term in confirm:
            if term and term not in self._static_terms:
                self._pending_terms.pop(term, None)  # promote out of pending
                if term not in self._detected_terms:
                    self._detected_terms.append(term)

        # drop pending terms that were never confirmed in time
        for term in [t for t, s in self._pending_terms.items() if self._tick - s >= _PENDING_TTL]:
            del self._pending_terms[term]

        # evict oldest confirmed terms if over the cap
        if self._max_keyterms is not None:
            while len(self._detected_terms) > self._max_keyterms:
                self._detected_terms.pop(0)

        # update the STT if the keyterms changed
        if (new_keyterms := self.keyterms) != before and self._stt is not None:
            self._stt._update_session_keyterms(new_keyterms)
            before_set, new_set = set(before), set(new_keyterms)
            logger.debug(
                "keyterms changed",
                extra={
                    "added": [t for t in new_keyterms if t not in before_set],
                    "removed": [t for t in before if t not in new_set],
                },
            )


async def _detect_keyterms(
    llm: LLM,
    chat_ctx: ChatContext,
    *,
    instructions: str | None = None,
    current_keyterms: list[tuple[str, bool]] | None = None,
    timeout: float = _DETECTION_TIMEOUT,
) -> tuple[list[str], list[str], list[str]]:
    """Run one extraction pass via a forced function call.

    Returns ``(pending, confirm, remove)``.
    """
    current = current_keyterms or []
    user_msg = _format_input(chat_ctx, current)
    if user_msg is None:  # no transcript yet — nothing to detect
        return [], [], []
    req_ctx = ChatContext.empty()
    req_ctx.add_message(role="system", content=instructions or _DEFAULT_KEYTERM_INSTRUCTIONS)
    req_ctx.add_message(role="user", content=user_msg)
    try:
        response = await asyncio.wait_for(
            llm.chat(chat_ctx=req_ctx, tools=[_record_keyterms], tool_choice="required").collect(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("keyterm detection: pass timed out after %ss; skipping", timeout)
        return [], [], []
    result = _parse_tool_call(response.tool_calls)

    if lk_keyterms_debug:
        _debug_dump(user_msg, result)
    return result


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

    applied = [term for term, ok in current_keyterms if ok]
    candidates = [term for term, ok in current_keyterms if not ok]
    # always show both lists (even empty) so the model has explicit state to diff against
    sections = [
        "## Transcript (USER = raw STT, may be wrong; ASSISTANT = correct spelling)\n"
        + "\n\n".join(turns),  # blank line between turns
        "## Applied keyterms (biasing the recognizer now)\n" + (", ".join(applied) or "(none)"),
        "## Candidate keyterms (seen, not yet applied)\n" + (", ".join(candidates) or "(none)"),
        "Update the keyterms from the latest turns, then call `record_keyterms` once.",
    ]
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


def _debug_dump(
    user_msg: str,
    result: tuple[list[str], list[str], list[str]],
) -> None:
    """Log the input/output of one detection pass (gated by ``LK_KEYTERMS_DEBUG``)."""
    pending, confirm, remove = result
    logger.debug(
        "\n".join(
            [
                "──────── keyterm detection ────────",
                user_msg,
                "──── output ────",
                f"pending: {pending}",
                f"confirm: {confirm}",
                f"remove:  {remove}",
                "───────────────────────────────────",
            ]
        )
    )
