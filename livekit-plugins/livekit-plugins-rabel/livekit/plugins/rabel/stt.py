# Copyright 2026 Rabel Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Parts of this file are adapted from livekit-agents' MultiSpeakerAdapter, Copyright 2023 LiveKit,
# Inc., licensed under the Apache License 2.0.

"""Bell around any streaming speech-to-text.

`rabel.STT(inner)` passes the inner speech-to-text's events on and sends the turn's words so far
to the Rabel API (version 3 of its conversation). When Bell answers with a guess of the turn's
final words, the wrapper emits the guess as one PREFLIGHT_TRANSCRIPT. LiveKit starts the LLM on it
(preemptive generation) and keeps that reply only if the turn's final words match the guess.

A guess is sent only within the session's limits for preemptive generation (`max_retries` and
`max_speech_duration`, read from the running session). It is sent only while LiveKit has two early
starts left after it: one for the caller's final words and one spare for a turn the caller goes on
with after a pause.

The rules the comments cite:

1. Events pass unchanged unless a rule below holds them.
2. An interim that agrees with the live guess (a beginning of it, perhaps ending in part of one of
   its words) is held; one equal to it passes.
3. An interim that differs from the guess only in a last word the speech-to-text is unsure of is
   held an update or two (`hold_line`, `hold_always` and `hold_updates` from the API's `ready`).
4. Any other interim that disagrees with the guess passes and cancels it.
5. The speech-to-text's own early start on the guess's words is held; any other replaces it.
6. Limits: a guess only while LiveKit keeps two early starts after it, counted from commit to
   commit, and before the speech-to-text's own early start in the turn; none where LiveKit makes
   no early reply (preemptive generation off, turns ended by hand, a realtime model, the agent's
   reply under way).
6a. After the words cancel a guess, Bell may guess again. One guess is live at a time, and a new
    one that comes during rule 3's hold waits for its outcome.
7. A final equal to a cancelled guess that LiveKit still remembers is sent first as an early start.
8. A guess is held while LiveKit's own early reply on a final waits for its commit. It is taken if
   the speech-to-text's early start replaces that reply, else dropped.
9. With no early start left after Bell's guess took one, the speech-to-text's early start is held
   and the reply under way runs on.

Fails open: when the API is slow, down, refuses, errors or does not answer version 3, the events
are exactly the inner speech-to-text's and one line is logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import re
import time
import weakref
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any, cast

import aiohttp

from livekit import rtc
from livekit.agents import (
    LanguageCode,
    get_job_context,
    llm,
    stt,
    utils,
)
from livekit.agents.tokenize.basic import split_words
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from ._bell import BellConnection
from ._form import CallForm, final_form, written
from .log import logger

if TYPE_CHECKING:
    from livekit.agents.voice.events import ConversationItemAddedEvent

DEFAULT_BASE_URL = "https://api.rabellabs.com"
# LiveKit's preemptive generation defaults (voice/turn.py, PreemptiveGenerationOptions): at most 3
# early starts a turn (`max_retries`), none after 10 s of speech (`max_speech_duration`). The
# session's own values are read where they can be (`_host_limits`); these are used when they cannot.
MAX_EARLY_STARTS = 3
MAX_SPEECH_S = 10.0
# rule 3: a speech-to-text may change a word it has shown; a last word under this confidence is
# taken as unsure. The API sends the line in `ready.hold_line`; this is used only when it sends no
# usable one.
DEFAULT_HOLD_LINE = 0.9
# rule 3's other two settings, from `ready.hold_always` and `ready.hold_updates`; these are used
# when the API does not send them: the line applies (not any last word), and an interim is held
# one update
DEFAULT_HOLD_ALWAYS = False
DEFAULT_HOLD_UPDATES = 1
# the version of the Rabel API's conversation this client speaks
CONTRACT_VERSION = 3
DEFAULT_AGENT = "default"
MIN_CHANCE_RANGE = (0.67, 0.99)
# a guess that arrives later than this after the words it was made from is not used
# (the Rabel API's docs)
DEFAULT_ANSWER_TIMEOUT = 0.5
_AGENT_NAME = re.compile(r"[a-z0-9_-]{1,64}")
# the report's head start is a number from 0 to the conversation's limit in ms
# (the Rabel API's docs)
_MAX_HEAD_START_MS = 3_600_000

Words = tuple[str, ...]
# the module's class under a name that SpeechStream.__init__'s `stt` parameter does not hide
_RecognizeStream = stt.RecognizeStream


def words_of(text: str | None) -> Words:
    """The words LiveKit compares when it keeps or drops an early reply
    (agent_activity._transcripts_equivalent): split on whitespace and per character for scripts
    without spaces, punctuation ignored, case-folded."""
    if not text:
        return ()
    return tuple(
        w.casefold()
        for w, _, _ in split_words(text, ignore_punctuation=True, split_character=True)
        if w
    )


def _is_begin(a: Words, b: Words) -> bool:
    """a is a beginning of b, or equals it."""
    return len(a) <= len(b) and b[: len(a)] == a


def _part_word(a: Words, b: Words) -> bool:
    """a's words before its last are a beginning of b, and a's last word is a beginning (not all) of
    b's word in that place."""
    n = len(a)
    return (
        1 <= n <= len(b)
        and a[:-1] == b[: n - 1]
        and a[-1] != b[n - 1]
        and b[n - 1].startswith(a[-1])
    )


def _agrees(a: Words, b: Words) -> bool:
    """a agrees with the guess b (rule 2): a is a beginning of b, or differs from one only in a
    part-word at its end."""
    return _is_begin(a, b) or _part_word(a, b)


def _check_agent(agent: str) -> str:
    if not isinstance(agent, str) or not _AGENT_NAME.fullmatch(agent):
        raise ValueError(
            "rabel.STT: agent must be 1 to 64 lower-case letters, digits, '-' or '_' "
            f"(got {agent!r})"
        )
    return agent


def _check_report(report: bool) -> bool:
    if not isinstance(report, bool):
        raise ValueError(f"rabel.STT: report must be True or False (got {report!r})")
    return report


def _check_min_chance(min_chance: float | None) -> float | None:
    """None (not set: the API's balanced setting), or a number clamped to 0.67 to 0.99 as the API
    clamps it (with one logged line when it is moved)."""
    if min_chance is None:
        return None
    if isinstance(min_chance, bool) or not isinstance(min_chance, (int, float)):
        raise ValueError(f"rabel.STT: min_chance must be a number (got {min_chance!r})")
    lo, hi = MIN_CHANCE_RANGE
    if not lo <= min_chance <= hi:
        clamped = min(max(float(min_chance), lo), hi)
        logger.warning(f"Bell: min_chance {min_chance} is outside {lo} to {hi}; {clamped} is used")
        return clamped
    return float(min_chance)


def _check_learn(learn: bool) -> bool:
    if not isinstance(learn, bool):
        raise ValueError(f"rabel.STT: learn must be True or False (got {learn!r})")
    return learn


def _check_answer_timeout(answer_timeout: float) -> float:
    if (
        isinstance(answer_timeout, bool)
        or not isinstance(answer_timeout, (int, float))
        or not answer_timeout > 0
    ):
        raise ValueError(
            f"rabel.STT: answer_timeout must be over 0 seconds (got {answer_timeout!r})"
        )
    return float(answer_timeout)


def _tail_only(w: Words, g: Words) -> bool:
    """w differs from g only in its last word, and is not longer than g."""
    n = len(w)
    return 1 <= n <= len(g) and w[:-1] == g[: n - 1] and w[-1] != g[n - 1]


@dataclasses.dataclass(frozen=True)
class _HostLimits:
    """What LiveKit does with an early start in the session this speech-to-text serves."""

    # LiveKit starts an early reply on an early start at all: preemptive generation is on, turns are
    # not ended by hand (`turn_detection="manual"`), and the reply comes from an LLM (not a realtime
    # model)
    early_replies: bool = True
    max_starts: int = MAX_EARLY_STARTS  # `max_retries`: early starts LiveKit makes in one turn
    max_speech_s: float = MAX_SPEECH_S  # `max_speech_duration`: none after this much speech
    # the agent has a reply under way that the caller has not interrupted: LiveKit starts no early
    # reply until it is interrupted or done (agent_activity's on_preemptive_generation)
    agent_speaking: bool = False


_DEFAULT_LIMITS = _HostLimits()


def _host_limits() -> _HostLimits:
    """The session's settings, as LiveKit resolves them for the running agent (agent_activity: the
    agent's own `turn_handling`, turn detection and LLM override the session's). Read from the job's
    primary session each time, so a change or a handoff to another agent is followed. LiveKit's
    defaults when there is no job or no session, and for any value that cannot be read."""
    try:
        ctx = get_job_context(required=False)
        if ctx is None:
            return _DEFAULT_LIMITS
        session = ctx.primary_session
    except Exception:  # noqa: BLE001 -- no session started for this job
        return _DEFAULT_LIMITS
    try:
        try:
            agent = session.current_agent
        except RuntimeError:  # the session is not running an agent yet
            agent = None
        opts: dict[str, Any] = dict(session.options.preemptive_generation)
        agent_opts = getattr(agent, "_turn_handling", None)
        agent_opts = (
            agent_opts.get("preemptive_generation") if isinstance(agent_opts, dict) else None
        )
        if isinstance(agent_opts, dict):
            opts.update(agent_opts)
        elif isinstance(agent_opts, bool):
            opts["enabled"] = agent_opts

        turn_detection = session.turn_detection
        reply_model = session.llm
        if agent is not None:
            if utils.is_given(agent.turn_detection):
                turn_detection = agent.turn_detection
            if utils.is_given(agent.llm):
                reply_model = agent.llm

        enabled = opts.get("enabled", True)
        max_starts = opts.get("max_retries", MAX_EARLY_STARTS)
        max_speech_s = opts.get("max_speech_duration", MAX_SPEECH_S)
        if isinstance(max_starts, bool) or not isinstance(max_starts, int) or max_starts < 0:
            max_starts = MAX_EARLY_STARTS
        if isinstance(max_speech_s, bool) or not isinstance(max_speech_s, (int, float)):
            max_speech_s = MAX_SPEECH_S
        return _HostLimits(
            early_replies=(
                enabled is not False
                and turn_detection != "manual"
                and (reply_model is None or isinstance(reply_model, llm.LLM))
            ),
            max_starts=max_starts,
            max_speech_s=float(max_speech_s),
            agent_speaking=(
                (speech := session.current_speech) is not None and not speech.interrupted
            ),
        )
    except Exception as e:  # noqa: BLE001
        # a setting that cannot be read never touches the speech-to-text's events
        logger.debug(
            f"Bell: the session's settings could not be read ({e!r}); LiveKit's defaults are used"
        )
        return _DEFAULT_LIMITS


class STT(stt.STT):
    def __init__(
        self,
        stt: stt.STT,
        *,
        agent: str = DEFAULT_AGENT,
        min_chance: float | None = None,
        answer_timeout: float = DEFAULT_ANSWER_TIMEOUT,
        learn: bool = True,
        report: bool = True,
        enabled: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Wrap a streaming speech-to-text with Bell.

        Args:
            stt: The speech-to-text to wrap. It keeps every job it has: its words, its end of turn,
                its own early starts.
            agent: Your agent's name, which Bell learns for: 1 to 64 lower-case letters, digits,
                '-' or '_'. Defaults to "default". "demo-bank" is a ready-made agent, already
                trained on a bank's callers, that every account can use; it never learns.
            min_chance: How sure Bell must be of a guess before it sends it, from 0.67 to 0.99.
                Defaults to None: not sent, and Bell uses its balanced setting. Set, every guess
                must also be at least that sure: fewer guesses, so fewer wrong ones and fewer right
                ones. A value outside is clamped, as the API clamps it, and one line is logged.
            answer_timeout: Seconds after the words it was made from within which a guess is still
                used. Defaults to 0.5. A later guess is dropped and one line is logged.
            learn: False means the calls through this wrapper teach Bell nothing (for a test or a
                benchmark). Defaults to True.
            report: True (the default) tells the Rabel API at each turn's end whether Bell's guess
                was kept or dropped, with the head start of a kept one, so your agent's status can
                show them. No words are added. False leaves it out.
            enabled: False makes the wrapper a plain pass-through.
            api_key: Your Rabel API key. Defaults to the RABEL_API_KEY environment variable.
            base_url: The Rabel API's address. Defaults to RABEL_API_URL, else
                https://api.rabellabs.com.
            http_session: An aiohttp session to use for the API's WebSocket.
        """
        if not stt.capabilities.streaming:
            raise ValueError(
                "rabel.STT needs a streaming speech-to-text "
                "(wrap a batch one in stt.StreamAdapter first)"
            )
        caps = stt.capabilities
        if not caps.chat_context:
            # rule 8 sees LiveKit's commit by the caller's message LiveKit adds to the conversation,
            # which LiveKit forwards only to a speech-to-text that takes the conversation; the items
            # reach the wrapped one only if it takes them
            caps = dataclasses.replace(caps, chat_context=True)
        super().__init__(capabilities=caps)
        self._stt = stt
        self._agent = _check_agent(agent)
        self._min_chance = _check_min_chance(min_chance)
        self._answer_timeout = _check_answer_timeout(answer_timeout)
        self._learn = _check_learn(learn)
        self._report = _check_report(report)
        self._enabled = enabled
        self._api_key = api_key or os.environ.get("RABEL_API_KEY", "")
        self._base_url = base_url or os.environ.get("RABEL_API_URL", DEFAULT_BASE_URL)
        self._http_session = http_session
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()
        if self._enabled and not self._api_key:
            logger.warning(
                "Bell: no Rabel API key (set RABEL_API_KEY); "
                "the speech-to-text passes through unchanged"
            )
            self._enabled = False
        self._stt.on("metrics_collected", self._on_metrics_collected)
        # the wrapped speech-to-text's own metrics are re-emitted; this wrapper adds no
        # second measurement
        self._recognize_metrics_needed = False

    @property
    def wrapped_stt(self) -> stt.STT:
        return self._stt

    @property
    def model(self) -> str:
        return self._stt.model

    @property
    def provider(self) -> str:
        return self._stt.provider

    def update_options(
        self,
        *,
        agent: NotGivenOr[str] = NOT_GIVEN,
        min_chance: NotGivenOr[float | None] = NOT_GIVEN,
        answer_timeout: NotGivenOr[float] = NOT_GIVEN,
        learn: NotGivenOr[bool] = NOT_GIVEN,
        report: NotGivenOr[bool] = NOT_GIVEN,
        enabled: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """Takes effect from the next speech stream (answer_timeout and report at once).
        min_chance=None goes back to Bell's balanced setting."""
        if utils.is_given(agent):
            self._agent = _check_agent(agent)
        if utils.is_given(min_chance):
            self._min_chance = _check_min_chance(min_chance)
        if utils.is_given(answer_timeout):
            self._answer_timeout = _check_answer_timeout(answer_timeout)
        if utils.is_given(learn):
            self._learn = _check_learn(learn)
        if utils.is_given(report):
            self._report = _check_report(report)
        if utils.is_given(enabled):
            self._enabled = bool(enabled) and bool(self._api_key)

    def _hello(self) -> dict[str, Any]:
        """The first message's fields besides the key.

        `min_chance` only when the developer set one.
        """
        hello: dict[str, Any] = {
            "type": "hello",
            "v": CONTRACT_VERSION,
            "agent": self._agent,
            "host": "livekit",
            "learn": self._learn,
        }
        if self._min_chance is not None:
            hello["min_chance"] = self._min_chance
        return hello

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._stt._update_session_keyterms(keyterms)

    def _push_conversation_item(self, ev: ConversationItemAddedEvent) -> None:
        """Passed on to the wrapped speech-to-text when it takes the conversation context.

        Bell sends none of it; the caller's message tells the hookup LiveKit has committed a turn
        (rule 8).
        """
        if self._stt.capabilities.chat_context:
            self._stt._push_conversation_item(ev)
        if getattr(getattr(ev, "item", None), "role", None) == "user":
            for s in list(self._streams):
                s._on_livekit_commit()

    def prewarm(self) -> None:
        self._stt.prewarm()

    def _on_metrics_collected(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)

    async def aclose(self) -> None:
        self._stt.off("metrics_collected", self._on_metrics_collected)

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        return await self._stt.recognize(buffer, language=language, conn_options=conn_options)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        stream = SpeechStream(stt=self, language=language, conn_options=conn_options)
        self._streams.add(stream)
        return stream


@dataclasses.dataclass
class _Turn:
    speech_start: float
    # finals inside the turn (speech-to-texts that segment)
    committed: list[str] = dataclasses.field(default_factory=list)
    words: Words = ()  # the words so far
    # the words sent to Bell, in order
    history: list[Words] = dataclasses.field(default_factory=list)
    starts: int = 0  # early starts LiveKit has made this turn
    stt_starts: int = 0  # of them, the speech-to-text's own (rule 6)
    # Bell's guesses passed on this turn
    # (rule 6a: a new one after the words cancelled the one before)
    guesses: int = 0
    running: Words | None = None  # the words of LiveKit's running early start
    live: Words | None = None  # Bell's guess, while it is the running early start
    bell_sent: bool = False
    held: stt.SpeechEvent | None = None  # rule 3's held interim
    held_n: int = 0  # rule 3: updates held in a row
    sent_heard: bool = False
    last_alt: stt.SpeechData | None = None
    # the speech-to-text's text of the words after its finals, as it last showed them
    shown: str = ""
    # the Bell conversation this turn goes to: the one open when the turn began, else None (a
    # conversation opened mid-turn starts with the next turn)
    conversation: int | None = None
    # when each words-so-far was sent
    heard_at: dict[Words, float] = dataclasses.field(default_factory=dict)
    # rule 8: a guess held while LiveKit's own early reply waits
    held_guess: dict[str, Any] | None = None
    waiting_guess: dict[str, Any] | None = None  # rule 6a: a new guess waiting on rule 3's hold
    # the report (the Rabel API's docs): when Bell's guess was passed on; when the last final
    # arrived and whether Bell's guess was then the running early start with the final words equal
    # to it and LiveKit's remembered text unchanged by the final (and the head start then); whether
    # LiveKit committed the turn while it was
    sent_at: float | None = None
    final_at: float | None = None
    final_kept: bool = False
    head_start_ms: int | None = None
    kept_committed: bool = False


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self, *, stt: STT, language: NotGivenOr[str], conn_options: APIConnectOptions
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._rstt = stt
        self._language = language
        self._turn: _Turn | None = None
        self._bell: BellConnection | None = None
        self._broken = False
        self.counts: dict[str, int] = {}
        # the inner speech-to-text's stream, and the wall-clock time at which LiveKit says the
        # caller's timeline began (from the start_time_offset it sets on this stream); both
        # None until known
        self._inner: _RecognizeStream | None = None
        self._timeline_start: float | None = None
        # what LiveKit remembers of the early starts it was given (audio_recognition:
        # `_audio_preflight_transcript`, reset at each final): the text of the last early start
        # passed since the last final, and whether it was Bell's guess and the words have since
        # cancelled its reply (rule 7)
        self._lk_pre: str | None = None
        self._lk_pre_stale = False
        # whether LiveKit's turn text (audio_recognition: `_audio_transcript`, the finals since its
        # last commit) is not empty; LiveKit's comparison at the final, which the report follows,
        # depends on it
        self._lk_has_text = False
        # rule 8: LiveKit's own early reply made on a final is waiting for LiveKit's commit
        self._lk_waiting = False
        # rule 6: the early starts LiveKit has made since its last commit, as LiveKit counts them
        # (agent_activity: `_preemptive_generation_count`, reset only when a turn is completed): a
        # turn of the speech-to-text's can end and another begin before LiveKit commits, and
        # LiveKit's count runs on over both
        self._lk_starts = 0
        # how this call's speech-to-text has written its final transcripts, for a guess in a
        # language other than English
        self._form = CallForm()
        # what this stream's guesses did for the host, for the summary logged when it
        # closes (numbers only)
        self.summary: dict[str, int] = {
            "turns": 0,
            "guesses": 0,
            "kept": 0,
            "dropped": 0,
            "head_start_ms": 0,
        }

    @property
    def start_time_offset(self) -> float:
        return cast(float, stt.RecognizeStream.start_time_offset.fget(self))  # type: ignore[attr-defined]

    @start_time_offset.setter
    def start_time_offset(self, value: float) -> None:
        # LiveKit sets this on the hookup's stream so the vendor's word times count from the
        # session's start, not from the stream's first audio. The vendor's own stream is the inner
        # one, so it gets the same timeline: now if it exists, else when it is made (and again on
        # every retry's new inner stream).
        stt.RecognizeStream.start_time_offset.fset(self, value)  # type: ignore[attr-defined]
        self._timeline_start = time.time() - value
        if self._inner is not None:
            self._hand_on_offset(self._inner)

    def _hand_on_offset(self, inner: stt.RecognizeStream) -> None:
        if self._timeline_start is None:
            return
        try:
            inner.start_time_offset = max(0.0, time.time() - self._timeline_start)
        except Exception as e:  # noqa: BLE001
            # the vendor's times then count from its first audio, as before
            logger.warning(f"Bell: could not hand the time offset to the speech-to-text ({e!r})")

    def _count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[stt.SpeechEvent]) -> None:
        # the wrapped stream reports usage for this audio and its metrics are re-emitted; only the
        # retry reset is kept
        async for ev in event_aiter:
            if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                self._mark_provider_progress()

    async def _run(self) -> None:
        async def _forward_input(stream: stt.RecognizeStream) -> None:
            async for frame in self._input_ch:
                if isinstance(frame, rtc.AudioFrame):
                    stream.push_frame(frame)
                elif isinstance(frame, self._FlushSentinel):
                    stream.flush()
            with contextlib.suppress(RuntimeError):
                stream.end_input()

        async def _forward_output(stream: stt.RecognizeStream) -> None:
            async for ev in stream:
                self._on_inner_event(ev)

        stream = self._rstt._stt.stream(language=self._language, conn_options=self._conn_options)
        stream._on_provider_progress = self._mark_provider_progress
        self._hand_on_offset(stream)
        self._inner = stream
        if self._rstt._enabled:
            try:
                hello = self._rstt._hello()
                self._bell = BellConnection(
                    url=self._rstt._base_url,
                    api_key=self._rstt._api_key,
                    hello=hello,
                    on_guess=self._on_guess,
                    http_session=self._rstt._http_session,
                )
                self._bell.start()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Bell: could not start ({e!r}); the speech-to-text passes through unchanged"
                )
                self._bell = None
        tasks = [
            asyncio.create_task(_forward_input(stream), name="rabel.SpeechStream.forward_input"),
            asyncio.create_task(_forward_output(stream), name="rabel.SpeechStream.forward_output"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            if self._bell is not None:
                self._log_summary()
                await self._bell.aclose()
            self._inner = None
            await stream.aclose()

    # -------------------------------------------------- the inner speech-to-text's events

    def _emit(self, ev: stt.SpeechEvent) -> None:
        self._event_ch.send_nowait(ev)

    def _on_inner_event(self, ev: stt.SpeechEvent) -> None:
        if self._bell is None or self._broken:
            self._emit(ev)
            return
        try:
            out = self._apply(ev)
        except Exception as e:  # noqa: BLE001
            # a fault of the wrapper's own never touches the inner events
            logger.warning(
                f"Bell: the wrapper failed ({e!r}); "
                "the speech-to-text passes through unchanged from here"
            )
            self._broken = True
            out = [ev]
        for out_ev in out:
            self._emit(out_ev)

    def _apply(self, ev: stt.SpeechEvent) -> list[stt.SpeechEvent]:
        T = stt.SpeechEventType
        if ev.type == T.START_OF_SPEECH:
            self._end_turn()
            self._turn = self._new_turn(ev.speech_start_time or time.time())
            return [ev]
        if ev.type == T.END_OF_SPEECH:
            self._end_turn()
            return [ev]
        if ev.type not in (T.INTERIM_TRANSCRIPT, T.PREFLIGHT_TRANSCRIPT, T.FINAL_TRANSCRIPT):
            return [ev]

        alt = ev.alternatives[0] if ev.alternatives else None
        text = alt.text if alt is not None else ""
        if not text:
            return [ev]
        t = self._turn
        if t is None:
            t = self._turn = self._new_turn(time.time())
        t.last_alt = alt

        if ev.type == T.FINAL_TRANSCRIPT:
            try:
                self._form.final(text)
            except Exception:  # noqa: BLE001
                # the form is read only for another language's guess
                self._count("form_not_learned")
            t.committed.append(text)
            t.words = words_of(" ".join(t.committed))
            t.shown = ""
            t.held = None
            t.waiting_guess = None
            pre = self._lk_pre
            # LiveKit's own comparison at the final (LiveKit's audio_recognition): its turn's text
            # with this final against its remembered early-start text, as strings, both built as
            # f"{turn text} {words}".lstrip() on the same turn text; when they differ it starts its
            # own early reply on the final, and the reply started on Bell's guess is replaced. On a
            # turn text that is not empty that is the final against the early start exactly; on an
            # empty one, both left-stripped.
            if pre is None:
                lk_unchanged = False  # LiveKit remembers "", and a final is never empty here
            elif self._lk_has_text:
                lk_unchanged = text == pre
            else:
                lk_unchanged = text.lstrip() == pre.lstrip()
            self._lk_has_text = True
            # the report: Bell's guess is kept when it is the running early start as the final words
            # arrive, they equal it, and the final leaves LiveKit's remembered text (the guess's)
            # unchanged by LiveKit's own comparison
            t.final_at = time.perf_counter()
            t.final_kept = t.live is not None and t.words == t.live and lk_unchanged
            if t.final_kept and t.sent_at is not None:
                t.head_start_ms = _ms(t.final_at - t.sent_at)
            out = [ev]
            if self._lk_pre_stale and pre is not None and text.strip() == pre.strip():
                # rule 7: LiveKit still remembers the cancelled guess's text, finds this final
                # unchanged and would start no early reply on it; the final goes first as the early
                # start on its own words, as LiveKit makes it
                self._count("final_sent_as_early_start")
                out = [
                    dataclasses.replace(
                        ev,
                        type=T.PREFLIGHT_TRANSCRIPT,
                        alternatives=[dataclasses.replace(a) for a in ev.alternatives],
                    ),
                    ev,
                ]
                changed = True
            else:
                # LiveKit starts its own early reply on a final that changes its text
                # (audio_recognition: transcript_changed)
                changed = pre is None or text.strip() != pre.strip()
            if changed:
                lim = _host_limits()
                if (
                    lim.early_replies
                    and not lim.agent_speaking
                    and self._lk_starts < lim.max_starts
                    and time.time() - t.speech_start <= lim.max_speech_s
                ):
                    # rule 6: LiveKit's own early start on this final (or rule 7's) counts
                    # against its turn
                    self._lk_starts += 1
            self._lk_pre, self._lk_pre_stale = None, False
            if changed or (t.running is not None and t.running == t.words):
                # rule 8: an early reply now waits for LiveKit's commit: LiveKit's own, made on this
                # final, or the running early start (Bell's guess, or the speech-to-text's own)
                # whose words this final equals, which LiveKit keeps
                self._lk_waiting = True
            return out

        w = words_of(" ".join([*t.committed, text]))
        t.words = w
        t.shown = text
        self._send_heard(t, text, ev)

        if ev.type == T.PREFLIGHT_TRANSCRIPT:
            if t.live is not None and w == t.live:
                # rule 5: the reply for these words is already running
                self._count("eager_held_back")
                t.held = None
                return []
            lim = _host_limits()
            if t.bell_sent and t.running is not None and self._lk_starts >= lim.max_starts:
                # rule 9: no early start is left (Bell's guess took one); LiveKit would cancel the
                # running reply and start none, so the reply under way is left to run
                self._count("eager_held_no_start_left")
                return []
            # passed: LiveKit cancels the running early start, then starts on these words if
            # within its limits
            t.live, t.held = None, None
            t.waiting_guess = None
            if (
                lim.early_replies
                and not lim.agent_speaking
                and self._lk_starts < lim.max_starts
                and time.time() - t.speech_start <= lim.max_speech_s
            ):
                t.starts += 1
                t.stt_starts += 1
                self._lk_starts += 1
                t.running = w
            else:
                t.running = None
            self._lk_pre, self._lk_pre_stale = text, False
            if self._lk_waiting:
                # rule 8: the speech-to-text's own early start has replaced LiveKit's waiting reply
                # (as without Bell); a held guess is taken now, as if it arrived after it
                self._lk_waiting = False
                return [ev, *self._release_held(t)]
            return [ev]

        # an interim
        g = t.live
        if g is None:
            if t.running is not None and w != t.running:
                t.running = None  # LiveKit cancels an early start the words have moved past
            return [ev]
        unsure = self._unsure_tail(alt, w, g)
        if t.held is not None:
            t.held = None
            if _agrees(w, g):
                self._count("hold_agreed")
                if t.waiting_guess is not None:
                    t.waiting_guess = None
                    self._count("waiting_guess_dropped")  # rule 6a: the running guess stands
                return [ev] if w == g else []
            if unsure and t.held_n < self._hold_updates():
                self._count("tail_held_again")  # rule 3, a further update
                t.held, t.held_n = ev, t.held_n + 1
                return []
            # rule 7: the guess's reply is cancelled below, its text still remembered
            self._lk_pre_stale = True
            self._count("hold_passed")
            t.live = t.running = None
            return [ev, *self._release_waiting(t)]
        if w == g:
            return [ev]
        if _is_begin(w, g):
            self._count("interim_held_back")  # rule 2
            return []
        if _part_word(w, g):
            self._count("part_word_held_back")  # rule 2, a part-word at the end
            return []
        if unsure:
            self._count("tail_held")  # rule 3
            t.held, t.held_n = ev, 1
            return []
        # rule 7: the guess's reply is cancelled below, its text still remembered
        self._lk_pre_stale = True
        self._count("guess_cancelled_by_words")  # rule 4
        t.live = t.running = None
        return [ev]

    def _unsure_tail(self, alt: stt.SpeechData | None, w: Words, g: Words) -> bool:
        """Rule 3's test: w differs from the guess g only in its last word, and that word is one the
        speech-to-text is unsure of (its confidence under the API's `hold_line`), or any last word
        when the API's `hold_always` is true."""
        if not _tail_only(w, g):
            return False
        b = self._bell
        always = b.hold_always if b is not None else None
        if always is None:
            always = DEFAULT_HOLD_ALWAYS
        if always:
            return True
        conf = None
        if alt is not None and alt.words:
            c = getattr(alt.words[-1], "confidence", None)
            conf = c if isinstance(c, (int, float)) else None
        line = b.hold_line if b is not None else None
        if line is None:
            line = DEFAULT_HOLD_LINE
        return conf is not None and conf < line

    def _hold_updates(self) -> int:
        """How many updates in a row rule 3 may hold an interim.

        The API's `hold_updates`, else one.
        """
        n = self._bell.hold_updates if self._bell is not None else None
        return DEFAULT_HOLD_UPDATES if n is None else n

    def _new_turn(self, speech_start: float) -> _Turn:
        b = self._bell
        return _Turn(
            speech_start=speech_start,
            conversation=b.conversation if b is not None and b.ready else None,
        )

    def _turn_open(self, t: _Turn) -> bool:
        """The turn's conversation is still the open one."""
        b = self._bell
        return (
            b is not None
            and b.ready
            and t.conversation is not None
            and t.conversation == b.conversation
        )

    def _send_heard(self, t: _Turn, text: str, ev: stt.SpeechEvent) -> None:
        assert self._bell is not None
        if not self._turn_open(t):
            t.conversation = None  # its conversation closed: the rest of this turn is not sent
            return
        t.history.append(t.words)
        t.heard_at[t.words] = time.perf_counter()
        t.sent_heard = True
        # only the words: no word times, confidences, event kind or vendor
        self._bell.heard(" ".join([*t.committed, text]))

    def _end_turn(self) -> None:
        t = self._turn
        self._turn = None
        if t is not None and t.held_guess is not None:
            t.held_guess = None
            # rule 8: LiveKit's reply waited the whole turn
            self._count("held_guess_not_sent_turn_over")
        if t is None:
            return
        guess, head_start_ms = self._report_of(t)
        # every guess passed on this turn counts: the last as the report says, each one before it
        # dropped (the words cancelled it, rule 6a)
        dropped_before = max(t.guesses - 1, 0)
        if t.words:
            self.summary["turns"] += 1
        if guess != "none":
            self.summary["guesses"] += 1 + dropped_before
            self.summary[guess] += 1
            self.summary["dropped"] += dropped_before
        if guess == "kept":
            self.summary["head_start_ms"] += head_start_ms or 0
            # in the form and at the level of LiveKit's own "using preemptive generation" line
            # (LiveKit's agent_activity)
            logger.debug("Bell's guess kept", extra={"head_start_ms": head_start_ms})
        if t.sent_heard and self._bell is not None and self._turn_open(t):
            text = " ".join(t.committed) if t.committed else " ".join(t.words)
            if self._rstt._report:
                self._bell.end_of_turn(
                    text, guess=guess, head_start_ms=head_start_ms, dropped_before=dropped_before
                )
            else:
                self._bell.end_of_turn(text)

    def _report_of(self, t: _Turn) -> tuple[str, int | None]:
        """What happened to Bell's guess in the turn (the Rabel API's docs): "none" when none was
        passed on, "kept" with its head start when LiveKit uses the reply started on it, "dropped"
        otherwise."""
        if not t.bell_sent:
            return "none", None
        if t.kept_committed or (t.final_kept and t.live is not None):
            return "kept", t.head_start_ms
        return "dropped", None

    def _log_summary(self) -> None:
        """One line when the stream closes: numbers only, no word of the call."""
        s = self.summary
        with contextlib.suppress(Exception):  # a fault here never reaches the host
            logger.info(
                f"Bell: {s['turns']} turns, {s['guesses']} guesses, {s['kept']} kept, "
                f"{s['dropped']} dropped, "
                f"{s['head_start_ms']} ms head start in all",
                extra={
                    "turns": s["turns"],
                    "guesses": s["guesses"],
                    "kept": s["kept"],
                    "dropped": s["dropped"],
                    "head_start_ms_total": s["head_start_ms"],
                },
            )

    # -------------------------------------------------- LiveKit's commit (rule 8)

    def _on_livekit_commit(self) -> None:
        """LiveKit has added the caller's message to the conversation: it has committed a turn and
        taken (or dropped) the early reply that waited. A guess held for it is not sent now:
        LiveKit's reply to that turn is then the agent's current speech, and LiveKit starts no early
        reply until it is interrupted or done (agent_activity's on_preemptive_generation), while the
        guess's text would stay remembered (rule 7's case) for the next final."""
        try:
            self._lk_waiting = False
            # rule 6: LiveKit's count of early starts starts again at its commit
            self._lk_starts = 0
            # LiveKit's turn text restarts at its commit, so the remembered text can no longer
            # equal a final
            self._lk_pre, self._lk_pre_stale = None, False
            self._lk_has_text = False
            t = self._turn
            if t is not None and t.final_kept and t.live is not None:
                # the report: LiveKit has taken the reply started on Bell's guess
                t.kept_committed = True
            if t is not None and t.held_guess is not None:
                t.held_guess = None
                self._count("held_guess_not_sent_commit")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Bell: LiveKit's commit could not be handled ({e!r})")

    def _release_held(self, t: _Turn) -> list[stt.SpeechEvent]:
        """Rule 8: the held guess, taken as if it arrived now (every check but its arrival time is
        made again)."""
        m, t.held_guess = t.held_guess, None
        if m is None:
            return []
        try:
            ev = self._take_guess(m, released=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Bell: a held guess could not be handled ({e!r})")
            return []
        return [ev] if ev is not None else []

    def _release_waiting(self, t: _Turn) -> list[stt.SpeechEvent]:
        """Rule 6a: the guess that waited on rule 3's hold, taken now that the hold has passed the
        interim (every check but its arrival time is made again)."""
        m, t.waiting_guess = t.waiting_guess, None
        if m is None:
            return []
        try:
            ev = self._take_guess(m, released=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Bell: a waiting guess could not be handled ({e!r})")
            return []
        return [ev] if ev is not None else []

    # -------------------------------------------------- Bell's guesses

    def _on_guess(self, m: dict[str, Any]) -> None:
        try:
            ev = self._take_guess(m)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Bell: a guess could not be handled ({e!r})")
            return
        if ev is not None:
            self._emit(ev)

    def _wrong_data(self, key: str, line: str) -> None:
        self._count(key)
        if self._bell is not None:
            self._bell.log_once("wrong_data", f"Bell: {line}; not used")

    def _take_guess(self, m: dict[str, Any], released: bool = False) -> stt.SpeechEvent | None:
        """A guess from the API as the early start to send, or None. `released`: a guess rule 8
        held, taken again (it arrived in time, so its arrival time is not checked again)."""
        b = self._bell
        assert b is not None
        text, heard, chance = m.get("text"), m.get("heard"), m.get("chance")
        # wrong data (the Rabel API's docs): a guess this client cannot read, one with no chance,
        # one under the min_chance in force, or one made from words this stream never sent
        if not isinstance(text, str) or not text.strip() or not isinstance(heard, str):
            self._wrong_data(
                "guess_unreadable", "the Rabel API sent a guess this client cannot read"
            )
            return None
        if chance is not None and _unit(chance) is None:
            self._wrong_data(
                "guess_unreadable", f"the Rabel API sent a guess with chance {chance!r}"
            )
            return None
        # the line in force: the API's (it says the value it took), else the developer's; with none
        # set the API serves its balanced setting, and no line is added
        min_chance = b.min_chance if b.min_chance is not None else self._rstt._min_chance
        if chance is None:
            self._wrong_data("guess_no_chance", "the Rabel API sent a guess with no chance")
            return None
        if min_chance is not None and chance < min_chance:
            self._wrong_data(
                "guess_below_min_chance",
                f"the Rabel API sent a guess at chance {chance}, under min_chance {min_chance}",
            )
            return None
        t = self._turn
        if t is None or self._broken or not self._turn_open(t):
            self._count("guess_no_turn")  # the turn it answers is over
            return None
        lim = _host_limits()
        if not lim.early_replies:
            # rule 6: LiveKit starts no early reply in this session (preemptive generation off,
            # turns ended by hand, or a realtime model in place of the LLM). A guess would start
            # nothing; it would only stand in LiveKit's text in place of the caller's words, and a
            # turn ended by hand would commit it as what the caller said.
            self._count("guess_no_early_reply")
            return None
        if lim.agent_speaking:
            # rule 6: the agent's reply is under way and the caller has not interrupted it. LiveKit
            # starts no early reply now, so the guess would start nothing, and its words would count
            # as the caller's in LiveKit's check of how many words interrupt the agent (`min_words`)
            self._count("guess_while_agent_speaks")
            return None
        g = words_of(text)
        h = words_of(heard)
        if not g:
            self._wrong_data("guess_unreadable", "the Rabel API sent a guess with no words")
            return None
        # the guess must still agree with every word heard since it was made
        # (a newer guess replaces a stale one)
        idx = next((i for i in range(len(t.history) - 1, -1, -1) if t.history[i] == h), None)
        if idx is None:
            self._wrong_data(
                "guess_unknown_heard", "the Rabel API sent a guess made from words never sent"
            )
            return None
        late = time.perf_counter() - t.heard_at.get(h, time.perf_counter())
        if not released and late > self._rstt._answer_timeout:
            self._count("guess_slow")
            b.log_once(
                "slow",
                f"Bell: a guess came {late * 1000:.0f} ms after its words, over answer_timeout "
                f"{self._rstt._answer_timeout * 1000:.0f} ms; not used",
            )
            return None
        if t.live is not None:
            if t.held is not None and not released:
                # rule 6a: rule 3 holds an interim against the running guess; the new guess waits
                # for the hold's outcome
                t.waiting_guess = m
                self._count("guess_waits_on_hold")
                return None
            self._count("guess_while_live")  # rule 6a: one guess alive
            return None
        if any(not _agrees(w, g) for w in t.history[idx + 1 :]):
            self._count("guess_stale")
            return None
        # the end bet: Bell's bet that the turn ends on the words it was made from, sent as the
        # early start on them
        end_bet = g == h
        if not end_bet and g == t.words:
            # the speech-to-text has shown the guessed words since the guess was made: Bell's guess
            # is now that the turn ends on these words, and it is sent as the early start on them.
            # Some speech-to-text services show a turn's last words in one burst, faster than the
            # answer time.
            self._count("guess_reached_by_words")
        if t.running is not None and g == t.running:
            self._count("guess_same_as_running")
            return None
        if self._lk_starts + 1 > lim.max_starts - 2:
            # rule 6: after this guess LiveKit must still have two early starts left in its turn:
            # one for the caller's final words, one spare for a turn the caller goes on with
            # after a pause
            self._count("guess_no_spare_start")
            return None
        if t.stt_starts >= 1:
            # rule 6: the speech-to-text has made its own early start in this turn: a guess now
            # would cancel that start, or leave the speech-to-text one start for the rest of
            # the turn
            self._count("guess_after_stt_start")
            return None
        if time.time() - t.speech_start > lim.max_speech_s:
            self._count("guess_past_10s")
            return None
        committed = words_of(" ".join(t.committed))
        if not _is_begin(committed, g):
            self._count("guess_against_final")
            return None
        if self._lk_waiting:
            # rule 8: LiveKit's own early reply made on a final waits for its commit; an early start
            # now would replace it (LiveKit keeps one waiting at a time), and if LiveKit then
            # commits that final alone, its reply starts again
            t.held_guess = m
            self._count("guess_held_while_livekit_waits")
            return None
        # the guess in the final's form: the words so far exactly as the speech-to-text showed them,
        # Bell's added words formed after them (LiveKit's default turn detection keeps an early
        # reply only when the final is the same string)
        shown_words = words_of(t.shown)
        if t.words == committed + shown_words and _is_begin(t.words, g):
            added = g[len(t.words) :]
            if _english(t.last_alt):
                text = final_form(t.shown, added)
            else:
                # another language: no rule of English, the guess is written as this call's
                # finals have been
                text = written(t.shown, added, self._form)
        elif committed:
            text = " ".join(g[len(committed) :])
        if not text.strip():
            # an end bet on words that are all the speech-to-text's finals has no text of its
            # own to start on
            self._count("guess_no_text")
            return None
        alt = t.last_alt
        lang = (
            alt.language
            if alt is not None
            else LanguageCode(self._language if utils.is_given(self._language) else "")
        )
        sd = stt.SpeechData(
            language=lang, text=text, confidence=alt.confidence if alt is not None else 0.0
        )
        t.bell_sent = True
        t.guesses += 1
        t.waiting_guess = None
        t.sent_at = time.perf_counter()
        t.starts += 1
        self._lk_starts += 1
        t.running = t.live = g
        t.held = None
        self._lk_pre, self._lk_pre_stale = text, False
        self._count("guess_sent")
        if end_bet:
            self._count("end_bet_sent")
        logger.debug(
            "Bell: a guess sent as an early start",
            extra={"chance": chance, "words_so_far": len(t.words)},
        )
        return stt.SpeechEvent(type=stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, alternatives=[sd])


def _english(alt: stt.SpeechData | None) -> bool:
    """The speech-to-text says the words are English, or does not say one language (none given,
    "multi"): the guess is then written by the English rules, as it always was."""
    lang = getattr(alt, "language", None)
    base = str(getattr(lang, "language", lang) or "").lower()
    return base in ("", "en", "multi", "und")


def _unit(v: Any) -> float | None:
    """A number from 0 to 1, else None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
        return None
    return float(v)


def _ms(seconds: float) -> int:
    """A head start in whole ms, from 0 to the API's limit."""
    return max(0, min(round(seconds * 1000), _MAX_HEAD_START_MS))
