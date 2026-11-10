"""Emotion markers in the LLM's text, timed against the audio the TTS makes of it.

``emotion_tts_node`` wraps ``Agent.default.tts_node``. On the way in, it parses the
text for expression markers (``_tags``) and strips the ones the TTS would read aloud.
On the way out, it matches the TTS's word timings against the text to find when each
marker's words are spoken, and attaches the result to the audio frames as
:data:`USERDATA_EMOTION_CUES`, where ``KeyframeAudioOutput`` writes them into the
framed stream.

The planner keeps a cue ahead of its audio: while a marker waits for its word timing,
frames past the last timed word are held back, up to ``max_cue_hold`` seconds. With a
TTS that reports no timings at all, only a marker before a reply's first word can be placed.
"""

from __future__ import annotations

import re
import weakref
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Sequence
from typing import Any

from livekit import rtc
from livekit.agents import utils
from livekit.agents.tts._provider_format import drop_bracket_cues
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT, TimedString
from livekit.agents.voice import Agent, AgentSession, ModelSettings

from ._tags import MAX_TAG_LENGTH, Marker, MarkerStyle, TagParser, normalize
from .log import logger
from .types import EMOTIONS, USERDATA_EMOTION_CUES, Emotion, EmotionCue

# an aligned token this long can match ahead of the coverage position; a shorter one
# only at the start or the very end of the reply
_MIN_SKIP_MATCH_CHARS = 6
_MIN_END_MATCH_CHARS = 2
# an alignment may drop a reply's first letters (ElevenLabs does), so until a token has
# matched, one may match this many characters in, however short
_MAX_DROPPED_CHARS = 8

DEFAULT_MAX_CUE_HOLD = 0.5
"""Seconds of audio held back while a marker waits for its word timing."""

# any XML-shaped tag: expr markers, and the provider-native tags they lower to, which a
# provider's alignment may echo between the words
_XML_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
# the end of a tag whose start an alignment dropped, as ElevenLabs may at the start of a
# reply: "[happy] Oh" aligned as "ppy] Oh"
_TAG_TAIL_RE = re.compile(rf"^\s*[^\s\[\]]{{0,{MAX_TAG_LENGTH}}}\]")

# sessions already warned that their transcripts show [tag] markers
_warned_sessions: weakref.WeakSet[AgentSession[Any]] = weakref.WeakSet()


def _clean_aligned(text: str) -> str:
    """Normalized spoken characters of a TTS-aligned token, markup removed."""
    return normalize(_XML_TAG_RE.sub("", text))


def _drop_tag_tail(tokens: list[TimedString]) -> list[TimedString]:
    """``tokens`` without a tag tail opening their first word (``_TAG_TAIL_RE``).

    Aligned tokens are whole words, so the tail of a one-word tag is one token, or the
    front of one ("ppy]Oh").
    """
    for i, token in enumerate(tokens):
        if not token.strip():
            continue
        if (match := _TAG_TAIL_RE.match(token)) is None:
            return tokens
        rest = token[match.end() :].removeprefix(" ")
        if not rest:
            return [*tokens[:i], *tokens[i + 1 :]]
        cut = TimedString(
            rest,
            start_time=token.start_time,
            end_time=token.end_time,
            confidence=token.confidence,
            start_time_offset=token.start_time_offset,
            speaker_id=token.speaker_id,
        )
        return [*tokens[:i], cut, *tokens[i + 1 :]]
    return tokens


class _BracketFilter:
    """Removes ``[tag]`` markers from a reply's TTS alignment.

    The TTS aligns each tag as words of its own, split at its spaces. The filtered tokens
    are the transcript, and what the cue planner matches against the text.

    ``drop_bracket_cues`` removes whole tags, across tokens. An alignment that drops the
    reply's first characters can also start inside its opening tag, so the first words
    lose the tail of one (``_drop_tag_tail``).
    """

    def __init__(self) -> None:
        self._held: list[TimedString] = []
        self._started = False

    def push(self, tokens: list[TimedString]) -> list[TimedString]:
        if not self._started and any(t.strip() for t in tokens):
            self._started = True
            tokens = _drop_tag_tail(tokens)
        return drop_bracket_cues(tokens, self._held)

    @property
    def holding(self) -> bool:
        """Whether tokens are held back in case they open a tag."""
        return bool(self._held)

    def flush(self) -> list[TimedString]:
        """The tokens held back in case they opened a tag, at the end of the reply."""
        return drop_bracket_cues([], self._held, final=True)


def _warn_brackets_in_transcript(session: AgentSession[Any], *, aligned: bool) -> None:
    if session in _warned_sessions:
        return
    _warned_sessions.add(session)
    if aligned:
        logger.warning(
            "[tag] markers show in the agent transcript: set use_tts_aligned_transcript=True "
            "on the AgentSession or Agent to keep them out"
        )
    else:
        logger.warning(
            "[tag] markers show in the agent transcript: keeping them out needs word "
            "timings, which this TTS does not report (the ElevenLabs plugin reports them; "
            "ElevenLabs through LiveKit Inference does not)"
        )


class _CuePlanner:
    """Turns markers (text positions) into cues (audio times) and attaches them to frames.

    Positions are normalized characters of the spoken text (``_tags.normalize``); times
    are seconds from the start of the reply's audio.
    """

    def __init__(self, *, aligned: bool, max_cue_hold: float) -> None:
        self._aligned = aligned
        self._max_cue_hold = max_cue_hold

        self.spoken = ""
        """Normalized spoken text sent to the TTS so far (``TagParser.spoken``)."""
        self.keep_last = False
        """Hold the newest frame back: the caller may still attach data to the last one."""
        self._text_done = False
        self._pending: deque[Marker] = deque()
        self._resolved: list[tuple[float, Emotion]] = []

        # how far the alignment has walked through the spoken text, and when that was
        self._cover_pos = 0
        self._cover_time = 0.0

        # frames: received from the TTS, released to the caller, and held in between
        self._received_end = 0.0
        self._held: deque[tuple[float, rtc.AudioFrame]] = deque()
        self._held_duration = 0.0
        self._warned_unplaceable = False

    @property
    def received_duration(self) -> float:
        return self._received_end

    # -- text side -------------------------------------------------------------

    def add_marker(self, marker: Marker) -> None:
        if marker.anchor == 0:
            # an opening marker is the reply's expression from its first sample
            self._resolved.append((0.0, marker.emotion))
        elif not self._aligned:
            # nothing to place it against; once per reply, so the log stays readable
            if not self._warned_unplaceable:
                self._warned_unplaceable = True
                logger.warning(
                    "ignoring an emotion marker inside a reply: the TTS reports no word "
                    "timings, so only a marker before a reply's first word moves the face"
                )
        elif marker.anchor < self._cover_pos:
            # the alignment already walked past it (a word matched ahead)
            self._resolved.append((self._cover_time, marker.emotion))
        else:
            self._pending.append(marker)

    def text_done(self) -> None:
        self._text_done = True

    # -- alignment side --------------------------------------------------------

    def add_timings(self, tokens: Sequence[TimedString]) -> None:
        if not self._aligned:
            return
        for token in tokens:
            word = _clean_aligned(token)
            if not word:
                continue
            at = self.spoken.find(word, self._cover_pos)
            if at == -1:
                continue
            end = at + len(word)
            ends_text = self._text_done and end == len(self.spoken)
            if not (
                at == self._cover_pos
                or len(word) >= _MIN_SKIP_MATCH_CHARS
                or (self._cover_pos == 0 and at <= _MAX_DROPPED_CHARS)
                or (ends_text and len(word) >= _MIN_END_MATCH_CHARS)
            ):
                continue

            start = token.start_time if utils.is_given(token.start_time) else None
            self._cover_pos = end
            if start is not None:
                self._cover_time = token.end_time if utils.is_given(token.end_time) else start
            else:
                # a token without a time: the frames so far are the best we know
                start = self._received_end
                self._cover_time = start

            # a marker before or inside this word changes the face as the word starts
            while self._pending and self._pending[0].anchor < end:
                marker = self._pending.popleft()
                self._resolved.append((start, marker.emotion))

    # -- frame side ------------------------------------------------------------

    def push_frame(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        start = self._received_end
        self._received_end += frame.duration
        self._held.append((start, frame))
        self._held_duration += frame.duration
        return self._release()

    def finish(self) -> list[rtc.AudioFrame]:
        """End of the reply: place what is still pending, then release everything."""
        total = len(self.spoken)
        while self._pending:
            marker = self._pending.popleft()
            fraction = marker.anchor / total if total else 1.0
            self._resolved.append((fraction * self._received_end, marker.emotion))

        out = self._release(all_frames=True)
        if self._resolved:
            # cues past the last frame ride on it. _may_release kept that frame back while
            # a cue could still land past it, so it is in `out`, not yet yielded.
            if out:
                self._attach(out[-1], self._received_end - out[-1].duration, include_all=True)
            else:
                logger.debug(
                    "dropping %d emotion cues: no audio left to carry them", len(self._resolved)
                )
            self._resolved.clear()

        return out

    def _release(self, *, all_frames: bool = False) -> list[rtc.AudioFrame]:
        out: list[rtc.AudioFrame] = []
        while self._held:
            start, frame = self._held[0]
            end = start + frame.duration
            if not all_frames and not self._may_release(end):
                break
            self._held.popleft()
            self._held_duration -= frame.duration
            self._attach(frame, start)
            out.append(frame)
        return out

    def _may_release(self, frame_end: float) -> bool:
        if frame_end >= self._received_end - 1e-6 and (
            self.keep_last
            or self._pending
            or any(at >= frame_end - 1e-6 for at, _ in self._resolved)
        ):
            # the newest frame waits while a cue may still land on or past it, or the caller
            # asks: finish() puts whatever is left on the last frame, which must not have
            # been yielded
            return False
        if not self._pending:
            return True
        # a pending marker resolves no earlier than the alignment has reached, so audio
        # before that cannot be the cue's; audio past it waits, up to the cap
        return (
            frame_end <= self._cover_time + 1e-6 or self._held_duration > self._max_cue_hold + 1e-6
        )

    def _attach(self, frame: rtc.AudioFrame, start: float, *, include_all: bool = False) -> None:
        # a cue on the boundary belongs to the frame that starts there
        end = start + frame.duration - 1e-6
        due = [c for c in self._resolved if include_all or c[0] < end]
        if not due:
            return
        self._resolved = [c for c in self._resolved if c not in due]
        due.sort(key=lambda c: c[0])
        cues: list[EmotionCue] = list(frame.userdata.get(USERDATA_EMOTION_CUES) or ())
        cues.extend(EmotionCue(emotion, at - start) for at, emotion in due)
        frame.userdata[USERDATA_EMOTION_CUES] = cues


InnerNode = Callable[[AsyncIterable[str]], AsyncIterable[rtc.AudioFrame]]


async def _emotion_frames(
    text: AsyncIterable[str],
    inner_node: InnerNode,
    *,
    style: MarkerStyle = "expr",
    forward_expr: bool,
    aligned: bool,
    emotions: Sequence[Emotion] = EMOTIONS,
    max_cue_hold: float = DEFAULT_MAX_CUE_HOLD,
) -> AsyncGenerator[rtc.AudioFrame, None]:
    """Run ``inner_node`` on the marker-parsed text; yield its frames with emotion cues.

    Args:
        text: The LLM text, markers included.
        inner_node: Synthesizes a text stream into audio frames (``Agent.default.tts_node``
            with the agent and settings bound).
        style: The markers the text carries (``_tags.MarkerStyle``).
        forward_expr: Pass ``<expr>`` markup through to the TTS (it lowers the markup
            itself); otherwise strip it so the TTS does not read it.
        aligned: The TTS reports word (or sentence) timings on its frames.
        emotions: Emotions a marker may resolve to.
        max_cue_hold: Seconds of audio held back while a marker waits for its timing.
    """
    parser = TagParser(style=style, emotions=emotions, forward_expr=forward_expr)
    planner = _CuePlanner(aligned=aligned, max_cue_hold=max_cue_hold)
    brackets = _BracketFilter() if style == "brackets" else None

    def _take(out: str, markers: list[Marker]) -> str:
        # the markers come first: each is anchored before the text forwarded with it
        planner.spoken = parser.spoken
        for marker in markers:
            planner.add_marker(marker)
        return out

    async def _parsed_text() -> AsyncGenerator[str, None]:
        try:
            async for chunk in text:
                if out := _take(*parser.push(chunk)):
                    yield out
            if out := _take(*parser.flush()):
                yield out
        finally:
            planner.spoken = parser.spoken
            planner.text_done()

    inner = inner_node(_parsed_text())
    try:
        async for frame in inner:
            timed = [
                t if isinstance(t, TimedString) else TimedString(t)
                for t in frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])
                if isinstance(t, str)
            ]
            if brackets is not None and timed:
                timed = brackets.push(timed)
                frame.userdata[USERDATA_TIMED_TRANSCRIPT] = timed
                # tokens held back in case they open a tag ride on the last frame if the
                # reply ends first
                planner.keep_last = brackets.holding
            planner.add_timings(timed)
            for released in planner.push_frame(frame):
                yield released
        tail = brackets.flush() if brackets is not None else []
        planner.add_timings(tail)
        last = planner.finish()
        if tail and last:
            last[-1].userdata[USERDATA_TIMED_TRANSCRIPT] = [
                *last[-1].userdata.get(USERDATA_TIMED_TRANSCRIPT, []),
                *tail,
            ]
        for released in last:
            yield released
    finally:
        aclose = getattr(inner, "aclose", None)
        if aclose is not None:
            await aclose()


def emotion_tts_node(
    agent: Agent,
    text: AsyncIterable[str],
    model_settings: ModelSettings,
    *,
    emotions: Sequence[Emotion] = EMOTIONS,
    style: MarkerStyle = "expr",
    max_cue_hold: float = DEFAULT_MAX_CUE_HOLD,
) -> AsyncGenerator[rtc.AudioFrame, None]:
    """``Agent.default.tts_node`` with the LLM's expression markers turned into timed cues.

    Use it as the agent's ``tts_node``::

        class MyAgent(Agent):
            def tts_node(self, text, model_settings):
                return keyframe.emotion_tts_node(self, text, model_settings)

    It is a drop-in for ``Agent.default.tts_node``, so a ``tts_node`` that processes the
    text before synthesis, or the frames after, can call it in the same place.

    Markers are ``<expr type="expression" label="..."/>`` (LiveKit's expressive markup).
    With expressive mode active the markup is passed through, so one marker steers both
    the voice and the face; otherwise it is stripped before the TTS. ``style="brackets"``
    also reads ``[tag]`` audio tags, which always go on to the TTS. They stay out of the
    transcript only with ``use_tts_aligned_transcript=True`` and a TTS that reports word
    timings; otherwise a warning is logged once per session.

    With a TTS that reports word timings, cues land on the words themselves; a
    non-streaming TTS gives sentence starts. A streaming TTS without timings moves the
    face only on a marker before a reply's first word; later markers are ignored, with a
    warning once per reply.

    Args:
        emotions: Emotions a marker may resolve to; a label naming none sends no cue.
        style: The markers the LLM writes: pass the ``style`` given to
            :func:`emotion_instructions`. In the default ``"expr"`` style, square
            brackets are text.
        max_cue_hold: Seconds of audio held back while a marker waits for its word
            timing, so the cue reaches the avatar before its audio.
    """

    activity = agent._get_activity_or_raise()
    session_tts = activity.tts
    expressive = activity._resolve_expressive_options() is not None
    aligned = session_tts is not None and bool(
        session_tts.capabilities.aligned_transcript or not session_tts.capabilities.streaming
    )
    # the transcript leaves [tag] markers out only when it is built from word timings
    if style == "brackets" and not (aligned and activity.use_tts_aligned_transcript):
        _warn_brackets_in_transcript(activity.session, aligned=aligned)

    def _inner(parsed: AsyncIterable[str]) -> AsyncIterable[rtc.AudioFrame]:
        return Agent.default.tts_node(agent, parsed, model_settings)

    return _emotion_frames(
        text,
        _inner,
        style=style,
        forward_expr=expressive,
        aligned=aligned,
        emotions=emotions,
        max_cue_hold=max_cue_hold,
    )


def emotion_instructions(
    *, emotions: Sequence[Emotion] = EMOTIONS, style: MarkerStyle = "expr"
) -> str:
    """Prompt text that teaches the LLM the markers :func:`emotion_tts_node` reads.

    Args:
        emotions: The faces to offer.
        style: ``"expr"`` teaches ``<expr type="expression" label="..."/>``. With
            expressive mode (``AgentSession(expressive=...)``) pass it as
            ``tts_instructions_append`` so one marker steers voice and face; with any
            other TTS add it to the agent's instructions and it drives the face only.
            ``"brackets"`` teaches ``[label]`` tags, which ElevenLabs v3/v4 voices read
            natively; add it to the agent's instructions. Pass the same ``style`` to
            :func:`emotion_tts_node`.

    The text has no braces, so it is safe inside the expressive template.
    """
    names = ", ".join(emotions)
    if style == "expr":
        return (
            "Your face is animated from your expression markers: an "
            '<expr type="expression" label="..."/> marker also sets the expression you '
            "hold until the next one. Start every reply with one, and add one wherever "
            "the mood changes, right before the words it applies to. The face knows only "
            f"these expressions: {names}. Use only those names as expression labels; a "
            "marker with any other label leaves the face as it was."
        )
    if style == "brackets":
        return (
            "Your face is animated from emotion tags in your reply: a tag in square "
            "brackets sets the expression you hold until the next one. Start every reply "
            "with one, and add one wherever the mood changes, right before the words it "
            f"applies to. Available tags: {' '.join(f'[{e}]' for e in emotions)}. Use "
            "only these names in emotion tags. Other audio tags are fine alongside them, "
            "but they don't change your face."
        )
    raise ValueError(f"unknown marker style {style!r}")
