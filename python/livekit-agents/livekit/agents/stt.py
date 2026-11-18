from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, ClassVar, TypeAlias, assert_never

from livekit import rtc

from . import utils
from ._generated import facades_stt as _facades, public_stt as _public
from ._generated.public_stt import *  # noqa: F403
from ._generated.public_stt import SpeechEnded, SpeechStarted, Transcript
from ._generated.stt import TEvent
from ._generated.vad import VAD, VADEventType, VADStream
from ._generated.websocket import WsFrame
from ._websocket import (
    Batch,
    Completion,
    InboundT,
    Keepalive,
    OutboundT,
    WebSocketHost,
    WebSocketKind,
    WebSocketMultiplexer,
    WebSocketPacing,
    WebSocketProtocol,
    WebSocketTarget,
    frames_of,
    log_skipped,
    protocol_error,
)
from .types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from .utils import aio

__all__ = [
    "Completion",
    "EndOfAudio",
    "Flush",
    "Keepalive",
    "PRE_ROLL",
    "Ready",
    "Report",
    "SessionFinished",
    "WebSocketMultiplexer",
    "WebSocketPacing",
    "WebSocketStream",
    "WebSocketTarget",
    "StreamAdapter",
    "FallbackAdapter",
    "log_skipped",
    "protocol_error",
]
__all__ += _public.__all__


class FallbackAdapter(_facades.FallbackAdapter[TEvent]):
    """Several STT behind one: a failed one is skipped until it recovers. A subclass that relays
    its members' own events names them: ``FallbackAdapter[Literal["my_event"]]``."""

    def _core_retries_itself(self) -> bool:
        return True

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        for stt in self._members:
            stt._update_session_keyterms(keyterms)

    def _push_conversation_item(self, role: str, text: str) -> None:
        for stt in self._members:
            stt._push_conversation_item(role, text)

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        frames = buffer if isinstance(buffer, list) else [buffer]
        return await self._core_native_recognize(
            frames, language if language else None, conn_options=conn_options
        )

    def prewarm(self) -> None:
        self._members[0].prewarm()


class StreamAdapter(_facades.StreamAdapter):
    """A recognize-only STT as a streaming one: the VAD cuts the audio into utterances, each
    recognized on its own."""

    def _core_retries_itself(self) -> bool:
        return True

    @property
    def model(self) -> str:
        return self._stt.model

    @property
    def provider(self) -> str:
        return self._stt.provider

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._stt._update_session_keyterms(keyterms)

    def _push_conversation_item(self, role: str, text: str) -> None:
        self._stt._push_conversation_item(role, text)

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        return await self._stt.recognize(
            buffer=buffer, language=language, conn_options=conn_options
        )

    def prewarm(self) -> None:
        self._stt.prewarm()


Report: TypeAlias = Transcript | SpeechStarted | SpeechEnded
"""What a stream reports of what its provider heard."""


Flush = RecognizeStream.Flush
"""The end of a segment of the audio, an input item ``encode`` receives."""


@dataclass(frozen=True)
class EndOfAudio:
    """The last input item ``encode`` receives: the audio ended (the provider's goodbye)."""


@dataclass(frozen=True)
class Ready:
    """Reported once the provider is ready for audio after the connection's setup (with
    ``awaits_ready``)."""


@dataclass(frozen=True)
class SessionFinished:
    """Reported once the provider finished its session: after the end of the audio the stream
    ends; with ``new_session_per_flush``, the audio after the flush goes on a new connection."""


class WebSocketStream(
    WebSocketProtocol[rtc.AudioFrame | Flush | EndOfAudio, InboundT, OutboundT], RecognizeStream
):
    """A recognition over the provider's socket: the plugin writes ``connect``, ``encode`` (audio
    frames, ``Flush`` and ``EndOfAudio``) and ``on_message``, which reports what the provider
    heard; the framework runs the socket. Each attempt is one socket session; the stream's
    attempts own the retries, and each connection's end closes the speech it left open. A
    transcript naming no request id carries the provider's (see ``request_id``)."""

    frame_ms: int | None = None
    """Audio is re-chunked to frames of this many milliseconds before ``encode``."""
    new_session_per_flush: ClassVar[bool] = False
    """The provider finishes its session at each flush (its end frame comes from ``encode``): the
    audio after the flush waits until it reports ``SessionFinished`` or closes, then goes on a new
    connection."""
    audio_idle: float | None = None
    """Seconds without audio after which ``on_audio_idle`` runs, once per silence."""
    truncate_silence: ClassVar[bool] = False
    """Audio goes out only while the stream's ``vad`` hears speech, led by the ``PRE_ROLL``
    before it so a turn's start is not clipped, and each speech's end is a ``Flush``. For a
    provider that bills the audio sent and detects no speech itself; without a ``vad`` every
    frame goes."""

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        vad: VAD | None = None,
    ) -> None:
        """
        Args:
            vad: What hears the speech ``truncate_silence`` lets through.
        """
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self._completion_received = False
        self._gate = _SilenceGate(vad) if self.truncate_silence and vad is not None else None
        # the audio the gate lets through is the usage, not the audio pushed
        self._measures_sent = self._core_measured and self._gate is not None
        if self._gate is not None:
            self._core_measured = False
        self._websocket = WebSocketHost(self, _Recognition(self, self._gate))

    @abstractmethod
    def on_message(self, message: InboundT) -> None:
        """One received message: ``report`` what the provider heard, ``Ready`` once it is ready
        for audio, ``SessionFinished`` once it finished its session."""

    def on_audio_idle(self) -> bytes | OutboundT | None:
        """No audio went out for ``audio_idle`` seconds: the frame to send (a finalize)."""
        return None

    def report(self, report: Report | Ready | SessionFinished) -> None:
        """What the provider heard or answered."""
        match report:
            case Ready():
                self._websocket.signal("ready")
            case SessionFinished():
                self._completion_received = True
                self._websocket.signal("session_finished")
            case Transcript():
                if not report.request_id and self.request_id:
                    report.request_id = self.request_id
                super().report(report)
            case SpeechStarted() | SpeechEnded():
                super().report(report)
            case _:
                assert_never(report)

    def nothing_pending(self) -> bool:
        """Whether all audio sent has been answered. Override for protocols that track
        outstanding items independently of their speech and transcript events."""
        return super()._core_settled()

    def _core_settled(self) -> bool:
        # The attempt's response deadline must use the protocol's pending items too.
        if self.completion == "peer_closed" or (
            self.completion == "session_finished" and not self._completion_received
        ):
            return False
        return self.nothing_pending()

    def _websocket_count_sent_audio(self, seconds: float) -> None:
        if self._measures_sent and seconds > 0:
            if (usage := self._core_attempts.audio_taken(seconds)) is not None:
                self.send(usage)

    async def _run(self) -> None:
        # the stream's attempts own the retries: the socket runs once
        socket = self._websocket.run(replace(self._conn_options, max_retry=0))
        if self._gate is None:
            await socket
            return
        vad_stream = self._gate.open()
        try:
            async with aio.TaskSet() as tasks:
                await tasks.first_completed(socket, self._gate.listen(vad_stream))
        finally:
            await vad_stream.aclose()


PRE_ROLL = 0.3
"""Seconds of audio before a speech that ``truncate_silence`` sends with it."""
# how long the VAD may take to report a speech start after the speech began
_START_LAG = 0.5
# float slack when comparing audio times summed in different orders
_EPSILON = 1e-6


@dataclass
class _Speech:
    """One speech the VAD heard, in seconds of the audio its stream was given."""

    start: float
    end: float | None = None

    def covers(self, start: float, end: float) -> bool:
        """Whether audio from ``start`` to ``end`` belongs to it or to its pre-roll."""
        return self.start - PRE_ROLL < end - _EPSILON and (
            self.end is None or start < self.end - _EPSILON
        )


class _SilenceGate:
    """The audio of a ``truncate_silence`` stream: each frame held until its VAD classified it,
    then sent if it belongs to a speech (or its pre-roll), else dropped once no late speech start
    can claim it. A speech still open lets its frames through at once."""

    def __init__(self, vad: VAD) -> None:
        self._vad = vad
        self._vad_stream: VADStream | None = None
        # the frames not yet decided, each with where it starts in the VAD stream's audio
        self._held: deque[tuple[float, rtc.AudioFrame]] = deque()
        self._given = 0.0
        self._classified = 0.0
        self._speeches: deque[_Speech] = deque()
        # the speech the last frame sent belongs to: its end flushes the segment
        self._sending: _Speech | None = None

    def open(self) -> VADStream:
        """The VAD stream of an attempt: the frames still held are classified again on it."""
        self._vad_stream = self._vad.stream()
        self._given = self._classified = 0.0
        self._speeches.clear()
        self._sending = None
        held = [frame for _, frame in self._held]
        self._held.clear()
        self._hold(held)
        return self._vad_stream

    async def listen(self, vad_stream: VADStream) -> None:
        """Reads the VAD's verdicts until its stream ends."""
        async for event in vad_stream:
            self._classified = max(self._classified, event.timestamp)
            match event.type:
                case VADEventType.START_OF_SPEECH:
                    self._speeches.append(_Speech(event.timestamp - event.speech_duration))
                case VADEventType.END_OF_SPEECH:
                    if self._speeches and self._speeches[-1].end is None:
                        self._speeches[-1].end = event.timestamp
                case VADEventType.INFERENCE_DONE:
                    pass
                case _:
                    assert_never(event.type)

    def _hold(self, audio: list[rtc.AudioFrame]) -> None:
        assert self._vad_stream is not None, "a gate holds audio only during an attempt"
        for frame in audio:
            self._held.append((self._given, frame))
            self._given += frame.duration
            self._vad_stream.push_frame(frame)

    def take(self, audio: list[rtc.AudioFrame]) -> list[rtc.AudioFrame | Flush]:
        """``audio`` held, then the frames decided to go, in order, up to the first speech's
        end (a ``Flush``): what follows it goes on the next call."""
        self._hold(audio)
        out: list[rtc.AudioFrame | Flush] = []
        while True:
            sending = self._sending
            if (
                sending is not None
                and sending.end is not None
                and (not self._held or self._held[0][0] >= sending.end - _EPSILON)
            ):
                self._sending = None
                out.append(Flush())
                break
            if not self._held:
                break
            start, frame = self._held[0]
            end = start + frame.duration
            speech = next((s for s in self._speeches if s.covers(start, end)), None)
            if speech is not None:
                self._sending = speech
                out.append(frame)
            elif end > self._classified - PRE_ROLL - _START_LAG + _EPSILON:
                # not classified yet, or a pre-roll a late start may still claim
                break
            self._held.popleft()
        while self._speeches and self._speeches[0] is not self._sending:
            first = self._speeches[0]
            if first.end is None or (self._held and first.end > self._held[0][0]):
                break
            self._speeches.popleft()
        return out


class _Recognition(WebSocketKind[rtc.AudioFrame | Flush | EndOfAudio]):
    """The recognition's terms for its socket: audio in, re-chunked (and gated to speech with
    ``truncate_silence``); a speech in progress is never cut by a replacement; a provider that
    ends its session at a flush holds the audio after it until it did."""

    def __init__(self, stream: WebSocketStream[Any, Any], gate: _SilenceGate | None) -> None:
        self._stream = stream
        self._gate = gate
        self._chunks: utils.media.AudioChunker | None = None
        self._ended = False
        # audio went out since the provider last finished a session, and whether a flush asked it
        # to finish the one in progress
        self._audio_in_session = False
        self._ending_session = False
        self._session_over = asyncio.Event()
        self._session_over.set()

    @property
    def label(self) -> str:
        return self._stream._stt.label

    def _held(self) -> list[rtc.AudioFrame]:
        return self._chunks.flush() if self._chunks is not None else []

    def _admit(self, audio: list[rtc.AudioFrame]) -> list[rtc.AudioFrame | Flush]:
        """What of ``audio`` goes now: all of it, or what the silence gate let through."""
        if self._gate is None:
            return list(audio)
        admitted = self._gate.take(audio)
        self._stream._websocket_count_sent_audio(
            sum(item.duration for item in admitted if isinstance(item, rtc.AudioFrame))
        )
        return admitted

    def _batch(self, items: list[rtc.AudioFrame | Flush], fresh: float) -> Batch:
        """``items`` to send, paced by no more than the ``fresh`` seconds of input they came
        with."""
        sent = sum(item.duration for item in items if isinstance(item, rtc.AudioFrame))
        self._audio_in_session |= sent > 0
        if any(isinstance(item, Flush) for item in items):
            self._ending_flush()
        return Batch(list(items), min(sent, fresh))

    def _ending_flush(self) -> None:
        if self._stream.new_session_per_flush:
            self._ending_session = True
            self._session_over.clear()

    async def take(self) -> Batch | None:
        stream = self._stream
        if self._ended:
            return None
        await self._session_over.wait()
        if self._gate is not None and (decided := self._admit([])):
            return self._batch(decided, 0.0)
        try:
            item = await stream.input.recv()
        except aio.ChanClosed:
            self._ended = True
            if not self._audio_in_session and self._gate is None:
                # the provider holds no session to say goodbye to
                return None
            held = self._admit(self._held())
            self._audio_in_session |= any(isinstance(i, rtc.AudioFrame) for i in held)
            if not self._audio_in_session:
                return None
            pace = sum(i.duration for i in held if isinstance(i, rtc.AudioFrame))
            return Batch([*held, EndOfAudio()], pace)
        match item:
            case rtc.AudioFrame():
                if self._chunks is None:
                    self._chunks = utils.media.AudioChunker(stream.frame_ms)
                audio = self._chunks.push(item)
                return self._batch(self._admit(audio), sum(frame.duration for frame in audio))
            case Flush():
                raw = self._held()
                held = self._admit(raw)
                if not (self._audio_in_session or held):
                    return Batch([])
                if held and isinstance(held[-1], Flush):
                    # the speech's end already ended the segment
                    return self._batch(held, sum(frame.duration for frame in raw))
                return self._batch([*held, item], sum(frame.duration for frame in raw))
            case _:
                assert_never(item)

    def owes_reply(self) -> bool:
        # the audio sent owes its transcripts
        return True

    def nothing_pending(self) -> bool:
        return self._stream.nothing_pending()

    def expects_close(self) -> bool:
        # a flush that ends the session, or the end of the audio, asked the provider to close
        return self._ending_session or self._ended

    def can_reconnect(self) -> bool:
        return self._stream.nothing_pending()

    def retrying(self, error: Exception) -> None:
        self._stream._core_retrying(error)

    def connecting(self, first: bool) -> None:
        self._stream._completion_received = False
        # the attempt's start already closed the speech the one before left open
        if not first:
            self._stream._end_speech()

    def session_finished(self) -> None:
        self._audio_in_session = self._ending_session = False
        self._session_over.set()

    def input_idle_after(self) -> float | None:
        return self._stream.audio_idle

    def input_idle(self) -> list[WsFrame]:
        return frames_of(self._stream.on_audio_idle())
