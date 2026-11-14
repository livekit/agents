from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Any, ClassVar, TypeAlias, assert_never

from livekit import rtc

from . import utils
from ._generated import facades_stt as _facades, public_stt as _public
from ._generated.public_stt import *  # noqa: F403
from ._generated.public_stt import SpeechEnded, SpeechStarted, Transcript
from ._generated.websocket import WsFrame
from ._websocket import (
    Batch,
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
    protocol_error,
)
from .types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from .utils import aio

__all__ = [
    "EndOfAudio",
    "Flush",
    "Keepalive",
    "Ready",
    "Report",
    "SessionFinished",
    "WebSocketMultiplexer",
    "WebSocketPacing",
    "WebSocketStream",
    "WebSocketTarget",
    "StreamAdapter",
    "FallbackAdapter",
    "protocol_error",
]
__all__ += _public.__all__


class FallbackAdapter(_facades.FallbackAdapter):
    """Several STT behind one: a failed one is skipped until it recovers."""

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

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self._websocket = WebSocketHost(self, _Recognition(self))

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
                self._websocket.signal("session_finished")
            case Transcript():
                if not report.request_id and self.request_id:
                    report.request_id = self.request_id
                super().report(report)
            case SpeechStarted() | SpeechEnded():
                super().report(report)
            case _:
                assert_never(report)

    async def _run(self) -> None:
        # the stream's attempts own the retries: the socket runs once
        await self._websocket.run(replace(self._conn_options, max_retry=0))


class _Recognition(WebSocketKind[rtc.AudioFrame | Flush | EndOfAudio]):
    """The recognition's terms for its socket: audio in, re-chunked; a speech in progress is
    never cut by a replacement; a provider that ends its session at a flush holds the audio after
    it until it did."""

    def __init__(self, stream: WebSocketStream[Any, Any]) -> None:
        self._stream = stream
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

    async def take(self) -> Batch | None:
        stream = self._stream
        if self._ended:
            return None
        await self._session_over.wait()
        try:
            item = await stream.input.recv()
        except aio.ChanClosed:
            self._ended = True
            if not self._audio_in_session:
                # the provider holds no session to say goodbye to
                return None
            held = self._held()
            return Batch([*held, EndOfAudio()], sum(audio.duration for audio in held))
        match item:
            case rtc.AudioFrame():
                if self._chunks is None:
                    self._chunks = utils.media.AudioChunker(stream.frame_ms)
                audio = self._chunks.push(item)
                self._audio_in_session |= bool(audio)
                return Batch(list(audio), sum(frame.duration for frame in audio))
            case Flush():
                held = self._held()
                if not (self._audio_in_session or held):
                    return Batch([])
                if stream.new_session_per_flush:
                    self._ending_session = True
                    self._session_over.clear()
                return Batch([*held, item], sum(audio.duration for audio in held))
            case _:
                assert_never(item)

    def owes_reply(self) -> bool:
        # the audio sent owes its transcripts
        return True

    def nothing_pending(self) -> bool:
        return self._stream._core_settled()

    def expects_close(self) -> bool:
        # a flush that ends the session, or the end of the audio, asked the provider to close
        return self._ending_session or self._ended

    def can_reconnect(self) -> bool:
        return not self._stream.speaking

    def retrying(self, error: Exception) -> None:
        self._stream._core_retrying(error)

    def connecting(self, first: bool) -> None:
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
