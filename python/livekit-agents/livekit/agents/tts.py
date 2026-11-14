from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, assert_never

from livekit import rtc

from . import _core, tokenize, utils
from ._generated import facades_tts as _facades, public_tts as _public
from ._generated.public_tts import *  # noqa: F403
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
from .types import (
    NOT_GIVEN,
    USERDATA_TIMED_TRANSCRIPT,
    APIConnectOptions,
    NotGivenOr,
    TranscriptChunk,
)

if TYPE_CHECKING:
    from ._generated.usage import TokenUsage

__all__ = [
    "EndOfText",
    "Keepalive",
    "ProviderRequestId",
    "Ready",
    "WebSocketMultiplexer",
    "WebSocketPacing",
    "WebSocketStream",
    "WebSocketTarget",
    "StreamAdapter",
    "FallbackAdapter",
    "protocol_error",
]
__all__ += _public.__all__


class _EmitterOutput:
    def __init__(self, ch: utils.aio.Chan[SynthesizedAudio]) -> None:
        self._ch = ch

    def send(
        self,
        frame: rtc.AudioFrame,
        request_id: str,
        segment_id: str,
        is_final: bool,
        timed_transcript: list[TranscriptChunk],
        usage: TokenUsage | None,
        characters_count: int | None,
        provider_request_ids: list[str],
    ) -> None:
        if self._ch.closed:
            return
        frame.userdata[USERDATA_TIMED_TRANSCRIPT] = timed_transcript
        self._ch.send_nowait(
            SynthesizedAudio(
                frame=frame,
                request_id=request_id,
                segment_id=segment_id,
                is_final=is_final,
                usage=usage,
                characters_count=characters_count,
                provider_request_ids=provider_request_ids,
            )
        )

    def closed(self) -> bool:
        return self._ch.closed


class FallbackAdapter(_facades.FallbackAdapter):
    """Several TTS behind one: a failed one is skipped until it recovers. Each prepares its text
    by its own options; ``options`` are the serving TTS's."""

    def prewarm(self) -> None:
        self._members[0].prewarm()

    def update_options(
        self,
        *,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Change how every TTS of the chain prepares its text."""
        for member in self._members:
            member.update_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing)

    def _core_options(self) -> _core.TtsOptionsCell:
        return self._active._core_options()


class StreamAdapter(_facades.StreamAdapter):
    """A non-streaming TTS as a streaming one: each chunk of the text, as the wrapped TTS's
    options split it, is synthesized on its own."""

    class Markup(TTS.Markup):
        # a pass-through speaks whatever dialect it wraps
        def _provider_key(self) -> str:
            assert isinstance(self._tts, StreamAdapter)
            return self._tts._tts.markup._provider_key()

    def _core_options(self) -> _core.TtsOptionsCell:
        # the wrapped TTS's: the adapter prepares its text as that TTS would, and
        # update_options changes them
        return self._tts._core_options()

    @property
    def model(self) -> str:
        return self._tts.model

    @property
    def provider(self) -> str:
        return self._tts.provider

    def prewarm(self) -> None:
        self._tts.prewarm()


@dataclass(frozen=True)
class EndOfText:
    """The last input item ``encode`` receives: every segment is done, and the stream ends."""


@dataclass(frozen=True)
class Ready:
    """Reported once the provider is ready for text after the connection's setup (with
    ``awaits_ready``)."""


@dataclass(frozen=True)
class ProviderRequestId:
    """Reported when the provider names its id for the audio that follows: it reaches that audio,
    its metrics and the stream's errors."""

    request_id: str


class WebSocketStream(
    WebSocketProtocol[SegmentInput | EndOfText, InboundT, OutboundT], SynthesizeStream
):
    """A synthesis over the provider's socket, in segments. The framework cuts the input into
    segments at each flush and names them; the plugin writes ``connect`` (or sets a
    ``multiplexer``), ``encode`` for ``SegmentStart``, ``SegmentText``, ``SegmentEnd`` and
    ``EndOfText``, and ``on_message``, which reports each segment's audio, words, usage and done.

    The framework plays the segments in order (a later segment's audio waits for the ones
    before it), completes each once, and times a read out only while a segment whose text ended
    is unanswered: never while the input waits for more text. ``EndOfText`` comes once every
    segment is done, and the stream then ends. Each attempt is one socket session; the stream's
    attempts own the retries and replay the text. The provider's request ids (the connection's,
    and each ``ProviderRequestId`` reported) reach the emitted audio.
    """

    mime_type: str = "audio/pcm"
    """The encoding of the audio reported."""
    connection_per_segment: ClassVar[bool] = False
    """Each segment is its own provider session: the next segment waits until the segment is
    done (its ``SegmentDone``, or the provider's close, which completes it), then goes on a new
    connection."""
    text_idle: float | None = None
    """Seconds without text after which ``on_text_idle`` runs, once per silence."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._segments: _core.TtsSegments | None = None
        self._synthesis = _Synthesis(self)
        self._websocket = WebSocketHost(self, self._synthesis)

    @abstractmethod
    def on_message(self, message: InboundT) -> None:
        """One received message: ``report`` what it carries for its segment, ``Ready`` once the
        provider is ready for text, a ``ProviderRequestId`` naming its id for the audio that
        follows."""

    def on_text_idle(self) -> bytes | OutboundT | None:
        """No text went out for ``text_idle`` seconds: the frame to send; reporting
        ``SegmentDone`` may end a segment whose answers all came."""
        return None

    def report(self, output: SegmentOutput | Ready | ProviderRequestId) -> None:
        """What the provider sent for a segment, or answered. ``encode`` may report the
        ``SegmentDone`` of a segment whose answers all came, for a provider that answers each
        text on its own."""
        match output:
            case Ready():
                self._websocket.signal("ready")
            case ProviderRequestId(request_id=request_id):
                self._websocket.name_request(request_id)
                self._synthesis.opened(request_id)
            case SegmentDone():
                self._synthesis.done(output)
            case SegmentAudio() | SegmentWords() | SegmentUsage():
                if self._segments is not None:
                    self._segments.deliver(output)
            case _:
                assert_never(output)

    async def _run(self, output_emitter: _core.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type=self.mime_type,
            stream=True,
        )
        self._segments = _core.TtsSegments(output_emitter, self._core_attempts, self._tts.label)
        # the stream's attempts own the retries: the socket runs once
        await self._websocket.run(replace(self._conn_options, max_retry=0))


class _Synthesis(WebSocketKind[SegmentInput | EndOfText]):
    """The synthesis's terms for its socket: segments in, a segment in progress never cut by a
    replacement; with ``connection_per_segment``, the next segment waits for the one before to be
    done, which ends its session."""

    def __init__(self, stream: WebSocketStream[Any, Any]) -> None:
        self._stream = stream
        self._ended = False
        # a segment's end went to a provider that takes a session per segment, and whether it is
        # done without the provider saying its session is
        self._ending_session = False
        self._done_unsaid = False
        self._session_over = asyncio.Event()
        self._session_over.set()

    @property
    def label(self) -> str:
        return self._stream._tts.label

    def done(self, output: SegmentDone) -> None:
        if (segments := self._stream._segments) is not None:
            segments.deliver(output)
        if self._ending_session:
            self._ending_session = False
            if self.host.reading:
                # the provider's own message: its session is over
                self.host.signal("session_finished")
            else:
                self._done_unsaid = True
            self._session_over.set()

    async def take(self) -> Batch | None:
        stream = self._stream
        segments = stream._segments
        assert segments is not None
        if self._ended:
            return None
        await self._session_over.wait()
        if self._done_unsaid:
            self._done_unsaid = False
            self.host.end_session()
        while (item := segments.next()) is None:
            try:
                taken = await stream.input.recv()
            except utils.aio.ChanClosed:
                await segments.settled()
                self._ended = True
                return Batch([EndOfText()])
            if isinstance(taken, SpeechChunk):
                segments.push(taken)
            else:
                segments.flush()
        match item:
            case SegmentText():
                stream.mark_started()
            case SegmentEnd() if stream.connection_per_segment:
                self._ending_session = True
                self._session_over.clear()
            case SegmentStart() | SegmentEnd():
                pass
            case _:
                assert_never(item)
        return Batch([item])

    def owes_reply(self) -> bool:
        segments = self._stream._segments
        return segments is not None and bool(segments.open)

    def can_reconnect(self) -> bool:
        return not self.owes_reply()

    def expects_close(self) -> bool:
        # a segment that ends its session, or the end of the text, asked the provider to close
        return self._ending_session or self._ended

    def retrying(self, error: Exception) -> None:
        self._stream._core_retrying(error)

    def opened(self, request_id: str | None) -> None:
        if request_id is not None and (emitter := self._stream._emitter) is not None:
            emitter.provider_request_id(request_id)

    def session_finished(self) -> None:
        if (segments := self._stream._segments) is not None:
            segments.complete_ended()
        self._ending_session = False
        self._session_over.set()

    def contexts(self) -> list[str]:
        # each open segment is a context on a multiplexed provider
        segments = self._stream._segments
        return segments.open if segments is not None else []

    def input_idle_after(self) -> float | None:
        return self._stream.text_idle

    def input_idle(self) -> list[WsFrame]:
        return frames_of(self._stream.on_text_idle())
