from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from livekit import rtc

from . import _core, tokenize, utils
from ._generated import facades_tts as _facades, public_tts as _public
from ._generated.public_tts import *  # noqa: F403
from .http.socket import (
    Done,
    EndOfInput,
    InboundT,
    OutboundT,
    SocketProtocol,
    _single_retry_owner,
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
    "SocketStream",
    "StreamAdapter",
    "FallbackAdapter",
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


class SocketStream(
    SocketProtocol[SpeechChunk | SynthesizeStream.Flush, InboundT, OutboundT], SynthesizeStream
):
    """A synthesis over the provider's socket: the plugin writes ``connect``, ``encode`` (text
    chunks, ``Flush`` and ``EndOfInput``) and ``on_message``, which returns the audio it carries;
    the framework runs the socket. Each attempt is one socket session; the stream's attempts own
    the retries and replay the text."""

    mime_type: str = "audio/pcm"
    """The encoding of the audio ``on_message`` returns."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._socket_init()
        self._socket_emitter: _core.AudioEmitter | None = None
        self._socket_segment = False

    @abstractmethod
    def on_message(self, message: InboundT) -> bytes | Iterable[bytes] | Done | None:
        """One received message: the audio it carries, ``DONE`` once the provider finished."""

    async def _run(self, output_emitter: _core.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type=self.mime_type,
            stream=True,
        )
        self._socket_emitter = output_emitter
        self._socket_segment = False
        await self._socket_run(_single_retry_owner(self._conn_options))

    async def _socket_input(self) -> SpeechChunk | SynthesizeStream.Flush | EndOfInput:
        try:
            item = await self.input.recv()
        except utils.aio.ChanClosed:
            return EndOfInput()
        if isinstance(item, SpeechChunk):
            self.mark_started()
        return item

    def _socket_boundary(self, item: object) -> bool:
        return isinstance(item, SynthesizeStream.Flush)

    def _socket_label(self) -> str:
        return self._tts.label

    def _socket_deliver(self, answer: bytes | Iterable[bytes] | None) -> None:
        emitter = self._socket_emitter
        if answer is None or emitter is None:
            return
        if not self._socket_segment:
            emitter.start_segment(segment_id=utils.shortuuid())
            self._socket_segment = True
        for audio in [answer] if isinstance(answer, bytes) else answer:
            emitter.push(audio)
