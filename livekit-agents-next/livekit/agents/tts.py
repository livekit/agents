"""Text to speech: the base classes a provider subclasses (generated from the core's
declarations), the expressive markup helpers over the core's per-provider tables
(``_core.Markup``), and the adapters, whose engines are the core's (``_core.FallbackTts``,
``_core.tts_adapter_stream``, ``_core._core_PacedSentences``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from livekit import rtc

from . import _core, tokenize, utils
from ._generated import facades_tts as _facades, public_tts as _public
from ._generated.public_tts import *  # noqa: F403
from .tokenize import SentenceStream
from .types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    USERDATA_TIMED_TRANSCRIPT,
    APIConnectOptions,
    NotGivenOr,
    TranscriptChunk,
)

if TYPE_CHECKING:
    from ._generated.usage import TokenUsage

__all__ = [
    "StreamAdapter",
    "FallbackAdapter",
    "SentenceStreamPacer",
]
__all__ += _public.__all__


class _EmitterOutput:
    """Where the native emitter's output goes: ``SynthesizedAudio`` into the synthesis channel,
    and the provider's request ids onto its current span."""

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
    """Several TTS behind one: a failed one is skipped until it recovers."""

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FallbackChunkedStream:
        return FallbackChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FallbackSynthesizeStream:
        return FallbackSynthesizeStream(tts=self, conn_options=conn_options)

    def prewarm(self) -> None:
        self._members[0].prewarm()


class StreamAdapter(TTS):
    """A non-streaming TTS as a streaming one: the text is split into sentences, each
    synthesized on its own."""

    def __init__(
        self,
        *,
        tts: TTS,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        text_pacing: SentenceStreamPacer | bool = False,
    ) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=True, aligned_transcript=True),
            sample_rate=tts.sample_rate,
            num_channels=tts.num_channels,
        )
        self._wrapped_tts = tts
        self._sentence_tokenizer = sentence_tokenizer or None
        self._stream_pacer: SentenceStreamPacer | None = (
            SentenceStreamPacer() if text_pacing is True else text_pacing or None
        )

    class Markup(TTS.Markup):
        # a pass-through speaks whatever dialect it wraps
        def _provider_key(self) -> str:
            assert isinstance(self._tts, StreamAdapter)
            return self._tts._wrapped_tts.markup._provider_key()

    def _core_lowers_markup(self) -> bool:
        # its sentences are lowered here, unlike a natively streaming plugin's input
        return True

    @property
    def model(self) -> str:
        return self._wrapped_tts.model

    @property
    def provider(self) -> str:
        return self._wrapped_tts.provider

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return self._wrapped_tts.synthesize(text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> StreamAdapterWrapper:
        return StreamAdapterWrapper(tts=self, conn_options=conn_options)

    def prewarm(self) -> None:
        self._wrapped_tts.prewarm()


class StreamAdapterWrapper(SynthesizeStream):
    """``StreamAdapter``'s stream: the core's sentence adapter (``_core.tts_adapter_stream``)
    splits the text into sentences (by the adapter's ``sentence_tokenizer`` when it has one),
    paces them when ``text_pacing`` is on, lowers the wrapped TTS's markup when expressive, and
    synthesizes each one with the wrapped TTS, retried on its own within ``conn_options``.
    Its time to first audio is measured from the first text pushed."""

    _tts: StreamAdapter
    _core_native_retries = True

    async def _core_open(self) -> _core.TtsStream:
        tokenizer, pacer = self._tts._sentence_tokenizer, self._tts._stream_pacer
        return await _core.tts_adapter_stream(
            self._tts._wrapped_tts,
            self._conn_options,
            expressive=self._expressive,
            sentences=tokenizer.stream() if tokenizer else None,
            pacing=pacer,
        )

    _run = SynthesizeStream._core_run_native


class SentenceStreamPacer:
    def __init__(self, *, min_remaining_audio: float = 5.0, max_text_length: int = 300) -> None:
        """Paces the text sent to a TTS: sentences are batched and sent when the audio left to
        play runs low, which wastes less on interruptions and gives the TTS more context.

        The first sentence goes at once. Each later batch waits until the synthesized audio
        stopped growing and at most ``min_remaining_audio`` seconds of it are left to play
        (measured from the first audio pushed), then joins whole sentences with a space until
        it reaches ``max_text_length`` characters.

        Args:
            min_remaining_audio: The audio left, in seconds, below which the next batch goes.
            max_text_length: The longest text sent at once.
        """
        self.min_remaining_audio = min_remaining_audio
        self.max_text_length = max_text_length

    def wrap(self, sent_stream: SentenceStream, audio_emitter: AudioEmitter) -> StreamPacerWrapper:
        """``sent_stream``'s sentences, paced by the audio ``audio_emitter`` pushed. The core
        reads ``sent_stream`` and runs the pacing (``_core._core_PacedSentences``); text pushed
        to the wrapper goes to ``sent_stream``, and closing the wrapper closes it."""
        return StreamPacerWrapper(sent_stream, audio_emitter, pacer=self)
