# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from .. import _bindings, _core
from .._bindings import core_leg, core_legs
from ..log import logger
from ..tokenize import SentenceStream
from ..types import DEFAULT_API_CONNECT_OPTIONS
from . import tts_availability
from .tokenizer import TokenData
from .tts import TTS, ChunkedStream, SynthesizeStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    from livekit.agents.tokenize import SentenceTokenizer
    from livekit.agents.tts import AudioEmitter

    from .. import tts
    from .chat_items import MetricsMetadata
    from .connect import APIConnectOptions
    from .llm_fallback import AvailabilityChanged
    from .stream_pacer import SentenceStreamPacer, TextPacing


class FallbackAdapter(TTS[Literal["tts_availability_changed"]]):
    """`tts.FallbackAdapter`: several TTS behind one; a failed one is skipped until it recovers."""

    if TYPE_CHECKING:
        _members: list[TTS[Any]]

    def __init__(
        self,
        tts: Sequence[TTS[Any]],
        *,
        max_retry_per_tts: int = 2,
        attempt_timeout: float | None = None,
        sample_rate: int | None = None,
    ) -> None:
        """Several TTS behind one: a failed one is skipped until it recovers.

        Args:
            tts: The TTS, in order of preference.
            max_retry_per_tts: Retries of each before moving to the next.
            attempt_timeout: Caps each limit of one attempt; ``None`` keeps the caller's
                ``conn_options`` limits.
            sample_rate: The output rate; the highest of the TTS by default, the others
                resampled.
        """
        self._native = _core.FallbackTts(
            core_legs(tts),
            max_retry_per_tts=max_retry_per_tts,
            attempt_timeout=attempt_timeout,
            sample_rate=sample_rate,
            on_availability=self,
        )
        self._members = list(tts)
        super().__init__(
            capabilities=self._native.capabilities,
            sample_rate=self._native.sample_rate,
            num_channels=self._native.num_channels,
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, FallbackAdapter, "synthesize", "stream")
        return self if replaced else self._native

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:
        self.emit(
            "tts_availability_changed",
            tts_availability.AvailabilityChangedEvent(self._members[ev.index], ev.available),
        )

    def _core_all_unavailable(self) -> None:
        logger.error("all TTSs are unavailable, retrying..")

    @property
    def _active(self) -> TTS[Any]:
        return self._members[self._native.active_index]

    @property
    def model(self) -> str:
        return self._active.model

    @property
    def provider(self) -> str:
        return self._active.provider

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        return self._active.metrics_metadata

    async def aclose(self) -> None:
        await super().aclose()
        await self._native.aclose()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return FallbackChunkedStream(
            tts=cast("tts.FallbackAdapter", self),
            input_text=text,
            conn_options=conn_options,
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        return FallbackSynthesizeStream(
            tts=cast("tts.FallbackAdapter", self),
            conn_options=conn_options,
        )


class FallbackChunkedStream(ChunkedStream):
    """`tts.FallbackAdapter.synthesize()`: the chain's synthesis, which retries and falls back."""

    _core_native_retries = True
    _tts: tts.FallbackAdapter

    def __init__(
        self,
        *,
        tts: tts.FallbackAdapter,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)

    async def _core_open(self) -> _core.TtsStream:
        return await self._tts._native.synthesize(self._input_text, conn_options=self._conn_options)

    _run = ChunkedStream._core_run_native


class FallbackSynthesizeStream(SynthesizeStream):
    """`tts.FallbackAdapter.stream()`: the chain's synthesis, which retries and falls back."""

    _core_native_retries = True
    _tts: tts.FallbackAdapter

    def __init__(self, *, tts: tts.FallbackAdapter, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)

    async def _core_open(self) -> _core.TtsStream:
        return await self._tts._native.stream(
            conn_options=self._conn_options,
            expressive=self._expressive,
        )

    _run = SynthesizeStream._core_run_native


class StreamAdapter(TTS):
    """`tts.StreamAdapter`: a non-streaming TTS as a streaming one."""

    def __init__(
        self,
        *,
        tts: TTS[Any],
        sentence_tokenizer: SentenceTokenizer | None = None,
        text_pacing: TextPacing | None = None,
    ) -> None:
        """A non-streaming TTS as a streaming one: the text is split into sentences, each
        synthesized on its own.

        Args:
            tts: The non-streaming TTS.
            sentence_tokenizer: What splits the text; `tokenize.SentenceTokenizer`'s sentences by default.
            text_pacing: Batches sentences as the audio plays out; ``True`` for the default
                ``SentenceStreamPacer``.
        """
        self._native = _core.TtsStreamAdapter(
            tts=core_leg(tts),
            sentence_tokenizer=sentence_tokenizer,
            text_pacing=text_pacing,
        )
        self._tts = tts
        super().__init__(
            capabilities=self._native.capabilities,
            sample_rate=self._native.sample_rate,
            num_channels=self._native.num_channels,
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, StreamAdapter, "stream")
        return self if replaced else self._native

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        return StreamAdapterWrapper(tts=cast("tts.StreamAdapter", self), conn_options=conn_options)


class StreamAdapterWrapper(SynthesizeStream):
    """`tts.StreamAdapter.stream()`: the text split into sentences, paced when ``text_pacing`` is
    on, each sentence synthesized by the wrapped TTS and retried on its own within
    ``conn_options``. Its time to first audio is measured from the first text pushed.
    """

    _core_native_retries = True
    _tts: tts.StreamAdapter

    def __init__(self, *, tts: tts.StreamAdapter, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)

    async def _core_open(self) -> _core.TtsStream:
        return await self._tts._native.stream(
            conn_options=self._conn_options,
            expressive=self._expressive,
        )

    _run = SynthesizeStream._core_run_native


class StreamPacerWrapper(SentenceStream):
    """`tts.StreamPacerWrapper`: a sentence stream paced by the audio its emitter pushed."""

    def __init__(
        self,
        sent_stream: SentenceStream,
        audio_emitter: AudioEmitter,
        *,
        pacer: SentenceStreamPacer,
    ) -> None:
        super().__init__()
        self._sent_stream = sent_stream
        self._native = _core._core_PacedSentences(sent_stream, audio_emitter, pacer=pacer)

    def push_text(self, text: str) -> None:
        self._sent_stream.push_text(text)

    def flush(self) -> None:
        self._sent_stream.flush()

    def end_input(self) -> None:
        self._sent_stream.end_input()

    async def __anext__(self) -> TokenData:
        if (value := await self._native.next()) is None:
            raise StopAsyncIteration
        return TokenData(token=value)

    async def aclose(self) -> None:
        await self._native.aclose()


FallbackChunkedStream.__module__ = "livekit.agents.tts"
FallbackSynthesizeStream.__module__ = "livekit.agents.tts"
StreamAdapterWrapper.__module__ = "livekit.agents.tts"
StreamPacerWrapper.__module__ = "livekit.agents.tts"
