# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .. import _bindings, _core
from .._bindings import core_legs
from ..log import logger
from ..tokenize import SentenceStream
from . import tts_availability
from .tokenizer import TokenData
from .tts import TTS, ChunkedStream, SynthesizeStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    from livekit.agents.tts import AudioEmitter

    from .. import tts
    from .chat_items import MetricsMetadata
    from .llm_fallback import AvailabilityChanged
    from .stream_pacer import SentenceStreamPacer


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
            attempt_timeout: Caps each phase of one attempt; ``None`` keeps the caller's
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


class FallbackChunkedStream(ChunkedStream):
    """`tts.FallbackAdapter.synthesize()`: the chain's synthesis, which retries and falls back."""

    _core_native_retries = True
    _tts: tts.FallbackAdapter

    async def _core_open(self) -> _core.TtsStream:
        return await self._tts._native.synthesize(self._input_text, self._conn_options)

    _run = ChunkedStream._core_run_native


class FallbackSynthesizeStream(SynthesizeStream):
    """`tts.FallbackAdapter.stream()`: the chain's synthesis, which retries and falls back."""

    _core_native_retries = True
    _tts: tts.FallbackAdapter

    async def _core_open(self) -> _core.TtsStream:
        return await self._tts._native.stream(self._conn_options, self._expressive)

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
StreamPacerWrapper.__module__ = "livekit.agents.tts"
