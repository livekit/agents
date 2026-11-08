# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from .._bindings import core_legs
from ..tokenize import SentenceStream
from . import tts_availability
from .tokenizer import TokenData
from .tts import ChunkedStream, SynthesizeStream

if TYPE_CHECKING:
    from livekit.agents.tts import TTS, AudioEmitter

    from .. import tts
    from .llm_fallback import AvailabilityChanged
    from .stream_pacer import SentenceStreamPacer


class FallbackAdapter:
    """`tts.FallbackAdapter`'s chain, built once per loop from the adapter's attributes; a failed
    TTS is skipped until it recovers.
    """

    if TYPE_CHECKING:
        _members: list[TTS[Any]]
        _max_retry_per_tts: int
        _attempt_timeout: float | None
        _sample_rate: int

        def _core_all_unavailable(self) -> None: ...

        def emit(self, event: str, *args: Any) -> None: ...

    @property
    def _native(self) -> _core.FallbackTts:
        return _bindings.on_loop(
            self,
            lambda: _core.FallbackTts(
                core_legs(self._members),
                max_retry_per_tts=self._max_retry_per_tts,
                attempt_timeout=self._attempt_timeout,
                sample_rate=self._sample_rate,
                on_availability=self,
            ),
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
