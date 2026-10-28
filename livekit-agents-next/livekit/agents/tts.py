"""Text to speech: the base classes a provider subclasses (generated from the core's
declarations), the expressive markup helpers over the core's per-provider tables
(``_core.Markup``), and the adapters, whose engines are the core's (``_core.FallbackTts``,
``_core.tts_adapter_stream``, ``_core.SentencePacer``)."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from . import _bindings, _core, tokenize, utils
from ._generated.tts import (
    TTS,
    ChunkedStream,
    MarkupInfo,
    SynthesizedAudio,
    SynthesizeStream,
    TTSCapabilities,
    TTSError,
)
from .log import logger
from .tokenize import SentenceStream, TokenData
from .types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr

if TYPE_CHECKING:
    from .llm.chat_context import MetricsMetadata

__all__ = [
    "TTS",
    "SynthesizedAudio",
    "SynthesizeStream",
    "TTSCapabilities",
    "MarkupInfo",
    "StreamAdapterWrapper",
    "StreamAdapter",
    "ChunkedStream",
    "AvailabilityChangedEvent",
    "FallbackAdapter",
    "FallbackChunkedStream",
    "FallbackSynthesizeStream",
    "AudioEmitter",
    "TTSError",
    "SentenceStreamPacer",
]


# The core's class under its public name: `audio/pcm` is framed as it is, any other container
# decoded (ffmpeg) as it arrives.
AudioEmitter = _core.AudioEmitter


class ExpressiveTag(TypedDict):
    """An expressive markup tag stripped from a transcript: the tag name (or the expr marker's
    type) and its payload (the ``value`` attribute, the expr ``label`` or the inner text)."""

    type: str
    value: str


def split_all_markup(text: str) -> tuple[str, list[ExpressiveTag]]:
    """Strip every provider's expressive XML markup; brackets are prose and stay."""
    clean, tags = _core.split_all_markup(text)
    return clean, [ExpressiveTag(type=t["type"], value=t["value"]) for t in tags]


def strip_all_markup(text: str) -> str:
    return _core.strip_all_markup(text)  # type: ignore[no-any-return]


def split_expr_markup(text: str) -> tuple[str, list[ExpressiveTag]]:
    """Strip only the ``<expr/>`` dialect, returning the clean text and its markers; the
    provider-native tags stay in the words."""
    clean, tags = _core.split_expr_markup(text)
    return clean, [ExpressiveTag(type=t["type"], value=t["value"]) for t in tags]


def strip_expr_markup(text: str) -> str:
    return split_expr_markup(text)[0]


def _into(output_emitter: AudioEmitter, *, stream: bool) -> Any:
    """What forwards a core synthesis's audio into ``output_emitter``, one segment if streamed."""
    started = not stream

    def out(audio: Any) -> None:
        nonlocal started
        if not started:
            started = True
            output_emitter.start_segment(segment_id=audio["segment_id"] or utils.shortuuid())
        if words := audio["timed_transcript"]:
            output_emitter.push_timed_transcript(words)
        output_emitter.push_frame(audio["frame"])

    return out


async def _drain_native(native: Any, output_emitter: AudioEmitter, *, stream: bool) -> None:
    """Forward a core synthesis (``_core.*TtsStream``) into ``output_emitter``."""
    out = _into(output_emitter, stream=stream)
    while (audio := await native.recv()) is not None:
        out(audio)


@dataclass
class AvailabilityChangedEvent:
    tts: TTS
    available: bool


class FallbackAdapter(TTS[Literal["tts_availability_changed"]]):
    """Several TTS behind one: a failed one is skipped until it recovers."""

    def __init__(
        self,
        tts: list[TTS],
        *,
        max_retry_per_tts: int = 2,
        sample_rate: int | None = None,
    ) -> None:
        """
        Args:
            tts: The TTS, in order of preference.
            max_retry_per_tts: Retries of each before moving to the next.
            sample_rate: The output rate; the highest of the TTS by default, the others
                resampled.
        """
        if not tts:
            raise ValueError("at least one TTS instance must be provided.")
        if len({t.num_channels for t in tts}) != 1:
            raise ValueError("all TTS must have the same number of channels")
        self._tts_instances = tts
        self._max_retry_per_tts = max_retry_per_tts
        sample_rate = sample_rate or max(t.sample_rate for t in tts)
        super().__init__(
            # the chain's own merge, read before any chain is built
            capabilities=TTSCapabilities(
                **_core.tts_fallback_capabilities([t.capabilities for t in tts])
            ),
            sample_rate=sample_rate,
            num_channels=tts[0].num_channels,
        )

    @property
    def _native(self) -> _core.FallbackTts:
        # one per running loop: its members are called on the loop the chain was built on; a
        # member without a duplex path streams through the core's sentence adapter
        return _bindings.on_loop(
            self,
            lambda: _core.FallbackTts(
                [_bindings.core_leg(t) for t in self._tts_instances],
                max_retry_per_tts=self._max_retry_per_tts,
                sample_rate=self.sample_rate,
                on_availability=self,
            ),
        )

    def _core_leg(self) -> Any:
        if _bindings.overrides(self, FallbackAdapter, "synthesize", "stream"):
            return self
        return self._native

    def _core_availability_changed(self, ev: dict[str, Any]) -> None:
        self.emit(
            "tts_availability_changed",
            AvailabilityChangedEvent(
                tts=self._tts_instances[ev["index"]], available=ev["available"]
            ),
        )

    def _core_all_unavailable(self) -> None:
        logger.error("all TTSs are unavailable, retrying..")

    @property
    def _active(self) -> TTS:
        chains = _bindings.built_on_loop(self)
        return self._tts_instances[chains[0].active_index if chains else 0]

    @property
    def model(self) -> str:
        return self._active.model

    @property
    def provider(self) -> str:
        return self._active.provider

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        return self._active.metrics_metadata

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FallbackChunkedStream:
        # the core's chain retries and falls back: the stream does not retry on top of it
        no_retry = replace(conn_options, max_retry=0)
        return FallbackChunkedStream(tts=self, input_text=text, conn_options=no_retry)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FallbackSynthesizeStream:
        return FallbackSynthesizeStream(tts=self, conn_options=replace(conn_options, max_retry=0))

    def prewarm(self) -> None:
        self._tts_instances[0].prewarm()

    async def aclose(self) -> None:
        for chain in _bindings.built_on_loop(self):
            await chain.aclose()


def _initialize(output_emitter: AudioEmitter, tts: TTS, *, stream: bool) -> None:
    output_emitter.initialize(
        request_id=utils.shortuuid(),
        sample_rate=tts.sample_rate,
        num_channels=tts.num_channels,
        mime_type="audio/pcm",
        stream=stream,
    )


class FallbackChunkedStream(ChunkedStream):
    _tts: FallbackAdapter

    async def _run(self, output_emitter: AudioEmitter) -> None:
        _initialize(output_emitter, self._tts, stream=False)
        native = await self._tts._native.synthesize(
            self._input_text, conn_options=self._conn_options
        )
        try:
            await _drain_native(native, output_emitter, stream=False)
        finally:
            await native.aclose()


class FallbackSynthesizeStream(SynthesizeStream):
    _tts: FallbackAdapter

    async def _run(self, output_emitter: AudioEmitter) -> None:
        _initialize(output_emitter, self._tts, stream=True)
        native = await self._tts._native.stream(conn_options=self._conn_options)
        await _duplex(self, native, output_emitter, self.input, lambda text: text)


async def _duplex(
    stream: SynthesizeStream,
    native: Any,
    output_emitter: AudioEmitter,
    texts: Any,
    lower: Any,
    push: Any = None,
) -> None:
    """Text from ``texts`` into a core duplex synthesis (``push``, its ``push_text`` by default),
    its audio into ``output_emitter``."""
    push = push or native.push_text

    def feed(text: Any) -> None:
        if isinstance(text, SynthesizeStream.Flush):
            native.flush()
        elif text := lower(text):
            stream.mark_started()
            push(text)

    await _bindings.pump(native, texts, feed, _into(output_emitter, stream=True))


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

    def _set_expressive(self, enabled: bool) -> None:
        super()._set_expressive(enabled)
        self._wrapped_tts._set_expressive(enabled)

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
    _tts: StreamAdapter

    def __init__(self, *, tts: StreamAdapter, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        # each sentence is retried by the core's adapter, not the whole stream
        self._core_attempts.conn = replace(conn_options, max_retry=0)
        # fixed for this synthesis: the framework sets it before stream()
        self._expressive = tts._wrapped_tts._expressive

    async def _run(self, output_emitter: AudioEmitter) -> None:
        markup = self._tts._wrapped_tts.markup
        lowering = bool(markup._provider_key()) and self._expressive
        # a marker must never be split across two sentences, so markup needs an xml-aware split
        tokenizer = self._tts._sentence_tokenizer or tokenize.blingfire.SentenceTokenizer(
            retain_format=True, xml_aware=lowering
        )
        sentences: SentenceStream = tokenizer.stream()
        if self._tts._stream_pacer:
            sentences = self._tts._stream_pacer.wrap(sentences, output_emitter)
        _initialize(output_emitter, self._tts, stream=True)
        native = await _core.tts_adapter_stream(self._tts._wrapped_tts, self._conn_options)

        async def _forward() -> None:
            async for data in self.input:
                if isinstance(data, self.Flush):
                    sentences.flush()
                else:
                    sentences.push_text(markup.normalize(data) if lowering else data)
            sentences.end_input()

        def _lower(text: str) -> str:
            text = text.strip()
            return markup.convert(markup.normalize(text)).strip() if lowering and text else text

        forward = asyncio.create_task(_forward())
        try:
            await _duplex(self, native, output_emitter, _tokens(sentences), _lower)
        finally:
            await utils.aio.cancel_and_wait(forward)
            await sentences.aclose()


async def _tokens(sentences: SentenceStream) -> Any:
    async for ev in sentences:
        yield ev.token


class SentenceStreamPacer:
    def __init__(self, *, min_remaining_audio: float = 5.0, max_text_length: int = 300) -> None:
        """Paces the text sent to a TTS: sentences are batched and sent when the audio left to
        play runs low, which wastes less on interruptions and gives the TTS more context.

        Args:
            min_remaining_audio: The audio left, in seconds, below which the next batch goes.
            max_text_length: The longest text sent at once.
        """
        self._min_remaining_audio = min_remaining_audio
        self._max_text_length = max_text_length

    def wrap(self, sent_stream: SentenceStream, audio_emitter: AudioEmitter) -> StreamPacerWrapper:
        return StreamPacerWrapper(sent_stream, audio_emitter, pacer=self)


class StreamPacerWrapper(SentenceStream):
    def __init__(
        self,
        sent_stream: SentenceStream,
        audio_emitter: AudioEmitter,
        *,
        pacer: SentenceStreamPacer,
    ) -> None:
        super().__init__()
        self._sent_stream = sent_stream
        self._audio_emitter = audio_emitter
        self._pacer = _core.SentencePacer(pacer._min_remaining_audio, pacer._max_text_length)
        self._wakeup = asyncio.Event()
        self._recv_atask = asyncio.create_task(self._recv_task())
        self._send_atask = asyncio.create_task(self._send_task())
        self._send_atask.add_done_callback(lambda _: self._event_ch.close())

    def push_text(self, text: str) -> None:
        self._sent_stream.push_text(text)

    def flush(self) -> None:
        self._sent_stream.flush()

    def end_input(self) -> None:
        self._sent_stream.end_input()

    async def aclose(self) -> None:
        await self._sent_stream.aclose()
        await utils.aio.cancel_and_wait(self._recv_atask, self._send_atask)

    async def _recv_task(self) -> None:
        try:
            async for ev in self._sent_stream:
                self._pacer.push(ev.token)
                self._wakeup.set()
        finally:
            self._pacer.end()
            self._wakeup.set()

    async def _send_task(self) -> None:
        while True:
            self._wakeup.clear()
            text, remaining, delay = self._pacer.poll(
                time.time(), self._audio_emitter.pushed_duration()
            )
            if text:
                self._event_ch.send_nowait(TokenData(token=text))
                logger.debug(
                    "sent text to tts", extra={"lk.pii.text": text, "remaining_audio": remaining}
                )
            if delay is None:
                return
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), delay)
