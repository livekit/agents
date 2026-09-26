from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from typing import Any, ClassVar

from .. import tokenize, utils
from ..metrics.provider_request import _provider_request_context
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from .stream_pacer import SentenceStreamPacer
from .tts import (
    TTS,
    AudioEmitter,
    ChunkedStream,
    SynthesizedAudio,
    SynthesizeStream,
    TTSCapabilities,
)

# already a retry mechanism in TTS.synthesize, don't retry in stream adapter
DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout
)


class StreamAdapter(TTS):
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
        self._explicit_tokenizer = bool(sentence_tokenizer)
        self._sentence_tokenizer = sentence_tokenizer or tokenize.blingfire.SentenceTokenizer(
            retain_format=True
        )
        self._markup_tokenizer: tokenize.SentenceTokenizer | None = None
        self._stream_pacer: SentenceStreamPacer | None = None
        if text_pacing is True:
            self._stream_pacer = SentenceStreamPacer()
        elif isinstance(text_pacing, SentenceStreamPacer):
            self._stream_pacer = text_pacing

        self._wrapped_tts.on("metrics_collected", self._on_metrics_collected)
        self._wrapped_tts.on("provider_request_completed", self._on_provider_request_completed)

    class Markup(TTS.Markup):
        # a pass-through speaks whatever dialect it wraps
        def _provider_key(self) -> str:
            assert isinstance(self._tts, StreamAdapter)
            return self._tts._wrapped_tts.markup._provider_key()

    def _tokenizer_for(self, *, lowering: bool) -> tokenize.SentenceTokenizer:
        """The sentence tokenizer for one synthesis.

        A marker must never be split across two tokens -- the sentence-level lowering in
        :class:`StreamAdapterWrapper` would see half a tag and send the halves on as
        words. A label is free-form English and may well contain a period, so an
        unguarded tokenizer really does split them. The framework passes an xml-aware
        tokenizer when it builds the adapter itself; a caller relying on the default gets
        one here, and only while markup is actually flowing, so a plain turn never pays
        the stray-``<`` stall.
        """
        if not lowering or self._explicit_tokenizer:
            return self._sentence_tokenizer
        if self._markup_tokenizer is None:
            self._markup_tokenizer = tokenize.blingfire.SentenceTokenizer(
                retain_format=True, xml_aware=True
            )
        return self._markup_tokenizer

    def _set_expressive(self, enabled: bool) -> None:
        # StreamAdapterWrapper reads the wrapped instance's flag, so an adapter handed
        # straight to the session has to pass this through
        super()._set_expressive(enabled)
        self._wrapped_tts._set_expressive(enabled)

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

    def _on_metrics_collected(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)

    def _on_provider_request_completed(self, *args: Any, **kwargs: Any) -> None:
        self.emit("provider_request_completed", *args, **kwargs)

    async def aclose(self) -> None:
        self._wrapped_tts.off("metrics_collected", self._on_metrics_collected)
        self._wrapped_tts.off("provider_request_completed", self._on_provider_request_completed)


class StreamAdapterWrapper(SynthesizeStream):
    _tts_request_span_name: ClassVar[str] = "tts_stream_adapter"
    _emit_provider_request_attempts: ClassVar[bool] = False

    def __init__(self, *, tts: StreamAdapter, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS)
        self._tts: StreamAdapter = tts
        self._wrapped_tts_conn_options = conn_options
        # Snapshot whether expressive is active now, while the framework holds it fixed
        # for this synthesis (set synchronously before stream()). _run happens later in
        # its own task, and _expressive lives on the shared TTS, so another turn or
        # session could flip it in between.
        self._expressive = tts._wrapped_tts._expressive

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SynthesizedAudio]) -> None:
        async for _ in event_aiter:
            pass

    async def _run(self, output_emitter: AudioEmitter) -> None:
        # the framework's input path for every non-streaming TTS, and the first place
        # whole sentences exist
        markup = self._tts._wrapped_tts.markup
        lowering = bool(markup._provider_key()) and self._expressive

        sent_stream = self._tts._tokenizer_for(lowering=lowering).stream()
        if self._tts._stream_pacer:
            sent_stream = self._tts._stream_pacer.wrap(
                sent_stream=sent_stream,
                audio_emitter=output_emitter,
            )

        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=True,
        )

        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        async def _forward_input() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                    continue

                sent_stream.push_text(markup.normalize(data) if lowering else data)

            sent_stream.end_input()

        async def _synthesize() -> None:
            from ..voice.io import TimedString

            duration = 0.0
            async for ev in sent_stream:
                output_emitter.push_timed_transcript(
                    TimedString(text=ev.token, start_time=duration)
                )

                if not (text := ev.token.strip()):
                    continue

                if lowering:
                    # re-normalize: a marker split across two input chunks isn't caught
                    # by the per-chunk pass above
                    if not (text := markup.convert(markup.normalize(text)).strip()):
                        continue

                self._mark_started()
                with _provider_request_context(
                    self._provider_request_tracker.operation_id,
                    self._provider_request_tracker.fallback_index,
                    self._provider_request_tracker.purpose,
                ):
                    async with self._tts._wrapped_tts.synthesize(
                        text, conn_options=self._wrapped_tts_conn_options
                    ) as tts_stream:
                        async for audio in tts_stream:
                            output_emitter.push_frame(audio.frame)
                            duration += audio.frame.duration
                        output_emitter.flush()

        tasks = [
            asyncio.create_task(_forward_input()),
            asyncio.create_task(_synthesize()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await sent_stream.aclose()
            await utils.aio.cancel_and_wait(*tasks)
