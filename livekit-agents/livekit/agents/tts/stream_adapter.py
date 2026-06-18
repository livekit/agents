from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from typing import Any, ClassVar

from .. import tokenize, utils
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
        self._sentence_tokenizer = sentence_tokenizer or tokenize.blingfire.SentenceTokenizer(
            retain_format=True
        )
        self._stream_pacer: SentenceStreamPacer | None = None
        if text_pacing is True:
            self._stream_pacer = SentenceStreamPacer()
        elif isinstance(text_pacing, SentenceStreamPacer):
            self._stream_pacer = text_pacing

        self._wrapped_tts.on("metrics_collected", self._on_metrics_collected)

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

    async def aclose(self) -> None:
        self._wrapped_tts.off("metrics_collected", self._on_metrics_collected)


class StreamAdapterWrapper(SynthesizeStream):
    _tts_request_span_name: ClassVar[str] = "tts_stream_adapter"

    def __init__(self, *, tts: StreamAdapter, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS)
        self._tts: StreamAdapter = tts
        self._wrapped_tts_conn_options = conn_options

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SynthesizedAudio]) -> None:
        pass  # do nothing

    async def _run(self, output_emitter: AudioEmitter) -> None:
        sent_stream = self._tts._sentence_tokenizer.stream()
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

                sent_stream.push_text(data)

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

                async with self._tts._wrapped_tts.synthesize(
                    text, conn_options=self._wrapped_tts_conn_options
                ) as tts_stream:
                    async for audio in tts_stream:
                        output_emitter.push(audio.frame.data.tobytes())
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
