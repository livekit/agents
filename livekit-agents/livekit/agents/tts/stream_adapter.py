from __future__ import annotations

import asyncio
from typing import AsyncIterable

from .. import tokenize, utils
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from .tts import (
    TTS,
    ChunkedStream,
    SynthesizedAudio,
    SynthesizeStream,
    TTSCapabilities,
)


class StreamAdapter(TTS):
    def __init__(
        self,
        *,
        tts: TTS,
        sentence_tokenizer: tokenize.SentenceTokenizer,
    ) -> None:
        super().__init__(
            capabilities=TTSCapabilities(
                streaming=True,
            ),
            sample_rate=tts.sample_rate,
            num_channels=tts.num_channels,
        )
        self._tts = tts
        self._sentence_tokenizer = sentence_tokenizer

        @self._tts.on("metrics_collected")
        def _forward_metrics(*args, **kwargs):
            self.emit("metrics_collected", *args, **kwargs)

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        return self._tts.synthesize(text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "StreamAdapterWrapper":
        return StreamAdapterWrapper(
            tts=self,
            conn_options=conn_options,
            wrapped_tts=self._tts,
            sentence_tokenizer=self._sentence_tokenizer,
        )


class StreamAdapterWrapper(SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
        wrapped_tts: TTS,
        sentence_tokenizer: tokenize.SentenceTokenizer,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._wrapped_tts = wrapped_tts
        self._sent_stream = sentence_tokenizer
        self._segments_ch = utils.aio.Chan[tokenize.SentenceStream]()

    async def _metrics_monitor_task(
        self, event_aiter: AsyncIterable[SynthesizedAudio]
    ) -> None:
        pass  # do nothing

    async def _run(self) -> None:
        async def _tokenize_input():
            """tokenize text"""
            input_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if input_stream is None:
                        # new segment (after flush for e.g)
                        input_stream = self._sent_stream.stream()
                        self._segments_ch.send_nowait(input_stream)

                    input_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if input_stream is not None:
                        input_stream.end_input()

                    input_stream = None
            self._segments_ch.close()

        async def _run_segments():
            async for input_stream in self._segments_ch:
                await _synthesize(input_stream)

        async def _synthesize(input_stream):
            async for ev in input_stream:
                last_audio: SynthesizedAudio | None = None
                async for audio in self._wrapped_tts.synthesize(
                    ev.token, segment_id=ev.segment_id
                ):
                    if last_audio is not None:
                        self._event_ch.send_nowait(last_audio)

                    last_audio = audio

                if last_audio is not None and ev.is_final:
                    last_audio.is_final = True
                    self._event_ch.send_nowait(last_audio)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
