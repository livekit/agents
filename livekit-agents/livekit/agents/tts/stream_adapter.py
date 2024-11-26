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
        self._sent_stream = sentence_tokenizer.stream()

    async def _metrics_monitor_task(
        self, event_aiter: AsyncIterable[SynthesizedAudio]
    ) -> None:
        pass  # do nothing

    async def _run(self) -> None:
        async def _forward_input():
            """forward input to vad"""
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_stream.flush()
                    continue
                self._sent_stream.push_text(data)

            self._sent_stream.end_input()

        async def _synthesize():
            async for ev in self._sent_stream:
                last_audio: SynthesizedAudio | None = None
                async for audio in self._wrapped_tts.synthesize(ev.token):
                    if last_audio is not None:
                        self._event_ch.send_nowait(last_audio)

                    last_audio = audio

                if last_audio is not None:
                    last_audio.is_final = True
                    self._event_ch.send_nowait(last_audio)

        tasks = [
            asyncio.create_task(_forward_input()),
            asyncio.create_task(_synthesize()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
