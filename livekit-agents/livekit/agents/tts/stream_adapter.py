from __future__ import annotations

import asyncio
from .. import tokenize, utils
from ..log import logger
from .tts import (
    TTS,
    TTSCapabilities,
    ChunkedStream,
    SynthesizeStream,
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

    def synthesize(self, text: str) -> ChunkedStream:
        return self._tts.synthesize(text=text)

    def stream(self) -> SynthesizeStream:
        return StreamAdapterWrapper(
            tts=self._tts,
            sentence_tokenizer=self._sentence_tokenizer,
        )


class StreamAdapterWrapper(SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        sentence_tokenizer: tokenize.SentenceTokenizer,
    ) -> None:
        super().__init__()
        self._tts = tts
        self._sent_stream = sentence_tokenizer.stream()
        self._main_task = asyncio.create_task(self._run())

    def push_text(self, token: str | None) -> None:
        self._check_not_closed()
        self._sent_stream.push_text(token)

    async def aclose(self) -> None:
        if self._closed:
            return

        self._do_close()
        await self._sent_stream.aclose()
        await utils.aio.gracefully_cancel(self._main_task)

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        try:
            async for ev in self._sent_stream:
                if ev.type == tokenize.TokenEventType.TOKEN:
                    async for audio in self._tts.synthesize(ev.token):
                        audio.end_of_segment = False
                        self._event_ch.put_nowait(audio)

        finally:
            self._event_ch.put_nowait(None)
