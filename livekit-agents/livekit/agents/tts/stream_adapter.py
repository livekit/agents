from __future__ import annotations

import asyncio
import contextlib
from typing import Optional

from .. import tokenize
from .tts import (
    TTS,
    ChunkedStream,
    SynthesisEvent,
    SynthesisEventType,
    SynthesizeStream,
)


class StreamAdapter(TTS):
    def __init__(
        self, *, tts: TTS, sentence_tokenizer: tokenize.SentenceTokenizer
    ) -> None:
        super().__init__(
            streaming_supported=True,
            sample_rate=tts.sample_rate,
            num_channels=tts.num_channels,
        )
        self._tts = tts
        self._sentence_tokenizer = sentence_tokenizer

    def synthesize(self, text: str) -> ChunkedStream:
        return self._tts.synthesize(text=text)

    def stream(self) -> SynthesizeStream:
        return StreamAdapterWrapper(
            tts=self._tts, sentence_tokenizer=self._sentence_tokenizer
        )


class StreamAdapterWrapper(SynthesizeStream):
    def __init__(
        self, *, tts: TTS, sentence_tokenizer: tokenize.SentenceTokenizer
    ) -> None:
        super().__init__()
        self._closed = False
        self._tts = tts
        self._sentence_tokenizer = sentence_tokenizer
        self._sentence_stream = self._sentence_tokenizer.stream()
        self._event_queue = asyncio.Queue[Optional[SynthesisEvent]]()
        self._main_task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        async for ev in self._sentence_stream:
            if ev.type == tokenize.TokenEventType.STARTED:
                self._event_queue.put_nowait(
                    SynthesisEvent(type=SynthesisEventType.STARTED)
                )
            elif ev.type == tokenize.TokenEventType.TOKEN:
                async for audio in self._tts.synthesize(text=ev.token):
                    self._event_queue.put_nowait(
                        SynthesisEvent(type=SynthesisEventType.AUDIO, audio=audio)
                    )
            elif ev.type == tokenize.TokenEventType.FINISHED:
                self._event_queue.put_nowait(
                    SynthesisEvent(type=SynthesisEventType.FINISHED)
                )
        self._event_queue.put_nowait(None)

    def push_text(self, token: str | None) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._sentence_stream.push_text(token)

    async def aclose(self, *, wait: bool = True) -> None:
        self._closed = True
        if not wait:
            self._main_task.cancel()

        await self._sentence_stream.aclose(wait=wait)
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def __anext__(self) -> SynthesisEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt
