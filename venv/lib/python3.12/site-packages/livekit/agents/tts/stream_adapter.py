import asyncio
import logging
from typing import AsyncIterable

from ..tokenize import SentenceStream, SentenceTokenizer
from .tts import (
    TTS,
    SynthesisEvent,
    SynthesisEventType,
    SynthesizedAudio,
    SynthesizeStream,
)


class StreamAdapterWrapper(SynthesizeStream):
    def __init__(self, tts: TTS, sentence_stream: SentenceStream) -> None:
        super().__init__()
        self._closed = False
        self._tts = tts
        self._sentence_stream = sentence_stream
        self._queue = asyncio.Queue[str]()
        self._event_queue = asyncio.Queue[SynthesisEvent]()

        self._main_task = asyncio.create_task(self._run())

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"speech task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

    async def _run(self) -> None:
        while True:
            try:
                sentence = await self._sentence_stream.__anext__()
                audio = await self._tts.synthesize(text=sentence.text)
                self._event_queue.put_nowait(
                    SynthesisEvent(type=SynthesisEventType.AUDIO, audio=audio)
                )
            except asyncio.CancelledError:
                break

        self._closed = True

    def push_text(self, token: str) -> None:
        self._sentence_stream.push_text(token)

    async def flush(self) -> None:
        await self._sentence_stream.flush()

    async def aclose(self) -> None:
        self._main_task.cancel()
        try:
            await self._main_task
        except asyncio.CancelledError:
            pass

    async def __anext__(self) -> SynthesisEvent:
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()


class StreamAdapter(TTS):
    def __init__(self, tts: TTS, tokenizer: SentenceTokenizer) -> None:
        super().__init__(streaming_supported=True)
        self._tts = tts
        self._tokenizer = tokenizer

    def synthesize(self, *, text: str) -> AsyncIterable[SynthesizedAudio]:
        return self._tts.synthesize(text=text)

    def stream(self) -> SynthesizeStream:
        return StreamAdapterWrapper(self._tts, self._tokenizer.stream())
