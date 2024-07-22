from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Optional, Union

from .. import tokenize, utils
from ..log import logger
from .tts import (
    TTS,
    ChunkedStream,
    SynthesizedAudio,
    SynthesizeStream,
)


class StreamAdapter(TTS):
    def __init__(
        self,
        *,
        tts: TTS,
        sentence_tokenizer: tokenize.SentenceTokenizer,
        max_concurrent_requests: int = 5,
    ) -> None:
        super().__init__(
            streaming_supported=True,
            sample_rate=tts.sample_rate,
            num_channels=tts.num_channels,
        )
        self._tts = tts
        self._sentence_tokenizer = sentence_tokenizer
        self._max_concurrent_requests = max(1, max_concurrent_requests)

    def synthesize(self, text: str) -> ChunkedStream:
        return self._tts.synthesize(text=text)

    def stream(self) -> SynthesizeStream:
        return StreamAdapterWrapper(
            tts=self._tts,
            sentence_tokenizer=self._sentence_tokenizer,
            max_concurrent_requests=self._max_concurrent_requests,
        )


@dataclass
class _SynthTask:
    sentence: str
    audio_rx: utils.aio.ChanReceiver[SynthesizedAudio]


class _SegmentStart: ...


class _SegmentEnd: ...


class StreamAdapterWrapper(SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        sentence_tokenizer: tokenize.SentenceTokenizer,
        max_concurrent_requests: int,
    ) -> None:
        super().__init__()
        self._closed = False
        self._tts = tts
        self._sent_tokenizer = sentence_tokenizer
        self._sent_stream = sentence_tokenizer.stream()

        self._event_q = asyncio.Queue[Optional[SynthesisEvent]]()
        self._sync_q = asyncio.Queue[
            Optional[Union[_SegmentStart, _SynthTask, _SegmentEnd]]
        ]()
        self._bg_task = asyncio.create_task(self._run())

        self._synth_tasks: set[asyncio.Task[None]] = set()
        self._sem = asyncio.Semaphore(max_concurrent_requests)

    def push_text(self, token: str | None) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._sent_stream.push_text(token)

    async def aclose(self, *, wait: bool = True) -> None:
        self._closed = True
        if not wait:
            self._bg_task.cancel()

        await self._sent_stream.aclose()
        with contextlib.suppress(asyncio.CancelledError):
            await self._bg_task

    async def _forward(self) -> None:
        while True:
            token = await self._sync_q.get()
            if token is None:
                break

            if isinstance(token, _SegmentStart):
                self._event_q.put_nowait(
                    SynthesisEvent(type=SynthesisEventType.STARTED)
                )

            elif isinstance(token, _SynthTask):
                async for audio in token.audio_rx:
                    self._event_q.put_nowait(
                        SynthesisEvent(type=SynthesisEventType.AUDIO, audio=audio)
                    )
            elif isinstance(token, _SegmentEnd):  # type: ignore
                self._event_q.put_nowait(
                    SynthesisEvent(type=SynthesisEventType.FINISHED)
                )

    async def _synthesize(
        self, sentence: str, audio_tx: utils.aio.ChanSender[SynthesizedAudio]
    ) -> None:
        async with self._sem:
            stream = self._tts.synthesize(text=sentence)
            try:
                async for audio in stream:
                    audio_tx.send_nowait(audio)

            finally:
                audio_tx.close()
                await stream.aclose()

    async def _schedule(self) -> None:
        async for ev in self._sent_stream:
            if ev.type == tokenize.TokenEventType.STARTED:
                self._sync_q.put_nowait(_SegmentStart())

            elif ev.type == tokenize.TokenEventType.TOKEN:
                audio_rx = audio_tx = utils.aio.Chan[SynthesizedAudio]()
                task = asyncio.create_task(self._synthesize(ev.token, audio_tx))
                self._synth_tasks.add(task)
                task.add_done_callback(self._synth_tasks.discard)
                self._sync_q.put_nowait(
                    _SynthTask(sentence=ev.token, audio_rx=audio_rx)
                )
            elif ev.type == tokenize.TokenEventType.FINISHED:
                self._sync_q.put_nowait(_SegmentEnd())

        self._sync_q.put_nowait(None)

    async def _run(self) -> None:
        try:
            await asyncio.gather(self._forward(), self._schedule())
        except Exception:
            logger.exception("tts stream adapter failed")
        finally:
            for task in self._synth_tasks:
                task.cancel()

            await asyncio.gather(*self._synth_tasks, return_exceptions=True)
            self._event_q.put_nowait(None)

    async def __anext__(self) -> SynthesisEvent:
        evt = await self._event_q.get()
        if evt is None:
            raise StopAsyncIteration

        return evt
