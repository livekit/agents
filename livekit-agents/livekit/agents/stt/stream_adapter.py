import contextlib
import asyncio
import logging
from typing import Optional
from ..vad import VADStream, VADEventType
from ..utils import merge_frames, AudioBuffer
from .stt import (
    STT,
    SpeechStream,
    SpeechEvent,
)
from livekit import rtc


class StreamAdapter(STT):
    def __init__(
        self,
        stt: STT,
        vad_stream: VADStream,
    ) -> None:
        super().__init__(streaming_supported=True)
        self._vad_stream = vad_stream
        self._stt = stt

    async def recognize(self, *, buffer: AudioBuffer, language: Optional[str] = None):
        return await self._stt.recognize(
            buffer=buffer,
            language=language,
        )

    def stream(
        self,
        *,
        language: Optional[str] = None,
    ) -> SpeechStream:
        return StreamAdapterWrapper(
            self._vad_stream,
            self._stt,
            language=language,
        )


class StreamAdapterWrapper(SpeechStream):
    def __init__(
        self,
        vad_stream: VADStream,
        stt: STT,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self._vad_stream = vad_stream
        self._stt = stt
        self._event_queue = asyncio.Queue[SpeechEvent]()
        self._closed = False
        self._args = args
        self._kwargs = kwargs

        self._main_task = asyncio.create_task(self._run())

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"stream adapter task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

    async def _run(self) -> None:
        # listen to vad events and send to stt on END_SPEAKING
        try:
            async for event in self._vad_stream:
                if event.type != VADEventType.END_SPEAKING:
                    continue

                merged_frames = merge_frames(event.speech)
                event = await self._stt.recognize(
                    buffer=merged_frames, *self._args, **self._kwargs
                )
                self._event_queue.put_nowait(event)
        except asyncio.CancelledError:
            pass

        self._closed = True

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._vad_stream.push_frame(frame)

    async def aclose(self) -> None:
        await self._vad_stream.aclose()
        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def flush(self) -> None:
        await self._vad_stream.flush()

    async def __anext__(self) -> SpeechEvent:
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()
