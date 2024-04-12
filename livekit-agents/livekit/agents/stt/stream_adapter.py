from __future__ import annotations

import asyncio
import contextlib
import logging

from livekit import rtc
from livekit.agents import stt

from ..log import logger
from ..utils import AudioBuffer, merge_frames
from ..vad import VADEventType, VADStream
from .stt import (
    STT,
    SpeechEvent,
    SpeechStream,
)


class StreamAdapter(STT):
    def __init__(
        self,
        stt: STT,
        vad_stream: VADStream,
    ) -> None:
        super().__init__(streaming_supported=True)
        self._vad = vad_stream
        self._stt = stt

    async def recognize(self, *, buffer: AudioBuffer, language: str | None = None):
        return await self._stt.recognize(
            buffer=buffer,
            language=language,
        )

    def stream(
        self,
        *,
        language: str | None = None,
    ) -> SpeechStream:
        return StreamAdapterWrapper(
            self._vad,
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
        self._vad = vad_stream
        self._stt = stt
        self._event_queue = asyncio.Queue[SpeechEvent | None]()
        self._closed = False
        self._args = args
        self._kwargs = kwargs

        self._main_task = asyncio.create_task(self._run())

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logger.error(f"stream adapter task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

    async def _run(self) -> None:
        try:
            async for event in self._vad:
                if event.type == VADEventType.START_OF_SPEECH:
                    start_event = SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    self._event_queue.put_nowait(start_event)
                elif event.type == VADEventType.END_OF_SPEECH:
                    merged_frames = merge_frames(event.speech)
                    event = await self._stt.recognize(
                        buffer=merged_frames, *self._args, **self._kwargs
                    )
                    self._event_queue.put_nowait(event)

                    end_event = SpeechEvent(
                        type=stt.SpeechEventType.END_OF_SPEECH,
                        alternatives=[event.alternatives[0]],
                    )
                    self._event_queue.put_nowait(end_event)
        except Exception as e:
            logging.exception(f"stream adapter failed: {e}")
        finally:
            self._event_queue.put_nowait(None)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._vad.push_frame(frame)

    async def aclose(self, *, wait: bool = True) -> None:
        self._closed = True
        if not wait:
            self._main_task.cancel()

        await self._vad.aclose(wait=wait)
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def __anext__(self) -> SpeechEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration
        return evt
