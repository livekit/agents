from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Optional

from livekit import rtc

from ..utils import AudioBuffer, merge_frames
from ..vad import VAD, VADEventType
from .stt import (
    STT,
    SpeechEvent,
    SpeechEventType,
    SpeechStream,
)


class StreamAdapter(STT):
    def __init__(
        self,
        *,
        stt: STT,
        vad: VAD,
    ) -> None:
        super().__init__(streaming_supported=True)
        self._vad = vad
        self._stt = stt

    @property
    def wrapped_stt(self) -> STT:
        return self._stt

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
        vad: VAD,
        stt: STT,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self._vad = vad
        self._stt = stt
        self._event_queue = asyncio.Queue[Optional[SpeechEvent]]()
        self._main_task = asyncio.create_task(self._run())
        self._vad_stream = self._vad.stream()
        self._closed = False
        self._args = args
        self._kwargs = kwargs

    # TODO(theomonnom): smarter adapter, create interim results using another STT?
    async def _run(self) -> None:
        try:
            async for event in self._vad_stream:
                if event.type == VADEventType.START_OF_SPEECH:
                    start_event = SpeechEvent(SpeechEventType.START_OF_SPEECH)
                    self._event_queue.put_nowait(start_event)
                elif event.type == VADEventType.END_OF_SPEECH:
                    merged_frames = merge_frames(event.frames)
                    event = await self._stt.recognize(
                        buffer=merged_frames, *self._args, **self._kwargs
                    )
                    self._event_queue.put_nowait(event)

                    final_event = SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[event.alternatives[0]],
                    )
                    self._event_queue.put_nowait(final_event)

                    end_event = SpeechEvent(
                        type=SpeechEventType.END_OF_SPEECH,
                        alternatives=[event.alternatives[0]],
                    )
                    self._event_queue.put_nowait(end_event)
        except Exception:
            logging.exception("stt stream adapter failed")
        finally:
            self._event_queue.put_nowait(None)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._vad_stream.push_frame(frame)

    async def aclose(self, *, wait: bool = True) -> None:
        self._closed = True
        if not wait:
            self._main_task.cancel()

        await self._vad_stream.aclose(wait=wait)
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def __anext__(self) -> SpeechEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration
        return evt
