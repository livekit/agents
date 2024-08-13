from __future__ import annotations

import asyncio
from typing import Any

from .. import utils
from ..log import logger
from ..vad import VAD, VADEventType
from .stt import STT, SpeechEvent, SpeechEventType, SpeechStream, STTCapabilities


class StreamAdapter(STT):
    def __init__(self, *, stt: STT, vad: VAD) -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=False)
        )
        self._vad = vad
        self._stt = stt

    @property
    def wrapped_stt(self) -> STT:
        return self._stt

    async def recognize(
        self, buffer: utils.AudioBuffer, *, language: str | None = None
    ):
        return await self._stt.recognize(buffer=buffer, language=language)

    def stream(self, *, language: str | None = None) -> SpeechStream:
        return StreamAdapterWrapper(self._vad, self._stt, language=language)


class StreamAdapterWrapper(SpeechStream):
    def __init__(self, vad: VAD, stt: STT, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._vad = vad
        self._stt = stt
        self._vad_stream = self._vad.stream()
        self._args = args
        self._kwargs = kwargs

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        async def _forward_input():
            """forward input to vad"""
            async for input in self._input_ch:
                if isinstance(input, self._FlushSentinel):
                    self._vad_stream.flush()
                    continue
                self._vad_stream.push_frame(input)

            self._vad_stream.end_input()

        async def _recognize():
            """recognize speech from vad"""
            async for event in self._vad_stream:
                if event.type == VADEventType.START_OF_SPEECH:
                    self._event_ch.send_nowait(
                        SpeechEvent(SpeechEventType.START_OF_SPEECH)
                    )
                elif event.type == VADEventType.END_OF_SPEECH:
                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.END_OF_SPEECH,
                        )
                    )

                    merged_frames = utils.merge_frames(event.frames)
                    t_event = await self._stt.recognize(
                        buffer=merged_frames, *self._args, **self._kwargs
                    )

                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[t_event.alternatives[0]],
                        )
                    )

        tasks = [
            asyncio.create_task(_forward_input(), name="forward_input"),
            asyncio.create_task(_recognize(), name="recognize"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
