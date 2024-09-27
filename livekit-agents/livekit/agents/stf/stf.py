from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Union

from livekit import rtc

from ..utils import aio
from ..tts import SynthesizedAudio


class STF(ABC):
    def __init__(
        self, *, width: int, height: int, frame_rate: int
    ) -> None:
        self._width = width
        self._height = height
        self._frame_rate = frame_rate

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_rate(self) -> int:
        return self._frame_rate

    @abstractmethod
    def idle_stream(self) -> SynthesizeStream: ...

    @abstractmethod
    def speech_stream(self) -> SynthesizeStream: ...


# TODO: add generation from text for the tts models that doesn't support alignment
class SynthesizeStream(ABC):
    class _FlushSentinel:
        pass

    def __init__(self):
        self._input_ch = aio.Chan[Union[SynthesizedAudio, SynthesizeStream._FlushSentinel]]()
        self._event_ch = aio.Chan[rtc.VideoFrameEvent]()
        self._task = asyncio.create_task(self._main_task(), name="STV._main_task")
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @abstractmethod
    async def _main_task(self) -> None: ...

    def push_audio(self, chunk: SynthesizedAudio) -> None:
        """Push some audio to synthesize face frames"""
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(chunk)

    def flush(self) -> None:
        """Mark the end of the current segment"""
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        """Mark the end of input, no more audio will be pushed"""
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        """Close ths stream immediately"""
        self._input_ch.close()
        await aio.gracefully_cancel(self._task)
        self._event_ch.close()

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self._input_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")

    async def __anext__(self) -> rtc.VideoFrameEvent:
        return await self._event_ch.__anext__()

    def __aiter__(self) -> AsyncIterator[rtc.VideoFrameEvent]:
        return self
