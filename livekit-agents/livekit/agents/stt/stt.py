from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import AsyncIterator, List, Union

from livekit import rtc

from ..utils import AudioBuffer, aio


@unique
class SpeechEventType(str, Enum):
    START_OF_SPEECH = "start_of_speech"
    """indicate the start of speech
    if the STT doesn't support this event, this will be emitted as the same time as the first INTERIM_TRANSCRIPT"""
    INTERIM_TRANSCRIPT = "interim_transcript"
    """interim transcript, useful for real-time transcription"""
    FINAL_TRANSCRIPT = "final_transcript"
    """final transcript, emitted when the STT is confident enough that a certain
    portion of speech will not change"""
    END_OF_SPEECH = "end_of_speech"
    """indicate the end of speech, emitted when the user stops speaking"""


@dataclass
class SpeechData:
    language: str
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0  # [0, 1]


@dataclass
class SpeechEvent:
    type: SpeechEventType
    alternatives: List[SpeechData] = field(default_factory=list)


@dataclass
class STTCapabilities:
    streaming: bool
    interim_results: bool


class STT(ABC):
    def __init__(self, *, capabilities: STTCapabilities) -> None:
        self._capabilities = capabilities

    @property
    def capabilities(self) -> STTCapabilities:
        return self._capabilities

    @abstractmethod
    async def recognize(
        self, buffer: AudioBuffer, *, language: str | None = None
    ) -> SpeechEvent:
        pass

    def stream(self, *, language: str | None = None) -> "SpeechStream":
        raise NotImplementedError(
            "streaming is not supported by this STT, please use a different STT or use a StreamAdapter"
        )

    async def aclose(self) -> None:
        """
        Close the STT, and every stream/requests associated with it
        """
        pass


class SpeechStream(ABC):
    class _FlushSentinel:
        pass

    def __init__(self):
        self._input_ch = aio.Chan[Union[rtc.AudioFrame, SpeechStream._FlushSentinel]]()
        self._event_ch = aio.Chan[SpeechEvent]()
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @abstractmethod
    def _main_task(self) -> None: ...

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push audio to be recognized"""
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(frame)

    def flush(self) -> None:
        """Mark the end of the current segment"""
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        """Mark the end of input, no more text will be pushed"""
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        """Close ths stream immediately"""
        self._input_ch.close()
        await aio.gracefully_cancel(self._task)
        self._event_ch.close()

    async def __anext__(self) -> SpeechEvent:
        return await self._event_ch.__anext__()

    def __aiter__(self) -> AsyncIterator[SpeechEvent]:
        return self

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self._input_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")
