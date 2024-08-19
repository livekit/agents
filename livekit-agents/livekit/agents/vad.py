import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import AsyncIterator, List, Union

from livekit import rtc

from .utils import aio


@unique
class VADEventType(str, Enum):
    START_OF_SPEECH = "start_of_speech"
    INFERENCE_DONE = "inference_done"
    END_OF_SPEECH = "end_of_speech"


@dataclass
class VADEvent:
    type: VADEventType
    """type of the event"""
    samples_index: int
    """index of the samples when the event was fired"""
    speech_duration: float
    """duration of the speech in seconds"""
    silence_duration: float
    """duration of the silence in seconds"""
    frames: List[rtc.AudioFrame] = field(default_factory=list)
    """list of audio frames of the speech

    start_of_speech: contains the complete audio chunks that triggered the detection)
    end_of_speech: contains the complete user speech
    """
    probability: float = 0.0
    """smoothed probability of the speech (only for INFERENCE_DONE event)"""
    inference_duration: float = 0.0
    """duration of the inference in seconds (only for INFERENCE_DONE event)"""
    speaking: bool = False
    """whether speech was detected in the frames"""


@dataclass
class VADCapabilities:
    update_interval: float


class VAD(ABC):
    def __init__(self, *, capabilities: VADCapabilities) -> None:
        self._capabilities = capabilities

    @property
    def capabilities(self) -> VADCapabilities:
        return self._capabilities

    @abstractmethod
    def stream(self) -> "VADStream":
        pass


class VADStream(ABC):
    class _FlushSentinel:
        pass

    def __init__(self):
        self._input_ch = aio.Chan[Union[rtc.AudioFrame, VADStream._FlushSentinel]]()
        self._event_ch = aio.Chan[VADEvent]()
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @abstractmethod
    def _main_task(self) -> None: ...

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push some text to be synthesized"""
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

    async def __anext__(self) -> VADEvent:
        return await self._event_ch.__anext__()

    def __aiter__(self) -> AsyncIterator[VADEvent]:
        return self

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self._input_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")
