from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum, unique
from typing import Literal, Union

from livekit import rtc

from .utils import aio


@unique
class BargeinEventType(str, Enum):
    INFERENCE_DONE = "inference_done"
    BARGEIN = "bargein"


@dataclass
class BargeinEvent:
    """
    Represents an event detected by the Bargein detection model.
    """

    type: BargeinEventType
    """Type of the bargein event (e.g., inference done, bargein)."""

    timestamp: float
    """Timestamp (in seconds) when the event was fired."""

    is_bargein: bool = False
    """Whether bargein is detected (only for `INFERENCE_DONE` events)."""

    inference_duration: float = 0.0
    """Time taken to perform the inference, in seconds."""


class BargeinDetector(ABC, rtc.EventEmitter[Literal["bargein_detected"]]):
    def __init__(self) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "unknown"

    @property
    def label(self) -> str:
        return self._label

    @abstractmethod
    def stream(self) -> BargeinStream: ...


class BargeinStream(ABC):
    class _AgentSpeechStartedSentinel:
        pass

    class _AgentSpeechEndedSentinel:
        pass

    class _OverlapSpeechStartedSentinel:
        pass

    class _OverlapSpeechEndedSentinel:
        pass

    class _FlushSentinel:
        pass

    def __init__(self, bargein_detector: BargeinDetector) -> None:
        self._bargein_detector = bargein_detector
        self._last_activity_time = time.perf_counter()
        self._input_ch = aio.Chan[
            Union[
                rtc.AudioFrame,
                BargeinStream._OverlapSpeechStartedSentinel,
                BargeinStream._OverlapSpeechEndedSentinel,
                BargeinStream._FlushSentinel,
            ]
        ]()
        self._event_ch = aio.Chan[BargeinEvent]()
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @abstractmethod
    async def _main_task(self) -> None: ...

    def start_agent_speech(self) -> None:
        """Mark the start of the agent's speech"""
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(self._AgentSpeechStartedSentinel())

    def end_agent_speech(self) -> None:
        """Mark the end of the agent's speech"""
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(self._AgentSpeechEndedSentinel())

    def start_overlap_speech(self) -> None:
        """Mark the start of the overlap speech"""
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(self._OverlapSpeechStartedSentinel())

    def end_overlap_speech(self) -> None:
        """Mark the end of the overlap speech"""
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(self._OverlapSpeechEndedSentinel())

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push some audio frame to be analyzed"""
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(frame)

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
        """Close the stream immediately"""
        self._input_ch.close()
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()

    async def __anext__(self) -> BargeinEvent:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904

            raise StopAsyncIteration from None

        return val

    def __aiter__(self) -> AsyncIterator[BargeinEvent]:
        return self

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self._input_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")
