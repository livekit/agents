from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum, unique
from typing import Generic, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc

from ._exceptions import APIConnectionError, APIError
from .log import logger
from .types import APIConnectOptions
from .utils import aio, log_exceptions


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


class BargeinError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["bargein_error"] = "bargein_error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


TEvent = TypeVar("TEvent")


class BargeinDetector(
    ABC, rtc.EventEmitter[Union[Literal["bargein_detected", "error"], TEvent]], Generic[TEvent]
):
    def __init__(self, *, sample_rate: int) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._sample_rate = sample_rate

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "unknown"

    @property
    def label(self) -> str:
        return self._label

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            BargeinError(
                timestamp=time.time(),
                label=self._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

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

    def __init__(self, bargein_detector: BargeinDetector, conn_options: APIConnectOptions) -> None:
        self._bargein_detector = bargein_detector
        self._last_activity_time = time.perf_counter()
        self._input_ch = aio.Chan[
            Union[
                rtc.AudioFrame,
                BargeinStream._AgentSpeechStartedSentinel,
                BargeinStream._AgentSpeechEndedSentinel,
                BargeinStream._OverlapSpeechStartedSentinel,
                BargeinStream._OverlapSpeechEndedSentinel,
                BargeinStream._FlushSentinel,
            ]
        ]()
        self._event_ch = aio.Chan[BargeinEvent]()
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())
        self._num_retries = 0
        self._conn_options = conn_options
        self._sample_rate = bargein_detector._sample_rate
        self._resampler: rtc.AudioResampler | None = None

    @abstractmethod
    async def _run(self) -> None: ...

    @log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        max_retries = self._conn_options.max_retry

        while self._num_retries <= max_retries:
            try:
                return await self._run()
            except APIError as e:
                if max_retries == 0:
                    self._emit_error(e, recoverable=False)
                    raise
                elif self._num_retries == max_retries:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"failed to detect bargein after {self._num_retries} attempts",
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)

                    retry_interval = self._conn_options._interval_for_retry(self._num_retries)
                    logger.warning(
                        f"failed to detect bargein, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={
                            "bargein_detector": self._bargein_detector._label,
                            "attempt": self._num_retries,
                        },
                    )
                    await asyncio.sleep(retry_interval)

                self._num_retries += 1

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._bargein_detector.emit(
            "error",
            BargeinError(
                timestamp=time.time(),
                label=self._bargein_detector._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

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

    def push_frame(
        self,
        frame: rtc.AudioFrame
        | BargeinStream._AgentSpeechStartedSentinel
        | BargeinStream._AgentSpeechEndedSentinel
        | BargeinStream._OverlapSpeechStartedSentinel
        | BargeinStream._OverlapSpeechEndedSentinel,
    ) -> None:
        """Push some audio frame to be analyzed"""
        self._check_input_not_ended()
        self._check_not_closed()

        if not isinstance(frame, rtc.AudioFrame):
            self._input_ch.send_nowait(frame)
            return

        if self._sample_rate != frame.sample_rate:
            if not self._resampler:
                self._resampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=self._sample_rate,
                    num_channels=1,
                    quality=rtc.AudioResamplerQuality.LOW,
                )
            elif self._resampler._input_rate != frame.sample_rate:
                raise ValueError("the sample rate of the input frames must be consistent")

        if self._resampler:
            frames = self._resampler.push(frame)
            for frame in frames:
                self._input_ch.send_nowait(frame)
        else:
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
