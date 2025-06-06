from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, unique
from types import TracebackType
from typing import Generic, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc

from .._exceptions import APIConnectionError, APIError
from ..log import logger
from ..metrics import STTMetrics
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import AudioBuffer, aio, is_given
from ..utils.audio import calculate_audio_duration


@unique
class SpeechEventType(str, Enum):
    START_OF_SPEECH = "start_of_speech"
    """indicate the start of speech
    if the STT doesn't support this event, this will be emitted as the same time as the first INTERIM_TRANSCRIPT"""  # noqa: E501
    INTERIM_TRANSCRIPT = "interim_transcript"
    """interim transcript, useful for real-time transcription"""
    FINAL_TRANSCRIPT = "final_transcript"
    """final transcript, emitted when the STT is confident enough that a certain
    portion of speech will not change"""
    RECOGNITION_USAGE = "recognition_usage"
    """usage event, emitted periodically to indicate usage metrics"""
    END_OF_SPEECH = "end_of_speech"
    """indicate the end of speech, emitted when the user stops speaking"""


@dataclass
class SpeechData:
    language: str
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0  # [0, 1]
    speaker_id: str | None = None


@dataclass
class RecognitionUsage:
    audio_duration: float


@dataclass
class SpeechEvent:
    type: SpeechEventType
    request_id: str = ""
    alternatives: list[SpeechData] = field(default_factory=list)
    recognition_usage: RecognitionUsage | None = None


@dataclass
class STTCapabilities:
    streaming: bool
    interim_results: bool


class STTError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["stt_error"] = "stt_error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


TEvent = TypeVar("TEvent")


class STT(
    ABC,
    rtc.EventEmitter[Union[Literal["metrics_collected", "error"], TEvent]],
    Generic[TEvent],
):
    def __init__(self, *, capabilities: STTCapabilities) -> None:
        super().__init__()
        self._capabilities = capabilities
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def label(self) -> str:
        return self._label

    @property
    def capabilities(self) -> STTCapabilities:
        return self._capabilities

    @abstractmethod
    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent: ...

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        for i in range(conn_options.max_retry + 1):
            try:
                start_time = time.perf_counter()
                event = await self._recognize_impl(
                    buffer, language=language, conn_options=conn_options
                )
                duration = time.perf_counter() - start_time
                stt_metrics = STTMetrics(
                    request_id=event.request_id,
                    timestamp=time.time(),
                    duration=duration,
                    label=self._label,
                    audio_duration=calculate_audio_duration(buffer),
                    streamed=False,
                )
                self.emit("metrics_collected", stt_metrics)
                return event

            except APIError as e:
                retry_interval = conn_options._interval_for_retry(i)
                if conn_options.max_retry == 0:
                    self._emit_error(e, recoverable=False)
                    raise
                elif i == conn_options.max_retry:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"failed to recognize speech after {conn_options.max_retry + 1} attempts",
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)
                    logger.warning(
                        f"failed to recognize speech, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={
                            "tts": self._label,
                            "attempt": i + 1,
                            "streamed": False,
                        },
                    )

                await asyncio.sleep(retry_interval)

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

        raise RuntimeError("unreachable")

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            STTError(
                timestamp=time.time(),
                label=self._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        raise NotImplementedError(
            "streaming is not supported by this STT, please use a different STT or use a StreamAdapter"  # noqa: E501
        )

    async def aclose(self) -> None:
        """Close the STT, and every stream/requests associated with it"""
        ...

    async def __aenter__(self) -> STT:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def prewarm(self) -> None:
        """Pre-warm connection to the STT service"""
        pass


class RecognizeStream(ABC):
    class _FlushSentinel:
        """Sentinel to mark when it was flushed"""

        pass

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ):
        """
        Args:
        sample_rate : int or None, optional
            The desired sample rate for the audio input.
            If specified, the audio input will be automatically resampled to match
            the given sample rate before being processed for Speech-to-Text.
            If not provided (None), the input will retain its original sample rate.
        """
        self._stt = stt
        self._conn_options = conn_options
        self._input_ch = aio.Chan[Union[rtc.AudioFrame, RecognizeStream._FlushSentinel]]()
        self._event_ch = aio.Chan[SpeechEvent]()

        self._event_aiter, monitor_aiter = aio.itertools.tee(self._event_ch, 2)
        self._metrics_task = asyncio.create_task(
            self._metrics_monitor_task(monitor_aiter), name="STT._metrics_task"
        )

        self._num_retries = 0
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

        self._needed_sr = sample_rate if is_given(sample_rate) else None
        self._pushed_sr = 0
        self._resampler: rtc.AudioResampler | None = None

    @abstractmethod
    async def _run(self) -> None: ...

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
                        f"failed to recognize speech after {self._num_retries} attempts",
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)

                    retry_interval = self._conn_options._interval_for_retry(self._num_retries)
                    logger.warning(
                        f"failed to recognize speech, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={
                            "tts": self._stt._label,
                            "attempt": self._num_retries,
                            "streamed": True,
                        },
                    )
                    await asyncio.sleep(retry_interval)

                self._num_retries += 1

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._stt.emit(
            "error",
            STTError(
                timestamp=time.time(),
                label=self._stt._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SpeechEvent]) -> None:
        """Task used to collect metrics"""

        async for ev in event_aiter:
            if ev.type == SpeechEventType.RECOGNITION_USAGE:
                assert ev.recognition_usage is not None, (
                    "recognition_usage must be provided for RECOGNITION_USAGE event"
                )

                stt_metrics = STTMetrics(
                    request_id=ev.request_id,
                    timestamp=time.time(),
                    duration=0.0,
                    label=self._stt._label,
                    audio_duration=ev.recognition_usage.audio_duration,
                    streamed=True,
                )

                self._stt.emit("metrics_collected", stt_metrics)
            elif ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                # reset the retry count after a successful recognition
                self._num_retries = 0

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push audio to be recognized"""
        self._check_input_not_ended()
        self._check_not_closed()

        if self._pushed_sr and self._pushed_sr != frame.sample_rate:
            raise ValueError("the sample rate of the input frames must be consistent")

        self._pushed_sr = frame.sample_rate

        if self._needed_sr and self._needed_sr != frame.sample_rate:
            if not self._resampler:
                self._resampler = rtc.AudioResampler(
                    frame.sample_rate,
                    self._needed_sr,
                    quality=rtc.AudioResamplerQuality.HIGH,
                )

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

        if self._resampler:
            for frame in self._resampler.flush():
                self._input_ch.send_nowait(frame)

        self._input_ch.send_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        """Mark the end of input, no more audio will be pushed"""
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        """Close ths stream immediately"""
        self._input_ch.close()
        await aio.cancel_and_wait(self._task)

        if self._metrics_task is not None:
            await self._metrics_task

    async def __anext__(self) -> SpeechEvent:
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904

            raise StopAsyncIteration from None

        return val

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

    async def __aenter__(self) -> RecognizeStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


SpeechStream = RecognizeStream  # deprecated alias
