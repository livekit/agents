"""
Speech-to-Text (STT) abstraction layer for voice recognition.

Provides interfaces for:
- Real-time audio transcription
- Streaming and batch processing
- Speech event handling
- Metrics collection
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
from types import TracebackType
from typing import AsyncIterable, AsyncIterator, Generic, List, Literal, TypeVar, Union

from livekit import rtc

from .._exceptions import APIConnectionError, APIError
from ..log import logger
from ..metrics import STTMetrics
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from ..utils import AudioBuffer, aio
from ..utils.audio import calculate_audio_duration


@unique
class SpeechEventType(str, Enum):
    """Events emitted during speech recognition."""
    START_OF_SPEECH = "start_of_speech"
    """Indicates speech detection beginning (may coincide with first interim transcript)
    if the STT doesn't support this event, this will be emitted as the same time as the first INTERIM_TRANSCRIPT"""
    INTERIM_TRANSCRIPT = "interim_transcript"
    """Partial transcription with low latency, may be updated"""
    FINAL_TRANSCRIPT = "final_transcript"
    """Stable transcription unlikely to change"""
    RECOGNITION_USAGE = "recognition_usage"
    """Periodic usage metrics update"""
    END_OF_SPEECH = "end_of_speech"
    """Signals speech has stopped"""


@dataclass
class SpeechData:
    """Recognized speech segment with timing and confidence."""
    language: str            # Detected language code
    text: str                # Transcribed text
    start_time: float = 0.0  # Audio start offset in seconds
    end_time: float = 0.0    # Audio end offset in seconds
    confidence: float = 0.0  # Recognition confidence [0-1]


@dataclass
class RecognitionUsage:
    """Resource usage metrics for recognition."""
    audio_duration: float    # Total processed audio in seconds


@dataclass
class SpeechEvent:
    """Container for STT events and associated data."""
    type: SpeechEventType         # Event category
    request_id: str = ""          # Unique identifier for recognition request
    alternatives: List[SpeechData] = field(default_factory=list)  # Possible transcriptions
    recognition_usage: RecognitionUsage | None = None  # Usage data for RECOGNITION_USAGE events


@dataclass
class STTCapabilities:
    """Describes features supported by the STT implementation."""
    streaming: bool         # Supports real-time audio streaming
    interim_results: bool   # Provides partial results during recognition


TEvent = TypeVar("TEvent")


class STT(
    ABC,
    rtc.EventEmitter[Union[Literal["metrics_collected"], TEvent]],
    Generic[TEvent],
):
    """Abstract base class for Speech-to-Text implementations."""
    
    def __init__(self, *, capabilities: STTCapabilities) -> None:
        """
        Args:
            capabilities: Supported features of this STT implementation
        """
        super().__init__()
        self._capabilities = capabilities
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def label(self) -> str:
        """Identifier for metrics and logging."""
        return self._label

    @property
    def capabilities(self) -> STTCapabilities:
        """Supported features of this STT engine."""
        return self._capabilities

    @abstractmethod
    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent: 
        """Implementation-specific recognition logic (to be overridden)"""

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        """Process audio buffer and return transcription.
        
        Args:
            buffer: Audio data to transcribe
            language: Optional language hint
            conn_options: Connection/retry configuration
            
        Returns:
            SpeechEvent with recognition results
            
        Raises:
            APIConnectionError: After repeated failures
        """
        for i in range(conn_options.max_retry + 1):
            try:
                start_time = time.perf_counter()
                event = await self._recognize_impl(
                    buffer, language=language, conn_options=conn_options
                )
                duration = time.perf_counter() - start_time
                
                # Emit performance metrics
                stt_metrics = STTMetrics(
                    request_id=event.request_id,
                    timestamp=time.time(),
                    duration=duration,
                    label=self._label,
                    audio_duration=calculate_audio_duration(buffer),
                    streamed=False,
                    error=None,
                )
                self.emit("metrics_collected", stt_metrics)
                return event

            except APIError as e:
                if conn_options.max_retry == 0:
                    raise
                elif i == conn_options.max_retry:
                    raise APIConnectionError(
                        f"failed to recognize speech after {conn_options.max_retry + 1} attempts",
                    ) from e
                else:
                    logger.warning(
                        f"failed to recognize speech, retrying in {conn_options.retry_interval}s",
                        exc_info=e,
                        extra={
                            "tts": self._label,
                            "attempt": i + 1,
                            "streamed": False,
                        },
                    )

                await asyncio.sleep(conn_options.retry_interval)

        raise RuntimeError("unreachable")

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "RecognizeStream":
        """Create streaming recognition session.
        
        Raises:
            NotImplementedError: If streaming not supported
        """
        raise NotImplementedError(
            "streaming is not supported by this STT, please use a different STT or use a StreamAdapter"
        )

    async def aclose(self) -> None:
        """Release resources and terminate ongoing operations."""
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


class RecognizeStream(ABC):
    """Base class for streaming speech recognition sessions."""
    
    class _FlushSentinel:
        """Internal marker for buffer flush operations"""

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        sample_rate: int | None = None,
    ):
        """
        Args:
            stt: Parent STT instance
            conn_options: Connection/retry configuration
            sample_rate: Target input sample rate (enables auto-resampling)
        """
        self._stt = stt
        self._conn_options = conn_options
        self._input_ch = aio.Chan[Union[rtc.AudioFrame, RecognizeStream._FlushSentinel]]()
        self._event_ch = aio.Chan[SpeechEvent]()

        # Create parallel iterators for event processing
        self._event_aiter, monitor_aiter = aio.itertools.tee(self._event_ch, 2)
        
        # Start metrics collection task
        self._metrics_task = asyncio.create_task(
            self._metrics_monitor_task(monitor_aiter), name="STT._metrics_task"
        )

        # Main processing task
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

        # Audio processing state
        self._needed_sr = sample_rate
        self._pushed_sr = 0
        self._resampler: rtc.AudioResampler | None = None

    @abstractmethod
    async def _run(self) -> None:
        """Stream processing implementation (to be overridden)"""

    async def _main_task(self) -> None:
        """Wrapper task with retry logic"""
        for i in range(self._conn_options.max_retry + 1):
            try:
                return await self._run()
            except APIError as e:
                if self._conn_options.max_retry == 0:
                    raise
                elif i == self._conn_options.max_retry:
                    raise APIConnectionError(
                        f"failed to recognize speech after {self._conn_options.max_retry + 1} attempts",
                    ) from e
                else:
                    logger.warning(
                        f"failed to recognize speech, retrying in {self._conn_options.retry_interval}s",
                        exc_info=e,
                        extra={
                            "tts": self._stt._label,
                            "attempt": i + 1,
                            "streamed": True,
                        },
                    )

                await asyncio.sleep(self._conn_options.retry_interval)

    async def _metrics_monitor_task(
        self, event_aiter: AsyncIterable[SpeechEvent]
    ) -> None:
        """Collect and report recognition metrics"""
        start_time = time.perf_counter()

        async for ev in event_aiter:
            if ev.type == SpeechEventType.RECOGNITION_USAGE:
                assert ev.recognition_usage is not None, (
                    "recognition_usage must be provided for RECOGNITION_USAGE event"
                )
              
                duration = time.perf_counter() - start_time
                stt_metrics = STTMetrics(
                    request_id=ev.request_id,
                    timestamp=time.time(),
                    duration=duration,
                    label=self._stt._label,
                    audio_duration=ev.recognition_usage.audio_duration,
                    streamed=True,
                    error=None,
                )

                self._stt.emit("metrics_collected", stt_metrics)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Submit audio frame for processing.
        
        Args:
            frame: Audio data to process
            
        Raises:
            ValueError: On sample rate inconsistency
            RuntimeError: If stream closed
        """
        self._check_input_not_ended()
        self._check_not_closed()

        # Validate sample rate consistency
        if self._pushed_sr and self._pushed_sr != frame.sample_rate:
            raise ValueError("the sample rate of the input frames must be consistent")

        self._pushed_sr = frame.sample_rate

        # Initialize resampler if needed
        if self._needed_sr and self._needed_sr != frame.sample_rate:
            if not self._resampler:
                self._resampler = rtc.AudioResampler(
                    frame.sample_rate,
                    self._needed_sr,
                    quality=rtc.AudioResamplerQuality.HIGH,
                )

        # Process frame through resampler
        if self._resampler:
            for frame in self._resampler.push(frame):
                self._input_ch.send_nowait(frame)
        else:
            self._input_ch.send_nowait(frame)

    def flush(self) -> None:
        """Finalize current audio segment processing"""
        self._check_input_not_ended()
        self._check_not_closed()

        # Process remaining resampler data
        if self._resampler:
            for frame in self._resampler.flush():
                self._input_ch.send_nowait(frame)

        self._input_ch.send_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        """Signal end of audio input stream"""
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        """Immediately terminate recognition stream"""
        self._input_ch.close()
        await aio.cancel_and_wait(self._task)

        if self._metrics_task is not None:
            await self._metrics_task

    async def __anext__(self) -> SpeechEvent:
        """Get next speech event from stream"""
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc from None

            raise StopAsyncIteration

        return val

    def __aiter__(self) -> AsyncIterator[SpeechEvent]:
        """Support async iteration over speech events"""
        return self

    def _check_not_closed(self) -> None:
        """Validate stream is still active"""
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        """Verify input is still being accepted"""
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


SpeechStream = RecognizeStream  # Backwards compatibility alias
