"""
Text-to-Speech (TTS) abstraction layer for voice synthesis.

Provides unified interfaces for:
- Streaming and non-streaming synthesis
- Metrics collection
- Error handling and retries
- Audio frame management
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import TracebackType
from typing import AsyncIterable, AsyncIterator, Generic, Literal, TypeVar, Union

from livekit import rtc

from .._exceptions import APIConnectionError, APIError
from ..log import logger
from ..metrics import TTSMetrics
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from ..utils import aio


@dataclass
class SynthesizedAudio:
    """Container for synthesized audio data and metadata."""
    frame: rtc.AudioFrame
    """Synthesized audio frame"""
    request_id: str
    """Request ID (one segment could be made up of multiple requests)"""
    is_final: bool = False
    """Whether this is latest frame of the segment (streaming only)"""
    segment_id: str = ""
    """Segment ID, each segment is separated by a flush (streaming only)"""
    delta_text: str = ""
    """Current segment of the synthesized audio (streaming only)"""


@dataclass
class TTSCapabilities:
    """Describes capabilities of a TTS implementation."""
    streaming: bool
    """Whether this TTS supports streaming (generally using websockets)"""


TEvent = TypeVar("TEvent")


class TTS(
    ABC,
    rtc.EventEmitter[Union[Literal["metrics_collected"], TEvent]],
    Generic[TEvent],
):
    """Abstract base class for Text-to-Speech implementations.
    
    Handles both streaming and non-streaming synthesis with:
    - Automatic retries
    - Metrics collection
    - Resource management
    """

    def __init__(
        self, *, capabilities: TTSCapabilities, sample_rate: int, num_channels: int
    ) -> None:
        """
        Args:
            capabilities: Supported features of the TTS implementation
            sample_rate: Output audio sample rate in Hz
            num_channels: Number of audio channels in output
        """
        super().__init__()
        self._capabilities = capabilities
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def label(self) -> str:
        """Identifier for metrics and logging."""
        return self._label

    @property
    def capabilities(self) -> TTSCapabilities:
        """Supported features of this TTS implementation."""
        return self._capabilities

    @property
    def sample_rate(self) -> int:
        """Output audio sample rate in Hz."""
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        """Number of audio channels in output."""
        return self._num_channels

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """Synthesize text into audio with chunked response handling.
        
        Args:
            text: Input text to synthesize
            conn_options: Connection and retry configuration
            
        Returns:
            ChunkedStream: Async stream of audio frames
        """
        ...

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create a streaming synthesis session.
        
        Args:
            conn_options: Connection and retry configuration
            
        Returns:
            SynthesizeStream: Interactive synthesis stream
            
        Raises:
            NotImplementedError: If streaming not supported
        """
        raise NotImplementedError(
            "streaming is not supported by this TTS, please use a different TTS or use a StreamAdapter"
        )

    async def aclose(self) -> None:
        """Clean up any resources used by the TTS."""
        ...

    async def __aenter__(self) -> TTS:
        """Context manager entry point."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit with resource cleanup."""
        await self.aclose()


class ChunkedStream(ABC):
    """Base class for handling chunked HTTP responses from TTS services.
    
    Manages:
    - Async iteration over audio chunks
    - Error handling and retries
    - Metrics collection
    """

    def __init__(
        self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        """
        Args:
            tts: Parent TTS instance
            input_text: Text being synthesized
            conn_options: Connection and retry configuration
        """
        self._input_text = input_text
        self._tts = tts
        self._conn_options = conn_options
        self._event_ch = aio.Chan[SynthesizedAudio]()

        # Split stream for metrics monitoring
        self._event_aiter, monitor_aiter = aio.itertools.tee(self._event_ch, 2)
        self._metrics_task = asyncio.create_task(
            self._metrics_monitor_task(monitor_aiter), name="TTS._metrics_task"
        )
        self._synthesize_task = asyncio.create_task(
            self._main_task(), name="TTS._synthesize_task"
        )
        self._synthesize_task.add_done_callback(lambda _: self._event_ch.close())

    @property
    def input_text(self) -> str:
        """Original text being synthesized."""
        return self._input_text

    @property
    def done(self) -> bool:
        """Check if synthesis is complete."""
        return self._synthesize_task.done()

    @property
    def exception(self) -> BaseException | None:
        """Get exception if synthesis failed."""
        return self._synthesize_task.exception()

    async def _metrics_monitor_task(
        self, event_aiter: AsyncIterable[SynthesizedAudio]
    ) -> None:
        """Background task for collecting performance metrics."""
        start_time = time.perf_counter()
        audio_duration = 0.0
        ttfb = -1.0  # Time to first byte
        request_id = ""

        async for ev in event_aiter:
            request_id = ev.request_id
            if ttfb == -1.0:
                ttfb = time.perf_counter() - start_time

            audio_duration += ev.frame.duration

        # Final metrics emission
        duration = time.perf_counter() - start_time
        metrics = TTSMetrics(
            timestamp=time.time(),
            request_id=request_id,
            ttfb=ttfb,
            duration=duration,
            characters_count=len(self._input_text),
            audio_duration=audio_duration,
            cancelled=self._synthesize_task.cancelled(),
            label=self._tts._label,
            streamed=False,
            error=None,
        )
        self._tts.emit("metrics_collected", metrics)

    async def collect(self) -> rtc.AudioFrame:
        """Combine all audio chunks into a single frame.
        
        Returns:
            rtc.AudioFrame: Combined audio data
        """
        frames = []
        async for ev in self:
            frames.append(ev.frame)

        return rtc.combine_audio_frames(frames)

    @abstractmethod
    async def _run(self) -> None:
        """Implementation-specific synthesis logic."""
        ...

    async def _main_task(self) -> None:
        """Main synthesis loop with retry handling."""
        for i in range(self._conn_options.max_retry + 1):
            try:
                return await self._run()
            except APIError as e:
                if self._conn_options.max_retry == 0:
                    raise
                elif i == self._conn_options.max_retry:
                    raise APIConnectionError(
                        f"failed to synthesize speech after {self._conn_options.max_retry + 1} attempts",
                    ) from e
                else:
                    logger.warning(
                        f"failed to synthesize speech, retrying in {self._conn_options.retry_interval}s",
                        exc_info=e,
                        extra={
                            "tts": self._tts._label,
                            "attempt": i + 1,
                            "streamed": False,
                        },
                    )

                await asyncio.sleep(self._conn_options.retry_interval)

    async def aclose(self) -> None:
        """Clean up resources and stop synthesis."""
        await aio.cancel_and_wait(self._synthesize_task)
        self._event_ch.close()
        await self._metrics_task

    async def __anext__(self) -> SynthesizedAudio:
        """Get next audio chunk."""
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if not self._synthesize_task.cancelled() and (
                exc := self._synthesize_task.exception()
            ):
                raise exc from None

            raise StopAsyncIteration

        return val

    def __aiter__(self) -> AsyncIterator[SynthesizedAudio]:
        """Support async iteration over audio chunks."""
        return self

    async def __aenter__(self) -> ChunkedStream:
        """Context manager entry point."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit with cleanup."""
        await self.aclose()


class SynthesizeStream(ABC):
    """Base class for interactive streaming synthesis sessions.
    
    Features:
    - Incremental text input
    - Real-time audio output
    - Segment management
    - Metrics tracking
    """

    class _FlushSentinel:
        """Internal marker for segment boundaries."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        """
        Args:
            tts: Parent TTS instance
            conn_options: Connection and retry configuration
        """
        super().__init__()
        self._tts = tts
        self._conn_options = conn_options
        self._input_ch = aio.Chan[Union[str, SynthesizeStream._FlushSentinel]]()
        self._event_ch = aio.Chan[SynthesizedAudio]()
        self._event_aiter, self._monitor_aiter = aio.itertools.tee(self._event_ch, 2)

        self._task = asyncio.create_task(self._main_task(), name="TTS._main_task")
        self._task.add_done_callback(lambda _: self._event_ch.close())
        self._metrics_task: asyncio.Task | None = None  # Lazy initialization

        # Metrics tracking state
        self._mtc_pending_texts: list[str] = []
        self._mtc_text = ""

    @abstractmethod
    async def _run(self) -> None:
        """Implementation-specific streaming logic."""
        ...

    async def _main_task(self) -> None:
        """Main streaming loop with retry handling."""
        for i in range(self._conn_options.max_retry + 1):
            try:
                return await self._run()
            except APIError as e:
                if self._conn_options.max_retry == 0:
                    raise
                elif i == self._conn_options.max_retry:
                    raise APIConnectionError(
                        f"failed to synthesize speech after {self._conn_options.max_retry + 1} attempts",
                    ) from e
                else:
                    logger.warning(
                        f"failed to synthesize speech, retrying in {self._conn_options.retry_interval}s",
                        exc_info=e,
                        extra={
                            "tts": self._tts._label,
                            "attempt": i + 1,
                            "streamed": True,
                        },
                    )

                await asyncio.sleep(self._conn_options.retry_interval)

    async def _metrics_monitor_task(
        self, event_aiter: AsyncIterable[SynthesizedAudio]
    ) -> None:
        """Background task for streaming metrics collection."""
        start_time = time.perf_counter()
        audio_duration = 0.0
        ttfb = -1.0  # Time to first byte
        request_id = ""

        def _emit_metrics():
            """Emit metrics for completed segments."""
            nonlocal start_time, audio_duration, ttfb, request_id
            duration = time.perf_counter() - start_time

            if not self._mtc_pending_texts:
                return

            text = self._mtc_pending_texts.pop(0)
            if not text:
                return

            metrics = TTSMetrics(
                timestamp=time.time(),
                request_id=request_id,
                ttfb=ttfb,
                duration=duration,
                characters_count=len(text),
                audio_duration=audio_duration,
                cancelled=self._task.cancelled(),
                label=self._tts._label,
                streamed=True,
                error=None,
            )
            self._tts.emit("metrics_collected", metrics)

            # Reset tracking for next segment
            audio_duration = 0.0
            ttfb = -1.0
            request_id = ""
            start_time = time.perf_counter()

        async for ev in event_aiter:
            if ttfb == -1.0:
                ttfb = time.perf_counter() - start_time

            audio_duration += ev.frame.duration
            request_id = ev.request_id

            if ev.is_final:
                _emit_metrics()

        # Emit final metrics if any remaining data
        if request_id:
            _emit_metrics()

    def push_text(self, token: str) -> None:
        """Add text to the current synthesis segment.
        
        Args:
            token: Text to synthesize
            
        Raises:
            RuntimeError: If stream is closed or input ended
        """
        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(
                self._metrics_monitor_task(self._monitor_aiter),
                name="TTS._metrics_task",
            )

        self._mtc_text += token
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(token)

    def flush(self) -> None:
        """Finalize current synthesis segment."""
        if self._mtc_text:
            self._mtc_pending_texts.append(self._mtc_text)
            self._mtc_text = ""

        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        """Signal end of input data."""
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        """Immediately terminate the stream."""
        self._input_ch.close()
        await aio.cancel_and_wait(self._task)

        if self._metrics_task is not None:
            await self._metrics_task

    def _check_not_closed(self) -> None:
        """Verify stream is still active."""
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        """Verify input is still accepted."""
        if self._input_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")

    async def __anext__(self) -> SynthesizedAudio:
        """Get next audio chunk from stream."""
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc from None

            raise StopAsyncIteration

        return val

    def __aiter__(self) -> AsyncIterator[SynthesizedAudio]:
        """Support async iteration over audio chunks."""
        return self

    async def __aenter__(self) -> SynthesizeStream:
        """Context manager entry point."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit with cleanup."""
        await self.aclose()
