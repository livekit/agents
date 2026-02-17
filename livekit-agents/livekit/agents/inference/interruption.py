from __future__ import annotations

import asyncio
import json
import math
import os
import struct
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter_ns
from typing import Annotated, Any, Literal, TypeAlias

import aiohttp
import numpy as np
import numpy.typing as npt
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from livekit import rtc

from .._exceptions import APIConnectionError, APIError, APIStatusError
from ..log import logger
from ..telemetry import trace_types
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import (
    AudioArrayBuffer,
    BoundedDict,
    aio,
    http_context,
    is_given,
    log_exceptions,
    shortuuid,
)
from ._utils import (
    DEFAULT_INFERENCE_URL,
    STAGING_INFERENCE_URL,
    create_access_token,
    get_default_inference_url,
)

SAMPLE_RATE = 16000
THRESHOLD = 0.5
MIN_INTERRUPTION_DURATION = 0.025 * 2  # 25ms per frame, 2 consecutive frames
MAX_AUDIO_DURATION = 3  # 3 seconds
DETECTION_INTERVAL = 0.1  # 0.1 second
AUDIO_PREFIX_DURATION = 1.0  # 1.0 second
REMOTE_INFERENCE_TIMEOUT = 1
_FRAMES_PER_SECOND = 40


class InterruptionDetectionError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["interruption_detection_error"] = "interruption_detection_error"
    timestamp: float = Field(default_factory=time.time)
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


@dataclass(slots=True, kw_only=True)
class InterruptionOptions:
    sample_rate: int
    """The sample rate of the audio frames, defaults to 16000Hz"""
    threshold: float
    """The threshold for the interruption detection, defaults to 0.5"""
    min_frames: int
    """The minimum number of frames to detect a interruption, defaults to 50ms/2 frames"""
    max_audio_duration: float
    """The maximum audio duration for the interruption detection, including the audio prefix, defaults to 3 seconds"""
    audio_prefix_duration: float
    """The audio prefix duration for the interruption detection, defaults to 1.0 seconds"""
    detection_interval: float
    """The interval between detections, defaults to 0.1 seconds"""
    inference_timeout: float
    """The timeout for the interruption detection, defaults to 1 second"""
    base_url: str
    api_key: str
    api_secret: str
    use_proxy: bool
    """Whether to use the inference instead of the hosted API"""


@dataclass(slots=True, kw_only=True)
class InterruptionCacheEntry:
    """Typed cache entry for interruption inference results."""

    created_at: int = field(default_factory=time.perf_counter_ns)
    """The timestamp when the cache entry was created, in nanoseconds. Used only for indexing and latency calculation."""
    speech_input: npt.NDArray[np.int16] | None = None
    total_duration: float | None = None
    prediction_duration: float | None = None
    detection_delay: float | None = None
    probabilities: npt.NDArray[np.float32] | None = None
    is_interruption: bool | None = None

    def get_total_duration(self, default: float = 0.0) -> float:
        """RTT (Round Trip Time) time taken to perform the inference, in seconds."""
        return self.total_duration if self.total_duration is not None else default

    def get_prediction_duration(self, default: float = 0.0) -> float:
        """Time taken to perform the inference from the model side, in seconds."""
        return self.prediction_duration if self.prediction_duration is not None else default

    def get_detection_delay(self, default: float = 0.0) -> float:
        """Total time from the onset of the speech to the final prediction, in seconds."""
        return self.detection_delay if self.detection_delay is not None else default

    def get_probability(self, default: float = 0.0) -> float:
        """The conservative estimated probability of the interruption event."""
        return (
            _estimate_probability(self.probabilities) if self.probabilities is not None else default
        )


class InterruptionEvent(BaseModel):
    """
    Represents an event detected by the interruption detection model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["user_interruption_detected", "user_non_interruption_detected"]
    """Type of the interruption event (e.g., interruption or non-interruption)."""

    timestamp: float
    """Timestamp (in seconds) when the event was fired."""

    is_interruption: bool = False
    """Whether interruption is detected."""

    total_duration: float = 0.0
    """RTT (Round Trip Time) time taken to perform the inference, in seconds."""

    prediction_duration: float = 0.0
    """Time taken to perform the inference from the model side, in seconds."""

    detection_delay: float = 0.0
    """Total time from the onset of the speech to the final prediction, in seconds."""

    overlap_speech_started_at: float | None = None
    """Timestamp (in seconds) when the overlap speech started. Useful for emitting held transcripts."""

    speech_input: npt.NDArray[np.int16] | None = None
    """The audio input that was used for the inference."""

    probabilities: npt.NDArray[np.float32] | None = None
    """The raw probabilities for the interruption detection."""

    probability: float = 0.0
    """The conservative estimated probability of the interruption event."""

    @classmethod
    def from_cache_entry(
        cls,
        *,
        entry: InterruptionCacheEntry,
        is_interruption: bool,
        started_at: float | None = None,
        ended_at: float | None = None,
    ) -> InterruptionEvent:
        """Initialize the event from a cache entry.

        Args:
            entry: The cache entry to initialize the event from.
            is_interruption: Whether the interruption is detected.
            started_at: The timestamp when the overlap speech started.
            ended_at: The timestamp when the overlap speech ended.

        Returns:
            The initialized event.
        """
        return cls(
            type="user_interruption_detected"
            if is_interruption
            else "user_non_interruption_detected",
            timestamp=ended_at or time.time(),
            is_interruption=is_interruption,
            overlap_speech_started_at=started_at,
            speech_input=entry.speech_input,
            probabilities=entry.probabilities,
            total_duration=entry.get_total_duration(),
            detection_delay=entry.get_detection_delay(),
            prediction_duration=entry.get_prediction_duration(),
            probability=entry.get_probability(),
        )


# Default empty entry used when cache misses occur
_EMPTY_CACHE_ENTRY = InterruptionCacheEntry(created_at=0)


# region: Sentinel classes
class _AgentSpeechStartedSentinel:
    pass


class _AgentSpeechEndedSentinel:
    pass


class _OverlapSpeechStartedSentinel:
    def __init__(
        self,
        speech_duration: float,
        started_at: float,
        user_speaking_span: trace.Span | None = None,
    ) -> None:
        self._speech_duration = speech_duration
        self._user_speaking_span = user_speaking_span
        self._started_at = started_at


class _OverlapSpeechEndedSentinel:
    def __init__(self, ended_at: float) -> None:
        self._ended_at = ended_at


class _FlushSentinel:
    pass


# endregion: Sentinel classes

InterruptionDataFrameType: TypeAlias = (
    rtc.AudioFrame
    | _AgentSpeechStartedSentinel
    | _AgentSpeechEndedSentinel
    | _OverlapSpeechStartedSentinel
    | _OverlapSpeechEndedSentinel
    | _FlushSentinel
)


class AdaptiveInterruptionDetector(
    rtc.EventEmitter[
        Literal["user_interruption_detected", "user_non_interruption_detected", "error"]
    ],
):
    def __init__(
        self,
        *,
        threshold: float = THRESHOLD,
        min_interruption_duration: float = MIN_INTERRUPTION_DURATION,
        max_audio_duration: float = MAX_AUDIO_DURATION,
        audio_prefix_duration: float = AUDIO_PREFIX_DURATION,
        detection_interval: float = DETECTION_INTERVAL,
        inference_timeout: float = REMOTE_INFERENCE_TIMEOUT,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Initialize a AdaptiveInterruptionDetector instance.

        Args:
            threshold (float, optional): The threshold for the interruption detection, defaults to 0.5.
            min_interruption_duration (float, optional): The minimum duration, in seconds, of the interruption event, defaults to 50ms.
            max_audio_duration (float, optional): The maximum audio duration, including the audio prefix, in seconds, for the interruption detection, defaults to 3s.
            audio_prefix_duration (float, optional): The audio prefix duration, in seconds, for the interruption detection, defaults to 0.5s.
            detection_interval (float, optional): The interval between detections, in seconds, for the interruption detection, defaults to 0.1s.
            inference_timeout (float, optional): The timeout for the interruption detection, defaults to 1 second.
            base_url (str, optional): The base URL for the interruption detection, defaults to the shared LIVEKIT_REMOTE_EOT_URL environment variable.
            api_key (str, optional): The API key for the interruption detection, defaults to the LIVEKIT_INFERENCE_API_KEY environment variable.
            api_secret (str, optional): The API secret for the interruption detection, defaults to the LIVEKIT_INFERENCE_API_SECRET environment variable.
            http_session (aiohttp.ClientSession, optional): The HTTP session to use for the interruption detection.
        """
        super().__init__()
        if max_audio_duration > 3.0:
            raise ValueError("max_audio_duration must be less than or equal to 3.0 seconds")

        lk_base_url = (
            base_url
            if base_url
            else os.getenv("LIVEKIT_REMOTE_EOT_URL", get_default_inference_url())
        )
        lk_api_key: str = api_key if api_key else ""
        lk_api_secret: str = api_secret if api_secret else ""
        # use LiveKit credentials if using the inference service (production or staging)
        is_inference_url = lk_base_url in (DEFAULT_INFERENCE_URL, STAGING_INFERENCE_URL)
        if is_inference_url:
            lk_api_key = (
                api_key
                if api_key
                else os.getenv("LIVEKIT_INFERENCE_API_KEY", os.getenv("LIVEKIT_API_KEY", ""))
            )
            if not lk_api_key:
                raise ValueError(
                    "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable"
                )

            lk_api_secret = (
                api_secret
                if api_secret
                else os.getenv("LIVEKIT_INFERENCE_API_SECRET", os.getenv("LIVEKIT_API_SECRET", ""))
            )
            if not lk_api_secret:
                raise ValueError(
                    "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental variable"
                )

            use_proxy = True
        else:
            use_proxy = False

        self._opts = InterruptionOptions(
            sample_rate=SAMPLE_RATE,
            threshold=threshold,
            min_frames=math.ceil(min_interruption_duration * _FRAMES_PER_SECOND),
            max_audio_duration=max_audio_duration,
            audio_prefix_duration=audio_prefix_duration,
            detection_interval=detection_interval,
            inference_timeout=inference_timeout,
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
            use_proxy=use_proxy,
        )
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._sample_rate = SAMPLE_RATE
        self._session = http_session
        self._streams = weakref.WeakSet[InterruptionHttpStream | InterruptionWebSocketStream]()

        logger.info(
            "adaptive interruption detector initialized",
            extra={
                "base_url": self._opts.base_url,
                "detection_interval": self._opts.detection_interval,
                "audio_prefix_duration": self._opts.audio_prefix_duration,
                "max_audio_duration": self._opts.max_audio_duration,
                "min_frames": self._opts.min_frames,
                "threshold": self._opts.threshold,
                "inference_timeout": self._opts.inference_timeout,
                "use_proxy": self._opts.use_proxy,
            },
        )

    @property
    def model(self) -> str:
        return "adaptive interruption"

    @property
    def provider(self) -> str:
        return "livekit"

    @property
    def label(self) -> str:
        return self._label

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            InterruptionDetectionError(
                label=self._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = http_context.http_session()
        return self._session

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> InterruptionHttpStream | InterruptionWebSocketStream:
        stream: InterruptionHttpStream | InterruptionWebSocketStream
        if self._opts.use_proxy:
            stream = InterruptionWebSocketStream(model=self, conn_options=conn_options)
        else:
            stream = InterruptionHttpStream(model=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        threshold: NotGivenOr[float] = NOT_GIVEN,
        min_interruption_duration: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(threshold):
            self._opts.threshold = threshold
        if is_given(min_interruption_duration):
            self._opts.min_frames = math.ceil(min_interruption_duration * _FRAMES_PER_SECOND)

        for stream in self._streams:
            stream.update_options(
                threshold=threshold, min_interruption_duration=min_interruption_duration
            )


class InterruptionStreamBase(ABC):
    def __init__(
        self, *, model: AdaptiveInterruptionDetector, conn_options: APIConnectOptions
    ) -> None:
        self._model = model
        self._opts = model._opts
        self._session = model._ensure_session()

        self._input_ch = aio.Chan[InterruptionDataFrameType]()
        self._event_ch = aio.Chan[InterruptionEvent]()
        self._audio_buffer = AudioArrayBuffer(
            buffer_size=int(self._opts.max_audio_duration * self._opts.sample_rate),
            dtype=np.int16,
            sample_rate=self._opts.sample_rate,
        )
        self._cache = BoundedDict[int, InterruptionCacheEntry](maxsize=10)
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

        self._num_retries = 0
        self._conn_options = conn_options
        self._sample_rate = self._opts.sample_rate

        self._overlap_speech_started_at: float | None = None
        self._user_speech_span: trace.Span | None = None
        self._agent_speech_started: bool = False
        self._overlap_speech_started: bool = False
        self._overlap_count: int = 0
        self._accumulated_samples: int = 0
        self._batch_size: int = int(self._opts.detection_interval * self._opts.sample_rate)
        self._prefix_size: int = int(self._opts.audio_prefix_duration * self._opts.sample_rate)

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
                        f"failed to detect interruption after {self._num_retries} attempts",
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)

                    retry_interval = self._conn_options._interval_for_retry(self._num_retries)
                    logger.warning(
                        f"failed to detect interruption, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={
                            "model": self._model._label,
                            "attempt": self._num_retries,
                        },
                    )
                    await asyncio.sleep(retry_interval)

                self._num_retries += 1

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._model._emit_error(api_error, recoverable)

    def push_frame(self, frame: InterruptionDataFrameType) -> None:
        """Push some audio frame to be analyzed"""
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(frame)

    def flush(self) -> None:
        """Mark the end of the current segment"""
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(_FlushSentinel())

    def end_input(self) -> None:
        """Mark the end of input, no more audio will be pushed"""
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        """Close the stream immediately"""
        self._input_ch.close()
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()

    async def __anext__(self) -> InterruptionEvent:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904

            raise StopAsyncIteration from None

        return val

    def __aiter__(self) -> AsyncIterator[InterruptionEvent]:
        return self

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self._input_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")

    @staticmethod
    def _update_user_speech_span(
        user_speech_span: trace.Span, entry: InterruptionCacheEntry
    ) -> None:
        user_speech_span.set_attribute(
            trace_types.ATTR_IS_INTERRUPTION, str(entry.is_interruption).lower()
        )
        user_speech_span.set_attribute(
            trace_types.ATTR_INTERRUPTION_PROBABILITY, entry.get_probability()
        )
        user_speech_span.set_attribute(
            trace_types.ATTR_INTERRUPTION_TOTAL_DURATION, entry.get_total_duration()
        )
        user_speech_span.set_attribute(
            trace_types.ATTR_INTERRUPTION_PREDICTION_DURATION, entry.get_prediction_duration()
        )
        user_speech_span.set_attribute(
            trace_types.ATTR_INTERRUPTION_DETECTION_DELAY, entry.get_detection_delay()
        )

    @log_exceptions(logger=logger)
    async def _forward_data(self, output_ch: aio.Chan[npt.NDArray[np.int16]]) -> None:
        """Preprocess the audio data and forward it to the output channel for inference."""

        def _reset_state() -> None:
            self._agent_speech_started = False
            self._overlap_speech_started = False
            self._overlap_count = 0
            self._accumulated_samples = 0

            self._audio_buffer.reset()
            self._cache.clear()
            self._user_speech_span = None

        async for input_frame in self._input_ch:
            match input_frame:
                case _FlushSentinel():
                    continue
                case _AgentSpeechStartedSentinel() | _AgentSpeechEndedSentinel():
                    _reset_state()
                    self._agent_speech_started = isinstance(
                        input_frame, _AgentSpeechStartedSentinel
                    )
                    continue
                case _OverlapSpeechStartedSentinel() if self._agent_speech_started:
                    self._overlap_speech_started_at = input_frame._started_at
                    self._user_speech_span = input_frame._user_speaking_span
                    self._overlap_speech_started = True
                    self._accumulated_samples = 0
                    self._overlap_count += 1
                    # include the audio prefix in the window and
                    # only shift (remove leading silence) when the first overlap speech started
                    # otherwise, keep the existing data
                    if self._overlap_count == 1:
                        shift_size = max(
                            0,
                            len(self._audio_buffer)
                            - (
                                int(input_frame._speech_duration * self._sample_rate)
                                + self._prefix_size
                            ),
                        )
                        self._audio_buffer.shift(shift_size)
                    logger.trace(
                        "overlap speech started, starting interruption inference",
                        extra={
                            "overlap_count": self._overlap_count,
                        },
                    )
                    self._cache.clear()
                    continue
                case _OverlapSpeechEndedSentinel():
                    if self._overlap_speech_started and self._overlap_speech_started_at is not None:
                        logger.trace("overlap speech ended, stopping interruption inference")
                        self._user_speech_span = None
                        _, last_entry = self._cache.pop_if(
                            lambda entry: (
                                entry.total_duration is not None and entry.total_duration > 0
                            )
                        )
                        if last_entry is None:
                            logger.trace("no request made for overlap speech")
                        ev = InterruptionEvent.from_cache_entry(
                            entry=last_entry or _EMPTY_CACHE_ENTRY,
                            is_interruption=False,
                            started_at=self._overlap_speech_started_at,
                            ended_at=input_frame._ended_at,
                        )
                        self.send(ev)

                    self._overlap_speech_started = False
                    self._accumulated_samples = 0
                    self._overlap_speech_started_at = None
                    # we don't clear the cache here since responses might be in flight
                case rtc.AudioFrame() if self._agent_speech_started:
                    samples_written = self._audio_buffer.push_frame(input_frame)
                    self._accumulated_samples += samples_written
                    if (
                        self._accumulated_samples >= self._batch_size
                        and self._overlap_speech_started
                    ):
                        output_ch.send_nowait(self._audio_buffer.read())
                        self._accumulated_samples = 0

        output_ch.close()

    def send(self, event: InterruptionEvent) -> None:
        self._event_ch.send_nowait(event)
        self._model.emit(event.type, event)


# region: HTTP Stream


class InterruptionResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    created_at: int
    is_bargein: bool
    prediction_duration: float
    probabilities: npt.NDArray[np.float32] = Field(..., exclude=True)


class InterruptionHttpStream(InterruptionStreamBase):
    def __init__(
        self, *, model: AdaptiveInterruptionDetector, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(model=model, conn_options=conn_options)

    def update_options(
        self,
        *,
        threshold: NotGivenOr[float] = NOT_GIVEN,
        min_interruption_duration: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(threshold):
            self._opts.threshold = threshold
        if is_given(min_interruption_duration):
            self._opts.min_frames = math.ceil(min_interruption_duration * _FRAMES_PER_SECOND)

    @log_exceptions(logger=logger)
    async def _run(self) -> None:

        @log_exceptions(logger=logger)
        async def _send_task(input_ch: aio.Chan[npt.NDArray[np.int16]]) -> None:
            async for data in input_ch:
                if (
                    overlap_speech_started_at := self._overlap_speech_started_at
                ) is None or not self._overlap_speech_started:
                    continue
                resp: InterruptionResponse = await self.predict(data)
                created_at = resp.created_at
                self._cache[created_at] = entry = InterruptionCacheEntry(
                    created_at=created_at,
                    speech_input=data,
                    prediction_duration=resp.prediction_duration,
                    total_duration=(time.perf_counter_ns() - created_at) / 1e9,
                    detection_delay=time.time() - overlap_speech_started_at,
                    probabilities=resp.probabilities,
                    is_interruption=resp.is_bargein,
                )
                if entry.is_interruption and self._overlap_speech_started:
                    logger.debug("user interruption detected")
                    if self._user_speech_span:
                        self._update_user_speech_span(self._user_speech_span, entry)
                        self._user_speech_span = None
                    ev = InterruptionEvent.from_cache_entry(
                        entry=entry,
                        is_interruption=True,
                        started_at=overlap_speech_started_at,
                        ended_at=time.time(),
                    )
                    self.send(ev)
                    self._overlap_speech_started = False

        data_ch = aio.Chan[npt.NDArray[np.int16]]()
        tasks = [
            asyncio.create_task(self._forward_data(data_ch)),
            asyncio.create_task(_send_task(data_ch)),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await aio.cancel_and_wait(*tasks)

    @log_exceptions(logger=logger)
    async def predict(self, waveform: np.ndarray) -> InterruptionResponse:
        created_at = perf_counter_ns()
        try:
            async with self._session.post(
                url=f"{self._opts.base_url}/bargein?threshold={self._opts.threshold}&min_frames={int(self._opts.min_frames)}&created_at={int(created_at)}",
                headers={
                    "Content-Type": "application/octet-stream",
                    "Authorization": f"Bearer {create_access_token(self._opts.api_key, self._opts.api_secret)}",
                },
                data=waveform.tobytes(),
                timeout=aiohttp.ClientTimeout(total=self._opts.inference_timeout),
            ) as resp:
                try:
                    resp.raise_for_status()
                    data: dict[str, Any] = await resp.json()
                    result = InterruptionResponse.model_validate(
                        data
                        | {
                            "prediction_duration": (time.perf_counter_ns() - created_at) / 1e9,
                            "probabilities": np.array(
                                data.get("probabilities", []), dtype=np.float32
                            ),
                        }
                    )

                    logger.trace(
                        "interruption inference done",
                        extra={
                            "created_at": created_at,
                            "is_interruption": result.is_bargein,
                            "prediction_duration": result.prediction_duration,
                        },
                    )
                    return result
                except Exception as e:
                    msg = await resp.text()
                    raise APIError(f"error during interruption prediction: {e}", body=msg) from e
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            raise APIError(f"interruption inference timeout: {e}") from e
        except APIError as e:
            raise e
        except Exception as e:
            raise APIError(f"error during interruption prediction: {e}") from e


# endregion


# region: WebSocket Stream


# region: WebSocket messages
class InterruptionWSMessageType(str, Enum):
    SESSION_CREATE = "session.create"
    SESSION_CLOSE = "session.close"
    SESSION_CREATED = "session.created"
    SESSION_CLOSED = "session.closed"
    INTERRUPTION_DETECTED = "bargein_detected"
    INFERENCE_DONE = "inference_done"
    ERROR = "error"


class InterruptionWSSessionCreatedMessage(BaseModel):
    type: Literal[InterruptionWSMessageType.SESSION_CREATED] = (
        InterruptionWSMessageType.SESSION_CREATED
    )


class InterruptionWSSessionCreateSettings(BaseModel):
    sample_rate: int
    num_channels: int
    threshold: float
    min_frames: int
    encoding: Literal["s16le"]


class InterruptionWSSessionCreateMessage(BaseModel):
    type: Literal[InterruptionWSMessageType.SESSION_CREATE] = (
        InterruptionWSMessageType.SESSION_CREATE
    )
    settings: InterruptionWSSessionCreateSettings


class InterruptionWSSessionCloseMessage(BaseModel):
    type: Literal[InterruptionWSMessageType.SESSION_CLOSE] = InterruptionWSMessageType.SESSION_CLOSE


class InterruptionWSSessionClosedMessage(BaseModel):
    type: Literal[InterruptionWSMessageType.SESSION_CLOSED] = (
        InterruptionWSMessageType.SESSION_CLOSED
    )


class InterruptionWSDetectedMessage(BaseModel):
    type: Literal[InterruptionWSMessageType.INTERRUPTION_DETECTED] = (
        InterruptionWSMessageType.INTERRUPTION_DETECTED
    )
    created_at: int
    prediction_duration: float = Field(default=0.0)
    probabilities: list[float] = Field(default_factory=list)


class InterruptionWSInferenceDoneMessage(BaseModel):
    type: Literal[InterruptionWSMessageType.INFERENCE_DONE] = (
        InterruptionWSMessageType.INFERENCE_DONE
    )
    created_at: int
    prediction_duration: float = Field(default=0.0)
    probabilities: list[float] = Field(default_factory=list)


class InterruptionWSErrorMessage(BaseModel):
    type: Literal[InterruptionWSMessageType.ERROR] = InterruptionWSMessageType.ERROR
    message: str
    code: int
    session_id: str


AnyInterruptionWSMessage: TypeAlias = (
    InterruptionWSSessionCreateMessage
    | InterruptionWSSessionCreatedMessage
    | InterruptionWSSessionCloseMessage
    | InterruptionWSSessionClosedMessage
    | InterruptionWSDetectedMessage
    | InterruptionWSInferenceDoneMessage
    | InterruptionWSErrorMessage
)
InterruptionWSMessage: TypeAdapter[AnyInterruptionWSMessage] = TypeAdapter(
    Annotated[AnyInterruptionWSMessage, Field(discriminator="type")]
)

# endregion


class InterruptionWebSocketStream(InterruptionStreamBase):
    def __init__(
        self, *, model: AdaptiveInterruptionDetector, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(model=model, conn_options=conn_options)
        self._request_id = str(shortuuid("interruption_request_"))
        self._reconnect_event = asyncio.Event()

    def update_options(
        self,
        *,
        threshold: NotGivenOr[float] = NOT_GIVEN,
        min_interruption_duration: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(threshold):
            self._opts.threshold = threshold
        if is_given(min_interruption_duration):
            self._opts.min_frames = math.ceil(min_interruption_duration * _FRAMES_PER_SECOND)
        self._reconnect_event.set()

    @log_exceptions(logger=logger)
    async def _run(self) -> None:
        closing_ws = False

        @log_exceptions(logger=logger)
        async def send_task(
            ws: aiohttp.ClientWebSocketResponse, input_ch: aio.Chan[npt.NDArray[np.int16]]
        ) -> None:
            nonlocal closing_ws

            async for audio_data in input_ch:
                created_at = perf_counter_ns()
                header = struct.pack("<Q", created_at)  # 8 bytes
                await ws.send_bytes(header + audio_data.tobytes())
                self._cache[created_at] = InterruptionCacheEntry(
                    created_at=created_at,
                    speech_input=audio_data,
                )

            closing_ws = True
            msg = InterruptionWSSessionCloseMessage(
                type=InterruptionWSMessageType.SESSION_CLOSE,
            )
            await ws.send_str(msg.model_dump_json())

        @log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            while True:
                ws_msg = await ws.receive()
                if ws_msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return
                    raise APIStatusError(
                        message=f"LiveKit Adaptive Interruption connection closed unexpectedly: {ws_msg.data}",
                        status_code=ws.close_code or -1,
                        body=f"{ws_msg.data=} {ws_msg.extra=}",
                    )

                if ws_msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning(
                        "unexpected LiveKit Adaptive Interruption message type %s", ws_msg.type
                    )
                    continue

                data = json.loads(ws_msg.data)
                msg: AnyInterruptionWSMessage = InterruptionWSMessage.validate_python(data)

                match msg:
                    case (
                        InterruptionWSSessionCreatedMessage() | InterruptionWSSessionClosedMessage()
                    ):
                        pass
                    case InterruptionWSDetectedMessage():
                        created_at = msg.created_at
                        if (
                            overlap_speech_started_at := self._overlap_speech_started_at
                        ) is not None and self._overlap_speech_started:
                            entry = self._cache.set_or_update(
                                created_at,
                                lambda c=created_at: InterruptionCacheEntry(created_at=c),  # type: ignore[misc]
                                total_duration=(perf_counter_ns() - created_at) / 1e9,
                                probabilities=np.array(msg.probabilities, dtype=np.float32),
                                is_interruption=True,
                                prediction_duration=msg.prediction_duration,
                                detection_delay=time.time() - overlap_speech_started_at,
                            )
                            if self._user_speech_span:
                                self._update_user_speech_span(self._user_speech_span, entry)
                                self._user_speech_span = None
                            logger.debug(
                                "interruption detected",
                                extra={
                                    "total_duration": entry.get_total_duration(),
                                    "prediction_duration": entry.get_prediction_duration(),
                                    "detection_delay": entry.get_detection_delay(),
                                    "probability": entry.get_probability(),
                                },
                            )
                            ev = InterruptionEvent.from_cache_entry(
                                entry=entry,
                                is_interruption=True,
                                started_at=overlap_speech_started_at,
                                ended_at=time.time(),
                            )
                            self.send(ev)
                            self._overlap_speech_started = False
                    case InterruptionWSInferenceDoneMessage():
                        created_at = msg.created_at
                        if (
                            overlap_speech_started_at := self._overlap_speech_started_at
                        ) is not None and self._overlap_speech_started:
                            entry = self._cache.set_or_update(
                                created_at,
                                lambda c=created_at: InterruptionCacheEntry(created_at=c),  # type: ignore[misc]
                                total_duration=(perf_counter_ns() - created_at) / 1e9,
                                prediction_duration=msg.prediction_duration,
                                probabilities=np.array(msg.probabilities, dtype=np.float32),
                                is_interruption=False,
                                detection_delay=time.time() - overlap_speech_started_at,
                            )
                            logger.trace(
                                "interruption inference done",
                                extra={
                                    "total_duration": entry.get_total_duration(),
                                    "prediction_duration": entry.get_prediction_duration(),
                                    "probability": entry.get_probability(),
                                },
                            )
                    case InterruptionWSErrorMessage():
                        raise APIError(
                            f"LiveKit Adaptive Interruption returned error: {msg.code}",
                            body=msg.message,
                        )
                    case _:
                        logger.warning(
                            "received unexpected message from LiveKit Adaptive Interruption: %s",
                            data,
                        )

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            data_ch = aio.Chan[npt.NDArray[np.int16]]()
            try:
                closing_ws = False
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(self._forward_data(data_ch)),
                    asyncio.create_task(send_task(ws, data_ch)),
                    asyncio.create_task(recv_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    closing_ws = True
                    if ws is not None and not ws.closed:
                        await ws.close()
                        ws = None
                    await aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    try:
                        tasks_group.exception()
                    except asyncio.CancelledError:
                        pass
            finally:
                closing_ws = True
                if ws is not None and not ws.closed:
                    await ws.close()

    @log_exceptions(logger=logger)
    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Connect to the LiveKit Adaptive Interruption WebSocket."""
        settings = InterruptionWSSessionCreateSettings(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            threshold=self._opts.threshold,
            min_frames=self._model._opts.min_frames,
            encoding="s16le",
        )

        base_url = self._opts.base_url
        if base_url.startswith(("http://", "https://")):
            base_url = base_url.replace("http", "ws", 1)
        headers = {
            "Authorization": f"Bearer {create_access_token(self._opts.api_key, self._opts.api_secret)}"
        }
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(f"{base_url}/bargein", headers=headers),
                self._conn_options.timeout,
            )
        except (
            aiohttp.ClientConnectorError,
            asyncio.TimeoutError,
            aiohttp.ClientResponseError,
        ) as e:
            if isinstance(e, aiohttp.ClientResponseError) and e.status == 429:
                raise APIStatusError(
                    "LiveKit Adaptive Interruption quota exceeded", status_code=e.status
                ) from e
            raise APIConnectionError("failed to connect to LiveKit Adaptive Interruption") from e

        try:
            msg = InterruptionWSSessionCreateMessage(
                type=InterruptionWSMessageType.SESSION_CREATE,
                settings=settings,
            )
            await ws.send_str(msg.model_dump_json())
        except Exception as e:
            await ws.close()
            raise APIConnectionError(
                "failed to send session.create message to LiveKit Adaptive Interruption"
            ) from e

        return ws


# endregion


def _estimate_probability(
    probabilities: npt.NDArray[np.float32] | None, window_size: float = MIN_INTERRUPTION_DURATION
) -> float:
    """
    Estimate the probability of the interruption event based on the probabilities of the frames.
    The estimated probability is the maximum of the minimum of every window_size consecutive frames.
    """
    if probabilities is None:
        return 0.0

    n_th = math.ceil(window_size / 0.025)  # 25ms per frame
    if len(probabilities) < n_th:
        return 0.0

    # return the n-th maximum of the probabilities
    return float(np.partition(probabilities, -n_th)[-n_th])
