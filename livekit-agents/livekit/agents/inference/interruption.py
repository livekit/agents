from __future__ import annotations

import asyncio
import json
import math
import os
import struct
import time
import warnings
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, overload

import aiohttp
import numpy as np
import numpy.typing as npt
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc

from .._exceptions import APIConnectionError, APIError, APIStatusError
from ..log import logger
from ..telemetry import trace_types
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import aio, http_context, is_given, log_exceptions, shortuuid
from ._utils import (
    DEFAULT_INFERENCE_URL,
    STAGING_INFERENCE_URL,
    create_access_token,
    get_default_inference_url,
)

if TYPE_CHECKING:
    from ..vad import VAD

SAMPLE_RATE = 16000
THRESHOLD = 0.5
MIN_INTERRUPTION_DURATION = 0.025 * 2  # 25ms per frame, 2 consecutive frames
MAX_AUDIO_DURATION = 3  # 3 seconds
DETECTION_INTERVAL = 0.1  # 0.1 second
AUDIO_PREFIX_DURATION = 1.0  # 1.0 second
REMOTE_INFERENCE_TIMEOUT = 1
_FRAMES_PER_SECOND = 40

MSG_INPUT_AUDIO = "input_audio"
MSG_SESSION_CREATE = "session.create"
MSG_SESSION_CLOSE = "session.close"
MSG_SESSION_CREATED = "session.created"
MSG_SESSION_CLOSED = "session.closed"
# fun fact: this was called bargein during development
MSG_INTERRUPTION_DETECTED = "bargein_detected"
MSG_INFERENCE_DONE = "inference_done"
MSG_ERROR = "error"


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


class InterruptionDetectionError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["interruption_detection_error"] = "interruption_detection_error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


@dataclass
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


@dataclass
class InterruptionCacheEntry:
    """Typed cache entry for interruption inference results."""

    created_at: int
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


# Default empty entry used when cache misses occur
_EMPTY_CACHE_ENTRY = InterruptionCacheEntry(created_at=0)


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
                timestamp=time.time(),
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
    class _AgentSpeechStartedSentinel:
        pass

    class _AgentSpeechEndedSentinel:
        pass

    class _OverlapSpeechStartedSentinel:
        def __init__(
            self,
            speech_duration: float | None = None,
            user_speaking_span: trace.Span | None = None,
        ) -> None:
            self._speech_duration = speech_duration or 0.0
            self._user_speaking_span = user_speaking_span

    class _OverlapSpeechEndedSentinel:
        pass

    class _FlushSentinel:
        pass

    def __init__(
        self, *, model: AdaptiveInterruptionDetector, conn_options: APIConnectOptions
    ) -> None:
        self._model = model
        self._opts = model._opts
        self._session = model._ensure_session()
        self._last_activity_time = time.perf_counter()
        self._input_ch = aio.Chan[
            rtc.AudioFrame
            | InterruptionStreamBase._AgentSpeechStartedSentinel
            | InterruptionStreamBase._AgentSpeechEndedSentinel
            | InterruptionStreamBase._OverlapSpeechStartedSentinel
            | InterruptionStreamBase._OverlapSpeechEndedSentinel
            | InterruptionStreamBase._FlushSentinel
        ]()
        self._event_ch = aio.Chan[InterruptionEvent]()
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())
        self._num_retries = 0
        self._conn_options = conn_options
        self._sample_rate = self._opts.sample_rate
        self._resampler: rtc.AudioResampler | None = None
        self._overlap_speech_started_at: float | None = None
        self._user_speech_span: trace.Span | None = None

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
        self._model.emit(
            "error",
            InterruptionDetectionError(
                timestamp=time.time(),
                label=self._model._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    def push_frame(
        self,
        frame: rtc.AudioFrame
        | InterruptionStreamBase._AgentSpeechStartedSentinel
        | InterruptionStreamBase._AgentSpeechEndedSentinel
        | InterruptionStreamBase._OverlapSpeechStartedSentinel
        | InterruptionStreamBase._OverlapSpeechEndedSentinel,
    ) -> None:
        """Push some audio frame to be analyzed"""
        self._check_input_not_ended()
        self._check_not_closed()

        if not isinstance(frame, rtc.AudioFrame):
            if isinstance(frame, InterruptionStreamBase._OverlapSpeechStartedSentinel):
                self._overlap_speech_started_at = time.time() - frame._speech_duration
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
        data_chan = aio.Chan[npt.NDArray[np.int16]]()
        overlap_speech_started: bool = False
        cache: _BoundedCache[int, InterruptionCacheEntry] = _BoundedCache(max_len=10)

        @log_exceptions(logger=logger)
        async def _forward_data() -> None:
            nonlocal data_chan
            nonlocal overlap_speech_started
            nonlocal cache

            agent_speech_started: bool = False
            overlap_count: int = 0
            inference_s16_data = np.zeros(
                int(self._opts.max_audio_duration * self._opts.sample_rate), dtype=np.int16
            )
            start_idx: int = 0
            accumulated_samples: int = 0

            async for input_frame in self._input_ch:
                # start accumulating user speech when agent starts speaking
                if isinstance(input_frame, InterruptionStreamBase._AgentSpeechStartedSentinel):
                    agent_speech_started = True
                    overlap_speech_started = False
                    overlap_count = 0
                    accumulated_samples = 0
                    start_idx = 0
                    cache.clear()
                    continue

                # reset state when agent stops speaking
                if isinstance(input_frame, InterruptionStreamBase._AgentSpeechEndedSentinel):
                    agent_speech_started = False
                    overlap_speech_started = False
                    overlap_count = 0
                    accumulated_samples = 0
                    start_idx = 0
                    cache.clear()
                    continue

                # start inferencing against the overlap speech
                if (
                    isinstance(input_frame, InterruptionStreamBase._OverlapSpeechStartedSentinel)
                    and agent_speech_started
                ):
                    self._user_speech_span = input_frame._user_speaking_span
                    overlap_speech_started = True
                    accumulated_samples = 0
                    overlap_count += 1
                    # include the audio prefix in the window and
                    # only shift (remove leading silence) when the first overlap speech started
                    # otherwise, keep the existing data
                    if overlap_count == 1:
                        shift_size = min(
                            start_idx,
                            int(input_frame._speech_duration * self._sample_rate)
                            + int(self._opts.audio_prefix_duration * self._sample_rate),
                        )
                        inference_s16_data[:shift_size] = inference_s16_data[
                            start_idx - shift_size : start_idx
                        ].copy()
                        start_idx = shift_size
                    logger.trace(
                        "overlap speech started, starting interruption inference",
                        extra={
                            "overlap_count": overlap_count,
                            "start_idx": start_idx,
                        },
                    )
                    cache.clear()
                    continue

                if isinstance(input_frame, InterruptionStreamBase._OverlapSpeechEndedSentinel):
                    if overlap_speech_started:
                        logger.trace("overlap speech ended, stopping interruption inference")
                        self._user_speech_span = None
                        _, last_entry = cache.pop(
                            lambda x: x.total_duration is not None and x.total_duration > 0
                        )
                        if last_entry is None:
                            logger.trace("no request made for overlap speech")
                        entry = last_entry or _EMPTY_CACHE_ENTRY
                        ev = InterruptionEvent(
                            type="user_non_interruption_detected",
                            timestamp=time.time(),
                            is_interruption=False,
                            overlap_speech_started_at=self._overlap_speech_started_at,
                            speech_input=entry.speech_input,
                            probabilities=entry.probabilities,
                            total_duration=entry.get_total_duration(),
                            detection_delay=entry.get_detection_delay(),
                            prediction_duration=entry.get_prediction_duration(),
                            probability=entry.get_probability(),
                        )
                        self._event_ch.send_nowait(ev)
                        self._model.emit("user_non_interruption_detected", ev)
                    overlap_speech_started = False
                    accumulated_samples = 0
                    # we don't clear the cache here since responses might be in flight
                    continue

                if isinstance(input_frame, InterruptionStreamBase._FlushSentinel):
                    continue

                if not agent_speech_started or not isinstance(input_frame, rtc.AudioFrame):
                    continue

                if input_frame.sample_rate != self._sample_rate:
                    raise ValueError("the sample rate of the input frames must be consistent")

                start_idx, samples_written = _write_to_inference_s16_data(
                    input_frame,
                    start_idx,
                    inference_s16_data,
                    self._opts.max_audio_duration,
                )
                accumulated_samples += samples_written
                if (
                    accumulated_samples
                    >= int(self._opts.detection_interval * self._opts.sample_rate)
                    and overlap_speech_started
                ):
                    data_chan.send_nowait(inference_s16_data[:start_idx].copy())
                    accumulated_samples = 0

            data_chan.close()

        @log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal overlap_speech_started
            nonlocal cache
            async for data in data_chan:
                if self._overlap_speech_started_at is None:
                    continue
                resp = await self.predict(data)
                created_at = resp["created_at"]
                cache[created_at] = entry = InterruptionCacheEntry(
                    created_at=created_at,
                    total_duration=(time.perf_counter_ns() - created_at) / 1e9,
                    speech_input=data,
                    detection_delay=time.time() - self._overlap_speech_started_at,
                    probabilities=resp["probabilities"],
                    prediction_duration=resp["prediction_duration"],
                    is_interruption=resp["is_bargein"],
                )
                if overlap_speech_started and entry.is_interruption:
                    logger.debug("user interruption detected")
                    if self._user_speech_span:
                        self._update_user_speech_span(self._user_speech_span, entry)
                        self._user_speech_span = None
                    ev = InterruptionEvent(
                        type="user_interruption_detected",
                        timestamp=time.time(),
                        overlap_speech_started_at=self._overlap_speech_started_at,
                        is_interruption=entry.is_interruption,
                        speech_input=entry.speech_input,
                        probabilities=entry.probabilities,
                        total_duration=entry.get_total_duration(),
                        prediction_duration=entry.get_prediction_duration(),
                        detection_delay=entry.get_detection_delay(),
                        probability=entry.get_probability(),
                    )
                    self._event_ch.send_nowait(ev)
                    self._model.emit("user_interruption_detected", ev)
                    overlap_speech_started = False

        tasks = [
            asyncio.create_task(_forward_data()),
            asyncio.create_task(_send_task()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await aio.cancel_and_wait(*tasks)

    @log_exceptions(logger=logger)
    async def predict(self, waveform: np.ndarray) -> dict[str, Any]:
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
                    # {
                    #     "created_at": int,
                    #     "is_bargein": bool,
                    #     "probabilities": list[float], optional, will be converted to numpy array
                    # }
                    data["prediction_duration"] = (time.perf_counter_ns() - created_at) / 1e9
                    data["probabilities"] = np.array(
                        data.get("probabilities", []), dtype=np.float32
                    )
                    logger.trace(
                        "interruption inference done",
                        extra={
                            "created_at": created_at,
                            "is_interruption": data["is_bargein"],
                            "prediction_duration": data["prediction_duration"],
                        },
                    )
                    return data
                except Exception as e:
                    msg = await resp.text()
                    raise APIError(f"error during interruption prediction: {e}", body=msg) from e
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            raise APIError(f"interruption inference timeout: {e}") from e
        except APIError as e:
            raise e
        except Exception as e:
            raise APIError(f"error during interruption prediction: {e}") from e


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
        overlap_speech_started: bool = False
        cache: _BoundedCache[int, InterruptionCacheEntry] = _BoundedCache(max_len=10)
        self._user_speech_span = None

        @log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            nonlocal overlap_speech_started
            nonlocal cache

            agent_speech_started: bool = False
            overlap_count: int = 0
            inference_s16_data = np.zeros(
                int(self._opts.max_audio_duration * self._opts.sample_rate), dtype=np.int16
            )
            start_idx: int = 0
            accumulated_samples: int = 0

            async for input_frame in self._input_ch:
                # start accumulating user speech when agent starts speaking
                if isinstance(input_frame, InterruptionStreamBase._AgentSpeechStartedSentinel):
                    agent_speech_started = True
                    overlap_speech_started = False
                    accumulated_samples = 0
                    overlap_count = 0
                    start_idx = 0
                    cache.clear()
                    continue

                # reset state when agent stops speaking
                if isinstance(input_frame, InterruptionStreamBase._AgentSpeechEndedSentinel):
                    agent_speech_started = False
                    overlap_speech_started = False
                    accumulated_samples = 0
                    overlap_count = 0
                    start_idx = 0
                    cache.clear()
                    continue

                # start inferencing against the overlap speech
                if (
                    isinstance(input_frame, InterruptionStreamBase._OverlapSpeechStartedSentinel)
                    and agent_speech_started
                ):
                    self._user_speech_span = input_frame._user_speaking_span
                    overlap_speech_started = True
                    accumulated_samples = 0
                    overlap_count += 1
                    if overlap_count == 1:
                        shift_size = min(
                            start_idx,
                            int(input_frame._speech_duration * self._sample_rate)
                            + int(self._opts.audio_prefix_duration * self._sample_rate),
                        )
                        inference_s16_data[:shift_size] = inference_s16_data[
                            start_idx - shift_size : start_idx
                        ].copy()
                        start_idx = shift_size
                    logger.trace(
                        "overlap speech started, starting interruption inference",
                        extra={
                            "overlap_count": overlap_count,
                            "start_idx": start_idx,
                        },
                    )
                    cache.clear()
                    continue

                # end inferencing against the overlap speech
                if isinstance(input_frame, InterruptionStreamBase._OverlapSpeechEndedSentinel):
                    if overlap_speech_started:
                        logger.trace("overlap speech ended, stopping interruption inference")
                        self._user_speech_span = None
                        # only pop the last complete request
                        _, last_entry = cache.pop(lambda x: x.get_total_duration() > 0)
                        if last_entry is None:
                            logger.trace("no request made for overlap speech")
                        entry = last_entry or _EMPTY_CACHE_ENTRY
                        ev = InterruptionEvent(
                            type="user_non_interruption_detected",
                            timestamp=time.time(),
                            is_interruption=False,
                            overlap_speech_started_at=self._overlap_speech_started_at,
                            speech_input=entry.speech_input,
                            probabilities=entry.probabilities,
                            total_duration=entry.get_total_duration(),
                            detection_delay=entry.get_detection_delay(),
                            prediction_duration=entry.get_prediction_duration(),
                            probability=entry.get_probability(),
                        )
                        self._event_ch.send_nowait(ev)
                        self._model.emit("user_non_interruption_detected", ev)
                    overlap_speech_started = False
                    accumulated_samples = 0
                    continue

                if isinstance(input_frame, InterruptionStreamBase._FlushSentinel):
                    continue

                if not agent_speech_started or not isinstance(input_frame, rtc.AudioFrame):
                    continue

                if input_frame.sample_rate != self._sample_rate:
                    raise ValueError(
                        f"sample rate mismatch: {input_frame.sample_rate} != {self._sample_rate}"
                    )

                start_idx, samples_written = _write_to_inference_s16_data(
                    input_frame,
                    start_idx,
                    inference_s16_data,
                    self._opts.max_audio_duration,
                )
                accumulated_samples += samples_written
                if (
                    accumulated_samples >= int(self._opts.detection_interval * self._sample_rate)
                    and overlap_speech_started
                ):
                    created_at = perf_counter_ns()
                    header = struct.pack("<Q", created_at)  # 8 bytes
                    await ws.send_bytes(header + inference_s16_data[:start_idx].tobytes())
                    cache[created_at] = InterruptionCacheEntry(
                        created_at=created_at,
                        speech_input=inference_s16_data[:start_idx].copy(),
                    )
                    accumulated_samples = 0

            closing_ws = True
            finalize_msg = {
                "type": MSG_SESSION_CLOSE,
            }
            await ws.send_str(json.dumps(finalize_msg))

        @log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            nonlocal overlap_speech_started
            nonlocal cache

            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return
                    raise APIStatusError(
                        message=f"LiveKit Interruption connection closed unexpectedly: {msg.data}"
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected LiveKit Interruption message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                msg_type = data["type"]

                if msg_type == MSG_SESSION_CREATED:
                    pass
                elif msg_type == MSG_INTERRUPTION_DETECTED:
                    created_at = int(data["created_at"])
                    if overlap_speech_started and self._overlap_speech_started_at is not None:
                        entry = cache.set_or_update(
                            created_at,
                            lambda c=created_at: InterruptionCacheEntry(created_at=c),  # type: ignore[misc]
                            total_duration=(perf_counter_ns() - created_at) / 1e9,
                            probabilities=np.array(data.get("probabilities", []), dtype=np.float32),
                            is_interruption=True,
                            prediction_duration=data.get("prediction_duration", 0.0),
                            detection_delay=time.time() - self._overlap_speech_started_at,
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
                        ev = InterruptionEvent(
                            type="user_interruption_detected",
                            timestamp=time.time(),
                            is_interruption=True,
                            total_duration=entry.get_total_duration(),
                            prediction_duration=entry.get_prediction_duration(),
                            overlap_speech_started_at=self._overlap_speech_started_at,
                            speech_input=entry.speech_input,
                            probabilities=entry.probabilities,
                            detection_delay=entry.get_detection_delay(),
                            probability=entry.get_probability(),
                        )
                        self._event_ch.send_nowait(ev)
                        self._model.emit("user_interruption_detected", ev)
                        overlap_speech_started = False

                elif msg_type == MSG_INFERENCE_DONE:
                    created_at = int(data["created_at"])
                    if self._overlap_speech_started_at is not None:
                        entry = cache.set_or_update(
                            created_at,
                            lambda c=created_at: InterruptionCacheEntry(created_at=c),  # type: ignore[misc]
                            total_duration=(perf_counter_ns() - created_at) / 1e9,
                            prediction_duration=data.get("prediction_duration", 0.0),
                            probabilities=np.array(data.get("probabilities", []), dtype=np.float32),
                            is_interruption=data.get("is_bargein", False),
                            detection_delay=time.time() - self._overlap_speech_started_at,
                        )
                        logger.trace(
                            "interruption inference done",
                            extra={
                                "total_duration": entry.get_total_duration(),
                                "prediction_duration": entry.get_prediction_duration(),
                            },
                        )
                elif msg_type == MSG_SESSION_CLOSED:
                    pass
                elif msg_type == MSG_ERROR:
                    raise APIError(f"LiveKit Interruption returned error: {msg.data}")
                else:
                    logger.warning(
                        "received unexpected message from LiveKit Interruption: %s", data
                    )

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                closing_ws = False
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
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
        """Connect to the LiveKit Interruption WebSocket."""
        params: dict[str, Any] = {
            "settings": {
                "sample_rate": self._opts.sample_rate,
                "num_channels": 1,
                "threshold": self._opts.threshold,
                "min_frames": self._model._opts.min_frames,
                "encoding": "s16le",
            },
        }

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
                    "LiveKit Interruption quota exceeded", status_code=e.status
                ) from e
            raise APIConnectionError("failed to connect to LiveKit Interruption") from e

        try:
            params["type"] = MSG_SESSION_CREATE
            await ws.send_str(json.dumps(params))
        except Exception as e:
            await ws.close()
            raise APIConnectionError(
                "failed to send session.create message to LiveKit Interruption"
            ) from e

        return ws


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


def _write_to_inference_s16_data(
    frame: rtc.AudioFrame, start_idx: int, out_data: np.ndarray, max_audio_duration: float
) -> tuple[int, int]:
    """Write the audio frame to the output data array and return the new start index and the number of samples written."""

    max_window_size = int(max_audio_duration * frame.sample_rate)

    if frame.samples_per_channel > out_data.shape[0]:
        raise ValueError("frame samples are greater than the max window size")

    # shift the data to the left if the window would overflow
    if (shift := start_idx + frame.samples_per_channel - max_window_size) > 0:
        out_data[: start_idx - shift] = out_data[shift:start_idx].copy()
        start_idx -= shift

    slice_ = out_data[start_idx : start_idx + frame.samples_per_channel]
    if frame.num_channels > 1:
        arr_i16 = np.frombuffer(
            frame.data, dtype=np.int16, count=frame.samples_per_channel * frame.num_channels
        ).reshape(-1, frame.num_channels)
        slice_[:] = (np.sum(arr_i16, axis=1, dtype=np.int32) // frame.num_channels).astype(np.int16)
    else:
        slice_[:] = np.frombuffer(frame.data, dtype=np.int16, count=frame.samples_per_channel)
    start_idx += frame.samples_per_channel
    return start_idx, frame.samples_per_channel


_K = TypeVar("_K")
_V = TypeVar("_V")


class _BoundedCache(Generic[_K, _V]):
    def __init__(
        self, max_len: int = 5, default_key: _K | None = None, default_value: _V | None = None
    ) -> None:
        self._cache: OrderedDict[_K, _V] = OrderedDict()
        self._max_len = max_len
        self._default_key = default_key
        self._default_value = default_value

    def __setitem__(self, key: _K, value: _V) -> None:
        self._cache[key] = value
        if len(self._cache) > self._max_len:
            self._cache.popitem(last=False)

    def __getitem__(self, key: _K) -> _V:
        return self._cache[key]

    @overload
    def get(self, key: _K) -> _V | None: ...

    @overload
    def get(self, key: _K, default: _V) -> _V: ...

    def get(self, key: _K, default: _V | None = None) -> _V | None:
        return self._cache.get(key, default)

    def update(self, key: _K, **kwargs: Any) -> _V | None:
        """Update fields on an existing cache entry. Returns the updated entry or None if not found."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        for field_name, value in kwargs.items():
            if value is not None and hasattr(entry, field_name):
                setattr(entry, field_name, value)
        return entry

    def set_or_update(self, key: _K, factory: Callable[[], _V], **kwargs: Any) -> _V:
        """Get existing entry and update it, or create a new one using factory."""
        entry = self.get(key)
        if entry is None:
            entry = factory()
            self[key] = entry
        for field_name, value in kwargs.items():
            if value is not None and hasattr(entry, field_name):
                setattr(entry, field_name, value)
        return entry

    def pop(self, predicate: Callable[[_V], bool] | None = None) -> tuple[_K | None, _V | None]:
        if predicate is None:
            if self._cache:
                return self._cache.popitem(last=True)
            return self._default_key, self._default_value

        # Find and remove only the matching entry, preserving others
        for key in reversed(list(self._cache.keys())):
            if predicate(self._cache[key]):
                return key, self._cache.pop(key)
        return self._default_key, self._default_value

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def keys(self) -> list[_K]:
        return list(self._cache.keys())


def handle_deprecation(
    vad: NotGivenOr[VAD | None] = NOT_GIVEN,
    allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    interruption_handling: NotGivenOr[Literal["adaptive", "vad", False]] = NOT_GIVEN,
) -> Literal["adaptive", "vad", False]:
    if is_given(allow_interruptions):
        warnings.warn(
            "`allow_interruptions` is deprecated, use `interruption_handling` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if is_given(interruption_handling):
            raise ValueError(
                "both `allow_interruptions` and `interruption_handling` are provided,"
                " please only use `interruption_handling` instead"
            )
        if allow_interruptions:
            if vad is None:
                raise ValueError("`allow_interruptions` is True but `vad` is not provided")
            return "adaptive"
        return False

    if is_given(interruption_handling):
        if vad is None and interruption_handling in {"adaptive", "vad"}:
            raise ValueError("`vad` is not provided but `interruption_handling` is not False")
        return cast(Literal["adaptive", "vad", False], interruption_handling)

    if vad is None:
        return False

    return "adaptive"
