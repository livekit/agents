from __future__ import annotations

import asyncio
import json
import math
import os
import struct
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum, unique
from time import perf_counter_ns
from typing import Any, Callable, Generic, Literal, TypeVar, Union, overload

import aiohttp
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc

from .._exceptions import APIConnectionError, APIError, APIStatusError
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import aio, http_context, is_given, log_exceptions, shortuuid
from ._utils import create_access_token

SAMPLE_RATE = 16000
THRESHOLD = 0.65
MIN_BARGEIN_DURATION = 0.025 * 2  # 25ms per frame
MAX_WINDOW_SIZE = 3 * 16000  # 3 seconds at 16000 Hz
STEP_SIZE = int(0.2 * 16000)  # 0.2 second at 16000 Hz
PREFIX_SIZE = int(0.5 * 16000)  # 0.5 second at 16000 Hz
REMOTE_INFERENCE_TIMEOUT = 1
DEFAULT_BASE_URL = "https://agent-gateway.livekit.cloud/v1"

MSG_INPUT_AUDIO = "input_audio"
MSG_SESSION_CREATE = "session.create"
MSG_SESSION_CLOSE = "session.close"
MSG_SESSION_CREATED = "session.created"
MSG_SESSION_CLOSED = "session.closed"
MSG_BARGEIN_DETECTED = "bargein_detected"
MSG_INFERENCE_DONE = "inference_done"
MSG_ERROR = "error"


@unique
class BargeinEventType(str, Enum):
    BARGEIN = "bargein"
    OVERLAP_SPEECH_ENDED = "overlap_speech_ended"


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
    """Whether bargein is detected."""

    total_duration: float = 0.0
    """Time taken to perform the inference, in seconds."""

    detection_delay: float = 0.0
    """Total time from the onset of the speech to the detection of the bargein, in seconds."""

    overlap_speech_started_at: float | None = None
    """Timestamp (in seconds) when the overlap speech started. Useful for emitting held transcripts."""

    speech_input: npt.NDArray[np.int16] | None = None
    """The audio input that was used for the inference."""

    probabilities: npt.NDArray[np.float32] | None = None
    """The probabilities for the bargein detection."""


class BargeinError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["bargein_error"] = "bargein_error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


@dataclass
class BargeinOptions:
    sample_rate: int
    """The sample rate of the audio frames, defaults to 16000Hz"""
    threshold: float
    """The threshold for the bargein detection, defaults to 0.65"""
    min_frames: int
    """The minimum number of frames to detect a bargein, defaults to 50ms/2 frames"""
    max_window_size: int
    """The maximum window size for the bargein detection, defaults to 3s at 16000Hz = 48000 samples"""
    prefix_size: int
    """The prefix size for the bargein detection, defaults to 0.5s at 16000Hz = 8000 samples"""
    step_size: int
    """The step size for the bargein detection, defaults to 0.1s at 16000Hz = 1600 samples"""
    inference_timeout: float
    """The timeout for the bargein detection, defaults to 1 second"""
    base_url: str
    api_key: str
    api_secret: str
    # whether to use the inference instead of the hosted API
    use_proxy: bool


class BargeinDetector(
    rtc.EventEmitter[Literal["bargein_detected", "overlap_speech_ended", "error"]],
):
    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = THRESHOLD,
        min_bargein_duration: float = MIN_BARGEIN_DURATION,
        max_window_size: int = MAX_WINDOW_SIZE,
        prefix_size: int = PREFIX_SIZE,
        step_size: int = STEP_SIZE,
        inference_timeout: float = REMOTE_INFERENCE_TIMEOUT,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        use_proxy: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__()
        lk_base_url = (
            base_url
            if base_url
            else os.environ.get("LIVEKIT_BARGEIN_INFERENCE_URL", DEFAULT_BASE_URL)
        )

        lk_api_key: str = api_key if api_key else ""
        lk_api_secret: str = api_secret if api_secret else ""
        # use LiveKit credentials if using the default base URL (inference gateway)
        if lk_base_url == DEFAULT_BASE_URL:
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

        self._opts = BargeinOptions(
            sample_rate=sample_rate,
            threshold=threshold,
            min_frames=math.ceil(min_bargein_duration * 40),  # 40 frames per second
            max_window_size=max_window_size,
            prefix_size=prefix_size,
            step_size=step_size,
            inference_timeout=inference_timeout,
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
            use_proxy=use_proxy,
        )
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._sample_rate = sample_rate
        self._session = http_session
        self._streams = weakref.WeakSet[Union[BargeinHttpStream, BargeinWebSocketStream]]()

        logger.info(
            "bargein detector initialized",
            extra={
                "base_url": self._opts.base_url,
                "step_size": self._opts.step_size,
                "prefix_size": self._opts.prefix_size,
                "max_window_size": self._opts.max_window_size,
                "min_frames": self._opts.min_frames,
                "threshold": self._opts.threshold,
                "inference_timeout": self._opts.inference_timeout,
                "use_proxy": self._opts.use_proxy,
            },
        )

    @property
    def model(self) -> str:
        return "bargein"

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
            BargeinError(
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
    ) -> BargeinHttpStream | BargeinWebSocketStream:
        stream: BargeinHttpStream | BargeinWebSocketStream
        if self._opts.use_proxy:
            stream = BargeinWebSocketStream(bargein_detector=self, conn_options=conn_options)
        else:
            stream = BargeinHttpStream(bargein_detector=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        threshold: NotGivenOr[float] = NOT_GIVEN,
        min_bargein_duration: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(threshold):
            self._opts.threshold = threshold
        if is_given(min_bargein_duration):
            self._opts.min_frames = math.ceil(min_bargein_duration * 40)

        for stream in self._streams:
            stream.update_options(threshold=threshold, min_bargein_duration=min_bargein_duration)


class BargeinStreamBase(ABC):
    class _AgentSpeechStartedSentinel:
        pass

    class _AgentSpeechEndedSentinel:
        pass

    class _OverlapSpeechStartedSentinel:
        def __init__(self, speech_duration: float | None = None) -> None:
            self._speech_duration = speech_duration or 0.0

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
                BargeinStreamBase._AgentSpeechStartedSentinel,
                BargeinStreamBase._AgentSpeechEndedSentinel,
                BargeinStreamBase._OverlapSpeechStartedSentinel,
                BargeinStreamBase._OverlapSpeechEndedSentinel,
                BargeinStreamBase._FlushSentinel,
            ]
        ]()
        self._event_ch = aio.Chan[BargeinEvent]()
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())
        self._num_retries = 0
        self._conn_options = conn_options
        self._sample_rate = bargein_detector._sample_rate
        self._resampler: rtc.AudioResampler | None = None
        self._overlap_speech_started_at: float | None = None

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

    def push_frame(
        self,
        frame: rtc.AudioFrame
        | BargeinStreamBase._AgentSpeechStartedSentinel
        | BargeinStreamBase._AgentSpeechEndedSentinel
        | BargeinStreamBase._OverlapSpeechStartedSentinel
        | BargeinStreamBase._OverlapSpeechEndedSentinel,
    ) -> None:
        """Push some audio frame to be analyzed"""
        self._check_input_not_ended()
        self._check_not_closed()

        if not isinstance(frame, rtc.AudioFrame):
            if isinstance(frame, BargeinStreamBase._OverlapSpeechStartedSentinel):
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


class BargeinHttpStream(BargeinStreamBase):
    def __init__(
        self, *, bargein_detector: BargeinDetector, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(bargein_detector=bargein_detector, conn_options=conn_options)
        self._model: BargeinDetector = bargein_detector
        self._opts = bargein_detector._opts

    def update_options(
        self,
        *,
        threshold: NotGivenOr[float] = NOT_GIVEN,
        min_bargein_duration: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(threshold):
            self._opts.threshold = threshold
        if is_given(min_bargein_duration):
            self._opts.min_frames = math.ceil(min_bargein_duration * 40)

    @log_exceptions(logger=logger)
    async def _run(self) -> None:
        data_chan = aio.Chan[npt.NDArray[np.int16]]()
        overlap_speech_started: bool = False
        cache: _BoundedCache[int, dict[str, Any]] = _BoundedCache(
            max_len=5, default_key=0, default_value={}
        )

        @log_exceptions(logger=logger)
        async def _forward_data() -> None:
            nonlocal data_chan
            nonlocal overlap_speech_started
            nonlocal cache

            agent_speech_started: bool = False
            inference_s16_data = np.zeros(self._model._opts.max_window_size, dtype=np.int16)
            start_idx: int = 0
            accumulated_samples: int = 0

            async for input_frame in self._input_ch:
                # start accumulating user speech when agent starts speaking
                if isinstance(input_frame, BargeinStreamBase._AgentSpeechStartedSentinel):
                    agent_speech_started = True
                    overlap_speech_started = False
                    accumulated_samples = 0
                    start_idx = 0
                    continue

                # reset state when agent stops speaking
                if isinstance(input_frame, BargeinStreamBase._AgentSpeechEndedSentinel):
                    agent_speech_started = False
                    overlap_speech_started = False
                    accumulated_samples = 0
                    start_idx = 0
                    cache.clear()
                    continue

                # start inferencing against the overlap speech
                if (
                    isinstance(input_frame, BargeinStreamBase._OverlapSpeechStartedSentinel)
                    and agent_speech_started
                ):
                    logger.debug("overlap speech started, starting barge-in inference")
                    overlap_speech_started = True
                    accumulated_samples = 0
                    shift_size = min(
                        start_idx,
                        int(input_frame._speech_duration * self._sample_rate)
                        + self._model._opts.prefix_size,
                    )
                    inference_s16_data[:shift_size] = inference_s16_data[
                        start_idx - shift_size : start_idx
                    ]
                    start_idx = shift_size
                    cache.clear()
                    continue

                if isinstance(input_frame, BargeinStreamBase._OverlapSpeechEndedSentinel):
                    if overlap_speech_started:
                        _, last_request = cache.pop(lambda x: x.get("total_duration", 0) > 0)
                        last_request = last_request or {}
                        if not last_request:
                            logger.debug("no request made for overlap speech")
                        probas = last_request.get("probabilities", [])
                        total_duration = last_request.get("total_duration", 0.0)
                        ev = BargeinEvent(
                            type=BargeinEventType.OVERLAP_SPEECH_ENDED,
                            timestamp=time.time(),
                            overlap_speech_started_at=self._overlap_speech_started_at,
                            speech_input=last_request.get("speech_input", None),
                            probabilities=np.array(probas, dtype=np.float32) if probas else None,
                            total_duration=total_duration,
                            detection_delay=time.time()
                            - self._overlap_speech_started_at
                            + total_duration,
                        )
                        self._event_ch.send_nowait(ev)
                        self._bargein_detector.emit("overlap_speech_ended", ev)
                    overlap_speech_started = False
                    accumulated_samples = 0
                    start_idx = 0
                    continue

                if isinstance(input_frame, BargeinStreamBase._FlushSentinel):
                    continue

                if not agent_speech_started or not isinstance(input_frame, rtc.AudioFrame):
                    continue

                start_idx, samples_written = _write_to_inference_s16_data(
                    input_frame,
                    start_idx,
                    inference_s16_data,
                    self._model._opts.max_window_size,
                )
                accumulated_samples += samples_written
                if accumulated_samples >= self._model._opts.step_size and overlap_speech_started:
                    data_chan.send_nowait(inference_s16_data[:start_idx].copy())
                    accumulated_samples = 0

        @log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal overlap_speech_started
            nonlocal cache
            async for data in data_chan:
                resp = await self.predict(data)
                is_bargein = resp["is_bargein"]
                total_duration = (time.perf_counter_ns() - resp["created_at"]) / 1e9
                cache[resp["created_at"]] = {
                    "total_duration": total_duration,
                    "speech_input": data,
                } | resp
                if overlap_speech_started and is_bargein:
                    ev = BargeinEvent(
                        type=BargeinEventType.BARGEIN,
                        timestamp=time.time(),
                        overlap_speech_started_at=self._overlap_speech_started_at,
                        total_duration=total_duration,
                        is_bargein=is_bargein,
                        speech_input=data,
                        probabilities=np.array(resp.get("probabilities", []), dtype=np.float32)
                        if resp.get("probabilities", [])
                        else None,
                        detection_delay=time.time()
                        - self._overlap_speech_started_at
                        + total_duration,
                    )
                    self._event_ch.send_nowait(ev)
                    self._bargein_detector.emit("bargein_detected", ev)
                    overlap_speech_started = False

        tasks = [
            asyncio.create_task(_forward_data()),
            asyncio.create_task(_send_task()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            data_chan.close()

    async def predict(self, waveform: np.ndarray) -> dict[str, Any]:
        created_at = perf_counter_ns()
        async with http_context.http_session().post(
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
                #     "probabilities": list[float], optional
                # }
                logger.debug(
                    "inference done",
                    extra={
                        "created_at": created_at,
                        "is_bargein": data["is_bargein"],
                        "prediction_duration": (time.perf_counter_ns() - created_at) / 1e9,
                    },
                )
                return data
            except Exception as e:
                msg = await resp.text()
                logger.error("error during bargein prediction", extra={"response": msg})
                raise APIError(f"error during bargein prediction: {e}", body=msg) from e


class BargeinWebSocketStream(BargeinStreamBase):
    def __init__(
        self, *, bargein_detector: BargeinDetector, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(bargein_detector=bargein_detector, conn_options=conn_options)
        self._model: BargeinDetector = bargein_detector
        self._opts = bargein_detector._opts
        self._session = bargein_detector._ensure_session()
        self._request_id = str(shortuuid("bargein_request_"))
        self._reconnect_event = asyncio.Event()

    def update_options(
        self,
        *,
        threshold: NotGivenOr[float] = NOT_GIVEN,
        min_bargein_duration: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(threshold):
            self._opts.threshold = threshold
        if is_given(min_bargein_duration):
            self._opts.min_frames = math.ceil(min_bargein_duration * 40)
        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False
        overlap_speech_started: bool = False
        cache: _BoundedCache[int, dict[str, Any]] = _BoundedCache(
            max_len=5, default_key=0, default_value={}
        )

        @log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            nonlocal overlap_speech_started
            nonlocal cache

            agent_speech_started: bool = False
            inference_s16_data = np.zeros(self._model._opts.max_window_size, dtype=np.int16)
            start_idx: int = 0
            accumulated_samples: int = 0

            async for input_frame in self._input_ch:
                # start accumulating user speech when agent starts speaking
                if isinstance(input_frame, BargeinStreamBase._AgentSpeechStartedSentinel):
                    agent_speech_started = True
                    overlap_speech_started = False
                    accumulated_samples = 0
                    start_idx = 0
                    continue

                # reset state when agent stops speaking
                if isinstance(input_frame, BargeinStreamBase._AgentSpeechEndedSentinel):
                    agent_speech_started = False
                    overlap_speech_started = False
                    accumulated_samples = 0
                    start_idx = 0
                    cache.clear()
                    continue

                # start inferencing against the overlap speech
                if (
                    isinstance(input_frame, BargeinStreamBase._OverlapSpeechStartedSentinel)
                    and agent_speech_started
                ):
                    logger.debug("overlap speech started, starting barge-in inference")
                    overlap_speech_started = True
                    accumulated_samples = 0
                    shift_size = min(
                        start_idx,
                        int(input_frame._speech_duration * self._sample_rate)
                        + self._model._opts.prefix_size,
                    )
                    inference_s16_data[:shift_size] = inference_s16_data[
                        start_idx - shift_size : start_idx
                    ]
                    start_idx = shift_size
                    cache.clear()
                    continue

                if isinstance(input_frame, BargeinStreamBase._OverlapSpeechEndedSentinel):
                    if overlap_speech_started:
                        logger.debug("overlap speech ended, stopping barge-in inference")
                        # only pop the last complete request
                        _, last_request = cache.pop(lambda x: x.get("total_duration", 0) > 0)
                        last_request = last_request or {}
                        if not last_request:
                            logger.debug("no request made for overlap speech")
                        total_duration = last_request.get("total_duration", 0.0)
                        speech_input = last_request.get("speech_input", None)
                        probas = last_request.get("probabilities", [])
                        ev = BargeinEvent(
                            type=BargeinEventType.OVERLAP_SPEECH_ENDED,
                            timestamp=time.time(),
                            overlap_speech_started_at=self._overlap_speech_started_at,
                            speech_input=speech_input,
                            probabilities=np.array(probas, dtype=np.float32) if probas else None,
                            total_duration=total_duration,
                            detection_delay=time.time()
                            - self._overlap_speech_started_at
                            + total_duration,
                        )
                        self._event_ch.send_nowait(ev)
                        self._bargein_detector.emit("overlap_speech_ended", ev)
                    overlap_speech_started = False
                    accumulated_samples = 0
                    start_idx = 0
                    continue

                if isinstance(input_frame, BargeinStreamBase._FlushSentinel):
                    continue

                if not agent_speech_started or not isinstance(input_frame, rtc.AudioFrame):
                    continue

                assert input_frame.sample_rate == self._model._opts.sample_rate, (
                    f"sample rate mismatch: {input_frame.sample_rate} != {self._model._opts.sample_rate}"
                )

                start_idx, samples_written = _write_to_inference_s16_data(
                    input_frame,
                    start_idx,
                    inference_s16_data,
                    self._model._opts.max_window_size,
                )
                accumulated_samples += samples_written
                if accumulated_samples >= self._model._opts.step_size and overlap_speech_started:
                    created_at = perf_counter_ns()
                    header = struct.pack("<Q", created_at)  # 8 bytes
                    await ws.send_bytes(header + inference_s16_data[:start_idx].tobytes())
                    logger.debug(
                        "sending inference data to LiveKit Bargein",
                        extra={
                            "created_at": created_at,
                            "samples": len(inference_s16_data[:start_idx]),
                        },
                    )
                    cache[created_at] = {
                        "speech_input": inference_s16_data[:start_idx].copy(),
                    }
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
                        message=f"LiveKit Bargein connection closed unexpectedly: {msg.data}"
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected LiveKit Bargein message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                msg_type = data["type"]

                if msg_type == MSG_SESSION_CREATED:
                    pass
                elif msg_type == MSG_BARGEIN_DETECTED:
                    created_at = int(data["created_at"])
                    if overlap_speech_started:
                        total_duration = (perf_counter_ns() - created_at) / 1e9
                        prediction_duration = data.get("prediction_duration", 0.0)
                        probas = data.get("probabilities", [])
                        speech_input = cache.get(created_at, {}).get("speech_input", None)
                        cache[created_at] = {
                            "total_duration": total_duration,
                            "speech_input": speech_input,
                            "probabilities": data.get("probabilities", []),
                            "is_bargein": True,
                            "created_at": created_at,
                        }
                        logger.debug(
                            "bargein detected",
                            extra={
                                "total_duration": total_duration,
                                "prediction_duration": prediction_duration,
                                "samples": len(speech_input) if speech_input is not None else 0,
                            },
                        )
                        ev = BargeinEvent(
                            type=BargeinEventType.BARGEIN,
                            timestamp=time.time(),
                            is_bargein=True,
                            total_duration=total_duration,
                            overlap_speech_started_at=self._overlap_speech_started_at,
                            speech_input=cache[created_at].get("speech_input", None),
                            probabilities=np.array(probas, dtype=np.float32) if probas else None,
                            detection_delay=time.time()
                            - self._overlap_speech_started_at
                            + total_duration,
                        )
                        self._event_ch.send_nowait(ev)
                        self._bargein_detector.emit("bargein_detected", ev)
                        overlap_speech_started = False
                elif msg_type == MSG_INFERENCE_DONE:
                    created_at = int(data["created_at"])
                    total_duration = (perf_counter_ns() - created_at) / 1e9
                    prediction_duration = data.get("prediction_duration", 0.0)
                    speech_input = cache.get(created_at, {}).get("speech_input", None)
                    if speech_input is not None and len(speech_input) > 0:
                        logger.debug(
                            "inference done",
                            extra={
                                "total_duration": total_duration,
                                "prediction_duration": prediction_duration,
                                "samples": len(speech_input),
                            },
                        )
                        cache[created_at] = {
                            "total_duration": total_duration,
                            "speech_input": speech_input,
                            "probabilities": data.get("probabilities", []),
                            "is_bargein": data.get("is_bargein", False),
                            "created_at": created_at,
                        }
                    else:
                        logger.debug(
                            "inference done but cache expired",
                            extra={
                                "created_at": created_at,
                                "total_duration": total_duration,
                                "prediction_duration": prediction_duration,
                            },
                        )
                elif msg_type == MSG_SESSION_CLOSED:
                    pass
                elif msg_type == MSG_ERROR:
                    raise APIError(f"LiveKit Bargein returned error: {msg.data}")
                else:
                    logger.warning("received unexpected message from LiveKit Bargein: %s", data)

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
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
                    await aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    try:
                        tasks_group.exception()
                    except asyncio.CancelledError:
                        pass
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Connect to the LiveKit Bargein WebSocket."""
        params: dict[str, Any] = {
            "settings": {
                "sample_rate": self._opts.sample_rate,
                "num_channels": 1,
                "threshold": self._model._opts.threshold,
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
            params["type"] = MSG_SESSION_CREATE
            await ws.send_str(json.dumps(params))
        except (
            aiohttp.ClientConnectorError,
            asyncio.TimeoutError,
            aiohttp.ClientResponseError,
        ) as e:
            if isinstance(e, aiohttp.ClientResponseError) and e.status == 429:
                raise APIStatusError("LiveKit Bargein quota exceeded", status_code=e.status) from e
            raise APIConnectionError("failed to connect to LiveKit Bargein") from e
        return ws


def _write_to_inference_s16_data(
    frame: rtc.AudioFrame, start_idx: int, out_data: np.ndarray, max_window_size: int
) -> tuple[int, int]:
    """Write the audio frame to the output data array and return the new start index and the number of samples written."""

    if frame.samples_per_channel > out_data.shape[0]:
        raise ValueError("frame samples are greater than the max window size")

    # shift the data to the left if the window would overflow
    if (shift := start_idx + frame.samples_per_channel - max_window_size) > 0:
        out_data[: start_idx - shift] = out_data[shift:start_idx]
        start_idx -= shift

    slice_ = out_data[start_idx : start_idx + frame.samples_per_channel]
    if frame.num_channels > 1:
        arr_i16 = np.frombuffer(
            frame.data, dtype=np.int16, count=frame.samples_per_channel * frame.num_channels
        ).reshape(-1, frame.num_channels)
        slice_[:] = (np.sum(arr_i16, axis=1, dtype=np.int32) // frame.num_channels).astype(np.int16)
    else:
        slice_[:] = frame.data
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
