from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum, unique
from time import perf_counter_ns
from typing import Any, Literal, Union

import aiohttp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc

from .._exceptions import APIConnectionError, APIError, APIStatusError
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import aio, http_context, is_given, log_exceptions, shortuuid
from ._utils import create_access_token

SAMPLE_RATE = 16000
THRESHOLD = 0.95
MIN_BARGEIN_DURATION = 0.05  # 25ms per frame
MAX_WINDOW_SIZE = 3 * 16000  # 3 seconds at 16000 Hz
STEP_SIZE = int(0.1 * 16000)  # 0.1 second at 16000 Hz
REMOTE_INFERENCE_TIMEOUT = 1
DEFAULT_BASE_URL = "https://agent-gateway.livekit.cloud/v1"

# WebSocket input message types
MSG_INPUT_AUDIO = "input_audio"
MSG_SESSION_CREATE = "session.create"
MSG_SESSION_CLOSE = "session.close"

# WebSocket output message types
MSG_SESSION_CREATED = "session.created"
MSG_SESSION_CLOSED = "session.closed"
MSG_BARGEIN_DETECTED = "bargein_detected"
MSG_ERROR = "error"


@unique
class BargeinEventType(str, Enum):
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
    """Whether bargein is detected."""

    inference_duration: float = 0.0
    """Time taken to perform the inference, in seconds."""

    overlap_speech_started_at: float | None = None
    """Timestamp (in seconds) when the overlap speech started. Useful for emitting held transcripts."""


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
    threshold: float
    min_frames: int
    max_window_size: int
    step_size: int
    inference_timeout: float
    base_url: str
    api_key: str
    api_secret: str
    use_proxy: bool


class BargeinDetector(
    rtc.EventEmitter[Literal["bargein_detected", "error"]],
):
    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = THRESHOLD,
        min_bargein_duration: float = MIN_BARGEIN_DURATION,
        max_window_size: int = MAX_WINDOW_SIZE,
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

        lk_api_key: str = ""
        lk_api_secret: str = ""
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

        self._opts = BargeinOptions(
            sample_rate=sample_rate,
            threshold=threshold,
            min_frames=math.ceil(min_bargein_duration * 40),  # 40 frames per second
            max_window_size=max_window_size,
            step_size=step_size,
            inference_timeout=inference_timeout,
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
            use_proxy=use_proxy if is_given(use_proxy) else lk_base_url == DEFAULT_BASE_URL,
        )
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._sample_rate = sample_rate
        self._session = http_session
        self._streams = weakref.WeakSet[Union[BargeinHttpStream, BargeinWebSocketStream]]()

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

    @staticmethod
    def encode_waveform(array: np.ndarray) -> str:
        return base64.b64encode(array.tobytes()).decode("utf-8")

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
                self._overlap_speech_started_at = time.time()
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
        data_chan = aio.Chan[np.ndarray]()
        overlap_speech_started: bool = False

        @log_exceptions(logger=logger)
        async def _forward_data() -> None:
            nonlocal data_chan
            nonlocal overlap_speech_started

            agent_speech_started: bool = False
            inference_f32_data = np.zeros(self._model._opts.max_window_size, dtype=np.float32)
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
                    continue

                # start inferencing against the overlap speech
                if (
                    isinstance(input_frame, BargeinStreamBase._OverlapSpeechStartedSentinel)
                    and agent_speech_started
                ):
                    logger.debug("overlap speech started, starting barge-in inference")
                    overlap_speech_started = True
                    continue

                if isinstance(input_frame, BargeinStreamBase._OverlapSpeechEndedSentinel):
                    if overlap_speech_started:
                        logger.debug("overlap speech ended, stopping barge-in inference")
                    overlap_speech_started = False
                    continue

                if isinstance(input_frame, BargeinStreamBase._FlushSentinel):
                    continue

                if not agent_speech_started or not isinstance(input_frame, rtc.AudioFrame):
                    continue

                start_idx, samples_written = _write_to_inference_f32_data(
                    input_frame,
                    start_idx,
                    inference_f32_data,
                    self._model._opts.max_window_size,
                )
                accumulated_samples += samples_written
                if accumulated_samples >= self._model._opts.step_size and overlap_speech_started:
                    data_chan.send_nowait(inference_f32_data[:start_idx])
                    accumulated_samples = 0

        @log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal overlap_speech_started
            async for data in data_chan:
                start_time = time.perf_counter()
                is_bargein = await self.predict(data)
                inference_duration = time.perf_counter() - start_time
                if overlap_speech_started and is_bargein:
                    ev = BargeinEvent(
                        type=BargeinEventType.BARGEIN,
                        timestamp=time.time(),
                        overlap_speech_started_at=self._overlap_speech_started_at,
                        inference_duration=inference_duration,
                        is_bargein=is_bargein,
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

    async def predict(self, waveform: np.ndarray) -> bool:
        created_at = perf_counter_ns()
        request = {
            "waveform": self._model.encode_waveform(waveform),
            "threshold": self._opts.threshold,
            "min_frames": self._opts.min_frames,
            "created_at": created_at,
        }
        async with http_context.http_session().post(
            url=f"{self._opts.base_url}/bargein",
            headers={
                "Authorization": f"Bearer {create_access_token(self._opts.api_key, self._opts.api_secret)}",
            },
            json=request,
            timeout=aiohttp.ClientTimeout(total=self._opts.inference_timeout),
        ) as resp:
            try:
                resp.raise_for_status()
                data = await resp.json()
                is_bargein: bool | None = data.get("is_bargein")
                inference_duration = (perf_counter_ns() - created_at) / 1e9
                if isinstance(is_bargein, bool):
                    logger.debug(
                        "bargein prediction",
                        extra={
                            "is_bargein": is_bargein,
                            "duration": inference_duration,
                        },
                    )
                    return is_bargein
            except Exception as e:
                msg = await resp.text()
                logger.error("error during bargein prediction", extra={"response": msg})
                raise APIError(f"error during bargein prediction: {e}", body=msg) from e
            return False


class BargeinWebSocketStream(BargeinStreamBase):
    def __init__(
        self, *, bargein_detector: BargeinDetector, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(bargein_detector=bargein_detector, conn_options=conn_options)
        self._model: BargeinDetector = bargein_detector
        self._opts = bargein_detector._opts
        self._session = bargein_detector._ensure_session()
        self._request_id = str(shortuuid("bargein_request_"))

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

    async def _run(self) -> None:
        closing_ws = False
        overlap_speech_started: bool = False

        @log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            nonlocal overlap_speech_started

            agent_speech_started: bool = False
            inference_f32_data = np.zeros(self._model._opts.max_window_size, dtype=np.float32)
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
                    continue

                # start inferencing against the overlap speech
                if (
                    isinstance(input_frame, BargeinStreamBase._OverlapSpeechStartedSentinel)
                    and agent_speech_started
                ):
                    logger.debug("overlap speech started, starting barge-in inference")
                    overlap_speech_started = True
                    continue

                if isinstance(input_frame, BargeinStreamBase._OverlapSpeechEndedSentinel):
                    if overlap_speech_started:
                        logger.debug("overlap speech ended, stopping barge-in inference")
                    overlap_speech_started = False
                    continue

                if isinstance(input_frame, BargeinStreamBase._FlushSentinel):
                    continue

                if not agent_speech_started or not isinstance(input_frame, rtc.AudioFrame):
                    continue

                assert input_frame.sample_rate == self._model._opts.sample_rate, (
                    f"sample rate mismatch: {input_frame.sample_rate} != {self._model._opts.sample_rate}"
                )

                start_idx, samples_written = _write_to_inference_f32_data(
                    input_frame,
                    start_idx,
                    inference_f32_data,
                    self._model._opts.max_window_size,
                )
                accumulated_samples += samples_written
                if accumulated_samples >= self._model._opts.step_size and overlap_speech_started:
                    msg = {
                        "type": MSG_INPUT_AUDIO,
                        "audio": self._model.encode_waveform(inference_f32_data[:start_idx]),
                        "created_at": perf_counter_ns(),
                    }
                    await ws.send_str(json.dumps(msg))
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

            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return
                    raise APIStatusError(message="LiveKit Bargein connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected LiveKit Bargein message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                msg_type = data.get("type")
                created_at = data.get("created_at", 0.0)
                if msg_type == MSG_SESSION_CREATED:
                    pass
                elif msg_type == MSG_BARGEIN_DETECTED:
                    if overlap_speech_started:
                        logger.debug("bargein detected")
                        inference_duration = (perf_counter_ns() - created_at) / 1e9
                        ev = BargeinEvent(
                            type=BargeinEventType.BARGEIN,
                            timestamp=time.time(),
                            is_bargein=True,
                            inference_duration=inference_duration,
                            overlap_speech_started_at=self._overlap_speech_started_at,
                        )
                        self._event_ch.send_nowait(ev)
                        self._bargein_detector.emit("bargein_detected", ev)
                        overlap_speech_started = False
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
                try:
                    await asyncio.gather(*tasks)
                finally:
                    await aio.gracefully_cancel(*tasks)
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Connect to the LiveKit STT WebSocket."""
        params: dict[str, Any] = {
            "settings": {
                "sample_rate": self._opts.sample_rate,
                "num_channels": 1,
                "threshold": self._model._opts.threshold,
                "min_frames": self._model._opts.min_frames,
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
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            if isinstance(e, aiohttp.ClientResponseError) and e.status == 429:
                raise APIStatusError("LiveKit Bargein quota exceeded", status_code=e.status) from e
            raise APIConnectionError("failed to connect to LiveKit Bargein") from e
        return ws


def _write_to_inference_f32_data(
    frame: rtc.AudioFrame, start_idx: int, out_data: np.ndarray, max_window_size: int
) -> tuple[int, int]:
    """Write the audio frame to the output data array and return the new start index and the number of samples written."""

    if frame.samples_per_channel > out_data.shape[0]:
        raise ValueError("frame samples are greater than the max window size")

    # shift the data to the left if the window would overflow
    if (shift := start_idx + frame.samples_per_channel - max_window_size) > 0:
        out_data[: start_idx - shift] = out_data[shift:start_idx]
        start_idx -= shift

    # convert data to f32
    slice_ = out_data[start_idx : start_idx + frame.samples_per_channel]
    if frame.num_channels > 1:
        np.sum(frame.data, axis=1, dtype=np.float32, out=slice_)
    else:
        slice_[:] = frame.data
    slice_ /= np.iinfo(np.int16).max * frame.num_channels
    start_idx += frame.samples_per_channel
    return start_idx, frame.samples_per_channel
