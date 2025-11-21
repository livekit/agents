from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import time
import weakref
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import aiohttp
import numpy as np

from livekit import rtc

from .. import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    NotGivenOr,
    get_job_context,
    utils,
)
from ..bargein import (
    BargeinDetector as BargeinDetectorBase,
    BargeinEvent,
    BargeinEventType,
    BargeinStream as BargeinStreamBase,
)
from ..log import logger
from ..utils import aio, is_given
from ._utils import create_access_token

SAMPLE_RATE = 16000
THRESHOLD = 0.95
MIN_BARGEIN_DURATION = 0.05  # 25ms per frame
MAX_WINDOW_SIZE = 3 * 16000  # 3 seconds at 16000 Hz
STEP_SIZE = int(0.2 * 16000)  # 0.2 second at 16000 Hz
REMOTE_INFERENCE_TIMEOUT = 1
DEFAULT_BASE_URL = "https://agent-gateway.livekit.cloud/v1"


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


class BargeinDetector(BargeinDetectorBase):
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
        super().__init__(sample_rate=sample_rate)

        lk_base_url = (
            base_url
            if base_url
            else os.environ.get("LIVEKIT_BARGEIN_INFERENCE_URL", DEFAULT_BASE_URL)
        )

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
            use_proxy=use_proxy if utils.is_given(use_proxy) else lk_base_url == DEFAULT_BASE_URL,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[BargeinStreamBase]()

    @property
    def model(self) -> str:
        return "bargein"

    @property
    def provider(self) -> str:
        return "livekit"

    @staticmethod
    def encode_waveform(array: np.ndarray) -> str:
        return base64.b64encode(array.tobytes()).decode("utf-8")

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> BargeinHttpStream | BargeinWebSocketStream:
        if self._opts.use_proxy:
            stream = BargeinWebSocketStream(bargein_detector=self, conn_options=conn_options)
        else:
            stream = BargeinHttpStream(bargein_detector=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        threshold: float = NOT_GIVEN,
        min_frames: int = NOT_GIVEN,
    ) -> None:
        if is_given(threshold):
            self._opts.threshold = threshold
        if is_given(min_frames):
            self._opts.min_frames = min_frames

        for stream in self._streams:
            stream.update_options(threshold=threshold, min_frames=min_frames)


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
        threshold: float = NOT_GIVEN,
        min_frames: int = NOT_GIVEN,
    ) -> None:
        if is_given(threshold):
            self._opts.threshold = threshold
        if is_given(min_frames):
            self._opts.min_frames = min_frames

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        data_chan = aio.Chan[np.ndarray]()
        overlap_speech_started: bool = False

        @utils.log_exceptions(logger=logger)
        async def _forward_data() -> None:
            nonlocal data_chan
            nonlocal overlap_speech_started

            agent_speech_started: bool = False
            inference_f32_data = np.zeros(self._model._opts.max_window_size, dtype=np.float32)
            start_idx: int = 0
            accumulated_samples: int = 0

            async for input_frame in self._input_ch:
                # start accumulating user speech
                if isinstance(input_frame, BargeinStreamBase._AgentSpeechStartedSentinel):
                    agent_speech_started = True
                    overlap_speech_started = False
                    accumulated_samples = 0
                    start_idx = 0
                    continue

                # end of agent speech, reset state
                if isinstance(input_frame, BargeinStreamBase._AgentSpeechEndedSentinel):
                    agent_speech_started = False
                    overlap_speech_started = False
                    accumulated_samples = 0
                    start_idx = 0
                    continue

                # start inferencing against the overlap speech
                if agent_speech_started and isinstance(
                    input_frame, BargeinStreamBase._OverlapSpeechStartedSentinel
                ):
                    logger.debug("overlap speech started")
                    overlap_speech_started = True
                    continue

                if isinstance(input_frame, BargeinStreamBase._OverlapSpeechEndedSentinel):
                    logger.debug("overlap speech ended")
                    overlap_speech_started = False
                    continue

                if isinstance(input_frame, BargeinStreamBase._FlushSentinel):
                    continue

                if not agent_speech_started:
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

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal overlap_speech_started
            async for data in data_chan:
                start_time = time.perf_counter()
                is_bargein = await self.predict(data)
                inference_duration = time.perf_counter() - start_time
                self._event_ch.send_nowait(
                    BargeinEvent(
                        type=BargeinEventType.INFERENCE_DONE,
                        timestamp=time.time(),
                        is_bargein=is_bargein,
                        inference_duration=inference_duration,
                    )
                )

                if is_bargein:
                    ev = BargeinEvent(
                        type=BargeinEventType.BARGEIN,
                        timestamp=time.time(),
                    )
                    self._event_ch.send_nowait(ev)
                    self._bargein_detector.emit("bargein_detected", ev)
                    overlap_speech_started = False

        tasks = [
            asyncio.create_task(_send_task()),
            asyncio.create_task(_forward_data()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            data_chan.close()

    async def predict(self, waveform: np.ndarray) -> bool:
        started_at = perf_counter()
        ctx = get_job_context()
        request = {
            "jobId": ctx.job.id,
            "workerId": ctx.worker_id,
            "waveform": self._model.encode_waveform(waveform),
            "threshold": self._opts.threshold,
            "min_frames": self._opts.min_frames,
        }
        agent_id = os.getenv("LIVEKIT_AGENT_ID")
        if agent_id:
            request["agentId"] = agent_id

        async with utils.http_context.http_session().post(
            url=f"{self._opts.base_url}/bargein",
            headers={
                "Authorization": f"Bearer {create_access_token(self._opts.api_key, self._opts.api_secret)}",
            },
            json=request,
            timeout=aiohttp.ClientTimeout(total=self._opts.inference_timeout),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            is_bargein: bool | None = data.get("is_bargein")
            if isinstance(is_bargein, bool):
                logger.debug(
                    "bargein prediction",
                    extra={
                        "is_bargein": is_bargein,
                        "duration": perf_counter() - started_at,
                    },
                )
                return is_bargein
            return False


class BargeinWebSocketStream(BargeinStreamBase):
    def __init__(
        self, *, bargein_detector: BargeinDetector, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(bargein_detector=bargein_detector, conn_options=conn_options)
        self._model: BargeinDetector = bargein_detector
        self._opts = bargein_detector._opts
        self._session = bargein_detector._ensure_session()
        self._request_id = str(utils.shortuuid("bargein_request_"))

    def update_options(
        self,
        *,
        threshold: float = NOT_GIVEN,
        min_frames: int = NOT_GIVEN,
    ) -> None:
        if is_given(threshold):
            self._opts.threshold = threshold
        if is_given(min_frames):
            self._opts.min_frames = min_frames

    async def _run(self) -> None:
        closing_ws = False
        overlap_speech_started: bool = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            nonlocal overlap_speech_started

            agent_speech_started: bool = False
            inference_f32_data = np.zeros(self._model._opts.max_window_size, dtype=np.float32)
            start_idx: int = 0
            accumulated_samples: int = 0

            async for input_frame in self._input_ch:
                # start accumulating user speech
                if isinstance(input_frame, BargeinStreamBase._AgentSpeechStartedSentinel):
                    agent_speech_started = True
                    overlap_speech_started = False
                    accumulated_samples = 0
                    start_idx = 0
                    continue

                # end of agent speech, reset state
                if isinstance(input_frame, BargeinStreamBase._AgentSpeechEndedSentinel):
                    agent_speech_started = False
                    overlap_speech_started = False
                    accumulated_samples = 0
                    start_idx = 0
                    continue

                # start inferencing against the overlap speech
                if agent_speech_started and isinstance(
                    input_frame, BargeinStreamBase._OverlapSpeechStartedSentinel
                ):
                    logger.debug("overlap speech started")
                    overlap_speech_started = True
                    continue

                if isinstance(input_frame, BargeinStreamBase._OverlapSpeechEndedSentinel):
                    logger.debug("overlap speech ended")
                    overlap_speech_started = False
                    continue

                if isinstance(input_frame, BargeinStreamBase._FlushSentinel):
                    continue

                if not agent_speech_started:
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
                        "type": "input_audio",
                        "audio": self._model.encode_waveform(inference_f32_data[:start_idx]),
                        "sample_rate": self._model._opts.sample_rate,
                        "num_channels": 1,
                        "threshold": self._model._opts.threshold,
                        "min_frames": self._model._opts.min_frames,
                        "created_at": time.time(),
                    }
                    await ws.send_str(json.dumps(msg))
                    accumulated_samples = 0

            closing_ws = True
            finalize_msg = {
                "type": "session.finalize",
            }
            await ws.send_str(json.dumps(finalize_msg))

        @utils.log_exceptions(logger=logger)
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
                if msg_type == "session.created":
                    pass
                elif msg_type == "inference_done":
                    is_bargein_result = data.get("is_bargein", False)
                    logger.debug("inference done", extra={"is_bargein": is_bargein_result})
                    self._event_ch.send_nowait(
                        BargeinEvent(
                            type=BargeinEventType.INFERENCE_DONE,
                            timestamp=time.time(),
                            is_bargein=is_bargein_result,
                            inference_duration=data.get("delta", 0.0),
                        )
                    )
                elif msg_type == "bargein_detected":
                    logger.debug("bargein detected")
                    ev = BargeinEvent(
                        type=BargeinEventType.BARGEIN,
                        timestamp=time.time(),
                    )
                    self._event_ch.send_nowait(ev)
                    self._bargein_detector.emit("bargein_detected", ev)
                    overlap_speech_started = False

                elif msg_type == "session.finalized":
                    pass
                elif msg_type == "session.closed":
                    pass
                elif msg_type == "error":
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
                    await utils.aio.gracefully_cancel(*tasks)
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Connect to the LiveKit STT WebSocket."""
        params: dict[str, Any] = {
            "settings": {
                "sample_rate": str(self._opts.sample_rate),
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
            params["type"] = "session.create"
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
