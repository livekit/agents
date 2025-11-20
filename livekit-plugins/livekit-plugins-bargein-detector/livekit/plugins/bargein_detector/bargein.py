from __future__ import annotations

import base64
import math
import os
import time
from time import perf_counter

import aiohttp
import numpy as np

from livekit import agents, rtc
from livekit.agents import get_job_context, utils
from livekit.agents.bargein import (
    BargeinDetector as BargeinDetectorBase,
    BargeinEvent,
    BargeinEventType,
    BargeinStream as BargeinStreamBase,
)
from livekit.agents.inference._utils import create_access_token

from .log import logger

MAX_WINDOW_SIZE = 3 * 16000  # 3 seconds at 16000 Hz
STEP_SIZE = int(0.2 * 16000)  # 0.2 second at 16000 Hz
REMOTE_INFERENCE_TIMEOUT = 1
DEFAULT_BASE_URL = "https://agent-gateway.livekit.cloud/v1"


class BargeinDetector(BargeinDetectorBase):
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        threshold: float = 0.95,
        min_bargein_duration: float = 0.05,
        max_window_size: int = MAX_WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        inference_timeout: float = REMOTE_INFERENCE_TIMEOUT,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        super().__init__()
        self._sample_rate: int = sample_rate
        self._max_window_size: int = max_window_size
        self._step_size: int = step_size
        self._inference_timeout: float = inference_timeout
        self._threshold: float = threshold
        self._min_frames: int = math.ceil(min_bargein_duration * 40)  # 40 frames per second

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

        self._base_url: str = lk_base_url
        self._api_key: str = lk_api_key
        self._api_secret: str = lk_api_secret

    @property
    def model(self) -> str:
        return "bargein"

    @property
    def provider(self) -> str:
        return "livekit"

    @staticmethod
    def encode_waveform(array: np.ndarray) -> str:
        return base64.b64encode(array.tobytes()).decode("utf-8")

    def stream(self) -> BargeinStream:
        return BargeinStream(bargein_detector=self)


class BargeinStream(BargeinStreamBase):
    def __init__(self, *, bargein_detector: BargeinDetector) -> None:
        super().__init__(bargein_detector=bargein_detector)
        self._model: BargeinDetector = bargein_detector
        self._input_sample_rate: int | None = None

    @agents.utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        inference_f32_data = np.zeros(self._model._max_window_size, dtype=np.float32)
        inference_frames: list[rtc.AudioFrame] = []
        resampler: rtc.AudioResampler | None = None

        start_idx: int = 0
        accumulated_samples: int = 0
        agent_speech_started: bool = False
        overlap_speech_started: bool = False

        @agents.utils.log_exceptions(logger=logger)
        def write_to_inference_f32_data(frame: rtc.AudioFrame) -> tuple[int, int]:
            nonlocal start_idx
            nonlocal inference_f32_data

            if frame.samples_per_channel > inference_f32_data.shape[0]:
                raise ValueError("frame samples are greater than the max window size")

            # shift the data to the left if the window would overflow
            if (shift := start_idx + frame.samples_per_channel - self._model._max_window_size) > 0:
                inference_f32_data[: start_idx - shift] = inference_f32_data[shift:start_idx]
                start_idx -= shift

            # convert data to f32
            slice_ = inference_f32_data[start_idx : start_idx + frame.samples_per_channel]
            if frame.num_channels > 1:
                np.sum(frame.data, axis=1, dtype=np.float32, out=slice_)
            else:
                slice_[:] = frame.data
            slice_ /= np.iinfo(np.int16).max * frame.num_channels
            start_idx += frame.samples_per_channel
            return start_idx, frame.samples_per_channel

        async for input_frame in self._input_ch:
            # start accumulating user speech
            if isinstance(input_frame, BargeinStreamBase._AgentSpeechStartedSentinel):
                agent_speech_started = True
                overlap_speech_started = False
                inference_frames = []
                accumulated_samples = 0
                start_idx = 0
                continue

            # end of agent speech, reset state
            if isinstance(input_frame, BargeinStreamBase._AgentSpeechEndedSentinel):
                agent_speech_started = False
                overlap_speech_started = False
                inference_frames = []
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

            if not self._input_sample_rate:
                self._input_sample_rate = input_frame.sample_rate
                if self._input_sample_rate != self._model._sample_rate:
                    resampler = rtc.AudioResampler(
                        input_rate=self._input_sample_rate,
                        output_rate=self._model._sample_rate,
                        quality=rtc.AudioResamplerQuality.QUICK,
                    )

            elif self._input_sample_rate != input_frame.sample_rate:
                logger.error("a frame with another sample rate was already pushed")
                continue

            if resampler is not None:
                inference_frames.extend(resampler.push(input_frame))
            else:
                inference_frames.append(input_frame)

            for i, frame in enumerate(inference_frames):
                start_time = time.perf_counter()
                end_idx, samples_written = write_to_inference_f32_data(frame)
                accumulated_samples += samples_written
                if accumulated_samples >= self._model._step_size and overlap_speech_started:
                    is_bargein = await self.predict(inference_f32_data[:end_idx])
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
                        inference_frames = inference_frames[i:]
                        self._bargein_detector.emit("bargein_detected", ev)
                        overlap_speech_started = False
                        break
            else:
                inference_frames = []

    async def predict(self, waveform: np.ndarray) -> bool:
        started_at = perf_counter()
        ctx = get_job_context()
        request = {
            "jobId": ctx.job.id,
            "workerId": ctx.worker_id,
            "waveform": self._model.encode_waveform(waveform),
            "threshold": self._model._threshold,
            "min_frames": self._model._min_frames,
        }
        agent_id = os.getenv("LIVEKIT_AGENT_ID")
        if agent_id:
            request["agentId"] = agent_id

        async with utils.http_context.http_session().post(
            url=f"{self._model._base_url}/bargein",
            headers={
                "Authorization": f"Bearer {create_access_token(self._model._api_key, self._model._api_secret)}",
            },
            json=request,
            timeout=aiohttp.ClientTimeout(total=self._model._inference_timeout),
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
