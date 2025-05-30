from __future__ import annotations

import os
import sys
from collections.abc import AsyncGenerator, AsyncIterator

import cv2
import numpy as np
from loguru import logger as _logger

from bithuman import AsyncBithuman  # type: ignore
from livekit import rtc
from livekit.agents import NOT_GIVEN, AgentSession, NotGivenOr, utils
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    QueueAudioOutput,
    VideoGenerator,
)

from .log import logger

_logger.remove()
_logger.add(sys.stdout, level="INFO")


class BitHumanException(Exception):
    """Exception for BitHuman errors"""


class AvatarSession:
    """A Beyond Presence avatar session"""

    def __init__(
        self,
        *,
        model_path: NotGivenOr[str | None] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        api_token: NotGivenOr[str] = NOT_GIVEN,
        runtime: NotGivenOr[AsyncBithuman | None] = NOT_GIVEN,
    ) -> None:
        self._api_url = api_url or (os.getenv("BITHUMAN_API_URL") or NOT_GIVEN)
        self._api_secret = api_secret or (os.getenv("BITHUMAN_API_SECRET") or NOT_GIVEN)
        self._api_token = api_token or (os.getenv("BITHUMAN_API_TOKEN") or NOT_GIVEN)
        self._model_path = model_path or (os.getenv("BITHUMAN_MODEL_PATH") or NOT_GIVEN)

        if self._api_secret is None and self._api_token is None:
            raise BitHumanException("BITHUMAN_API_SECRET or BITHUMAN_API_TOKEN must be set")
        if self._model_path is None:
            raise BitHumanException("BITHUMAN_MODEL_PATH must be set")

        self._avatar_runner: AvatarRunner | None = None
        self._runtime = runtime

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        if self._runtime:
            runtime = self._runtime
        else:
            kwargs = {
                "model_path": self._model_path,
            }
            if self._api_secret:
                kwargs["api_secret"] = self._api_secret
            if self._api_token:
                kwargs["token"] = self._api_token
            if self._api_url:
                kwargs["api_url"] = self._api_url

            runtime = await AsyncBithuman.create(**kwargs)
            await runtime.start()

        video_generator = BithumanGenerator(runtime)

        output_width, output_height = video_generator.video_resolution
        avatar_options = AvatarOptions(
            video_width=output_width,
            video_height=output_height,
            video_fps=video_generator.video_fps,
            audio_sample_rate=video_generator.audio_sample_rate,
            audio_channels=1,
        )

        audio_buffer = QueueAudioOutput(sample_rate=runtime.settings.INPUT_SAMPLE_RATE)
        # create avatar runner
        self._avatar_runner = AvatarRunner(
            room=room,
            video_gen=video_generator,
            audio_recv=audio_buffer,
            options=avatar_options,
        )
        await self._avatar_runner.start()

        agent_session.output.audio = audio_buffer


class BithumanGenerator(VideoGenerator):
    def __init__(self, runtime: AsyncBithuman):
        self._runtime = runtime

    @property
    def video_resolution(self) -> tuple[int, int]:
        frame = self._runtime.get_first_frame()
        if frame is None:
            raise ValueError("Failed to read frame")
        return frame.shape[1], frame.shape[0]

    @property
    def video_fps(self) -> int:
        return self._runtime.settings.FPS  # type: ignore

    @property
    def audio_sample_rate(self) -> int:
        return self._runtime.settings.INPUT_SAMPLE_RATE  # type: ignore

    @utils.log_exceptions(logger=logger)
    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            await self._runtime.flush()
            return
        await self._runtime.push_audio(bytes(frame.data), frame.sample_rate, last_chunk=False)

    def clear_buffer(self) -> None:
        self._runtime.interrupt()

    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._stream_impl()

    async def _stream_impl(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        def create_video_frame(image: np.ndarray) -> rtc.VideoFrame:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            return rtc.VideoFrame(
                width=image.shape[1],
                height=image.shape[0],
                type=rtc.VideoBufferType.RGBA,
                data=image.tobytes(),
            )

        async for frame in self._runtime.run():
            if frame.bgr_image is not None:
                video_frame = create_video_frame(frame.bgr_image)
                yield video_frame

            audio_chunk = frame.audio_chunk
            if audio_chunk is not None:
                audio_frame = rtc.AudioFrame(
                    data=audio_chunk.bytes,
                    sample_rate=audio_chunk.sample_rate,
                    num_channels=1,
                    samples_per_channel=len(audio_chunk.array),
                )
                yield audio_frame

            if frame.end_of_speech:
                yield AudioSegmentEnd()

    async def stop(self) -> None:
        await self._runtime.stop()
