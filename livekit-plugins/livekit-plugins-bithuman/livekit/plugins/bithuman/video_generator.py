from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING

import cv2
import numpy as np

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.avatar import AudioSegmentEnd, VideoGenerator

from .log import logger

if TYPE_CHECKING:
    from bithuman import AsyncBithuman  # type: ignore


class BithumanGenerator(VideoGenerator):
    def __init__(self, runtime: AsyncBithuman):
        self._runtime = runtime

        self._capturing = False
        self._playback_enabled = asyncio.Event()
        self._playback_enabled.set()

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
            self._capturing = False
            return

        if not self._capturing:
            self._capturing = True

        if not self._playback_enabled.is_set():
            await self._playback_enabled.wait()
            if not self._capturing:
                return

        await self._runtime.push_audio(bytes(frame.data), frame.sample_rate, last_chunk=False)

    def clear_buffer(self) -> None:
        self._runtime.interrupt()
        self._capturing = False

    def resume(self) -> None:
        self._playback_enabled.set()

    def pause(self) -> None:
        self._runtime.interrupt()
        self._playback_enabled.clear()

    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._stream_impl()

    async def _stream_impl(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        def create_video_frame(image: np.ndarray) -> rtc.VideoFrame:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return rtc.VideoFrame(
                width=image.shape[1],
                height=image.shape[0],
                type=rtc.VideoBufferType.RGB24,
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
