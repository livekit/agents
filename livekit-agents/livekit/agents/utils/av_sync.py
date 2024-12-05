import asyncio
import logging
import time
from collections import deque
from typing import Optional, Union

import livekit.agents.utils as utils
from livekit import rtc

logger = logging.getLogger(__name__)


class AVSynchronizer:
    """Synchronize audio and video capture.

    Usage:
        av_sync = AVSynchronizer(
            audio_source=audio_source,
            video_source=video_source,
            video_fps=video_fps,
        )

        async for video_frame, audio_frame in video_generator:
            await av_sync.push(video_frame)
            await av_sync.push(audio_frame)
    """

    def __init__(
        self,
        *,
        audio_source: rtc.AudioSource,
        video_source: rtc.VideoSource,
        video_fps: float,
        video_queue_size_ms: float = 100,
        _max_delay_tolerance_ms: float = 300,
    ):
        self._audio_source = audio_source
        self._video_source = video_source
        self._video_fps = video_fps
        self._video_queue_size_ms = video_queue_size_ms
        self._max_delay_tolerance_ms = _max_delay_tolerance_ms

        self._stopped = False

        self._video_queue_max_size = int(
            self._video_fps * self._video_queue_size_ms / 1000
        )
        if self._video_queue_size_ms > 0:
            # ensure queue is bounded if queue size is specified
            self._video_queue_max_size = max(1, self._video_queue_max_size)

        self._video_queue = asyncio.Queue[rtc.VideoFrame](
            maxsize=self._video_queue_max_size
        )
        self._fps_controller = _FPSController(
            expected_fps=self._video_fps,
            max_delay_tolerance_ms=self._max_delay_tolerance_ms,
        )
        self._capture_video_task = asyncio.create_task(self._capture_video())

    async def push(self, frame: Union[rtc.VideoFrame, rtc.AudioFrame]) -> None:
        if isinstance(frame, rtc.AudioFrame):
            # TODO: test if frame duration is too long
            await self._audio_source.capture_frame(frame)
            return

        await self._video_queue.put(frame)

    async def clear_queue(self) -> None:
        self._audio_source.clear_queue()
        while not self._video_queue.empty():
            await self._video_queue.get()

    async def _capture_video(self) -> None:
        while not self._stopped:
            frame = await self._video_queue.get()
            async with self._fps_controller:
                self._video_source.capture_frame(frame)

    async def aclose(self) -> None:
        self._stopped = True
        if self._capture_video_task:
            await utils.aio.gracefully_cancel(self._capture_video_task)

    @property
    def actual_fps(self) -> float:
        return self._fps_controller.actual_fps


class _FPSController:
    def __init__(
        self, *, expected_fps: float, max_delay_tolerance_ms: float = 300
    ) -> None:
        """Controls frame rate by adjusting sleep time based on actual FPS.

        Usage:
            async with _FPSController(expected_fps=30):
                # process frame
                pass

        Args:
            expected_fps: Target frames per second
            max_delay_tolerance_ms: Maximum delay tolerance in milliseconds
        """
        self._expected_fps = expected_fps
        self._frame_interval = 1.0 / expected_fps
        self._max_delay_tolerance_secs = max_delay_tolerance_ms / 1000

        self._next_frame_time: Optional[float] = None
        self._fps_calc_winsize = max(2, int(1.0 * expected_fps))
        self._send_timestamps: deque[float] = deque(maxlen=self._fps_calc_winsize)

    async def __aenter__(self) -> None:
        await self.wait_next_process()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.after_process()

    async def wait_next_process(self) -> None:
        """Wait until it's time for the next frame.

        Adjusts sleep time based on actual FPS to maintain target rate.
        """
        current_time = time.perf_counter()

        # initialize the next frame time
        if self._next_frame_time is None:
            self._next_frame_time = current_time

        # calculate sleep time
        sleep_time = self._next_frame_time - current_time
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        else:
            # check if significantly behind schedule
            if -sleep_time > self._max_delay_tolerance_secs:
                logger.warning(
                    f"Frame capture was behind schedule for "
                    f"{-sleep_time * 1000:.2f} ms"
                )
                self._next_frame_time = time.perf_counter()

    def after_process(self) -> None:
        """Update timing information after processing a frame."""
        assert (
            self._next_frame_time is not None
        ), "wait_next_process must be called first"

        # update timing information
        self._send_timestamps.append(time.perf_counter())

        # calculate next frame time
        self._next_frame_time += self._frame_interval

    @property
    def expected_fps(self) -> float:
        return self._expected_fps

    @property
    def actual_fps(self) -> float:
        """Get current average FPS."""
        if len(self._send_timestamps) < 2:
            return 0

        return (len(self._send_timestamps) - 1) / (
            self._send_timestamps[-1] - self._send_timestamps[0]
        )
