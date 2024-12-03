import asyncio
import logging
import time
from collections import deque

import livekit.agents.utils as utils
from livekit import rtc

logger = logging.getLogger(__name__)


class AVSynchronizer:
    """Synchronize audio and video capture.

    Usage:
        av_sync = AVSynchronizer(
            audio_source=audio_source,
            video_source=video_source,
            video_sample_rate=video_sample_rate,
        )
        av_sync.start()

        async for video_frame, audio_frame in video_generator:
            av_sync.push(video_frame)
            av_sync.push(audio_frame)
    """

    def __init__(
        self,
        *,
        audio_source: rtc.AudioSource,
        video_source: rtc.VideoSource,
        video_sample_rate: float,
        video_queue_size_ms: float = 1000,
        _max_delay_tolerance_ms: float = 300,
    ):
        self._audio_source = audio_source
        self._video_source = video_source
        self._video_sample_rate = video_sample_rate

        self._video_queue_size_ms = video_queue_size_ms
        self._video_queue_max_size = int(video_sample_rate * video_queue_size_ms / 1000)
        self._video_queue = asyncio.Queue[rtc.VideoFrame](
            maxsize=self._video_queue_max_size
        )

        self._fps_controller = _FPSController(
            expected_fps=video_sample_rate,
            max_delay_tolerance_ms=_max_delay_tolerance_ms,
        )
        self._stopped = False

    def start(self) -> None:
        self._capture_video_task = asyncio.create_task(self._capture_video_task())

    async def push(self, frame: rtc.VideoFrame | rtc.AudioFrame) -> None:
        if isinstance(frame, rtc.AudioFrame):
            await self._audio_source.capture_frame(frame)
            return

        await self._video_queue.put(frame)

    async def _capture_video_task(self) -> None:
        while not self._stopped:
            frame = await self._video_queue.get()

            await self._fps_controller.wait_next_process()
            self._video_source.capture_frame(frame)
            self._fps_controller.after_process()

    async def aclose(self) -> None:
        self._stopped = True
        if self._capture_video_task:
            await utils.aio.gracefully_cancel(self._capture_video_task)


class _FPSController:
    def __init__(
        self, *, expected_fps: float, max_delay_tolerance_ms: float = 300
    ) -> None:
        """Controls frame rate by adjusting sleep time based on actual FPS.

        Usage:
            fps_controller = _FPSController(expected_fps=30, max_delay_tolerance_ms=300)
            while True:
                await fps_controller.wait_next_frame()
                # process frame
                await fps_controller.after_process()

        Args:
            expected_fps: Target frames per second
            max_delay_tolerance_ms: Maximum delay tolerance in milliseconds
        """
        self._expected_fps = expected_fps
        self._frame_interval = 1.0 / expected_fps

        self._max_delay_tolerance_secs = max_delay_tolerance_ms / 1000

        self._next_frame_time = None
        self._send_timestamps = deque(maxlen=self._fps_calc_winsize)

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
