import asyncio
import logging
import os
import signal
import sys
import time
from collections import deque
from typing import AsyncIterator, Generator, Optional, Union

import cv2
import numpy as np
from dotenv import load_dotenv
from livekit import api, rtc

from .io import AudioFlushSentinel
from .worker import AvatarWorker, MediaOptions

# ensure LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are set
load_dotenv()

logger = logging.getLogger(__name__)


class VideoGenerator:
    def __init__(self, media_options: MediaOptions):
        self.media_options = media_options
        self._audio_queue: asyncio.Queue[Union[rtc.AudioFrame, AudioFlushSentinel]] = (
            asyncio.Queue()
        )

    async def push_audio(self, frame: rtc.AudioFrame | AudioFlushSentinel) -> None:
        await self._audio_queue.put(frame)

    def clear_buffer(self) -> None:
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def stream(
        self, av_sync: Optional[rtc.AVSynchronizer] = None
    ) -> AsyncIterator[
        tuple[rtc.VideoFrame, Optional[rtc.AudioFrame]] | AudioFlushSentinel
    ]:
        audio_samples_per_frame = int(
            self.media_options.audio_sample_rate / self.media_options.video_fps
        )

        # initialize background canvas
        background = np.zeros(
            (self.media_options.video_height, self.media_options.video_width, 4),
            dtype=np.uint8,
        )
        background.fill(255)

        # initialize audio buffer (n_samples, n_channels)
        audio_buffer = np.zeros((0, self.media_options.audio_channels), dtype=np.int16)
        wave_visualizer = WaveformVisualizer()

        def _generate_idle_frame() -> rtc.VideoFrame:
            idle_frame = background.copy()
            fps = av_sync.actual_fps if av_sync else None
            wave_visualizer.draw(
                idle_frame,
                audio_samples=np.zeros((1, self.media_options.audio_channels)),
                fps=fps,
            )
            return self._np_to_video_frame(idle_frame)

        def _generate_non_idle_frame(
            audio_frame: rtc.AudioFrame,
        ) -> Generator[tuple[rtc.VideoFrame, rtc.AudioFrame], None, None]:
            # NOTE: Audio frames and video frames have different frequencies,
            #       so we accumulate audio samples in a buffer before
            #       generating corresponding video frames
            nonlocal audio_buffer
            audio_samples = np.frombuffer(audio_frame.data, dtype=np.int16).reshape(
                -1, audio_frame.num_channels
            )  # (n_samples, n_channels)
            audio_buffer = np.concatenate([audio_buffer, audio_samples], axis=0)

            while audio_buffer.shape[0] >= audio_samples_per_frame:
                sub_samples = audio_buffer[:audio_samples_per_frame, :]
                audio_buffer = audio_buffer[audio_samples_per_frame:, :]

                canvas = background.copy()
                fps = av_sync.actual_fps if av_sync else None
                wave_visualizer.draw(canvas, sub_samples, fps=fps)
                video_frame = self._np_to_video_frame(canvas)
                sub_audio_frame = rtc.AudioFrame(
                    data=sub_samples.tobytes(),
                    sample_rate=audio_frame.sample_rate,
                    num_channels=sub_samples.shape[1],
                    samples_per_channel=sub_samples.shape[0],
                )
                yield video_frame, sub_audio_frame

        while True:
            try:
                # timeout has to be shorter than the frame interval to avoid starvation
                audio_frame = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=0.5 / self.media_options.video_fps
                )
            except asyncio.TimeoutError:
                # generate frame without audio (e.g. silence state)
                video_frame = _generate_idle_frame()
                yield video_frame, None
                await asyncio.sleep(0)
                continue

            if isinstance(audio_frame, AudioFlushSentinel):
                # reset the audio buffer when the audio finished
                audio_buffer = np.zeros(
                    (0, self.media_options.audio_channels), dtype=np.int16
                )
                yield AudioFlushSentinel()
                continue

            for video_frame, audio_frame in _generate_non_idle_frame(audio_frame):
                yield video_frame, audio_frame

    def _np_to_video_frame(self, image: np.ndarray) -> rtc.VideoFrame:
        return rtc.VideoFrame(
            width=image.shape[1],
            height=image.shape[0],
            type=rtc.VideoBufferType.RGBA,
            data=image.tobytes(),
        )


class WaveformVisualizer:
    def __init__(self, history_length: int = 1000):
        self.history_length = history_length
        self.volume_history: deque[float] = deque(maxlen=history_length)
        self.start_time = time.time()

    def draw_timestamp(self, canvas: np.ndarray, fps: float):
        height, width = canvas.shape[:2]
        text = f"{time.time() - self.start_time:.1f}s @ {fps:.1f}fps"
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font_face, font_scale, thickness
        )
        x = (width - text_width) // 2
        y = int((height - text_height) * 0.4 + baseline)
        cv2.putText(canvas, text, (x, y), font_face, font_scale, (0, 0, 0), thickness)

    def draw_current_wave(
        self, canvas: np.ndarray, audio_samples: np.ndarray
    ) -> np.ndarray:
        """Draw the current waveform and return the current values"""
        height, width = canvas.shape[:2]
        center_y = height // 2 + 100

        normalized_samples = audio_samples.astype(np.float32) / 32767.0
        normalized_samples = normalized_samples.mean(axis=1)  # (samples,)
        num_points = min(width, len(normalized_samples))

        if len(normalized_samples) > num_points:
            indices = np.linspace(0, len(normalized_samples) - 1, num_points, dtype=int)
            plot_data = normalized_samples[indices]
        else:
            plot_data = normalized_samples

        x_coords = np.linspace(0, width, num_points, dtype=int)
        y_coords = (plot_data * 200) + center_y

        cv2.line(canvas, (0, center_y), (width, center_y), (200, 200, 200), 1)
        points = np.column_stack((x_coords, y_coords.astype(int)))
        for i in range(len(points) - 1):
            cv2.line(canvas, tuple(points[i]), tuple(points[i + 1]), (0, 255, 0), 2)

        return plot_data

    def draw_volume_history(self, canvas: np.ndarray, current_volume: float):
        height, width = canvas.shape[:2]
        center_y = height // 2

        self.volume_history.append(current_volume)
        cv2.line(
            canvas, (0, center_y - 250), (width, center_y - 250), (200, 200, 200), 1
        )

        volume_x = np.linspace(0, width, len(self.volume_history), dtype=int)
        volume_y = center_y - 250 + (np.array(self.volume_history) * 200)
        points = np.column_stack((volume_x, volume_y.astype(int)))
        for i in range(len(points) - 1):
            cv2.line(canvas, tuple(points[i]), tuple(points[i + 1]), (255, 0, 0), 2)

    def draw(
        self,
        canvas: np.ndarray,
        audio_samples: np.ndarray,
        fps: Optional[float] = None,
    ):
        if fps:
            self.draw_timestamp(canvas, fps)
        plot_data = self.draw_current_wave(canvas, audio_samples)
        current_volume = np.abs(plot_data).mean()
        self.draw_volume_history(canvas, current_volume)


async def entrypoint(room: rtc.Room, room_name: str):
    token = (
        api.AccessToken()
        .with_identity("video-publisher")
        .with_name("Video Publisher")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                agent=True,
            )
        )
        .to_jwt()
    )
    url = os.getenv("LIVEKIT_URL")
    logging.info("connecting to %s", url)

    try:
        await room.connect(url, token)
        logging.info("connected to room %s", room.name)
    except rtc.ConnectError as e:
        logging.error("failed to connect to the room: %s", e)
        return

    media_options = MediaOptions(
        video_width=1280,
        video_height=720,
        video_fps=30,
        audio_sample_rate=16000,
        audio_channels=1,
    )
    video_generator = VideoGenerator(media_options)
    worker = AvatarWorker(
        room, video_generator=video_generator, media_options=media_options
    )
    await worker.start()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python example.py <room-name>")
        sys.exit(1)

    room_name = sys.argv[1]
    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)

    async def cleanup():
        await room.disconnect()
        loop.stop()

    asyncio.ensure_future(entrypoint(room, room_name))
    for signal in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()
