import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterable, Optional, Union

import numpy as np
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.utils.av_sync import AVSynchronizer

try:
    import cv2
except ImportError:
    raise RuntimeError(
        "cv2 is required to run this example, "
        "install with `pip install opencv-python`"
    )

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class MediaInfo:
    video_width: int
    video_height: int
    video_fps: float
    audio_sample_rate: int
    audio_channels: int


class _AudioEndSentinel:
    pass


async def audio_generator(
    media_info: MediaInfo,
    output_audio: asyncio.Queue[Union[rtc.AudioFrame, _AudioEndSentinel]],
):
    """Generates audio frames with alternating sine wave and silence periods"""
    frequency = 480  # Hz
    amplitude = 0.5
    period = 7.0
    sine_duration = 5.0  # Duration of sine wave in each period
    chunk_size = 1024

    while True:
        current_time = 0.0

        # Generate audio for sine_duration seconds
        while current_time < sine_duration:
            t = np.linspace(
                current_time,
                current_time + chunk_size / media_info.audio_sample_rate,
                num=chunk_size,
                endpoint=False,
            )
            # Create volume envelope using sine wave
            volume = np.abs(np.sin(2 * np.pi * current_time / sine_duration))
            samples = amplitude * volume * np.sin(2 * np.pi * frequency * t)

            # Convert to int16, (samples, channels)
            samples = (samples[:, np.newaxis] * 32767).astype(np.int16)
            if media_info.audio_channels > 1:
                samples = np.repeat(samples, media_info.audio_channels, axis=1)

            # Create audio frame
            audio_frame = rtc.AudioFrame(
                data=samples.tobytes(),
                sample_rate=media_info.audio_sample_rate,
                num_channels=samples.shape[1],
                samples_per_channel=samples.shape[0],
            )
            await output_audio.put(audio_frame)
            current_time += chunk_size / media_info.audio_sample_rate
            await asyncio.sleep(0)
        await output_audio.put(_AudioEndSentinel())

        # Simulate silence
        silence_duration = period - sine_duration
        await asyncio.sleep(silence_duration)


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

    def draw(self, canvas: np.ndarray, audio_samples: np.ndarray, fps: float):
        self.draw_timestamp(canvas, fps)
        plot_data = self.draw_current_wave(canvas, audio_samples)
        current_volume = np.abs(plot_data).mean()
        self.draw_volume_history(canvas, current_volume)


async def video_generator(
    media_info: MediaInfo,
    input_audio: asyncio.Queue[Union[rtc.AudioFrame, _AudioEndSentinel]],
    av_sync: AVSynchronizer,  # only used for drawing the actual fps on the video
) -> AsyncIterable[tuple[rtc.VideoFrame, Optional[rtc.AudioFrame]]]:
    canvas = np.zeros(
        (media_info.video_height, media_info.video_width, 4), dtype=np.uint8
    )
    canvas.fill(255)

    def _np_to_video_frame(image: np.ndarray) -> rtc.VideoFrame:
        return rtc.VideoFrame(
            width=image.shape[1],
            height=image.shape[0],
            type=rtc.VideoBufferType.RGBA,
            data=image.tobytes(),
        )

    audio_samples_per_frame = int(media_info.audio_sample_rate / media_info.video_fps)
    audio_buffer = np.zeros((0, media_info.audio_channels), dtype=np.int16)
    wave_visualizer = WaveformVisualizer()
    while True:
        try:
            # timeout has to be shorter than the frame interval to avoid starvation
            audio_frame = await asyncio.wait_for(
                input_audio.get(), timeout=0.5 / media_info.video_fps
            )
        except asyncio.TimeoutError:
            # generate frame without audio (e.g. silence state)
            new_frame = canvas.copy()
            wave_visualizer.draw(new_frame, np.zeros((1, 2)), av_sync.actual_fps)
            video_frame = _np_to_video_frame(new_frame)
            yield video_frame, None

            # speed is controlled by the video fps in av_sync
            await asyncio.sleep(0)
            continue

        if isinstance(audio_frame, _AudioEndSentinel):
            # drop the audio buffer when the audio finished
            audio_buffer = np.zeros((media_info.audio_channels, 0), dtype=np.int16)
            continue

        audio_samples = np.frombuffer(audio_frame.data, dtype=np.int16).reshape(
            -1, audio_frame.num_channels
        )  # (samples, channels)
        # accumulate audio samples to the buffer
        audio_buffer = np.concatenate([audio_buffer, audio_samples], axis=0)

        while audio_buffer.shape[0] >= audio_samples_per_frame:
            sub_samples = audio_buffer[:audio_samples_per_frame, :]
            audio_buffer = audio_buffer[audio_samples_per_frame:, :]

            new_frame = canvas.copy()
            wave_visualizer.draw(new_frame, sub_samples, av_sync.actual_fps)
            video_frame = _np_to_video_frame(new_frame)
            sub_audio_frame = rtc.AudioFrame(
                data=sub_samples.tobytes(),
                sample_rate=audio_frame.sample_rate,
                num_channels=sub_samples.shape[1],
                samples_per_channel=sub_samples.shape[0],
            )
            yield video_frame, sub_audio_frame


async def entrypoint(job: JobContext):
    await job.connect()
    room = job.room

    # Create media info
    media_info = MediaInfo(
        video_width=1280,
        video_height=720,
        video_fps=30.0,
        audio_sample_rate=48000,
        audio_channels=2,
    )

    # Create video and audio sources/tracks
    queue_size_ms = 50
    video_source = rtc.VideoSource(
        width=media_info.video_width,
        height=media_info.video_height,
    )
    audio_source = rtc.AudioSource(
        sample_rate=media_info.audio_sample_rate,
        num_channels=media_info.audio_channels,
        queue_size_ms=queue_size_ms,
    )

    video_track = rtc.LocalVideoTrack.create_video_track("video", video_source)
    audio_track = rtc.LocalAudioTrack.create_audio_track("audio", audio_source)

    # Publish tracks
    video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)

    await room.local_participant.publish_track(video_track, video_options)
    await room.local_participant.publish_track(audio_track, audio_options)

    # Create AV synchronizer
    av_sync = AVSynchronizer(
        audio_source=audio_source,
        video_source=video_source,
        video_fps=media_info.video_fps,
        video_queue_size_ms=queue_size_ms,
    )

    # Start audio generator
    audio_queue = asyncio.Queue[Union[rtc.AudioFrame, _AudioEndSentinel]](maxsize=1)
    audio_task = asyncio.create_task(audio_generator(media_info, audio_queue))

    try:
        async for video_frame, audio_frame in video_generator(
            media_info, audio_queue, av_sync=av_sync
        ):
            await av_sync.push(video_frame)
            if audio_frame:
                await av_sync.push(audio_frame)
    finally:
        audio_task.cancel()
        await av_sync.aclose()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            job_memory_warn_mb=400,
        )
    )
