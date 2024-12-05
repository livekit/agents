import asyncio
import logging
import time
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

            # Convert to int16
            samples = (samples[np.newaxis, :] * 32767).astype(np.int16)
            if media_info.audio_channels > 1:
                samples = np.repeat(samples, media_info.audio_channels, axis=0)

            # Create audio frame
            audio_frame = rtc.AudioFrame(
                data=samples.tobytes(),
                sample_rate=media_info.audio_sample_rate,
                num_channels=media_info.audio_channels,
                samples_per_channel=chunk_size,
            )
            await output_audio.put(audio_frame)
            current_time += chunk_size / media_info.audio_sample_rate
            await asyncio.sleep(0)
        await output_audio.put(_AudioEndSentinel())

        # Simulate silence
        silence_duration = period - sine_duration
        await asyncio.sleep(silence_duration)


def _draw_timestamp(canvas: np.ndarray, duration: float, fps: float):
    height, width = canvas.shape[:2]
    text = f"{duration:.1f}s @ {fps:.1f}fps"
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font_face, font_scale, thickness
    )
    x = (width - text_width) // 2
    y = int((height - text_height) * 0.4 + baseline)
    cv2.putText(canvas, text, (x, y), font_face, font_scale, (0, 0, 0), thickness)


def _draw_wave(canvas: np.ndarray, audio_samples: np.ndarray):
    """Draws an audio waveform visualization"""
    height, width = canvas.shape[:2]
    center_y = height // 2 + 100

    # Normalize audio samples to [-1, 1]
    normalized_samples = audio_samples.astype(np.float32) / 32767.0

    num_points = min(width, len(normalized_samples[0]))
    if len(normalized_samples[0]) > num_points:
        indices = np.linspace(0, len(normalized_samples[0]) - 1, num_points, dtype=int)
        plot_data = normalized_samples[0][indices]
    else:
        plot_data = normalized_samples[0]

    x_coords = np.linspace(0, width, num_points, dtype=int)
    y_coords = (plot_data * 200) + center_y  # Scale the wave amplitude

    # Draw the center line and waveform
    cv2.line(canvas, (0, center_y), (width, center_y), (200, 200, 200), 1)
    points = np.column_stack((x_coords, y_coords.astype(int)))
    for i in range(len(points) - 1):
        cv2.line(canvas, tuple(points[i]), tuple(points[i + 1]), (0, 255, 0), 2)


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
    audio_buffer = np.zeros((media_info.audio_channels, 0), dtype=np.int16)
    start_time = time.time()
    while True:
        try:
            # timeout has to be shorter than the frame interval to avoid starvation
            audio_frame = await asyncio.wait_for(
                input_audio.get(), timeout=0.5 / media_info.video_fps
            )
        except asyncio.TimeoutError:
            # generate frame without audio (e.g. silence state)
            new_frame = canvas.copy()
            _draw_timestamp(new_frame, time.time() - start_time, av_sync.actual_fps)
            _draw_wave(new_frame, np.zeros((1, 2)))
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
            audio_frame.num_channels, -1
        )
        # accumulate audio samples to the buffer
        audio_buffer = np.concatenate([audio_buffer, audio_samples], axis=1)
        while audio_buffer.shape[1] >= audio_samples_per_frame:
            sub_samples = audio_buffer[:, :audio_samples_per_frame]
            audio_buffer = audio_buffer[:, audio_samples_per_frame:]

            new_frame = canvas.copy()
            _draw_timestamp(new_frame, time.time() - start_time, av_sync.actual_fps)
            _draw_wave(new_frame, sub_samples)
            video_frame = _np_to_video_frame(new_frame)
            sub_audio_frame = rtc.AudioFrame(
                data=sub_samples.tobytes(),
                sample_rate=audio_frame.sample_rate,
                num_channels=audio_frame.num_channels,
                samples_per_channel=sub_samples.shape[1],
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
        audio_channels=1,
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
