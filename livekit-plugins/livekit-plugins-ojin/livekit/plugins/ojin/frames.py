from __future__ import annotations

import io

from PIL import Image

from livekit import rtc


def pcm16_bytes_to_audio_frame(data: bytes, sample_rate: int, channels: int) -> rtc.AudioFrame:
    """Convert PCM int16 bytes to a LiveKit AudioFrame.

    Args:
        data: Raw PCM int16 bytes (little-endian).
        sample_rate: Sample rate in Hz (e.g. 16000).
        channels: Number of audio channels (e.g. 1 for mono).

    Returns:
        An rtc.AudioFrame with the converted data.
    """
    # Each sample is 2 bytes (int16), total samples across all channels
    total_samples = len(data) // 2
    samples_per_channel = total_samples // channels

    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=samples_per_channel,
    )


def decode_jpeg_to_rgb24_sync(data: bytes) -> tuple[int, int, bytes]:
    """Decode JPEG bytes to RGB24 pixel data (synchronous, CPU-bound).

    This function should be called via asyncio.to_thread() to avoid
    blocking the event loop.

    Args:
        data: JPEG-encoded image bytes.

    Returns:
        A tuple of (width, height, rgb24_bytes).
    """
    with Image.open(io.BytesIO(data)) as source_image:
        rgb_image = source_image.convert("RGB")
        width, height = rgb_image.size
        rgb_bytes = rgb_image.tobytes()
    return width, height, rgb_bytes


def rgb24_to_video_frame(width: int, height: int, data: bytes) -> rtc.VideoFrame:
    """Convert RGB24 byte data to a LiveKit VideoFrame.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        data: RGB24 pixel data bytes.

    Returns:
        An rtc.VideoFrame with type RGB24.
    """
    return rtc.VideoFrame(
        width=width,
        height=height,
        type=rtc.VideoBufferType.RGB24,
        data=data,
    )
