"""Tests for frame conversion utilities."""

import io
import struct

from PIL import Image

from livekit import rtc
from livekit.plugins.ojin.frames import (
    decode_jpeg_to_rgb24_sync,
    pcm16_bytes_to_audio_frame,
    rgb24_to_video_frame,
)


def test_pcm16_to_audio_frame_40ms():
    """640 samples mono @16kHz = 40ms frame."""
    num_samples = 640
    sample_rate = 16000
    channels = 1
    # Generate PCM16 bytes (little-endian int16)
    data = struct.pack(f"<{num_samples}h", *range(num_samples))

    frame = pcm16_bytes_to_audio_frame(data, sample_rate=sample_rate, channels=channels)

    assert isinstance(frame, rtc.AudioFrame)
    assert frame.sample_rate == sample_rate
    assert frame.num_channels == channels
    assert frame.samples_per_channel == num_samples


def test_decode_jpeg_to_rgb24_sync():
    """Valid JPEG returns correct dimensions and RGB24 data."""
    # Create a small test JPEG
    img = Image.new("RGB", (64, 48), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    width, height, rgb_data = decode_jpeg_to_rgb24_sync(jpeg_bytes)

    assert width == 64
    assert height == 48
    assert len(rgb_data) == 64 * 48 * 3  # RGB24: 3 bytes per pixel


def test_rgb24_to_video_frame():
    """VideoFrame should have correct type and dimensions."""
    width, height = 64, 48
    data = bytes(width * height * 3)

    frame = rgb24_to_video_frame(width, height, data)

    assert isinstance(frame, rtc.VideoFrame)
    assert frame.width == width
    assert frame.height == height
    assert frame.type == rtc.VideoBufferType.RGB24
