from __future__ import annotations

import pytest
from ojin.stv import FrameType, STVAudioFrame, STVVideoFrame

from livekit import rtc
from livekit.plugins.ojin.frames import (
    downmix_to_mono,
    is_silence,
    to_audio_frame,
    to_video_frame,
)

# Hermetic: driven by a fake Ojin client, no network and no credentials.
pytestmark = pytest.mark.unit


def _video(width: int = 1024, height: int = 1024, *, rgb: bytes | None = b"", **kwargs):
    pixels = rgb if rgb != b"" else bytes(width * height * 3)
    return STVVideoFrame(
        rgb=pixels,
        source_bytes=kwargs.pop("source_bytes", b"\xff\xd8jpeg"),
        width=width,
        height=height,
        frame_type=kwargs.pop("frame_type", FrameType.SPEECH),
        pts=0,
        **kwargs,
    )


def _audio(pcm: bytes, sample_rate: int = 24000, num_channels: int = 1) -> STVAudioFrame:
    return STVAudioFrame(pcm=pcm, sample_rate=sample_rate, num_channels=num_channels, pts=0)


def test_video_frame_round_trip() -> None:
    frame = to_video_frame(_video())

    assert frame is not None
    assert (frame.width, frame.height) == (1024, 1024)
    assert frame.type == rtc.VideoBufferType.RGB24
    assert len(frame.data) == 1024 * 1024 * 3


def test_video_frame_keeps_non_square_geometry() -> None:
    frame = to_video_frame(_video(736, 1216))

    assert frame is not None
    assert (frame.width, frame.height) == (736, 1216)
    assert len(frame.data) == 736 * 1216 * 3


def test_held_tick_still_converts() -> None:
    """A held tick has empty source_bytes but rgb repeats the previous image."""
    frame = to_video_frame(_video(source_bytes=b"", frame_type=FrameType.IDLE))

    assert frame is not None
    assert len(frame.data) == 1024 * 1024 * 3


def test_video_frame_without_pixels_returns_none() -> None:
    assert to_video_frame(_video(rgb=None)) is None


def test_audio_frame_derives_sample_count_mono() -> None:
    frame = to_audio_frame(_audio(bytes(1920)))

    assert frame.sample_rate == 24000
    assert frame.num_channels == 1
    assert frame.samples_per_channel == 960


def test_audio_frame_derives_sample_count_stereo() -> None:
    frame = to_audio_frame(_audio(bytes(1920), num_channels=2))

    assert frame.num_channels == 2
    assert frame.samples_per_channel == 480


def test_is_silence_detects_synthesized_fill() -> None:
    assert is_silence(_audio(bytes(1920)))
    assert is_silence(_audio(b""))


def test_is_silence_false_for_real_audio() -> None:
    assert not is_silence(_audio(bytes(1918) + b"\x01\x00"))


def test_downmix_passes_mono_through_untouched() -> None:
    pcm = b"\x01\x02\x03\x04"

    assert downmix_to_mono(pcm, 1) is pcm


def test_downmix_averages_stereo_channels() -> None:
    import numpy as np

    stereo = np.array([[1000, 2000], [-400, -600]], dtype=np.int16).tobytes()

    mono = downmix_to_mono(stereo, 2)

    assert len(mono) == len(stereo) // 2
    assert np.frombuffer(mono, dtype=np.int16).tolist() == [1500, -500]
