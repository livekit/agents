"""Clova sends complete PCM audio at the recognition service's sample rate."""

import struct

import pytest

from livekit.plugins.clova.common import resample_audio

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("input_rate, output_rate", [(48000, 16000), (16000, 48000)])
def test_resampling_keeps_the_complete_audio_duration(input_rate: int, output_rate: int) -> None:
    samples = input_rate // 5
    pcm = struct.pack(f"<{samples}h", *([1000, -1000] * (samples // 2)))
    converted = resample_audio(pcm, input_rate, output_rate)
    assert len(converted) // 2 == output_rate // 5
    assert any(converted)


def test_resampling_rejects_incomplete_pcm_sample() -> None:
    with pytest.raises(ValueError):
        resample_audio(b"\x01\x02\x03", 48000, 16000)
