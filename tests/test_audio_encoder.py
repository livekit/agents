from __future__ import annotations

import pytest

from livekit import rtc
from livekit.agents.utils.codecs import AudioStreamEncoder


def _frame(*, samples_per_channel: int, sample_rate: int = 16000) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples_per_channel,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples_per_channel,
    )


@pytest.mark.asyncio
async def test_opus_encoder_reports_samples_for_emitted_bytes() -> None:
    encoder = AudioStreamEncoder(codec="opus", sample_rate=16000, num_channels=1)

    total_input_samples = 0
    for _ in range(10):
        frame = _frame(samples_per_channel=320)
        total_input_samples += frame.samples_per_channel
        encoder.push(frame)
    encoder.end_input()

    emitted: list[tuple[int, int]] = []
    async for encoded in encoder:
        if encoded is not None and encoded.data:
            emitted.append((len(encoded.data), encoded.num_samples))

    assert [num_samples for _, num_samples in emitted] == [320, 2880]
    assert sum(num_samples for _, num_samples in emitted) == total_input_samples
