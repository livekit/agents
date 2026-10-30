"""What ``_LocalTransport`` hands the native EOT model: the last ``EOT_MAX_SAMPLES`` of audio,
and the turn length in samples counted from the user turn start since the last flush."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from livekit import rtc
from livekit.agents.inference.eot.base import TurnDetectorOptions
from livekit.agents.inference.eot.languages import ThresholdOptions
from livekit.agents.inference.eot.transports import EOT_MAX_SAMPLES, _LocalTransport

pytestmark = pytest.mark.audio_eot


class _RecordingEOT:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None]] = []

    def predict(self, pcm: np.ndarray, turn_samples: int | None = None) -> float:
        self.calls.append((len(pcm), turn_samples))
        return 0.5


def _frame(num_samples: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=bytes(2 * num_samples),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=num_samples,
    )


@pytest.fixture
def transport() -> _LocalTransport:
    opts = TurnDetectorOptions(
        sample_rate=16000, thresholds=ThresholdOptions("turn-detector-v1-mini")
    )
    with patch("livekit.agents.inference.eot.transports._EOT", _RecordingEOT):
        return _LocalTransport(opts=opts)


async def _predict(transport: _LocalTransport) -> tuple[int, int | None]:
    stream = MagicMock()
    transport.attach(stream)
    transport.run_inference("request")
    await asyncio.gather(*transport._tasks)
    stream._resolve_prediction.assert_called_once()
    calls = transport._eot.calls  # type: ignore[attr-defined]
    return calls[-1]


async def test_turn_samples_count_from_the_turn_start_past_the_window(
    transport: _LocalTransport,
) -> None:
    transport.push_frame(_frame(8000))  # silence before the user speaks
    transport.push_frame(_frame(4000))
    transport.on_start_of_turn(0.25)
    for _ in range(2 * EOT_MAX_SAMPLES // 1600):
        transport.push_frame(_frame(1600))

    window, turn_samples = await _predict(transport)

    assert window == EOT_MAX_SAMPLES
    assert turn_samples == 4000 + 2 * EOT_MAX_SAMPLES // 1600 * 1600


async def test_flush_forgets_the_turn_start(transport: _LocalTransport) -> None:
    transport.push_frame(_frame(16000))
    transport.on_start_of_turn(0.1)
    transport.flush()
    transport.push_frame(_frame(6000))

    window, turn_samples = await _predict(transport)

    assert window == 6000
    assert turn_samples == 6000


async def test_turn_covers_at_least_the_audio_the_model_reads(
    transport: _LocalTransport,
) -> None:
    transport.push_frame(_frame(EOT_MAX_SAMPLES))
    transport.on_start_of_turn(0.05)  # a start later than the window's first sample

    _, turn_samples = await _predict(transport)

    assert turn_samples == EOT_MAX_SAMPLES


async def test_turn_longer_than_the_audio_clamps_to_flush(transport: _LocalTransport) -> None:
    transport.push_frame(_frame(1600))
    transport.on_start_of_turn(1.0)

    _, turn_samples = await _predict(transport)

    assert turn_samples == 1600
