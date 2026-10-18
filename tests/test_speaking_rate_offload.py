"""SpeakingRateStream should not monopolize the event loop while draining a backlog."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import numpy as np
import pytest

from livekit import rtc
from livekit.agents.voice.transcription._speaking_rate import (
    SpeakingRateDetector,
    SpeakingRateStream,
)

pytestmark = pytest.mark.unit


def _sine_frame(
    *,
    sample_rate: int = 16000,
    duration: float = 0.1,
    freq: float = 220.0,
    amplitude: float = 0.3,
) -> rtc.AudioFrame:
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    samples = (amplitude * np.sin(2 * np.pi * freq * t) * np.iinfo(np.int16).max).astype(np.int16)
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=n,
    )


@pytest.mark.asyncio
async def test_speaking_rate_stream_yields_events_for_burst() -> None:
    detector = SpeakingRateDetector(window_size=0.25, step_size=0.1, sample_rate=16000)
    stream = detector.stream()

    # ~1.2s of audio pushed as a burst (faster than real time)
    for _ in range(12):
        stream.push_frame(_sine_frame())
    stream.end_input()

    events = [event async for event in stream]
    await stream.aclose()

    assert len(events) >= 1
    assert all(e.speaking_rate >= 0 for e in events)


@pytest.mark.asyncio
async def test_speaking_rate_compute_does_not_block_event_loop() -> None:
    """CPU work in _compute_speaking_rate must not starve concurrent loop tasks (#7921)."""
    detector = SpeakingRateDetector(window_size=0.25, step_size=0.1, sample_rate=16000)
    stream = detector.stream()

    ticks = 0

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    def _slow_compute(self: SpeakingRateStream, audio: np.ndarray, sample_rate: int) -> float:
        # Simulate the expensive STFT path without depending on machine speed.
        time.sleep(0.05)
        return 1.0

    ticker = asyncio.create_task(_ticker())
    try:
        with patch.object(SpeakingRateStream, "_compute_speaking_rate", _slow_compute):
            for _ in range(8):
                stream.push_frame(_sine_frame(duration=0.1))
            stream.end_input()
            events = [event async for event in stream]
            await stream.aclose()
    finally:
        ticker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ticker

    assert len(events) >= 1
    # Several 50ms computes ran off-loop; the ticker should have advanced many times.
    assert ticks >= 5, f"event loop starved during drain (ticks={ticks})"
