"""The console speaker holds back the head of each segment until ``PREBUFFER_DURATION`` of audio
is queued or the segment is flushed. A duplex model streams at real-time pace, so without this
margin every network hiccup drained the buffer and played as a zero-filled click mid-word."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from livekit import rtc
from livekit.agents.cli._legacy import SAMPLE_RATE, ConsoleAudioOutput

pytestmark = pytest.mark.unit

BLOCK = SAMPLE_RATE // 10  # one sounddevice callback


def _frame(value: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=np.full(BLOCK, value, dtype=np.int16).tobytes(),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=BLOCK,
    )


def _read(out: ConsoleAudioOutput) -> np.ndarray:
    buf = np.empty((BLOCK, 1), dtype=np.int16)
    out.read_into(buf, BLOCK)
    return buf[:, 0]


async def test_segment_head_waits_for_prebuffer() -> None:
    out = ConsoleAudioOutput(asyncio.get_running_loop())
    await out.capture_frame(_frame(1))
    assert not _read(out).any()
    await out.capture_frame(_frame(2))
    await out.capture_frame(_frame(3))
    assert _read(out)[0] == 1


async def test_flush_releases_a_short_segment() -> None:
    out = ConsoleAudioOutput(asyncio.get_running_loop())
    await out.capture_frame(_frame(1))
    out.flush()
    assert _read(out)[0] == 1
    await asyncio.sleep(0.01)


async def test_underrun_mid_segment_does_not_reprime() -> None:
    out = ConsoleAudioOutput(asyncio.get_running_loop())
    for v in (1, 2, 3):
        await out.capture_frame(_frame(v))
    for _ in range(3):
        _read(out)
    assert not _read(out).any()
    await out.capture_frame(_frame(4))
    assert _read(out)[0] == 4


async def test_next_segment_primes_again() -> None:
    out = ConsoleAudioOutput(asyncio.get_running_loop())
    await out.capture_frame(_frame(1))
    out.flush()
    _read(out)
    await asyncio.sleep(0.01)
    await out.capture_frame(_frame(2))
    assert not _read(out).any()
