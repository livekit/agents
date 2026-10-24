"""Tests for fax (CNG tone) detection."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from livekit import rtc
from livekit.agents.voice.fax import FaxDetectedEvent, FaxDetector
from livekit.agents.voice.fax.tone import _CNGToneDetector

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

PHONE_RATE = 8000
AGENT_RATE = 16000


def _tone(freqs: list[float], secs: float, amp: float = 8000.0) -> np.ndarray:
    t = np.arange(int(PHONE_RATE * secs)) / PHONE_RATE
    out = np.zeros_like(t)
    for f in freqs:
        out += amp / len(freqs) * np.sin(2 * np.pi * f * t)
    return out


def _silence(secs: float) -> np.ndarray:
    return np.zeros(int(PHONE_RATE * secs))


def _cng(cycles: int, amp: float = 8000.0) -> np.ndarray:
    return np.concatenate([np.concatenate([_tone([1100], 0.5, amp), _silence(3.0)])] * cycles)


def _fake_speech(secs: float) -> np.ndarray:
    # harmonic voice with a wandering pitch and syllable-like bursts
    t = np.arange(int(PHONE_RATE * secs)) / PHONE_RATE
    f0 = 170 + 70 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / PHONE_RATE
    voice = sum(np.sin(k * phase) / k for k in range(1, 20))
    return 4000 * voice * (np.sin(2 * np.pi * 3 * t) > 0)


def _phone_line(x: np.ndarray, noise: float = 200.0) -> np.ndarray:
    """Add line noise, round-trip through G.711 mu-law, and resample 8k -> 16k."""
    rng = np.random.default_rng(0)
    x = np.clip((x + rng.normal(0, noise, len(x))) / 32768.0, -1, 1)
    y = np.sign(x) * np.log1p(255 * np.abs(x)) / np.log1p(255)
    y = np.round(y * 127) / 127
    x = np.sign(y) * np.expm1(np.abs(y) * np.log1p(255)) / 255 * 32768.0
    up = np.interp(np.arange(len(x) * 2) / 2, np.arange(len(x)), x)
    return np.clip(up, -32768, 32767).astype(np.int16)


def _detect(audio: np.ndarray) -> float | None:
    det = _CNGToneDetector(AGENT_RATE)
    for i in range(0, len(audio), 320):  # 20 ms frames
        if det.push(audio[i : i + 320]):
            return det.detected_at
    return None


@pytest.mark.parametrize(
    "audio",
    [
        pytest.param(_cng(5), id="cng"),
        pytest.param(_cng(5, amp=1500), id="cng_quiet"),
        pytest.param(np.concatenate([_silence(1.0), _cng(3)]), id="cng_after_silence"),
    ],
)
def test_detects_cng(audio: np.ndarray) -> None:
    detected_at = _detect(_phone_line(audio))
    assert detected_at is not None
    assert 3.5 <= detected_at <= 6.0


@pytest.mark.parametrize(
    "audio",
    [
        pytest.param(np.concatenate([_silence(1), _tone([1100], 0.5), _silence(8)]), id="one_beep"),
        pytest.param(_tone([1100], 10), id="steady_1100hz"),
        pytest.param(
            np.concatenate([np.concatenate([_tone([480, 620], 0.5), _silence(0.5)])] * 10),
            id="us_busy",
        ),
        pytest.param(
            np.concatenate([np.concatenate([_tone([1100], 0.5), _silence(0.5)])] * 10),
            id="wrong_spacing",
        ),
        pytest.param(_fake_speech(10), id="speech"),
        pytest.param(_silence(10), id="silence"),
    ],
)
def test_ignores_non_fax(audio: np.ndarray) -> None:
    assert _detect(_phone_line(audio)) is None


async def _frames(audio: np.ndarray) -> AsyncIterator[rtc.AudioFrame]:
    """Yield 20 ms frames, paced in (virtual) real time."""
    for i in range(0, len(audio) - 319, 320):
        chunk = audio[i : i + 320]
        yield rtc.AudioFrame(chunk.tobytes(), AGENT_RATE, 1, len(chunk))
        await asyncio.sleep(0.02)


async def test_detector_emits_event() -> None:
    detector = FaxDetector(rtc.Room())
    events: list[FaxDetectedEvent] = []
    detector.on("fax_detected", events.append)

    await detector._detect("sip_caller", _frames(_phone_line(_cng(5))))

    assert detector.result is not None
    assert detector.result.participant_identity == "sip_caller"
    assert 3.5 <= detector.result.detected_after <= 6.0
    assert events == [detector.result]


async def test_detector_stops_after_timeout() -> None:
    detector = FaxDetector(rtc.Room(), timeout=2.0)
    # the second beep would arrive at 3.5 s, after the timeout
    await detector._detect("sip_caller", _frames(_phone_line(_cng(5))))
    assert detector.result is None


async def test_detector_times_out_without_audio() -> None:
    async def no_frames() -> AsyncIterator[rtc.AudioFrame]:
        await asyncio.sleep(60)
        yield rtc.AudioFrame(b"\x00" * 640, AGENT_RATE, 1, 320)

    detector = FaxDetector(rtc.Room(), timeout=2.0)
    await detector._detect("sip_caller", no_frames())
    assert detector.result is None
