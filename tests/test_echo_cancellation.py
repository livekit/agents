from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from livekit import rtc
from livekit.agents.voice.room_io._echo_cancellation import (
    _AdaptiveEchoCanceller,
    _RenderHistory,
)

pytestmark = pytest.mark.unit


def _frame(pcm: NDArray[np.int16], rate: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(pcm.tobytes(), rate, 1, len(pcm))


def _signal(rate: int, seconds: int, seed: int = 3734) -> NDArray[np.int16]:
    rng = np.random.default_rng(seed)
    pcm = np.convolve(rng.normal(0, 5000, rate * seconds), [0.2, 0.6, 0.2], mode="same")
    return pcm.astype(np.int16)


def _echo(reference: NDArray[np.int16], rate: int, delay_ms: int) -> NDArray[np.int16]:
    delayed = np.zeros(len(reference))
    for tap_ms, gain in [(0, 0.4), (20, 0.12)]:
        offset = rate * (delay_ms + tap_ms) // 1000
        delayed[offset:] += gain * reference[: len(reference) - offset]
    return delayed.astype(np.int16)


def _run(
    reference: NDArray[np.int16],
    capture: NDArray[np.int16],
    rate: int,
    *,
    frame_ms: int = 50,
) -> tuple[NDArray[np.int16], list[tuple[float, int]]]:
    aec = _AdaptiveEchoCanceller(sample_rate=rate)
    samples = rate * frame_ms // 1000
    output = np.zeros_like(capture)
    locks: list[tuple[float, int]] = []
    for i in range(0, len(capture), samples):
        timestamp = 1000 + i / rate
        aec.push_render(_frame(reference[i : i + samples], rate), started_at=timestamp)
        frame = _frame(capture[i : i + samples], rate)
        processed = aec.process_capture(frame, started_at=timestamp)
        output[i : i + samples] = np.frombuffer(processed.data, np.int16)
        if aec.bulk_delay_ms is None:
            assert processed is frame
        elif not locks or aec.bulk_delay_ms != locks[-1][1]:
            locks.append((i / rate, aec.bulk_delay_ms))
    return output, locks


def _reduction(before: NDArray[np.int16], after: NDArray[np.int16]) -> float:
    return float(
        10 * np.log10(np.mean(before.astype(float) ** 2) / np.mean(after.astype(float) ** 2))
    )


@pytest.mark.parametrize(
    ("rate", "delay_ms"), [(8000, 400), (16000, 1350), (24000, 2300), (32000, 1350), (48000, 400)]
)
def test_delayed_echo_is_aligned_and_suppressed(rate: int, delay_ms: int) -> None:
    reference = _signal(rate, 8)
    capture = _echo(reference, rate, delay_ms)
    output, locks = _run(reference, capture, rate)

    assert locks
    assert len(locks) == 1
    assert abs(locks[0][1] + 80 - delay_ms) <= 20
    assert locks[0][0] < delay_ms / 1000 + 1.2
    assert _reduction(capture[5 * rate :], output[5 * rate :]) > 10


@pytest.mark.parametrize("new_delay_ms", [1750, 1260])
def test_delay_change_reacquires_without_losing_near_end_speech(new_delay_ms: int) -> None:
    rate = 16000
    reference = _signal(rate, 13)
    capture = _echo(reference, rate, 1350)
    capture[5 * rate :] = _echo(reference, rate, new_delay_ms)[5 * rate :]
    # A voiced caller with varying pitch and syllable amplitude, independent of the render.
    t = np.arange(len(reference)) / rate
    phase = 2 * np.pi * np.cumsum(140 + 25 * np.sin(2 * np.pi * 2 * t)) / rate
    voice = sum(
        np.sin(k * phase + 0.3 * k) * np.exp(-(((k * 140 - 1000) / 700) ** 2)) for k in range(1, 20)
    )
    near_end = (voice * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * 2500).astype(np.int16)
    capture[10 * rate :] += near_end[10 * rate :]
    output, locks = _run(reference, capture, rate)

    assert len(locks) == 2
    assert abs(locks[0][1] + 80 - 1350) <= 20
    assert abs(locks[1][1] + 80 - new_delay_ms) <= 20
    assert locks[1][0] < 6.5
    assert _reduction(capture[8 * rate : 10 * rate], output[8 * rate : 10 * rate]) > 10
    # Check the known caller signal, not total output energy, during double-talk.
    # APM adds a short filter delay; allow up to 10 ms when comparing the waveforms.
    caller = near_end[11 * rate : 12 * rate].astype(float)
    candidates = [output[11 * rate + lag : 12 * rate + lag].astype(float) for lag in range(161)]
    result = max(candidates, key=lambda candidate: np.dot(caller, candidate))
    assert np.corrcoef(caller, result)[0, 1] > 0.8
    assert np.dot(caller, result) / np.dot(caller, caller) > 0.7


@pytest.mark.parametrize("reference_kind", ["missing", "unrelated", "tone", "dual_tone"])
def test_ambiguous_or_missing_reference_passes_through(reference_kind: str) -> None:
    rate = 16000
    capture = _signal(rate, 5)
    if reference_kind == "missing":
        reference = np.zeros_like(capture)
    elif reference_kind == "unrelated":
        reference = _signal(rate, 5, seed=19)
    else:
        reference = (3000 * np.sin(2 * np.pi * 500 * np.arange(len(capture)) / rate)).astype(
            np.int16
        )
        if reference_kind == "dual_tone":
            reference += (3000 * np.sin(2 * np.pi * 1200 * np.arange(len(capture)) / rate)).astype(
                np.int16
            )
        capture = reference.copy()
    output, locks = _run(reference, capture, rate)
    assert not locks
    np.testing.assert_array_equal(output, capture)


def test_future_playout_cannot_explain_current_capture() -> None:
    rate = 16000
    aec = _AdaptiveEchoCanceller(sample_rate=rate)
    reference = _signal(rate, 5)
    capture = _echo(reference, rate, 400)
    for i in range(0, len(reference), 800):
        timestamp = 1000 + i / rate
        aec.push_render(_frame(reference[i : i + 800], rate), started_at=timestamp + 1)
        frame = _frame(capture[i : i + 800], rate)
        assert aec.process_capture(frame, started_at=timestamp) is frame
        assert aec.bulk_delay_ms is None


def test_capture_gap_and_track_reset_require_a_new_delay_estimate() -> None:
    rate = 16000
    aec = _AdaptiveEchoCanceller(sample_rate=rate)
    reference = _signal(rate, 2)
    capture = _echo(reference, rate, 400)
    for i in range(0, len(reference), 800):
        aec.push_render(_frame(reference[i : i + 800], rate), started_at=i / rate)
        aec.process_capture(_frame(capture[i : i + 800], rate), started_at=i / rate)
    assert aec.bulk_delay_ms is not None
    frame = _frame(capture[:800], rate)
    assert aec.process_capture(frame, started_at=4) is frame
    assert aec.bulk_delay_ms is None
    aec.reset_capture()
    # Already-played output remains available for echoes after a track replacement.
    np.testing.assert_array_equal(
        aec._render.read(started_at=1.5, samples=800), reference[24000:24800]
    )


def test_render_timeline_retains_played_tail_and_discards_future_audio() -> None:
    history = _RenderHistory(8000)
    history.push(np.full(800, 10, dtype=np.int16), started_at=100)
    history.push(np.full(800, 20, dtype=np.int16), started_at=100.1)
    history.discard_after(100.125)
    history.push(np.full(800, 30, dtype=np.int16), started_at=100.3)

    expected = np.concatenate(
        [np.full(800, 10), np.full(200, 20), np.zeros(1400), np.full(800, 30)]
    )
    np.testing.assert_array_equal(history.read(started_at=100, samples=3200), expected)
    history.push(np.full(800, 40, dtype=np.int16), started_at=104)
    assert not history.read(started_at=100, samples=3200).any()


def test_stereo_render_is_resampled_on_the_playout_timeline() -> None:
    aec = _AdaptiveEchoCanceller(sample_rate=16000)
    mono = _signal(48000, 1)
    stereo = np.column_stack([mono, mono]).astype(np.int16)
    for i in range(0, len(mono), 2400):
        aec.push_render(
            rtc.AudioFrame(stereo[i : i + 2400].tobytes(), 48000, 2, 2400),
            started_at=100 + i / 48000,
        )
    converted = aec._render.read(started_at=100, samples=16000)
    resampler = rtc.AudioResampler(48000, 16000)
    expected = np.concatenate(
        [np.frombuffer(f.data, np.int16) for f in resampler.push(_frame(mono, 48000))]
    )
    # Native resampling dithers the least significant bits independently on each run.
    np.testing.assert_allclose(converted[: len(expected)], expected, atol=2)
    assert not aec._render.read(started_at=99, samples=16000).any()

    # A stop flushes the resampler's played tail before trimming the unplayed part.
    aec.discard_render_after(100.99)
    tail = np.concatenate([np.frombuffer(f.data, np.int16) for f in resampler.flush()])
    expected = np.concatenate([expected, tail])
    converted = aec._render.read(started_at=100, samples=16000)
    np.testing.assert_allclose(converted[:15840], expected[:15840], atol=2)
    assert not converted[15840:].any()
