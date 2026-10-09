"""Hermetic tests for the Krisp turn-taking and interruption-prediction streams.

The Krisp SDK is replaced by a scripted session pair, so these exercise only the
LiveKit-facing logic: when a ``predict()`` resolves, and which overlap verdicts
are emitted. They do not import or require the ``krisp_audio`` wheel.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from livekit import rtc
from livekit.agents.inference import InterruptionDetectionError, OverlappingSpeechEvent
from livekit.agents.inference.interruption import (
    _AgentSpeechEndedSentinel,
    _AgentSpeechStartedSentinel,
    _OverlapSpeechEndedSentinel,
    _OverlapSpeechStartedSentinel,
)
from livekit.plugins.krisp import interruption, turn_detector

pytestmark = pytest.mark.unit

FRAME_SAMPLES = 320  # 20 ms at 16 kHz

# (voice, probability) per frame, given (frame index, bot_speaking)
Script = Callable[[int, bool], tuple[bool, float]]


class _ScriptedPair:
    def __init__(self, script: Script) -> None:
        self.frame_samples = FRAME_SAMPLES
        self.script = script
        self.calls: list[bool] = []  # bot_speaking per processed frame
        self.closed = False
        self.created_in: threading.Thread | None = None

    def process(self, frame: np.ndarray, bot_speaking: bool) -> tuple[bool, float]:
        self.calls.append(bot_speaking)
        return self.script(len(self.calls) - 1, bot_speaking)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("tt.kef", "ip.kef", "vad.kef"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setenv("KRISP_VIVA_TT_MODEL_PATH", str(tmp_path / "tt.kef"))
    monkeypatch.setenv("KRISP_VIVA_IP_MODEL_PATH", str(tmp_path / "ip.kef"))
    monkeypatch.setenv("KRISP_VIVA_VAD_MODEL_PATH", str(tmp_path / "vad.kef"))
    return tmp_path


def _install(monkeypatch: pytest.MonkeyPatch, module: object, script: Script) -> _ScriptedPair:
    pair = _ScriptedPair(script)

    def factory(**_: object) -> _ScriptedPair:
        pair.created_in = threading.current_thread()
        return pair

    monkeypatch.setattr(module, "_SessionPair", factory)
    return pair


def _install_failing(monkeypatch: pytest.MonkeyPatch, module: object) -> None:
    def factory(**_: object) -> _ScriptedPair:
        raise RuntimeError("bad model")

    monkeypatch.setattr(module, "_SessionPair", factory)


def _frames(n: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=np.zeros(n * FRAME_SAMPLES, dtype=np.int16).tobytes(),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=n * FRAME_SAMPLES,
    )


async def _drain() -> None:
    for _ in range(20):
        await asyncio.sleep(0)
    await asyncio.sleep(0.05)  # lets the worker-thread calls (model load, processing) finish


# region: turn detector


def _speech_then_silence(speech: int, *, ready_after: int = 10) -> Script:
    """Voice for ``speech`` frames, then silence whose TT result is ready after ``ready_after``."""

    def script(i: int, bot_speaking: bool) -> tuple[bool, float]:
        if i < speech:
            return True, 0.0
        silent = i - speech
        return False, 0.0 if silent < ready_after else 0.9 + silent / 1000

    return script


async def test_predict_waits_for_fresh_result(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, turn_detector, _speech_then_silence(5))
    stream = turn_detector.TurnDetector().stream()

    stream.push_audio(_frames(5 + 3))  # speech plus a short pause, no result yet
    fut = stream.predict()
    await _drain()
    assert not fut.done()

    stream.push_audio(_frames(10))  # result becomes ready during this pause
    await _drain()
    assert fut.done()
    # resolved with the first fresh value (silent frame 10), not a later one
    assert fut.result().end_of_turn_probability == pytest.approx(0.910)
    await stream.aclose()


async def test_predict_resolves_immediately_when_ready(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, turn_detector, _speech_then_silence(5))
    stream = turn_detector.TurnDetector().stream()

    stream.push_audio(_frames(5 + 15))
    await _drain()
    fut = stream.predict()
    await _drain()
    assert fut.done()
    # resolved with the latest value of the pause (silent frame 14)
    assert fut.result().end_of_turn_probability == pytest.approx(0.914)
    await stream.aclose()


async def test_cancel_inference_resolves_with_zero(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, turn_detector, _speech_then_silence(5))
    stream = turn_detector.TurnDetector().stream()

    stream.push_audio(_frames(5))
    fut = stream.predict()
    stream.cancel_inference()
    assert fut.result().end_of_turn_probability == 0.0
    await stream.aclose()


async def test_new_speech_invalidates_previous_result(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def script(i: int, bot_speaking: bool) -> tuple[bool, float]:
        if i < 5 or 20 <= i < 25:
            return True, 0.0
        return False, 0.0 if (5 <= i < 15 or 25 <= i < 35) else 0.8

    _install(monkeypatch, turn_detector, script)
    stream = turn_detector.TurnDetector().stream()

    stream.push_audio(_frames(27))  # first turn ready, then speech again, short pause
    fut = stream.predict()
    await _drain()
    assert not fut.done()  # the earlier result belongs to the previous pause
    await stream.aclose()


async def test_flush_resets_tt_with_bot_speaking(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = _install(monkeypatch, turn_detector, _speech_then_silence(2))
    stream = turn_detector.TurnDetector().stream()

    stream.push_audio(_frames(3))
    stream.flush(reason="turn committed")
    stream.push_audio(_frames(3))
    await _drain()
    assert pair.calls == [False, False, False, True, False, False]
    await stream.aclose()
    assert pair.closed


def _pause_rising(speech: int, *, start: float, step: float) -> Script:
    """Voice for ``speech`` frames, then a pause whose TT result (ready after 10 frames)
    starts at ``start`` and rises by ``step`` per frame."""

    def script(i: int, bot_speaking: bool) -> tuple[bool, float]:
        if i < speech:
            return True, 0.0
        silent = i - speech
        return False, 0.0 if silent < 10 else start + step * (silent - 10)

    return script


async def test_predict_waits_for_threshold_crossing(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, turn_detector, _pause_rising(5, start=0.2, step=0.02))
    stream = turn_detector.TurnDetector().stream()

    stream.push_audio(_frames(5 + 12))  # first fresh values (0.20, 0.22) are below 0.5
    fut = stream.predict()
    await _drain()
    assert not fut.done()

    stream.push_audio(_frames(20))
    await _drain()
    assert fut.done()
    # resolved on the first frame at or above the threshold (silent frame 25)
    assert fut.result().end_of_turn_probability == pytest.approx(0.5)
    await stream.aclose()


async def test_predict_answers_latest_value_after_budget(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, turn_detector, _pause_rising(5, start=0.2, step=0.001))
    stream = turn_detector.TurnDetector(max_prediction_delay=0.2).stream()  # 10 frames

    stream.push_audio(_frames(5 + 12))
    fut = stream.predict()
    stream.push_audio(_frames(9))
    await _drain()
    assert not fut.done()

    stream.push_audio(_frames(5))
    await _drain()
    assert fut.done()
    # budget ends on the 10th frame after the request (silent frame 21)
    assert fut.result().end_of_turn_probability == pytest.approx(0.211)
    await stream.aclose()


async def test_tt_models_load_off_the_event_loop(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = _install(monkeypatch, turn_detector, _speech_then_silence(0))
    stream = turn_detector.TurnDetector().stream()
    assert pair.created_in is None  # stream() returns without loading the models

    await _drain()
    assert pair.created_in is not None
    assert pair.created_in is not threading.main_thread()
    await stream.aclose()
    assert pair.closed


async def test_tt_load_failure_commits_turns(models: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_failing(monkeypatch, turn_detector)
    stream = turn_detector.TurnDetector().stream()
    await _drain()

    stream.push_audio(_frames(3))  # ignored, no error
    fut = stream.predict()
    assert fut.result().end_of_turn_probability == 1.0
    await stream.aclose()


async def test_detector_satisfies_streaming_protocol(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from livekit.agents.voice.turn import _StreamingTurnDetector, _StreamingTurnDetectorStream

    _install(monkeypatch, turn_detector, _speech_then_silence(0))
    detector = turn_detector.TurnDetector()
    stream = detector.stream()
    assert isinstance(detector, _StreamingTurnDetector)
    assert isinstance(stream, _StreamingTurnDetectorStream)
    await stream.aclose()


# endregion

# region: interruption detector


async def _collect(stream: interruption.InterruptionDetectorStream) -> list[OverlappingSpeechEvent]:
    events: list[OverlappingSpeechEvent] = []

    async def _read() -> None:
        async for ev in stream:
            events.append(ev)

    task = asyncio.create_task(_read())
    await _drain()
    task.cancel()
    return events


def _ip_script(probabilities: list[float]) -> Script:
    def script(i: int, bot_speaking: bool) -> tuple[bool, float]:
        p = probabilities[i] if i < len(probabilities) else 0.0
        return p > 0.0, p if bot_speaking else 0.0

    return script


async def test_interruption_detected_during_overlap(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = _install(monkeypatch, interruption, _ip_script([0.0, 0.2, 0.4, 0.7, 0.9]))
    detector = interruption.InterruptionDetector()
    emitted: list[OverlappingSpeechEvent] = []
    detector.on("overlapping_speech", emitted.append)
    stream = detector.stream()

    stream.push_frame(_AgentSpeechStartedSentinel())
    stream.push_frame(_OverlapSpeechStartedSentinel(speech_duration=0.0, started_at=1.0))
    stream.push_frame(_frames(5))
    stream.push_frame(_OverlapSpeechEndedSentinel(ended_at=2.0))
    events = await _collect(stream)

    assert [ev.is_interruption for ev in events] == [True]  # one verdict per overlap
    assert events[0].probability == pytest.approx(0.7)
    assert emitted == events
    assert pair.calls == [True] * 5
    await stream.aclose()


async def test_backchannel_reported_when_overlap_ends(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, interruption, _ip_script([0.1, 0.2, 0.1]))
    stream = interruption.InterruptionDetector().stream()

    stream.push_frame(_AgentSpeechStartedSentinel())
    stream.push_frame(_OverlapSpeechStartedSentinel(speech_duration=0.0, started_at=1.0))
    stream.push_frame(_frames(3))
    stream.push_frame(_OverlapSpeechEndedSentinel(ended_at=1.5, agent_ended=False))
    events = await _collect(stream)

    assert len(events) == 1
    assert events[0].is_interruption is False
    assert events[0].detection_delay == pytest.approx(0.5)
    await stream.aclose()


async def test_no_verdict_without_agent_speech(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = _install(monkeypatch, interruption, _ip_script([0.9, 0.9]))
    stream = interruption.InterruptionDetector().stream()

    stream.push_frame(_OverlapSpeechStartedSentinel(speech_duration=0.0, started_at=1.0))
    stream.push_frame(_frames(2))
    stream.push_frame(_OverlapSpeechEndedSentinel(ended_at=2.0))
    events = await _collect(stream)

    assert events == []
    assert pair.calls == [False, False]  # IP idles while the bot is not speaking
    await stream.aclose()


async def test_agent_speech_end_switches_bot_speaking_off(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = _install(monkeypatch, interruption, _ip_script([0.1] * 4))
    stream = interruption.InterruptionDetector().stream()

    stream.push_frame(_AgentSpeechStartedSentinel())
    stream.push_frame(_frames(2))
    stream.push_frame(_AgentSpeechEndedSentinel())
    stream.push_frame(_frames(2))
    await _collect(stream)

    assert pair.calls == [True, True, False, False]
    await stream.aclose()


async def test_ip_load_failure_emits_unrecoverable_error(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_failing(monkeypatch, interruption)
    detector = interruption.InterruptionDetector()
    errors: list[InterruptionDetectionError] = []
    detector.on("error", errors.append)
    stream = detector.stream()

    stream.push_frame(_AgentSpeechStartedSentinel())
    stream.push_frame(_frames(2))  # still accepted after the failure
    await _drain()

    assert len(errors) == 1
    assert errors[0].recoverable is False
    await stream.aclose()


# endregion
