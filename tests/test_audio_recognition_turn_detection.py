"""Integration tests for ``AudioRecognition`` audio turn-detection wiring.

Covers two concerns the FSM-level tests can't reach:

1. ``_turn_detection_task`` — the stream's emitted predictions trigger
   ``_run_eou_detection`` plus deactivate the stream on a positive prediction.

2. The speaking-guard race in ``_run_eou_detection``: setting
   ``_user_speaking_event`` mid-bounce must abort the commit so a
   late-arriving SOS doesn't ship the prior turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents.utils import aio
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.turn import (
    TurnDetectionEvent,
    TurnDetectorOptions,
    _AudioTurnDetector,
    _AudioTurnDetectorStream,
)


def _make_opts() -> TurnDetectorOptions:
    return TurnDetectorOptions(
        sample_rate=16000,
        # Materialized table — the stream reads `self._opts.thresholds.get(lang)`
        # for the deactivate-on-positive-prediction check, so a populated entry
        # is required to exercise that branch.
        thresholds={"en": 0.5},
    )


class _ParkingTransport:
    """Transport that parks forever in `run()`; aclose cancels."""

    def bind(self, stream: _AudioTurnDetectorStream) -> None:
        pass

    async def run(self) -> None:
        await asyncio.Future()

    def transport_ready(self) -> bool:
        return True

    def start_inference(self, request_id: str) -> None:
        pass

    def stop_inference(self, *, reason: str | None) -> None:
        pass

    async def push_frame(self, frame: Any) -> None:
        pass

    async def flush(self, sentinel: Any) -> None:
        pass

    def detach(self) -> None:
        pass


class _RecordingStream(_AudioTurnDetectorStream):
    """Records deactivate calls and lets tests inject predictions via ``emit``."""

    def __init__(self, *, detector: Any, opts: TurnDetectorOptions) -> None:
        self.deactivate_calls: list[str | None] = []
        super().__init__(detector=detector, opts=opts, transport=_ParkingTransport())

    def deactivate(self, trigger: str | None = None) -> None:
        self.deactivate_calls.append(trigger)
        super().deactivate(trigger)

    def emit(self, probability: float) -> None:
        """Push an event directly onto the stream's channel — bypasses
        request_id dedup so tests can exercise the consumer-side
        ``is_inference_running`` stale-event guard. Mirrors the
        ``_last_prediction`` cache that ``_handle_prediction`` would normally
        set."""
        event = TurnDetectionEvent(
            type="eot_prediction",
            last_speaking_time=time.time(),
            end_of_turn_probability=probability,
        )
        self._last_prediction = event
        self._emit_event(event)


def _make_recognition_shell() -> AudioRecognition:
    """Build an AudioRecognition with only the attrs `_turn_detection_task` touches."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._hooks = MagicMock()
    ar._hooks.retrieve_chat_ctx.return_value = MagicMock(copy=MagicMock(return_value=MagicMock()))
    ar._closing = asyncio.Event()
    ar._tasks = set()
    ar._last_language = None
    ar._run_eou_detection = MagicMock()
    return ar


def _make_detector_stub() -> MagicMock:
    detector = MagicMock(spec=_AudioTurnDetector)
    detector.model = "test"
    detector.provider = "livekit"

    async def unlikely_threshold(_lang: Any) -> float:
        return 0.5

    async def supports_language(_lang: Any) -> bool:
        return True

    detector.unlikely_threshold = unlikely_threshold
    detector.supports_language = supports_language
    return detector


class TestTurnDetectionTaskPredictions:
    """Predictions emitted by the stream drive `_run_eou_detection` + may
    deactivate the stream on a positive prediction."""

    async def test_subthreshold_prediction_does_not_deactivate(self) -> None:
        """The turn-detection subscriber no longer re-fires `_run_eou_detection`
        (the vad-EOS bounce already covers that via the cached prediction).
        A below-threshold prediction only updates `last_prediction`; the
        stream stays active and `_run_eou_detection` is not called from here."""
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        task = asyncio.create_task(recognition._turn_detection_task(stream, None))
        try:
            # FSM must be running so the event isn't filtered as stale.
            stream.warmup()
            stream.activate(trigger="test")
            stream.emit(0.3)  # below threshold 0.5 → no deactivate
            for _ in range(5):
                await asyncio.sleep(0)

            recognition._run_eou_detection.assert_not_called()
            assert stream.last_prediction is not None
            assert stream.last_prediction.end_of_turn_probability == 0.3
            # Stream wasn't deactivated for sub-threshold prediction.
            assert "positive eou prediction" not in stream.deactivate_calls
        finally:
            await aio.cancel_and_wait(task)

    async def test_positive_prediction_deactivates_stream(self) -> None:
        """A prediction >= unlikely_threshold deactivates the stream so a
        subsequent intra-speech silence can trigger a fresh warmup — letting
        the bounce task re-evaluate on more recent audio if the user keeps
        talking through the endpointing delay."""
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        task = asyncio.create_task(recognition._turn_detection_task(stream, None))
        try:
            stream.warmup()
            stream.activate(trigger="test")
            stream.emit(0.9)  # >= 0.5 threshold
            for _ in range(5):
                await asyncio.sleep(0)

            assert "positive eou prediction" in stream.deactivate_calls
        finally:
            await aio.cancel_and_wait(task)

    async def test_prediction_skipped_when_inference_not_running(self) -> None:
        """An event delivered while the FSM is idle (no active turn) is
        treated as stale: the subscriber skips the threshold/deactivate path,
        even for a positive probability."""
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        task = asyncio.create_task(recognition._turn_detection_task(stream, None))
        try:
            # No warmup → no pending request; emit anyway.
            stream.emit(0.9)
            for _ in range(5):
                await asyncio.sleep(0)

            assert "positive eou prediction" not in stream.deactivate_calls
        finally:
            await aio.cancel_and_wait(task)


# ---------------------------------------------------------------------------
# Speaking-guard race
# ---------------------------------------------------------------------------


def _make_full_recognition_for_eou() -> AudioRecognition:
    """Wire enough of AudioRecognition to drive `_run_eou_detection` against
    a fake audio turn-detector — used by the speaking-guard tests."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._hooks = MagicMock()
    ar._hooks.on_end_of_turn.return_value = False  # don't commit
    ar._stt = None
    ar._audio_transcript = ""
    ar._turn_detection_mode = "vad"

    # turn_detector must be an _AudioTurnDetector for the speaking-guard
    # variant to be chosen.
    ar._turn_detector = MagicMock(spec=_AudioTurnDetector)

    stream_mock = MagicMock()
    stream_mock.supports_language = AsyncMock(return_value=True)
    stream_mock.predict_end_of_turn = AsyncMock(return_value=0.0)
    stream_mock.unlikely_threshold = AsyncMock(return_value=0.5)
    stream_mock.flush = MagicMock()
    ar._turn_detector_stream = stream_mock

    endpointing = MagicMock()
    endpointing.min_delay = 0.01
    endpointing.max_delay = 0.5  # long enough for the guard to fire mid-sleep
    ar._endpointing = endpointing

    ar._ensure_user_turn_span = MagicMock(  # type: ignore[method-assign]
        return_value=MagicMock(is_recording=MagicMock(return_value=False))
    )
    ar._user_turn_span = None
    ar._user_turn_start = None
    ar._user_speaking_event = asyncio.Event()
    ar._speaking = False
    ar._final_transcript_confidence = []
    ar._stt_request_ids = []
    ar._last_speaking_time = None
    ar._last_final_transcript_time = None
    ar._speech_start_time = None
    ar._vad_speech_started = False
    ar._end_of_turn_task = None
    ar._user_turn_committed = False
    ar._vad = None
    ar._last_language = None
    ar._closing = asyncio.Event()
    return ar


def _make_chat_ctx_stub() -> MagicMock:
    """ChatContext stub that survives the `.copy()` + `.add_message` +
    `.items[-N:]` calls inside `_run_eou_detection`."""
    ctx = MagicMock()
    ctx.copy = MagicMock(return_value=ctx)
    ctx.add_message = MagicMock()
    ctx.items = []
    return ctx


class TestSpeakingGuardRace:
    async def test_speaking_event_during_bounce_aborts_commit(self) -> None:
        """Regression: a VAD SOS during the endpointing-delay window must
        cancel the in-flight bounce so the prior turn doesn't commit."""
        ar = _make_full_recognition_for_eou()
        chat_ctx = _make_chat_ctx_stub()

        ar._run_eou_detection(chat_ctx, trigger="vad")

        # The bounce is sleeping ~0.5 s. Fire the speaking event well inside
        # that window — the guard's `asyncio.wait(FIRST_COMPLETED)` returns
        # with `speaking_task` done and the bounce gets cancelled.
        await asyncio.sleep(0.05)
        ar._user_speaking_event.set()

        assert ar._end_of_turn_task is not None
        with contextlib.suppress(asyncio.CancelledError):
            await ar._end_of_turn_task

        ar._hooks.on_end_of_turn.assert_not_called()

    async def test_speaking_at_entry_returns_immediately(self) -> None:
        """Variant: speaking already True when `_run_eou_detection` is
        called → the guard short-circuits without spawning the bounce."""
        ar = _make_full_recognition_for_eou()
        ar._speaking = True
        chat_ctx = _make_chat_ctx_stub()

        ar._run_eou_detection(chat_ctx, trigger="vad")

        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task
        ar._hooks.on_end_of_turn.assert_not_called()
        # predict_end_of_turn should not have been awaited — the guard
        # bailed before the bounce task started.
        assert ar._turn_detector_stream.predict_end_of_turn.call_count == 0


class _FakeVad:
    """Stand-in for a VAD that supports the ``min_silence_duration`` property
    contract from ``agents.vad.VAD``: getter exposes the current value,
    setter applies a new one."""

    def __init__(self, min_silence_duration: float) -> None:
        self._min_silence_duration = min_silence_duration
        self.setter_calls: list[float] = []

    @property
    def min_silence_duration(self) -> float | None:
        return self._min_silence_duration

    @min_silence_duration.setter
    def min_silence_duration(self, duration: float) -> None:
        self.setter_calls.append(duration)
        self._min_silence_duration = duration


def _make_recognition_for_override() -> AudioRecognition:
    """Minimal AudioRecognition wired for the override helpers — no tasks."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._vad = None
    ar._turn_detector = None
    ar._vad_min_silence_orig = None
    ar._warned_vad_silence_override = False
    return ar


class TestVadMinSilenceOverride:
    """``audio EOT`` needs ~200ms of trailing silence; the VAD must report
    END_OF_SPEECH no earlier than that. When the user pairs an audio EOT
    detector with a VAD configured to a shorter silence, ``AudioRecognition``
    bumps the VAD up — and restores the original on detach."""

    def test_audio_detector_bumps_low_min_silence(self) -> None:
        ar = _make_recognition_for_override()
        ar._vad = _FakeVad(min_silence_duration=0.1)
        ar._turn_detector = MagicMock(spec=_AudioTurnDetector)

        ar._maybe_apply_vad_silence_override()

        assert ar._vad.min_silence_duration == pytest.approx(0.25)
        assert ar._vad_min_silence_orig == pytest.approx(0.1)
        assert ar._vad.setter_calls == [pytest.approx(0.25)]

    def test_audio_detector_leaves_high_min_silence_alone(self) -> None:
        ar = _make_recognition_for_override()
        ar._vad = _FakeVad(min_silence_duration=0.5)
        ar._turn_detector = MagicMock(spec=_AudioTurnDetector)

        ar._maybe_apply_vad_silence_override()

        assert ar._vad.min_silence_duration == pytest.approx(0.5)
        assert ar._vad_min_silence_orig is None
        assert ar._vad.setter_calls == []

    def test_swap_to_text_detector_restores_original(self) -> None:
        ar = _make_recognition_for_override()
        ar._vad = _FakeVad(min_silence_duration=0.1)
        ar._turn_detector = MagicMock(spec=_AudioTurnDetector)
        ar._maybe_apply_vad_silence_override()
        assert ar._vad.min_silence_duration == pytest.approx(0.25)

        ar._revert_vad_silence_override()

        assert ar._vad.min_silence_duration == pytest.approx(0.1)
        assert ar._vad_min_silence_orig is None

    def test_double_apply_is_idempotent(self) -> None:
        ar = _make_recognition_for_override()
        ar._vad = _FakeVad(min_silence_duration=0.1)
        ar._turn_detector = MagicMock(spec=_AudioTurnDetector)

        ar._maybe_apply_vad_silence_override()
        ar._maybe_apply_vad_silence_override()

        assert ar._vad.setter_calls == [pytest.approx(0.25)]
        assert ar._vad_min_silence_orig == pytest.approx(0.1)

    def test_non_audio_detector_skips(self) -> None:
        ar = _make_recognition_for_override()
        ar._vad = _FakeVad(min_silence_duration=0.1)
        ar._turn_detector = MagicMock()  # not an _AudioTurnDetector

        ar._maybe_apply_vad_silence_override()

        assert ar._vad.min_silence_duration == pytest.approx(0.1)
        assert ar._vad_min_silence_orig is None

    def test_no_vad_skips(self) -> None:
        ar = _make_recognition_for_override()
        ar._vad = None
        ar._turn_detector = MagicMock(spec=_AudioTurnDetector)

        ar._maybe_apply_vad_silence_override()  # must not raise

        assert ar._vad_min_silence_orig is None

    def test_vad_without_min_silence_knob_skips_silently(self) -> None:
        """A VAD whose ``min_silence_duration`` getter returns ``None``
        (i.e. doesn't expose the knob — the ``agents.vad.VAD`` default)
        is left alone."""

        class _VadWithoutKnob:
            def __init__(self) -> None:
                self.setter_calls: list[float] = []

            @property
            def min_silence_duration(self) -> float | None:
                return None

            @min_silence_duration.setter
            def min_silence_duration(self, duration: float) -> None:
                self.setter_calls.append(duration)

        vad = _VadWithoutKnob()
        ar = _make_recognition_for_override()
        ar._vad = vad
        ar._turn_detector = MagicMock(spec=_AudioTurnDetector)

        ar._maybe_apply_vad_silence_override()  # must not raise

        assert ar._vad_min_silence_orig is None
        assert vad.setter_calls == []

    def test_info_logged_once_per_lifetime(self, caplog: pytest.LogCaptureFixture) -> None:
        """Second override (e.g., after VAD swap) doesn't double-log."""
        ar = _make_recognition_for_override()
        ar._turn_detector = MagicMock(spec=_AudioTurnDetector)

        with caplog.at_level(logging.INFO, logger="livekit.agents"):
            ar._vad = _FakeVad(min_silence_duration=0.1)
            ar._maybe_apply_vad_silence_override()
            # simulate VAD swap: clear snapshot, attach a new low VAD, re-apply
            ar._vad_min_silence_orig = None
            ar._vad = _FakeVad(min_silence_duration=0.05)
            ar._maybe_apply_vad_silence_override()

        bumped = [r for r in caplog.records if "bumping vad min_silence_duration" in r.message]
        assert len(bumped) == 1

    async def test_update_turn_detector_drives_apply_and_revert(self) -> None:
        """Integration: `update_turn_detector(audio)` runs apply; switching
        back to None runs revert. Locks in the call-site wiring."""
        ar = _make_recognition_for_override()
        # Wire the task fields update_turn_detector touches.
        ar._turn_detection_atask = None
        ar._tasks = set()
        ar._vad = _FakeVad(min_silence_duration=0.1)
        # Stub the task body so update_turn_detector's create_task is a no-op.
        ar._turn_detection_task = AsyncMock()  # type: ignore[method-assign]

        detector = MagicMock(spec=_AudioTurnDetector)
        detector.stream.return_value = MagicMock()

        try:
            ar.update_turn_detector(detector)
            assert ar._vad.min_silence_duration == pytest.approx(0.25)
            assert ar._vad_min_silence_orig == pytest.approx(0.1)

            ar.update_turn_detector(None)
            assert ar._vad.min_silence_duration == pytest.approx(0.1)
            assert ar._vad_min_silence_orig is None
        finally:
            # Drain the cancel_and_wait fire-and-forget task spawned above.
            await asyncio.gather(*list(ar._tasks), return_exceptions=True)
