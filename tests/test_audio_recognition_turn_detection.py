"""Integration tests for ``AudioRecognition`` audio turn-detection wiring.

Covers two concerns the FSM-level tests can't reach:

1. ``_turn_detection_task`` — VAD events forward to the stream's
   ``activate`` / ``deactivate`` / ``push_audio`` / ``flush`` calls in the right shape, and
   the stream's emitted predictions trigger ``_run_eou_detection`` plus
   deactivate the stream on a positive prediction.

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

from livekit import rtc
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import aio
from livekit.agents.vad import VADEvent, VADEventType
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
        base_url="",
        api_key="",
        api_secret="",
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        # Materialized table — the stream reads `self._opts.thresholds.get(lang)`
        # for the deactivate-on-positive-prediction check, so a populated entry
        # is required to exercise that branch.
        thresholds={"en": 0.5},
    )


def _vad_event(event_type: VADEventType) -> VADEvent:
    return VADEvent(
        type=event_type,
        samples_index=0,
        timestamp=0.0,
        speech_duration=0.0,
        silence_duration=0.0,
    )


def _pcm_frame(samples: int = 320) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=samples,
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
    """Records every public-surface call the wiring task is supposed to make,
    and lets tests inject predictions via ``emit``."""

    def __init__(self, *, detector: Any, opts: TurnDetectorOptions) -> None:
        self.activate_calls: list[str | None] = []
        self.deactivate_calls: list[str | None] = []
        self.push_audio_calls: list[rtc.AudioFrame] = []
        self.flush_calls: list[str | None] = []
        super().__init__(detector=detector, opts=opts, transport=_ParkingTransport())

    def activate(self, trigger: str | None = None) -> None:
        self.activate_calls.append(trigger)
        super().activate(trigger)

    def deactivate(self, trigger: str | None = None) -> None:
        self.deactivate_calls.append(trigger)
        super().deactivate(trigger)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self.push_audio_calls.append(frame)
        super().push_audio(frame)

    def flush(self, reason: str | None = None, *, keep_tail_ms: int = 0) -> None:
        self.flush_calls.append(reason)
        super().flush(reason, keep_tail_ms=keep_tail_ms)

    def emit(self, probability: float) -> None:
        """Push an event directly onto the stream's channel — bypasses
        request_id dedup so tests can exercise the consumer-side
        ``is_inference_running`` stale-event guard."""
        self._emit_event(
            TurnDetectionEvent(
                type="eot_prediction",
                last_speaking_time=time.time(),
                end_of_turn_probability=probability,
            )
        )


def _make_recognition_shell() -> AudioRecognition:
    """Build an AudioRecognition with only the attrs `_turn_detection_task` touches."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._hooks = MagicMock()
    ar._hooks.retrieve_chat_ctx.return_value = MagicMock(copy=MagicMock(return_value=MagicMock()))
    ar._closing = asyncio.Event()
    ar._tasks = set()
    ar._latest_eot_prediction = None
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


class TestTurnDetectionTaskForwarding:
    """VAD events / audio frames / sentinels round-trip through the channel
    into the right stream call."""

    async def test_vad_sos_calls_deactivate(self) -> None:
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        ch: aio.Chan[Any] = aio.Chan()
        task = asyncio.create_task(recognition._turn_detection_task(stream, ch, None))
        try:
            ch.send_nowait(_vad_event(VADEventType.START_OF_SPEECH))
            for _ in range(5):
                await asyncio.sleep(0)
            assert "vad sos" in stream.deactivate_calls
        finally:
            ch.close()
            await aio.cancel_and_wait(task)

    async def test_vad_eos_calls_activate(self) -> None:
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        ch: aio.Chan[Any] = aio.Chan()
        task = asyncio.create_task(recognition._turn_detection_task(stream, ch, None))
        try:
            ch.send_nowait(_vad_event(VADEventType.END_OF_SPEECH))
            for _ in range(5):
                await asyncio.sleep(0)
            assert "vad eos" in stream.activate_calls
        finally:
            ch.close()
            await aio.cancel_and_wait(task)

    async def test_audio_frame_pushed_to_stream(self) -> None:
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        ch: aio.Chan[Any] = aio.Chan()
        task = asyncio.create_task(recognition._turn_detection_task(stream, ch, None))
        try:
            frame = _pcm_frame()
            ch.send_nowait(frame)
            for _ in range(5):
                await asyncio.sleep(0)
            assert stream.push_audio_calls == [frame]
        finally:
            ch.close()
            await aio.cancel_and_wait(task)

    async def test_flush_sentinel_calls_flush(self) -> None:
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        ch: aio.Chan[Any] = aio.Chan()
        task = asyncio.create_task(recognition._turn_detection_task(stream, ch, None))
        try:
            ch.send_nowait(_AudioTurnDetectorStream._FlushSentinel(reason="turn"))
            for _ in range(5):
                await asyncio.sleep(0)
            assert stream.flush_calls == ["turn"]
        finally:
            ch.close()
            await aio.cancel_and_wait(task)

    async def test_manual_bool_dispatches_to_activate_deactivate(self) -> None:
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        ch: aio.Chan[Any] = aio.Chan()
        task = asyncio.create_task(recognition._turn_detection_task(stream, ch, None))
        try:
            ch.send_nowait(True)
            ch.send_nowait(False)
            for _ in range(5):
                await asyncio.sleep(0)
            assert "manual" in stream.activate_calls
            assert "manual" in stream.deactivate_calls
        finally:
            ch.close()
            await aio.cancel_and_wait(task)


class TestTurnDetectionTaskPredictions:
    """Predictions emitted by the stream drive `_run_eou_detection` + may
    deactivate the stream on a positive prediction."""

    async def test_prediction_invokes_run_eou_detection(self) -> None:
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        ch: aio.Chan[Any] = aio.Chan()
        task = asyncio.create_task(recognition._turn_detection_task(stream, ch, None))
        try:
            # FSM must be running so the event isn't filtered as stale.
            stream.warmup()
            stream.activate(trigger="test")
            stream.emit(0.3)  # below threshold 0.5 → no deactivate
            for _ in range(5):
                await asyncio.sleep(0)

            assert recognition._run_eou_detection.called
            kwargs = recognition._run_eou_detection.call_args.kwargs
            assert kwargs["trigger"] == "turn_detector"
            assert kwargs["latest_eot_prediction"].end_of_turn_probability == 0.3
            # Stream wasn't deactivated for sub-threshold prediction.
            assert "positive eou prediction" not in stream.deactivate_calls
        finally:
            ch.close()
            await aio.cancel_and_wait(task)

    async def test_positive_prediction_deactivates_stream(self) -> None:
        """A prediction >= unlikely_threshold deactivates the stream so a
        subsequent intra-speech silence can trigger a fresh warmup — letting
        the bounce task re-evaluate on more recent audio if the user keeps
        talking through the endpointing delay."""
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        ch: aio.Chan[Any] = aio.Chan()
        task = asyncio.create_task(recognition._turn_detection_task(stream, ch, None))
        try:
            stream.warmup()
            stream.activate(trigger="test")
            stream.emit(0.9)  # >= 0.5 threshold
            for _ in range(5):
                await asyncio.sleep(0)

            assert "positive eou prediction" in stream.deactivate_calls
        finally:
            ch.close()
            await aio.cancel_and_wait(task)

    async def test_prediction_skipped_when_inference_not_running(self) -> None:
        """An event delivered while the FSM is idle (no active turn) is
        treated as stale and must NOT trigger `_run_eou_detection`."""
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        ch: aio.Chan[Any] = aio.Chan()
        task = asyncio.create_task(recognition._turn_detection_task(stream, ch, None))
        try:
            # No warmup → no pending request; emit anyway.
            stream.emit(0.9)
            for _ in range(5):
                await asyncio.sleep(0)

            recognition._run_eou_detection.assert_not_called()
        finally:
            ch.close()
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
    ar._latest_eot_prediction = None
    ar._last_commit_time = None
    ar._last_committed_speaking_time = None
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

        ar._run_eou_detection(chat_ctx, trigger="turn_detector")

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

        ar._run_eou_detection(chat_ctx, trigger="turn_detector")

        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task
        ar._hooks.on_end_of_turn.assert_not_called()
        # predict_end_of_turn should not have been awaited — the guard
        # bailed before the bounce task started.
        assert ar._turn_detector_stream.predict_end_of_turn.call_count == 0


class TestLateSttAfterAudioCommit:
    """STT finals can arrive after the audio EOT model has already committed
    the turn. With the corresponding audio already flushed, running predict
    would evaluate near-silence, so the bounce skips the predict in that
    case. The guard is structural — gated on a prior commit, VAD presence,
    and absence of a fresh VAD START_OF_SPEECH since commit."""

    @staticmethod
    def _arm_late_after_commit(ar: AudioRecognition) -> None:
        """Set the state that marks 'we've committed, no fresh SOS since'."""
        ar._last_commit_time = time.time() - 0.1
        ar._last_committed_speaking_time = ar._last_commit_time
        ar._vad = MagicMock()
        ar._speech_start_time = None
        ar._audio_transcript = "late text"

    async def test_skips_predict_when_late_after_commit(self) -> None:
        ar = _make_full_recognition_for_eou()
        self._arm_late_after_commit(ar)
        chat_ctx = _make_chat_ctx_stub()

        ar._run_eou_detection(chat_ctx, trigger="stt")

        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task
        assert ar._turn_detector_stream.predict_end_of_turn.call_count == 0

    async def test_logs_warning_when_late_after_commit(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ar = _make_full_recognition_for_eou()
        self._arm_late_after_commit(ar)
        chat_ctx = _make_chat_ctx_stub()

        with caplog.at_level(logging.WARNING):
            ar._run_eou_detection(chat_ctx, trigger="stt")

        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task
        assert any("stt final arrived late" in record.message for record in caplog.records)

    async def test_commits_via_bounce_after_min_delay(self) -> None:
        """Skipping predict still goes through the existing bounce: sleep
        min_delay then call on_end_of_turn."""
        ar = _make_full_recognition_for_eou()
        self._arm_late_after_commit(ar)
        ar._hooks.on_end_of_turn.return_value = True  # actually commit
        chat_ctx = _make_chat_ctx_stub()

        ar._run_eou_detection(chat_ctx, trigger="stt")

        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task
        ar._hooks.on_end_of_turn.assert_called_once()
        # transcript cleared, stream flushed at commit
        assert ar._audio_transcript == ""
        ar._turn_detector_stream.flush.assert_called_once()

    async def test_fresh_sos_after_commit_runs_normal_predict(self) -> None:
        """A fresh VAD START_OF_SPEECH since the last commit clears the
        'late' classification — predict runs as usual."""
        ar = _make_full_recognition_for_eou()
        self._arm_late_after_commit(ar)
        ar._speech_start_time = time.time()  # fresh SOS happened
        chat_ctx = _make_chat_ctx_stub()

        ar._run_eou_detection(chat_ctx, trigger="stt")

        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task
        assert ar._turn_detector_stream.predict_end_of_turn.call_count == 1

    async def test_no_prior_commit_runs_normal_predict(self) -> None:
        """Before the first commit, ``_last_commit_time`` is None — the gate
        doesn't fire and predict runs."""
        ar = _make_full_recognition_for_eou()
        self._arm_late_after_commit(ar)
        ar._last_commit_time = None
        chat_ctx = _make_chat_ctx_stub()

        ar._run_eou_detection(chat_ctx, trigger="stt")

        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task
        assert ar._turn_detector_stream.predict_end_of_turn.call_count == 1

    async def test_no_vad_runs_normal_predict(self) -> None:
        """Without VAD there's no SOS signal to compare against — fall back
        to the normal predict path."""
        ar = _make_full_recognition_for_eou()
        self._arm_late_after_commit(ar)
        ar._vad = None
        chat_ctx = _make_chat_ctx_stub()

        ar._run_eou_detection(chat_ctx, trigger="stt")

        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task
        assert ar._turn_detector_stream.predict_end_of_turn.call_count == 1

    async def test_non_stt_trigger_runs_normal_predict(self) -> None:
        """Other triggers (vad, turn_detector, manual) never take this
        short-circuit even when the rest of the state matches."""
        ar = _make_full_recognition_for_eou()
        self._arm_late_after_commit(ar)
        chat_ctx = _make_chat_ctx_stub()

        ar._run_eou_detection(chat_ctx, trigger="turn_detector")

        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task
        assert ar._turn_detector_stream.predict_end_of_turn.call_count == 1


class _FakeVad:
    """Stand-in for a VAD that exposes the duck-typed surface the override
    mechanism probes: ``_opts.min_silence_duration`` (readable) and
    ``update_options(min_silence_duration=...)`` (writable)."""

    def __init__(self, min_silence_duration: float) -> None:
        self._opts = MagicMock()
        self._opts.min_silence_duration = min_silence_duration
        self.update_options_calls: list[float] = []

    def update_options(self, *, min_silence_duration: float) -> None:
        self.update_options_calls.append(min_silence_duration)
        self._opts.min_silence_duration = min_silence_duration


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

        assert ar._vad._opts.min_silence_duration == pytest.approx(0.25)
        assert ar._vad_min_silence_orig == pytest.approx(0.1)
        assert ar._vad.update_options_calls == [pytest.approx(0.25)]

    def test_audio_detector_leaves_high_min_silence_alone(self) -> None:
        ar = _make_recognition_for_override()
        ar._vad = _FakeVad(min_silence_duration=0.5)
        ar._turn_detector = MagicMock(spec=_AudioTurnDetector)

        ar._maybe_apply_vad_silence_override()

        assert ar._vad._opts.min_silence_duration == pytest.approx(0.5)
        assert ar._vad_min_silence_orig is None
        assert ar._vad.update_options_calls == []

    def test_swap_to_text_detector_restores_original(self) -> None:
        ar = _make_recognition_for_override()
        ar._vad = _FakeVad(min_silence_duration=0.1)
        ar._turn_detector = MagicMock(spec=_AudioTurnDetector)
        ar._maybe_apply_vad_silence_override()
        assert ar._vad._opts.min_silence_duration == pytest.approx(0.25)

        ar._revert_vad_silence_override()

        assert ar._vad._opts.min_silence_duration == pytest.approx(0.1)
        assert ar._vad_min_silence_orig is None

    def test_double_apply_is_idempotent(self) -> None:
        ar = _make_recognition_for_override()
        ar._vad = _FakeVad(min_silence_duration=0.1)
        ar._turn_detector = MagicMock(spec=_AudioTurnDetector)

        ar._maybe_apply_vad_silence_override()
        ar._maybe_apply_vad_silence_override()

        assert ar._vad.update_options_calls == [pytest.approx(0.25)]
        assert ar._vad_min_silence_orig == pytest.approx(0.1)

    def test_non_audio_detector_skips(self) -> None:
        ar = _make_recognition_for_override()
        ar._vad = _FakeVad(min_silence_duration=0.1)
        ar._turn_detector = MagicMock()  # not an _AudioTurnDetector

        ar._maybe_apply_vad_silence_override()

        assert ar._vad._opts.min_silence_duration == pytest.approx(0.1)
        assert ar._vad_min_silence_orig is None

    def test_no_vad_skips(self) -> None:
        ar = _make_recognition_for_override()
        ar._vad = None
        ar._turn_detector = MagicMock(spec=_AudioTurnDetector)

        ar._maybe_apply_vad_silence_override()  # must not raise

        assert ar._vad_min_silence_orig is None

    def test_vad_without_update_options_skips_silently(self) -> None:
        """A VAD that exposes ``_opts.min_silence_duration`` but no
        ``update_options`` is still safe — we just don't override it."""
        ar = _make_recognition_for_override()
        vad = MagicMock(spec=["_opts"])  # no update_options attribute
        vad._opts = MagicMock()
        vad._opts.min_silence_duration = 0.1
        ar._vad = vad
        ar._turn_detector = MagicMock(spec=_AudioTurnDetector)

        ar._maybe_apply_vad_silence_override()  # must not raise

        assert ar._vad_min_silence_orig is None

    def test_vad_update_options_without_min_silence_kwarg_skips(self) -> None:
        """A VAD whose ``update_options`` doesn't accept ``min_silence_duration``
        (different kwargs entirely) must not be called — would raise TypeError."""

        class _VadWithDifferentUpdateOptions:
            def __init__(self) -> None:
                self._opts = MagicMock()
                self._opts.min_silence_duration = 0.1
                self.update_options_calls: list[Any] = []

            def update_options(self, *, threshold: float) -> None:
                self.update_options_calls.append(threshold)

        vad = _VadWithDifferentUpdateOptions()
        ar = _make_recognition_for_override()
        ar._vad = vad
        ar._turn_detector = MagicMock(spec=_AudioTurnDetector)

        ar._maybe_apply_vad_silence_override()  # must not raise

        assert ar._vad_min_silence_orig is None
        assert vad.update_options_calls == []

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
        ar._turn_detection_ch = None
        ar._tasks = set()
        ar._vad = _FakeVad(min_silence_duration=0.1)
        # Stub the task body so update_turn_detector's create_task is a no-op.
        ar._turn_detection_task = AsyncMock()  # type: ignore[method-assign]

        detector = MagicMock(spec=_AudioTurnDetector)
        detector.stream.return_value = MagicMock()

        try:
            ar.update_turn_detector(detector)
            assert ar._vad._opts.min_silence_duration == pytest.approx(0.25)
            assert ar._vad_min_silence_orig == pytest.approx(0.1)

            ar.update_turn_detector(None)
            assert ar._vad._opts.min_silence_duration == pytest.approx(0.1)
            assert ar._vad_min_silence_orig is None
        finally:
            # Drain the cancel_and_wait fire-and-forget task spawned above.
            await asyncio.gather(*list(ar._tasks), return_exceptions=True)
