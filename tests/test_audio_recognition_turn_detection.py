"""Integration tests for ``AudioRecognition`` audio turn-detection wiring.

Covers two concerns the FSM-level tests can't reach:

1. ``_turn_detection_task`` — VAD events forward to the stream's
   ``set_active`` / ``push_audio`` / ``flush`` calls in the right shape, and
   the stream's emitted predictions trigger ``_run_eou_detection`` plus
   deactivate the stream on a positive prediction.

2. The speaking-guard race in ``_run_eou_detection``: setting
   ``_user_speaking_event`` mid-bounce must abort the commit so a
   late-arriving SOS doesn't ship the prior turn.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from livekit import rtc
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import aio
from livekit.agents.vad import VADEvent, VADEventType
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.turn import (
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


class _RecordingStream(_AudioTurnDetectorStream):
    """Records every public-surface call the wiring task is supposed to make,
    and lets tests inject predictions via ``emit``."""

    def __init__(self, *, detector: Any, opts: TurnDetectorOptions) -> None:
        self.set_active_calls: list[tuple[bool, str | None]] = []
        self.push_audio_calls: list[rtc.AudioFrame] = []
        self.flush_calls: list[str | None] = []
        super().__init__(detector=detector, opts=opts)

    def _transport_ready(self) -> bool:
        return True

    async def _run_transport(self) -> None:
        # Park forever; aclose cancels.
        await asyncio.Future()

    def set_active(self, active: bool, trigger: str | None = None) -> None:
        self.set_active_calls.append((active, trigger))
        super().set_active(active, trigger)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self.push_audio_calls.append(frame)
        super().push_audio(frame)

    def flush(self, reason: str | None = None, *, keep_tail_ms: int = 0) -> None:
        self.flush_calls.append(reason)
        super().flush(reason, keep_tail_ms=keep_tail_ms)

    def emit(self, probability: float) -> None:
        """Push a prediction into the stream's event channel as a real
        backend would."""
        self._emit_prediction(probability)


def _make_recognition_shell() -> AudioRecognition:
    """Build an AudioRecognition with only the attrs `_turn_detection_task` touches."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._hooks = MagicMock()
    ar._hooks.retrieve_chat_ctx.return_value = MagicMock(copy=MagicMock(return_value=MagicMock()))
    ar._closing = asyncio.Event()
    ar._tasks = set()
    ar._latest_eou_prediction = None
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

    async def test_vad_sos_calls_set_active_false(self) -> None:
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        ch: aio.Chan[Any] = aio.Chan()
        task = asyncio.create_task(recognition._turn_detection_task(stream, ch, None))
        try:
            ch.send_nowait(_vad_event(VADEventType.START_OF_SPEECH))
            for _ in range(5):
                await asyncio.sleep(0)
            assert (False, "vad sos") in stream.set_active_calls
        finally:
            ch.close()
            await aio.cancel_and_wait(task)

    async def test_vad_eos_calls_set_active_true(self) -> None:
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        ch: aio.Chan[Any] = aio.Chan()
        task = asyncio.create_task(recognition._turn_detection_task(stream, ch, None))
        try:
            ch.send_nowait(_vad_event(VADEventType.END_OF_SPEECH))
            for _ in range(5):
                await asyncio.sleep(0)
            assert (True, "vad eos") in stream.set_active_calls
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

    async def test_manual_bool_calls_set_active(self) -> None:
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        ch: aio.Chan[Any] = aio.Chan()
        task = asyncio.create_task(recognition._turn_detection_task(stream, ch, None))
        try:
            ch.send_nowait(True)
            ch.send_nowait(False)
            for _ in range(5):
                await asyncio.sleep(0)
            assert (True, "manual") in stream.set_active_calls
            assert (False, "manual") in stream.set_active_calls
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
            stream.set_active(True, trigger="test")
            stream.emit(0.3)  # below threshold 0.5 → no deactivate
            for _ in range(5):
                await asyncio.sleep(0)

            assert recognition._run_eou_detection.called
            kwargs = recognition._run_eou_detection.call_args.kwargs
            assert kwargs["trigger"] == "turn_detector"
            assert kwargs["latest_eou_prediction"].end_of_turn_probability == 0.3
            # Stream wasn't deactivated for sub-threshold prediction.
            assert (False, "positive eou prediction") not in stream.set_active_calls
        finally:
            ch.close()
            await aio.cancel_and_wait(task)

    async def test_positive_prediction_deactivates_stream(self) -> None:
        """A prediction >= unlikely_threshold flips the stream off so the
        next audio doesn't keep the warmup window open."""
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        ch: aio.Chan[Any] = aio.Chan()
        task = asyncio.create_task(recognition._turn_detection_task(stream, ch, None))
        try:
            stream.warmup()
            stream.set_active(True, trigger="test")
            stream.emit(0.9)  # >= 0.5 threshold
            for _ in range(5):
                await asyncio.sleep(0)

            assert (False, "positive eou prediction") in stream.set_active_calls
        finally:
            ch.close()
            await aio.cancel_and_wait(task)

    async def test_prediction_skipped_when_inference_not_running(self) -> None:
        """An event delivered while the FSM is DEACTIVATED (no active turn)
        is treated as stale and must NOT trigger `_run_eou_detection`."""
        recognition = _make_recognition_shell()
        stream = _RecordingStream(detector=_make_detector_stub(), opts=_make_opts())
        ch: aio.Chan[Any] = aio.Chan()
        task = asyncio.create_task(recognition._turn_detection_task(stream, ch, None))
        try:
            # No warmup → FSM is DEACTIVATED; emit anyway.
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
    ar._latest_eou_prediction = None
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
