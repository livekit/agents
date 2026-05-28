"""Integration tests for ``AudioRecognition`` audio turn-detection wiring.

Covers concerns the FSM-level tests can't reach:

1. The speaking-guard race in ``_run_eou_detection``: setting
   ``_user_speaking_event`` mid-bounce must abort the commit so a
   late-arriving SOS doesn't ship the prior turn.

2. ``on_eot_prediction`` dedup across the vad-EOS and stt-final triggers that
   share one cached prediction, and the ``update_turn_detector`` swap wiring.

The deactivate-on-positive-prediction behavior now lives in the stream FSM
itself; see ``test_turn_detection_fsm.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents import vad
from livekit.agents.utils import aio
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.turn import (
    TurnDetectionEvent,
    _StreamingTurnDetector,
    _StreamingTurnDetectorStream,
)

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

    # turn_detector must be an _StreamingTurnDetector for the speaking-guard
    # variant to be chosen.
    ar._turn_detector = MagicMock(spec=_StreamingTurnDetector)

    # spec= on _StreamingTurnDetectorStream so the runtime_checkable isinstance
    # narrowing in audio_recognition's last_prediction reads sees the mock as
    # the streaming flavor.
    stream_mock = MagicMock(spec=_StreamingTurnDetectorStream)
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
    ar._last_emitted_prediction = None
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


def _inference_done(*, raw_speech: float) -> vad.VADEvent:
    """An ``INFERENCE_DONE`` event carrying ``raw_speech`` accumulated speech and no
    silence — the shape the silero/inference VAD emits each inference window."""
    return vad.VADEvent(
        type=vad.VADEventType.INFERENCE_DONE,
        samples_index=0,
        timestamp=0.0,
        speech_duration=0.0,
        silence_duration=0.0,
        raw_accumulated_speech=raw_speech,
        raw_accumulated_silence=0.0,
    )


class TestSubThresholdSpeakingSpike:
    """A noise spike can push ``raw_accumulated_speech`` above zero on ``INFERENCE_DONE``
    without ever reaching ``START_OF_SPEECH`` — so no ``END_OF_SPEECH`` fires to clear
    ``_user_speaking_event``. It must be cleared when speech drops back to zero, or the
    speaking-guard aborts every subsequent ``_StreamingTurnDetector`` commit forever."""

    async def test_subthreshold_spike_is_cleared(self) -> None:
        ar = _make_full_recognition_for_eou()

        # spike crosses the activation threshold: event gets set, no SOS.
        await ar._on_vad_event(_inference_done(raw_speech=0.1))
        assert ar._user_speaking_event.is_set()

        # spike subsides before min_speech_duration: accumulation resets to 0.
        await ar._on_vad_event(_inference_done(raw_speech=0.0))
        assert not ar._user_speaking_event.is_set()

    async def test_zero_speech_during_confirmed_turn_keeps_event(self) -> None:
        """Inside a confirmed turn (post-SOS, ``_speaking`` True) a momentary zero-speech
        window must NOT clear the event — ``END_OF_SPEECH`` owns the clear there."""
        ar = _make_full_recognition_for_eou()
        ar._speaking = True
        ar._user_speaking_event.set()

        await ar._on_vad_event(_inference_done(raw_speech=0.0))
        assert ar._user_speaking_event.is_set()

    async def test_stale_spike_does_not_block_next_commit(self) -> None:
        """End-to-end symptom: after a spike sets-then-clears, the EOU bounce must run to
        completion instead of being aborted by a stuck ``_user_speaking_event``."""
        ar = _make_full_recognition_for_eou()

        await ar._on_vad_event(_inference_done(raw_speech=0.1))
        await ar._on_vad_event(_inference_done(raw_speech=0.0))
        assert not ar._user_speaking_event.is_set()

        chat_ctx = _make_chat_ctx_stub()
        ar._run_eou_detection(chat_ctx, trigger="vad")
        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task

        ar._hooks.on_end_of_turn.assert_called_once()


class TestEotPredictionDedup:
    """Both EOU triggers in a turn (vad EOS + stt final) read the same cached
    ``TurnDetectionEvent`` from the audio stream. ``on_eot_prediction`` must
    fire exactly once for that single prediction."""

    async def test_vad_then_stt_emits_eot_prediction_once(self) -> None:
        """Regression for duplicate ``EotPredictionEvent``: the vad-trigger
        bounce emits and then parks in the endpointing sleep; the stt-trigger
        cancels it and runs a second bounce that reads the *same* cached
        prediction. Without identity dedup both bounces emit; with it, only
        the first does."""
        ar = _make_full_recognition_for_eou()
        chat_ctx = _make_chat_ctx_stub()

        # One prediction per inference window — both triggers read this object
        # by reference via ``turn_detector_stream.last_prediction``.
        cached = TurnDetectionEvent(
            type="eot_prediction",
            last_speaking_time=time.time(),
            end_of_turn_probability=0.2,  # below 0.5 threshold → endpointing max_delay
            inference_duration=0.05,
            detection_delay=0.1,
        )
        ar._turn_detector_stream.last_prediction = cached

        # vad trigger: bounce emits, then parks in the ~0.5s endpointing sleep.
        ar._run_eou_detection(chat_ctx, trigger="vad")
        for _ in range(5):
            await asyncio.sleep(0)
        await asyncio.sleep(0.02)
        assert ar._hooks.on_eot_prediction.call_count == 1

        # stt trigger: cancels the parked vad bounce and runs a fresh one that
        # reads the same cached prediction. Dedup must suppress a second emit.
        ar._run_eou_detection(chat_ctx, trigger="stt")
        for _ in range(5):
            await asyncio.sleep(0)
        await asyncio.sleep(0.02)

        assert ar._hooks.on_eot_prediction.call_count == 1
        assert ar._last_emitted_prediction is cached

        if ar._end_of_turn_task is not None:
            await aio.cancel_and_wait(ar._end_of_turn_task)

    async def test_no_cached_prediction_emits_every_bounce(self) -> None:
        """When there's no cached ``TurnDetectionEvent`` (``last_prediction`` is
        ``None`` — text-based detectors, or an audio timeout), each bounce must
        still emit: identity dedup only applies to a shared cached object."""
        ar = _make_full_recognition_for_eou()
        ar._turn_detector_stream.last_prediction = None
        chat_ctx = _make_chat_ctx_stub()

        ar._run_eou_detection(chat_ctx, trigger="vad")
        for _ in range(5):
            await asyncio.sleep(0)
        await asyncio.sleep(0.02)
        assert ar._hooks.on_eot_prediction.call_count == 1

        ar._run_eou_detection(chat_ctx, trigger="stt")
        for _ in range(5):
            await asyncio.sleep(0)
        await asyncio.sleep(0.02)
        assert ar._hooks.on_eot_prediction.call_count == 2

        if ar._end_of_turn_task is not None:
            await aio.cancel_and_wait(ar._end_of_turn_task)

    async def test_text_detector_emits_every_bounce(self) -> None:
        """A text-based detector (not an ``_StreamingTurnDetector``) has no
        streaming inference window — ``last_prediction`` is ``None`` — so it
        emits ``on_eot_prediction`` on every bounce, never deduped."""
        ar = _make_full_recognition_for_eou()
        # Swap the audio detector for a text one and give it a transcript so
        # ``_run_eou_detection`` selects the text detector.
        text_detector = MagicMock()  # not an _StreamingTurnDetector
        text_detector.supports_language = AsyncMock(return_value=True)
        text_detector.predict_end_of_turn = AsyncMock(return_value=0.2)
        text_detector.unlikely_threshold = AsyncMock(return_value=0.5)
        text_detector.last_prediction = None
        ar._turn_detector = text_detector
        ar._turn_detector_stream = None
        ar._audio_transcript = "hello there"
        chat_ctx = _make_chat_ctx_stub()

        ar._run_eou_detection(chat_ctx, trigger="vad")
        for _ in range(5):
            await asyncio.sleep(0)
        await asyncio.sleep(0.02)
        assert ar._hooks.on_eot_prediction.call_count == 1

        ar._run_eou_detection(chat_ctx, trigger="stt")
        for _ in range(5):
            await asyncio.sleep(0)
        await asyncio.sleep(0.02)
        assert ar._hooks.on_eot_prediction.call_count == 2

        if ar._end_of_turn_task is not None:
            await aio.cancel_and_wait(ar._end_of_turn_task)

    async def test_clear_user_turn_allows_next_turn_to_emit(self) -> None:
        """``clear_user_turn`` resets the dedup guard so the next turn's first
        prediction (a distinct object) emits again."""
        ar = _make_full_recognition_for_eou()
        prev = TurnDetectionEvent(
            type="eot_prediction",
            last_speaking_time=time.time(),
            end_of_turn_probability=0.2,
        )
        ar._last_emitted_prediction = prev

        # Wire only the bits ``clear_user_turn`` touches beyond the eou helper.
        ar._audio_interim_transcript = ""
        ar._audio_preflight_transcript = ""
        ar._stt_request_ids = []
        ar._turn_detector_stream.flush = MagicMock()
        ar.update_stt = MagicMock()  # type: ignore[method-assign]

        ar.clear_user_turn()

        assert ar._last_emitted_prediction is None


class _FakeVad:
    """Read-only stand-in exposing the ``min_silence_duration`` knob that
    ``AudioRecognition`` validates against (read duck-typed via ``getattr``)."""

    def __init__(self, min_silence_duration: float | None) -> None:
        self._min_silence_duration = min_silence_duration

    @property
    def min_silence_duration(self) -> float | None:
        return self._min_silence_duration


def _make_recognition_for_validation() -> AudioRecognition:
    """Minimal AudioRecognition wired for the VAD-silence validation — no tasks."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._vad = None
    ar._turn_detector = None
    ar._turn_detector_stream = None
    return ar


class TestVadMinSilenceRequirement:
    """``audio EOT`` needs ~200ms of trailing silence; the VAD must report
    END_OF_SPEECH no earlier than that. Rather than mutate the user's VAD,
    ``AudioRecognition`` fails fast when ``min_silence_duration`` is too low
    for an audio-EOT pairing."""

    def test_low_min_silence_with_audio_detector_raises(self) -> None:
        ar = _make_recognition_for_validation()
        ar._vad = _FakeVad(min_silence_duration=0.1)
        ar._turn_detector = MagicMock(spec=_StreamingTurnDetector)

        with pytest.raises(ValueError, match="min_silence_duration"):
            ar._check_vad_silence_requirement()

    def test_adequate_min_silence_passes(self) -> None:
        ar = _make_recognition_for_validation()
        ar._vad = _FakeVad(min_silence_duration=0.5)
        ar._turn_detector = MagicMock(spec=_StreamingTurnDetector)

        ar._check_vad_silence_requirement()  # must not raise

    def test_non_audio_detector_skips(self) -> None:
        ar = _make_recognition_for_validation()
        ar._vad = _FakeVad(min_silence_duration=0.05)
        ar._turn_detector = MagicMock()  # not an _StreamingTurnDetector

        ar._check_vad_silence_requirement()  # must not raise

    def test_no_vad_skips(self) -> None:
        ar = _make_recognition_for_validation()
        ar._vad = None
        ar._turn_detector = MagicMock(spec=_StreamingTurnDetector)

        ar._check_vad_silence_requirement()  # must not raise

    def test_vad_without_min_silence_knob_skips(self) -> None:
        """A VAD that doesn't expose ``min_silence_duration`` can't be
        validated, so the pairing is allowed (no raise)."""
        ar = _make_recognition_for_validation()
        ar._vad = MagicMock(spec=[])  # no min_silence_duration attribute
        ar._turn_detector = MagicMock(spec=_StreamingTurnDetector)

        ar._check_vad_silence_requirement()  # must not raise

    def test_update_turn_detector_validates_pairing(self) -> None:
        """Integration: attaching an audio detector over a too-low VAD raises
        through the ``update_turn_detector`` call site, before any stream is
        built."""
        ar = _make_recognition_for_validation()
        ar._tasks = set()
        ar._vad = _FakeVad(min_silence_duration=0.1)

        detector = MagicMock(spec=_StreamingTurnDetector)

        with pytest.raises(ValueError, match="min_silence_duration"):
            ar.update_turn_detector(detector)

        # Aborted before building a stream — and without calling .stream().
        assert ar._turn_detector_stream is None
        detector.stream.assert_not_called()
