"""Integration tests for ``AudioRecognition`` audio turn-detection wiring.

Recognition owns all streaming turn-detection policy: it holds the in-flight
inference request's future (``_turn_detector_prediction_fut``), starts
requests on VAD events only, awaits the future with the model-specific
``prediction_timeout`` in the eou bounce, and flushes the stream on turn
commits.
Covered here:

1. Resumed speech during the endpointing window: a ``START_OF_SPEECH`` mid-bounce
   cancels the in-flight eou task so the prior turn doesn't ship, while a
   sub-``min_speech_duration`` VAD spike (no SOS/EOS) must not block the next commit.

2. ``on_eot_prediction`` dedup across the vad-EOS and stt-final triggers that
   share one resolved prediction future, and the ``update_turn_detector``
   swap wiring.

3. The prediction-future lifecycle against VAD events: requests start
   exclusively on the silence tick, resumed speech inside a still-open VAD
   segment rearms the next pause, SOS teardown, the flushed-turn short-circuit
   for late stt finals, and the predict-timeout fallback signal.

The stream-side request lifecycle lives in ``test_turn_detection_fsm.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
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

pytestmark = pytest.mark.audio_eot

# ---------------------------------------------------------------------------
# Resumed-speech handling during the endpointing window
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
    # narrowing in audio_recognition's streaming branch sees the mock as the
    # streaming flavor.
    stream_mock = MagicMock(spec=_StreamingTurnDetectorStream)
    stream_mock.supports_language = AsyncMock(return_value=True)
    stream_mock.unlikely_threshold = AsyncMock(return_value=0.5)
    # backchannel disabled by default (server sent no thresholds); the
    # backchannel-emit tests override this with a positive threshold.
    stream_mock.backchannel_threshold = AsyncMock(return_value=None)
    # each call hands out a fresh pending future, mirroring the real
    # predict; tests install resolved/pending futures directly on
    # ar._turn_detector_prediction_fut to model cached/awaiting predictions
    stream_mock.predict = MagicMock(side_effect=asyncio.Future)
    stream_mock.flush = MagicMock()
    stream_mock.cancel_inference = MagicMock()
    stream_mock.prediction_timeout = 0.01
    ar._turn_detector_stream = stream_mock
    ar._turn_detector_prediction_fut = None
    ar._turn_detector_flushed = False
    ar._turn_detector_late_prediction_warned = False
    ar._agent_speaking = False
    ar._interruption_enabled = False
    ar._interruption_ch = None
    ar._vad_base_turn_detection = False

    endpointing = MagicMock()
    endpointing.min_delay = 0.01
    endpointing.max_delay = 0.5  # long enough for the guard to fire mid-sleep
    ar._endpointing = endpointing

    ar._ensure_user_turn_span = MagicMock(  # type: ignore[method-assign]
        return_value=MagicMock(is_recording=MagicMock(return_value=False))
    )
    ar._user_turn_span = None
    ar._user_turn_start = None
    ar._user_silence_ev = asyncio.Event()
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


def _resolved_prediction(
    probability: float,
    *,
    inference_duration: float | None = None,
    detection_delay: float | None = None,
    backchannel_probability: float | None = None,
) -> tuple[asyncio.Future[TurnDetectionEvent], TurnDetectionEvent]:
    """A resolved prediction future, as if the transport already answered."""
    event = TurnDetectionEvent(
        type="eot_prediction",
        last_speaking_time=time.time(),
        end_of_turn_probability=probability,
        inference_duration=inference_duration,
        detection_delay=detection_delay,
        backchannel_probability=backchannel_probability,
    )
    fut: asyncio.Future[TurnDetectionEvent] = asyncio.Future()
    fut.set_result(event)
    return fut, event


class TestResumedSpeechAbortsCommit:
    async def test_sos_during_bounce_cancels_commit(self) -> None:
        """Regression: a VAD ``START_OF_SPEECH`` during the endpointing-delay
        window cancels the in-flight bounce so the prior turn doesn't commit.
        The prior speaking-guard race was replaced by this SOS teardown."""
        ar = _make_full_recognition_for_eou()
        chat_ctx = _make_chat_ctx_stub()
        # sub-threshold prediction (0.2 < 0.5) extends endpointing to max_delay
        ar._turn_detector_prediction_fut, _ = _resolved_prediction(0.2)

        ar._run_eou_detection(chat_ctx, trigger="vad")
        task = ar._end_of_turn_task
        assert task is not None

        # The bounce is parked in the ~0.5 s endpointing sleep. Resumed speech
        # well inside that window tears the bounce down (audio_recognition's
        # SOS handler cancels ``_end_of_turn_task``).
        await asyncio.sleep(0.05)
        await ar._on_vad_event(_start_of_speech())

        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert task.cancelled()
        ar._hooks.on_end_of_turn.assert_not_called()


def _inference_done(*, raw_speech: float, raw_silence: float = 0.0) -> vad.VADEvent:
    """An ``INFERENCE_DONE`` event carrying the accumulated speech/silence —
    the shape the silero/inference VAD emits each inference window."""
    return vad.VADEvent(
        type=vad.VADEventType.INFERENCE_DONE,
        samples_index=0,
        timestamp=0.0,
        speech_duration=0.0,
        silence_duration=0.0,
        raw_accumulated_speech=raw_speech,
        raw_accumulated_silence=raw_silence,
    )


def _start_of_speech() -> vad.VADEvent:
    return vad.VADEvent(
        type=vad.VADEventType.START_OF_SPEECH,
        samples_index=0,
        timestamp=0.0,
        speech_duration=0.5,
        silence_duration=0.0,
    )


def _end_of_speech() -> vad.VADEvent:
    return vad.VADEvent(
        type=vad.VADEventType.END_OF_SPEECH,
        samples_index=0,
        timestamp=0.0,
        speech_duration=0.0,
        silence_duration=0.3,
    )


class TestSubThresholdSpeakingSpike:
    """A noise spike can push ``raw_accumulated_speech`` above zero on ``INFERENCE_DONE``
    without ever reaching ``START_OF_SPEECH`` — so no SOS/EOS fires. Resumed speech is
    gated on a real SOS, not on a momentary spike, so the spike must not block a later
    ``_StreamingTurnDetector`` commit (the regression that wedged the turn forever)."""

    async def test_stale_spike_does_not_block_next_commit(self) -> None:
        ar = _make_full_recognition_for_eou()

        # spike crosses the activation threshold then subsides before
        # min_speech_duration: no SOS, no EOS.
        await ar._on_vad_event(_inference_done(raw_speech=0.1))
        await ar._on_vad_event(_inference_done(raw_speech=0.0))

        chat_ctx = _make_chat_ctx_stub()
        ar._run_eou_detection(chat_ctx, trigger="vad")
        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task

        ar._hooks.on_end_of_turn.assert_called_once()


class TestEotPredictionDedup:
    """Both EOU triggers in a turn (vad EOS + stt final) read the same resolved
    prediction future. ``on_eot_prediction`` must fire exactly once for that
    single prediction."""

    async def test_vad_then_stt_emits_eot_prediction_once(self) -> None:
        """Regression for duplicate ``EotPredictionEvent``: the vad-trigger
        bounce emits and then parks in the endpointing sleep; the stt-trigger
        cancels it and runs a second bounce that reads the *same* resolved
        future. Without identity dedup both bounces emit; with it, only the
        first does."""
        ar = _make_full_recognition_for_eou()
        chat_ctx = _make_chat_ctx_stub()

        # One prediction per inference request — both triggers read this event
        # by reference from the held future.
        fut, cached = _resolved_prediction(
            0.2,  # below 0.5 threshold → endpointing max_delay
            inference_duration=0.05,
            detection_delay=0.1,
        )
        ar._turn_detector_prediction_fut = fut

        # vad trigger: bounce emits, then parks in the ~0.5s endpointing sleep.
        ar._run_eou_detection(chat_ctx, trigger="vad")
        for _ in range(5):
            await asyncio.sleep(0)
        await asyncio.sleep(0.02)
        assert ar._hooks.on_eot_prediction.call_count == 1

        # stt trigger: cancels the parked vad bounce and runs a fresh one that
        # reads the same resolved future. Dedup must suppress a second emit.
        ar._run_eou_detection(chat_ctx, trigger="stt")
        for _ in range(5):
            await asyncio.sleep(0)
        await asyncio.sleep(0.02)

        assert ar._hooks.on_eot_prediction.call_count == 1
        assert ar._last_emitted_prediction is cached

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


class TestBackchannelOpportunityEmit:
    """``on_agent_backchannel_opportunity`` fires whenever the backchannel
    probability clears its threshold, regardless of end-of-turn; the event carries
    the end-of-turn probability and threshold so AgentActivity can gauge how close
    the pause is to a reply."""

    @staticmethod
    async def _drive(ar: AudioRecognition, chat_ctx: MagicMock) -> None:
        ar._run_eou_detection(chat_ctx, trigger="vad")
        for _ in range(5):
            await asyncio.sleep(0)
        await asyncio.sleep(0.02)
        if ar._end_of_turn_task is not None:
            await aio.cancel_and_wait(ar._end_of_turn_task)

    async def test_emits_with_eot_context_when_turn_continues(self) -> None:
        ar = _make_full_recognition_for_eou()
        ar._turn_detector_stream.backchannel_threshold = AsyncMock(return_value=0.5)
        ar._last_language = "en"
        chat_ctx = _make_chat_ctx_stub()
        # eot 0.2 < unlikely 0.5 → turn continues; backchannel 0.8 >= 0.5 → emit
        ar._turn_detector_prediction_fut, _ = _resolved_prediction(0.2, backchannel_probability=0.8)

        await self._drive(ar, chat_ctx)

        ar._hooks.on_agent_backchannel_opportunity.assert_called_once()
        ev = ar._hooks.on_agent_backchannel_opportunity.call_args.args[0]
        assert ev.probability == pytest.approx(0.8)
        assert ev.threshold == pytest.approx(0.5)
        assert ev.language == "en"
        assert ev.end_of_turn_probability == pytest.approx(0.2)
        assert ev.end_of_turn_threshold == pytest.approx(0.5)

    async def test_emits_with_eot_context_when_turn_ends(self) -> None:
        """The turn-continuing gate was dropped: a backchannel above threshold
        still fires at end-of-turn, carrying the EOT context (probability past the
        threshold) so AgentActivity can let it lead the reply."""
        ar = _make_full_recognition_for_eou()
        ar._turn_detector_stream.backchannel_threshold = AsyncMock(return_value=0.5)
        chat_ctx = _make_chat_ctx_stub()
        # eot 0.9 >= unlikely 0.5 → turn ends; backchannel 0.8 >= 0.5 → still emits
        ar._turn_detector_prediction_fut, _ = _resolved_prediction(0.9, backchannel_probability=0.8)

        await self._drive(ar, chat_ctx)

        ar._hooks.on_agent_backchannel_opportunity.assert_called_once()
        ev = ar._hooks.on_agent_backchannel_opportunity.call_args.args[0]
        assert ev.end_of_turn_probability == pytest.approx(0.9)
        assert ev.end_of_turn_threshold == pytest.approx(0.5)

    async def test_no_emit_below_threshold(self) -> None:
        ar = _make_full_recognition_for_eou()
        ar._turn_detector_stream.backchannel_threshold = AsyncMock(return_value=0.7)
        chat_ctx = _make_chat_ctx_stub()
        # backchannel 0.4 < 0.7 → no emit (turn continues at eot 0.2)
        ar._turn_detector_prediction_fut, _ = _resolved_prediction(0.2, backchannel_probability=0.4)

        await self._drive(ar, chat_ctx)

        ar._hooks.on_agent_backchannel_opportunity.assert_not_called()

    async def test_no_emit_when_backchannel_disabled(self) -> None:
        ar = _make_full_recognition_for_eou()
        # default helper threshold is None (server sent no backchannel defaults)
        chat_ctx = _make_chat_ctx_stub()
        ar._turn_detector_prediction_fut, _ = _resolved_prediction(0.2, backchannel_probability=0.9)

        await self._drive(ar, chat_ctx)

        ar._hooks.on_agent_backchannel_opportunity.assert_not_called()

    async def test_no_emit_for_text_detector(self) -> None:
        """A text detector produces no streaming prediction event, so there is
        no backchannel probability to act on."""
        ar = _make_full_recognition_for_eou()
        text_detector = MagicMock()  # not an _StreamingTurnDetector
        text_detector.supports_language = AsyncMock(return_value=True)
        text_detector.predict_end_of_turn = AsyncMock(return_value=0.2)
        text_detector.unlikely_threshold = AsyncMock(return_value=0.5)
        ar._turn_detector = text_detector
        ar._turn_detector_stream = None
        ar._audio_transcript = "hello there"
        chat_ctx = _make_chat_ctx_stub()

        await self._drive(ar, chat_ctx)

        ar._hooks.on_agent_backchannel_opportunity.assert_not_called()


class TestPredictionFutureLifecycle:
    """The held prediction future against VAD events: requests start on the
    silence tick, are rearmed by resumed speech or SOS, and the flushed-turn
    flag blocks new requests until fresh speech."""

    async def test_silence_tick_starts_request_once(self) -> None:
        ar = _make_full_recognition_for_eou()
        ar._speaking = True

        await ar._on_vad_event(_inference_done(raw_speech=0.0, raw_silence=0.3))
        await ar._on_vad_event(_inference_done(raw_speech=0.0, raw_silence=0.4))

        assert ar._turn_detector_stream.predict.call_count == 1
        assert ar._turn_detector_prediction_fut is not None

    async def test_resumed_speech_without_sos_rearms_next_pause(self) -> None:
        """A short intra-segment pause can resolve a prediction before Silero
        emits EOS. When speech resumes without a new SOS, the cached
        prediction must be dropped so the next pause gets a fresh window."""
        ar = _make_full_recognition_for_eou()
        ar._speaking = True

        await ar._on_vad_event(_inference_done(raw_speech=0.0, raw_silence=0.3))
        first_fut = ar._turn_detector_prediction_fut
        assert first_fut is not None
        first_fut.set_result(
            TurnDetectionEvent(
                type="eot_prediction",
                last_speaking_time=time.time(),
                end_of_turn_probability=0.1,
            )
        )

        await ar._on_vad_event(_inference_done(raw_speech=0.1, raw_silence=0.0))

        ar._turn_detector_stream.cancel_inference.assert_called_once_with()
        assert ar._turn_detector_prediction_fut is None

        await ar._on_vad_event(_inference_done(raw_speech=0.0, raw_silence=0.3))

        assert ar._turn_detector_stream.predict.call_count == 2
        assert ar._turn_detector_prediction_fut is not None
        assert ar._turn_detector_prediction_fut is not first_fut

    async def test_silence_tick_starts_request_while_agent_speaking(self) -> None:
        """The agent-speaking gate was dropped: the silence tick warms a
        prediction during the user's pause even while the agent is still
        speaking, so an overlapping/interrupting turn still gets an EOT
        window."""
        ar = _make_full_recognition_for_eou()
        ar._speaking = True
        ar._agent_speaking = True

        await ar._on_vad_event(_inference_done(raw_speech=0.0, raw_silence=0.3))

        assert ar._turn_detector_stream.predict.call_count == 1
        assert ar._turn_detector_prediction_fut is not None

    async def test_eos_consumes_silence_tick_request_without_predicting(self) -> None:
        """EOS no longer starts an inference request — the silence tick owns
        that. EOS consumes the already-armed future and runs the eou bounce."""
        ar = _make_full_recognition_for_eou()
        ar._speaking = True
        ar._vad_base_turn_detection = True
        fut, _ = _resolved_prediction(0.9)
        ar._turn_detector_prediction_fut = fut

        await ar._on_vad_event(_end_of_speech())

        assert ar._turn_detector_stream.predict.call_count == 0
        assert ar._turn_detector_prediction_fut is fut
        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task
        ar._hooks.on_eot_prediction.assert_called_once()

    async def test_eos_runs_eou_even_while_agent_speaking(self) -> None:
        """The agent-speaking gate was dropped from the EOS handler: the eou
        bounce runs regardless of agent speech. Whether anything commits is
        then decided downstream by the transcript/interruption guards, not by
        the VAD handler."""
        ar = _make_full_recognition_for_eou()
        ar._speaking = True
        ar._agent_speaking = True
        ar._vad_base_turn_detection = True
        fut, _ = _resolved_prediction(0.9)
        ar._turn_detector_prediction_fut = fut

        await ar._on_vad_event(_end_of_speech())

        assert ar._turn_detector_stream.predict.call_count == 0
        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task
        ar._hooks.on_eot_prediction.assert_called_once()

    async def test_sos_tears_down_request_and_rearms(self) -> None:
        ar = _make_full_recognition_for_eou()
        ar._turn_detector_prediction_fut = asyncio.Future()
        ar._turn_detector_flushed = True

        await ar._on_vad_event(_start_of_speech())

        ar._turn_detector_stream.cancel_inference.assert_called_once_with()
        assert ar._turn_detector_prediction_fut is None
        assert ar._turn_detector_flushed is False

    async def test_eos_never_starts_request(self) -> None:
        """Inference requests start exclusively on the silence tick. EOS does
        not start one (no-prediction turns commit on min_delay) and leaves a
        held future untouched for the eou bounce."""
        ar = _make_full_recognition_for_eou()

        await ar._on_vad_event(_end_of_speech())
        assert ar._turn_detector_stream.predict.call_count == 0
        assert ar._turn_detector_prediction_fut is None

        fut, _ = _resolved_prediction(0.9)
        ar._turn_detector_prediction_fut = fut
        await ar._on_vad_event(_end_of_speech())
        assert ar._turn_detector_stream.predict.call_count == 0
        assert ar._turn_detector_prediction_fut is fut

    async def test_late_stt_final_after_flush_short_circuits(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A late stt final after the turn was flushed must not start an
        inference request and must not commit a second turn (issue #6504):
        committing here would ship a turn with a null ``end_of_turn_probability``
        and split one utterance into two. It warns once, then logs at debug level,
        and leaves the transcript untouched so it folds into the next turn."""
        caplog.set_level(logging.WARNING, logger="livekit.agents")
        ar = _make_full_recognition_for_eou()
        ar._turn_detector_flushed = True
        ar._audio_transcript = "date needed now"
        chat_ctx = _make_chat_ctx_stub()

        for _ in range(2):
            ar._run_eou_detection(chat_ctx, trigger="stt")
            assert ar._end_of_turn_task is not None
            await ar._end_of_turn_task

        assert ar._turn_detector_stream.predict.call_count == 0
        ar._hooks.on_eot_prediction.assert_not_called()
        # the late transcript is genuinely skipped, never committed with a null prediction
        ar._hooks.on_end_of_turn.assert_not_called()
        # transcript is preserved so it merges into the next turn instead of being lost
        assert ar._audio_transcript == "date needed now"
        flush_warnings = [
            r for r in caplog.records if "after turn has been committed" in r.getMessage()
        ]
        assert len(flush_warnings) == 1

    async def test_predict_timeout_signals_fallback_and_drops_future(self) -> None:
        """A pending future timing out at the model-specific
        ``prediction_timeout`` commits without a prediction — no synthetic
        emission, no threshold lookup — and reports the timeout to the stream
        (first one promotes the cloud→local fallback)."""
        ar = _make_full_recognition_for_eou()
        ar._turn_detector_prediction_fut = asyncio.Future()
        chat_ctx = _make_chat_ctx_stub()

        ar._run_eou_detection(chat_ctx, trigger="vad")
        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task

        ar._turn_detector_stream.cancel_inference.assert_called_once_with(timed_out=True)
        assert ar._turn_detector_prediction_fut is None
        ar._hooks.on_eot_prediction.assert_not_called()
        ar._turn_detector_stream.unlikely_threshold.assert_not_called()
        ar._hooks.on_end_of_turn.assert_called_once()

    async def test_commit_flushes_stream_and_marks_turn_flushed(self) -> None:
        ar = _make_full_recognition_for_eou()
        ar._hooks.on_end_of_turn.return_value = True  # commit
        fut, _ = _resolved_prediction(0.9)  # confident → no max_delay extension
        ar._turn_detector_prediction_fut = fut
        chat_ctx = _make_chat_ctx_stub()

        ar._run_eou_detection(chat_ctx, trigger="vad")
        assert ar._end_of_turn_task is not None
        await ar._end_of_turn_task

        ar._turn_detector_stream.flush.assert_called_once_with(reason="turn committed")
        assert ar._turn_detector_prediction_fut is None
        assert ar._turn_detector_flushed is True


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
    """``TurnDetector`` needs ~200ms of trailing silence; the VAD must report
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
