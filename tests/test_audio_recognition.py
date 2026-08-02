"""Unit tests for ``AudioRecognition``.

Recognition owns the audio fan-out to STT/VAD/AMD/interruption, the wall-clock
anchors those signals produce, the end-of-turn timing metrics derived from them,
and the resource handoff between activities. Each concern gets a section below:

1. ``_push_audio`` fan-out: who receives the real frame vs. an STT substitute,
   and the anchors stamped on the way through.
2. ``_last_speaking_time`` precedence when an STT transcript lands: a VAD anchor
   outranks the provider's audio-stream timestamp.
3. ``_compute_end_of_turn_metrics``: crafted timestamps in, delays (or ``None``)
   out — no audio, no STT/VAD.
4. ``_user_turn_start`` / ``_speech_start_time`` lifecycle across VAD bursts.
5. ``aclose`` against pre-cancelled tasks.
6. STT pipeline / turn-detector stream reuse across an agent handoff.

Streaming turn-detection policy is a separate (``audio_eot``) suite; see
``test_audio_recognition_turn_detection.py``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from livekit import rtc
from livekit.agents import Agent, stt
from livekit.agents.types import NOT_GIVEN
from livekit.agents.utils import aio
from livekit.agents.vad import VADEvent, VADEventType
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import (
    AudioRecognition,
    _compute_end_of_turn_metrics,
    _STTPipeline,
)
from livekit.agents.voice.turn import _StreamingTurnDetector

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Audio fan-out (_push_audio)
# ---------------------------------------------------------------------------


def _make_frame(byte: int = 0x11, samples: int = 160, sample_rate: int = 16000) -> rtc.AudioFrame:
    data = bytes([byte, byte]) * samples
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


def _make_push_audio_recognition() -> AudioRecognition:
    """Build an AudioRecognition stub with just the attributes ``push_audio`` reads."""
    ar = object.__new__(AudioRecognition)
    ar._sample_rate = None  # type: ignore[attr-defined]
    ar._stt_pipeline = MagicMock()  # type: ignore[attr-defined]
    # the input anchor lives on the pipeline (see _STTPipeline.input_started_at)
    ar._stt_pipeline.input_started_at = None  # type: ignore[attr-defined]
    ar._vad_ch = MagicMock()  # type: ignore[attr-defined]
    ar._interruption_ch = MagicMock()  # type: ignore[attr-defined]
    ar._session = MagicMock()  # type: ignore[attr-defined]
    ar._turn_detector_stream = None  # type: ignore[attr-defined]
    return ar


def test_push_audio_routes_real_frame_everywhere_by_default() -> None:
    ar = _make_push_audio_recognition()
    frame = _make_frame()

    ar._push_audio(frame)

    ar._stt_pipeline.audio_ch.send_nowait.assert_called_once_with(frame)
    ar._vad_ch.send_nowait.assert_called_once_with(frame)
    ar._session.amd.push_audio.assert_called_once_with(frame)
    ar._interruption_ch.send_nowait.assert_called_once_with(frame)


def test_push_audio_substitutes_stt_frame_only_on_stt_path() -> None:
    ar = _make_push_audio_recognition()
    real = _make_frame(byte=0x11)
    silence = _make_frame(byte=0x00)

    ar._push_audio(real, stt_frame=silence)

    # STT pipeline sees the substitute (silence), nothing else does.
    ar._stt_pipeline.audio_ch.send_nowait.assert_called_once_with(silence)
    ar._vad_ch.send_nowait.assert_called_once_with(real)
    ar._session.amd.push_audio.assert_called_once_with(real)
    ar._interruption_ch.send_nowait.assert_called_once_with(real)


def test_push_audio_skips_optional_consumers_when_unset() -> None:
    ar = _make_push_audio_recognition()
    ar._stt_pipeline = None  # type: ignore[attr-defined]
    ar._vad_ch = None  # type: ignore[attr-defined]
    ar._interruption_ch = None  # type: ignore[attr-defined]
    ar._session.amd = None

    # Should not raise even when every downstream consumer is absent.
    ar._push_audio(_make_frame())


def test_push_audio_records_sample_rate_and_input_start() -> None:
    ar = _make_push_audio_recognition()
    frame = _make_frame(sample_rate=24000)

    ar._push_audio(frame)

    assert ar._sample_rate == 24000  # type: ignore[attr-defined]
    assert ar._input_started_at is not None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# `_last_speaking_time` anchor precedence on STT transcripts
# ---------------------------------------------------------------------------
#
# The anchor feeds transcription_delay / end_of_utterance_delay (see
# _compute_end_of_turn_metrics below). STT events carry provider audio-stream
# timestamps that are mapped to wall clock with a fixed offset, so any skew
# between the provider's audio clock and wall clock corrupts the mapping: when
# the mapped anchor runs ahead of `now` it is clamped to `now`, which makes
# transcription_delay collapse to ~0; when it lags it inflates instead.
#
# A VAD anchor is measured against the local clock and has neither failure mode,
# so it wins whenever one exists for the segment — including the session's
# default VAD. The STT timestamp is only a fallback for "no VAD at all" or "VAD
# missed this segment" (the anchor is reset to None on each turn commit).


def _make_stt_event_recognition(
    *,
    vad: object | None,
    input_started_at: float,
    turn_detection_mode: str = "vad",
) -> AudioRecognition:
    """Wire the attributes ``_on_stt_event`` touches for transcript / speech events."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._session.amd = None
    ar._hooks = MagicMock()
    ar._vad = vad
    # the session loads a default VAD when the user didn't pass one; it still
    # produces a usable anchor, so this flag must not gate anchor precedence
    ar._using_default_vad = True
    ar._turn_detection_mode = turn_detection_mode
    ar._user_turn_committed = False
    ar._vad_base_turn_detection = False
    ar._user_silence_ev = asyncio.Event()
    ar._speaking = False
    ar._interruption_enabled = False
    ar._transcript_buffer = []
    ar._backchannel_boundary = None
    ar._ignore_user_transcript_until = NOT_GIVEN
    ar._stt_request_ids = []
    ar._end_of_turn_task = None
    ar._audio_transcript = ""
    ar._audio_interim_transcript = ""
    ar._audio_preflight_transcript = ""
    ar._final_transcript_confidence = []
    ar._final_transcript_received = asyncio.Event()
    ar._last_language = None
    ar._last_final_transcript_time = None
    ar._turn_tracker = MagicMock()
    ar._sample_rate = None
    ar._vad_ch = None
    ar._interruption_ch = None
    ar._stt_pipeline = MagicMock()
    ar._stt_pipeline.input_started_at = input_started_at
    # audio timeline anchored at `input_started_at`, flowing at realtime; the mapping
    # itself is covered by TestAudioTimelineMapping below
    ar._stt_pipeline.wall_time_at = lambda audio_time: input_started_at + audio_time
    ar._check_user_turn_limit = MagicMock()  # type: ignore[method-assign]
    # STT-driven turn detection reaches the span + eou machinery; the anchor is
    # what these tests pin down, so the bounce itself is stubbed out
    ar._speech_start_time = None
    ar._user_turn_span = None
    ar._vad_speech_started = False
    ar._vad_stream = None
    ar._ensure_user_turn_span = MagicMock(  # type: ignore[method-assign]
        return_value=MagicMock(is_recording=MagicMock(return_value=False))
    )
    ar._run_eou_detection = MagicMock()  # type: ignore[method-assign]
    ar._turn_detector_stream = None
    ar._turn_detector_prediction_fut = None
    return ar


def _final_transcript(end_time: float) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(language="en", text="hello there", confidence=1.0, end_time=end_time)
        ],
    )


async def test_vad_anchor_survives_a_skewed_stt_timestamp() -> None:
    # the provider's audio clock runs 2s ahead of wall clock: end_time maps to a
    # wall clock in the future, which the STT anchor would clamp down to `now`
    now = time.time()
    ar = _make_stt_event_recognition(vad=MagicMock(), input_started_at=now - 10.0)
    vad_anchor = now - 0.6
    ar._last_speaking_time = vad_anchor

    await ar._on_stt_event(_final_transcript(end_time=12.0))

    assert ar._last_speaking_time == vad_anchor
    assert ar._last_final_transcript_time is not None
    # this is what the EOU metrics report as transcription_delay
    assert ar._last_final_transcript_time - ar._last_speaking_time == pytest.approx(0.6, abs=0.1)


async def test_stt_timestamp_used_when_vad_missed_the_segment() -> None:
    now = time.time()
    ar = _make_stt_event_recognition(vad=MagicMock(), input_started_at=now - 10.0)
    ar._last_speaking_time = None  # reset on the previous turn commit, VAD never fired

    await ar._on_stt_event(_final_transcript(end_time=9.4))

    assert ar._last_speaking_time == pytest.approx(now - 0.6, abs=0.1)


async def test_stt_timestamp_used_without_vad() -> None:
    # no VAD at all: the STT anchor must keep being refreshed, otherwise the
    # previous turn's stale value would be reported
    now = time.time()
    ar = _make_stt_event_recognition(vad=None, input_started_at=now - 10.0)
    ar._last_speaking_time = now - 30.0

    await ar._on_stt_event(_final_transcript(end_time=9.4))

    assert ar._last_speaking_time == pytest.approx(now - 0.6, abs=0.1)


async def test_preflight_transcript_keeps_the_vad_anchor() -> None:
    """Preflight transcripts refresh the anchor on the same rules as finals.

    They land *earlier* than the final, so letting a skewed preflight timestamp
    through would corrupt the anchor before the final even arrives.
    """
    now = time.time()
    ar = _make_stt_event_recognition(vad=MagicMock(), input_started_at=now - 10.0)
    vad_anchor = now - 0.6
    ar._last_speaking_time = vad_anchor

    await ar._on_stt_event(
        stt.SpeechEvent(
            type=stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(language="en", text="hello", confidence=1.0, end_time=12.0)
            ],
        )
    )

    assert ar._last_speaking_time == vad_anchor


async def test_preflight_transcript_falls_back_to_stt_without_vad() -> None:
    now = time.time()
    ar = _make_stt_event_recognition(vad=None, input_started_at=now - 10.0)
    ar._last_speaking_time = None

    await ar._on_stt_event(
        stt.SpeechEvent(
            type=stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(language="en", text="hello", confidence=1.0, end_time=9.4)
            ],
        )
    )

    assert ar._last_speaking_time == pytest.approx(now - 0.6, abs=0.1)


# --- STT-driven turn detection (turn_detection="stt") ----------------------
#
# Providers with native end-of-turn detection (Deepgram Flux, AssemblyAI) send
# END_OF_SPEECH instead of relying on VAD silence. The anchor rules are the same
# as for transcripts, but this is the path where "when did the user stop
# speaking" is decided by the provider, so it gets its own coverage.


def _stt_speech_event(type_: stt.SpeechEventType, end_time: float) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=type_,
        alternatives=[stt.SpeechData(language="en", text="", confidence=1.0, end_time=end_time)],
    )


async def test_stt_end_of_speech_keeps_the_vad_anchor() -> None:
    """A provider EOT signal must not overwrite a fresher VAD anchor.

    The provider's ``end_time`` is an audio-stream position mapped to wall clock
    through a fixed offset; when that mapping skews forward the anchor collapses
    onto the arrival instant, which zeroes transcription_delay *and* makes the
    endpointing wait start from the wrong moment.
    """
    now = time.time()
    ar = _make_stt_event_recognition(
        vad=MagicMock(), input_started_at=now - 10.0, turn_detection_mode="stt"
    )
    vad_anchor = now - 0.6
    ar._last_speaking_time = vad_anchor

    await ar._on_stt_event(_stt_speech_event(stt.SpeechEventType.END_OF_SPEECH, end_time=12.0))

    assert ar._last_speaking_time == vad_anchor
    assert ar._user_turn_committed is True
    ar._run_eou_detection.assert_called_once()
    assert ar._run_eou_detection.call_args.kwargs["trigger"] == "stt"


async def test_stt_end_of_speech_anchors_on_provider_time_without_vad() -> None:
    """With no VAD, the provider's EOT timestamp is the only signal available."""
    now = time.time()
    ar = _make_stt_event_recognition(
        vad=None, input_started_at=now - 10.0, turn_detection_mode="stt"
    )
    ar._last_speaking_time = None

    await ar._on_stt_event(_stt_speech_event(stt.SpeechEventType.END_OF_SPEECH, end_time=9.4))

    assert ar._last_speaking_time == pytest.approx(now - 0.6, abs=0.1)


async def test_stt_start_of_speech_seeds_the_anchor_unconditionally() -> None:
    """START_OF_SPEECH re-seeds the anchor even when a VAD anchor exists.

    This is deliberate — it opens a new turn, so the previous turn's anchor must
    not survive into it. Pinned because it is the one site that still writes a
    provider timestamp over a VAD value.
    """
    now = time.time()
    ar = _make_stt_event_recognition(
        vad=MagicMock(), input_started_at=now - 10.0, turn_detection_mode="stt"
    )
    ar._last_speaking_time = now - 30.0  # stale, from the previous turn

    await ar._on_stt_event(_stt_speech_event(stt.SpeechEventType.START_OF_SPEECH, end_time=9.4))

    assert ar._last_speaking_time == pytest.approx(now - 0.6, abs=0.1)
    assert ar._speaking is True


# ---------------------------------------------------------------------------
# End-of-turn timing metrics (_compute_end_of_turn_metrics)
# ---------------------------------------------------------------------------
#
# These exercise the computation in isolation, with crafted timestamps (no
# audio, no STT/VAD). They pin down the behaviour described in issue #6093:
# when the internal `_last_speaking_time` anchor (reported as
# `stopped_speaking_at`) is stale and predates the start of the current turn,
# the previous code emitted wildly inflated `transcription_delay` /
# `end_of_turn_delay` values (often >200s) instead of skipping the calculation.


def test_normal_turn_produces_small_bounded_delays() -> None:
    """A well-ordered turn yields the expected sub-second delays."""
    started = 1000.0
    stopped = 1005.0  # finished speaking 5s after starting
    last_final = 1005.2  # final transcript landed 0.2s later
    now = 1005.4  # turn committed 0.4s after the user stopped

    metrics = _compute_end_of_turn_metrics(
        speech_start_time=started,
        last_speaking_time=stopped,
        last_final_transcript_time=last_final,
        now=now,
    )

    assert metrics.started_speaking_at == started
    assert metrics.stopped_speaking_at == stopped
    assert metrics.transcription_delay == pytest.approx(0.2)
    assert metrics.end_of_turn_delay == pytest.approx(0.4)


def test_stale_anchor_predating_turn_start_is_skipped() -> None:
    """Regression for issue #6093.

    When the turn detector commits a turn whose ``_last_speaking_time`` anchor was
    never refreshed for this segment, the anchor can be from a much earlier point
    in the session and predate ``speech_start_time``. The old code passed the
    not-None guard and computed ``end_of_turn_delay = now - last_speaking_time``,
    yielding ~220s. The metric must instead be skipped (left as ``None``) rather
    than reported as a bogus huge value.
    """
    # numbers mirror the issue payload: stopped_speaking_at ~220s before the start
    started = 1781342804.815377
    stopped = 1781342584.6181495  # ~220s BEFORE `started` — stale anchor
    last_final = 1781342804.9027314
    now = 1781342804.9027314

    metrics = _compute_end_of_turn_metrics(
        speech_start_time=started,
        last_speaking_time=stopped,
        last_final_transcript_time=last_final,
        now=now,
    )

    assert metrics.started_speaking_at is None
    assert metrics.stopped_speaking_at is None
    assert metrics.transcription_delay is None
    assert metrics.end_of_turn_delay is None


def test_anchor_equal_to_start_is_accepted() -> None:
    """An anchor exactly at the turn start is valid (boundary, delay == 0)."""
    started = 2000.0
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=started,
        last_speaking_time=started,
        last_final_transcript_time=started,
        now=started + 0.3,
    )

    assert metrics.started_speaking_at == started
    assert metrics.stopped_speaking_at == started
    assert metrics.transcription_delay == 0.0
    assert metrics.end_of_turn_delay == pytest.approx(0.3)


def test_missing_speaking_anchor_skips_everything() -> None:
    """Without the anchor every delay is unmeasurable (unreliable VAD/STT timing)."""
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=1000.0,
        last_speaking_time=None,
        last_final_transcript_time=1005.2,
        now=1006.0,
    )

    assert metrics.started_speaking_at is None
    assert metrics.stopped_speaking_at is None
    assert metrics.transcription_delay is None
    assert metrics.end_of_turn_delay is None


def test_unknown_turn_start_still_reports_the_transcription_delay() -> None:
    """A turn opened by a transcript that arrived after the previous commit has no
    turn-start anchor of its own (it is cleared on commit, and the late fragment
    never got a start-of-speech). How long transcription took is still measurable
    from the two transcript-side anchors, so it must not be dropped — reporting it
    as ``None`` here is what surfaced downstream as a hard ``0.0``.
    """
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=None,
        last_speaking_time=1005.0,
        last_final_transcript_time=1005.2,
        now=1005.3,
    )

    assert metrics.started_speaking_at is None  # genuinely unknown
    assert metrics.stopped_speaking_at == 1005.0
    assert metrics.transcription_delay == pytest.approx(0.2)
    assert metrics.end_of_turn_delay == pytest.approx(0.3)


def test_missing_final_transcript_still_reports_the_end_of_turn_delay() -> None:
    """A turn committed without a transcript (manual commit, empty STT) still has a
    measurable end-of-turn delay; only the transcription delay is unknown."""
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=1000.0,
        last_speaking_time=1005.0,
        last_final_transcript_time=None,
        now=1006.0,
    )

    assert metrics.started_speaking_at == 1000.0
    assert metrics.stopped_speaking_at == 1005.0
    assert metrics.transcription_delay is None
    assert metrics.end_of_turn_delay == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# `_user_turn_start` lifecycle across VAD bursts
# ---------------------------------------------------------------------------


class TestUserTurnStartPersistence:
    """``AudioRecognition._user_turn_start`` reflects the turn-level start of
    speech and persists across multiple VAD bursts within the same logical turn.

    Within a single user turn the VAD can produce several
    START_OF_SPEECH/END_OF_SPEECH cycles separated by short silences (e.g. the
    user says "Hello." then pauses briefly before continuing with the rest of
    their utterance). End-of-turn detection is decoupled from VAD: a turn is
    only considered ended once the EOT logic in ``_bounce_eou_task`` runs and
    clears the per-turn state.

    ``_speech_start_time`` reflects the *latest* VAD burst start (it is
    overwritten by every new SOS) and is used as the start of the per-burst
    ``user_speaking`` OTEL spans. ``_user_turn_start`` is set alongside the
    ``_user_turn_span`` on the first SOS of a turn and cleared together with the
    span on EOT cleanup. It is the value passed into ``_bounce_eou_task`` and
    ultimately ends up as ``started_speaking_at`` on the EOT metrics report.
    """

    pytestmark = [pytest.mark.virtual_time, pytest.mark.no_concurrent]

    def _create_audio_recognition(self) -> AudioRecognition:
        """Create an AudioRecognition instance with mocked dependencies."""
        with patch.object(AudioRecognition, "__init__", lambda self, *args, **kwargs: None):
            audio_recognition = AudioRecognition.__new__(AudioRecognition)

        # state read/written by _on_vad_event SOS/EOS branches
        audio_recognition._speech_start_time = None
        audio_recognition._vad_speech_started = False
        # _speaking is a property backed by this event (set == silent/not speaking)
        audio_recognition._user_silence_ev = asyncio.Event()
        audio_recognition._speaking = False
        audio_recognition._agent_speaking = False
        audio_recognition._turn_detector_stream = None
        audio_recognition._end_of_turn_task = None
        audio_recognition._user_turn_span = None
        audio_recognition._user_turn_start = None
        audio_recognition._user_turn_committed = False
        # disable EOU detection from EOS branch — we're testing VAD state, not EOT
        audio_recognition._vad_base_turn_detection = False
        audio_recognition._turn_detection_mode = None
        audio_recognition._stt = None
        audio_recognition._stt_model = None
        audio_recognition._stt_provider = None
        audio_recognition._audio_transcript = ""
        audio_recognition._last_speaking_time = None

        # collaborators
        audio_recognition._hooks = MagicMock()
        audio_recognition._session = MagicMock()
        audio_recognition._session.amd = None
        audio_recognition._session._room_io = None

        return audio_recognition

    @staticmethod
    def _vad_event(
        type_: VADEventType,
        *,
        speech_duration: float = 0.0,
        silence_duration: float = 0.0,
        inference_duration: float = 0.0,
    ) -> VADEvent:
        return VADEvent(
            type=type_,
            samples_index=0,
            timestamp=time.time(),
            speech_duration=speech_duration,
            silence_duration=silence_duration,
            inference_duration=inference_duration,
        )

    async def test_first_sos_sets_user_turn_start(self):
        """A single START_OF_SPEECH event sets _user_turn_start to the
        back-calculated burst start (time.time() - speech_duration - inference_duration).
        """
        audio_recognition = self._create_audio_recognition()

        before = time.time()
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=5.0)
        )
        after = time.time()

        assert audio_recognition._user_turn_start is not None
        assert before - 5.0 - 0.5 <= audio_recognition._user_turn_start <= after - 5.0 + 0.5

    async def test_user_turn_start_persists_across_intra_turn_bursts(self):
        """
        Within a single turn, VAD may fire multiple START_OF_SPEECH/END_OF_SPEECH
        cycles before EOT detection commits the turn. `_user_turn_start` must
        reflect the *first* burst's start and persist across subsequent bursts —
        it is only cleared by the EOT cleanup in `_bounce_eou_task`, alongside
        the `_user_turn_span` it travels with.

        Sequence:
            SOS (burst 1, speech_duration=5.0)  →  _user_turn_start = T1
            EOS (burst 1)
            SOS (burst 2, speech_duration=0.0)  →  _user_turn_start should remain T1

        `_speech_start_time` (per-burst, used for OTEL spans) is allowed to be
        overwritten by the second SOS — that's a separate concern.
        """
        audio_recognition = self._create_audio_recognition()

        # Burst 1 — speech started ~5s before this event fired
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=5.0)
        )
        first_burst_start = audio_recognition._user_turn_start
        assert first_burst_start is not None

        # End of burst 1
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.END_OF_SPEECH, speech_duration=5.0, silence_duration=0.6)
        )

        # Brief silence between bursts — same logical turn (no EOT yet)
        await asyncio.sleep(0.05)

        # Burst 2 — speech started "right now" (speech_duration=0)
        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=0.0)
        )

        assert audio_recognition._user_turn_start == pytest.approx(first_burst_start, abs=0.01), (
            "_user_turn_start was overwritten by the second SOS within the same turn. "
            f"Expected {first_burst_start:.3f}, got {audio_recognition._user_turn_start:.3f}. "
            "It should only be cleared by the EOT cleanup in _bounce_eou_task."
        )

    async def test_speech_start_time_updates_per_burst(self):
        """
        `_speech_start_time` is per-burst by design (used as the start of OTEL
        `user_speaking` spans), so it *should* update when a new SOS fires
        after an EOS. This test pins down that behaviour so we don't regress it.
        """
        audio_recognition = self._create_audio_recognition()

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=5.0)
        )
        first_burst_speech_start = audio_recognition._speech_start_time
        assert first_burst_speech_start is not None

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.END_OF_SPEECH, speech_duration=5.0, silence_duration=0.6)
        )

        await asyncio.sleep(0.05)

        await audio_recognition._on_vad_event(
            self._vad_event(VADEventType.START_OF_SPEECH, speech_duration=0.0)
        )

        # _speech_start_time should now reflect the second burst's start, not the first
        assert audio_recognition._speech_start_time is not None
        assert audio_recognition._speech_start_time > first_burst_speech_start


# ---------------------------------------------------------------------------
# aclose against pre-cancelled tasks
# ---------------------------------------------------------------------------


class TestAudioRecognitionAclose:
    """``AudioRecognition.aclose()`` handles pre-cancelled tasks gracefully.

    Before the fix, if ``_commit_user_turn_atask`` or ``_end_of_turn_task`` were
    cancelled before ``aclose()`` was called, awaiting them would raise
    CancelledError and propagate up, causing cleanup to fail. The fix wraps
    these awaits in try-except blocks to catch CancelledError.
    """

    pytestmark = [pytest.mark.virtual_time, pytest.mark.no_concurrent]

    def _create_audio_recognition(self) -> AudioRecognition:
        """Create an AudioRecognition instance with mocked dependencies."""
        with patch.object(AudioRecognition, "__init__", lambda self, *args, **kwargs: None):
            audio_recognition = AudioRecognition.__new__(AudioRecognition)

        # Initialize required attributes manually
        audio_recognition._session = MagicMock()
        audio_recognition._hooks = MagicMock()
        audio_recognition._closing = asyncio.Event()
        audio_recognition._tasks = set()
        audio_recognition._stt_consumer_atask = None
        audio_recognition._stt_pipeline = None
        audio_recognition._vad_atask = None
        audio_recognition._interruption_atask = None
        audio_recognition._turn_detector_stream = None
        audio_recognition._commit_user_turn_atask = None
        audio_recognition._end_of_turn_task = None
        audio_recognition._vad_ch = None
        audio_recognition._interruption_ch = None
        audio_recognition._audio_input_atask = None
        audio_recognition._backchannel_boundary_timer = None
        audio_recognition._AudioRecognition__stt_context = None

        return audio_recognition

    async def test_unprotected_await_blocks_subsequent_cleanup(self):
        """
        DEMONSTRATES THE BUG: Mimics the old aclose() pattern where CancelledError
        on the first task prevents the second task from ever being cleaned up.

        Old aclose() pattern:
            if self._commit_user_turn_atask is not None:
                await self._commit_user_turn_atask  # <-- raises CancelledError!
            # ... other cleanup ...
            if self._end_of_turn_task is not None:
                await self._end_of_turn_task  # <-- NEVER REACHED

        This leaves _end_of_turn_task orphaned and never awaited.
        """

        async def long_running_task():
            await asyncio.sleep(10)

        # Create two tasks like aclose() has
        commit_user_turn_atask = asyncio.create_task(long_running_task())
        end_of_turn_task = asyncio.create_task(long_running_task())

        # Cancel the first task (simulating external cancellation before aclose)
        commit_user_turn_atask.cancel()
        await asyncio.sleep(0)

        second_task_awaited = False

        async def old_aclose_pattern():
            """Mimics the OLD aclose() without try-except protection."""
            nonlocal second_task_awaited

            # First await - this raises CancelledError and exits
            if commit_user_turn_atask is not None:
                await commit_user_turn_atask

            # Second await - NEVER REACHED due to exception above
            if end_of_turn_task is not None:
                second_task_awaited = True
                end_of_turn_task.cancel()
                try:
                    await end_of_turn_task
                except asyncio.CancelledError:
                    pass

        # Run old_aclose_pattern with a timeout to prove it fails
        with pytest.raises(asyncio.CancelledError):
            await old_aclose_pattern()

        # The second task was never awaited - it's orphaned
        assert not second_task_awaited, "Second task cleanup was never reached"
        assert not end_of_turn_task.done(), "Second task is still running (orphaned)"

        # Cleanup the orphaned task
        end_of_turn_task.cancel()
        try:
            await end_of_turn_task
        except asyncio.CancelledError:
            pass

    async def test_aclose_handles_precancelled_tasks_gracefully(self):
        """
        PROVES THE FIX: Both tasks are properly cleaned up even when pre-cancelled.

        Fixed aclose() pattern:
            if self._commit_user_turn_atask is not None:
                try:
                    await self._commit_user_turn_atask
                except asyncio.CancelledError:
                    pass  # <-- Catches the error, continues cleanup
            # ... other cleanup ...
            if self._end_of_turn_task is not None:
                try:
                    await self._end_of_turn_task
                except asyncio.CancelledError:
                    pass  # <-- This is now reached!
        """
        audio_recognition = self._create_audio_recognition()

        async def long_running_task():
            await asyncio.sleep(10)

        # Create and cancel both tasks before aclose()
        commit_task = asyncio.create_task(long_running_task())
        commit_task.cancel()

        end_of_turn_task = asyncio.create_task(long_running_task())
        end_of_turn_task.cancel()

        await asyncio.sleep(0)

        audio_recognition._commit_user_turn_atask = commit_task
        audio_recognition._end_of_turn_task = end_of_turn_task

        # With the fix, aclose() completes without raising
        await audio_recognition._aclose()

        # Verify cleanup completed
        assert audio_recognition._closing.is_set()
        # Both tasks are now done (not orphaned)
        assert commit_task.done()
        assert end_of_turn_task.done()


# ---------------------------------------------------------------------------
# Resource reuse across an agent handoff
# ---------------------------------------------------------------------------


def _make_activity(agent: Agent, stt: object, turn_detection: object = None) -> MagicMock:
    act = MagicMock(spec=AgentActivity)
    act.agent = agent
    act._audio_recognition = MagicMock()
    act._audio_recognition._detach_stt = AsyncMock(return_value=MagicMock())
    act._audio_recognition._detach_turn_detector = MagicMock(return_value=MagicMock())
    type(act).stt = PropertyMock(return_value=stt)
    # turn detector reuse checks read this; None disables the reuse branch
    act._turn_detection = turn_detection
    # rt session reuse checks need these
    act._rt_session = None
    type(act).llm = PropertyMock(return_value=None)
    type(act).tools = PropertyMock(return_value=[])
    return act


async def _detach_stt_if_reusable(old: MagicMock, new: MagicMock) -> object | None:
    """Call the real _detach_reusable_resources, return stt_pipeline."""
    resources = await AgentActivity._detach_reusable_resources(old, new)
    return resources.stt_pipeline


async def _detach_turn_detector_if_reusable(old: MagicMock, new: MagicMock) -> object | None:
    """Call the real _detach_reusable_resources, return turn_detector_stream."""
    resources = await AgentActivity._detach_reusable_resources(old, new)
    return resources.turn_detector_stream


def _stub_recognition() -> AudioRecognition:
    ar = object.__new__(AudioRecognition)
    ar._stt_consumer_atask = None  # type: ignore[attr-defined]
    ar._stt_pipeline = None  # type: ignore[attr-defined]
    ar._transcript_buffer = MagicMock()  # type: ignore[attr-defined]
    ar._ignore_user_transcript_until = NOT_GIVEN  # type: ignore[attr-defined]
    return ar


class TestSTTPipelineReuse:
    """STT pipeline reuse via ``_detach_reusable_resources``."""

    pytestmark = pytest.mark.concurrent

    async def test_reusable_same_class_same_stt(self) -> None:
        """Two plain Agent instances sharing the same STT object → reusable."""
        shared_stt = MagicMock()
        old = _make_activity(Agent(instructions="a"), shared_stt)
        new = _make_activity(Agent(instructions="b"), shared_stt)

        result = await _detach_stt_if_reusable(old, new)
        assert result is not None  # detach_stt was called
        old._audio_recognition._detach_stt.assert_awaited_once()

    async def test_not_reusable_different_stt_instance(self) -> None:
        """Different STT instances (different connections) → not reusable."""
        old = _make_activity(Agent(instructions="a"), MagicMock())
        new = _make_activity(Agent(instructions="b"), MagicMock())

        result = await _detach_stt_if_reusable(old, new)
        assert result is None

    async def test_not_reusable_no_stt(self) -> None:
        """Either side missing STT → not reusable."""
        shared_stt = MagicMock()
        old = _make_activity(Agent(instructions="a"), None)
        new = _make_activity(Agent(instructions="b"), shared_stt)

        result = await _detach_stt_if_reusable(old, new)
        assert result is None

    async def test_not_reusable_different_stt_node_override(self) -> None:
        """Both subclasses define their own stt_node override → not reusable."""
        shared_stt = MagicMock()

        class AgentA(Agent):
            def stt_node(
                self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
            ) -> None:
                return None

        class AgentB(Agent):
            def stt_node(
                self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
            ) -> None:
                return None

        old = _make_activity(AgentA(instructions="a"), shared_stt)
        new = _make_activity(AgentB(instructions="b"), shared_stt)

        result = await _detach_stt_if_reusable(old, new)
        assert result is None

    async def test_not_reusable_subclass_inherits_custom_stt_node(self) -> None:
        """Both agents share a custom stt_node via inheritance → not reusable.

        The pipeline is bound to the old agent's `self`; a custom stt_node may access
        self.session/activity inside the yield loop, which raises after detach.
        Only the default Agent.stt_node is known to be safe to reuse.
        """
        shared_stt = MagicMock()

        class AgentA(Agent):
            def stt_node(
                self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
            ) -> None:
                return None

        class AgentB(AgentA):
            pass  # inherits AgentA.stt_node without overriding

        assert AgentA.stt_node is AgentB.stt_node  # sanity check

        old = _make_activity(AgentA(instructions="a"), shared_stt)
        new = _make_activity(AgentB(instructions="b"), shared_stt)

        result = await _detach_stt_if_reusable(old, new)
        assert result is None

    async def test_not_reusable_subclass_overrides_stt_node(self) -> None:
        """Old agent has stt_node; new agent's subclass overrides it differently → not reusable."""
        shared_stt = MagicMock()

        class AgentA(Agent):
            def stt_node(
                self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
            ) -> None:
                return None

        class AgentB(AgentA):
            def stt_node(  # type: ignore[override]
                self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
            ) -> None:
                return None

        old = _make_activity(AgentA(instructions="a"), shared_stt)
        new = _make_activity(AgentB(instructions="b"), shared_stt)

        result = await _detach_stt_if_reusable(old, new)
        assert result is None

    async def test_not_reusable_no_audio_recognition(self) -> None:
        """Old activity has no audio recognition → not reusable."""
        shared_stt = MagicMock()
        old = _make_activity(Agent(instructions="a"), shared_stt)
        old._audio_recognition = None
        new = _make_activity(Agent(instructions="b"), shared_stt)

        result = await _detach_stt_if_reusable(old, new)
        assert result is None


class TestTurnDetectorStreamReuse:
    """Turn detector stream reuse via ``_detach_reusable_resources``."""

    pytestmark = pytest.mark.concurrent

    async def test_turn_detector_reusable_same_instance(self) -> None:
        """Same TurnDetector instance carries over → live stream is detached for reuse."""
        shared_detector = MagicMock(spec=_StreamingTurnDetector)
        old = _make_activity(Agent(instructions="a"), MagicMock(), turn_detection=shared_detector)
        new = _make_activity(Agent(instructions="b"), MagicMock(), turn_detection=shared_detector)

        result = await _detach_turn_detector_if_reusable(old, new)
        assert result is not None
        old._audio_recognition._detach_turn_detector.assert_called_once()

    async def test_turn_detector_not_reusable_different_instance(self) -> None:
        """Different detector instances → not reusable (old stream torn down normally)."""
        old = _make_activity(
            Agent(instructions="a"),
            MagicMock(),
            turn_detection=MagicMock(spec=_StreamingTurnDetector),
        )
        new = _make_activity(
            Agent(instructions="b"),
            MagicMock(),
            turn_detection=MagicMock(spec=_StreamingTurnDetector),
        )

        result = await _detach_turn_detector_if_reusable(old, new)
        assert result is None
        old._audio_recognition._detach_turn_detector.assert_not_called()

    async def test_turn_detector_not_reusable_when_new_opts_out(self) -> None:
        """New agent resolves to no turn detection (e.g. realtime server-side) → not reusable."""
        shared_detector = MagicMock(spec=_StreamingTurnDetector)
        old = _make_activity(Agent(instructions="a"), MagicMock(), turn_detection=shared_detector)
        new = _make_activity(Agent(instructions="b"), MagicMock(), turn_detection=None)

        result = await _detach_turn_detector_if_reusable(old, new)
        assert result is None
        old._audio_recognition._detach_turn_detector.assert_not_called()


class TestInputAnchorAcrossHandoff:
    """The input-time anchor (``end_time=0`` wall clock) survives pipeline reuse."""

    pytestmark = pytest.mark.concurrent

    async def test_input_anchor_preserved_when_pipeline_reused(self) -> None:
        """_update_stt must not reset a reused pipeline's input anchor.

        The STT ``end_time`` clock is relative to the original stream start. Re-anchoring
        ``input_started_at`` to the handoff time would desync the two and push the derived
        speaking time minutes into the future (see the 68s end-of-turn stall).
        """
        reused = object.__new__(_STTPipeline)
        reused.input_started_at = 1000.0  # anchored during the previous activity
        ch = aio.Chan()  # type: ignore[var-annotated]
        ch.close()  # closed channel → the swapped-in consumer exits immediately
        reused._event_ch = ch  # type: ignore[attr-defined]

        ar = _stub_recognition()
        ar._update_stt(MagicMock(), pipeline=reused)

        assert ar._input_started_at == 1000.0  # carried over, not reset to None
        if ar._stt_consumer_atask is not None:
            await ar._stt_consumer_atask

    async def test_reused_pipeline_rebinds_stt_node(self) -> None:
        """_update_stt must rebind a reused pipeline to the current activity's node.

        The pipeline is created bound to the agent that first started it. After a
        close-based handoff that agent's activity is torn down, so recreating the
        stream through the stale node would raise and stop the pump, leaving the
        agent permanently deaf. Reuse must point recreation at the live node.
        """
        reused = object.__new__(_STTPipeline)
        reused.input_started_at = None  # type: ignore[attr-defined]
        reused._stt_node = MagicMock(name="previous_agent_node")  # type: ignore[attr-defined]
        ch = aio.Chan()  # type: ignore[var-annotated]
        ch.close()  # closed channel → the swapped-in consumer exits immediately
        reused._event_ch = ch  # type: ignore[attr-defined]

        new_node = MagicMock(name="current_agent_node")
        ar = _stub_recognition()
        ar._update_stt(new_node, pipeline=reused)

        assert reused._stt_node is new_node
        if ar._stt_consumer_atask is not None:
            await ar._stt_consumer_atask

    def test_input_anchor_reads_through_to_pipeline(self) -> None:
        """The anchor lives on the pipeline so it travels with the stream across handoff."""
        ar = _stub_recognition()
        assert ar._input_started_at is None  # no pipeline attached yet

        pipeline = object.__new__(_STTPipeline)
        pipeline.input_started_at = 1234.5
        ar._stt_pipeline = pipeline  # type: ignore[attr-defined]

        assert ar._input_started_at == 1234.5  # read-only view of the pipeline's anchor


# ---------------------------------------------------------------------------
# Audio timeline -> wall clock mapping (_STTPipeline.wall_time_at)
# ---------------------------------------------------------------------------


class TestAudioTimelineMapping:
    """STT timestamps are positions on the audio timeline the pipeline fed the
    provider; turning one into a wall clock is what ``transcription_delay`` and
    the barge-in hold window are built on.

    The mapping is anchored on the most recent push, so it only depends on audio
    delivered between that position and now. A one-shot anchor (first frame +
    elapsed wall clock) instead accumulates every divergence between the two
    timelines and never recovers.
    """

    pytestmark = [pytest.mark.virtual_time, pytest.mark.no_concurrent]

    @staticmethod
    def _pipeline() -> _STTPipeline:
        pipeline = object.__new__(_STTPipeline)
        pipeline.input_started_at = None
        pipeline._audio_duration = 0.0
        pipeline._wall_at_audio_end = 0.0
        return pipeline

    @classmethod
    async def _push(cls, pipeline: _STTPipeline, seconds: float, *, realtime: bool = True) -> None:
        """Feed `seconds` of audio in 20ms steps, optionally consuming wall clock.

        A frame carrying 20ms of audio arrives *after* those 20ms have elapsed, so
        the sleep comes first — that ordering is what makes the last push land on
        ``now``, exactly as it does in the live pipeline.
        """
        steps = int(seconds / 0.02)
        for _ in range(steps):
            if realtime:
                await asyncio.sleep(0.02)
            if pipeline.input_started_at is None:
                pipeline.input_started_at = time.time() - 0.02
            pipeline.note_audio_pushed(0.02)

    async def test_unmapped_before_audio_starts(self) -> None:
        assert self._pipeline().wall_time_at(1.0) is None

    async def test_realtime_delivery_maps_to_the_matching_wall_clock(self) -> None:
        pipeline = self._pipeline()
        await self._push(pipeline, 5.0)

        now = time.time()
        # the provider transcribed up to 0.6s before the audio we have sent
        assert pipeline.wall_time_at(pipeline.audio_duration - 0.6) == pytest.approx(
            now - 0.6, abs=0.01
        )

    async def test_a_stalled_input_does_not_bias_later_positions(self) -> None:
        """A muted or stalled track advances wall clock without advancing audio.

        A one-shot anchor would report every subsequent utterance as ~30s late;
        re-anchoring on the last push confines the gap to positions behind it.
        """
        pipeline = self._pipeline()
        await self._push(pipeline, 1.0)
        await asyncio.sleep(30.0)  # track muted: no frames
        await self._push(pipeline, 2.0)

        now = time.time()
        assert pipeline.wall_time_at(pipeline.audio_duration - 0.5) == pytest.approx(
            now - 0.5, abs=0.01
        )

    async def test_a_burst_does_not_push_positions_into_the_future(self) -> None:
        """Catch-up burst: 2s of audio delivered in ~no wall clock.

        The old fixed offset mapped later positions ahead of ``now``, where the
        ``min(..., now)`` clamp silently turned transcription_delay into 0.
        """
        pipeline = self._pipeline()
        await self._push(pipeline, 1.0)
        await self._push(pipeline, 2.0, realtime=False)  # jitter-buffer catch-up
        await self._push(pipeline, 1.0)

        now = time.time()
        assert pipeline.wall_time_at(pipeline.audio_duration) <= now + 0.01
        assert pipeline.wall_time_at(pipeline.audio_duration - 0.4) == pytest.approx(
            now - 0.4, abs=0.01
        )

    async def test_synthetic_audio_keeps_earlier_positions_anchored(self) -> None:
        """The manual-commit flush injects silence far faster than realtime.

        It has to count toward the timeline (the provider's positions include it),
        but must not drag everything behind it into the past — the user stopped
        speaking when they stopped, not 2s earlier.
        """
        pipeline = self._pipeline()
        await self._push(pipeline, 3.0)
        speech_end_position = pipeline.audio_duration - 0.5
        expected = pipeline.wall_time_at(speech_end_position)

        for _ in range(10):  # 2s of flush silence, no wall clock consumed
            pipeline.note_audio_pushed(0.2, synthetic=True)

        assert pipeline.wall_time_at(speech_end_position) == pytest.approx(expected, abs=0.001)

    async def test_burst_delivered_audio_does_not_collapse_the_anchor(self) -> None:
        """End to end through ``_push_audio`` + ``_on_stt_event``.

        This is the original failure: audio arriving faster than realtime pushed
        the fixed-offset mapping ahead of ``now``, ``min(..., now)`` clamped it to
        the transcript-arrival instant, and transcription_delay reported ~0.
        """
        ar = _make_stt_event_recognition(vad=None, input_started_at=0.0)
        pipeline = object.__new__(_STTPipeline)
        pipeline.input_started_at = None
        pipeline._audio_duration = 0.0
        pipeline._wall_at_audio_end = 0.0
        pipeline._audio_ch = aio.Chan()  # type: ignore[var-annotated]
        ar._stt_pipeline = pipeline
        ar._last_speaking_time = None

        async def push(seconds: float, *, realtime: bool) -> None:
            for _ in range(int(seconds / 0.02)):
                if realtime:
                    await asyncio.sleep(0.02)
                ar._push_audio(_make_frame(samples=320))  # 20ms @ 16kHz

        await push(3.0, realtime=True)
        await push(2.0, realtime=False)  # jitter-buffer catch-up
        await push(1.0, realtime=True)

        # the provider transcribed up to 0.4s behind the audio it has received
        await ar._on_stt_event(_final_transcript(end_time=pipeline.audio_duration - 0.4))

        now = time.time()
        assert ar._last_speaking_time == pytest.approx(now - 0.4, abs=0.02)
        assert ar._last_final_transcript_time is not None
        assert ar._last_final_transcript_time - ar._last_speaking_time == pytest.approx(
            0.4, abs=0.02
        )

    async def test_synthetic_audio_arriving_first_anchors_like_real_audio(self) -> None:
        """Nothing in the framework injects synthetic audio before real input (the
        flush needs a sample rate, which only a real frame sets), but anchoring off
        the never-set 0.0 wall time would map every position to the epoch.
        """
        pipeline = self._pipeline()
        pipeline.input_started_at = time.time()
        pipeline.note_audio_pushed(0.2, synthetic=True)

        assert pipeline.wall_time_at(pipeline.audio_duration) == pytest.approx(
            time.time(), abs=0.01
        )
