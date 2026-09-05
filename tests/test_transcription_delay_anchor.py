"""Regression tests for the ``_last_speaking_time`` anchor behind
``transcription_delay``.

``transcription_delay`` is (wall clock when the final transcript arrived) minus
(wall clock when the user stopped talking). The second term is the
``_last_speaking_time`` anchor: a VAD measures it against the local clock — the
session's auto-loaded default VAD included — and it is the better estimate
whenever it exists. The provider-derived value (``end_time + input_started_at``,
clamped to ``now``) takes over when there is no VAD anchor to beat: no VAD at
all, or a segment the VAD missed.

``turn_detection="stt"`` adds one exception: the provider owns the turn boundary
there, so a *real* provider timestamp replaces the VAD anchor, as does an explicit
``END_OF_SPEECH``. A transcript with a missing ``end_time`` does not — the estimate
would collapse to the transcript-arrival instant and report a ~0 delay.

Every test below runs under both turn detection modes; where the two disagree,
the expectation is spelled out per mode rather than duplicated into a second
test.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from livekit.agents import stt
from livekit.agents.voice.audio_recognition import (
    AudioRecognition,
    _compute_end_of_turn_metrics,
)

pytestmark = pytest.mark.unit

# the anchor rules are shared by both modes, so every test asserts both
both_modes = pytest.mark.parametrize("mode", ["vad", "stt"])


def _make_recognition(
    *, vad: object | None, input_started_at: float, mode: str = "vad"
) -> AudioRecognition:
    """Wire the attributes ``_on_stt_event`` touches for transcript events."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._session.amd = None
    ar._session._room_io = None
    ar._hooks = MagicMock()
    ar._vad = vad
    ar._turn_detection_mode = mode
    ar._user_turn_committed = False
    ar._vad_base_turn_detection = False
    ar._user_silence_ev = asyncio.Event()
    ar._speaking = False
    ar._interruption_enabled = False
    ar._agent_speaking = False
    ar._transcript_buffer = []
    ar._transcript_gate_active = False
    ar._stt_aligned_transcript = False
    ar._backchannel_boundary = None
    ar._turn_transcript_received = False
    ar._transcription_timeout_handle = None
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
    ar._last_speaking_time = None
    ar._sample_rate = None
    ar._vad_ch = None
    ar._interruption_ch = None
    ar._stt_pipeline = MagicMock()
    ar._stt_pipeline.input_started_at = input_started_at
    ar._check_user_turn_limit = MagicMock()  # type: ignore[method-assign]
    ar._speech_start_time = None
    # only reached by the stt-mode END_OF_SPEECH / START_OF_SPEECH branches
    ar._user_turn_span = None
    ar._user_turn_start = None
    ar._eot_wait_span = None
    ar._eot_wait_started_at = None
    ar._eot_wait_rearms = 0
    ar._stt_model = None
    ar._stt_provider = None
    ar._vad_stream = None
    ar._vad_speech_started = False
    ar._run_eou_detection = MagicMock()  # type: ignore[method-assign]
    return ar


def _speech_event(kind: stt.SpeechEventType, end_time: float, text: str) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=kind,
        alternatives=[stt.SpeechData(language="en", text=text, confidence=1.0, end_time=end_time)],
    )


def _final_transcript(end_time: float) -> stt.SpeechEvent:
    return _speech_event(stt.SpeechEventType.FINAL_TRANSCRIPT, end_time, "hello there")


def _preflight_transcript(end_time: float) -> stt.SpeechEvent:
    return _speech_event(stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, end_time, "hello")


@both_modes
async def test_vad_anchor_survives_a_transcript_without_timestamps(mode: str) -> None:
    """The reported regression: with the (default) VAD anchor discarded, a
    provider that sends no timestamps anchors the turn at transcript arrival and
    transcription_delay reports ~0 instead of the actual STT latency.

    stt turn detection is no exception — there is no provider timestamp to
    prefer, so the estimate would be nothing but ``now``.
    """
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0, mode=mode)
    vad_anchor = now - 0.6  # VAD end-of-speech, 0.6s of transcription latency ago
    ar._last_speaking_time = vad_anchor

    await ar._on_stt_event(_final_transcript(end_time=0.0))  # no provider timestamps

    assert ar._last_speaking_time == vad_anchor
    assert ar._last_final_transcript_time is not None

    metrics = _compute_end_of_turn_metrics(
        speech_start_time=now - 3.0,
        last_speaking_time=ar._last_speaking_time,
        last_final_transcript_time=ar._last_final_transcript_time,
        now=time.time(),
    )
    assert metrics.transcription_delay == pytest.approx(0.6, abs=0.1)


@both_modes
async def test_usable_provider_timestamp_replaces_the_anchor_only_in_stt_mode(mode: str) -> None:
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0, mode=mode)
    ar._last_speaking_time = now - 0.6  # VAD end-of-speech
    provider_anchor = now - 1.0  # end_time 9.0 against input_started_at now - 10.0

    await ar._on_stt_event(_final_transcript(end_time=9.0))

    if mode == "stt":
        # the provider endpoints the turn in this mode, so its timestamp wins
        assert ar._last_speaking_time == pytest.approx(provider_anchor, abs=0.05)
    else:
        assert ar._last_speaking_time == pytest.approx(now - 0.6, abs=0.05)


@both_modes
async def test_skewed_provider_timestamp(mode: str) -> None:
    # the provider's audio clock runs ahead of wall clock: end_time maps into the
    # future and the estimate clamps to `now`, zeroing the delay
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0, mode=mode)
    vad_anchor = now - 0.6
    ar._last_speaking_time = vad_anchor

    await ar._on_stt_event(_final_transcript(end_time=12.0))

    if mode == "stt":
        # accepted: the provider owns the boundary here and a skewed timestamp is
        # indistinguishable from a good one, so the delay under-reports
        assert ar._last_speaking_time == pytest.approx(now, abs=0.05)
    else:
        assert ar._last_speaking_time == vad_anchor


@both_modes
async def test_provider_timestamp_used_when_vad_missed_the_segment(mode: str) -> None:
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0, mode=mode)
    ar._last_speaking_time = None  # reset on the previous turn commit, VAD never fired

    await ar._on_stt_event(_final_transcript(end_time=9.4))

    assert ar._last_speaking_time == pytest.approx(now - 0.6, abs=0.1)


@both_modes
async def test_provider_timestamp_used_without_vad(mode: str) -> None:
    # no VAD at all: the provider estimate must keep refreshing the anchor,
    # otherwise the previous turn's stale value would be reported
    now = time.time()
    ar = _make_recognition(vad=None, input_started_at=now - 10.0, mode=mode)
    ar._last_speaking_time = now - 30.0

    await ar._on_stt_event(_final_transcript(end_time=9.4))

    assert ar._last_speaking_time == pytest.approx(now - 0.6, abs=0.1)


@both_modes
async def test_preflight_transcript_follows_the_same_rules(mode: str) -> None:
    """Preflight transcripts refresh the anchor on the same rules as finals.

    They land *earlier* than the final, so a discarded anchor here corrupts the
    turn before the final even arrives.
    """
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0, mode=mode)
    ar._last_speaking_time = now - 0.6

    await ar._on_stt_event(_preflight_transcript(end_time=0.0))  # no provider timestamps

    assert ar._last_speaking_time == pytest.approx(now - 0.6, abs=0.05)

    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0, mode=mode)
    ar._last_speaking_time = now - 0.6

    await ar._on_stt_event(_preflight_transcript(end_time=9.0))  # provider says now - 1.0

    expected = now - 1.0 if mode == "stt" else now - 0.6
    assert ar._last_speaking_time == pytest.approx(expected, abs=0.05)


async def test_stt_end_of_speech_anchors_on_a_real_provider_timestamp() -> None:
    """The stt-mode END_OF_SPEECH branch commits the turn, so it anchors it too."""
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0, mode="stt")
    ar._last_speaking_time = now - 0.6

    await ar._on_stt_event(_speech_event(stt.SpeechEventType.END_OF_SPEECH, end_time=9.0, text=""))

    assert ar._last_speaking_time == pytest.approx(now - 1.0, abs=0.05)
    assert ar._user_turn_committed is True


async def test_stt_end_of_speech_prefers_the_provider_speech_end_time() -> None:
    """``speech_end_time`` is the provider's wall-clock speech end — the same
    quantity a VAD anchor holds (``StreamAdapter`` builds it as
    ``now - silence_duration - inference_duration``). It outranks the word
    timestamps, which only bound the last recognized word.
    """
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0, mode="stt")
    ar._last_speaking_time = now - 0.6
    speech_end_time = now - 0.4

    await ar._on_stt_event(
        stt.SpeechEvent(
            type=stt.SpeechEventType.END_OF_SPEECH,
            # word timestamps also present, and pointing somewhere else
            alternatives=[stt.SpeechData(language="en", text="", confidence=1.0, end_time=9.0)],
            speech_end_time=speech_end_time,
        )
    )

    assert ar._last_speaking_time == speech_end_time
    assert ar._user_turn_committed is True


async def test_stt_end_of_speech_clamps_a_future_speech_end_time() -> None:
    """A provider clock running ahead would push the anchor past ``now``. The
    metrics survive that (both delays clamp at 0), but ``extra_sleep`` does not:
    it adds ``last_speaking_time - now`` to the endpointing delay, so an
    unclamped anchor delays the turn commit by the skew.
    """
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0, mode="stt")
    ar._last_speaking_time = now - 0.6

    await ar._on_stt_event(
        stt.SpeechEvent(
            type=stt.SpeechEventType.END_OF_SPEECH,
            alternatives=[],
            speech_end_time=now + 5.0,  # provider clock runs 5s ahead
        )
    )

    assert ar._last_speaking_time == pytest.approx(now, abs=0.05)
    assert ar._last_speaking_time <= time.time()


async def test_stt_end_of_speech_without_timestamps_still_anchors_the_turn() -> None:
    """END_OF_SPEECH commonly carries no alternatives, leaving arrival time as the
    only estimate — but unlike an untimestamped transcript it is an *explicit*
    endpointing signal, so the provider is taken at its word and the anchor moves.

    Contrast ``test_vad_anchor_survives_a_transcript_without_timestamps[stt]``,
    where the same missing ``end_time`` on a FINAL_TRANSCRIPT keeps the VAD anchor:
    there the provider said nothing about the boundary, here it said "now".
    """
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0, mode="stt")
    ar._last_speaking_time = now - 0.6

    await ar._on_stt_event(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, alternatives=[]))

    assert ar._last_speaking_time == pytest.approx(now, abs=0.05)
    assert ar._user_turn_committed is True


@both_modes
async def test_wired_vad_speaking_state_reaches_the_transcript_hooks(mode: str) -> None:
    """``speaking is False`` on the transcript hooks is what arms the
    false-interruption resume timer (see ``on_final_transcript`` in
    agent_activity). ``self._vad`` is the resolved, wired VAD — the session's
    auto-loaded default included — and it drives ``_speaking``, so its state is
    reported rather than ``None`` (which silently disables the resume path)."""
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0, mode=mode)
    ar._speaking = False

    await ar._on_stt_event(_final_transcript(end_time=0.0))

    assert ar._hooks.on_final_transcript.call_args.kwargs["speaking"] is False

    # without a VAD, stt turn detection still tracks speaking from the provider's
    # own start/end of speech events; any other mode has no state to report
    ar = _make_recognition(vad=None, input_started_at=now - 10.0, mode=mode)
    ar._speaking = False

    await ar._on_stt_event(_final_transcript(end_time=0.0))

    expected = False if mode == "stt" else None
    assert ar._hooks.on_final_transcript.call_args.kwargs["speaking"] is expected
