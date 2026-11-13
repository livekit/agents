"""The ``_last_speaking_time`` anchor behind ``transcription_delay``.

A VAD anchor wins whenever it exists; the provider timestamp takes over without one, and in
``turn_detection="stt"`` a real provider timestamp or ``END_OF_SPEECH`` replaces it.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from livekit.agents import stt
from livekit.agents.voice import audio_recognition
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
    ar._session._root_span_context = None
    ar._session.amd = None
    ar._session._room_io = None
    ar._hooks = MagicMock()
    ar._vad = vad
    ar._turn_detection_mode = mode
    ar._user_turn_committed = False
    ar._stt_turn_ended = False
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
    ar._stt_events = []
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
    ar._turn_detector_stream = None
    ar._eou_wait_span = None
    ar._eou_wait_started_at_ns = None
    ar._eou_wait_rearms = 0
    ar._eou_wait_floor_ns = None
    ar._eou_wait_not_committed = 0
    ar._eou_detection_span = None
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
async def test_vad_anchor_survives_a_transcript_without_timestamps(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transcript without timestamps keeps the VAD anchor rather than reporting ~0 delay."""
    now = time.time()
    # the transcript is stamped with the same clock the anchor was derived from
    monkeypatch.setattr(audio_recognition.time, "time", lambda: now)
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
async def test_skewed_provider_timestamp(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # the provider's audio clock runs ahead of wall clock: end_time maps into the
    # future and the estimate clamps to `now`, zeroing the delay
    now = time.time()
    # the clamp reads the clock again, so a slow runner would drift past the tolerance
    monkeypatch.setattr(audio_recognition.time, "time", lambda: now)
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0, mode=mode)
    vad_anchor = now - 0.6
    ar._last_speaking_time = vad_anchor

    await ar._on_stt_event(_final_transcript(end_time=12.0))

    if mode == "stt":
        # accepted: the provider owns the boundary here and a skewed timestamp is
        # indistinguishable from a good one, so the delay under-reports
        assert now <= ar._last_speaking_time <= time.time()
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
    """Preflight transcripts refresh the anchor on the same rules as finals."""
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
    """``speech_end_time`` is a wall-clock speech end, so it outranks word timestamps."""
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
    """An anchor past ``now`` would delay the turn commit by the clock skew, so it is clamped."""
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

    # clamped to arrival time: bounded by the clock reads around the call rather than a
    # fixed tolerance, which a loaded CI runner blows through
    assert now <= ar._last_speaking_time <= time.time()


async def test_stt_end_of_speech_without_timestamps_still_anchors_the_turn() -> None:
    """An untimed END_OF_SPEECH is an explicit boundary, so the anchor moves to its arrival."""
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0, mode="stt")
    ar._last_speaking_time = now - 0.6

    await ar._on_stt_event(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, alternatives=[]))

    # the anchor moved from 0.6 s ago to arrival time (bounded by the clock reads around the call)
    assert now <= ar._last_speaking_time <= time.time()
    assert ar._user_turn_committed is True


@pytest.mark.parametrize("endpoint", ["wall", "word", "arrival"])
@pytest.mark.parametrize("transcript", [_final_transcript, _preflight_transcript])
async def test_untimed_transcript_preserves_the_current_stt_endpoint(
    monkeypatch, endpoint, transcript
):
    now = 1010.0
    monkeypatch.setattr(audio_recognition.time, "time", lambda: now)
    ar = _make_recognition(vad=None, input_started_at=1000.0, mode="stt")
    await ar._on_stt_event(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
    eos = stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
    expected = now
    if endpoint == "wall":
        eos.speech_end_time = expected = 1009.4
    elif endpoint == "word":
        eos.alternatives = [stt.SpeechData(language="en", text="", end_time=9.4)]
        expected = 1009.4
    await ar._on_stt_event(eos)

    now = 1010.5
    await ar._on_stt_event(transcript(end_time=0.0))
    assert ar._last_speaking_time == pytest.approx(expected)
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=1008.0,
        last_speaking_time=ar._last_speaking_time,
        last_final_transcript_time=ar._last_final_transcript_time,
        now=now,
    )
    assert metrics.transcription_delay == pytest.approx(now - expected)

    now = 1012.0
    await ar._on_stt_event(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
    now = 1012.5
    await ar._on_stt_event(transcript(end_time=0.0))
    assert ar._last_speaking_time == now


@both_modes
async def test_untimed_eos_does_not_inherit_an_earlier_segment_final(mode: str) -> None:
    """A final can stabilize a segment while the utterance continues."""
    now = time.time()
    ar = _make_recognition(vad=None, input_started_at=now - 10.0, mode=mode)
    await ar._on_stt_event(_final_transcript(end_time=9.0))
    assert ar._last_speaking_time == pytest.approx(now - 1.0, rel=0, abs=1e-6)

    before_eos = time.time()
    await ar._on_stt_event(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    if mode == "stt":
        assert before_eos <= ar._last_speaking_time <= time.time()
    else:
        assert ar._last_speaking_time == pytest.approx(now - 1.0, rel=0, abs=1e-6)


@both_modes
async def test_wired_vad_speaking_state_reaches_the_transcript_hooks(mode: str) -> None:
    """The wired VAD's speaking state reaches the transcript hooks, never ``None``."""
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
