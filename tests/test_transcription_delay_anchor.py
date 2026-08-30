"""Regression tests for the ``_last_speaking_time`` anchor behind
``transcription_delay``.

``transcription_delay`` is (wall clock when the final transcript arrived) minus
(wall clock when the user stopped talking). The second term is the
``_last_speaking_time`` anchor: a VAD end-of-speech is measured against the
local clock and is authoritative — the session's auto-loaded default VAD
included. The provider-derived estimate (``end_time + input_started_at``,
clamped to ``now``) is only a fallback for "no VAD at all" or "VAD missed this
segment", because it collapses to the transcript-arrival instant whenever the
provider sends no timestamps — which reports a ~0 delay.
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


def _make_recognition(*, vad: object | None, input_started_at: float) -> AudioRecognition:
    """Wire the attributes ``_on_stt_event`` touches for transcript events."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._session.amd = None
    ar._hooks = MagicMock()
    ar._vad = vad
    ar._turn_detection_mode = "vad"
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
    return ar


def _final_transcript(end_time: float) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(language="en", text="hello there", confidence=1.0, end_time=end_time)
        ],
    )


async def test_default_vad_anchor_survives_a_transcript_without_timestamps() -> None:
    """The reported regression: with the (default) VAD anchor discarded, a
    provider that sends no timestamps anchors the turn at transcript arrival and
    transcription_delay reports ~0 instead of the actual STT latency."""
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0)
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


async def test_default_vad_anchor_survives_a_skewed_provider_timestamp() -> None:
    # the provider's audio clock runs ahead of wall clock: end_time maps into the
    # future and the fallback estimate would clamp to `now`, zeroing the delay
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0)
    vad_anchor = now - 0.6
    ar._last_speaking_time = vad_anchor

    await ar._on_stt_event(_final_transcript(end_time=12.0))

    assert ar._last_speaking_time == vad_anchor


async def test_provider_timestamp_used_when_vad_missed_the_segment() -> None:
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0)
    ar._last_speaking_time = None  # reset on the previous turn commit, VAD never fired

    await ar._on_stt_event(_final_transcript(end_time=9.4))

    assert ar._last_speaking_time == pytest.approx(now - 0.6, abs=0.1)


async def test_provider_timestamp_used_without_vad() -> None:
    # no VAD at all: the provider estimate must keep refreshing the anchor,
    # otherwise the previous turn's stale value would be reported
    now = time.time()
    ar = _make_recognition(vad=None, input_started_at=now - 10.0)
    ar._last_speaking_time = now - 30.0

    await ar._on_stt_event(_final_transcript(end_time=9.4))

    assert ar._last_speaking_time == pytest.approx(now - 0.6, abs=0.1)


async def test_preflight_transcript_keeps_the_vad_anchor() -> None:
    """Preflight transcripts refresh the anchor on the same rules as finals.

    They land *earlier* than the final, so a discarded anchor here corrupts the
    turn before the final even arrives.
    """
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0)
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


async def test_wired_vad_speaking_state_reaches_the_transcript_hooks() -> None:
    """``speaking is False`` on the transcript hooks is what arms the
    false-interruption resume timer (see ``on_final_transcript`` in
    agent_activity). ``self._vad`` is the resolved, wired VAD — the session's
    auto-loaded default included — and it drives ``_speaking``, so its state is
    reported rather than ``None`` (which silently disables the resume path)."""
    now = time.time()
    ar = _make_recognition(vad=MagicMock(), input_started_at=now - 10.0)
    ar._speaking = False

    await ar._on_stt_event(_final_transcript(end_time=0.0))

    assert ar._hooks.on_final_transcript.call_args.kwargs["speaking"] is False

    # without any vad (and without stt turn detection) there is no state to report
    ar = _make_recognition(vad=None, input_started_at=now - 10.0)
    ar._speaking = False

    await ar._on_stt_event(_final_transcript(end_time=0.0))

    assert ar._hooks.on_final_transcript.call_args.kwargs["speaking"] is None
