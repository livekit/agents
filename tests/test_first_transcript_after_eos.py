"""Tests for the ``first_transcript_after_eos_delay`` latency metric.

The existing ``transcription_delay`` measures ``last_final_transcript_time -
last_speaking_time`` (last speaking anchor -> *final* transcript). The new
``first_transcript_after_eos_delay`` measures ``first_transcript_time -
end_of_speech_time`` (end-of-speech -> *first* transcript event), which is the
metric requested in https://github.com/livekit/agents/issues/4795 for
provider-latency comparison.

These tests drive ``AudioRecognition`` directly with crafted VAD/STT events
(no audio, no network) and assert on the anchors captured by the event
handlers and on the value reported through the ``on_end_of_turn`` hook.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents import stt
from livekit.agents.language import LanguageCode
from livekit.agents.vad import VADEvent, VADEventType
from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _create_audio_recognition() -> AudioRecognition:
    """Create an ``AudioRecognition`` with just the state the handlers touch."""
    with patch.object(AudioRecognition, "__init__", lambda self, *args, **kwargs: None):
        ar = AudioRecognition.__new__(AudioRecognition)

    ar._speech_start_time = None
    ar._vad_speech_started = False
    ar._user_silence_ev = asyncio.Event()
    ar._speaking = False
    ar._end_of_turn_task = None
    ar._user_turn_span = None
    ar._user_turn_start = None
    ar._user_turn_committed = False
    ar._vad_base_turn_detection = False
    ar._turn_detection_mode = None
    ar._stt = None
    ar._vad = None
    ar._stt_model = None
    ar._stt_provider = None
    ar._stt_request_ids = []
    ar._input_started_at = None
    ar._audio_transcript = ""
    ar._audio_interim_transcript = ""
    ar._audio_preflight_transcript = ""
    ar._last_speaking_time = None
    ar._last_final_transcript_time = None
    ar._last_language = None
    ar._final_transcript_received = asyncio.Event()
    ar._final_transcript_confidence = []
    ar._interruption_enabled = False
    ar._end_of_speech_time = None
    ar._first_transcript_after_eos_time = None

    ar._hooks = MagicMock()
    ar._session = MagicMock()
    ar._session.amd = None
    ar._session._room_io = None
    ar._session.options.turn_handling = {"user_turn_limit": {}}

    return ar


def _vad_eos() -> VADEvent:
    return VADEvent(
        type=VADEventType.END_OF_SPEECH,
        samples_index=0,
        timestamp=time.time(),
        speech_duration=1.0,
        silence_duration=0.6,
        inference_duration=0.0,
    )


def _final_transcript(text: str = "hello world") -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language=LanguageCode("en"), text=text, confidence=0.9)],
    )


@pytest.mark.asyncio
async def test_vad_end_of_speech_sets_end_of_speech_time():
    """A VAD END_OF_SPEECH event anchors ``_end_of_speech_time``."""
    ar = _create_audio_recognition()
    ar._speaking = True

    before = time.time()
    await ar._on_vad_event(_vad_eos())
    after = time.time()

    assert ar._end_of_speech_time is not None
    assert before <= ar._end_of_speech_time <= after


@pytest.mark.asyncio
async def test_first_transcript_after_eos_time_recorded_once():
    """The first transcript event after EOS anchors the time; later ones don't move it."""
    ar = _create_audio_recognition()
    ar._speaking = True

    await ar._on_vad_event(_vad_eos())
    assert ar._first_transcript_after_eos_time is None  # nothing yet

    await ar._on_stt_event(_final_transcript("hello"))
    first = ar._first_transcript_after_eos_time
    assert first is not None
    assert first >= ar._end_of_speech_time

    await asyncio.sleep(0.05)
    await ar._on_stt_event(_final_transcript("world"))
    assert ar._first_transcript_after_eos_time == pytest.approx(first, abs=1e-9), (
        "the anchor must capture the FIRST transcript after EOS, not be overwritten"
    )


@pytest.mark.asyncio
async def test_transcript_before_eos_does_not_set_anchor():
    """A transcript arriving while the user is still speaking is not 'after EOS'."""
    ar = _create_audio_recognition()
    ar._speaking = True

    # transcript before any end-of-speech -> no EOS anchor, no first-transcript anchor
    await ar._on_stt_event(_final_transcript("partial"))
    assert ar._end_of_speech_time is None
    assert ar._first_transcript_after_eos_time is None


@pytest.mark.asyncio
async def test_end_of_turn_info_reports_first_transcript_after_eos_delay():
    """The computed delay equals first_transcript_time - end_of_speech_time and is
    surfaced on ``_EndOfTurnInfo`` passed to ``on_end_of_turn``."""
    ar = _create_audio_recognition()

    captured: dict[str, object] = {}

    def _on_end_of_turn(info):
        captured["info"] = info
        return True

    ar._hooks.on_end_of_turn.side_effect = _on_end_of_turn
    ar._hooks.retrieve_chat_ctx.return_value = MagicMock(copy=lambda: MagicMock(items=[]))

    eos_t = 1000.0
    first_t = 1000.4
    ar._end_of_speech_time = eos_t
    ar._first_transcript_after_eos_time = first_t
    ar._last_speaking_time = eos_t
    ar._last_final_transcript_time = first_t
    ar._speech_start_time = 999.0
    ar._user_turn_start = 999.0
    ar._audio_transcript = "hello world"

    # call the metrics computation path directly via _run_eou_detection's inner task
    chat_ctx = MagicMock()
    chat_ctx.copy.return_value = chat_ctx
    chat_ctx.items = []
    ar._turn_detector = None
    ar._endpointing = MagicMock()
    ar._endpointing.min_delay = 0.0
    ar._endpointing.max_delay = 0.0
    ar._closing = asyncio.Event()

    with patch.object(ar, "_ensure_user_turn_span", return_value=MagicMock()):
        ar._run_eou_detection(chat_ctx)
        await ar._end_of_turn_task

    info = captured["info"]
    assert info.first_transcript_after_eos_delay == pytest.approx(0.4, abs=1e-6)
