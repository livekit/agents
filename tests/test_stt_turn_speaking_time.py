"""``stopped_speaking_at`` must come from the provider's word timestamps.

Issue #7651. With STT turn detection, ``_last_speaking_time`` -- reported as
``stopped_speaking_at``, and the anchor behind ``transcription_delay``,
``end_of_turn_delay`` and the agent's ``e2e_latency`` -- was overwritten on
``END_OF_SPEECH``.

Soniox, Deepgram and AssemblyAI send ``END_OF_SPEECH`` with no alternatives and
no ``speech_end_time``, so the fallback collapsed to ``time.time()``: the moment
the message arrived, which is late by the provider's endpointing delay (~0.6s
with Soniox) and by the whole pause when the mic goes quiet right after the last
word. ``transcription_delay`` was then always 0.

These drive ``_process_stt_event`` directly, in the style of
``test_audio_recognition_turn_detection.py``, because the behaviour is in the
core turn handling rather than in any one plugin.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from livekit.agents import LanguageCode
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = [pytest.mark.unit, pytest.mark.audio_eot]

# The pipeline stamps `input_started_at` on the first frame, so a word `end_time`
# of 2.0 means "two seconds of audio in", i.e. INPUT_STARTED_AT + 2.0 wall clock.
INPUT_STARTED_AT = 1_700_000_000.0
LAST_WORD_END = 2.0
ENDPOINTING_DELAY = 0.6


def _make_stt_recognition() -> AudioRecognition:
    """Enough of AudioRecognition to drive `_process_stt_event` in stt mode."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._session.amd = None
    ar._hooks = MagicMock()
    ar._stt = MagicMock()
    ar._stt_pipeline = MagicMock()
    ar._stt_pipeline.input_started_at = INPUT_STARTED_AT
    ar._vad = None
    ar._vad_stream = None
    ar._vad_speech_started = False
    ar._turn_detection_mode = "stt"
    ar._vad_base_turn_detection = False
    ar._last_language = LanguageCode("en")
    ar._audio_transcript = ""
    ar._audio_interim_transcript = ""
    ar._audio_preflight_transcript = ""
    ar._final_transcript_confidence = []
    ar._final_transcript_received = asyncio.Event()
    ar._last_final_transcript_time = None
    ar._last_speaking_time = None
    ar._last_stt_word_end_time = None
    ar._speech_start_time = None
    ar._user_silence_ev = asyncio.Event()
    ar._user_silence_ev.set()
    ar._speaking = False
    ar._user_turn_committed = False
    ar._end_of_turn_task = None
    # _run_eou_detection and the span helpers are not what these tests are about
    ar._run_eou_detection = MagicMock()
    ar._ensure_user_turn_span = MagicMock()
    ar._end_eou_wait_span = MagicMock()
    ar._check_user_turn_limit = MagicMock()
    ar._update_vad = MagicMock()
    return ar


def _final(text: str, end_time: float) -> SpeechEvent:
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            SpeechData(
                text=text,
                language=LanguageCode("en"),
                confidence=0.9,
                start_time=max(end_time - 1.0, 0.0),
                end_time=end_time,
            )
        ],
    )


def _end_of_speech(speech_end_time: float | None = None) -> SpeechEvent:
    """END_OF_SPEECH as the streaming providers send it: no alternatives."""
    ev = SpeechEvent(type=SpeechEventType.END_OF_SPEECH, alternatives=[])
    if speech_end_time is not None:
        ev.speech_end_time = speech_end_time
    return ev


def _start_of_speech() -> SpeechEvent:
    return SpeechEvent(type=SpeechEventType.START_OF_SPEECH, alternatives=[])


def test_end_of_speech_keeps_the_word_timestamp() -> None:
    """The anchor must survive an END_OF_SPEECH that carries no timing.

    This is the case in the issue: the final transcript puts the anchor on the
    last word, then END_OF_SPEECH arrives `ENDPOINTING_DELAY` later with nothing
    on it and used to move the anchor to that arrival time.
    """
    ar = _make_stt_recognition()
    expected = INPUT_STARTED_AT + LAST_WORD_END

    ar._process_stt_event(_start_of_speech())
    ar._process_stt_event(_final("hello there", LAST_WORD_END))
    assert ar._last_speaking_time == pytest.approx(expected)

    before_eos = time.time()
    ar._process_stt_event(_end_of_speech())

    assert ar._last_speaking_time == pytest.approx(expected)
    # the regression this pins: the anchor becoming ~now instead of the word end
    assert ar._last_speaking_time < before_eos


def test_explicit_speech_end_time_still_wins() -> None:
    """A provider that does timestamp END_OF_SPEECH keeps deciding the boundary."""
    ar = _make_stt_recognition()
    provider_end = INPUT_STARTED_AT + LAST_WORD_END + 0.05

    ar._process_stt_event(_start_of_speech())
    ar._process_stt_event(_final("hello there", LAST_WORD_END))
    ar._process_stt_event(_end_of_speech(speech_end_time=provider_end))

    assert ar._last_speaking_time == pytest.approx(provider_end)


def test_later_word_in_the_same_turn_moves_the_anchor_forward() -> None:
    """Two finals in one turn: the newest word end is the one kept."""
    ar = _make_stt_recognition()

    ar._process_stt_event(_start_of_speech())
    ar._process_stt_event(_final("hello", LAST_WORD_END))
    ar._process_stt_event(_final("there", LAST_WORD_END + 1.5))
    ar._process_stt_event(_end_of_speech())

    assert ar._last_speaking_time == pytest.approx(INPUT_STARTED_AT + LAST_WORD_END + 1.5)


def test_turn_without_word_timestamps_falls_back_to_arrival() -> None:
    """With nothing timestamped anywhere, `now` is still the only answer.

    A plugin that reports no word times must keep the previous behaviour rather
    than inherit an anchor from somewhere else.
    """
    ar = _make_stt_recognition()
    untimed = SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(text="hello", language=LanguageCode("en"), confidence=0.9)],
    )

    before = time.time()
    ar._process_stt_event(_start_of_speech())
    ar._process_stt_event(untimed)
    ar._process_stt_event(_end_of_speech())

    assert ar._last_speaking_time is not None
    assert before <= ar._last_speaking_time <= time.time()


def test_new_turn_does_not_inherit_the_previous_word_end() -> None:
    """A second, untimestamped turn must not reuse the first turn's word end.

    Otherwise the anchor would be seconds in the past at commit time, which
    `_compute_end_of_turn_metrics` drops as predating the turn start.
    """
    ar = _make_stt_recognition()

    ar._process_stt_event(_start_of_speech())
    ar._process_stt_event(_final("first turn", LAST_WORD_END))
    ar._process_stt_event(_end_of_speech())
    first_anchor = ar._last_speaking_time

    untimed = SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(text="second", language=LanguageCode("en"), confidence=0.9)],
    )
    before_second = time.time()
    ar._process_stt_event(_start_of_speech())
    ar._process_stt_event(untimed)
    ar._process_stt_event(_end_of_speech())

    assert first_anchor is not None
    assert ar._last_speaking_time is not None
    assert ar._last_speaking_time >= before_second
