"""Regression tests for the empty final ``MultiSpeakerAdapter`` emits for a suppressed segment.

With ``suppress_background_speaker=True`` the adapter forwards a background
segment's interim (diarizing providers report interims without a speaker id, so
they bypass the suppression check) and then drops the attributed final. In its
place it emits a ``FINAL_TRANSCRIPT`` whose only alternative has empty text:
"send an empty final transcript to clear the interim results" (see
``stt/multi_speaker_adapter.py``).

The transport cannot express a wire-level clear (an empty text stream carries
zero data chunks, which clients ignore), so ``AudioRecognition`` does not
forward the event. It does use the empty final to drop the interim buffered for
the suppressed segment. Otherwise those stale background words stay in
``_current_transcript`` - where they count toward the ``min_words`` interruption
gate - and ``commit_user_turn`` appends them to the user turn.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from livekit.agents import stt
from livekit.agents.language import LanguageCode
from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = pytest.mark.unit


def _make_recognition() -> AudioRecognition:
    """Wire the attributes the transcript path of ``_on_stt_event`` touches."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._session._root_span_context = None
    ar._session.amd = None
    ar._session._room_io = None
    ar._hooks = MagicMock()
    ar._vad = None
    ar._turn_detection_mode = "manual"
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
    ar._stt_pipeline.input_started_at = time.time()
    ar._check_user_turn_limit = MagicMock()  # type: ignore[method-assign]
    ar._speech_start_time = None
    ar._run_eou_detection = MagicMock()  # type: ignore[method-assign]
    # only reached by _commit_user_turn
    ar._stt = MagicMock()
    ar._closing = asyncio.Event()
    ar._commit_user_turn_atask = None
    return ar


def _final_transcript(text: str, *, speaker_id: str | None) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language=LanguageCode("en"),
                text=text,
                confidence=0.9,
                speaker_id=speaker_id,
            )
        ],
    )


def _interim_transcript(text: str) -> stt.SpeechEvent:
    # Deepgram sets speaker=None on every non-final result, so a diarizing
    # provider's interim carries no attribution and bypasses the adapter's
    # background-speaker suppression.
    return stt.SpeechEvent(
        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[stt.SpeechData(language=LanguageCode("en"), text=text, speaker_id=None)],
    )


def _empty_final_transcript() -> stt.SpeechEvent:
    # Exactly what MultiSpeakerAdapterWrapper._forward_output emits for a
    # suppressed final: one alternative with empty text and no language.
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language=LanguageCode(""), text="")],
    )


async def _feed_adapter_sequence(ar: AudioRecognition) -> None:
    """Feed the events the adapter emits for a suppressed background segment."""
    await ar._on_stt_event(_final_transcript("hello", speaker_id="S0"))
    await ar._on_stt_event(_interim_transcript("background chatter"))
    await ar._on_stt_event(_empty_final_transcript())


async def test_empty_final_clears_the_background_interim() -> None:
    ar = _make_recognition()

    await ar._on_stt_event(_final_transcript("hello", speaker_id="S0"))
    await ar._on_stt_event(_interim_transcript("background chatter"))

    # control: the speaker-less interim is buffered as usual
    assert ar._audio_interim_transcript == "background chatter"
    assert ar._current_transcript == "hello background chatter"

    await ar._on_stt_event(_empty_final_transcript())

    assert ar._audio_interim_transcript == ""
    assert ar._audio_preflight_transcript == ""
    assert ar._current_transcript == "hello"
    assert ar._final_transcript_received.is_set()


async def test_empty_final_without_buffered_interim_keeps_empty_state() -> None:
    """An empty final with nothing buffered stays a no-op for the turn text."""
    ar = _make_recognition()

    await ar._on_stt_event(_empty_final_transcript())

    assert ar._audio_interim_transcript == ""
    assert ar._audio_preflight_transcript == ""
    assert ar._current_transcript == ""
    assert ar._final_transcript_received.is_set()


async def test_empty_final_keeps_background_text_out_of_commit_user_turn() -> None:
    ar = _make_recognition()

    await _feed_adapter_sequence(ar)
    # the empty final just arrived, so the transcript is fresh and the commit
    # does not need to flush the STT and wait for another final
    ar._last_final_transcript_time = time.time()

    committed = await ar._commit_user_turn(
        audio_detached=False, transcript_timeout=0.01, stt_flush_duration=0.0
    )

    assert committed == "hello"
