"""A manually committed turn must survive a `clear_user_turn()` while the end-of-turn delay runs."""

from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import MagicMock

import pytest

from livekit.agents import llm
from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = pytest.mark.unit


def _make_recognition(
    *, mode: str = "manual", transcript: str = "hello there", min_delay: float = 0.0
) -> AudioRecognition:
    """Enough of `AudioRecognition` to drive `_run_eou_detection` without a session."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._session._root_span_context = None
    ar._session._room_io = None
    ar._session.amd = None
    ar._hooks = MagicMock()
    ar._hooks.on_end_of_turn.return_value = True
    ar._hooks.retrieve_chat_ctx.return_value = llm.ChatContext()
    ar._stt = MagicMock()
    ar._stt_pipeline = None
    ar._stt_model = None
    ar._stt_provider = None
    ar._vad = None
    ar._turn_detection_mode = mode
    ar._turn_detector = None
    ar._turn_detector_stream = None
    ar._turn_detector_prediction_fut = None
    ar._vad_base_turn_detection = False
    ar._agent_speaking = False
    ar._interruption_enabled = False
    ar._interruption_ch = None
    ar._turn_backchannel_over_agent = False
    ar._overlap_in_current_turn = False
    ar._active_vad_speech_started_at = None
    ar._transcription_timeout_handle = None
    ar._turn_speech_duration = 0.0
    ar._turn_transcript_received = False
    ar._audio_transcript = transcript
    ar._audio_interim_transcript = ""
    ar._audio_preflight_transcript = ""
    ar._final_transcript_received = asyncio.Event()
    ar._final_transcript_confidence = []
    ar._stt_request_ids = []
    ar._stt_events = []
    ar._last_speaking_time = None
    ar._last_final_transcript_time = None
    ar._speech_start_time = None
    ar._vad_speech_started = False
    ar._user_silence_ev = asyncio.Event()
    ar._speaking = False
    ar._end_of_turn_task = None
    ar._end_of_turn_committed = False
    ar._commit_user_turn_atask = None
    ar._user_turn_committed = False
    ar._last_language = None
    ar._user_turn_span = None
    ar._user_turn_start = None
    ar._eou_wait_span = None
    ar._eou_wait_started_at_ns = None
    ar._eou_wait_rearms = 0
    ar._eou_wait_floor_ns = None
    ar._eou_wait_not_committed = 0
    ar._user_turn_resumes = 0
    ar._eou_detection_span = None
    ar._closing = asyncio.Event()

    endpointing = MagicMock()
    endpointing.min_delay = min_delay
    endpointing.max_delay = max(min_delay, 0.0)
    ar._endpointing = endpointing
    # the stt pipeline is not what this file is about: only the turn-clearing decision is
    ar._update_stt = MagicMock()
    return ar


async def _await_bounce(ar: AudioRecognition) -> None:
    task = ar._end_of_turn_task
    assert task is not None
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_committed_transcript_survives_a_clear_before_the_delay_elapses() -> None:
    """`clear_user_turn()` blanks the live transcript; the committed turn must still carry the text."""
    ar = _make_recognition()

    ar._run_eou_detection(llm.ChatContext(), trigger="manual", transcript="hello there")
    # what clear_user_turn() does while the end-of-turn delay is still running
    ar._audio_transcript = ""

    await asyncio.wait_for(_await_bounce(ar), timeout=2.0)

    info = ar._hooks.on_end_of_turn.call_args[0][0]
    assert info.new_transcript == "hello there"


async def test_transcript_argument_reaches_the_turn_detector_context() -> None:
    """The captured transcript is what the end-of-turn detection sees, not the live field."""
    ar = _make_recognition()

    ar._run_eou_detection(llm.ChatContext(), trigger="manual", transcript="hello there")
    ar._audio_transcript = ""

    await asyncio.wait_for(_await_bounce(ar), timeout=2.0)

    info = ar._hooks.on_end_of_turn.call_args[0][0]
    user_items = [item for item in info.new_transcript.split() if item]
    assert user_items == ["hello", "there"]


async def test_vad_trigger_captures_the_transcript_at_schedule_time() -> None:
    """Every trigger captures the text when the turn is scheduled, not when the delay ends."""
    ar = _make_recognition(mode="vad", transcript="hello there")

    ar._run_eou_detection(llm.ChatContext(), trigger="vad")
    ar._audio_transcript = ""

    await asyncio.wait_for(_await_bounce(ar), timeout=2.0)

    info = ar._hooks.on_end_of_turn.call_args[0][0]
    assert info.new_transcript == "hello there"


async def test_clear_user_turn_after_a_manual_commit_still_delivers_the_turn() -> None:
    """The reported path: commit resolves, then clear_user_turn() lands during the eou delay."""
    ar = _make_recognition(min_delay=0.5)

    # a FINAL just landed, so the commit takes its fast path and resolves immediately
    ar._last_final_transcript_time = time.time()

    fut = ar._commit_user_turn(audio_detached=False, transcript_timeout=2.0)
    assert await asyncio.wait_for(fut, timeout=2.0) == "hello there"

    task = ar._end_of_turn_task
    assert task is not None
    ar._clear_user_turn()

    await asyncio.wait_for(asyncio.shield(task), timeout=3.0)

    info = ar._hooks.on_end_of_turn.call_args[0][0]
    assert info.new_transcript == "hello there"


async def test_clear_user_turn_cancels_a_turn_that_was_never_committed() -> None:
    """Clearing still drops the live turn: only a committed one is delivered."""
    ar = _make_recognition(mode="vad", min_delay=0.5)

    ar._run_eou_detection(llm.ChatContext(), trigger="vad")
    task = ar._end_of_turn_task
    assert task is not None

    ar._clear_user_turn()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not ar._hooks.on_end_of_turn.called


async def test_a_retained_committed_turn_does_not_erase_the_next_turn() -> None:
    """Once the live turn is replaced, the retained task must not clear the new turn's input."""
    ar = _make_recognition(min_delay=0.3)
    ar._last_final_transcript_time = time.time()

    fut = ar._commit_user_turn(audio_detached=False, transcript_timeout=2.0)
    assert await asyncio.wait_for(fut, timeout=2.0) == "hello there"
    task = ar._end_of_turn_task
    assert task is not None

    ar._clear_user_turn()
    # the next turn starts while the committed one is still waiting out the endpointing delay
    ar._audio_transcript = "second question"

    await asyncio.wait_for(asyncio.shield(task), timeout=3.0)

    assert ar._hooks.on_end_of_turn.call_args[0][0].new_transcript == "hello there"
    assert ar._audio_transcript == "second question"
