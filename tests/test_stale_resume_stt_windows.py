"""A pause backed by live speech evidence must not resume on a bare END_OF_SPEECH.

With STT-based turn detection (Cartesia Ink-2 / Deepgram Flux), START_OF_SPEECH is
speech-selective and a turn's FINAL_TRANSCRIPT can lag its speech windows: a caller
telling a long story produces SOS/EOS pairs (breaths) with word-bearing interims,
and the final that commits the turn arrives seconds later.

Today a bare EOS arms the false-interruption resume timer with the pause's own
timeout — which is 0 for a pre-playout pause taken by ``on_start_of_speech`` —
and ``_on_timeout`` only defers on an *open* ``_end_of_turn_task``. Between a
speech window's EOS and its (still pending) final there is no open decision, so
the stale pause resumes straight into the caller's breath, emits a word or two,
and is then killed by the commit. Observed in production twice in one call:
held replies leaking "I'm" … "I" into each pause of a caller's narration.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from livekit.agents import Agent
from livekit.agents.voice.agent_activity import AgentActivity, _PausedSpeechInfo

from .fake_io import FakeAudioOutput
from .test_false_interruption_resume import _session

pytestmark = pytest.mark.unit

FALSE_INTERRUPTION_TIMEOUT = 0.3


def _activity_with_pending_reply(session, *, interim_words: str) -> tuple[AgentActivity, MagicMock]:
    """An activity whose reply is current but unplayed, with the caller's words
    (interims) already in flight — the state right before a pre-playout pause."""
    activity = AgentActivity(Agent(instructions="test"), session)
    activity._scheduling_paused = False
    session.output.audio = FakeAudioOutput(can_pause=True)

    handle = MagicMock()
    handle.done.return_value = False
    handle.interrupted = False
    handle.allow_interruptions = True
    handle._agent_turn_context = None
    activity._current_speech = handle

    recognition = MagicMock()
    recognition._end_of_turn_task = None  # the final hasn't arrived: no bounce yet
    recognition._audio_interim_transcript = interim_words
    activity._audio_recognition = recognition
    return activity, handle


async def test_first_breath_must_not_resume_a_preplayout_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SOS pauses the pending reply (timeout=0); the caller's first breath (bare
    EOS, final still pending) must NOT resume it — the speech evidence (SOS +
    word-bearing interims) says a turn decision is coming."""
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = _session()
    activity, handle = _activity_with_pending_reply(session, interim_words="thirteen one oh eight")

    events: list[str] = []
    session.on("agent_false_interruption", lambda _: events.append("resume"))

    # the real pre-playout pause: caller starts speaking while the reply is unplayed
    activity.on_start_of_speech(None, speech_start_time=time.time())
    assert activity._paused_speech is not None
    assert activity._paused_speech.timeout == 0

    # mid-utterance breath: ink-2 closes the speech window; the final that will
    # commit this turn is still in flight
    activity.on_end_of_speech(None)

    await asyncio.sleep(0.25)
    await session.aclose()

    # the pause must survive the breath: it is owned by the upcoming turn
    # decision (which will commit and interrupt it), not by a 0-second timer
    assert events == []
    assert activity._paused_speech is not None


async def test_speech_window_gap_must_not_resume_a_held_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same shape with a mid-playout pause (configured timeout): the gap between
    two speech windows of ONE utterance must not resume the held reply. Distinct
    from test_resume_is_immediate_when_no_turn_decision_is_open: there the EOS is
    a lone noise glitch; here a real speech window (SOS, words) preceded it and
    its final is still pending."""
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = _session()
    activity, handle = _activity_with_pending_reply(
        session, interim_words="and this is like the sixth time"
    )
    activity._paused_speech = _PausedSpeechInfo(
        handle=handle, agent_state="speaking", timeout=FALSE_INTERRUPTION_TIMEOUT
    )

    events: list[str] = []
    session.on("agent_false_interruption", lambda _: events.append("resume"))

    activity.on_start_of_speech(None, speech_start_time=time.time())  # window 1 (keeps the pause)
    activity.on_end_of_speech(None)  # breath between windows; final still pending

    await asyncio.sleep(FALSE_INTERRUPTION_TIMEOUT + 0.15)
    await session.aclose()

    assert events == []
    assert activity._paused_speech is not None
