"""A reply must not start playout into the user's speech.

The pre-playout hold (``on_start_of_speech``) is edge-triggered, and
``_audio_forwarding_task`` used to ``resume()`` the output unconditionally the
moment the reply's TTS started streaming — so a user who resumed speaking
inside the reply's generation window (after the turn commit, before the first
TTS frame) had the reply launched into their speech, and any hold already
placed was released mid-utterance.

These tests cover the two halves of the fix:
- ``_hold_playout_if_user_speaking`` — the level check + pause bookkeeping;
- ``_audio_forwarding_task`` honoring ``hold_playout`` instead of blindly
  resuming.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.generation import _audio_forwarding_task, _AudioOutput

pytestmark = pytest.mark.unit


class _RecordingOutput:
    """Minimal AudioOutput stand-in for driving ``_audio_forwarding_task``."""

    def __init__(self) -> None:
        self.paused = 0
        self.resumed = 0
        self.flushed = 0
        self.sample_rate = None
        self.can_pause = True

    def pause(self) -> None:
        self.paused += 1

    def resume(self) -> None:
        self.resumed += 1

    def flush(self) -> None:
        self.flushed += 1

    def clear_buffer(self) -> None:
        pass


def _make_activity(
    *,
    agent_state: str = "thinking",
    user_state: str = "speaking",
    resume_false_interruption: bool = True,
) -> tuple[AgentActivity, _RecordingOutput]:
    activity: AgentActivity = AgentActivity.__new__(AgentActivity)
    output = _RecordingOutput()
    session = MagicMock()
    session.agent_state = agent_state
    session.user_state = user_state
    session.output.audio = output
    session.options.interruption = {
        "resume_false_interruption": resume_false_interruption,
        "false_interruption_timeout": 2.0,
    }
    activity._session = session
    activity._paused_speech = None
    return activity, output


def _speech(*, interrupted: bool = False, allow_interruptions: bool = True) -> Any:
    return MagicMock(interrupted=interrupted, allow_interruptions=allow_interruptions)


def test_holds_when_user_speaking_at_playout_start() -> None:
    activity, output = _make_activity()
    speech = _speech()

    assert activity._hold_playout_if_user_speaking(speech)
    assert output.paused == 1
    assert activity._paused_speech is not None
    assert activity._paused_speech.handle is speech
    assert activity._paused_speech.timeout == 0


def test_hold_preserves_upgraded_timeout_for_same_handle() -> None:
    # _interrupt_by_audio_activity may have raised this handle's timeout to
    # false_interruption_timeout; re-holding at playout start must not downgrade
    # it back to 0, or the resume fires the instant the user pauses
    from livekit.agents.voice.agent_activity import _PausedSpeechInfo

    activity, output = _make_activity()
    speech = _speech()
    activity._paused_speech = _PausedSpeechInfo(handle=speech, agent_state="thinking", timeout=2.0)

    assert activity._hold_playout_if_user_speaking(speech)
    assert output.paused == 1
    assert activity._paused_speech.timeout == 2.0
    assert activity._paused_speech.handle is speech


def test_no_hold_when_user_silent() -> None:
    activity, output = _make_activity(user_state="listening")
    assert not activity._hold_playout_if_user_speaking(_speech())
    assert output.paused == 0
    assert activity._paused_speech is None


def test_no_hold_when_agent_already_speaking() -> None:
    # overlap with audible agent speech belongs to the interruption paths
    activity, output = _make_activity(agent_state="speaking")
    assert not activity._hold_playout_if_user_speaking(_speech())
    assert output.paused == 0


def test_no_hold_when_pause_disabled() -> None:
    activity, output = _make_activity(resume_false_interruption=False)
    assert not activity._hold_playout_if_user_speaking(_speech())
    assert output.paused == 0


def test_no_hold_for_uninterruptible_speech() -> None:
    activity, output = _make_activity()
    assert not activity._hold_playout_if_user_speaking(_speech(allow_interruptions=False))
    assert not activity._hold_playout_if_user_speaking(_speech(interrupted=True))
    assert output.paused == 0


async def _no_frames() -> Any:
    if False:  # pragma: no cover - async generator
        yield


@pytest.mark.asyncio
async def test_forwarding_skips_resume_while_held() -> None:
    output = _RecordingOutput()
    out = _AudioOutput(audio=[], first_frame_fut=MagicMock())

    await _audio_forwarding_task(output, _no_frames(), out, hold_playout=lambda: True)

    assert output.resumed == 0
    assert output.flushed == 1


@pytest.mark.asyncio
async def test_forwarding_resumes_when_not_held() -> None:
    for hold_playout in (None, lambda: False):
        output = _RecordingOutput()
        out = _AudioOutput(audio=[], first_frame_fut=MagicMock())

        await _audio_forwarding_task(output, _no_frames(), out, hold_playout=hold_playout)

        assert output.resumed == 1
        assert output.flushed == 1
