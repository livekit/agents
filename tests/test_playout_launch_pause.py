from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.generation import _audio_forwarding_task, _AudioOutput
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_io import FakeAudioOutput

pytestmark = pytest.mark.unit


@pytest.fixture
async def activity() -> AsyncIterator[tuple[AgentActivity, FakeAudioOutput]]:
    session = AgentSession(
        turn_handling={
            "interruption": {
                "resume_false_interruption": True,
                "false_interruption_timeout": 2.0,
            }
        }
    )
    audio_output = FakeAudioOutput(can_pause=True)
    session.output.audio = audio_output
    agent_activity = AgentActivity(Agent(instructions="test"), session)
    session._update_agent_state("thinking")

    yield agent_activity, audio_output

    await session.aclose()


async def test_playout_launch_preserves_existing_pause(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    agent_activity._update_paused_speech(speech_handle, timeout=2.0)
    audio_output.pause()
    paused_at = audio_output._paused_at

    agent_activity._reconcile_playout_pause(speech_handle)

    assert paused_at is not None
    assert audio_output._paused_at == paused_at
    assert agent_activity._paused_speech is not None
    assert agent_activity._paused_speech.handle is speech_handle
    assert agent_activity._paused_speech.timeout == 2.0


async def test_playout_launch_pauses_when_sos_precedes_current_speech(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    agent_activity._user_silence_event.clear()

    agent_activity._reconcile_playout_pause(speech_handle)

    assert audio_output._paused_at is not None
    assert agent_activity._paused_speech is not None
    assert agent_activity._paused_speech.handle is speech_handle
    assert agent_activity._paused_speech.timeout == 0


async def test_playout_launch_resumes_when_user_is_silent(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    assert agent_activity._user_silence_event.is_set()

    agent_activity._reconcile_playout_pause(speech_handle)

    assert audio_output._paused_at is None
    assert agent_activity._paused_speech is None


async def test_playout_launch_releases_pause_when_interruptions_are_disabled(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    agent_activity._update_paused_speech(speech_handle, timeout=2.0)
    audio_output.pause()
    timer = MagicMock()
    agent_activity._false_interruption_timer = timer
    agent_activity._false_interruption_pending = True

    speech_handle.allow_interruptions = False
    agent_activity._reconcile_playout_pause(speech_handle)

    assert audio_output._paused_at is None
    assert agent_activity._paused_speech is None
    timer.cancel.assert_called_once_with()
    assert agent_activity._false_interruption_timer is None
    assert agent_activity._false_interruption_pending is False

    speech_handle.allow_interruptions = True
    agent_activity._reconcile_playout_pause(speech_handle)
    await agent_activity._cancel_speech_pause()

    assert audio_output._paused_at is None
    assert speech_handle.interrupted is False


async def test_playout_launch_releases_pause_for_interrupted_speech(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    agent_activity._update_paused_speech(speech_handle, timeout=2.0)
    audio_output.pause()

    speech_handle.interrupt()
    agent_activity._reconcile_playout_pause(speech_handle)

    assert audio_output._paused_at is None
    assert agent_activity._paused_speech is None


async def test_playout_launch_releases_pause_when_pause_is_disabled(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    agent_activity._update_paused_speech(speech_handle, timeout=2.0)
    audio_output.pause()
    agent_activity._session.options.interruption["resume_false_interruption"] = False

    agent_activity._reconcile_playout_pause(speech_handle)

    assert audio_output._paused_at is None
    assert agent_activity._paused_speech is None


async def test_playout_launch_releases_pause_when_audio_output_is_disabled(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    agent_activity._update_paused_speech(speech_handle, timeout=2.0)
    audio_output.pause()
    agent_activity._session.output.set_audio_enabled(False)

    agent_activity._reconcile_playout_pause(speech_handle)

    assert audio_output._paused_at is None
    assert agent_activity._paused_speech is None


async def test_audio_forwarding_reconciles_playout_pause_before_first_frame() -> None:
    order: list[str] = []
    audio_output = MagicMock()
    audio_output.sample_rate = None
    audio_output.capture_frame = AsyncMock(side_effect=lambda _: order.append("frame"))
    audio_output.flush.side_effect = lambda: order.append("flush")

    frame = MagicMock()

    async def _frames():
        yield frame

    out = _AudioOutput(
        audio=[],
        first_frame_fut=asyncio.Future(),
        captured_segments_before=0,
    )
    await _audio_forwarding_task(
        audio_output,
        _frames(),
        out,
        reconcile_playout_pause=lambda: order.append("reconcile"),
    )

    assert order == ["reconcile", "frame", "flush"]
    audio_output.resume.assert_not_called()
