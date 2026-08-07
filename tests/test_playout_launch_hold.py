from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.generation import _audio_forwarding_task, _AudioOutput
from livekit.agents.voice.speech_handle import SpeechHandle

pytestmark = pytest.mark.unit


def _make_activity() -> tuple[AgentActivity, MagicMock]:
    audio_output = MagicMock()
    audio_output.can_pause = True

    activity = object.__new__(AgentActivity)
    activity._session = SimpleNamespace(  # type: ignore[assignment]
        agent_state="thinking",
        user_state="listening",
        options=SimpleNamespace(
            interruption={
                "resume_false_interruption": True,
                "false_interruption_timeout": 2.0,
            }
        ),
        output=SimpleNamespace(audio=audio_output),
    )
    activity._user_silence_event = asyncio.Event()
    activity._user_silence_event.set()
    activity._paused_speech = None
    return activity, audio_output


async def test_playout_launch_preserves_existing_hold() -> None:
    activity, audio_output = _make_activity()
    speech_handle = SpeechHandle.create()
    activity._update_paused_speech(speech_handle, timeout=2.0)

    activity._prepare_audio_playout(speech_handle)

    audio_output.pause.assert_called_once_with()
    audio_output.resume.assert_not_called()
    assert activity._paused_speech is not None
    assert activity._paused_speech.handle is speech_handle
    assert activity._paused_speech.timeout == 2.0


async def test_playout_launch_recovers_hold_when_sos_precedes_current_speech() -> None:
    activity, audio_output = _make_activity()
    speech_handle = SpeechHandle.create()
    activity._user_silence_event.clear()

    activity._prepare_audio_playout(speech_handle)

    audio_output.pause.assert_called_once_with()
    audio_output.resume.assert_not_called()
    assert activity._paused_speech is not None
    assert activity._paused_speech.handle is speech_handle
    assert activity._paused_speech.timeout == 0


async def test_playout_launch_resumes_after_silence_gate_reopens() -> None:
    activity, audio_output = _make_activity()
    speech_handle = SpeechHandle.create()
    activity._session.user_state = "speaking"

    activity._prepare_audio_playout(speech_handle)

    audio_output.resume.assert_called_once_with()
    audio_output.pause.assert_not_called()
    assert activity._paused_speech is None


async def test_audio_forwarding_prepares_playout_before_first_frame() -> None:
    order: list[str] = []
    audio_output = MagicMock()
    audio_output.sample_rate = None
    audio_output.capture_frame = AsyncMock(side_effect=lambda _: order.append("frame"))
    audio_output.flush.side_effect = lambda: order.append("flush")

    frame = MagicMock()

    async def _frames():
        yield frame

    out = _AudioOutput(audio=[], first_frame_fut=asyncio.Future())
    await _audio_forwarding_task(
        audio_output,
        _frames(),
        out,
        prepare_playout=lambda: order.append("prepare"),
    )

    assert order == ["prepare", "frame", "flush"]
    audio_output.resume.assert_not_called()
