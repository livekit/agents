from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit import rtc
from livekit.agents import AgentStateChangedEvent, AudioConfig, BuiltinAudioClip
from livekit.agents.voice.background_audio import BackgroundAudioPlayer

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_thinking_state_change_during_aclose_does_not_play(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session: rtc.EventEmitter[str] = rtc.EventEmitter()
    room = MagicMock()
    room.local_participant.publish_track = AsyncMock()
    room.local_participant.track_publications = {}

    player = BackgroundAudioPlayer(
        thinking_sound=AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.5),
    )
    await player.start(room=room, agent_session=session)  # type: ignore[arg-type]

    # the agent starts thinking while the player is tearing down its mixer
    mixer_aclose = player._audio_mixer.aclose

    async def _aclose_while_thinking() -> None:
        session.emit(
            "agent_state_changed",
            AgentStateChangedEvent(old_state="listening", new_state="thinking"),
        )
        await mixer_aclose()

    player._audio_mixer.aclose = _aclose_while_thinking  # type: ignore[method-assign]

    with caplog.at_level(logging.ERROR):
        await player.aclose()

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
