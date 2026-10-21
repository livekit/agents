from __future__ import annotations

import asyncio
import contextlib
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import AgentStateChangedEvent, AudioConfig, BuiltinAudioClip
from livekit.agents.voice.background_audio import BackgroundAudioPlayer

pytestmark = pytest.mark.unit


async def _started_player() -> tuple[BackgroundAudioPlayer, rtc.EventEmitter[str]]:
    session: rtc.EventEmitter[str] = rtc.EventEmitter()
    room = MagicMock()
    room.local_participant.publish_track = AsyncMock()
    room.local_participant.track_publications = {}

    player = BackgroundAudioPlayer(
        thinking_sound=AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.5),
    )
    await player.start(room=room, agent_session=session)  # type: ignore[arg-type]
    return player, session


def _emit_thinking(session: rtc.EventEmitter[str]) -> None:
    session.emit(
        "agent_state_changed",
        AgentStateChangedEvent(old_state="listening", new_state="thinking"),
    )


@pytest.mark.asyncio
async def test_thinking_state_change_during_aclose_does_not_play(
    caplog: pytest.LogCaptureFixture,
) -> None:
    player, session = await _started_player()

    # the agent starts thinking while the player is tearing down its mixer
    mixer_aclose = player._audio_mixer.aclose

    async def _aclose_while_thinking() -> None:
        _emit_thinking(session)
        await mixer_aclose()

    player._audio_mixer.aclose = _aclose_while_thinking  # type: ignore[method-assign]

    with caplog.at_level(logging.ERROR):
        await player.aclose()

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.asyncio
async def test_cancelled_aclose_keeps_thinking_sounds() -> None:
    player, session = await _started_player()

    # a play task that needs another loop turn to finish cancelling
    release = asyncio.Event()

    async def _slow_to_cancel() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            raise

    slow = asyncio.create_task(_slow_to_cancel())
    player._play_tasks.append(slow)
    await asyncio.sleep(0)

    close = asyncio.create_task(player.aclose())
    await asyncio.sleep(0.01)
    close.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await close

    with patch.object(player, "play") as play:
        _emit_thinking(session)
    play.assert_called_once()

    release.set()
    player._play_tasks.remove(slow)
    await player.aclose()
