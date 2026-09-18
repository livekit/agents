from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from livekit import rtc
from livekit.agents.voice.background_audio import _TRACK_NAME, BackgroundAudioPlayer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("ending", ["ack", "already_disconnected", "disconnect", "cancel", "error"])
async def test_background_audio_close_releases_unpublish_on_room_disconnect(ending: str) -> None:
    room = rtc.EventEmitter()
    connected = ending != "already_disconnected"
    requested = asyncio.Event()
    acknowledged = asyncio.Event()
    cancelled = asyncio.Event()

    async def unpublish(sid: str) -> None:
        assert sid == "current_track"
        requested.set()
        try:
            await acknowledged.wait()
            if ending == "error":
                raise RuntimeError("unpublish failed")
        except asyncio.CancelledError:
            cancelled.set()
            raise

    room.isconnected = lambda: connected
    room.local_participant = SimpleNamespace(
        track_publications={
            "current_track": SimpleNamespace(name=_TRACK_NAME, sid="current_track")
        },
        unpublish_track=unpublish,
    )
    player = object.__new__(BackgroundAudioPlayer)
    player._room = room
    player._lock = asyncio.Lock()
    player._play_tasks = set()
    player._mixer_atask = asyncio.create_task(asyncio.Event().wait())
    player._audio_mixer = SimpleNamespace(aclose=AsyncMock())
    player._audio_source = SimpleNamespace(aclose=AsyncMock())
    player._agent_session = rtc.EventEmitter()
    player._agent_session.on("agent_state_changed", player._agent_state_changed)
    close = asyncio.create_task(player.aclose())
    try:
        if ending != "already_disconnected":
            await asyncio.wait_for(requested.wait(), 1)
            assert not close.done()
            if ending in {"ack", "error"}:
                acknowledged.set()
            elif ending == "disconnect":
                connected = False
                room.emit("disconnected", rtc.DisconnectReason.CLIENT_INITIATED)
            else:
                close.cancel()
        done, _ = await asyncio.wait([close], timeout=1)
        assert close in done, "background audio waits for an ack after room event delivery stopped"
        if ending == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await close
        else:
            await close
        assert requested.is_set() == (ending != "already_disconnected")
        assert cancelled.is_set() == (ending in {"disconnect", "cancel"})
        player._audio_mixer.aclose.assert_awaited_once()
        player._audio_source.aclose.assert_awaited_once()
        assert not player._agent_session._events.get("agent_state_changed")
        assert not room._events.get("disconnected")
    finally:
        close.cancel()
        await asyncio.gather(close, return_exceptions=True)
