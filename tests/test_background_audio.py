from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents.voice.background_audio import BackgroundAudioPlayer

pytestmark = pytest.mark.unit


async def _frames() -> AsyncGenerator[rtc.AudioFrame, None]:
    # long enough that the play task cannot finish on its own during the test
    for _ in range(1000):
        yield rtc.AudioFrame.create(48000, 1, 4800)
        await asyncio.sleep(0.01)


def _started_player() -> BackgroundAudioPlayer:
    """A player in the state `start()` leaves behind, without needing a live room.

    `start()` publishes a track to a real room; everything `play()` and `aclose()`
    touch is already built in `__init__`, so only the room, session and mixer task
    have to be stood in for.
    """
    player = BackgroundAudioPlayer()
    player._room = MagicMock()
    player._agent_session = None
    player._mixer_atask = asyncio.create_task(asyncio.sleep(3600))
    return player


async def test_aclose_does_not_leave_a_play_task_running() -> None:
    # aclose() passes *self._play_tasks to cancel_and_wait, which snapshots the list, and
    # only clears _mixer_atask (the flag play() gates on) after awaiting. A play() landing
    # in that window is never cancelled, so it outlives the player whose mixer and audio
    # source have already been closed.
    player = _started_player()

    closing = asyncio.create_task(player.aclose())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    with contextlib.suppress(RuntimeError):
        player.play(_frames())

    await closing

    still_running = [task for task in player._play_tasks if not task.done()]
    assert still_running == []


async def test_play_handle_from_a_closing_player_never_hangs() -> None:
    # whatever play() hands back has to reach playout-done, otherwise `await handle`
    # blocks forever: the mixer that would drive its generator is already closed
    player = _started_player()

    closing = asyncio.create_task(player.aclose())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    handle = None
    with contextlib.suppress(RuntimeError):
        handle = player.play(_frames())

    await closing

    if handle is not None:
        await asyncio.wait_for(handle.wait_for_playout(), timeout=1.0)
