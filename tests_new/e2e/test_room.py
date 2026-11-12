"""Room waits against a real SFU: an isolated ``livekit-server`` (``sfu.py``)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

import pytest

from livekit import rtc
from livekit.agents.utils import wait_for_participant, wait_for_track_publication

from .sfu import connected


def new_room() -> str:
    return "room-" + uuid.uuid4().hex


@pytest.mark.parametrize(
    "wait", [wait_for_participant, wait_for_track_publication], ids=lambda wait: wait.__name__
)
async def test_a_room_wait_raises_when_the_room_disconnects(
    sfu: str, wait: Callable[[rtc.Room], Awaitable[object]]
) -> None:
    """A wait on a room that disconnects raises instead of hanging.
    https://github.com/livekit/agents/issues/4861"""
    async with connected(sfu, new_room(), "observer") as room:
        waiting = asyncio.ensure_future(wait(room))
        await asyncio.sleep(0)  # the wait starts

        await room.disconnect()

        with pytest.raises(RuntimeError, match="disconnected"):
            await waiting


async def test_a_track_publication_wait_returns_a_later_audio_publication(sfu: str) -> None:
    """A wait started before anyone publishes returns the audio track published later.
    https://github.com/livekit/agents/pull/4864"""
    name = new_room()
    async with connected(sfu, name, "observer") as room:
        waiting = asyncio.ensure_future(wait_for_track_publication(room))

        async with connected(sfu, name, "publisher") as publisher:
            track = rtc.LocalAudioTrack.create_audio_track("mic", rtc.AudioSource(48000, 1))
            await publisher.local_participant.publish_track(track)
            publication = await waiting

    assert publication.kind == rtc.TrackKind.KIND_AUDIO
