"""Integration tests for wait_for_participant, wait_for_track_publication, and wait_for_agent.

These tests require a running LiveKit server (bootstrapped via the livekit_server fixture).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import pytest

from livekit import api, rtc
from livekit.agents.utils.participant import (
    wait_for_agent,
    wait_for_participant,
    wait_for_track_publication,
)

from .lk_server import LK_API_KEY, LK_API_SECRET, LK_URL, livekit_server  # noqa: F401

TIMEOUT = 5.0


def _make_token(
    identity: str,
    room: str,
    *,
    kind: str | None = None,
    agent: bool = False,
) -> str:
    token = (
        api.AccessToken(LK_API_KEY, LK_API_SECRET)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                agent=agent,
            )
        )
    )
    if kind is not None:
        token = token.with_kind(kind)
    return token.to_jwt()


@asynccontextmanager
async def connect_room(
    identity: str, room_name: str, *, kind: str | None = None, agent: bool = False
):
    room = rtc.Room()
    token = _make_token(identity, room_name, kind=kind, agent=agent)
    await room.connect(LK_URL, token)
    try:
        yield room
    finally:
        await room.disconnect()


def _room_name() -> str:
    return f"test-{uuid.uuid4()}"


async def _publish_audio_track(room: rtc.Room) -> None:
    source = rtc.AudioSource(48000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    await room.local_participant.publish_track(track)


async def _test_wait_disconnect(wait_fn: Callable[[rtc.Room], Awaitable[object]]) -> None:
    """Connect a room, start waiting, disconnect, and assert RuntimeError."""
    name = _room_name()
    room = rtc.Room()
    token = _make_token("observer", name)
    await room.connect(LK_URL, token)

    task = asyncio.ensure_future(wait_fn(room))
    # ensure the task is running and has registered its event handlers
    await asyncio.sleep(0.5)
    await room.disconnect()

    with pytest.raises(RuntimeError, match="disconnected"):
        await asyncio.wait_for(task, timeout=TIMEOUT)


# -- wait_for_participant tests --


@pytest.mark.usefixtures("livekit_server")
class TestWaitForParticipant:
    async def test_joins_after(self):
        """Room A waits, Room B joins afterwards -> returns B."""
        name = _room_name()
        async with connect_room("observer", name) as room_a:
            task = asyncio.ensure_future(wait_for_participant(room_a))
            await asyncio.sleep(0.5)

            async with connect_room("joiner", name):
                result = await asyncio.wait_for(task, timeout=TIMEOUT)
                assert result.identity == "joiner"

    async def test_already_joined(self):
        """Room B already present when A calls wait -> returns immediately."""
        name = _room_name()
        async with connect_room("first", name):
            await asyncio.sleep(0.5)
            async with connect_room("observer", name) as room_a:
                await asyncio.sleep(0.5)
                result = await asyncio.wait_for(wait_for_participant(room_a), timeout=TIMEOUT)
                assert result.identity == "first"

    async def test_with_identity(self):
        """Only returns participant matching specific identity."""
        name = _room_name()
        async with connect_room("observer", name) as room_a:
            task = asyncio.ensure_future(wait_for_participant(room_a, identity="target"))
            await asyncio.sleep(0.3)

            # Join with wrong identity first
            async with connect_room("decoy", name):
                await asyncio.sleep(0.3)
                assert not task.done()

                # Now join with the target identity
                async with connect_room("target", name):
                    result = await asyncio.wait_for(task, timeout=TIMEOUT)
                    assert result.identity == "target"

    async def test_with_kind(self):
        """Standard participant doesn't match; agent participant does."""
        name = _room_name()
        async with connect_room("observer", name) as room_a:
            task = asyncio.ensure_future(
                wait_for_participant(
                    room_a,
                    kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT,
                )
            )
            await asyncio.sleep(0.3)

            # Standard participant should not match
            async with connect_room("standard", name):
                await asyncio.sleep(0.3)
                assert not task.done()

                # Agent participant should match
                async with connect_room("agent-p", name, kind="agent", agent=True):
                    result = await asyncio.wait_for(task, timeout=TIMEOUT)
                    assert result.identity == "agent-p"

    async def test_disconnect(self):
        """Room disconnects while waiting -> RuntimeError raised."""
        await _test_wait_disconnect(wait_for_participant)


# -- wait_for_track_publication tests --


@pytest.mark.usefixtures("livekit_server")
class TestWaitForTrackPublication:
    async def test_published_after(self):
        """Observer waits, publisher publishes audio track -> returns publication."""
        name = _room_name()
        async with connect_room("observer", name) as room_a:
            task = asyncio.ensure_future(wait_for_track_publication(room_a))
            await asyncio.sleep(0.5)

            async with connect_room("publisher", name) as room_b:
                await _publish_audio_track(room_b)
                result = await asyncio.wait_for(task, timeout=TIMEOUT)
                assert result.kind == rtc.TrackKind.KIND_AUDIO

    async def test_already_published(self):
        """Track already published before wait -> returns immediately."""
        name = _room_name()
        async with connect_room("publisher", name) as room_b:
            await _publish_audio_track(room_b)
            await asyncio.sleep(0.5)

            async with connect_room("observer", name) as room_a:
                await asyncio.sleep(0.5)
                result = await asyncio.wait_for(wait_for_track_publication(room_a), timeout=TIMEOUT)
                assert result.kind == rtc.TrackKind.KIND_AUDIO

    async def test_disconnect(self):
        """Room disconnects while waiting -> RuntimeError raised."""
        await _test_wait_disconnect(wait_for_track_publication)


# -- wait_for_agent tests --


@pytest.mark.usefixtures("livekit_server")
class TestWaitForAgent:
    async def test_agent_joins(self):
        """Agent-kind participant detected correctly."""
        name = _room_name()
        async with connect_room("observer", name) as room_a:
            task = asyncio.ensure_future(wait_for_agent(room_a))
            await asyncio.sleep(0.5)

            async with connect_room("my-agent", name, kind="agent", agent=True):
                result = await asyncio.wait_for(task, timeout=TIMEOUT)
                assert result.identity == "my-agent"
                assert result.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
