"""An avatar's lifecycle: when a wait for its join returns at once, concurrent closes, and a start
after a close."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import VoiceSession
from tests.fakes import AgentRoom, FakeAvatar


async def test_waiting_for_the_join_of_an_avatar_never_started_returns_at_once() -> None:
    """An avatar that was never started has nothing to wait for."""
    avatar = FakeAvatar()

    await avatar.wait_for_join(timeout=0)


async def test_waiting_for_the_join_of_a_closed_avatar_returns_at_once() -> None:
    """Once the avatar is closed, a wait for its join returns at once."""
    avatar = FakeAvatar()
    await avatar.start(VoiceSession(vad=None), AgentRoom())

    await avatar.aclose()

    await avatar.wait_for_join(timeout=0)


async def test_concurrent_closes_both_return() -> None:
    """Two closes issued together both return, leaving the avatar closed."""
    avatar = FakeAvatar()
    await avatar.start(VoiceSession(vad=None), AgentRoom())

    await asyncio.gather(avatar.aclose(), avatar.aclose())

    await avatar.wait_for_join(timeout=0)


async def test_an_avatar_started_again_after_its_close_waits_for_a_new_join() -> None:
    """Starting a closed avatar again begins a new lifecycle that waits for the avatar to join."""
    avatar = FakeAvatar()
    session = VoiceSession(vad=None)
    await avatar.start(session, AgentRoom())
    await avatar.aclose()

    await avatar.start(session, AgentRoom())

    try:
        with pytest.raises(TimeoutError):
            await avatar.wait_for_join(timeout=0.01)
    finally:
        await avatar.aclose()


async def test_a_standalone_avatar_is_closed_when_its_session_closes() -> None:
    """Outside a job, closing the agent session closes its avatar. Catches an avatar that only a
    job's shutdown closed, left running (and billing) by a standalone session."""
    avatar = FakeAvatar()
    session = VoiceSession(vad=None)
    await avatar.start(session, AgentRoom())

    await session.aclose()

    assert avatar.closes == 1
    await avatar.wait_for_join(timeout=0)  # closed: returns at once
