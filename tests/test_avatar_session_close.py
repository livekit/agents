"""AvatarSession.aclose() must undo the audio route its start() installed.

Avatar plugins route the agent session's audio to the avatar via
``_attach_audio_output`` (a remembered ``replace_audio_tail``). If the avatar
fails to start and the app gives up on it, ``aclose()`` has to put the previous
route back, or the agent stays silent instead of degrading to regular audio
(livekit/agents#7276).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from livekit.agents.voice import AgentSession
from livekit.agents.voice.avatar import AvatarSession
from livekit.agents.voice.io import _AudioSinkProxy

from .fake_io import FakeAudioOutput

pytestmark = pytest.mark.unit


class _FakeAvatar(AvatarSession):
    """Mimics what an avatar plugin's start() does."""

    def __init__(self) -> None:
        super().__init__()
        self.sink = FakeAudioOutput()

    @property
    def avatar_identity(self) -> str:
        return "avatar-worker"

    async def start(self, agent_session: AgentSession, room) -> None:  # type: ignore[override]
        await super().start(agent_session, room)
        self._attach_audio_output(self.sink)


def _room() -> MagicMock:
    room = MagicMock()
    room.isconnected.return_value = False
    return room


async def test_aclose_clears_a_route_installed_over_nothing() -> None:
    # canonical ordering: avatar.start() runs before session.start(), so there is
    # no audio output yet; a failed avatar must leave the slot empty again so a
    # later session.start() sets up room audio normally
    session = AgentSession()
    assert session.output.audio is None

    avatar = _FakeAvatar()
    await avatar.start(session, _room())
    assert session.output.audio is avatar.sink

    await avatar.aclose()
    assert session.output.audio is None


async def test_aclose_restores_the_previous_output() -> None:
    session = AgentSession()
    previous = FakeAudioOutput()
    session.output.audio = previous

    avatar = _FakeAvatar()
    await avatar.start(session, _room())
    assert session.output.audio is avatar.sink

    await avatar.aclose()
    assert session.output.audio is previous


async def test_aclose_restores_the_tail_under_wrappers() -> None:
    # with wrappers present, the route swap happens below the proxy; the restore
    # must swap the tail back without disturbing the wrapper chain
    session = AgentSession()
    previous = FakeAudioOutput()
    proxy = _AudioSinkProxy(previous)
    session.output.audio = proxy

    avatar = _FakeAvatar()
    await avatar.start(session, _room())
    assert session.output.audio is proxy
    assert proxy.next_in_chain is avatar.sink

    await avatar.aclose()
    assert session.output.audio is proxy
    assert proxy.next_in_chain is previous


async def test_aclose_keeps_a_newer_route() -> None:
    # someone re-routed the audio after the avatar; their route wins
    session = AgentSession()
    avatar = _FakeAvatar()
    await avatar.start(session, _room())

    newer = FakeAudioOutput()
    session.output.replace_audio_tail(newer)

    await avatar.aclose()
    assert session.output.audio is newer


async def test_aclose_without_attach_is_a_no_op() -> None:
    # a plugin that never routed audio (e.g. start() failed before the attach)
    session = AgentSession()
    previous = FakeAudioOutput()
    session.output.audio = previous

    avatar = _FakeAvatar()
    avatar._attach_audio_output = lambda sink: None  # type: ignore[method-assign]
    await avatar.start(session, _room())

    await avatar.aclose()
    assert session.output.audio is previous


async def test_a_cancelled_aclose_still_restores_the_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # aclose can be cancelled mid participant-removal (job-shutdown deadline);
    # the route restore runs before any await so the cancellation can't skip it
    import asyncio

    session = AgentSession()
    previous = FakeAudioOutput()
    session.output.audio = previous

    removal_started = asyncio.Event()

    async def _hang(*args, **kwargs) -> None:
        removal_started.set()
        await asyncio.Event().wait()

    ctx = MagicMock()
    ctx.api.room.remove_participant = _hang
    monkeypatch.setattr(
        "livekit.agents.voice.avatar._types.get_job_context", lambda required=False: ctx
    )

    avatar = _FakeAvatar()
    room = _room()
    room.name = "avatar-room"  # a bare MagicMock .name breaks the protobuf request
    await avatar.start(session, room)
    assert session.output.audio is avatar.sink

    room.isconnected.return_value = True
    close_task = asyncio.create_task(avatar.aclose())
    await asyncio.wait_for(removal_started.wait(), timeout=1.0)
    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert session.output.audio is previous
