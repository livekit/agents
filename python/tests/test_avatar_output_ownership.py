"""Avatar close releases its own sink after another avatar replaces the audio tail."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from livekit import rtc
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.plugins.anam import AvatarSession
from livekit.plugins.anam.api import SessionResponse
from livekit.plugins.anam.types import PersonaConfig

pytestmark = pytest.mark.unit


class Output:
    def __init__(self):
        self.audio = None

    def replace_audio_tail(self, audio):
        self.audio = audio


class Sink:
    def __init__(self, *, blocked=False):
        self.closing = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.closed = False
        if not blocked:
            self.release.set()

    async def aclose(self):
        self.calls += 1
        self.closing.set()
        await self.release.wait()
        self.closed = True


@pytest.fixture
def setup(monkeypatch):
    class API:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def start_session(self, **kwargs):
            return SessionResponse(session_id="provider-session")

    async def start(self, agent_session, room):
        # these tests own the audio output, not the room's avatar lifecycle
        self._room, self._agent_session = room, agent_session

    monkeypatch.setattr(BaseAvatarSession, "start", start)
    monkeypatch.setattr("livekit.plugins.anam.avatar.AnamAPI", API)
    monkeypatch.setattr(
        "livekit.plugins.anam.avatar.get_job_context",
        lambda: SimpleNamespace(local_participant_identity="agent"),
    )
    agent = SimpleNamespace(output=Output(), on=Mock(), off=Mock())
    return agent


def avatar():
    return AvatarSession(
        persona_config=PersonaConfig(name="avatar", avatarId="avatar"), api_key="key"
    )


async def start(avatar, agent, room=None):
    await avatar.start(
        agent,
        room or SimpleNamespace(name="room", isconnected=lambda: False),
        livekit_url="wss://example.test",
        livekit_api_key="key",
        livekit_api_secret="a" * 32,
    )


async def test_avatar_close_keeps_a_replacement_sink_alive(setup, monkeypatch):
    first_sink, second_sink = Sink(), Sink()
    sinks = iter([first_sink, second_sink])
    monkeypatch.setattr(
        "livekit.plugins.anam.avatar.DataStreamAudioOutput", lambda **kwargs: next(sinks)
    )
    first, second = avatar(), avatar()
    try:
        await start(first, setup)
        await start(second, setup)
        assert setup.output.audio is second_sink
        await first.aclose()
        assert first_sink.closed is True
        assert second_sink.closed is False
        await first.aclose()
        assert first_sink.calls == 1
        assert setup.output.audio is second_sink
    finally:
        await first.aclose()
        await second.aclose()
    assert second_sink.closed is True
    assert second_sink.calls == 1


async def test_concurrent_avatar_close_waits_for_the_owned_sink(setup, monkeypatch):
    sink = Sink(blocked=True)
    monkeypatch.setattr("livekit.plugins.anam.avatar.DataStreamAudioOutput", lambda **kwargs: sink)
    session = avatar()
    closing = []
    try:
        await start(session, setup)
        closing.append(asyncio.create_task(session.aclose()))
        await asyncio.wait_for(sink.closing.wait(), 1)
        closing.append(asyncio.create_task(session.aclose()))
        await asyncio.sleep(0)
        assert all(not task.done() for task in closing)
        sink.release.set()
        await asyncio.gather(*closing)
        assert sink.closed is True
        calls = sink.calls
        await session.aclose()
        assert sink.calls == calls
    finally:
        sink.release.set()
        await asyncio.gather(*closing)
        await session.aclose()


async def test_avatar_close_releases_native_startup_with_an_actual_sdk_room(setup):
    from livekit.agents.voice.avatar import DataStreamAudioOutput

    class NativeAvatar(BaseAvatarSession):
        @property
        def avatar_identity(self):
            return "avatar"

        async def start(self, agent_session, room):
            await super().start(agent_session, room)
            self._attach_audio_output(
                agent_session,
                DataStreamAudioOutput(room, destination_identity=self.avatar_identity),
            )

    session = NativeAvatar()
    try:
        await session.start(setup, rtc.Room())
        await asyncio.wait_for(session.aclose(), 1)
        await asyncio.wait_for(session.aclose(), 1)
    finally:
        await session.aclose()


async def test_cancelled_runway_close_still_releases_its_native_sink(setup, monkeypatch):
    from livekit.plugins.runway import AvatarSession as RunwaySession

    entered, release = asyncio.Event(), asyncio.Event()
    sink = Sink()

    async def end(self, room):
        entered.set()
        await release.wait()

    monkeypatch.setattr(RunwaySession, "_create_session", AsyncMock())
    monkeypatch.setattr(RunwaySession, "_end_runway_realtime_session", end)
    monkeypatch.setattr(
        "livekit.plugins.runway.avatar.DataStreamAudioOutput", lambda **kwargs: sink
    )
    monkeypatch.setattr(
        "livekit.plugins.runway.avatar.get_job_context",
        lambda: SimpleNamespace(local_participant_identity="agent"),
    )
    session = RunwaySession(avatar_id="avatar", api_key="key")
    try:
        await start(session, setup)
        closing = asyncio.create_task(session.aclose())
        await asyncio.wait_for(entered.wait(), 1)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert sink.closed is True
    finally:
        release.set()
        await session.aclose()


async def test_liveavatar_close_settles_its_owned_connection_task(setup, monkeypatch):
    from livekit.plugins.liveavatar import AvatarSession as LiveAvatarSession
    from livekit.plugins.liveavatar.api import (
        SessionResponse,
        SessionStart,
        SessionStop,
        SessionToken,
    )

    entered, stopped = asyncio.Event(), asyncio.Event()

    async def main(self):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(LiveAvatarSession, "_main_task", main)
    monkeypatch.setattr(
        "livekit.plugins.liveavatar.avatar.get_job_context",
        lambda: SimpleNamespace(token_claims=lambda: SimpleNamespace(identity="agent")),
    )
    session = LiveAvatarSession(avatar_id="avatar", api_key="key")
    session._api.create_streaming_session = AsyncMock(
        return_value=SessionResponse(data=SessionToken(session_id="session", session_token="token"))
    )
    session._api.start_streaming_session = AsyncMock(
        return_value=SessionResponse(data=SessionStart(ws_url="wss://provider.test"))
    )
    session._api.stop_streaming_session = AsyncMock(return_value=SessionStop(code=200))
    try:
        await start(session, setup)
        await asyncio.wait_for(entered.wait(), 1)
        await session.aclose()
        assert stopped.is_set()
    finally:
        if session._main_atask is not None:
            session._main_atask.cancel()
            await asyncio.gather(session._main_atask, return_exceptions=True)
        await session.aclose()
