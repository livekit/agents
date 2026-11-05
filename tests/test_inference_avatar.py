"""`inference.AvatarSession` against a local gateway: one provider session per instance, and every
session the gateway creates is either held or terminated."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIStatusError
from livekit.agents.inference import AvatarSession, avatar as avatar_mod
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession

pytestmark = pytest.mark.unit

CREATED = {
    "session_id": "AVS_1",
    "provider_session_id": "ls_1",
    "terminate_token": "tt_1",
    "sample_rate": 24000,
}


class _Gateway:
    """The gateway's create and terminate routes: `answers` are the create replies in order
    (a status or a body); a create waits for `release` when it is cleared."""

    def __init__(self, *answers: int | dict[str, Any]) -> None:
        self.answers = list(answers)
        self.creates: list[dict[str, Any]] = []
        self.terminates: list[dict[str, Any]] = []
        self.create_received = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def _create(self, request: web.Request) -> web.Response:
        self.creates.append(await request.json())
        self.create_received.set()
        await self.release.wait()
        answer = self.answers.pop(0)
        if isinstance(answer, int):
            return web.json_response({"error": "refused"}, status=answer)
        return web.json_response(answer)

    async def _terminate(self, request: web.Request) -> web.Response:
        self.terminates.append(await request.json())
        return web.json_response({"terminated": True})

    @contextlib.asynccontextmanager
    async def serve(self) -> AsyncIterator[str]:
        app = web.Application()
        app.router.add_post("/v1/avatar/sessions", self._create)
        app.router.add_post("/v1/avatar/sessions/terminate", self._terminate)
        server = TestServer(app)
        await server.start_server()
        try:
            yield str(server.make_url("/v1"))
        finally:
            await server.close()


class _LocalParticipant:
    identity = "agent"


class _Room:
    """A connected room as `start()` reads it."""

    name = "room"
    local_participant = _LocalParticipant()
    remote_participants: dict[str, Any] = {}

    def isconnected(self) -> bool:
        return True

    @property
    def sid(self) -> Any:
        async def sid() -> str:
            return "RM_1"

        return sid()

    def on(self, *_: Any) -> None:
        pass

    def off(self, *_: Any) -> None:
        pass


class _AudioSink:
    def __init__(self, room: object, *, sample_rate: int | None = None, **kwargs: object) -> None:
        self.sample_rate = sample_rate
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _Output:
    def __init__(self) -> None:
        self.audio_tails: list[_AudioSink] = []

    def replace_audio_tail(self, sink: _AudioSink) -> None:
        self.audio_tails.append(sink)


class _VoiceSession:
    def __init__(self) -> None:
        self.output = _Output()

    def on(self, *_: Any) -> None:
        pass

    def off(self, *_: Any) -> None:
        pass


def _avatar(base_url: str) -> AvatarSession:
    return AvatarSession("lemonslice", base_url=base_url, api_key="key", api_secret="secret")


async def _start(avatar: AvatarSession, session: _VoiceSession | None = None) -> None:
    await avatar.start(
        session or _VoiceSession(),  # type: ignore[arg-type]
        _Room(),  # type: ignore[arg-type]
        livekit_url="wss://example.livekit.cloud",
    )


@pytest.fixture(autouse=True)
def _audio_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    """The avatar's audio sink is the room's; these tests own the gateway, not the room."""
    monkeypatch.setattr(avatar_mod, "DataStreamAudioOutput", _AudioSink)

    async def start(self: BaseAvatarSession, agent_session: Any, room: Any) -> None:
        self._room, self._agent_session = room, agent_session

    monkeypatch.setattr(BaseAvatarSession, "start", start)


async def test_a_second_start_while_the_first_creates_is_refused() -> None:
    gateway = _Gateway(CREATED)
    gateway.release.clear()
    async with gateway.serve() as base_url:
        avatar = _avatar(base_url)
        first = asyncio.create_task(_start(avatar))
        await gateway.create_received.wait()

        with pytest.raises(RuntimeError, match="only be called once"):
            await asyncio.wait_for(_start(avatar), 5)

        gateway.release.set()
        await first
        assert avatar.session_id == "AVS_1"
        await avatar.aclose()

    assert len(gateway.creates) == 1


async def test_aclose_terminates_the_provider_session() -> None:
    gateway = _Gateway(CREATED)
    async with gateway.serve() as base_url:
        avatar = _avatar(base_url)
        session = _VoiceSession()
        await _start(avatar, session)
        assert session.output.audio_tails[0].sample_rate == 24000
        assert avatar.provider_session_id == "ls_1"

        await avatar.aclose()
        assert session.output.audio_tails[0].closed

    assert gateway.terminates == [
        {"provider": "lemonslice", "provider_session_id": "ls_1", "terminate_token": "tt_1"}
    ]
    assert avatar.provider_session_id is None


async def test_a_failed_create_can_be_started_again() -> None:
    gateway = _Gateway(400, CREATED)
    async with gateway.serve() as base_url:
        avatar = _avatar(base_url)
        with pytest.raises(APIStatusError):
            await _start(avatar)

        await _start(avatar)
        assert avatar.session_id == "AVS_1"
        await avatar.aclose()

    assert len(gateway.creates) == 2
    assert len(gateway.terminates) == 1
