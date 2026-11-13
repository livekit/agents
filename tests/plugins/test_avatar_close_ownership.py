"""Avatar plugins' close: it waits for the audio sink it owns, survives its own cancellation and
settles the provider connection it opened."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, http
from livekit.plugins import anam, liveavatar, runway
from tests.fakes import until

if TYPE_CHECKING:
    from tests.plugins.conftest import AvatarHost


OPTIONS = APIConnectOptions(max_retry=0, connect_timeout=1, read_timeout=1)


class _Provider:
    """An avatar provider's REST API and event socket; ``hold`` keeps a session cancel hanging.
    The socket keeps what the avatar sent (``heard``) and how it ended (``ended``: ``CLOSE`` for a
    close frame, ``CLOSED`` for a dropped connection)."""

    def __init__(self) -> None:
        self.cancelling = asyncio.Event()
        self.hold = asyncio.Event()
        self.hold.set()
        self.heard: list[str] = []
        self.ended: aiohttp.WSMsgType | None = None
        self.url = ""

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/engine/session", self.anam_session)
        app.router.add_post("/v1/realtime_sessions", self.runway_session)
        app.router.add_delete("/v1/realtime_sessions/{id}", self.runway_cancel)
        app.router.add_post("/sessions/token", self.liveavatar_token)
        app.router.add_post("/sessions/start", self.liveavatar_start)
        app.router.add_post("/sessions/stop", self.liveavatar_stop)
        app.router.add_get("/events", self.events)
        return app

    async def anam_session(self, _: web.Request) -> web.Response:
        return web.json_response({"sessionId": "S1"})

    async def runway_session(self, _: web.Request) -> web.Response:
        return web.json_response({"id": "rs_1"})

    async def runway_cancel(self, _: web.Request) -> web.Response:
        self.cancelling.set()
        await self.hold.wait()
        return web.json_response({})

    async def liveavatar_token(self, _: web.Request) -> web.Response:
        return web.json_response({"data": {"session_id": "S1", "session_token": "token"}})

    async def liveavatar_start(self, _: web.Request) -> web.Response:
        return web.json_response({"data": {"ws_url": self.url + "/events"}})

    async def liveavatar_stop(self, _: web.Request) -> web.Response:
        return web.json_response({"code": 100})

    async def events(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_json({"type": "session.state_updated", "state": "connected"})
        while (message := await socket.receive()).type is aiohttp.WSMsgType.TEXT:
            self.heard.append(message.data)
        self.ended = message.type
        return socket


@pytest.fixture
async def provider() -> AsyncIterator[_Provider]:
    provider = _Provider()
    async with TestServer(provider.app()) as server, http.open():
        provider.url = str(server.make_url("")).rstrip("/")
        yield provider
        provider.hold.set()


def _anam(provider: _Provider) -> anam.AvatarSession:
    return anam.AvatarSession(
        persona_config=anam.PersonaConfig(name="n", avatarId="a"),
        api_key="key",
        api_url=provider.url,
        conn_options=OPTIONS,
    )


async def test_concurrent_closes_return_only_once_the_owned_sink_closed(
    provider: _Provider, avatar_host: AvatarHost
) -> None:
    """A close issued while another is still closing the avatar's audio sink waits for that
    sink too."""
    avatar = _anam(provider)
    await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)
    [sink] = avatar_host.sinks
    release, sink_closed = asyncio.Event(), asyncio.Event()

    async def close_sink() -> None:
        await release.wait()
        sink_closed.set()

    sink.aclose.side_effect = close_sink
    seen_closed: list[bool] = []

    async def close() -> None:
        await avatar.aclose()
        seen_closed.append(sink_closed.is_set())

    async def release_once_closing() -> None:
        await until(lambda: sink.aclose.await_count, "the avatar never closed its sink")
        release.set()

    await asyncio.gather(close(), close(), release_once_closing())

    assert seen_closed == [True, True]


async def test_cancelling_a_runway_close_still_closes_its_audio_sink(
    provider: _Provider, avatar_host: AvatarHost
) -> None:
    """A Runway close cancelled while its provider session is still being cancelled closes the
    avatar's audio sink anyway."""
    provider.hold.clear()
    avatar = runway.AvatarSession(
        avatar_id="avatar", api_key="key", api_url=provider.url, conn_options=OPTIONS
    )
    await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)
    [sink] = avatar_host.sinks
    closing = asyncio.ensure_future(avatar.aclose())
    await provider.cancelling.wait()

    closing.cancel()

    with pytest.raises(asyncio.CancelledError):
        await closing
    closed_on_cancel = sink.aclose.await_count
    provider.hold.set()
    await avatar.aclose()

    assert closed_on_cancel == 1


async def test_closing_a_liveavatar_sends_its_event_socket_a_close_frame(
    provider: _Provider, avatar_host: AvatarHost
) -> None:
    """Closing a LiveAvatar session closes the event socket it opened with a close frame, rather
    than leaving it open or dropping the connection."""
    avatar = liveavatar.AvatarSession(
        avatar_id="avatar", api_key="key", api_url=provider.url + "/sessions", conn_options=OPTIONS
    )
    await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)
    # it speaks once its side of the handshake is done; a close before that is
    # test_liveavatar's stalled handshake
    await until(lambda: provider.heard, "the avatar spoke on its event socket")

    await avatar.aclose()

    # the frame is sent before aclose() returns; the provider reads it on its own time
    await until(lambda: provider.ended, "the provider saw its event socket end")
    assert provider.ended is aiohttp.WSMsgType.CLOSE
