"""Runway avatar: closing the avatar ends the provider's realtime session."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, CloseEvent, CloseReason, http
from livekit.plugins.runway import AvatarSession

if TYPE_CHECKING:
    from tests_new.plugins.conftest import AvatarHost

OPTIONS = APIConnectOptions(max_retry=0, connect_timeout=1, read_timeout=1)


class _Runway:
    def __init__(self) -> None:
        self.deletes: list[tuple[str, str | None]] = []

    async def create(self, _: web.Request) -> web.Response:
        return web.json_response({"id": "rs_1"})

    async def cancel(self, request: web.Request) -> web.Response:
        self.deletes.append((request.path, request.headers.get("Authorization")))
        return web.json_response({})

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/realtime_sessions", self.create)
        app.router.add_delete("/v1/realtime_sessions/{id}", self.cancel)
        return app


async def test_closing_the_avatar_in_a_connected_room_sends_end_call_once(
    avatar_host: AvatarHost,
) -> None:
    """While the room is up, closing the session and the avatar sends one END_CALL, no cancel.

    https://github.com/livekit/agents/pull/5612 https://github.com/livekit/agents/pull/5623
    """
    published: list[tuple[dict[str, str], list[str]]] = []

    async def publish_data(
        payload: bytes, *, reliable: bool, destination_identities: list[str]
    ) -> None:
        published.append((json.loads(payload), destination_identities))

    # a connected room: the avatar only names it and publishes data to it
    room = cast(
        rtc.Room,
        SimpleNamespace(
            name="room-1",
            isconnected=lambda: True,
            local_participant=SimpleNamespace(publish_data=publish_data),
        ),
    )
    runway = _Runway()
    async with TestServer(runway.app()) as server, http.open():
        avatar = AvatarSession(
            avatar_id="avatar",
            api_key="runway-key",
            api_url=str(server.make_url("")).rstrip("/"),
            conn_options=OPTIONS,
        )
        await avatar.start(avatar_host.agent, room, **avatar_host.credentials)

        avatar_host.agent.emit("close", CloseEvent(reason=CloseReason.USER_INITIATED))
        await avatar.aclose()
        await avatar.aclose()

    assert published == [({"type": "END_CALL"}, ["runway-avatar-agent"])]
    assert runway.deletes == []


async def test_closing_the_avatar_after_the_room_disconnected_cancels_the_session(
    avatar_host: AvatarHost,
) -> None:
    """After the room is gone, closing the avatar cancels the Runway session by its id.

    https://github.com/livekit/agents/pull/5612 https://github.com/livekit/agents/pull/5623
    """
    room = cast(rtc.Room, SimpleNamespace(name="room-1", isconnected=lambda: False))
    runway = _Runway()
    async with TestServer(runway.app()) as server, http.open():
        avatar = AvatarSession(
            avatar_id="avatar",
            api_key="runway-key",
            api_url=str(server.make_url("")).rstrip("/"),
            conn_options=OPTIONS,
        )
        await avatar.start(avatar_host.agent, room, **avatar_host.credentials)

        await avatar.aclose()

    assert runway.deletes == [("/v1/realtime_sessions/rs_1", "Bearer runway-key")]
