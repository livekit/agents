"""Synthesia: the `SynthesiaAPI` session client against a real local HTTP peer."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APITimeoutError, http
from livekit.plugins.synthesia import AvatarConfig, AvatarSession
from livekit.plugins.synthesia.api import SESSION_PATH, SynthesiaAPI
from livekit.plugins.synthesia.errors import ErrorType, SynthesiaError
from livekit.plugins.synthesia.types import StartSessionRequest, StartSessionResponse

if TYPE_CHECKING:
    from tests.plugins.conftest import AvatarHost

API_KEY = "sk-secret-key"
NO_SLEEP = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0)


@dataclasses.dataclass
class _Call:
    url: str
    headers: dict[str, str]
    json: object


class _Peer:
    """Answers each POST with the next scripted (status, body, content_type)."""

    def __init__(self) -> None:
        self.outcomes: list[tuple[int, object, str]] = []
        self.calls: list[_Call] = []
        self.url = ""
        self.session: http.HttpSession | None = None

    async def respond(self, request: web.Request) -> web.Response:
        self.calls.append(_Call(str(request.url), dict(request.headers), await request.json()))
        status, body, content_type = self.outcomes.pop(0)
        return web.json_response(body, status=status, content_type=content_type)


@pytest.fixture
async def peer() -> AsyncIterator[_Peer]:
    peer = _Peer()
    app = web.Application()
    app.router.add_post("/{path:.*}", peer.respond)
    async with TestServer(app) as server, http.open() as session:
        peer.url = str(server.make_url("")).rstrip("/")
        peer.session = session
        yield peer


def _reply(peer: _Peer, status: int, body: object, content_type: str = "application/json") -> None:
    peer.outcomes.append((status, body, content_type))


async def _start(peer: _Peer) -> StartSessionResponse:
    client = SynthesiaAPI(
        api_key=API_KEY, api_url=peer.url, session=peer.session, conn_options=NO_SLEEP
    )
    return await client.start_session(
        StartSessionRequest(
            avatar_ids=["avatar-1"],
            livekit_url="wss://room.livekit.cloud",
            lk_token="lk-token-secret",
        )
    )


@pytest.mark.parametrize(
    ("code", "status", "expected"),
    [
        ("validation_error", 400, ErrorType.INVALID_SESSION_REQUEST),
        ("unknown_reference", 404, ErrorType.UNKNOWN_AVATAR),
        ("quota_exceeded", 402, ErrorType.QUOTA_EXCEEDED),
        ("rate_limited", 429, ErrorType.RATE_LIMITED),
        ("unauthenticated", 401, ErrorType.AUTH),
        ("not_authorized", 403, ErrorType.AUTH),
        ("forbidden", 403, ErrorType.AUTH),
        ("insufficient_scope", 403, ErrorType.AUTH),
        ("feature_not_in_plan", 403, ErrorType.FEATURE_NOT_IN_PLAN),
        ("feature_not_in_plan", 402, ErrorType.FEATURE_NOT_IN_PLAN),
        ("concurrency_limit", 429, ErrorType.CONCURRENCY_LIMIT),
        ("concurrency_limit_exceeded", 429, ErrorType.CONCURRENCY_LIMIT),
    ],
)
async def test_problem_type_maps_to_error_type(
    peer: _Peer, code: str, status: int, expected: ErrorType
) -> None:
    """https://github.com/livekit/agents/pull/7216"""
    problem = {
        "type": f"https://developers.synthesia.io/errors/{code}",
        "title": "t",
        "status": status,
        "code": code,
        "detail": "d",
    }
    _reply(peer, status, problem, "application/problem+json")
    with pytest.raises(SynthesiaError) as exc:
        await _start(peer)
    assert exc.value.type is expected


async def test_request_carries_auth_header_and_payload(peer: _Peer) -> None:
    """https://github.com/livekit/agents/pull/7216"""
    _reply(peer, 200, {"session_id": "sess_123"})
    await _start(peer)
    call = peer.calls[0]
    assert call.url == peer.url + SESSION_PATH
    assert call.headers["Authorization"] == API_KEY
    assert call.json == {
        "avatarIds": ["av_avatar-1"],
        "livekitUrl": "wss://room.livekit.cloud",
        "livekitToken": "lk-token-secret",
    }


async def test_secrets_never_logged(peer: _Peer, caplog: pytest.LogCaptureFixture) -> None:
    """https://github.com/livekit/agents/pull/7216"""
    caplog.set_level(logging.DEBUG, logger="livekit.plugins.synthesia")
    _reply(peer, 200, {"session_id": "sess_ok"})
    await _start(peer)
    _reply(peer, 401, {"error": {"code": "unauthorized"}})
    with pytest.raises(SynthesiaError):
        await _start(peer)
    assert API_KEY not in caplog.text
    assert "lk-token-secret" not in caplog.text


async def test_a_stalled_launch_fails_at_the_join_timeout(avatar_host: AvatarHost) -> None:
    """The session launch is bounded by ``join_timeout`` as one total deadline, not per read."""
    stalled = asyncio.Event()

    async def stall(request: web.Request) -> web.Response:
        stalled.set()
        await asyncio.Event().wait()  # never answers
        return web.json_response({"session_id": "late"})

    app = web.Application()
    app.router.add_post("/{path:.*}", stall)
    async with TestServer(app) as server, http.open():
        avatar = AvatarSession(
            AvatarConfig(avatar_ids=["avatar-1"]),
            api_key=API_KEY,
            api_url=str(server.make_url("")).rstrip("/"),
            join_timeout=0.05,
        )
        started = time.monotonic()

        with pytest.raises(APITimeoutError) as raised:
            await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)

    assert stalled.is_set()
    assert raised.value.timeout == "total"
    assert time.monotonic() - started < 5
