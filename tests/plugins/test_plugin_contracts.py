"""Contracts shared across plugins: avatar session creation and TTS sample-rate sync.

Each avatar starts through ``AvatarSession.start`` against a local provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from _pytest.mark import ParameterSet
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIError, APITimeoutError, http
from livekit.agents.voice.avatar import AvatarSession
from livekit.plugins import (
    anam,
    avatario,
    avatartalk,
    bey,
    bithuman,
    boson_avatar,
    did,
    keyframe,
    lemonslice,
    liveavatar,
    protoface,
    runway,
    simli,
    spatius,
    synthesia,
    tavus,
    trugen,
)

if TYPE_CHECKING:
    from tests.plugins.conftest import AvatarHost


@dataclass
class Avatar:
    """An avatar plugin against the local provider: ``session`` is the request (method and path)
    that creates the provider session, ``created`` its valid answer, ``others`` the valid
    answers to the plugin's other requests by method and path (``{}`` when absent)."""

    make: Callable[[str, APIConnectOptions], AvatarSession]
    session: str
    created: object
    others: dict[str, object] = field(default_factory=dict)


BOSON_SESSION = {"id": "s", "object": "avatar.livekit.session", "avatar_identity": "avatar-1"}

AVATARS: dict[str, Avatar] = {
    "anam": Avatar(
        lambda url, options: anam.AvatarSession(
            persona_config=anam.PersonaConfig(name="avatar", avatarId="avatar"),
            api_key="key",
            api_url=url,
            conn_options=options,
        ),
        "POST /v1/engine/session",
        {"sessionId": "s"},
    ),
    "avatario": Avatar(
        lambda url, options: avatario.AvatarSession(
            avatar_id="avatar", api_key="key", conn_options=options
        ),
        "POST /start-session",
        {"id": "s"},
    ),
    "avatartalk": Avatar(
        lambda url, options: avatartalk.AvatarSession(api_secret="key", api_url=url),
        "POST /livekit/create-session",
        {"task_id": "s"},
    ),
    "bey": Avatar(
        lambda url, options: bey.AvatarSession(api_key="key", api_url=url, conn_options=options),
        "POST /v1/session",
        {"id": "s"},
    ),
    "bithuman": Avatar(
        lambda url, options: bithuman.AvatarSession(
            api_secret="secret", avatar_id="avatar", api_url=url, conn_options=options
        ),
        "POST /",
        {"id": "s"},
    ),
    "boson_avatar": Avatar(
        lambda url, options: boson_avatar.AvatarSession(
            avatar_id="avatar",
            api_key="key",
            api_url=url,
            avatar_participant_identity="avatar-1",
            conn_options=options,
        ),
        "POST /sessions",
        {**BOSON_SESSION, "status": "active"},
        {"DELETE /sessions/s": {**BOSON_SESSION, "status": "terminated"}},
    ),
    "did": Avatar(
        lambda url, options: did.AvatarSession(
            agent_id="avatar", api_key="key", api_url=url, conn_options=options
        ),
        "POST /v2/agents/avatar/sessions/join",
        {"id": "s"},
    ),
    "keyframe": Avatar(
        lambda url, options: keyframe.AvatarSession(
            persona_id="avatar", api_key="key", api_url=url, conn_options=options
        ),
        "POST /v1/sessions/plugins/livekit",
        {"reservation_id": "s"},
    ),
    "lemonslice": Avatar(
        lambda url, options: lemonslice.AvatarSession(
            agent_id="avatar", api_key="key", api_url=url, conn_options=options
        ),
        "POST /",
        {"session_id": "s"},
    ),
    "liveavatar": Avatar(
        lambda url, options: liveavatar.AvatarSession(
            avatar_id="avatar", api_key="key", api_url=url, conn_options=options
        ),
        "POST /token",
        {"data": {"session_id": "s", "session_token": "token"}},
        {"POST /start": {"data": {"ws_url": "ws://127.0.0.1:9/"}}, "POST /stop": {"code": 100}},
    ),
    "protoface": Avatar(
        lambda url, options: protoface.AvatarSession(
            avatar_id="avatar", api_key="key", api_url=url, conn_options=options
        ),
        "POST /v1/sessions",
        {"id": "s"},
    ),
    "runway": Avatar(
        lambda url, options: runway.AvatarSession(
            avatar_id="avatar", api_key="key", api_url=url, conn_options=options
        ),
        "POST /v1/realtime_sessions",
        {"id": "s"},
    ),
    "simli": Avatar(
        lambda url, options: simli.AvatarSession(
            simli_config=simli.SimliConfig(api_key="key", face_id="face"), api_url=url
        ),
        "POST /integrations/livekit/agents",
        {},
        {"POST /compose/token": {"session_token": "token"}},
    ),
    "spatius": Avatar(
        lambda url, options: spatius.AvatarSession(
            api_key="key",
            app_id="app",
            avatar_id="avatar",
            console_endpoint_url=url,
            ingress_endpoint_url=url,
            sample_rate=24000,
        ),
        "POST /session-tokens",
        {"sessionToken": "token"},
    ),
    "synthesia": Avatar(
        lambda url, options: synthesia.AvatarSession(
            synthesia.AvatarConfig(avatar_ids=["avatar"]), api_key="key", api_url=url
        ),
        "POST /api/interactive-avatars/sessions",
        {"id": "s"},
    ),
    "tavus": Avatar(
        lambda url, options: tavus.AvatarSession(api_key="key", api_url=url, conn_options=options),
        "POST /conversations",
        {"conversation_id": "s"},
    ),
    "trugen": Avatar(
        lambda url, options: trugen.AvatarSession(api_key="key", conn_options=options),
        "POST /v1/sessions",
        {},
    ),
}

Reply = tuple[int, object] | None  # None holds the answer until the test ends

COMMIT = "https://github.com/livekit/agents/commit/9e79bc520b80553e3ca665884a2fac2beeb63042"
NOT_RETRIED = {
    "avatartalk": "avatartalk/api.py:52 posts without retry_request: a 503 fails start",
    "simli": "simli/avatar.py:125,132 post without retry_request: a 503 fails start",
    "spatius": "the spatius SDK's session-token POST (spatius/avatar.py:246) has no retries: "
    "a 503 fails start with SpatiusException",
    "synthesia": "synthesia/avatar.py:194 forces max_retry=0: a 503 fails start",
}
NO_READ_TIMEOUT = {
    "avatartalk": "avatartalk.AvatarSession takes no conn_options",
    "simli": "simli.AvatarSession takes no conn_options",
    "spatius": "spatius.AvatarSession takes no conn_options; the SDK's token POST has a fixed "
    "10 s total timeout",
    "synthesia": "synthesia.AvatarSession takes no conn_options; start uses the default "
    "10 s read timeout",
}


def _every_avatar(**marks: pytest.MarkDecorator) -> list[ParameterSet]:
    return [pytest.param(name, marks=marks.get(name, ())) for name in AVATARS]


async def _start(
    provider: str,
    replies: list[Reply],
    options: APIConnectOptions,
    requests: list[str],
    avatar_host: AvatarHost,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starts ``provider``'s avatar against a local provider that answers its session request
    with the next of ``replies`` (the last one repeats) and every other request validly,
    keeping each request's method and path in ``requests``."""
    avatar = AVATARS[provider]
    release = asyncio.Event()

    async def answer(request: web.Request) -> web.Response:
        route = f"{request.method} {request.path}"
        requests.append(route)
        if route != avatar.session:
            return web.json_response(avatar.others.get(route, {}))
        reply = replies[min(requests.count(route), len(replies)) - 1]
        if reply is None:
            await release.wait()
            return web.json_response(avatar.created)
        status, body = reply
        return web.json_response(body, status=status)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", answer)
    async with TestServer(app) as server, http.open():
        url = str(server.make_url("")).rstrip("/")
        monkeypatch.setattr("livekit.plugins.avatario.api.DEFAULT_API_URL", url)
        monkeypatch.setattr("livekit.plugins.trugen.avatar._BASE_API_URL", url)
        session = avatar.make(url, options)
        try:
            await session.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)
        finally:
            release.set()
            await session.aclose()


@pytest.mark.parametrize(
    "provider", ["anam", "avatario", "avatartalk", "did", "keyframe", "lemonslice", "liveavatar"]
)
async def test_invalid_success_response_does_not_repeat_creation(
    provider: str, avatar_host: AvatarHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful avatar POST with an invalid body breaks the provider's contract: it fails
    for good and is never repeated, since the provider already created the session."""
    requests: list[str] = []
    options = APIConnectOptions(max_retry=2, retry_interval=0)

    with pytest.raises(APIError) as error:
        await _start(provider, [(200, [])], options, requests, avatar_host, monkeypatch)

    assert error.value.kind == "protocol"
    assert requests.count(AVATARS[provider].session) == 1


@pytest.mark.parametrize("provider", ["avatario", "bey", "bithuman", "runway"])
async def test_zero_retries_still_creates_avatar(
    provider: str, avatar_host: AvatarHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With max_retry=0 the avatar is still created, once. https://github.com/livekit/agents/issues/7604"""
    requests: list[str] = []
    reply = (200, AVATARS[provider].created)

    await _start(
        provider, [reply], APIConnectOptions(max_retry=0), requests, avatar_host, monkeypatch
    )

    assert requests.count(AVATARS[provider].session) == 1


@pytest.mark.parametrize(
    "provider",
    _every_avatar(
        spatius=pytest.mark.xfail(
            strict=True,
            reason=f"{COMMIT}: a 401 on the spatius SDK's session-token POST fails start with "
            "SpatiusException (spatius/avatar.py:286), which is no APIError",
        ),
    ),
)
async def test_unauthorized_start_is_not_retried(
    provider: str, avatar_host: AvatarHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 401 on session creation fails start after one request with a non-retryable APIError.
    Catches an avatar that retries a rejected key."""
    requests: list[str] = []
    options = APIConnectOptions(max_retry=2, retry_interval=0)

    with pytest.raises(APIError) as error:
        await _start(provider, [(401, {})], options, requests, avatar_host, monkeypatch)

    assert error.value.kind == "auth"
    assert requests.count(AVATARS[provider].session) == 1


@pytest.mark.parametrize(
    "provider",
    _every_avatar(
        **{
            name: pytest.mark.xfail(strict=True, reason=f"{COMMIT}: {bug}")
            for name, bug in NOT_RETRIED.items()
        }
    ),
)
async def test_unavailable_start_is_retried(
    provider: str, avatar_host: AvatarHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 503 on session creation is retried within max_retry, and the retry's valid answer
    starts the avatar. Catches an avatar that gives up on a transient provider failure."""
    requests: list[str] = []
    replies: list[Reply] = [(503, {}), (200, AVATARS[provider].created)]
    options = APIConnectOptions(max_retry=1, retry_interval=0)

    await _start(provider, replies, options, requests, avatar_host, monkeypatch)

    assert requests.count(AVATARS[provider].session) == 2


@pytest.mark.parametrize(
    "provider",
    _every_avatar(**{name: pytest.mark.skip(reason=why) for name, why in NO_READ_TIMEOUT.items()}),
)
async def test_read_timeout_start_is_not_resent(
    provider: str, avatar_host: AvatarHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session request the provider received but did not answer within read_timeout fails
    start with APITimeoutError and is never sent again. Catches an avatar whose retry could
    create a second provider session."""
    requests: list[str] = []
    options = APIConnectOptions(max_retry=2, retry_interval=0, read_timeout=0.05)

    with pytest.raises(APITimeoutError):
        await _start(provider, [None], options, requests, avatar_host, monkeypatch)

    assert requests.count(AVATARS[provider].session) == 1


@pytest.mark.parametrize("plugin", ["inworld", "lmnt", "smallestai"])
def test_update_options_syncs_public_sample_rate(plugin: str) -> None:
    """``TTS.sample_rate``, which the fallback adapter resamples from, follows the option.

    https://github.com/livekit/agents/pull/7135
    """
    tts = __import__(f"livekit.plugins.{plugin}", fromlist=["TTS"]).TTS(api_key="key")
    assert tts.sample_rate != 16000

    tts.update_options(sample_rate=16000)

    assert tts.sample_rate == 16000
