"""Boson avatar: the hosted session API through the AvatarSession lifecycle, against a local peer.

https://github.com/livekit/agents/pull/7019
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from multidict import CIMultiDict

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, CloseEvent, http
from livekit.agents.voice.events import CloseReason
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.plugins.boson_avatar import (
    AvatarInfo,
    AvatarSession,
    BosonAvatarException,
    list_avatars,
)
from livekit.plugins.boson_avatar.api import AvatarSessionStartError
from tests_new.fakes import until

if TYPE_CHECKING:
    from tests_new.plugins.conftest import AvatarHost

# Any: decoded wire JSON, indexed by the provider's documented shape
Wire = dict[str, Any]

LOGGER = "livekit.plugins.boson_avatar"
PROVIDER_SECRET = "private-provider-response"
ACTIVE = {
    "id": "avatar-session-1",
    "object": "avatar.livekit.session",
    "status": "active",
    "avatar_identity": "avatar-1",
}


TRUNCATED = b"{"  # a payload cut off mid-body, then the connection drops


@dataclass
class Boson:
    """The local Boson API: its queued outcomes, the calls it got, and the methods it holds
    unanswered until their event is set."""

    outcomes: list[tuple[int, Mapping[str, object] | bytes | None]] = field(default_factory=list)
    calls: list[tuple[str, str, CIMultiDict[str], Wire | None]] = field(default_factory=list)
    holds: dict[str, asyncio.Event] = field(default_factory=dict)

    def methods(self) -> list[str]:
        return [call[0] for call in self.calls]


@pytest.fixture
async def boson(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Boson]:
    """A local Boson API answering each request with the next ``(status, payload)``."""
    peer = Boson()

    async def respond(request: web.Request) -> web.StreamResponse:
        body = await request.json() if request.can_read_body else None
        peer.calls.append((request.method, request.path, request.headers.copy(), body))
        if hold := peer.holds.get(request.method):
            await hold.wait()
        status, payload = peer.outcomes.pop(0)
        if payload is None:
            return web.Response(status=status)
        if isinstance(payload, bytes):
            headers = {"Content-Type": "application/json", "Content-Length": "100"}
            response = web.StreamResponse(status=status, headers=headers)
            await response.prepare(request)
            await response.write(payload)
            assert request.transport is not None
            request.transport.close()
            return response
        return web.json_response(payload, status=status)

    for name in ("BOSON_API_KEY", "BOSONAI_API_KEY", "BOSON_AVATAR_ID"):
        monkeypatch.delenv(name, raising=False)
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", respond)
    async with TestServer(app) as server, http.open():
        monkeypatch.setenv("BOSON_AVATAR_API_URL", str(server.make_url("/v1")))
        yield peer


def _avatar(max_retry: int = 0) -> AvatarSession:
    return AvatarSession(
        avatar_id="asset-1",
        api_key="boson-key",
        avatar_participant_identity="avatar-1",
        conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0),
    )


async def _start(avatar: AvatarSession, avatar_host: AvatarHost) -> str:
    return await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name.startswith(LOGGER) and r.levelno >= logging.WARNING]


async def test_start_and_close_use_the_hosted_contract(
    avatar_host: AvatarHost, boson: Boson
) -> None:
    """Start creates the session with a scoped avatar token; close deletes it."""
    boson.outcomes = [(201, ACTIVE), (204, None)]
    avatar = AvatarSession(
        avatar_id="asset-1",
        api_key="boson-key",
        avatar_participant_identity="avatar-1",
        avatar_participant_name="Demo Avatar",
        width=640,
        height=480,
        max_duration_seconds=900,
        conn_options=APIConnectOptions(max_retry=0),
    )

    session_id = await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)
    await avatar.aclose()

    assert session_id == "avatar-session-1"
    [(_, _, headers, create), delete] = boson.calls
    assert create is not None
    assert delete[:2] == ("DELETE", "/v1/sessions/avatar-session-1")
    assert headers["Authorization"] == "Bearer boson-key"
    uuid.UUID(headers["Idempotency-Key"])
    token = create["transport"].pop("participant_token")
    assert create == {
        "avatar_id": "asset-1",
        "transport": {
            "type": "livekit",
            "url": "wss://lk.test",
            "room_name": "room-1",
            "participant_identity": "avatar-1",
            "publisher_identity": "agent-1",
            "audio_source": "data_stream",
        },
        "output": {"width": 640, "height": 480},
        "max_duration_seconds": 900,
    }
    claims = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "=="))
    assert (claims["sub"], claims["name"], claims["kind"]) == ("avatar-1", "Demo Avatar", "agent")
    assert claims["video"]["room"] == "room-1"
    assert not claims["video"]["canSubscribe"]
    assert set(claims["video"]["canPublishSources"]) == {"camera", "microphone"}
    assert claims["attributes"][ATTRIBUTE_PUBLISH_ON_BEHALF] == "agent-1"
    [sink] = avatar_host.sinks
    assert sink.options["wait_remote_track"] == rtc.TrackKind.KIND_AUDIO


async def test_non_retryable_auth_error_preserves_status_and_request_id(
    avatar_host: AvatarHost, boson: Boson
) -> None:
    """A 401 keeps its status and request id and is not retried."""
    boson.outcomes = [(401, {"error": {"code": "invalid_api_key", "request_id": "req-1"}})]
    avatar = AvatarSession(
        avatar_id="asset-1", api_key="bad-key", conn_options=APIConnectOptions(max_retry=3)
    )

    with pytest.raises(APIStatusError) as raised:
        await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)
    await avatar.aclose()

    assert (raised.value.status_code, raised.value.request_id) == (401, "req-1")
    assert len(boson.calls) == 1


async def test_compensation_log_does_not_expose_provider_error(
    avatar_host: AvatarHost, boson: Boson, caplog: pytest.LogCaptureFixture
) -> None:
    """A session started for the wrong avatar is deleted; a failed delete logs no provider text."""
    boson.outcomes = [
        (201, {**ACTIVE, "avatar_identity": "wrong-avatar"}),
        *[(503, {"error": {"message": PROVIDER_SECRET}})] * 3,
    ]
    avatar = _avatar()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with pytest.raises(AvatarSessionStartError):
            await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)
        await avatar.aclose()

    [create, *deletes] = [call[:2] for call in boson.calls]
    assert create == ("POST", "/v1/sessions")
    assert deletes and set(deletes) == {("DELETE", "/v1/sessions/avatar-session-1")}
    record = _warnings(caplog)[0]
    assert record.exc_info is None
    assert record.__dict__["lk.pii.session_id"] == "avatar-session-1"
    assert PROVIDER_SECRET not in caplog.text


async def test_close_retries_a_failed_end_and_stays_idempotent(
    avatar_host: AvatarHost, boson: Boson, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed end is retried by the next close; later closes send nothing."""
    boson.outcomes = [(201, ACTIVE), (503, {"error": {"message": PROVIDER_SECRET}}), (204, None)]
    avatar = _avatar()
    await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await avatar.aclose()
        await avatar.aclose()
        await avatar.aclose()

    assert [call[0] for call in boson.calls] == ["POST", "DELETE", "DELETE"]
    assert avatar.session_id is None
    record = _warnings(caplog)[0]
    assert record.exc_info is None
    assert PROVIDER_SECRET not in caplog.text


async def test_start_twice_is_refused(avatar_host: AvatarHost, boson: Boson) -> None:
    """An AvatarSession starts once."""
    boson.outcomes = [(201, ACTIVE), (204, None)]
    avatar = _avatar()
    await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)

    with pytest.raises(RuntimeError, match="called twice"):
        await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)
    await avatar.aclose()


async def test_list_avatars_reads_the_provider_catalog(boson: Boson) -> None:
    """``list_avatars`` returns the project's avatars from an authenticated catalog GET."""
    catalog = [
        {"avatar_id": "asset_demo", "name": "Claire"},
        {"avatar_id": "asset_1", "name": "Emma"},
    ]
    boson.outcomes = [(200, {"object": "avatar.list", "data": catalog})]

    avatars = await list_avatars(api_key="boson-key", conn_options=APIConnectOptions(max_retry=0))

    assert avatars == [AvatarInfo("asset_demo", "Claire"), AvatarInfo("asset_1", "Emma")]
    [(method, path, headers, _)] = boson.calls
    assert (method, path) == ("GET", "/v1/avatars")
    assert headers["Authorization"] == "Bearer boson-key"


@pytest.mark.parametrize(
    "catalog",
    [
        {"object": "wrong", "data": []},
        {"object": "avatar.list", "data": {}},
        {"object": "avatar.list", "data": ["asset_demo"]},
        {"object": "avatar.list", "data": [{"avatar_id": "", "name": "Claire"}]},
        {"object": "avatar.list", "data": [{"avatar_id": "asset_demo"}]},
        {
            "object": "avatar.list",
            "data": [
                {"avatar_id": "asset_demo", "name": "Claire"},
                {"avatar_id": "asset_demo", "name": "Duplicate"},
            ],
        },
    ],
    ids=["object", "data", "entry", "empty-id", "no-name", "duplicate"],
)
async def test_list_avatars_rejects_an_invalid_catalog(boson: Boson, catalog: Wire) -> None:
    """A catalog that breaks the documented shape is refused, not partly returned."""
    boson.outcomes = [(200, catalog)]

    with pytest.raises(BosonAvatarException):
        await list_avatars(api_key="boson-key", conn_options=APIConnectOptions(max_retry=0))


@pytest.mark.parametrize(
    "options",
    [
        {"width": 640},
        {"width": 0, "height": 640},
        {"width": True, "height": 640},
        {"width": 1.5, "height": 640},
        {"width": "wide", "height": 640},
        {"max_duration_seconds": -1},
        {"max_duration_seconds": 1.5},
        {"max_duration_seconds": 14_401},
        {"idempotency_key": ""},
        {"idempotency_key": 123},
        {"idempotency_key": "application-session-1"},
    ],
)
def test_constructor_rejects_invalid_configuration(options: Wire) -> None:
    """Size, duration and idempotency key are validated before any request."""
    with pytest.raises(BosonAvatarException):
        AvatarSession(
            avatar_id="asset-1", api_key="boson-key", api_url="https://a.test/v1", **options
        )


@pytest.mark.parametrize(
    "api_url",
    [
        "/",
        "avatar.example/v1",
        "ftp://avatar.example/v1",
        "https:///v1",
        "https://avatar.example/v1?region=us",
        "https://avatar.example/v1#sessions",
        "https://user:password@avatar.example/v1",
        "https://avatar.example:invalid/v1",
        "http://avatar.example/v1",
    ],
)
def test_api_url_must_be_a_plain_https_base(api_url: str) -> None:
    """The API key is a bearer credential: the URL must be HTTPS, without credentials or query."""
    with pytest.raises(BosonAvatarException):
        AvatarSession(avatar_id="asset-1", api_key="boson-key", api_url=api_url)


@pytest.mark.parametrize(
    "api_url",
    [
        "http://localhost:8400/v1/",
        "http://worker.localhost:8400/v1/",
        "http://LOCALHOST:8400/v1/",
        "http://127.0.0.1:8400/v1/",
        "http://[::1]:8400/v1/",
    ],
)
def test_plain_http_is_allowed_for_a_loopback_api_url(api_url: str) -> None:
    """Local development may reach a loopback service over plain HTTP."""
    AvatarSession(avatar_id="asset-1", api_key="boson-key", api_url=api_url)


async def test_retry_reuses_one_idempotency_key(avatar_host: AvatarHost, boson: Boson) -> None:
    """A retried create sends the same Idempotency-Key, so Boson allocates one session."""
    boson.outcomes = [(503, {"error": {"code": "busy"}}), (201, ACTIVE)]

    await _start(_avatar(max_retry=1), avatar_host)

    keys = [headers["Idempotency-Key"] for _, _, headers, _ in boson.calls]
    assert len(keys) == 2
    assert keys[0] == keys[1]


async def test_job_idempotency_key_is_stable_across_redelivery(
    avatar_host: AvatarHost, boson: Boson
) -> None:
    """Inside a job, the default key is a UUID derived from the job, the same on each start."""
    boson.outcomes = [(201, ACTIVE), (201, ACTIVE)]

    await _start(_avatar(), avatar_host)
    await _start(_avatar(), avatar_host)

    first, second = [headers["Idempotency-Key"] for _, _, headers, _ in boson.calls]
    assert first == second
    uuid.UUID(first)


async def test_explicit_idempotency_key_overrides_the_job_default(
    avatar_host: AvatarHost, boson: Boson
) -> None:
    """An idempotency_key passed by the application is the one sent."""
    boson.outcomes = [(201, ACTIVE)]
    key = "123e4567-e89b-12d3-a456-426614174000"
    avatar = AvatarSession(
        avatar_id="asset-1",
        api_key="boson-key",
        avatar_participant_identity="avatar-1",
        idempotency_key=key,
        conn_options=APIConnectOptions(max_retry=0),
    )

    await _start(avatar, avatar_host)

    [(_, _, headers, _)] = boson.calls
    assert headers["Idempotency-Key"] == key


async def test_non_contract_success_status_is_not_retried(
    avatar_host: AvatarHost, boson: Boson
) -> None:
    """A 2xx other than 200/201 on create fails with its status and is never retried."""
    boson.outcomes = [(202, ACTIVE)]

    with pytest.raises(APIStatusError) as raised:
        await _start(_avatar(max_retry=3), avatar_host)

    assert raised.value.status_code == 202
    assert not raised.value.retryable
    assert boson.methods() == ["POST"]


@pytest.mark.parametrize(
    ("created", "methods"),
    [
        ({k: v for k, v in ACTIVE.items() if k != "id"}, ["POST"]),
        ({**ACTIVE, "avatar_identity": "wrong-avatar"}, ["POST", "DELETE"]),
        ({**ACTIVE, "status": "pending"}, ["POST", "DELETE"]),
    ],
    ids=["no-id", "wrong-identity", "not-active"],
)
async def test_invalid_created_session_is_refused_and_deleted(
    avatar_host: AvatarHost, boson: Boson, created: Wire, methods: list[str]
) -> None:
    """A create answer that breaks the contract fails start; a session it names is deleted."""
    boson.outcomes = [(201, created), (204, None)]

    with pytest.raises(BosonAvatarException):
        await _start(_avatar(), avatar_host)

    assert boson.methods() == methods


async def test_start_refuses_an_avatar_identity_equal_to_the_agent(
    avatar_host: AvatarHost, boson: Boson
) -> None:
    """The avatar cannot join as the agent it speaks for: nothing is created or rerouted."""
    avatar = AvatarSession(
        avatar_id="asset-1", api_key="boson-key", avatar_participant_identity="agent-1"
    )

    with pytest.raises(BosonAvatarException, match="must differ"):
        await _start(avatar, avatar_host)

    assert boson.calls == []
    avatar_host.audio_tail.assert_not_called()
    avatar_host.framework_start.assert_not_awaited()


async def test_failed_start_after_create_deletes_the_session(
    avatar_host: AvatarHost, boson: Boson
) -> None:
    """When routing audio fails after the session is created, start deletes that session."""
    boson.outcomes = [(201, ACTIVE), (204, None)]
    avatar_host.audio_tail.side_effect = RuntimeError("sink rejected")
    avatar = _avatar()

    with pytest.raises(RuntimeError, match="sink rejected"):
        await _start(avatar, avatar_host)

    assert boson.methods() == ["POST", "DELETE"]
    assert avatar.session_id is None


async def test_failed_compensation_keeps_the_session_for_close(
    avatar_host: AvatarHost, boson: Boson
) -> None:
    """A session whose start-time deletes failed is still deleted by the next close."""
    boson.outcomes = [
        (201, {**ACTIVE, "avatar_identity": "wrong-avatar"}),
        *[(503, None)] * 2,
        (204, None),
    ]
    avatar = _avatar()
    with pytest.raises(AvatarSessionStartError):
        await _start(avatar, avatar_host)
    kept = avatar.session_id

    await avatar.aclose()

    assert kept == "avatar-session-1"
    assert avatar.session_id is None
    assert boson.methods() == ["POST", "DELETE", "DELETE", "DELETE"]


@pytest.mark.parametrize(
    "ended",
    [
        (200, {"id": "avatar-session-1", "object": "avatar.livekit.session", "status": "active"}),
        (200, TRUNCATED),
    ],
    ids=["not-terminated", "truncated"],
)
async def test_close_keeps_the_session_when_the_end_is_not_confirmed(
    avatar_host: AvatarHost, boson: Boson, ended: tuple[int, Wire | bytes]
) -> None:
    """A delete answered without a terminated session leaves it for a later close."""
    boson.outcomes = [(201, ACTIVE), ended]
    avatar = _avatar()
    await _start(avatar, avatar_host)

    await avatar.aclose()

    assert boson.methods() == ["POST", "DELETE"]
    assert avatar.session_id == "avatar-session-1"


async def test_close_during_start_deletes_the_session_being_created(
    avatar_host: AvatarHost, boson: Boson
) -> None:
    """Closing while Boson creates the session fails start and deletes the session once."""
    boson.outcomes = [(201, ACTIVE), (204, None)]
    boson.holds["POST"] = asyncio.Event()
    avatar = _avatar()
    start = asyncio.create_task(_start(avatar, avatar_host))
    await until(lambda: boson.calls, "Boson receives the create")

    # answered once close has begun
    asyncio.get_running_loop().call_soon(boson.holds["POST"].set)
    await avatar.aclose()

    with pytest.raises(RuntimeError, match="closed while start"):
        await start
    assert boson.methods() == ["POST", "DELETE"]
    assert avatar.session_id is None


@pytest.mark.parametrize("cancels", [1, 2])
async def test_cancelled_start_deletes_the_session_being_created(
    avatar_host: AvatarHost, boson: Boson, cancels: int
) -> None:
    """Cancelling start, even repeatedly, while Boson creates the session still deletes it."""
    boson.outcomes = [(201, ACTIVE), (204, None)]
    boson.holds["POST"] = asyncio.Event()
    avatar = _avatar()
    start = asyncio.create_task(_start(avatar, avatar_host))
    await until(lambda: boson.calls, "Boson receives the create")

    start.cancel()
    if cancels == 2:  # again, once start is compensating
        asyncio.get_running_loop().call_soon(start.cancel)
    asyncio.get_running_loop().call_soon(boson.holds["POST"].set)
    with pytest.raises(asyncio.CancelledError):
        await start
    await avatar.aclose()

    assert boson.methods() == ["POST", "DELETE"]
    assert avatar.session_id is None


async def test_cancelling_start_does_not_cancel_its_compensating_delete(
    avatar_host: AvatarHost, boson: Boson
) -> None:
    """A start cancelled while deleting its failed session lets that delete finish."""
    boson.outcomes = [(201, ACTIVE), (204, None)]
    boson.holds["DELETE"] = asyncio.Event()
    avatar_host.audio_tail.side_effect = RuntimeError("sink rejected")
    avatar = _avatar()
    start = asyncio.create_task(_start(avatar, avatar_host))
    await until(lambda: "DELETE" in boson.methods(), "Boson receives the delete")

    start.cancel()
    with pytest.raises(RuntimeError, match="sink rejected"):
        await start
    boson.holds["DELETE"].set()
    await avatar.aclose()

    assert boson.methods() == ["POST", "DELETE"]
    assert avatar.session_id is None


async def test_cancelling_close_does_not_cancel_its_delete(
    avatar_host: AvatarHost, boson: Boson
) -> None:
    """A close cancelled mid-delete lets the delete finish; the next close sends nothing."""
    boson.outcomes = [(201, ACTIVE), (204, None)]
    avatar = _avatar()
    await _start(avatar, avatar_host)
    boson.holds["DELETE"] = asyncio.Event()
    close = asyncio.create_task(avatar.aclose())
    await until(lambda: "DELETE" in boson.methods(), "Boson receives the delete")

    close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close
    boson.holds["DELETE"].set()
    await avatar.aclose()

    assert boson.methods() == ["POST", "DELETE"]
    assert avatar.session_id is None


async def test_voice_session_close_ends_the_provider_session(
    avatar_host: AvatarHost, boson: Boson
) -> None:
    """When the voice session closes, the Boson session is deleted without an explicit close."""
    boson.outcomes = [(201, ACTIVE), (204, None)]
    avatar = _avatar()
    await _start(avatar, avatar_host)

    avatar_host.agent.emit("close", CloseEvent(reason=CloseReason.USER_INITIATED))
    await until(lambda: avatar.session_id is None, "the Boson session is deleted")

    assert boson.methods() == ["POST", "DELETE"]
