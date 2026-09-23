"""Unit tests for the Atmee avatar plugin.

Covers plugin registration, the ``AtmeeAPI`` HTTP client (avatar creation from a
portrait, avatar-session start / get / end, error mapping, retries) and the
``AvatarSession`` lifecycle (token minting, publish-on-behalf, audio routing,
shutdown). The Atmee API is a local scripted aiohttp server and the LiveKit room
is faked, so this runs offline.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import jwt
import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import api as lk_api
from livekit.agents import APIConnectOptions, Plugin
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.plugins import atmee
from livekit.plugins.atmee import (
    SUPPORTED_AVATAR_VERSIONS,
    AtmeeAPI,
    AtmeeAvatarNotReadyError,
    AtmeeException,
    AtmeeNoCapacityError,
)

pytestmark = [pytest.mark.unit, pytest.mark.plugin("atmee")]


def test_plugin_registered() -> None:
    titles = [p.title for p in Plugin.registered_plugins]
    assert "livekit.plugins.atmee" in titles


# --- fixtures: a scripted Atmee API and a minimal fake room -------------------

API_KEY = "sk_atmee_test_key"
AVATAR_ID = "0f0e0d0c-0b0a-4908-8706-050403020100"
SESSION_ID = "11111111-2222-4333-8444-555555555555"
LIVEKIT_SECRET = "the-developers-secret-the-developers-secret"


@dataclass
class Recorded:
    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    json: Any = None
    form: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scripted:
    status: int
    payload: Any = None
    body: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


class FakeAtmee:
    """A scripted stand-in for the Atmee API: answers each (method, path) from
    a queue of scripted responses and records every request it saw."""

    def __init__(self) -> None:
        self.requests: list[Recorded] = []
        self.scripts: dict[tuple[str, str], deque[Scripted]] = defaultdict(deque)
        self.url = ""

    def script(
        self,
        method: str,
        path: str,
        status: int,
        payload: Any = None,
        *,
        body: str | None = None,
        headers: dict[str, str] | None = None,
        times: int = 1,
    ) -> None:
        for _ in range(times):
            self.scripts[(method, path)].append(Scripted(status, payload, body, headers or {}))

    def calls(self, method: str, path: str) -> list[Recorded]:
        return [r for r in self.requests if r.method == method and r.path == path]

    async def handle(self, request: web.Request) -> web.StreamResponse:
        rec = Recorded(
            method=request.method,
            path=request.path,
            query=dict(request.query),
            headers=dict(request.headers.items()),
        )
        ctype = request.content_type
        if ctype == "application/json":
            rec.json = await request.json()
        elif ctype.startswith("multipart/"):
            reader = await request.multipart()
            async for part in reader:
                name = part.name or ""
                if part.filename:
                    rec.form[name] = {
                        "filename": part.filename,
                        "content_type": part.headers.get("Content-Type"),
                        "data": await part.read(decode=False),
                    }
                else:
                    rec.form[name] = (await part.read(decode=False)).decode()
        self.requests.append(rec)

        queue = self.scripts.get((request.method, request.path))
        if not queue:
            return web.json_response({"error": "unscripted", "message": request.path}, status=404)
        s = queue.popleft()
        if s.payload is not None:
            return web.Response(
                status=s.status,
                text=json.dumps(s.payload),
                content_type="application/json",
                headers=s.headers,
            )
        return web.Response(status=s.status, text=s.body or "", headers=s.headers)


@pytest_asyncio.fixture
async def fake_atmee(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeAtmee]:
    fa = FakeAtmee()
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", fa.handle)
    server = TestServer(app)
    await server.start_server()
    fa.url = str(server.make_url("")).rstrip("/")
    monkeypatch.setenv("ATMEE_API_URL", fa.url)
    try:
        yield fa
    finally:
        await server.close()


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATMEE_API_KEY", API_KEY)
    monkeypatch.setenv("ATMEE_API_URL", "https://api.unreachable.test")
    monkeypatch.setenv("LIVEKIT_URL", "wss://dev.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "APIdevkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", LIVEKIT_SECRET)


@pytest_asyncio.fixture
async def http_session() -> AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession() as session:
        yield session


class FakeParticipant:
    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.name = identity


class FakeRoom:
    """Just enough of rtc.Room for AvatarSession.start(): a name, a local
    participant, event registration, and a disconnected state so nothing
    tries to talk to a server."""

    def __init__(self, name: str = "dev-room-42", local_identity: str = "my-agent") -> None:
        self.name = name
        self.local_participant = FakeParticipant(local_identity)
        self.handlers: dict[str, list[Any]] = {}

    def isconnected(self) -> bool:
        return False

    def on(self, event: str, handler: Any = None) -> Any:
        if handler is None:  # decorator form

            def _register(h: Any) -> Any:
                self.handlers.setdefault(event, []).append(h)
                return h

            return _register
        self.handlers.setdefault(event, []).append(handler)
        return handler

    def off(self, event: str, handler: Any) -> None:
        if handler in self.handlers.get(event, []):
            self.handlers[event].remove(handler)

    def emit(self, event: str, *args: Any) -> None:
        for h in list(self.handlers.get(event, [])):
            h(*args)


class FakeOutput:
    def __init__(self) -> None:
        self.audio_tails: list[Any] = []

    def replace_audio_tail(self, out: Any) -> None:
        self.audio_tails.append(out)


class FakeAgentSession:
    def __init__(self) -> None:
        self.output = FakeOutput()
        self.handlers: dict[str, list[Any]] = {}

    def on(self, event: str, handler: Any) -> Any:
        self.handlers.setdefault(event, []).append(handler)
        return handler

    def off(self, event: str, handler: Any) -> None:
        if handler in self.handlers.get(event, []):
            self.handlers[event].remove(handler)

    def emit(self, *args: Any) -> None:
        pass


async def settle() -> None:
    """Let fire-and-forget tasks run."""
    for _ in range(10):
        await asyncio.sleep(0)


# --- AtmeeAPI -----------------------------------------------------------------

SESSIONS_PATH = f"/v1/avatars/{AVATAR_ID}/avatar_sessions"
FAST = APIConnectOptions(max_retry=3, retry_interval=0.0, timeout=5.0)


def _start_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "sessionId": SESSION_ID,
        "status": "initializing",
        "avatarParticipantIdentity": "atmee-avatar-agent",
        "agentIdentity": "my-agent",
        "roomName": "dev-room-42",
        "maxDurationSeconds": 3600,
        "billingMode": "metered",
    }
    body.update(overrides)
    return body


async def test_create_avatar_session_sends_key_and_payload(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script("POST", SESSIONS_PATH, 202, _start_body())
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    info = await api.create_avatar_session(
        AVATAR_ID,
        livekit_url="wss://dev.livekit.cloud",
        livekit_token="eyJ.tok",
        agent_identity="my-agent",
        max_duration_seconds=1800,
        metadata={"tenant": "t1"},
    )
    call = fake_atmee.calls("POST", SESSIONS_PATH)[0]
    assert call.headers["X-Api-Key"] == API_KEY
    assert call.json == {
        "livekitUrl": "wss://dev.livekit.cloud",
        "livekitToken": "eyJ.tok",
        "agentIdentity": "my-agent",
        "maxDurationSeconds": 1800,
        "metadata": {"tenant": "t1"},
    }
    assert call.query == {}
    assert info.session_id == SESSION_ID
    assert info.status == "initializing"
    assert info.agent_identity == "my-agent"
    assert info.room_name == "dev-room-42"
    assert info.max_duration_seconds == 3600
    assert info.billing_mode == "metered"
    assert info.avatar_version == "v1"
    assert api.api_url == fake_atmee.url


def test_only_v1_avatars_are_supported() -> None:
    assert SUPPORTED_AVATAR_VERSIONS == frozenset({"v1"})


async def test_create_avatar_session_explicit_v1_sends_no_version_field(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    # The API contract has no version field: the version is plugin-side only
    # and the request body is byte-for-byte what it was before.
    fake_atmee.script("POST", SESSIONS_PATH, 202, _start_body())
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    info = await api.create_avatar_session(
        AVATAR_ID, livekit_url="wss://x", livekit_token="t", avatar_version="v1"
    )
    assert info.avatar_version == "v1"
    assert fake_atmee.calls("POST", SESSIONS_PATH)[0].json == {
        "livekitUrl": "wss://x",
        "livekitToken": "t",
    }


async def test_create_avatar_session_rejects_v2_before_any_request(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script("POST", SESSIONS_PATH, 202, _start_body())
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    with pytest.raises(ValueError, match="avatar_version 'v2' is not supported") as exc:
        await api.create_avatar_session(
            AVATAR_ID,
            livekit_url="wss://x",
            livekit_token="t",
            avatar_version="v2",
        )
    assert "only 'v1'" in str(exc.value)
    assert fake_atmee.requests == []


async def test_create_avatar_session_wait_for_avatar_joined(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script("POST", SESSIONS_PATH, 202, _start_body(status="avatar_joined"))
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    info = await api.create_avatar_session(
        AVATAR_ID, livekit_url="wss://x", livekit_token="t", wait_for="avatar_joined"
    )
    assert info.status == "avatar_joined"
    assert fake_atmee.calls("POST", SESSIONS_PATH)[0].query == {"waitFor": "avatar_joined"}


async def test_no_capacity_is_typed_and_never_retried(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script(
        "POST",
        SESSIONS_PATH,
        503,
        {"error": "no_capacity", "message": "no rendering capacity right now"},
        headers={"Retry-After": "5"},
        times=3,
    )
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    with pytest.raises(AtmeeNoCapacityError) as exc:
        await api.create_avatar_session(AVATAR_ID, livekit_url="wss://x", livekit_token="t")
    assert len(fake_atmee.calls("POST", SESSIONS_PATH)) == 1
    assert exc.value.retry_after == 5.0
    assert exc.value.code == "no_capacity"
    assert exc.value.status_code == 503


async def test_4xx_is_final_with_code(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script(
        "POST",
        SESSIONS_PATH,
        400,
        {"error": "missing_publish_on_behalf", "message": "token lacks the attribute"},
        times=3,
    )
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    with pytest.raises(AtmeeException) as exc:
        await api.create_avatar_session(AVATAR_ID, livekit_url="wss://x", livekit_token="t")
    assert len(fake_atmee.calls("POST", SESSIONS_PATH)) == 1
    assert exc.value.code == "missing_publish_on_behalf"
    assert exc.value.status_code == 400
    assert "token lacks the attribute" in str(exc.value)


async def test_not_renderable_is_typed(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script(
        "POST", SESSIONS_PATH, 409, {"error": "avatar_not_renderable", "message": "no portrait"}
    )
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    with pytest.raises(AtmeeAvatarNotReadyError):
        await api.create_avatar_session(AVATAR_ID, livekit_url="wss://x", livekit_token="t")


async def test_session_create_is_never_retried(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    # The POST bills a session once the server accepts it; a retry after a
    # timeout or 5xx could open a second render, so it is final either way.
    fake_atmee.script(
        "POST", SESSIONS_PATH, 502, {"error": "avatar_start_failed", "message": "no face"}
    )
    fake_atmee.script("POST", SESSIONS_PATH, 202, _start_body())
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    with pytest.raises(AtmeeException) as exc:
        await api.create_avatar_session(AVATAR_ID, livekit_url="wss://x", livekit_token="t")
    assert exc.value.status_code == 502 and exc.value.code == "avatar_start_failed"
    assert len(fake_atmee.calls("POST", SESSIONS_PATH)) == 1


async def test_idempotent_calls_retry_5xx_then_raise(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    get_path = f"/v1/avatar_sessions/{SESSION_ID}"
    fake_atmee.script("GET", get_path, 502, body="bad gateway")
    fake_atmee.script("GET", get_path, 500, body="oops")
    fake_atmee.script("GET", get_path, 200, {"sessionId": SESSION_ID, "status": "active"})
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    got = await api.get_avatar_session(SESSION_ID)
    assert len(fake_atmee.calls("GET", get_path)) == 3
    assert got["status"] == "active"

    # max_retry=3 means three retries after the first attempt: four calls
    fake_atmee.script("GET", get_path, 500, body="down", times=4)
    with pytest.raises(AtmeeException) as exc:
        await api.get_avatar_session(SESSION_ID)
    assert exc.value.status_code == 500
    assert len(fake_atmee.calls("GET", get_path)) == 7


async def test_end_and_get_avatar_session(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    end_path = f"/v1/avatar_sessions/{SESSION_ID}/end"
    get_path = f"/v1/avatar_sessions/{SESSION_ID}"
    fake_atmee.script(
        "POST",
        end_path,
        200,
        {"sessionId": SESSION_ID, "status": "completed", "alreadyEnded": True},
    )
    fake_atmee.script("GET", get_path, 200, {"sessionId": SESSION_ID, "status": "completed"})
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    ended = await api.end_avatar_session(SESSION_ID)
    got = await api.get_avatar_session(SESSION_ID)
    assert fake_atmee.calls("POST", end_path)[0].headers["X-Api-Key"] == API_KEY
    assert ended["alreadyEnded"] is True
    assert got["status"] == "completed"


async def test_create_avatar_from_file_is_multipart(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession, tmp_path: Path
) -> None:
    portrait = tmp_path / "val.png"
    portrait.write_bytes(b"\x89PNG\r\n\x1a\nfakepng")
    fake_atmee.script(
        "POST",
        "/v1/avatars",
        201,
        {"avatarId": AVATAR_ID, "kind": "render_only", "status": "ready", "statusUrl": "x"},
    )
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    avatar_id = await api.create_avatar("Val", portrait, description="from a test")
    assert avatar_id == AVATAR_ID
    call = fake_atmee.calls("POST", "/v1/avatars")[0]
    assert call.form["name"] == "Val"
    assert call.form["description"] == "from a test"
    assert call.form["file"]["filename"] == "val.png"
    assert call.form["file"]["content_type"] == "image/png"
    assert call.form["file"]["data"] == b"\x89PNG\r\n\x1a\nfakepng"
    assert set(call.form) == {"name", "description", "file"}  # no version field


async def test_create_avatar_reports_v1_by_default(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script(
        "POST",
        "/v1/avatars",
        201,
        {"avatarId": AVATAR_ID, "kind": "render_only", "status": "ready"},
    )
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    info = await api.create_avatar_info("Val", b"jpegbytes")
    # `kind` comes from the API (voice/persona or not); `version` is the avatar
    # generation, filled in by the plugin because the API reports none.
    assert info.kind == "render_only"
    assert info.version == "v1"
    assert "version" not in info.raw
    assert set(fake_atmee.calls("POST", "/v1/avatars")[0].form) == {"name", "file"}


async def test_create_avatar_explicit_v1_keeps_manifest_unchanged(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script("POST", "/v1/avatars", 201, {"avatarId": AVATAR_ID, "status": "ready"})
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    info = await api.create_avatar_info("Val", "https://cdn.example/val.jpg", avatar_version="v1")
    assert info.version == "v1"
    assert fake_atmee.calls("POST", "/v1/avatars")[0].json == {
        "schemaVersion": 1,
        "name": "Val",
        "assets": {"image": {"url": "https://cdn.example/val.jpg"}},
    }


async def test_create_avatar_rejects_v2_before_any_request(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession, tmp_path: Path
) -> None:
    fake_atmee.script("POST", "/v1/avatars", 201, {"avatarId": AVATAR_ID, "status": "ready"})
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    with pytest.raises(ValueError, match="avatar_version 'v2' is not supported"):
        await api.create_avatar("Val", "https://cdn.example/val.jpg", avatar_version="v2")
    with pytest.raises(ValueError, match="only 'v1'"):
        await api.create_avatar_info("Val", b"jpegbytes", avatar_version="v2")
    missing = tmp_path / "does-not-exist.png"  # never read: the version check comes first
    with pytest.raises(ValueError):
        await api.create_avatar("Val", missing, avatar_version="v2")
    assert fake_atmee.requests == []


async def test_create_avatar_from_bytes(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script("POST", "/v1/avatars", 201, {"avatarId": AVATAR_ID, "status": "ready"})
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    await api.create_avatar("Val", b"webpbytes", content_type="image/webp")
    part = fake_atmee.calls("POST", "/v1/avatars")[0].form["file"]
    assert part["filename"] == "portrait.webp" and part["content_type"] == "image/webp"


async def test_create_avatar_from_url_is_json_manifest(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script(
        "POST",
        "/v1/avatars",
        201,
        {"avatarId": AVATAR_ID, "kind": "render_only", "status": "ready", "statusUrl": "x"},
    )
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    info = await api.create_avatar_info("Val", "https://cdn.example/val.jpg")
    assert fake_atmee.calls("POST", "/v1/avatars")[0].json == {
        "schemaVersion": 1,
        "name": "Val",
        "assets": {"image": {"url": "https://cdn.example/val.jpg"}},
    }
    assert info.ready and info.kind == "render_only"


def test_plaintext_api_url_is_refused() -> None:
    with pytest.raises(AtmeeException, match="https"):
        AtmeeAPI(api_url="http://api.example.com")
    AtmeeAPI(api_url="http://127.0.0.1:8080")  # loopback is fine for local development
    AtmeeAPI(api_url="https://api.example.com")


async def test_wait_until_ready_honours_its_timeout(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    path = f"/v1/avatars/{AVATAR_ID}"
    fake_atmee.script("GET", path, 200, {"avatarId": AVATAR_ID, "status": "building"}, times=5)
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    loop = asyncio.get_running_loop()
    began = loop.time()
    with pytest.raises(AtmeeException) as exc:
        await api.wait_until_ready(AVATAR_ID, timeout=0.2, poll_interval=30)
    assert exc.value.code == "timeout"
    assert loop.time() - began < 2  # the 30 s poll interval was capped by the deadline


async def test_wait_until_ready_polls(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    path = f"/v1/avatars/{AVATAR_ID}"
    fake_atmee.script("GET", path, 200, {"avatarId": AVATAR_ID, "status": "building"})
    fake_atmee.script(
        "GET", path, 200, {"avatarId": AVATAR_ID, "status": "ready", "kind": "conversational"}
    )
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    info = await api.wait_until_ready(AVATAR_ID, timeout=10, poll_interval=0)
    assert info.ready and info.kind == "conversational"
    assert len(fake_atmee.calls("GET", path)) == 2


def test_missing_api_key_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATMEE_API_KEY")
    with pytest.raises(AtmeeException):
        AtmeeAPI()


async def test_owns_session_outside_a_job(fake_atmee: FakeAtmee) -> None:
    fake_atmee.script(
        "GET", f"/v1/avatars/{AVATAR_ID}", 200, {"avatarId": AVATAR_ID, "status": "ready"}
    )
    api = AtmeeAPI(conn_options=FAST)
    async with api:
        info = await api.get_avatar(AVATAR_ID)
    assert info.ready
    assert api._session is None  # closed and dropped by aclose()


def test_error_str_carries_code_and_status() -> None:
    e = AtmeeException("nope", status_code=402, code="insufficient_credits")
    assert str(e) == "nope [insufficient_credits] (HTTP 402)"


async def test_malformed_success_is_a_typed_error(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    get_path = f"/v1/avatar_sessions/{SESSION_ID}"
    fake_atmee.script("GET", get_path, 200, body="<html>not json</html>")
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    with pytest.raises(AtmeeException) as exc:
        await api.get_avatar_session(SESSION_ID)
    assert exc.value.code == "invalid_response"
    assert len(fake_atmee.calls("GET", get_path)) == 1  # a 2xx is never retried


async def test_wait_until_ready_deadline_bounds_retries(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    path = f"/v1/avatars/{AVATAR_ID}"
    fake_atmee.script("GET", path, 502, body="bad gateway", times=20)
    slow_retries = APIConnectOptions(max_retry=5, retry_interval=30.0, timeout=5.0)
    api = AtmeeAPI(session=http_session, conn_options=slow_retries)
    loop = asyncio.get_running_loop()
    began = loop.time()
    with pytest.raises(AtmeeException) as exc:
        await api.wait_until_ready(AVATAR_ID, timeout=0.3, poll_interval=30)
    # deadline expiry is a timeout, not the last 5xx seen before it
    assert exc.value.code == "timeout"
    # the 30 s retry pauses were cut to the remaining budget
    assert loop.time() - began < 2


async def test_plain_503_is_retried_like_any_5xx(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    get_path = f"/v1/avatar_sessions/{SESSION_ID}"
    fake_atmee.script("GET", get_path, 503, body="Service Unavailable")  # e.g. from a proxy
    fake_atmee.script("GET", get_path, 200, {"sessionId": SESSION_ID, "status": "active"})
    api = AtmeeAPI(session=http_session, conn_options=FAST)
    got = await api.get_avatar_session(SESSION_ID)
    assert got["status"] == "active"
    assert len(fake_atmee.calls("GET", get_path)) == 2


# --- AvatarSession --------------------------------------------------------------

END_PATH = f"/v1/avatar_sessions/{SESSION_ID}/end"
SESSION_FAST = APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=5.0)


def _session_start_body() -> dict[str, Any]:
    return {
        "sessionId": SESSION_ID,
        "status": "initializing",
        "avatarParticipantIdentity": "atmee-avatar-agent",
        "agentIdentity": "my-agent",
        "roomName": "dev-room-42",
        "maxDurationSeconds": 3600,
        "billingMode": "metered",
    }


async def test_start_mints_token_posts_session_and_routes_audio(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script("POST", SESSIONS_PATH, 202, _session_start_body())
    fake_atmee.script("POST", END_PATH, 200, {"sessionId": SESSION_ID, "status": "completed"})
    room = FakeRoom()
    agent_session = FakeAgentSession()
    avatar = atmee.AvatarSession(
        avatar_id=AVATAR_ID,
        conn_options=SESSION_FAST,
        http_session=http_session,
        metadata={"k": "v"},
    )
    assert avatar.provider == "atmee"
    assert avatar.avatar_identity == "atmee-avatar-agent"

    await avatar.start(agent_session, room)  # type: ignore[arg-type]

    body = fake_atmee.calls("POST", SESSIONS_PATH)[0].json
    assert body["livekitUrl"] == "wss://dev.livekit.cloud"
    assert body["agentIdentity"] == "my-agent"
    assert body["maxDurationSeconds"] == 3600
    assert body["metadata"] == {"k": "v"}

    # The token is minted with the developer's own LiveKit credentials, for the
    # avatar participant, granting exactly this room, publishing on behalf of
    # the local agent.
    claims = lk_api.TokenVerifier("APIdevkey", LIVEKIT_SECRET).verify(body["livekitToken"])
    assert claims.identity == "atmee-avatar-agent"
    # TokenVerifier does not surface `kind`; read the raw payload for it.
    assert jwt.decode(body["livekitToken"], options={"verify_signature": False})["kind"] == "agent"
    assert claims.video is not None and claims.video.room == "dev-room-42"
    assert claims.video.room_join is True
    assert claims.attributes == {"lk.publish_on_behalf": "my-agent"}

    assert avatar.session_id == SESSION_ID
    assert avatar.session_info is not None and avatar.session_info.billing_mode == "metered"

    tails = agent_session.output.audio_tails
    assert len(tails) == 1 and isinstance(tails[0], DataStreamAudioOutput)
    assert tails[0]._destination_identity == "atmee-avatar-agent"
    assert tails[0].sample_rate == atmee.avatar.SAMPLE_RATE == 16000

    await avatar.aclose()
    await avatar.aclose()  # idempotent locally too
    assert len(fake_atmee.calls("POST", END_PATH)) == 1


async def test_avatar_participant_leaving_emits_and_ends_once(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script("POST", SESSIONS_PATH, 202, _session_start_body())
    fake_atmee.script("POST", END_PATH, 200, {"sessionId": SESSION_ID, "status": "completed"})
    room = FakeRoom()
    avatar = atmee.AvatarSession(
        avatar_id=AVATAR_ID, conn_options=SESSION_FAST, http_session=http_session
    )
    seen: list[str] = []
    avatar.on("avatar_disconnected", lambda p: seen.append(p.identity))

    await avatar.start(FakeAgentSession(), room)  # type: ignore[arg-type]

    room.emit("participant_disconnected", FakeParticipant("someone-else"))
    await settle()
    assert seen == []
    room.emit("participant_disconnected", FakeParticipant("atmee-avatar-agent"))
    await settle()
    room.emit("participant_disconnected", FakeParticipant("atmee-avatar-agent"))
    await settle()
    await avatar.aclose()
    assert len(fake_atmee.calls("POST", END_PATH)) == 1
    assert seen == ["atmee-avatar-agent"]


async def test_end_failure_never_breaks_close(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script("POST", SESSIONS_PATH, 202, _session_start_body())
    fake_atmee.script("POST", END_PATH, 500, body="down")
    avatar = atmee.AvatarSession(
        avatar_id=AVATAR_ID, conn_options=SESSION_FAST, http_session=http_session
    )
    await avatar.start(FakeAgentSession(), FakeRoom())  # type: ignore[arg-type]
    fake_atmee.script("POST", END_PATH, 500, body="down")  # max_retry=1: two attempts
    await avatar.aclose()  # must not raise
    assert len(fake_atmee.calls("POST", END_PATH)) == 2

    # the failed end was not recorded as done, so a later aclose retries it
    fake_atmee.script("POST", END_PATH, 200, {"sessionId": SESSION_ID, "alreadyEnded": False})
    await avatar.aclose()
    assert len(fake_atmee.calls("POST", END_PATH)) == 3
    await avatar.aclose()  # confirmed ended: no further request
    assert len(fake_atmee.calls("POST", END_PATH)) == 3


async def test_end_4xx_is_final(fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession) -> None:
    fake_atmee.script("POST", SESSIONS_PATH, 202, _session_start_body())
    fake_atmee.script("POST", END_PATH, 404, {"error": "not_found", "message": "gone"})
    avatar = atmee.AvatarSession(
        avatar_id=AVATAR_ID, conn_options=SESSION_FAST, http_session=http_session
    )
    await avatar.start(FakeAgentSession(), FakeRoom())  # type: ignore[arg-type]
    await avatar.aclose()
    await avatar.aclose()  # unknown session: nothing left to end, no retry
    assert len(fake_atmee.calls("POST", END_PATH)) == 1


async def test_start_is_one_shot(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script("POST", SESSIONS_PATH, 202, _session_start_body())
    avatar = atmee.AvatarSession(
        avatar_id=AVATAR_ID, conn_options=SESSION_FAST, http_session=http_session
    )
    await avatar.start(FakeAgentSession(), FakeRoom())  # type: ignore[arg-type]
    with pytest.raises(AtmeeException, match="already called"):
        await avatar.start(FakeAgentSession(), FakeRoom())  # type: ignore[arg-type]
    # the second call never reached the API, so no second billed render
    assert len(fake_atmee.calls("POST", SESSIONS_PATH)) == 1


async def test_start_without_session_id_is_rejected(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script("POST", SESSIONS_PATH, 202, {"status": "initializing"})
    avatar = atmee.AvatarSession(
        avatar_id=AVATAR_ID, conn_options=SESSION_FAST, http_session=http_session
    )
    with pytest.raises(AtmeeException) as exc:
        await avatar.start(FakeAgentSession(), FakeRoom())  # type: ignore[arg-type]
    assert exc.value.code == "invalid_response"
    assert avatar.session_id is None


async def test_construct_outside_a_job_without_http_session(fake_atmee: FakeAtmee) -> None:
    # no job context and no session passed: construction must not touch the
    # job's http context; the client creates (and aclose releases) its own
    avatar = atmee.AvatarSession(avatar_id=AVATAR_ID, conn_options=SESSION_FAST)
    fake_atmee.script("POST", SESSIONS_PATH, 202, _session_start_body())
    fake_atmee.script("POST", END_PATH, 200, {"sessionId": SESSION_ID})
    await avatar.start(FakeAgentSession(), FakeRoom())  # type: ignore[arg-type]
    await avatar.aclose()
    assert len(fake_atmee.calls("POST", END_PATH)) == 1


async def test_start_failure_is_typed(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script(
        "POST",
        SESSIONS_PATH,
        503,
        {"error": "no_capacity", "message": "busy"},
        headers={"Retry-After": "5"},
    )
    avatar = atmee.AvatarSession(
        avatar_id=AVATAR_ID, conn_options=SESSION_FAST, http_session=http_session
    )
    with pytest.raises(atmee.AtmeeNoCapacityError):
        await avatar.start(FakeAgentSession(), FakeRoom())  # type: ignore[arg-type]
    assert avatar.session_id is None
    await avatar.aclose()  # nothing to end
    assert fake_atmee.calls("POST", END_PATH) == []


async def test_requires_livekit_credentials(
    monkeypatch: pytest.MonkeyPatch, http_session: aiohttp.ClientSession
) -> None:
    monkeypatch.delenv("LIVEKIT_API_SECRET")
    avatar = atmee.AvatarSession(
        avatar_id=AVATAR_ID, conn_options=SESSION_FAST, http_session=http_session
    )
    with pytest.raises(atmee.AtmeeException):
        await avatar.start(FakeAgentSession(), FakeRoom())  # type: ignore[arg-type]
    await avatar.aclose()


def test_avatar_id_required() -> None:
    with pytest.raises(atmee.AtmeeException):
        atmee.AvatarSession(avatar_id="")


async def test_avatar_version_defaults_to_v1(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script("POST", SESSIONS_PATH, 202, _session_start_body())
    fake_atmee.script("POST", END_PATH, 200, {"sessionId": SESSION_ID, "status": "completed"})
    avatar = atmee.AvatarSession(
        avatar_id=AVATAR_ID, conn_options=SESSION_FAST, http_session=http_session
    )
    assert avatar.avatar_version == "v1"

    await avatar.start(FakeAgentSession(), FakeRoom())  # type: ignore[arg-type]

    assert avatar.session_info is not None
    assert avatar.session_info.avatar_version == "v1"
    # The API has no version field: the request body is unchanged.
    body = fake_atmee.calls("POST", SESSIONS_PATH)[0].json
    assert set(body) == {"livekitUrl", "livekitToken", "agentIdentity", "maxDurationSeconds"}
    await avatar.aclose()


async def test_explicit_v1_is_accepted(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script("POST", SESSIONS_PATH, 202, _session_start_body())
    fake_atmee.script("POST", END_PATH, 200, {"sessionId": SESSION_ID, "status": "completed"})
    avatar = atmee.AvatarSession(
        avatar_id=AVATAR_ID,
        conn_options=SESSION_FAST,
        http_session=http_session,
        avatar_version="v1",
    )
    await avatar.start(FakeAgentSession(), FakeRoom())  # type: ignore[arg-type]
    assert avatar.avatar_version == "v1"
    assert avatar.session_info is not None and avatar.session_info.avatar_version == "v1"
    assert "avatarVersion" not in fake_atmee.calls("POST", SESSIONS_PATH)[0].json
    await avatar.aclose()


def test_v2_is_rejected_at_construction(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    # v2 is the next avatar generation and not available through the plugin
    # yet: the constructor says so, before any token is minted or request sent.
    with pytest.raises(ValueError, match="avatar_version 'v2' is not supported") as exc:
        atmee.AvatarSession(
            avatar_id=AVATAR_ID,
            conn_options=SESSION_FAST,
            http_session=http_session,
            avatar_version="v2",
        )
    assert "only 'v1'" in str(exc.value)
    assert fake_atmee.requests == []


def test_unknown_version_is_rejected_at_construction(http_session: aiohttp.ClientSession) -> None:
    with pytest.raises(ValueError, match="avatar_version 'v3' is not supported"):
        atmee.AvatarSession(
            avatar_id=AVATAR_ID,
            conn_options=SESSION_FAST,
            http_session=http_session,
            avatar_version="v3",  # type: ignore[arg-type]
        )


async def test_wait_for_is_forwarded(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    body = _session_start_body()
    body["status"] = "avatar_joined"
    fake_atmee.script("POST", SESSIONS_PATH, 202, body)
    fake_atmee.script("POST", END_PATH, 200, {"sessionId": SESSION_ID, "status": "completed"})
    avatar = atmee.AvatarSession(
        avatar_id=AVATAR_ID,
        conn_options=SESSION_FAST,
        http_session=http_session,
        wait_for="avatar_joined",
        max_duration_seconds=600,
        avatar_participant_identity="val",
    )
    await avatar.start(FakeAgentSession(), FakeRoom())  # type: ignore[arg-type]
    call = fake_atmee.calls("POST", SESSIONS_PATH)[0]
    assert call.query == {"waitFor": "avatar_joined"}
    assert call.json["maxDurationSeconds"] == 600
    claims = lk_api.TokenVerifier("APIdevkey", LIVEKIT_SECRET).verify(call.json["livekitToken"])
    assert claims.identity == "val" and avatar.avatar_identity == "val"
    assert avatar.session_info is not None and avatar.session_info.status == "avatar_joined"
    await asyncio.sleep(0)
    await avatar.aclose()


async def test_agent_session_close_ends_the_render(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script("POST", SESSIONS_PATH, 202, _session_start_body())
    fake_atmee.script("POST", END_PATH, 200, {"sessionId": SESSION_ID})
    avatar = atmee.AvatarSession(
        avatar_id=AVATAR_ID, conn_options=SESSION_FAST, http_session=http_session
    )
    agent_session = FakeAgentSession()
    await avatar.start(agent_session, FakeRoom())  # type: ignore[arg-type]
    # AgentSession.aclose() without a job shutdown: the render must end too
    for handler in list(agent_session.handlers.get("close", [])):
        handler(None)
    await settle()
    assert len(fake_atmee.calls("POST", END_PATH)) == 1
    assert "close" not in agent_session.handlers or not agent_session.handlers["close"]


async def test_concurrent_aclose_never_closes_the_session_mid_end(
    fake_atmee: FakeAtmee, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_atmee.script("POST", SESSIONS_PATH, 202, _session_start_body())
    fake_atmee.script("POST", END_PATH, 500, body="down")
    fake_atmee.script("POST", END_PATH, 200, {"sessionId": SESSION_ID})
    no_retry = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=5.0)
    # no http_session passed: the client owns one, and aclose() closes it
    avatar = atmee.AvatarSession(avatar_id=AVATAR_ID, conn_options=no_retry)
    agent_session = FakeAgentSession()
    await avatar.start(agent_session, FakeRoom())  # type: ignore[arg-type]

    # In a job the base aclose() awaits the LiveKit API to remove the avatar
    # participant, and ending a render is a network round trip: model both,
    # and record how many ends are in flight whenever the HTTP client closes.
    from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession

    base_aclose = BaseAvatarSession.aclose

    async def slow_base_aclose(self: Any) -> None:
        await asyncio.sleep(0.01)
        await base_aclose(self)

    monkeypatch.setattr(BaseAvatarSession, "aclose", slow_base_aclose)
    in_flight = 0
    in_flight_at_close: list[int] = []
    real_end, real_close = avatar.api.end_avatar_session, avatar.api.aclose

    async def slow_end(session_id: str) -> dict[str, Any]:
        nonlocal in_flight
        in_flight += 1
        try:
            await asyncio.sleep(0.05)
            return await real_end(session_id)
        finally:
            in_flight -= 1

    async def recording_close() -> None:
        in_flight_at_close.append(in_flight)
        await real_close()

    monkeypatch.setattr(avatar.api, "end_avatar_session", slow_end)
    monkeypatch.setattr(avatar.api, "aclose", recording_close)

    # the agent session closes (background aclose) while the job shuts down (explicit aclose)
    for handler in list(agent_session.handlers.get("close", [])):
        handler(None)
    await avatar.aclose()
    for _ in range(100):
        await asyncio.sleep(0.01)
        if len(in_flight_at_close) >= 2:
            break

    assert in_flight_at_close and all(n == 0 for n in in_flight_at_close)
    assert avatar._ended  # the second close retried the failed end and it went through


async def test_failed_start_releases_the_base_session_hooks(
    fake_atmee: FakeAtmee, http_session: aiohttp.ClientSession
) -> None:
    fake_atmee.script(
        "POST",
        SESSIONS_PATH,
        503,
        {"error": "no_capacity", "message": "busy"},
        headers={"Retry-After": "5"},
    )
    avatar = atmee.AvatarSession(
        avatar_id=AVATAR_ID, conn_options=SESSION_FAST, http_session=http_session
    )
    agent_session, room = FakeAgentSession(), FakeRoom()
    with pytest.raises(atmee.AtmeeNoCapacityError):
        await avatar.start(agent_session, room)  # type: ignore[arg-type]
    # the listeners super().start() installed are gone again
    assert not agent_session.handlers.get("conversation_item_added")
    assert not room.handlers.get("connection_state_changed")
    # and the instance stays spent
    with pytest.raises(atmee.AtmeeException, match="already called"):
        await avatar.start(agent_session, room)  # type: ignore[arg-type]
