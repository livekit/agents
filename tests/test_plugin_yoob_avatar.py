"""Yoob avatar plugin tests: offline, against a local fake of Yoob's session API (``POST /api/v1/avatar/sessions``).

The character renders on the user's device, so the plugin's job is the agent's half: announce the character in its
participant attributes and open Yoob sessions for the people in the call over an RPC.

Async tests run by pytest-asyncio (``asyncio_mode = "auto"`` in the repo's pytest config).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectOptions

pytestmark = pytest.mark.unit

_FAST = APIConnectOptions(max_retry=2, retry_interval=0.01, timeout=2.0)
_SESSION = {"session_token": "st", "download_token": "dt", "heartbeat_seconds": 15}


class _FakeApi:
    def __init__(self, statuses: list[int] | None = None, body: str | None = None) -> None:
        self.statuses = statuses or []
        self.body = body
        self.requests: list[dict[str, Any]] = []
        self.headers: list[str] = []

    async def create(self, request: web.Request) -> web.Response:
        self.headers.append(request.headers.get("Authorization", ""))
        self.requests.append(await request.json())
        status = self.statuses.pop(0) if self.statuses else 200
        if status >= 400:
            return web.json_response(
                {"error": "no credit", "code": "payment_required"}, status=status
            )
        return web.Response(
            text=self.body or json.dumps(_SESSION), status=status, content_type="application/json"
        )


async def _serve(fake: _FakeApi) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_post("/api/v1/avatar/sessions", fake.create)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"http://127.0.0.1:{port}"


def _participant(kind: int) -> MagicMock:
    participant = MagicMock()
    participant.kind = kind
    return participant


def _room() -> MagicMock:
    room = MagicMock()
    room.isconnected.return_value = True
    room.local_participant.set_attributes = AsyncMock()
    room.remote_participants = {
        "user-1": _participant(rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD),
        "other-agent": _participant(rtc.ParticipantKind.PARTICIPANT_KIND_AGENT),
    }
    return room


def _call(caller: str) -> rtc.RpcInvocationData:
    return rtc.RpcInvocationData(
        request_id="r1",
        caller_identity=caller,
        payload="",
        response_timeout=10.0,
        method="yoob.avatar.session",
    )


@pytest.fixture(autouse=True)
def no_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("livekit.plugins.yoob.avatar.get_job_context", lambda *a, **k: None)


async def _avatar(base: str, **kwargs: Any):
    from livekit.plugins.yoob import AvatarSession

    avatar = AvatarSession(api_key="yoob_test_k", api_base=base, conn_options=_FAST, **kwargs)
    avatar._http_session = aiohttp.ClientSession()
    return avatar


async def test_start_announces_the_character_and_answers_the_users_device() -> None:
    fake = _FakeApi()
    runner, base = await _serve(fake)
    try:
        avatar = await _avatar(base, character="luna-realistic", version="2026.11.15.1")
        room, agent_session = _room(), MagicMock()
        await avatar.start(agent_session, room)

        room.local_participant.set_attributes.assert_awaited_once_with(
            {"yoob.avatar.character": "luna-realistic", "yoob.avatar.version": "2026.11.15.1"}
        )
        (method, handler), _ = room.local_participant.register_rpc_method.call_args
        assert method == "yoob.avatar.session"
        # the agent's own audio is left alone: the device plays it through the character
        agent_session.output.replace_audio_tail.assert_not_called()

        answer = await handler(_call("user-1"))
        assert json.loads(answer) == _SESSION
        assert fake.headers == ["Bearer yoob_test_k"]
        assert fake.requests == [{"characters": ["luna-realistic"]}]
        await avatar.wait_for_join(timeout=0.1)

        await avatar.aclose()
        room.local_participant.unregister_rpc_method.assert_called_once_with("yoob.avatar.session")
        await avatar._http_session.close()
    finally:
        await runner.cleanup()


async def test_only_people_in_the_call_get_sessions() -> None:
    fake = _FakeApi()
    runner, base = await _serve(fake)
    try:
        avatar = await _avatar(base)
        room = _room()
        await avatar.start(MagicMock(), room)
        for caller in ("other-agent", "not-in-room"):
            with pytest.raises(rtc.RpcError):
                await avatar._on_session_request(_call(caller))
        assert fake.requests == []
        await avatar._http_session.close()
    finally:
        await runner.cleanup()


async def test_sessions_per_minute_are_capped() -> None:
    fake = _FakeApi()
    runner, base = await _serve(fake)
    try:
        avatar = await _avatar(base, max_sessions_per_minute=2)
        await avatar.start(MagicMock(), _room())
        await avatar._on_session_request(_call("user-1"))
        await avatar._on_session_request(_call("user-1"))
        with pytest.raises(rtc.RpcError) as raised:
            await avatar._on_session_request(_call("user-1"))
        assert "too many" in raised.value.message
        assert len(fake.requests) == 2
        await avatar._http_session.close()
    finally:
        await runner.cleanup()


async def test_refusal_reaches_the_device_and_is_not_retried() -> None:
    fake = _FakeApi(statuses=[402])
    runner, base = await _serve(fake)
    try:
        avatar = await _avatar(base)
        await avatar.start(MagicMock(), _room())
        with pytest.raises(rtc.RpcError) as raised:
            await avatar._on_session_request(_call("user-1"))
        assert "payment_required" in (raised.value.data or "")
        assert "yoob_test_k" not in (raised.value.data or "") + raised.value.message
        assert len(fake.requests) == 1
        with pytest.raises(TimeoutError):
            await avatar.wait_for_join(timeout=0.05)
        await avatar._http_session.close()
    finally:
        await runner.cleanup()


async def test_server_errors_are_retried() -> None:
    fake = _FakeApi(statuses=[503, 503])
    runner, base = await _serve(fake)
    try:
        avatar = await _avatar(base)
        await avatar.start(MagicMock(), _room())
        assert json.loads(await avatar._on_session_request(_call("user-1"))) == _SESSION
        assert len(fake.requests) == 3
        await avatar._http_session.close()
    finally:
        await runner.cleanup()


async def test_invalid_session_is_not_retried() -> None:
    fake = _FakeApi(body="not json")
    runner, base = await _serve(fake)
    try:
        avatar = await _avatar(base)
        await avatar.start(MagicMock(), _room())
        with pytest.raises(rtc.RpcError):
            await avatar._on_session_request(_call("user-1"))
        assert len(fake.requests) == 1
        await avatar._http_session.close()
    finally:
        await runner.cleanup()


async def test_unreachable_api_is_an_rpc_error() -> None:
    avatar = await _avatar("http://127.0.0.1:9")
    await avatar.start(MagicMock(), _room())
    with pytest.raises(rtc.RpcError):
        await avatar._on_session_request(_call("user-1"))
    await avatar._http_session.close()


async def test_setup_errors_fail_early(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.yoob import AvatarSession, YoobAvatarException

    monkeypatch.delenv("YOOB_API_KEY", raising=False)
    with pytest.raises(YoobAvatarException):
        AvatarSession()
    with pytest.raises(YoobAvatarException):
        AvatarSession(api_key="k", character="../astrid")
    room = _room()
    room.isconnected.return_value = False
    with pytest.raises(YoobAvatarException):
        await AvatarSession(api_key="k").start(MagicMock(), room)
