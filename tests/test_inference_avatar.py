"""Unit tests for livekit.agents.inference.AvatarSession.

Covers model-string parsing, credential resolution, the gateway create call
(payload/headers/idempotency), error mapping, and that the response sample rate
drives the DataStream audio sink.
"""

from __future__ import annotations

import contextlib
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIStatusError
from livekit.agents.inference import avatar as avatar_mod
from livekit.agents.inference._utils import HEADER_INFERENCE_PROVIDER
from livekit.agents.inference.avatar import AvatarSession, _parse_avatar_model
from livekit.agents.types import APIConnectOptions

pytestmark = pytest.mark.unit


def _make_avatar(**kwargs: Any) -> AvatarSession:
    defaults: dict[str, Any] = {
        "model": "lemonslice",
        "api_key": "test-key",
        "api_secret": "test-secret",
        "base_url": "https://example.livekit.cloud/v1",
    }
    defaults.update(kwargs)
    return AvatarSession(**defaults)


class TestParseAvatarModel:
    @pytest.mark.parametrize(
        "model,provider,avatar_id",
        [
            ("lemonslice", "lemonslice", None),
            ("lemonslice/agent_abc", "lemonslice", "agent_abc"),
            ("lemonslice/", "lemonslice", None),
            ("bey/face_1", "bey", "face_1"),
        ],
    )
    def test_valid(self, model: str, provider: str, avatar_id: str | None) -> None:
        assert _parse_avatar_model(model) == (provider, avatar_id)

    @pytest.mark.parametrize("model", ["", "/foo", "  "])
    def test_invalid_provider_raises(self, model: str) -> None:
        with pytest.raises(ValueError):
            _parse_avatar_model(model)


class TestCredentials:
    def test_defaults_and_identity(self) -> None:
        av = _make_avatar()
        assert av.provider == "lemonslice"
        assert av.avatar_identity == "lemonslice-avatar-agent"

    def test_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LIVEKIT_INFERENCE_API_KEY", raising=False)
        monkeypatch.delenv("LIVEKIT_INFERENCE_API_SECRET", raising=False)
        monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "env-secret")
        av = AvatarSession("lemonslice", base_url="https://x/v1")
        assert av._api_key == "env-key"
        assert av._api_secret == "env-secret"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "LIVEKIT_INFERENCE_API_KEY",
            "LIVEKIT_API_KEY",
            "LIVEKIT_INFERENCE_API_SECRET",
            "LIVEKIT_API_SECRET",
        ):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ValueError):
            AvatarSession("lemonslice", base_url="https://x/v1")

    def test_custom_identity(self) -> None:
        av = _make_avatar(avatar_participant_identity="custom-id")
        assert av.avatar_identity == "custom-id"


@contextlib.asynccontextmanager
async def _gateway(handler):  # type: ignore[no-untyped-def]
    """Start an aiohttp test server exposing POST /v1/avatar/sessions."""
    app = web.Application()
    app.router.add_post("/v1/avatar/sessions", handler)
    server = TestServer(app)
    await server.start_server()
    session = aiohttp.ClientSession()
    try:
        base_url = str(server.make_url("/v1"))
        yield base_url, session
    finally:
        await session.close()
        await server.close()


async def _call_create(av: AvatarSession) -> dict[str, Any]:
    return await av._create_session(
        room_name="my-room",
        room_sid="RM_123",
        livekit_url="wss://example.livekit.cloud",
        worker_token="worker-token",
        agent_identity="agent-worker-1",
    )


async def test_create_session_success() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = await request.json()
        return web.json_response(
            {"session_id": "AVS_1", "provider_session_id": "ls_abc", "sample_rate": 16000}
        )

    async with _gateway(handler) as (base_url, session):
        av = _make_avatar(
            model="lemonslice",
            base_url=base_url,
            http_session=session,
            image_url="https://example.com/face.png",
            prompt="be expressive",
            idle_timeout=300,
        )
        resp = await _call_create(av)

    assert resp["session_id"] == "AVS_1"
    assert resp["provider_session_id"] == "ls_abc"

    body = captured["body"]
    assert body["provider"] == "lemonslice"
    assert body["livekit_token"] == "worker-token"
    assert body["avatar_identity"] == "lemonslice-avatar-agent"
    assert body["agent_identity"] == "agent-worker-1"
    assert body["room_sid"] == "RM_123"
    assert body["image_url"] == "https://example.com/face.png"
    assert body["prompt"] == "be expressive"
    assert body["idle_timeout_s"] == 300

    headers = captured["headers"]
    assert headers["Authorization"].startswith("Bearer ")
    assert headers[HEADER_INFERENCE_PROVIDER] == "lemonslice"
    assert headers["Idempotency-Key"]


async def test_avatar_id_from_model_string() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.Response:
        captured["body"] = await request.json()
        return web.json_response({"session_id": "AVS_1"})

    async with _gateway(handler) as (base_url, session):
        av = _make_avatar(model="lemonslice/agent_abc", base_url=base_url, http_session=session)
        await _call_create(av)

    assert captured["body"]["avatar_id"] == "agent_abc"
    assert "image_url" not in captured["body"]


async def test_idempotency_key_stable_across_retries() -> None:
    keys: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        keys.append(request.headers["Idempotency-Key"])
        if len(keys) < 3:
            return web.json_response({"error": "unavailable"}, status=503)
        return web.json_response({"session_id": "AVS_1"})

    async with _gateway(handler) as (base_url, session):
        av = _make_avatar(
            base_url=base_url,
            http_session=session,
            conn_options=APIConnectOptions(max_retry=3, retry_interval=0.0, timeout=5.0),
        )
        resp = await _call_create(av)

    assert resp["session_id"] == "AVS_1"
    assert len(keys) == 3
    assert len(set(keys)) == 1, "idempotency key must be stable across retries"


async def test_non_retryable_error_not_retried() -> None:
    calls = {"n": 0}

    async def handler(request: web.Request) -> web.Response:
        calls["n"] += 1
        return web.json_response({"error": "not enabled"}, status=403)

    async with _gateway(handler) as (base_url, session):
        av = _make_avatar(
            base_url=base_url,
            http_session=session,
            conn_options=APIConnectOptions(max_retry=3, retry_interval=0.0, timeout=5.0),
        )
        with pytest.raises(APIStatusError) as exc:
            await _call_create(av)

    assert exc.value.status_code == 403
    assert calls["n"] == 1, "a 403 is non-retryable and must not be retried"


async def test_server_error_retried_then_raised() -> None:
    calls = {"n": 0}

    async def handler(request: web.Request) -> web.Response:
        calls["n"] += 1
        return web.json_response({"error": "boom"}, status=502)

    async with _gateway(handler) as (base_url, session):
        av = _make_avatar(
            base_url=base_url,
            http_session=session,
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=5.0),
        )
        with pytest.raises(APIStatusError) as exc:
            await _call_create(av)

    assert exc.value.status_code == 502
    assert calls["n"] == 3, "502 is retryable: initial try + 2 retries"


class _FakeSink:
    def __init__(self, **kwargs: Any) -> None:
        self.sample_rate = kwargs.get("sample_rate")


class _FakeOutput:
    def __init__(self) -> None:
        self.sink: Any = None

    def replace_audio_tail(self, sink: Any) -> None:
        self.sink = sink


class _FakeAgentSession:
    def __init__(self) -> None:
        self.output = _FakeOutput()

    def on(self, *_a: Any, **_k: Any) -> None:
        pass

    def emit(self, *_a: Any, **_k: Any) -> None:
        pass


class _FakeRoom:
    name = "my-room"

    def isconnected(self) -> bool:
        # False so the base class registers a connection callback instead of
        # launching the participant-join wait task (which would hang offline).
        return False

    def on(self, *_a: Any, **_k: Any) -> None:
        pass

    def off(self, *_a: Any, **_k: Any) -> None:
        pass


class _FakeJobRoom:
    sid = "RM_123"


class _FakeJob:
    room = _FakeJobRoom()


class _FakeJobCtx:
    local_participant_identity = "agent-worker-1"
    job = _FakeJob()

    def add_shutdown_callback(self, *_a: Any, **_k: Any) -> None:
        pass


async def test_start_uses_response_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DataStream sink is created with the gateway-reported sample rate,
    not a hardcoded per-provider constant."""

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {"session_id": "AVS_1", "provider_session_id": "ls_1", "sample_rate": 24000}
        )

    monkeypatch.setattr(avatar_mod, "get_job_context", lambda *a, **k: _FakeJobCtx())
    monkeypatch.setattr(avatar_mod, "DataStreamAudioOutput", _FakeSink)

    async with _gateway(handler) as (base_url, session):
        av = _make_avatar(
            base_url=base_url,
            http_session=session,
            image_url="https://example.com/face.png",
        )
        agent_session = _FakeAgentSession()
        await av.start(
            agent_session,  # type: ignore[arg-type]
            _FakeRoom(),  # type: ignore[arg-type]
            livekit_url="wss://example.livekit.cloud",
            livekit_api_key="devkey",
            livekit_api_secret="devsecret",
        )

    assert av.session_id == "AVS_1"
    assert av.provider_session_id == "ls_1"
    assert isinstance(agent_session.output.sink, _FakeSink)
    assert agent_session.output.sink.sample_rate == 24000


async def test_aclose_terminates_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """aclose() calls the gateway terminate endpoint so the provider session
    stops billing instead of lingering until idle timeout."""
    terminate_bodies: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.Response:
        terminate_bodies.append(await request.json())
        return web.json_response({"terminated": True})

    app = web.Application()
    app.router.add_post("/v1/avatar/sessions/terminate", handler)
    server = TestServer(app)
    await server.start_server()
    session = aiohttp.ClientSession()
    try:
        av = _make_avatar(base_url=str(server.make_url("/v1")), http_session=session)
        av._provider_session_id = "ls_abc"
        # No room/agent were started; base aclose() no-ops on cleanup.
        await av.aclose()
    finally:
        await session.close()
        await server.close()

    assert len(terminate_bodies) == 1
    assert terminate_bodies[0] == {"provider": "lemonslice", "provider_session_id": "ls_abc"}
    # provider session id cleared so a second aclose() does not re-terminate.
    assert av.provider_session_id is None


async def test_aclose_without_session_is_noop() -> None:
    av = _make_avatar()
    # No provider session created; aclose() must not raise.
    await av.aclose()
