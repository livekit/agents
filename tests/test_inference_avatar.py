"""Unit tests for livekit.agents.inference.AvatarSession.

Covers model-string parsing, credential resolution, the gateway create call
(payload/headers/idempotency), error mapping, that the response sample rate
drives the DataStream audio sink, the terminate_token contract, double-start
guarding, and the standalone (no job context) room paths.
"""

from __future__ import annotations

import contextlib
from typing import Any

import aiohttp
import jwt
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
async def _gateway(handler, path: str = "/v1/avatar/sessions"):  # type: ignore[no-untyped-def]
    """Start an aiohttp test server exposing POST `path`."""
    app = web.Application()
    app.router.add_post(path, handler)
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
    not a hardcoded per-provider constant, and terminate_token is captured
    from the create response for later use by aclose()."""

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "session_id": "AVS_1",
                "provider_session_id": "ls_1",
                "terminate_token": "tt_1",
                "sample_rate": 24000,
            }
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
    assert av._terminate_token == "tt_1"
    assert isinstance(agent_session.output.sink, _FakeSink)
    assert agent_session.output.sink.sample_rate == 24000


async def test_start_mints_worker_token_with_expected_grants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The minted avatar worker token must carry the room-join grant and the
    lk.publish_on_behalf / lk.avatar_provider attributes the gateway and
    server-side metering depend on."""
    captured: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.Response:
        captured["body"] = await request.json()
        return web.json_response({"session_id": "AVS_1", "provider_session_id": "ls_1"})

    monkeypatch.setattr(avatar_mod, "get_job_context", lambda *a, **k: _FakeJobCtx())
    monkeypatch.setattr(avatar_mod, "DataStreamAudioOutput", _FakeSink)

    async with _gateway(handler) as (base_url, session):
        av = _make_avatar(base_url=base_url, http_session=session)
        agent_session = _FakeAgentSession()
        await av.start(
            agent_session,  # type: ignore[arg-type]
            _FakeRoom(),  # type: ignore[arg-type]
            livekit_url="wss://example.livekit.cloud",
            livekit_api_key="devkey",
            livekit_api_secret="devsecret",
        )

    assert captured["body"]["room_sid"] == "RM_123"
    assert captured["body"]["agent_identity"] == "agent-worker-1"

    claims = jwt.decode(captured["body"]["livekit_token"], options={"verify_signature": False})
    assert claims["kind"] == "agent"
    assert claims["sub"] == "lemonslice-avatar-agent"
    assert claims["video"]["roomJoin"] is True
    assert claims["video"]["room"] == "my-room"
    assert claims["attributes"] == {
        "lk.publish_on_behalf": "agent-worker-1",
        "lk.avatar_provider": "lemonslice",
    }


async def test_start_twice_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second start() must not create a second paid provider session or lose
    the first session's terminate handle."""
    calls = {"n": 0}

    async def handler(request: web.Request) -> web.Response:
        calls["n"] += 1
        return web.json_response({"session_id": "AVS_1", "provider_session_id": "ls_1"})

    monkeypatch.setattr(avatar_mod, "get_job_context", lambda *a, **k: _FakeJobCtx())
    monkeypatch.setattr(avatar_mod, "DataStreamAudioOutput", _FakeSink)

    async with _gateway(handler) as (base_url, session):
        av = _make_avatar(base_url=base_url, http_session=session)
        agent_session = _FakeAgentSession()
        await av.start(
            agent_session,  # type: ignore[arg-type]
            _FakeRoom(),  # type: ignore[arg-type]
            livekit_url="wss://example.livekit.cloud",
            livekit_api_key="devkey",
            livekit_api_secret="devsecret",
        )

        with pytest.raises(RuntimeError, match="only be called once"):
            await av.start(
                agent_session,  # type: ignore[arg-type]
                _FakeRoom(),  # type: ignore[arg-type]
                livekit_url="wss://example.livekit.cloud",
                livekit_api_key="devkey",
                livekit_api_secret="devsecret",
            )

    assert calls["n"] == 1, "second start() must not call the gateway again"
    assert av.provider_session_id == "ls_1", "the first session's terminate handle must survive"


class _FakeBrokenOutput:
    def replace_audio_tail(self, sink: Any) -> None:
        raise RuntimeError("boom")


class _FakeBrokenAgentSession(_FakeAgentSession):
    def __init__(self) -> None:
        super().__init__()
        self.output = _FakeBrokenOutput()  # type: ignore[assignment]


async def test_start_ids_set_before_audio_rebind(monkeypatch: pytest.MonkeyPatch) -> None:
    """session_id/provider_session_id/terminate_token must be recorded before
    the audio-output rebind, so a failure in the rebind still leaves a
    terminable (billable) session rather than an orphaned one."""

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {"session_id": "AVS_1", "provider_session_id": "ls_1", "terminate_token": "tt_1"}
        )

    monkeypatch.setattr(avatar_mod, "get_job_context", lambda *a, **k: _FakeJobCtx())

    async with _gateway(handler) as (base_url, session):
        av = _make_avatar(base_url=base_url, http_session=session)
        with pytest.raises(RuntimeError, match="boom"):
            await av.start(
                _FakeBrokenAgentSession(),  # type: ignore[arg-type]
                _FakeRoom(),  # type: ignore[arg-type]
                livekit_url="wss://example.livekit.cloud",
                livekit_api_key="devkey",
                livekit_api_secret="devsecret",
            )

    assert av.provider_session_id == "ls_1"
    assert av._terminate_token == "tt_1"


class _FakeLocalParticipant:
    identity = "standalone-agent"


class _FakeConnectedRoom:
    name = "my-room"

    def isconnected(self) -> bool:
        return True

    @property
    def local_participant(self) -> Any:
        return _FakeLocalParticipant()

    @property
    def sid(self) -> Any:
        async def _get() -> str:
            return "RM_789"

        return _get()

    def on(self, *_a: Any, **_k: Any) -> None:
        pass

    def off(self, *_a: Any, **_k: Any) -> None:
        pass


async def test_start_standalone_connected_room(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a job context, a connected room's local_participant/sid drive
    start() (guarded by isconnected(), not a bare local_participant access,
    which raises on a real rtc.Room instead of returning None)."""
    captured: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.Response:
        captured["body"] = await request.json()
        return web.json_response({"session_id": "AVS_1", "provider_session_id": "ls_1"})

    monkeypatch.setattr(avatar_mod, "get_job_context", lambda *a, **k: None)
    monkeypatch.setattr(avatar_mod, "DataStreamAudioOutput", _FakeSink)

    async with _gateway(handler) as (base_url, session):
        av = _make_avatar(base_url=base_url, http_session=session)
        await av.start(
            _FakeAgentSession(),  # type: ignore[arg-type]
            _FakeConnectedRoom(),  # type: ignore[arg-type]
            livekit_url="wss://example.livekit.cloud",
            livekit_api_key="devkey",
            livekit_api_secret="devsecret",
        )

        # isconnected() == True also makes the base class eagerly spawn a
        # join-wait task; this fake doesn't implement remote_participants
        # (out of scope for this test), so retrieve/discard its exception
        # rather than leaving it unretrieved at garbage-collection time.
        join_task = av._wait_avatar_join_task
        if join_task is not None:
            join_task.cancel()
            with contextlib.suppress(BaseException):
                await join_task

    assert captured["body"]["agent_identity"] == "standalone-agent"
    assert captured["body"]["room_sid"] == "RM_789"


async def test_start_standalone_disconnected_room_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a job context and a disconnected room, start() must raise the
    documented, actionable RuntimeError rather than an opaque error from the
    rtc layer (Room.local_participant raises rather than returning None)."""
    monkeypatch.setattr(avatar_mod, "get_job_context", lambda *a, **k: None)

    av = _make_avatar()
    with pytest.raises(RuntimeError, match="needs a connected room"):
        await av.start(
            _FakeAgentSession(),  # type: ignore[arg-type]
            _FakeRoom(),  # type: ignore[arg-type]
            livekit_url="wss://example.livekit.cloud",
            livekit_api_key="devkey",
            livekit_api_secret="devsecret",
        )


async def test_aclose_terminates_session() -> None:
    """aclose() calls the gateway terminate endpoint (with the terminate_token
    the gateway requires) so the provider session stops billing instead of
    lingering until idle timeout."""
    terminate_bodies: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.Response:
        terminate_bodies.append(await request.json())
        return web.json_response({"terminated": True})

    async with _gateway(handler, path="/v1/avatar/sessions/terminate") as (base_url, session):
        av = _make_avatar(base_url=base_url, http_session=session)
        av._provider_session_id = "ls_abc"
        av._terminate_token = "tt_abc"
        # No room/agent were started; base aclose() no-ops on cleanup.
        await av.aclose()

    assert len(terminate_bodies) == 1
    assert terminate_bodies[0] == {
        "provider": "lemonslice",
        "provider_session_id": "ls_abc",
        "terminate_token": "tt_abc",
    }
    # cleared on a confirmed terminate, so a second aclose() does not re-terminate.
    assert av.provider_session_id is None
    assert av._terminate_token is None


async def test_aclose_skips_terminate_without_token() -> None:
    """If the create response never returned a terminate_token, aclose() must
    not call the gateway at all (it would just 400) — it logs and moves on."""
    calls = {"n": 0}

    async def handler(request: web.Request) -> web.Response:
        calls["n"] += 1
        return web.json_response({"terminated": True})

    async with _gateway(handler, path="/v1/avatar/sessions/terminate") as (base_url, session):
        av = _make_avatar(base_url=base_url, http_session=session)
        av._provider_session_id = "ls_abc"
        # av._terminate_token left None, as if the gateway never returned one.
        await av.aclose()  # must not raise

    assert calls["n"] == 0, "terminate must not be attempted without a terminate_token"


async def test_aclose_terminate_failure_keeps_ids_and_runs_base_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed terminate must not raise, must leave the ids in place so a
    subsequent aclose() call can retry, and must not block the base class's
    own cleanup."""
    from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession

    base_aclose_calls = {"n": 0}
    orig_base_aclose = BaseAvatarSession.aclose

    async def spy_base_aclose(self: Any) -> None:
        base_aclose_calls["n"] += 1
        await orig_base_aclose(self)

    monkeypatch.setattr(BaseAvatarSession, "aclose", spy_base_aclose)

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": "boom"}, status=500)

    async with _gateway(handler, path="/v1/avatar/sessions/terminate") as (base_url, session):
        av = _make_avatar(base_url=base_url, http_session=session)
        av._provider_session_id = "ls_abc"
        av._terminate_token = "tt_abc"

        await av.aclose()  # must not raise

    assert base_aclose_calls["n"] == 1
    assert av.provider_session_id == "ls_abc", "must survive a failed terminate for a retry"
    assert av._terminate_token == "tt_abc"


async def test_aclose_without_session_is_noop() -> None:
    av = _make_avatar()
    # No provider session created; aclose() must not raise.
    await av.aclose()
