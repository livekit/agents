"""Unit tests for the Humanlike avatar plugin.

Covers plugin registration, configuration, the ``HumanlikeAPI`` HTTP client, and the
``AvatarSession`` lifecycle: video models, client-rendered models, an avatar that leaves
mid-call, and the renewal of capped sessions. The Humanlike API, the LiveKit room and the
avatar join are faked, so this runs offline.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
import pytest

from livekit import api, rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, Plugin
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.plugins import humanlike
from livekit.plugins.humanlike import avatar as avatar_module
from livekit.plugins.humanlike.api import DEFAULT_API_URL, HumanlikeAPI

pytestmark = [pytest.mark.unit, pytest.mark.plugin("humanlike")]

LK_URL = "wss://dev.livekit.cloud"
LK_KEY = "lk-api-key"
LK_SECRET = "lk-api-secret-that-is-long-enough-for-hs256"
_OPTS = APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=1.0)


def test_plugin_registered():
    assert "livekit.plugins.humanlike" in [p.title for p in Plugin.registered_plugins]


def test_to_ws_url():
    assert avatar_module._to_ws_url("https://x.livekit.cloud") == "wss://x.livekit.cloud"
    assert avatar_module._to_ws_url("http://localhost:7880") == "ws://localhost:7880"
    assert avatar_module._to_ws_url("wss://x.livekit.cloud") == "wss://x.livekit.cloud"


class TestConfig:
    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        monkeypatch.delenv("HUMANLIKE_API_KEY", raising=False)
        monkeypatch.delenv("HUMANLIKE_API_URL", raising=False)

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("HUMANLIKE_API_KEY", "hk_live_env")
        assert humanlike.AvatarSession()._api_key == "hk_live_env"

    def test_api_key_argument_wins(self, monkeypatch):
        monkeypatch.setenv("HUMANLIKE_API_KEY", "hk_live_env")
        assert humanlike.AvatarSession(api_key="hk_live_arg")._api_key == "hk_live_arg"

    def test_missing_api_key_raises(self):
        with pytest.raises(humanlike.HumanlikeException):
            humanlike.AvatarSession()

    def test_publishable_key_is_refused(self):
        with pytest.raises(humanlike.HumanlikeException, match="publishable"):
            humanlike.AvatarSession(api_key="hk_pub_browser")

    def test_api_url(self, monkeypatch):
        assert humanlike.AvatarSession(api_key="k")._api_url == DEFAULT_API_URL
        monkeypatch.setenv("HUMANLIKE_API_URL", "https://staging.example")
        assert humanlike.AvatarSession(api_key="k")._api_url == "https://staging.example"

    def test_defaults(self):
        session = humanlike.AvatarSession(api_key="k")
        assert session._model == "humanlike/homo-1"
        assert session.avatar_identity == "humanlike-avatar-agent"
        assert session.provider == "humanlike"


# -- HTTP client ---------------------------------------------------------------------------


class _Response:
    def __init__(self, status: int, body: Any = None) -> None:
        self.status = status
        self.ok = status < 400
        self._body = body

    async def text(self) -> str:
        return self._body if isinstance(self._body, str) else json.dumps(self._body)

    async def json(self, content_type: str | None = None) -> Any:
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _ScriptedHTTP:
    """Answers each request with the next step of `script`; an exception is raised instead."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(step, BaseException):
            raise step
        return step


def _client(http: _ScriptedHTTP) -> HumanlikeAPI:
    return HumanlikeAPI(
        "hk_live_test",
        "http://humanlike.test/",
        session=http,  # type: ignore[arg-type]
        conn_options=_OPTS,
    )


class TestApiClient:
    async def test_create_session_sends_auth_and_body(self):
        http = _ScriptedHTTP([_Response(201, {"id": "as_1"})])

        view = await _client(http).create_session({"model": "humanlike/homo-1"}, timeout=5)

        assert view == {"id": "as_1"}
        call = http.calls[0]
        assert call["method"] == "POST"
        assert call["url"] == "http://humanlike.test/v1/avatar/sessions"
        assert call["json"] == {"model": "humanlike/homo-1"}
        assert call["headers"]["Authorization"] == "Bearer hk_live_test"
        assert call["headers"]["Idempotency-Key"]
        assert call["timeout"].total == 5

    async def test_client_error_is_not_retried_and_names_the_problem(self):
        body = {"error": {"code": "unknown_model", "message": 'Unknown model "anam/cara-5"'}}
        http = _ScriptedHTTP([_Response(400, body)])

        with pytest.raises(APIStatusError) as exc_info:
            await _client(http).create_session({}, timeout=5)

        assert len(http.calls) == 1
        assert exc_info.value.status_code == 400
        assert 'Unknown model "anam/cara-5" (unknown_model)' in str(exc_info.value)

    async def test_server_error_is_retried_with_the_same_idempotency_key(self):
        http = _ScriptedHTTP(
            [_Response(503, "busy"), _Response(503, "busy"), _Response(201, {"id": "as_1"})]
        )

        view = await _client(http).create_session({}, timeout=5)

        assert view["id"] == "as_1"
        assert len(http.calls) == 3
        assert len({c["headers"]["Idempotency-Key"] for c in http.calls}) == 1

    async def test_persistent_server_error_keeps_its_status(self):
        http = _ScriptedHTTP([_Response(502, "bad gateway")])

        with pytest.raises(APIStatusError) as exc_info:
            await _client(http).create_session({}, timeout=5)

        assert len(http.calls) == _OPTS.max_retry + 1
        assert exc_info.value.status_code == 502

    async def test_network_error_is_retried_and_chained(self):
        http = _ScriptedHTTP([aiohttp.ClientConnectionError("connection refused")])

        with pytest.raises(APIConnectionError) as exc_info:
            await _client(http).create_session({}, timeout=5)

        assert len(http.calls) == _OPTS.max_retry + 1
        assert isinstance(exc_info.value.__cause__, aiohttp.ClientConnectionError)

    async def test_malformed_success_body_is_not_retried(self):
        http = _ScriptedHTTP([_Response(201, "not json")])

        with pytest.raises(APIConnectionError) as exc_info:
            await _client(http).create_session({}, timeout=5)

        assert len(http.calls) == 1
        assert exc_info.value.retryable is False

    async def test_end_and_renew(self):
        http = _ScriptedHTTP([_Response(204), _Response(201, {"id": "as_2"})])
        client = _client(http)

        assert await client.end_session("as_1") is None
        assert (await client.renew_session("as_1", {"transport": {}}, timeout=5))["id"] == "as_2"

        assert http.calls[0]["method"] == "DELETE"
        assert http.calls[0]["url"].endswith("/v1/avatar/sessions/as_1")
        assert http.calls[1]["url"].endswith("/v1/avatar/sessions/as_1/renew")


# -- AvatarSession -------------------------------------------------------------------------


class _Participant:
    def __init__(self, identity: str, sid: str) -> None:
        self.identity = identity
        self.sid = sid


class _LocalParticipant:
    def __init__(self, identity: str = "agent-1") -> None:
        self.identity = identity
        self.attributes: dict[str, str] = {}

    async def set_attributes(self, attributes: dict[str, str]) -> None:
        self.attributes.update(attributes)


class _Room:
    def __init__(self) -> None:
        self.name = "room-1"
        self.local_participant = _LocalParticipant()
        self.remote_participants: dict[str, _Participant] = {}
        self.connected = True
        self._handlers: dict[str, list[Any]] = {}

    def isconnected(self) -> bool:
        return self.connected

    def on(self, event: str, handler: Any) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Any) -> None:
        if handler in self._handlers.get(event, []):
            self._handlers[event].remove(handler)

    def listener_count(self, event: str) -> int:
        return len(self._handlers.get(event, []))

    def join(self, identity: str, sid: str) -> None:
        participant = _Participant(identity, sid)
        self.remote_participants[identity] = participant
        for handler in list(self._handlers.get("participant_connected", [])):
            handler(participant)

    def leave(self, identity: str, sid: str) -> None:
        if getattr(self.remote_participants.get(identity), "sid", None) == sid:
            del self.remote_participants[identity]
        for handler in list(self._handlers.get("participant_disconnected", [])):
            handler(_Participant(identity, sid))


class _Output:
    def __init__(self) -> None:
        self.audio: Any = None

    def replace_audio_tail(self, sink: Any) -> None:
        self.audio = sink


class _AgentSession:
    def __init__(self) -> None:
        self.output = _Output()

    def on(self, *args: Any) -> None:
        pass

    def off(self, *args: Any) -> None:
        pass

    def emit(self, *args: Any) -> None:
        pass


class _AudioOutput:
    def __init__(
        self, room: Any, *, destination_identity: str, sample_rate: int, wait_remote_track: Any
    ) -> None:
        self.destination_identity = destination_identity
        self.sample_rate = sample_rate
        self.wait_remote_track = wait_remote_track


class _Backend:
    """Stands in for HumanlikeAPI and records what the plugin asked for."""

    def __init__(self) -> None:
        self.view: dict[str, Any] = _video_view()
        self.renewed: dict[str, Any] = {"id": "as_2", "model": "humanlike/homo-1"}
        self.create_error: Exception | None = None
        self.renew_error: Exception | None = None
        self.created: list[dict[str, Any]] = []
        self.renewals: list[tuple[str, dict[str, Any]]] = []
        self.ended: list[str] = []
        self.on_renew: Any = None

    async def create_session(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        self.created.append(payload)
        if self.create_error is not None:
            raise self.create_error
        return self.view

    async def renew_session(
        self, session_id: str, payload: dict[str, Any], *, timeout: float
    ) -> dict[str, Any]:
        self.renewals.append((session_id, payload))
        if self.renew_error is not None:
            raise self.renew_error
        if self.on_renew is not None:
            self.on_renew()
        return self.renewed

    async def end_session(self, session_id: str, *, timeout: float = 15.0) -> None:
        self.ended.append(session_id)


def _video_view(**extra: Any) -> dict[str, Any]:
    return {
        "id": "as_1",
        "model": "humanlike/homo-1",
        "provider": "modal",
        "avatar_identity": "humanlike-avatar-agent",
        "audio_in": "livekit-datastream",
        "sample_rate": 24000,
        **extra,
    }


def _claims(token: str) -> api.access_token.Claims:
    return api.TokenVerifier(LK_KEY, LK_SECRET).verify(token)


def _raw_claims(token: str) -> dict[str, Any]:
    """The JWT payload as sent; TokenVerifier does not read back every field (``kind``)."""
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


class TestAvatarSession:
    @pytest.fixture(autouse=True)
    async def _fail_on_leaked_tasks(self):
        before = asyncio.all_tasks()
        yield
        await asyncio.sleep(0)
        leaked = {
            t
            for t in asyncio.all_tasks() - before
            if t is not asyncio.current_task() and not t.done()
        }
        assert not leaked, f"leaked tasks: {leaked}"

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        monkeypatch.setenv("LIVEKIT_URL", "https://dev.livekit.cloud")
        monkeypatch.setenv("LIVEKIT_API_KEY", LK_KEY)
        monkeypatch.setenv("LIVEKIT_API_SECRET", LK_SECRET)
        monkeypatch.setattr(avatar_module, "DataStreamAudioOutput", _AudioOutput)
        monkeypatch.setattr(avatar_module, "get_job_context", lambda required=True: None)

    @pytest.fixture
    def backend(self, monkeypatch) -> _Backend:
        backend = _Backend()
        monkeypatch.setattr(avatar_module, "HumanlikeAPI", lambda *args, **kwargs: backend)
        return backend

    @pytest.fixture
    def join(self, monkeypatch):
        """The avatar's join, which the base class waits for; set it to let start() finish."""
        joined = asyncio.Event()

        async def _wait(**kwargs: Any) -> None:
            await joined.wait()

        monkeypatch.setattr("livekit.agents.utils.wait_for_participant", _wait)
        monkeypatch.setattr("livekit.agents.utils.wait_for_track_publication", _wait)
        return joined

    def _session(self, **kwargs: Any) -> humanlike.AvatarSession:
        return humanlike.AvatarSession(api_key="hk_live_test", http_session=object(), **kwargs)  # type: ignore[arg-type]

    async def _start(
        self,
        session: humanlike.AvatarSession,
        room: _Room,
        agent: _AgentSession,
        join: asyncio.Event,
    ) -> None:
        task = asyncio.create_task(session.start(agent, room))  # type: ignore[arg-type]
        await _settle()
        room.join(session.avatar_identity, "PA_1")
        join.set()
        await task

    async def test_video_model_routes_speech_to_the_avatar(self, backend, join):
        room, agent = _Room(), _AgentSession()
        session = self._session(
            avatar_id="face-1", fallback=["anam/cara-4"], options={"expressiveness": 0.8}
        )

        await self._start(session, room, agent, join)

        body = backend.created[0]
        assert body["model"] == "humanlike/homo-1"
        assert body["avatar"] == {"id": "face-1"}
        assert body["fallback"] == ["anam/cara-4"]
        assert body["options"] == {"expressiveness": 0.8}
        transport = body["transport"]
        assert transport["url"] == LK_URL  # https:// from LiveKit Cloud becomes wss://
        assert transport["room_name"] == "room-1"
        assert transport["identity"] == "humanlike-avatar-agent"
        assert transport["agent_identity"] == "agent-1"

        claims = _claims(transport["token"])
        assert claims.identity == "humanlike-avatar-agent"
        assert _raw_claims(transport["token"])["kind"] == "agent"
        assert claims.video.room == "room-1" and claims.video.room_join
        assert claims.attributes[ATTRIBUTE_PUBLISH_ON_BEHALF] == "agent-1"
        assert LK_SECRET not in json.dumps(body)

        sink = agent.output.audio
        assert isinstance(sink, _AudioOutput)
        assert sink.destination_identity == "humanlike-avatar-agent"
        assert sink.sample_rate == 24000
        assert sink.wait_remote_track == rtc.TrackKind.KIND_VIDEO
        assert session.session_id == "as_1" and session.model == "humanlike/homo-1"
        assert session.render is None

        await session.aclose()
        assert backend.ended == ["as_1"]

    async def test_only_a_model_is_needed(self, backend, join):
        room, agent = _Room(), _AgentSession()
        session = self._session(model="humanlike/homo-1-pro")

        await self._start(session, room, agent, join)

        assert set(backend.created[0]) == {"model", "transport"}
        await session.aclose()

    async def test_missing_livekit_credentials_raise_before_any_request(self, backend, monkeypatch):
        monkeypatch.delenv("LIVEKIT_API_SECRET")
        session = self._session()

        with pytest.raises(humanlike.HumanlikeException, match="livekit_api_secret"):
            await session.start(_AgentSession(), _Room())  # type: ignore[arg-type]

        assert backend.created == []

    async def test_client_rendered_model_publishes_what_to_draw(self, backend, join):
        backend.view = _video_view(
            model="humanlike/homo-0", audio_in="none", render={"type": "lam", "face": "socrates"}
        )
        room, agent = _Room(), _AgentSession()
        session = self._session(model="humanlike/homo-0")

        await session.start(agent, room)  # type: ignore[arg-type]
        await session.wait_for_join(timeout=0.01)  # nothing joins; returns at once

        assert agent.output.audio is None  # the agent keeps its own audio track
        assert json.loads(room.local_participant.attributes[humanlike.ATTRIBUTE_RENDER]) == {
            "type": "lam",
            "face": "socrates",
        }
        assert session.render == {"type": "lam", "face": "socrates"}
        await session.aclose()
        assert backend.ended == ["as_1"]

    async def test_avatar_that_never_joins_is_ended(self, backend, join):
        session = self._session(join_timeout=0.05)

        with pytest.raises(humanlike.HumanlikeException, match="did not join"):
            await session.start(_AgentSession(), _Room())  # type: ignore[arg-type]

        assert backend.ended == ["as_1"]

    async def test_rejected_start_raises_the_api_error(self, backend, join):
        backend.create_error = APIStatusError("Unknown model (unknown_model)", status_code=400)
        session = self._session(model="anam/cara-5")

        with pytest.raises(APIStatusError):
            await session.start(_AgentSession(), _Room())  # type: ignore[arg-type]

        assert backend.ended == []  # nothing was started

    async def test_unsupported_audio_input_is_ended(self, backend, join):
        backend.view = _video_view(audio_in="websocket")
        session = self._session()

        with pytest.raises(humanlike.HumanlikeException, match="upgrade"):
            await session.start(_AgentSession(), _Room())  # type: ignore[arg-type]

        assert backend.ended == ["as_1"]

    async def test_avatar_leaving_mid_call_ends_the_session_once(self, backend, join):
        room, agent = _Room(), _AgentSession()
        session = self._session()
        await self._start(session, room, agent, join)

        room.leave("someone-else", "PA_9")
        room.leave(session.avatar_identity, "PA_1")
        room.leave(session.avatar_identity, "PA_1")
        await _settle()

        assert backend.ended == ["as_1"]
        await session.aclose()
        assert backend.ended == ["as_1"]

    async def test_avatar_leaving_with_the_room_is_left_to_aclose(self, backend, join):
        room, agent = _Room(), _AgentSession()
        session = self._session()
        await self._start(session, room, agent, join)

        room.connected = False
        room.leave(session.avatar_identity, "PA_1")
        await _settle()
        assert backend.ended == []

        await session.aclose()
        assert backend.ended == ["as_1"]

    async def test_capped_session_is_renewed_before_its_cap(self, backend, join, monkeypatch):
        monkeypatch.setattr(avatar_module, "_MIN_RENEW_DELAY_S", 0.0)
        soon = (
            (datetime.now(timezone.utc) + timedelta(seconds=45)).isoformat().replace("+00:00", "Z")
        )
        backend.view = _video_view(expires_at=soon)
        room, agent = _Room(), _AgentSession()
        session = self._session()

        def successor_joins() -> None:
            # LiveKit replaces the old participant: the successor joins, the predecessor leaves
            asyncio.get_running_loop().call_soon(room.join, session.avatar_identity, "PA_2")
            asyncio.get_running_loop().call_soon(room.leave, session.avatar_identity, "PA_1")

        backend.on_renew = successor_joins
        await self._start(session, room, agent, join)
        for _ in range(20):
            await asyncio.sleep(0.01)
            if backend.ended:
                break

        previous, payload = backend.renewals[0]
        assert previous == "as_1"
        assert _claims(payload["transport"]["token"]).identity == session.avatar_identity
        assert session.session_id == "as_2"
        assert backend.ended == ["as_1"]  # the predecessor, once the successor joined

        await session.aclose()
        assert backend.ended == ["as_1", "as_2"]

    async def test_failed_renewal_keeps_the_current_avatar(self, backend, join, monkeypatch):
        monkeypatch.setattr(avatar_module, "_MIN_RENEW_DELAY_S", 0.0)
        soon = (datetime.now(timezone.utc) + timedelta(seconds=45)).isoformat()
        backend.view = _video_view(expires_at=soon)
        backend.renew_error = APIStatusError("capacity", status_code=503)
        room, agent = _Room(), _AgentSession()
        session = self._session()

        await self._start(session, room, agent, join)
        for _ in range(20):
            await asyncio.sleep(0.01)
            if backend.renewals:
                break
        await _settle()

        assert session.session_id == "as_1"
        assert backend.ended == []
        await session.aclose()
        assert backend.ended == ["as_1"]

    async def test_aclose_is_idempotent_and_removes_listeners(self, backend, join):
        room, agent = _Room(), _AgentSession()
        session = self._session()
        await self._start(session, room, agent, join)
        assert room.listener_count("participant_disconnected") == 1

        await session.aclose()
        await session.aclose()

        assert backend.ended == ["as_1"]
        assert room.listener_count("participant_connected") == 0
        assert room.listener_count("participant_disconnected") == 0


async def test_usage_example(monkeypatch):
    """The README's snippet, end to end against fakes."""
    backend = _Backend()
    monkeypatch.setenv("HUMANLIKE_API_KEY", "hk_live_test")
    monkeypatch.setenv("LIVEKIT_URL", LK_URL)
    monkeypatch.setenv("LIVEKIT_API_KEY", LK_KEY)
    monkeypatch.setenv("LIVEKIT_API_SECRET", LK_SECRET)
    monkeypatch.setattr(avatar_module, "HumanlikeAPI", lambda *args, **kwargs: backend)
    monkeypatch.setattr(avatar_module, "DataStreamAudioOutput", _AudioOutput)
    monkeypatch.setattr(avatar_module, "get_job_context", lambda required=True: None)

    async def _joined(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr("livekit.agents.utils.wait_for_participant", _joined)
    monkeypatch.setattr("livekit.agents.utils.wait_for_track_publication", _joined)

    room, session = _Room(), _AgentSession()
    avatar = humanlike.AvatarSession(model="humanlike/homo-1", http_session=object())  # type: ignore[arg-type]
    await avatar.start(session, room=room)  # type: ignore[arg-type]

    assert backend.created[0]["model"] == "humanlike/homo-1"
    await avatar.aclose()
