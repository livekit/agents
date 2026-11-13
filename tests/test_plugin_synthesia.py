from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.plugins import synthesia
from livekit.plugins.synthesia.api import SESSION_PATH, SynthesiaAPI
from livekit.plugins.synthesia.errors import ErrorType, SynthesiaError
from livekit.plugins.synthesia.types import (
    AVATAR_IDENTITY,
    AVATAR_NAME,
    DEFAULT_API_URL,
    TOKEN_TTL,
    StartSessionRequest,
    StartSessionResponse,
)

pytestmark = pytest.mark.unit

ADA_ID = "03cee7ec-ac90-45ec-8c20-74a399cf3dc4"
SECOND_ID = "6d999451-039c-4bf2-9b88-c769ac2faa78"


class TestConfig:
    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        monkeypatch.delenv("SYNTHESIA_API_KEY", raising=False)
        monkeypatch.delenv("SYNTHESIA_API_URL", raising=False)

    @pytest.mark.parametrize(
        ("field", "env_var", "arg", "default"),
        [
            ("api_key", "SYNTHESIA_API_KEY", "arg-key", None),
            ("api_url", "SYNTHESIA_API_URL", "https://arg.example", DEFAULT_API_URL),
        ],
    )
    def test_env_fills_in_and_arg_overrides(self, monkeypatch, field, env_var, arg, default):
        config = synthesia.AvatarConfig(avatar_ids=[ADA_ID])
        extra = {} if field == "api_key" else {"api_key": "k"}
        if default is not None:
            assert getattr(synthesia.AvatarSession(config, **extra)._config, field) == default

        monkeypatch.setenv(env_var, "from-env")
        assert getattr(synthesia.AvatarSession(config, **extra)._config, field) == "from-env"
        avatar = synthesia.AvatarSession(config, **extra, **{field: arg})
        assert getattr(avatar._config, field) == arg

    def test_api_key_missing_raises(self):
        with pytest.raises(synthesia.SynthesiaError):
            synthesia.AvatarSession(synthesia.AvatarConfig(avatar_ids=[ADA_ID]))

    @pytest.mark.parametrize(
        ("avatar_ids", "match"),
        [
            ([], "between 1 and"),
            ([f"avatar-{i}" for i in range(6)], "between 1 and"),
            ("x", "list"),
        ],
    )
    def test_avatar_ids_are_validated(self, avatar_ids, match):
        with pytest.raises(ValueError, match=match):
            synthesia.AvatarConfig(avatar_ids=avatar_ids)

    def test_avatar_ids_are_copied_from_input(self):
        ids = [f"avatar-{i}" for i in range(5)]
        config = synthesia.AvatarConfig(avatar_ids=ids)
        ids.append("late")
        avatar = synthesia.AvatarSession(config, api_key="k")
        assert avatar._config.avatar_ids == tuple(ids[:5])

    def test_config_repr_redacts_api_key(self):
        avatar = synthesia.AvatarSession(
            synthesia.AvatarConfig(avatar_ids=[ADA_ID]), api_key="super-secret"
        )
        assert "super-secret" not in repr(avatar._config)


class TestErrors:
    @pytest.mark.parametrize(
        ("error_type", "retryable"),
        [
            (None, False),
            (ErrorType.AUTH, False),
            (ErrorType.FEATURE_NOT_IN_PLAN, False),
            (ErrorType.UNKNOWN_AVATAR, False),
            (ErrorType.QUOTA_EXCEEDED, False),
            (ErrorType.INVALID_ROOM_TOKEN, False),
            (ErrorType.LIVEKIT_CREDENTIALS_REJECTED, False),
            (ErrorType.INVALID_SESSION_REQUEST, False),
            (ErrorType.RATE_LIMITED, True),
            (ErrorType.CONCURRENCY_LIMIT, True),
            (ErrorType.TIMEOUT, True),
            (ErrorType.CONNECTION, True),
        ],
    )
    def test_retryable_follows_the_error_type(self, error_type, retryable):
        assert SynthesiaError("boom", type=error_type).retryable is retryable

    def test_retryable_can_be_overridden(self):
        assert SynthesiaError("boom", type=ErrorType.AUTH, retryable=True).retryable is True
        assert SynthesiaError("boom", type=ErrorType.CONNECTION, retryable=False).retryable is False

    def test_str_never_raises_on_non_string_message(self):
        assert "boom" in str(SynthesiaError({"context": ["boom"]}))


class TestApiClient:
    API_KEY = "sk-secret-key"
    API_URL = "https://api.example"
    NO_SLEEP = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0)
    ONE_ATTEMPT = dataclasses.replace(NO_SLEEP, max_retry=0)
    TWO_ATTEMPTS = dataclasses.replace(NO_SLEEP, max_retry=1)
    REQUEST_ID = "req_8c2f7d1e4b5a46f0a1b2c3d4e5f60718"

    class _FakeResponse:
        def __init__(self, status, *, body=None, headers=None, content_type="application/json"):
            self.status = status
            self._body = body
            self.headers = headers or {}
            self.content_type = content_type

        @property
        def ok(self):
            return 200 <= self.status < 300

        async def json(self, *, content_type=None):
            if self._body is None:
                raise aiohttp.ContentTypeError(None, None)
            if content_type is not None and self.content_type != content_type:
                raise aiohttp.ContentTypeError(None, None)
            return self._body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _RaisingCtx:
        def __init__(self, exc):
            self._exc = exc

        async def __aenter__(self):
            raise self._exc

        async def __aexit__(self, *exc):
            return False

    class _FakeSession:
        """Records each POST and yields the next queued response or raises a queued error."""

        def __init__(self, outcomes):
            self._outcomes = list(outcomes)
            self.calls = []

        def post(self, url, *, headers=None, json=None, timeout=None):
            self.calls.append({"url": url, "headers": headers, "json": json})
            outcome = self._outcomes.pop(0)
            if isinstance(outcome, Exception):
                return TestApiClient._RaisingCtx(outcome)
            return outcome

    def _request(self, avatar_ids=("avatar-1",)):
        return StartSessionRequest(
            avatar_ids=list(avatar_ids),
            livekit_url="wss://room.livekit.cloud",
            lk_token="lk-token-secret",
        )

    async def _start(self, *outcomes, conn_options=None, request=None):
        session = self._FakeSession(outcomes)
        client = SynthesiaAPI(
            api_key=self.API_KEY,
            api_url=self.API_URL,
            session=session,
            conn_options=conn_options or self.NO_SLEEP,
        )
        return session, await client.start_session(request or self._request())

    async def _error(self, *outcomes, conn_options=None):
        with pytest.raises(SynthesiaError) as exc:
            await self._start(*outcomes, conn_options=conn_options)
        return exc.value

    def _problem(self, code, *, status, title="t", detail=None, with_code=True, headers=None):
        body = {
            "type": f"https://developers.synthesia.io/errors/{code}",
            "title": title,
            "status": status,
            **({"code": code} if with_code else {}),
            **({"detail": detail} if detail is not None else {}),
        }
        return self._FakeResponse(
            status, body=body, headers=headers, content_type="application/problem+json"
        )

    @pytest.mark.parametrize(
        ("status", "body", "session_id"),
        [(200, {"session_id": "sess_123"}, "sess_123"), (201, {"id": "ses_9f3c"}, "ses_9f3c")],
    )
    async def test_success_returns_the_session_id(self, status, body, session_id):
        _, result = await self._start(self._FakeResponse(status, body=body))
        assert result == StartSessionResponse(session_id=session_id)

    async def test_request_uses_the_published_route_auth_and_wire_names(self):
        session, _ = await self._start(
            self._FakeResponse(201, body={"id": "ses_9f3c"}),
            request=self._request(["lead", "av_already-prefixed", "third"]),
        )
        call = session.calls[0]
        assert call["url"] == self.API_URL + SESSION_PATH
        assert SESSION_PATH == "/api/interactive-avatars/sessions"
        assert call["headers"]["Authorization"] == self.API_KEY
        assert call["json"] == {
            "avatarIds": ["av_lead", "av_already-prefixed", "av_third"],
            "livekitUrl": "wss://room.livekit.cloud",
            "livekitToken": "lk-token-secret",
        }

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
    async def test_problem_code_maps_to_error_type(self, code, status, expected):
        err = await self._error(self._problem(code, status=status, detail="d"))
        assert err.type is expected

    async def test_problem_detail_status_and_request_id_reach_the_error(self):
        response = self._problem(
            "validation_error",
            status=400,
            detail="The supplied livekitToken cannot be used to join the room.",
        )
        response._body["requestId"] = "req_1"
        err = await self._error(response)
        assert str(err) == "The supplied livekitToken cannot be used to join the room."
        assert err.body == response._body
        assert (err.status, err.request_id) == (400, "req_1")

    @pytest.mark.parametrize(
        ("with_code", "detail", "expected"),
        [
            (False, None, ErrorType.UNKNOWN_AVATAR),  # no code: the status decides
            (True, None, None),  # an unmapped code stays untyped
            (True, [{"msg": "x"}], None),
        ],
    )
    async def test_problem_falls_back_to_status_and_title(self, with_code, detail, expected):
        response = self._problem(
            "not_found", status=404, title="Not found", detail=detail, with_code=with_code
        )
        err = await self._error(response)
        assert err.type is expected
        assert str(err) == "Not found"

    @pytest.mark.parametrize(
        ("code", "status", "expected"),
        [
            ("unauthorized", 401, ErrorType.AUTH),
            ("invalid_api_key", 401, ErrorType.AUTH),
            ("unknown_avatar", 404, ErrorType.UNKNOWN_AVATAR),
            ("avatar_not_accessible", 404, ErrorType.UNKNOWN_AVATAR),
            ("quota_exceeded", 402, ErrorType.QUOTA_EXCEEDED),
            ("rate_limited", 429, ErrorType.RATE_LIMITED),
            ("invalid_token", 400, ErrorType.INVALID_ROOM_TOKEN),
            ("validation_error", 400, ErrorType.INVALID_SESSION_REQUEST),
            ("invalid_livekit_credentials", 401, ErrorType.LIVEKIT_CREDENTIALS_REJECTED),
        ],
    )
    async def test_legacy_error_code_maps_to_error_type(self, code, status, expected):
        err = await self._error(
            self._FakeResponse(status, body={"error": {"code": code, "message": "nope"}})
        )
        assert err.type is expected

    @pytest.mark.parametrize(
        ("status", "body", "expected", "message"),
        [
            (
                401,
                {
                    "error": "LiveKitCredentialsRejectedError",
                    "code": "invalid_livekit_credentials",
                    "context": {"lk_token": ["LiveKit rejected the supplied credentials"]},
                },
                ErrorType.LIVEKIT_CREDENTIALS_REJECTED,
                "LiveKit rejected the supplied credentials",
            ),
            (401, {"message": "Unauthorized"}, ErrorType.AUTH, None),
            (400, {"error": {"code": "teapot", "message": "x is required"}}, None, None),
            (
                403,
                {"error": "Forbidden", "context": "User is not authenticated"},
                ErrorType.AUTH,
                "User is not authenticated",
            ),
            (403, {"error": "Forbidden"}, ErrorType.AUTH, "Forbidden"),
            (
                400,
                {
                    "code": "validation_error",
                    "context": {"livekit_url": ["Must be a wss:// URL"]},
                    "error": "InvalidSessionRequestError",
                },
                ErrorType.INVALID_SESSION_REQUEST,
                "Must be a wss:// URL",
            ),
        ],
        ids=["lk-credentials", "bare-401", "unmapped", "flat-context", "flat-string", "dict-ctx"],
    )
    async def test_flat_error_bodies(self, status, body, expected, message):
        err = await self._error(self._FakeResponse(status, body=body))
        assert err.type is expected
        assert err.body == body
        if message is not None:
            assert message in str(err)

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            (
                _FakeResponse(
                    429, body={"error": {"code": "rate_limited"}}, headers={"Retry-After": "7"}
                ),
                7.0,
            ),
            (_FakeResponse(429, body={"error": {"code": "rate_limited"}, "retry_after": 3.5}), 3.5),
            (_FakeResponse(429, body={"error": {"code": "rate_limited", "retry_after": 9}}), 9.0),
            (_FakeResponse(429, body={"error": {"code": "rate_limited"}}), None),
        ],
        ids=["header", "body", "nested", "absent"],
    )
    async def test_rate_limited_reads_retry_after(self, response, expected):
        err = await self._error(response)
        assert err.type is ErrorType.RATE_LIMITED
        assert err.retry_after == expected

    @pytest.mark.parametrize(
        ("code", "headers", "retry_after"),
        [("rate_limited", {"Retry-After": "12"}, 12.0), ("concurrency_limit", {}, None)],
    )
    async def test_problem_429_reads_retry_after_when_sent(self, code, headers, retry_after):
        err = await self._error(self._problem(code, status=429, detail="d", headers=headers))
        assert err.retry_after == retry_after

    async def test_terminal_error_is_not_retried(self):
        session = self._FakeSession(
            [self._FakeResponse(401, body={"error": {"code": "unauthorized"}})]
        )
        client = SynthesiaAPI(
            api_key=self.API_KEY, api_url=self.API_URL, session=session, conn_options=self.NO_SLEEP
        )
        with pytest.raises(SynthesiaError):
            await client.start_session(self._request())
        assert len(session.calls) == 1

    async def test_5xx_surfaces_the_api_detail_and_request_id(self):
        response = self._problem(
            "service_unavailable",
            status=503,
            title="Service unavailable",
            detail="The session could not be started. Retry.",
        )
        response._body["requestId"] = self.REQUEST_ID
        err = await self._error(response, conn_options=self.ONE_ATTEMPT)
        assert str(err) == (
            "could not start a Synthesia session; last attempt: HTTP 503: "
            f"The session could not be started. Retry. (request {self.REQUEST_ID})"
        )
        assert err.type is ErrorType.CONNECTION
        assert (err.status, err.request_id) == (503, self.REQUEST_ID)

    async def test_5xx_legacy_body_message_is_kept(self):
        body = {"error": {"code": "internal", "message": "upstream is down"}}
        err = await self._error(self._FakeResponse(500, body=body), conn_options=self.ONE_ATTEMPT)
        assert str(err) == (
            "could not start a Synthesia session; last attempt: HTTP 500: upstream is down"
        )

    async def test_5xx_on_every_attempt_names_the_last_status(self):
        attempts = self.NO_SLEEP.max_retry + 1
        err = await self._error(*[self._FakeResponse(502) for _ in range(attempts)])
        assert str(err) == (
            f"could not start a Synthesia session after {attempts} attempts; last attempt: HTTP 502"
        )
        assert (err.body, err.status) == (None, 502)

    async def test_5xx_whose_body_fails_to_arrive_is_still_a_5xx(self):
        class _TruncatedResponse(self._FakeResponse):
            async def json(self, *, content_type=None):
                raise aiohttp.ClientPayloadError("truncated")

        err = await self._error(
            _TruncatedResponse(503), _TruncatedResponse(503), conn_options=self.TWO_ATTEMPTS
        )
        assert str(err) == (
            "could not start a Synthesia session after 2 attempts; last attempt: HTTP 503"
        )
        assert err.status == 503
        assert isinstance(err.__cause__, aiohttp.ClientPayloadError)

    async def test_last_attempt_decides_the_message_when_outcomes_mix(self):
        answered = self._problem("service_unavailable", status=503, title="Service unavailable")

        err = await self._error(aiohttp.ClientError(), answered, conn_options=self.TWO_ATTEMPTS)
        assert str(err) == (
            "could not start a Synthesia session after 2 attempts;"
            " last attempt: HTTP 503: Service unavailable"
        )
        assert (err.body, err.status) == (answered._body, 503)

        err = await self._error(answered, aiohttp.ClientError(), conn_options=self.TWO_ATTEMPTS)
        assert str(err) == (
            "could not start a Synthesia session after 2 attempts; last attempt: connection error"
        )
        assert (err.body, err.status) == (None, None)
        assert isinstance(err.__cause__, aiohttp.ClientError)

    async def test_transport_errors_retry_per_conn_options_then_raise_connection(self):
        session = self._FakeSession([aiohttp.ClientError(), aiohttp.ClientError()])
        client = SynthesiaAPI(
            api_key=self.API_KEY,
            api_url=self.API_URL,
            session=session,
            conn_options=self.TWO_ATTEMPTS,
        )
        with pytest.raises(SynthesiaError) as exc:
            await client.start_session(self._request())
        assert exc.value.type is ErrorType.CONNECTION
        assert len(session.calls) == 2

    async def test_recovers_after_transient_failure(self):
        session, result = await self._start(
            self._FakeResponse(503), self._FakeResponse(200, body={"session_id": "sess_ok"})
        )
        assert result.session_id == "sess_ok"
        assert len(session.calls) == 2

    async def test_malformed_success_body_raises(self):
        await self._error(self._FakeResponse(200, body={"unexpected": True}))

    async def test_secrets_never_logged(self, caplog):
        caplog.set_level(logging.DEBUG, logger="livekit.plugins.synthesia")
        await self._start(self._FakeResponse(200, body={"session_id": "sess_ok"}))
        await self._error(self._FakeResponse(401, body={"error": {"code": "unauthorized"}}))
        assert self.API_KEY not in caplog.text
        assert "lk-token-secret" not in caplog.text
        assert "lk-token-secret" not in repr(self._request())


class TestAvatarSession:
    LK_URL = "wss://dev.livekit.cloud"
    LK_KEY = "lk-api-key"
    LK_SECRET = "lk-api-secret-never-leaks"
    CUSTOM_IDENTITY = "avatar-host"

    class _FakeLocalParticipant:
        def __init__(self, identity="dev-agent"):
            self.identity = identity
            self.rpc_calls = []
            self.rpc_response = json.dumps({"status": "ok", "avatar_id": "swapped-id"})
            self.rpc_error = None

        async def perform_rpc(
            self, *, destination_identity, method, payload, response_timeout=None
        ):
            self.rpc_calls.append(
                {
                    "destination_identity": destination_identity,
                    "method": method,
                    "payload": payload,
                    "response_timeout": response_timeout,
                }
            )
            if self.rpc_error is not None:
                raise self.rpc_error
            return self.rpc_response

    class _Emitter:
        def __init__(self):
            self._handlers = {}

        def on(self, event, handler):
            self._handlers.setdefault(event, []).append(handler)

        def off(self, event, handler):
            if handler in self._handlers.get(event, []):
                self._handlers[event].remove(handler)

        def fire(self, event, *args):
            for handler in list(self._handlers.get(event, [])):
                handler(*args)

        def emit(self, *args, **kwargs):
            pass

        def listener_count(self, event):
            return len(self._handlers.get(event, []))

    class _FakeRoom(_Emitter):
        def __init__(self, name="dev-room"):
            super().__init__()
            self.name = name
            self.local_participant = TestAvatarSession._FakeLocalParticipant()
            self.remote_participants = {}
            self._connected = True

        def isconnected(self):
            return self._connected

    class _FakeAudioOutput:
        def __init__(self, room, *, destination_identity, wait_remote_track=None, **kwargs):
            self.room = room
            self.destination_identity = destination_identity
            self.wait_remote_track = wait_remote_track
            self.closed = False

        async def aclose(self):
            self.closed = True

    class _FakeOutput:
        def __init__(self):
            self.audio = None

        def replace_audio_tail(self, sink):
            self.audio = sink

    class _FakeAgentSession(_Emitter):
        def __init__(self):
            super().__init__()
            self._started = False
            self.output = TestAvatarSession._FakeOutput()

    class _Recorder:
        def __init__(self):
            self.requests = []
            self.error = None
            self.response = StartSessionResponse(session_id="sess_123")
            # set to an asyncio.Event to hold start_session() open until it is set
            self.gate = None

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
    def fake_audio_output(self, monkeypatch):
        monkeypatch.setattr(
            "livekit.plugins.synthesia.avatar.DataStreamAudioOutput", self._FakeAudioOutput
        )

    @pytest.fixture(autouse=True)
    def job_context(self, monkeypatch):
        ctx = type("FakeJobContext", (), {"local_participant_identity": "dev-agent"})()
        monkeypatch.setattr(
            "livekit.plugins.synthesia.avatar.get_job_context", lambda *, required=True: ctx
        )
        return ctx

    @pytest.fixture
    def api_recorder(self, monkeypatch):
        rec = self._Recorder()

        class _RecorderAPI:
            def __init__(self, **kwargs):
                pass

            async def start_session(self, request, *, conn_options=None):
                rec.requests.append(request)
                if rec.gate is not None:
                    await rec.gate.wait()
                if rec.error is not None:
                    raise rec.error
                return rec.response

        monkeypatch.setattr("livekit.plugins.synthesia.avatar.SynthesiaAPI", _RecorderAPI)
        return rec

    @pytest.fixture
    def instant_join(self, monkeypatch):
        async def _ok(**kwargs):
            return None

        monkeypatch.setattr("livekit.agents.utils.wait_for_participant", _ok)
        monkeypatch.setattr("livekit.agents.utils.wait_for_track_publication", _ok)

    @pytest.fixture
    def hanging_join(self, monkeypatch):
        async def _never(**kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr("livekit.agents.utils.wait_for_participant", _never)
        monkeypatch.setattr("livekit.agents.utils.wait_for_track_publication", _never)

    def _session(self, avatar_ids=(ADA_ID,), **kwargs):
        return synthesia.AvatarSession(
            synthesia.AvatarConfig(avatar_ids=list(avatar_ids)), api_key="syn-key", **kwargs
        )

    def _multi_session(self, **kwargs):
        return self._session((ADA_ID, SECOND_ID), **kwargs)

    async def _start(self, session, room, agent=None, **kwargs):
        await session.start(
            agent or self._FakeAgentSession(),
            room,
            livekit_url=self.LK_URL,
            livekit_api_key=self.LK_KEY,
            livekit_api_secret=self.LK_SECRET,
            **kwargs,
        )

    @staticmethod
    def _decode_jwt(token):
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))

    async def test_token_grants_attribute_and_ttl(self, api_recorder, instant_join, job_context):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)

        request = api_recorder.requests[0]
        assert request.avatar_ids == [ADA_ID]
        assert request.livekit_url == self.LK_URL
        claims = self._decode_jwt(request.lk_token)
        assert claims["kind"] == "agent"
        assert claims["video"]["roomJoin"] is True
        assert claims["video"]["room"] == room.name
        assert claims["video"]["canPublish"] and claims["video"]["canSubscribe"]
        assert claims["video"]["canPublishData"] is True
        assert (
            claims["attributes"][ATTRIBUTE_PUBLISH_ON_BEHALF]
            == job_context.local_participant_identity
        )
        assert claims["exp"] - claims["nbf"] == int(TOKEN_TTL.total_seconds())
        assert self.LK_SECRET not in request.lk_token
        assert self.LK_SECRET not in repr(session)
        assert self.LK_SECRET not in repr(session._config)

        await session.aclose()

    @pytest.mark.parametrize(
        ("kwargs", "identity", "name"),
        [
            ({}, AVATAR_IDENTITY, AVATAR_NAME),
            (
                {"avatar_participant_identity": "avatar-host", "avatar_participant_name": "Host"},
                "avatar-host",
                "Host",
            ),
        ],
    )
    async def test_participant_identity_and_name_reach_token_and_audio(
        self, api_recorder, instant_join, kwargs, identity, name
    ):
        agent = self._FakeAgentSession()
        session = self._session(**kwargs)
        await self._start(session, self._FakeRoom(), agent)

        claims = self._decode_jwt(api_recorder.requests[0].lk_token)
        assert (claims["sub"], claims["name"]) == (identity, name)
        assert session.avatar_identity == identity
        assert agent.output.audio.destination_identity == identity
        assert agent.output.audio.wait_remote_track == rtc.TrackKind.KIND_VIDEO
        await session.aclose()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"avatar_participant_identity": ""},
            {"avatar_participant_identity": "   "},
            {"avatar_participant_name": ""},
            {"avatar_participant_name": "   "},
        ],
    )
    def test_blank_identity_or_name_rejected_at_construction(self, kwargs):
        with pytest.raises(SynthesiaError, match=next(iter(kwargs))):
            self._session(**kwargs)

    async def test_falls_back_to_room_identity_without_a_job_context(
        self, api_recorder, instant_join, monkeypatch
    ):
        monkeypatch.setattr(
            "livekit.plugins.synthesia.avatar.get_job_context", lambda *, required=True: None
        )
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)

        claims = self._decode_jwt(api_recorder.requests[0].lk_token)
        assert claims["attributes"][ATTRIBUTE_PUBLISH_ON_BEHALF] == room.local_participant.identity
        await session.aclose()

    @pytest.mark.parametrize("identity", ["", "   ", None])
    async def test_blank_local_identity_fails_before_launch(
        self, api_recorder, instant_join, job_context, identity
    ):
        job_context.local_participant_identity = identity
        session = self._session()

        with pytest.raises(SynthesiaError, match="local participant"):
            await self._start(session, self._FakeRoom())

        assert api_recorder.requests == []
        await session.aclose()

    @pytest.mark.parametrize(("target", "active"), [(SECOND_ID, SECOND_ID), ("default", ADA_ID)])
    async def test_swap_avatar_sends_rpc_and_returns_active_id(
        self, api_recorder, instant_join, target, active
    ):
        room = self._FakeRoom()
        session = self._multi_session(avatar_participant_identity=self.CUSTOM_IDENTITY)
        await self._start(session, room)
        room.local_participant.rpc_response = json.dumps({"status": "ok", "avatar_id": active})

        assert await session.swap_avatar(target) == active

        call = room.local_participant.rpc_calls[0]
        assert call["destination_identity"] == self.CUSTOM_IDENTITY
        assert call["method"] == "swapAvatar"
        assert json.loads(call["payload"]) == {"avatar_id": target}
        assert call["response_timeout"] is not None
        await session.aclose()

    async def test_swap_avatar_rejects_id_not_in_config(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._multi_session()
        await self._start(session, room)

        with pytest.raises(SynthesiaError) as exc:
            await session.swap_avatar("not-in-the-launch-list")
        assert exc.value.type is ErrorType.UNKNOWN_AVATAR

        assert room.local_participant.rpc_calls == []
        await session.aclose()

    @pytest.mark.parametrize(
        ("response", "match"),
        [({"error": "swap timeout"}, "swap timeout"), ({"status": "weird"}, "weird")],
    )
    async def test_swap_avatar_error_response_raises(
        self, api_recorder, instant_join, response, match
    ):
        room = self._FakeRoom()
        session = self._multi_session()
        await self._start(session, room)
        room.local_participant.rpc_response = json.dumps(response)

        with pytest.raises(SynthesiaError, match=match):
            await session.swap_avatar(SECOND_ID)

        await session.aclose()

    async def test_swap_avatar_rpc_failure_raises_connection_error(
        self, api_recorder, instant_join
    ):
        room = self._FakeRoom()
        session = self._multi_session()
        await self._start(session, room)
        room.local_participant.rpc_error = RuntimeError("rpc transport down")

        with pytest.raises(SynthesiaError) as exc:
            await session.swap_avatar(SECOND_ID)
        assert exc.value.type is ErrorType.CONNECTION

        await session.aclose()

    async def test_swap_avatar_outside_a_started_session_raises(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._multi_session()
        with pytest.raises(SynthesiaError):
            await session.swap_avatar(SECOND_ID)

        await self._start(session, room)
        await session.aclose()
        with pytest.raises(SynthesiaError):
            await session.swap_avatar(SECOND_ID)
        assert room.local_participant.rpc_calls == []

    async def test_swap_avatar_during_teardown_raises(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._multi_session()
        await self._start(session, room)

        # a room disconnect schedules aclose() but the state flips to closed only once it runs
        room._connected = False
        room.fire("disconnected")
        assert session._teardown_task is not None

        with pytest.raises(SynthesiaError, match="started avatar session"):
            await session.swap_avatar(SECOND_ID)

        assert room.local_participant.rpc_calls == []
        await session._teardown_task

    @pytest.mark.parametrize(
        "given", ["internal-host:7880", "tcp://internal-host:7880", "proj.livekit.cloud"]
    )
    async def test_unusable_livekit_url_fails_fast(self, api_recorder, instant_join, given):
        session = self._session()
        with pytest.raises(SynthesiaError, match="livekit_url"):
            await session.start(
                self._FakeAgentSession(),
                self._FakeRoom(),
                livekit_url=given,
                livekit_api_key=self.LK_KEY,
                livekit_api_secret=self.LK_SECRET,
            )
        assert api_recorder.requests == []

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("https://proj.livekit.cloud", "wss://proj.livekit.cloud"),
            ("http://localhost:7880", "ws://localhost:7880"),
            ("wss://proj.livekit.cloud", "wss://proj.livekit.cloud"),
            ("ws://localhost:7880", "ws://localhost:7880"),
        ],
    )
    async def test_livekit_url_normalized_to_ws_scheme(
        self, api_recorder, instant_join, given, expected
    ):
        session = self._session()
        await session.start(
            self._FakeAgentSession(),
            self._FakeRoom(),
            livekit_url=given,
            livekit_api_key=self.LK_KEY,
            livekit_api_secret=self.LK_SECRET,
        )

        assert api_recorder.requests[0].livekit_url == expected
        await session.aclose()

    async def test_livekit_credentials_from_env(self, api_recorder, instant_join, monkeypatch):
        monkeypatch.setenv("LIVEKIT_URL", "https://proj.livekit.cloud")
        monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "env-secret")
        session = self._session()
        await session.start(self._FakeAgentSession(), self._FakeRoom())

        request = api_recorder.requests[0]
        assert request.livekit_url == "wss://proj.livekit.cloud"
        assert self._decode_jwt(request.lk_token)["iss"] == "env-key"
        await session.aclose()

    @pytest.mark.parametrize(
        ("url", "key", "secret"),
        [
            (LK_URL, None, None),
            (LK_URL, "   ", LK_SECRET),
            (LK_URL, LK_KEY, "   "),
            (LK_URL, "   ", "   "),
            ("   ", LK_KEY, LK_SECRET),
        ],
    )
    async def test_missing_or_blank_livekit_settings_fail_before_launch(
        self, api_recorder, monkeypatch, url, key, secret
    ):
        for var in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
            monkeypatch.delenv(var, raising=False)
        session = self._session()
        with pytest.raises(SynthesiaError):
            await session.start(
                self._FakeAgentSession(),
                self._FakeRoom(),
                livekit_url=url,
                **({} if key is None else {"livekit_api_key": key}),
                **({} if secret is None else {"livekit_api_secret": secret}),
            )
        assert api_recorder.requests == []

    @pytest.mark.virtual_time
    async def test_join_timeout_raises_and_tears_down(self, api_recorder, hanging_join):
        room = self._FakeRoom()
        agent = self._FakeAgentSession()
        session = self._session(join_timeout=5.0)

        with pytest.raises(SynthesiaError) as exc:
            await self._start(session, room, agent)
        assert exc.value.type is ErrorType.TIMEOUT

        assert agent.output.audio.closed is True
        assert session._audio_output is None
        assert room.listener_count("disconnected") == 0
        assert agent.listener_count("conversation_item_added") == 0

    async def test_failed_start_tears_down_and_can_be_retried(self, api_recorder, instant_join):
        api_recorder.error = SynthesiaError("bad key", type=ErrorType.AUTH)
        room = self._FakeRoom()
        agent = self._FakeAgentSession()
        session = self._session()

        with pytest.raises(SynthesiaError) as exc:
            await self._start(session, room, agent)
        assert exc.value.type is ErrorType.AUTH
        # the API call fails before the audio output is installed, so there is nothing to close
        assert agent.output.audio is None
        assert agent.listener_count("conversation_item_added") == 0

        api_recorder.error = None
        await self._start(session, room, agent)
        assert len(api_recorder.requests) == 2
        assert agent.output.audio is not None
        await session.aclose()

    async def test_double_start_is_idempotent(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)
        await self._start(session, room)

        assert len(api_recorder.requests) == 1
        await session.aclose()

    async def test_room_disconnect_logs_and_tears_down(self, api_recorder, instant_join, caplog):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)

        with caplog.at_level(logging.INFO, logger="livekit.plugins.synthesia"):
            room.fire("disconnected")
            await session._teardown_task

        assert caplog.text.count("avatar session ended") == 1

    async def test_track_drop_then_disconnect_logs_once(self, api_recorder, instant_join, caplog):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)

        with caplog.at_level(logging.INFO, logger="livekit.plugins.synthesia"):
            publication = type("Pub", (), {"kind": rtc.TrackKind.KIND_VIDEO})()
            participant = type("P", (), {"identity": session.avatar_identity})()
            room.fire("track_unpublished", publication, participant)
            room.fire("disconnected")
            await session._teardown_task

        assert caplog.text.count("avatar left the room unexpectedly") == 1
        assert "avatar session ended" not in caplog.text

    async def test_departure_handlers_key_on_the_avatar_identity(
        self, api_recorder, instant_join, caplog
    ):
        room = self._FakeRoom()
        session = self._session(avatar_participant_identity=self.CUSTOM_IDENTITY)
        await self._start(session, room)

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.synthesia"):
            # a participant on the default identity is a different avatar
            other = type("P", (), {"identity": AVATAR_IDENTITY})()
            video = type("Pub", (), {"kind": rtc.TrackKind.KIND_VIDEO})()
            room.fire("track_unpublished", video, other)
            room.fire("participant_disconnected", other)
            assert caplog.text == ""
            assert session._teardown_task is None

            room.fire("participant_disconnected", type("P", (), {"identity": "avatar-host"})())
            await session._teardown_task

        assert caplog.text.count("avatar left the room unexpectedly") == 1

    async def test_aclose_closes_audio_and_unregisters_handlers(self, api_recorder, instant_join):
        room = self._FakeRoom()
        agent = self._FakeAgentSession()
        session = self._session()
        await self._start(session, room, agent)
        audio = agent.output.audio

        await session.aclose()
        await session.aclose()

        assert audio.closed is True
        assert agent.output.audio is audio
        assert session._audio_output is None
        assert room.listener_count("disconnected") == 0
        assert room.listener_count("track_unpublished") == 0
        assert agent.listener_count("conversation_item_added") == 0

    async def test_aclose_completes_even_if_audio_close_raises(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)

        class _RaisingAudio(self._FakeAudioOutput):
            async def aclose(self):
                raise RuntimeError("audio boom")

        session._audio_output = _RaisingAudio(room, destination_identity=session.avatar_identity)

        with pytest.raises(RuntimeError):
            await session.aclose()

        # a second close must not hang waiting on the first
        await session.aclose()

    async def test_failed_teardown_blocks_restart(self, api_recorder, instant_join, monkeypatch):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)

        async def _boom(self):
            raise RuntimeError("teardown boom")

        monkeypatch.setattr(BaseAvatarSession, "aclose", _boom)

        room.fire("disconnected")
        with pytest.raises(RuntimeError):
            await session._teardown_task

        with pytest.raises(SynthesiaError):
            await self._start(session, room)

    async def test_wait_until_ready(self, api_recorder, instant_join):
        session = self._session()
        with pytest.raises(SynthesiaError, match="not started"):
            await session.wait_until_ready()

        await self._start(session, self._FakeRoom())
        await session.wait_until_ready()
        await session.aclose()

    async def test_background_start_returns_before_the_session_call(
        self, api_recorder, instant_join
    ):
        api_recorder.gate = asyncio.Event()
        room, agent = self._FakeRoom(), self._FakeAgentSession()
        session = self._session()
        await self._start(session, room, agent, wait_for_avatar=False)

        # the sink is installed up front and holds audio until the avatar's video track exists
        sink = agent.output.audio
        assert isinstance(sink, self._FakeAudioOutput)
        assert sink.wait_remote_track == rtc.TrackKind.KIND_VIDEO
        ready = asyncio.create_task(session.wait_until_ready())
        await asyncio.sleep(0)
        assert not ready.done()

        api_recorder.gate.set()
        await ready
        assert agent.output.audio is sink
        assert len(api_recorder.requests) == 1
        assert room.listener_count("disconnected") == 1
        await session.aclose()

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (SynthesiaError("bad key", type=ErrorType.AUTH), ErrorType.AUTH),
            (RuntimeError("socket closed"), None),
        ],
    )
    async def test_background_start_failure_raises_from_wait_until_ready(
        self, api_recorder, instant_join, error, expected
    ):
        api_recorder.error = error
        agent = self._FakeAgentSession()
        session = self._session()
        await self._start(session, self._FakeRoom(), agent, wait_for_avatar=False)
        sink = agent.output.audio

        with pytest.raises(SynthesiaError) as exc:
            await session.wait_until_ready()
        if expected is None:
            assert isinstance(exc.value.__cause__, RuntimeError)
        else:
            assert exc.value.type is expected
        assert sink.closed
        with pytest.raises(SynthesiaError, match="requires a started avatar session"):
            await session.swap_avatar("default")
        await session.aclose()

    async def test_background_start_failure_log_omits_the_message(
        self, api_recorder, instant_join, caplog
    ):
        api_recorder.error = SynthesiaError(
            "echoed: lk-token-secret", type=ErrorType.AUTH, status=401, request_id="req_1"
        )
        session = self._session()
        with caplog.at_level(logging.ERROR, logger="livekit.plugins.synthesia"):
            await self._start(session, self._FakeRoom(), wait_for_avatar=False)
            with pytest.raises(SynthesiaError, match="echoed"):
                await session.wait_until_ready()

        [record] = [r for r in caplog.records if r.getMessage() == "avatar failed to start"]
        assert "lk-token-secret" not in caplog.text
        assert record.exc_info is None
        assert (record.type, record.status, record.request_id) == ("auth", 401, "req_1")
        await session.aclose()

    @pytest.mark.virtual_time
    async def test_background_start_join_timeout(self, api_recorder, hanging_join):
        agent = self._FakeAgentSession()
        session = self._session(join_timeout=5.0)
        await self._start(session, self._FakeRoom(), agent, wait_for_avatar=False)
        sink = agent.output.audio

        with pytest.raises(SynthesiaError) as exc:
            await session.wait_until_ready()
        assert exc.value.type is ErrorType.TIMEOUT
        assert sink.closed
        await session.aclose()

    @pytest.mark.parametrize("let_it_run", [True, False])
    async def test_aclose_cancels_background_start(self, api_recorder, instant_join, let_it_run):
        api_recorder.gate = asyncio.Event()
        agent = self._FakeAgentSession()
        session = self._session()
        await self._start(session, self._FakeRoom(), agent, wait_for_avatar=False)
        if let_it_run:
            await asyncio.sleep(0)

        await session.aclose()

        with pytest.raises(SynthesiaError, match="closed while starting"):
            await asyncio.wait_for(session.wait_until_ready(), timeout=1)
        assert agent.output.audio.closed

    async def test_start_while_background_start_runs_raises(self, api_recorder, instant_join):
        api_recorder.gate = asyncio.Event()
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room, wait_for_avatar=False)

        with pytest.raises(SynthesiaError, match="already in progress"):
            await self._start(session, room)
        await session.aclose()
