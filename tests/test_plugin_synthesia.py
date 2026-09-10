"""Unit tests for the Synthesia interactive-avatar plugin.

Covers plugin registration, config validation, the error taxonomy, the
``SynthesiaAPI`` HTTP client, the ``AvatarSession`` lifecycle, and a
README-mirroring usage example. The Synthesia backend, the LiveKit room, and
the avatar-join wait are faked throughout, so this runs offline.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIError, Plugin
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.plugins import synthesia
from livekit.plugins.synthesia.api import SESSION_PATH, SynthesiaAPI
from livekit.plugins.synthesia.errors import ErrorType, SynthesiaError
from livekit.plugins.synthesia.types import (
    AVATAR_IDENTITY,
    AVATAR_NAME,
    DEFAULT_API_URL,
    DEFAULT_JOIN_TIMEOUT,
    TOKEN_TTL,
    StartSessionRequest,
    StartSessionResponse,
)

pytestmark = [pytest.mark.unit, pytest.mark.plugin("synthesia")]


class TestPluginRegistration:
    def test_plugin_registered(self):
        titles = [p.title for p in Plugin.registered_plugins]
        assert "livekit.plugins.synthesia" in titles


class TestConfig:
    ADA = "03cee7ec-ac90-45ec-8c20-74a399cf3dc4"
    SECOND = "6d999451-039c-4bf2-9b88-c769ac2faa78"

    @staticmethod
    def _config(avatar=None, **kwargs):
        return synthesia.AvatarConfig(avatar_ids=[avatar or TestConfig.ADA], **kwargs)

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        monkeypatch.delenv("SYNTHESIA_API_KEY", raising=False)
        monkeypatch.delenv("SYNTHESIA_API_URL", raising=False)

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("SYNTHESIA_API_KEY", "env-key")
        avatar = synthesia.AvatarSession(self._config())
        assert avatar._config.api_key == "env-key"

    def test_api_key_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("SYNTHESIA_API_KEY", "env-key")
        avatar = synthesia.AvatarSession(self._config(), api_key="arg-key")
        assert avatar._config.api_key == "arg-key"

    def test_api_key_missing_raises(self):
        with pytest.raises(synthesia.SynthesiaError):
            synthesia.AvatarSession(self._config())

    def test_api_url_default(self):
        avatar = synthesia.AvatarSession(self._config(), api_key="k")
        assert avatar._config.api_url == DEFAULT_API_URL

    def test_api_url_from_env(self, monkeypatch):
        monkeypatch.setenv("SYNTHESIA_API_URL", "https://staging.example")
        avatar = synthesia.AvatarSession(self._config(), api_key="k")
        assert avatar._config.api_url == "https://staging.example"

    def test_api_url_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("SYNTHESIA_API_URL", "https://staging.example")
        avatar = synthesia.AvatarSession(self._config(), api_key="k", api_url="https://arg.example")
        assert avatar._config.api_url == "https://arg.example"

    def test_avatar_id_passed_through(self):
        avatar = synthesia.AvatarSession(self._config("my-avatar-id"), api_key="k")
        assert avatar._config.avatar_ids == ("my-avatar-id",)

    def test_avatar_ids_order_preserved(self):
        config = synthesia.AvatarConfig(avatar_ids=[self.ADA, self.SECOND])
        avatar = synthesia.AvatarSession(config, api_key="k")
        assert avatar._config.avatar_ids == (self.ADA, self.SECOND)

    def test_avatar_ids_max_five_accepted(self):
        ids = [f"avatar-{i}" for i in range(5)]
        avatar = synthesia.AvatarSession(synthesia.AvatarConfig(avatar_ids=ids), api_key="k")
        assert avatar._config.avatar_ids == tuple(ids)

    def test_avatar_ids_empty_raises(self):
        with pytest.raises(ValueError, match="between 1 and"):
            synthesia.AvatarConfig(avatar_ids=[])

    def test_avatar_ids_oversize_raises(self):
        with pytest.raises(ValueError, match="between 1 and"):
            synthesia.AvatarConfig(avatar_ids=[f"avatar-{i}" for i in range(6)])

    def test_avatar_ids_bare_string_raises(self):
        with pytest.raises(ValueError, match="list"):
            synthesia.AvatarConfig(avatar_ids="lucas")

    def test_avatar_ids_copied_from_input(self):
        ids = [self.ADA]
        config = synthesia.AvatarConfig(avatar_ids=ids)
        ids.append(self.SECOND)
        avatar = synthesia.AvatarSession(config, api_key="k")
        assert avatar._config.avatar_ids == (self.ADA,)

    def test_join_timeout_default(self):
        avatar = synthesia.AvatarSession(self._config(), api_key="k")
        assert avatar._config.join_timeout == DEFAULT_JOIN_TIMEOUT

    def test_join_timeout_override(self):
        avatar = synthesia.AvatarSession(self._config(), api_key="k", join_timeout=5.0)
        assert avatar._config.join_timeout == 5.0

    def test_config_repr_redacts_api_key(self):
        avatar = synthesia.AvatarSession(self._config(), api_key="super-secret")
        assert "super-secret" not in repr(avatar._config)

    def test_identity_and_provider(self):
        avatar = synthesia.AvatarSession(self._config(), api_key="k")
        assert avatar.avatar_identity == "synthesia-avatar-agent"
        assert avatar.provider == "synthesia"


class TestErrors:
    _NON_RETRYABLE_TYPES = [
        ErrorType.AUTH,
        ErrorType.FEATURE_NOT_IN_PLAN,
        ErrorType.UNKNOWN_AVATAR,
        ErrorType.QUOTA_EXCEEDED,
        ErrorType.INVALID_ROOM_TOKEN,
        ErrorType.LIVEKIT_CREDENTIALS_REJECTED,
        ErrorType.INVALID_SESSION_REQUEST,
    ]

    _RETRYABLE_TYPES = [
        ErrorType.RATE_LIMITED,
        ErrorType.CONCURRENCY_LIMIT,
        ErrorType.TIMEOUT,
        ErrorType.CONNECTION,
    ]

    def test_base_is_api_error(self):
        assert issubclass(SynthesiaError, APIError)

    @pytest.mark.parametrize("error_type", [*_NON_RETRYABLE_TYPES, *_RETRYABLE_TYPES])
    def test_typed_errors_caught_by_base(self, error_type):
        with pytest.raises(SynthesiaError):
            raise SynthesiaError("boom", type=error_type)

    def test_untyped_error_defaults_not_retryable(self):
        assert SynthesiaError("boom").retryable is False

    @pytest.mark.parametrize("error_type", _NON_RETRYABLE_TYPES)
    def test_retryable_defaults_false(self, error_type):
        assert SynthesiaError("boom", type=error_type).retryable is False

    @pytest.mark.parametrize("error_type", _RETRYABLE_TYPES)
    def test_retryable_defaults_true(self, error_type):
        assert SynthesiaError("boom", type=error_type).retryable is True

    def test_retryable_can_be_overridden(self):
        assert SynthesiaError("boom", type=ErrorType.AUTH, retryable=True).retryable is True
        assert SynthesiaError("boom", type=ErrorType.CONNECTION, retryable=False).retryable is False

    def test_rate_limited_carries_retry_after(self):
        err = SynthesiaError("throttled", type=ErrorType.RATE_LIMITED, retry_after=12.5)
        assert err.retry_after == 12.5

    def test_retry_after_defaults_none(self):
        err = SynthesiaError("throttled", type=ErrorType.RATE_LIMITED)
        assert err.retry_after is None

    def test_error_type_defaults_none(self):
        assert SynthesiaError("boom").type is None

    def test_status_and_request_id_default_none(self):
        err = SynthesiaError("boom")
        assert (err.status, err.request_id) == (None, None)

    def test_forwards_status_and_request_id(self):
        err = SynthesiaError(
            "throttled", type=ErrorType.RATE_LIMITED, status=429, request_id="req_1"
        )
        assert (err.status, err.request_id) == (429, "req_1")

    def test_errors_exported_from_package(self):
        assert synthesia.SynthesiaError is SynthesiaError
        assert synthesia.ErrorType is ErrorType


class TestApiClient:
    API_KEY = "sk-secret-key"
    API_URL = "https://api.example"
    NO_SLEEP = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0)
    _REQUEST_ID = "req_8c2f7d1e4b5a46f0a1b2c3d4e5f60718"

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

    def _client(self, session, conn_options=None):
        return SynthesiaAPI(
            api_key=self.API_KEY,
            api_url=self.API_URL,
            session=session,
            conn_options=conn_options or self.NO_SLEEP,
        )

    def _request(self, avatar_ids=("avatar-1",)):
        return StartSessionRequest(
            avatar_ids=list(avatar_ids),
            livekit_url="wss://room.livekit.cloud",
            lk_token="lk-token-secret",
        )

    def _problem(self, code, *, status, title, detail=None, with_code=True, request_id=None):
        return {
            "type": f"https://developers.synthesia.io/errors/{code}",
            "title": title,
            "status": status,
            **({"code": code} if with_code else {}),
            **({"detail": detail} if detail is not None else {}),
            **({"requestId": request_id} if request_id is not None else {}),
        }

    async def test_success_returns_session_id(self):
        session = self._FakeSession([self._FakeResponse(200, body={"session_id": "sess_123"})])
        result = await self._client(session).start_session(self._request())
        assert result == StartSessionResponse(session_id="sess_123")

    async def test_request_body_uses_the_published_wire_names(self):
        created = {"id": "ses_9f3c", "status": "active"}
        session = self._FakeSession([self._FakeResponse(201, body=created)])
        await self._client(session).start_session(self._request(["ada-uuid", "kenji-uuid"]))
        assert session.calls[0]["json"] == {
            "avatarIds": ["av_ada-uuid", "av_kenji-uuid"],
            "livekitUrl": "wss://room.livekit.cloud",
            "livekitToken": "lk-token-secret",
        }

    async def test_avatar_ids_are_not_double_prefixed(self):
        created = {"id": "ses_9f3c", "status": "active"}
        session = self._FakeSession([self._FakeResponse(201, body=created)])
        await self._client(session).start_session(self._request(["av_already-prefixed"]))
        assert session.calls[0]["json"]["avatarIds"] == ["av_already-prefixed"]

    async def test_created_response_yields_the_public_session_id(self):
        created = {"id": "ses_9f3c", "status": "active"}
        session = self._FakeSession([self._FakeResponse(201, body=created)])
        result = await self._client(session).start_session(self._request())
        assert result == StartSessionResponse(session_id="ses_9f3c")

    async def test_problem_json_detail_becomes_the_error_message(self):
        body = self._problem(
            "validation_error",
            status=400,
            title="Invalid request",
            detail="The supplied livekitToken cannot be used to join the room.",
        )
        session = self._FakeSession(
            [self._FakeResponse(400, body=body, content_type="application/problem+json")]
        )
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert str(exc.value) == "The supplied livekitToken cannot be used to join the room."
        assert exc.value.type is ErrorType.INVALID_SESSION_REQUEST
        assert exc.value.body == body

    async def test_problem_body_without_code_falls_back_to_status(self):
        body = self._problem("not_found", status=404, title="Not found", with_code=False)
        session = self._FakeSession(
            [self._FakeResponse(404, body=body, content_type="application/problem+json")]
        )
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert exc.value.type is ErrorType.UNKNOWN_AVATAR

    async def test_unmapped_problem_code_is_untyped(self):
        body = self._problem("not_found", status=404, title="Not found")
        session = self._FakeSession(
            [self._FakeResponse(404, body=body, content_type="application/problem+json")]
        )
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert exc.value.type is None
        assert str(exc.value) == "Not found"

    @pytest.mark.parametrize("detail", [None, [{"msg": "x"}]], ids=["absent", "not-a-string"])
    async def test_problem_title_used_when_detail_is_unusable(self, detail):
        body = self._problem("not_found", status=404, title="Not found", detail=detail)
        session = self._FakeSession(
            [self._FakeResponse(404, body=body, content_type="application/problem+json")]
        )
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert str(exc.value) == "Not found"

    async def test_errors_carry_status_and_request_id(self):
        body = self._problem("forbidden", status=403, title="Forbidden", request_id="req_1")
        session = self._FakeSession(
            [self._FakeResponse(403, body=body, content_type="application/problem+json")]
        )
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert exc.value.type is ErrorType.AUTH
        assert (exc.value.status, exc.value.request_id) == (403, "req_1")

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
    async def test_problem_type_maps_to_error_type(self, code, status, expected):
        body = self._problem(code, status=status, title="t", detail="d")
        session = self._FakeSession(
            [self._FakeResponse(status, body=body, content_type="application/problem+json")]
        )
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert exc.value.type is expected

    @pytest.mark.parametrize(
        ("code", "expected", "headers", "retry_after"),
        [
            ("rate_limited", ErrorType.RATE_LIMITED, {"Retry-After": "12"}, 12.0),
            ("concurrency_limit", ErrorType.CONCURRENCY_LIMIT, {"Retry-After": "7"}, 7.0),
            ("concurrency_limit", ErrorType.CONCURRENCY_LIMIT, {}, None),
        ],
    )
    async def test_problem_429_reads_retry_after_when_sent(
        self, code, expected, headers, retry_after
    ):
        body = self._problem(code, status=429, title="t", detail="d")
        session = self._FakeSession(
            [
                self._FakeResponse(
                    429,
                    body=body,
                    headers=headers,
                    content_type="application/problem+json",
                )
            ]
        )
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert exc.value.type is expected
        assert exc.value.retry_after == retry_after

    async def test_session_url_matches_the_published_route(self):
        session = self._FakeSession([self._FakeResponse(200, body={"session_id": "sess_123"})])
        client = SynthesiaAPI(
            api_key=self.API_KEY,
            api_url="https://developers.example",
            session=session,
        )
        await client.start_session(self._request())
        assert (
            session.calls[0]["url"] == "https://developers.example/api/interactive-avatars/sessions"
        )

    async def test_request_carries_auth_header_and_payload(self):
        session = self._FakeSession([self._FakeResponse(200, body={"session_id": "sess_123"})])
        await self._client(session).start_session(self._request())
        call = session.calls[0]
        assert call["url"] == self.API_URL + SESSION_PATH
        assert call["headers"]["Authorization"] == self.API_KEY
        assert call["json"] == {
            "avatarIds": ["av_avatar-1"],
            "livekitUrl": "wss://room.livekit.cloud",
            "livekitToken": "lk-token-secret",
        }

    async def test_request_payload_carries_avatar_ids_in_order(self):
        session = self._FakeSession([self._FakeResponse(200, body={"session_id": "sess_123"})])
        await self._client(session).start_session(self._request(["lead", "second", "third"]))
        assert session.calls[0]["json"]["avatarIds"] == ["av_lead", "av_second", "av_third"]

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
    async def test_error_code_maps_to_error_type(self, code, status, expected):
        session = self._FakeSession(
            [self._FakeResponse(status, body={"error": {"code": code, "message": "nope"}})]
        )
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert exc.value.type is expected

    async def test_livekit_credentials_rejection_is_not_an_auth_error(self):
        body = {
            "error": "LiveKitCredentialsRejectedError",
            "code": "invalid_livekit_credentials",
            "context": {"lk_token": ["LiveKit rejected the supplied credentials"]},
        }
        session = self._FakeSession([self._FakeResponse(401, body=body)])
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert exc.value.type is ErrorType.LIVEKIT_CREDENTIALS_REJECTED

    async def test_bare_401_without_a_code_is_still_an_auth_error(self):
        session = self._FakeSession([self._FakeResponse(401, body={"message": "Unauthorized"})])
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert exc.value.type is ErrorType.AUTH

    async def test_unmapped_error_code_carries_backend_body(self):
        body = {"error": {"code": "teapot", "message": "avatar_ids is required"}}
        session = self._FakeSession([self._FakeResponse(400, body=body)])
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert exc.value.type is None
        assert exc.value.body == body

    async def test_flat_error_body_surfaces_context_message(self):
        body = {"error": "Forbidden", "context": "User is not authenticated"}
        session = self._FakeSession([self._FakeResponse(403, body=body)])
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert str(exc.value) == "User is not authenticated"
        assert exc.value.type is ErrorType.AUTH
        assert exc.value.body == body

    async def test_flat_string_error_used_when_no_context(self):
        body = {"error": "Forbidden"}
        session = self._FakeSession([self._FakeResponse(403, body=body)])
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert str(exc.value) == "Forbidden"
        assert exc.value.type is ErrorType.AUTH

    async def test_dict_context_renders_readable_message(self):
        body = {
            "code": "validation_error",
            "context": {"livekit_url": ["Must be a wss:// URL"]},
            "error": "InvalidSessionRequestError",
        }
        session = self._FakeSession([self._FakeResponse(400, body=body)])
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        text = str(exc.value)
        assert "livekit_url" in text
        assert "Must be a wss:// URL" in text
        assert exc.value.body == body

    def test_error_str_never_raises_on_non_string_message(self):
        exc = SynthesiaError({"context": ["boom"]})
        assert "boom" in str(exc)

    async def test_terminal_error_is_not_retried(self):
        session = self._FakeSession(
            [self._FakeResponse(401, body={"error": {"code": "unauthorized"}})]
        )
        with pytest.raises(SynthesiaError):
            await self._client(session).start_session(self._request())
        assert len(session.calls) == 1

    async def test_rate_limited_retry_after_from_header(self):
        session = self._FakeSession(
            [
                self._FakeResponse(
                    429,
                    body={"error": {"code": "rate_limited"}},
                    headers={"Retry-After": "7"},
                )
            ]
        )
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert exc.value.retry_after == 7.0

    async def test_rate_limited_retry_after_from_body(self):
        session = self._FakeSession(
            [self._FakeResponse(429, body={"error": {"code": "rate_limited"}, "retry_after": 3.5})]
        )
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert exc.value.retry_after == 3.5

    async def test_rate_limited_retry_after_from_nested_error(self):
        session = self._FakeSession(
            [self._FakeResponse(429, body={"error": {"code": "rate_limited", "retry_after": 9}})]
        )
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert exc.value.retry_after == 9.0

    async def test_5xx_surfaces_the_api_detail_and_request_id(self):
        body = self._problem(
            "service_unavailable",
            status=503,
            title="Service unavailable",
            detail="The session could not be started. Retry.",
            request_id=self._REQUEST_ID,
        )
        session = self._FakeSession(
            [self._FakeResponse(503, body=body, content_type="application/problem+json")]
        )
        one_attempt = dataclasses.replace(self.NO_SLEEP, max_retry=0)
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session, one_attempt).start_session(self._request())
        assert str(exc.value) == (
            "could not start a Synthesia session; last attempt: HTTP 503: "
            f"The session could not be started. Retry. (request {self._REQUEST_ID})"
        )
        assert exc.value.type is ErrorType.CONNECTION
        assert exc.value.body == body
        assert (exc.value.status, exc.value.request_id) == (503, self._REQUEST_ID)

    async def test_5xx_legacy_body_message_is_kept(self):
        body = {"error": {"code": "internal", "message": "upstream is down"}}
        session = self._FakeSession([self._FakeResponse(500, body=body)])
        one_attempt = dataclasses.replace(self.NO_SLEEP, max_retry=0)
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session, one_attempt).start_session(self._request())
        assert str(exc.value) == (
            "could not start a Synthesia session; last attempt: HTTP 500: upstream is down"
        )

    async def test_5xx_on_every_attempt_names_the_last_status(self):
        session = self._FakeSession(
            [self._FakeResponse(502) for _ in range(self.NO_SLEEP.max_retry + 1)]
        )
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert len(session.calls) == self.NO_SLEEP.max_retry + 1
        assert str(exc.value) == (
            f"could not start a Synthesia session after {self.NO_SLEEP.max_retry + 1} attempts;"
            " last attempt: HTTP 502"
        )
        assert exc.value.body is None
        assert exc.value.status == 502

    async def test_5xx_whose_body_fails_to_arrive_is_still_a_5xx(self):
        class _TruncatedResponse(self._FakeResponse):
            async def json(self, *, content_type=None):
                raise aiohttp.ClientPayloadError("truncated")

        two_attempts = dataclasses.replace(self.NO_SLEEP, max_retry=1)
        session = self._FakeSession([_TruncatedResponse(503) for _ in range(2)])
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session, two_attempts).start_session(self._request())
        assert str(exc.value) == (
            "could not start a Synthesia session after 2 attempts; last attempt: HTTP 503"
        )
        assert exc.value.status == 503
        assert isinstance(exc.value.__cause__, aiohttp.ClientPayloadError)

    async def test_last_attempt_decides_the_message_when_outcomes_mix(self):
        body = self._problem("service_unavailable", status=503, title="Service unavailable")
        answered = self._FakeResponse(503, body=body, content_type="application/problem+json")
        two_attempts = dataclasses.replace(self.NO_SLEEP, max_retry=1)

        session = self._FakeSession([aiohttp.ClientError(), answered])
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session, two_attempts).start_session(self._request())
        assert str(exc.value) == (
            "could not start a Synthesia session after 2 attempts;"
            " last attempt: HTTP 503: Service unavailable"
        )
        assert (exc.value.body, exc.value.status) == (body, 503)

        session = self._FakeSession([answered, aiohttp.ClientError()])
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session, two_attempts).start_session(self._request())
        assert str(exc.value) == (
            "could not start a Synthesia session after 2 attempts; last attempt: connection error"
        )
        assert (exc.value.body, exc.value.status) == (None, None)
        assert isinstance(exc.value.__cause__, aiohttp.ClientError)

    async def test_transport_error_retries_then_connection_error(self):
        session = self._FakeSession(
            [aiohttp.ClientError() for _ in range(self.NO_SLEEP.max_retry + 1)]
        )
        with pytest.raises(SynthesiaError) as exc:
            await self._client(session).start_session(self._request())
        assert exc.value.type is ErrorType.CONNECTION
        assert len(session.calls) == self.NO_SLEEP.max_retry + 1

    async def test_retry_count_honors_conn_options(self):
        conn = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0, max_retry=1)
        session = self._FakeSession([aiohttp.ClientError() for _ in range(2)])
        with pytest.raises(SynthesiaError):
            await self._client(session, conn).start_session(self._request())
        assert len(session.calls) == 2

    async def test_recovers_after_transient_failure(self):
        session = self._FakeSession(
            [self._FakeResponse(503), self._FakeResponse(200, body={"session_id": "sess_ok"})]
        )
        result = await self._client(session).start_session(self._request())
        assert result.session_id == "sess_ok"
        assert len(session.calls) == 2

    async def test_malformed_success_body_raises(self):
        session = self._FakeSession([self._FakeResponse(200, body={"unexpected": True})])
        with pytest.raises(SynthesiaError):
            await self._client(session).start_session(self._request())

    def test_request_repr_redacts_token(self):
        text = repr(self._request())
        assert "lk-token-secret" not in text

    async def test_secrets_never_logged(self, caplog):
        caplog.set_level(logging.DEBUG, logger="livekit.plugins.synthesia")
        ok = self._FakeSession([self._FakeResponse(200, body={"session_id": "sess_ok"})])
        await self._client(ok).start_session(self._request())
        bad = self._FakeSession([self._FakeResponse(401, body={"error": {"code": "unauthorized"}})])
        with pytest.raises(SynthesiaError):
            await self._client(bad).start_session(self._request())
        assert self.API_KEY not in caplog.text
        assert "lk-token-secret" not in caplog.text


class TestAvatarSession:
    LK_URL = "wss://dev.livekit.cloud"
    LK_KEY = "lk-api-key"
    LK_SECRET = "lk-api-secret-never-leaks"
    ADA_ID = "03cee7ec-ac90-45ec-8c20-74a399cf3dc4"
    SECOND_ID = "6d999451-039c-4bf2-9b88-c769ac2faa78"
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

    class _FakeRoom:
        def __init__(self, name="dev-room", connected=True):
            self.name = name
            self.local_participant = TestAvatarSession._FakeLocalParticipant()
            self.remote_participants = {}
            self._connected = connected
            self._handlers = {}

        def isconnected(self):
            return self._connected

        def on(self, event, handler):
            self._handlers.setdefault(event, []).append(handler)

        def off(self, event, handler):
            if handler in self._handlers.get(event, []):
                self._handlers[event].remove(handler)

        def fire(self, event, *args):
            for handler in list(self._handlers.get(event, [])):
                handler(*args)

        def listener_count(self, event):
            return len(self._handlers.get(event, []))

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

    class _FakeAgentSession:
        def __init__(self):
            self._started = False
            self.output = TestAvatarSession._FakeOutput()
            self._handlers = {}

        def on(self, event, handler):
            self._handlers.setdefault(event, []).append(handler)

        def off(self, event, handler):
            if handler in self._handlers.get(event, []):
                self._handlers[event].remove(handler)

        def emit(self, *args, **kwargs):
            pass

        def listener_count(self, event):
            return len(self._handlers.get(event, []))

    class _Recorder:
        def __init__(self):
            self.requests = []
            self.init_kwargs = None
            self.error = None
            self.response = StartSessionResponse(session_id="sess_123")

    class _RecorderAPI:
        def __init__(self, recorder):
            self._rec = recorder

        async def start_session(self, request, *, conn_options=None):
            self._rec.requests.append(request)
            if self._rec.error is not None:
                raise self._rec.error
            return self._rec.response

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

    @pytest.fixture
    def api_recorder(self, monkeypatch):
        rec = self._Recorder()

        def factory(**kwargs):
            rec.init_kwargs = kwargs
            return self._RecorderAPI(rec)

        monkeypatch.setattr("livekit.plugins.synthesia.avatar.SynthesiaAPI", factory)
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

    def _session(self, avatar=None, **kwargs):
        return synthesia.AvatarSession(
            synthesia.AvatarConfig(avatar_ids=[avatar or self.ADA_ID]), api_key="syn-key", **kwargs
        )

    def _multi_session(self):
        return synthesia.AvatarSession(
            synthesia.AvatarConfig(avatar_ids=[self.ADA_ID, self.SECOND_ID]), api_key="syn-key"
        )

    async def _start(self, session, room, agent=None):
        await session.start(
            agent or self._FakeAgentSession(),
            room,
            livekit_url=self.LK_URL,
            livekit_api_key=self.LK_KEY,
            livekit_api_secret=self.LK_SECRET,
        )

    @staticmethod
    def _decode_jwt(token):
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))

    async def test_token_grants_attribute_and_ttl(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)

        request = api_recorder.requests[0]
        assert request.avatar_ids == [self.ADA_ID]
        assert request.livekit_url == self.LK_URL
        claims = self._decode_jwt(request.lk_token)
        assert claims["sub"] == session.avatar_identity
        assert claims["kind"] == "agent"
        assert claims["video"]["roomJoin"] is True
        assert claims["video"]["room"] == room.name
        assert claims["video"]["canPublish"] and claims["video"]["canSubscribe"]
        assert claims["video"]["canPublishData"] is True
        assert claims["attributes"][ATTRIBUTE_PUBLISH_ON_BEHALF] == room.local_participant.identity
        assert claims["exp"] - claims["nbf"] == int(TOKEN_TTL.total_seconds())

        await session.aclose()

    @pytest.mark.parametrize("identity", ["", "   ", None])
    async def test_blank_local_identity_fails_before_launch(
        self, api_recorder, instant_join, identity
    ):
        room = self._FakeRoom()
        room.local_participant.identity = identity
        session = self._session()

        with pytest.raises(SynthesiaError, match="local participant"):
            await self._start(session, room)

        assert api_recorder.requests == []
        await session.aclose()

    async def test_configured_avatar_reaches_request(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._session(avatar=self.ADA_ID)
        await self._start(session, room)

        assert api_recorder.requests[0].avatar_ids == [self.ADA_ID]
        await session.aclose()

    async def test_swap_avatar_sends_rpc_and_returns_active_id(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._multi_session()
        await self._start(session, room)
        room.local_participant.rpc_response = json.dumps(
            {"status": "ok", "avatar_id": self.SECOND_ID}
        )

        active = await session.swap_avatar(self.SECOND_ID)

        assert active == self.SECOND_ID
        call = room.local_participant.rpc_calls[0]
        assert call["destination_identity"] == session.avatar_identity
        assert call["method"] == "swapAvatar"
        assert json.loads(call["payload"]) == {"avatar_id": self.SECOND_ID}
        assert call["response_timeout"] is not None
        await session.aclose()

    async def test_swap_avatar_accepts_default(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._multi_session()
        await self._start(session, room)
        room.local_participant.rpc_response = json.dumps({"status": "ok", "avatar_id": self.ADA_ID})

        active = await session.swap_avatar("default")

        assert active == self.ADA_ID
        assert json.loads(room.local_participant.rpc_calls[0]["payload"]) == {
            "avatar_id": "default"
        }
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

    async def test_swap_avatar_worker_error_raises(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._multi_session()
        await self._start(session, room)
        room.local_participant.rpc_response = json.dumps({"error": "swap timeout"})

        with pytest.raises(SynthesiaError, match="swap timeout"):
            await session.swap_avatar(self.SECOND_ID)

        await session.aclose()

    async def test_swap_avatar_during_teardown_raises(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._multi_session()
        await self._start(session, room)

        # A room disconnect schedules the aclose() task but _state only flips to
        # CLOSED once that task runs; a swap in that window must be rejected.
        room._connected = False
        room.fire("disconnected")
        assert session._teardown_task is not None

        with pytest.raises(SynthesiaError, match="started avatar session"):
            await session.swap_avatar(self.SECOND_ID)

        assert room.local_participant.rpc_calls == []
        await session._teardown_task

    async def test_swap_avatar_unrecognized_response_carries_raw(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._multi_session()
        await self._start(session, room)
        room.local_participant.rpc_response = json.dumps({"status": "weird"})

        with pytest.raises(SynthesiaError, match="weird"):
            await session.swap_avatar(self.SECOND_ID)

        await session.aclose()

    async def test_swap_avatar_rpc_failure_raises_connection_error(
        self, api_recorder, instant_join
    ):
        room = self._FakeRoom()
        session = self._multi_session()
        await self._start(session, room)
        room.local_participant.rpc_error = RuntimeError("rpc transport down")

        with pytest.raises(SynthesiaError) as exc:
            await session.swap_avatar(self.SECOND_ID)
        assert exc.value.type is ErrorType.CONNECTION

        await session.aclose()

    async def test_swap_avatar_before_start_raises(self):
        session = self._multi_session()
        with pytest.raises(SynthesiaError):
            await session.swap_avatar(self.SECOND_ID)

    async def test_swap_avatar_after_close_raises(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._multi_session()
        await self._start(session, room)
        await session.aclose()

        with pytest.raises(SynthesiaError):
            await session.swap_avatar(self.SECOND_ID)
        assert room.local_participant.rpc_calls == []

    @pytest.mark.parametrize(
        "given", ["internal-host:7880", "tcp://internal-host:7880", "proj.livekit.cloud"]
    )
    async def test_unusable_livekit_url_fails_fast(self, api_recorder, instant_join, given):
        room = self._FakeRoom()
        session = self._session()
        with pytest.raises(SynthesiaError, match="livekit_url"):
            await session.start(
                self._FakeAgentSession(),
                room,
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
        room = self._FakeRoom()
        session = self._session()
        await session.start(
            self._FakeAgentSession(),
            room,
            livekit_url=given,
            livekit_api_key=self.LK_KEY,
            livekit_api_secret=self.LK_SECRET,
        )

        assert api_recorder.requests[0].livekit_url == expected
        await session.aclose()

    async def test_livekit_url_from_env_normalized_to_ws_scheme(
        self, api_recorder, instant_join, monkeypatch
    ):
        monkeypatch.setenv("LIVEKIT_URL", "https://proj.livekit.cloud")
        monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "env-secret")
        room = self._FakeRoom()
        session = self._session()
        await session.start(self._FakeAgentSession(), room)

        assert api_recorder.requests[0].livekit_url == "wss://proj.livekit.cloud"
        await session.aclose()

    async def test_secret_never_serialized(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)

        token = api_recorder.requests[0].lk_token
        assert self.LK_SECRET not in token
        assert self.LK_SECRET not in repr(session)
        assert self.LK_SECRET not in repr(session._config)
        await session.aclose()

    async def test_audio_output_wired_to_avatar(self, api_recorder, instant_join):
        room = self._FakeRoom()
        agent = self._FakeAgentSession()
        session = self._session()
        await self._start(session, room, agent)

        assert agent.output.audio.destination_identity == session.avatar_identity
        assert agent.output.audio.wait_remote_track == rtc.TrackKind.KIND_VIDEO
        await session.aclose()

    async def test_join_timeout_raises_and_tears_down(self, api_recorder, hanging_join):
        room = self._FakeRoom()
        agent = self._FakeAgentSession()
        session = self._session(join_timeout=0.05)

        with pytest.raises(SynthesiaError) as exc:
            await self._start(session, room, agent)
        assert exc.value.type is ErrorType.TIMEOUT

        assert agent.output.audio.closed is True
        assert session._audio_output is None
        assert room.listener_count("disconnected") == 0
        assert agent.listener_count("conversation_item_added") == 0

    async def test_start_failure_raises_mapped_and_tears_down(self, api_recorder, instant_join):
        api_recorder.error = SynthesiaError("bad key", type=ErrorType.AUTH)
        room = self._FakeRoom()
        agent = self._FakeAgentSession()
        session = self._session()

        with pytest.raises(SynthesiaError) as exc:
            await self._start(session, room, agent)
        assert exc.value.type is ErrorType.AUTH

        # The Synthesia API call fails before the audio output is ever
        # installed, so there is nothing to close.
        assert agent.output.audio is None
        assert session._audio_output is None
        assert agent.listener_count("conversation_item_added") == 0

    async def test_mid_session_track_drop_logs_and_tears_down(
        self, api_recorder, instant_join, caplog
    ):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.synthesia"):
            publication = type("Pub", (), {"kind": rtc.TrackKind.KIND_VIDEO})()
            participant = type("P", (), {"identity": session.avatar_identity})()
            room.fire("track_unpublished", publication, participant)
            await session._teardown_task

        assert caplog.text.count("avatar left the room unexpectedly") == 1
        assert room.listener_count("track_unpublished") == 0

    async def test_participant_disconnect_logs_and_tears_down(
        self, api_recorder, instant_join, caplog
    ):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.synthesia"):
            participant = type("P", (), {"identity": session.avatar_identity})()
            room.fire("participant_disconnected", participant)
            await session._teardown_task

        assert caplog.text.count("avatar left the room unexpectedly") == 1

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

    async def test_retry_after_failed_start(self, api_recorder, instant_join):
        room = self._FakeRoom()
        agent = self._FakeAgentSession()
        session = self._session()

        api_recorder.error = SynthesiaError("transient", type=ErrorType.AUTH)
        with pytest.raises(SynthesiaError):
            await self._start(session, room, agent)

        api_recorder.error = None
        await self._start(session, room, agent)

        assert len(api_recorder.requests) == 2
        assert agent.output.audio is not None
        await session.aclose()

    async def test_room_disconnect_logs_and_tears_down(self, api_recorder, instant_join, caplog):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)

        with caplog.at_level(logging.INFO, logger="livekit.plugins.synthesia"):
            room.fire("disconnected")
            await session._teardown_task

        assert caplog.text.count("avatar session ended") == 1

    async def test_teardown_closes_audio_output(self, api_recorder, instant_join):
        room = self._FakeRoom()
        agent = self._FakeAgentSession()
        session = self._session()
        await self._start(session, room, agent)

        audio = agent.output.audio
        await session.aclose()

        assert audio.closed is True
        assert agent.output.audio is audio
        assert session._audio_output is None

    async def test_aclose_sets_close_done_even_if_audio_close_raises(
        self, api_recorder, instant_join
    ):
        room = self._FakeRoom()
        agent = self._FakeAgentSession()
        session = self._session()
        await self._start(session, room, agent)

        class _RaisingAudio(self._FakeAudioOutput):
            async def aclose(self):
                raise RuntimeError("audio boom")

        session._audio_output = _RaisingAudio(room, destination_identity=session.avatar_identity)

        with pytest.raises(RuntimeError):
            await session.aclose()

        assert session._close_done.is_set()
        # A second close must not hang waiting on the event.
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

    async def test_double_start_is_idempotent(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)
        await self._start(session, room)

        assert len(api_recorder.requests) == 1
        await session.aclose()

    async def test_double_close_is_idempotent(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)
        await session.aclose()
        await session.aclose()

    async def test_missing_livekit_credentials_raises(self, api_recorder):
        session = self._session()
        with pytest.raises(SynthesiaError):
            await session.start(self._FakeAgentSession(), self._FakeRoom(), livekit_url=self.LK_URL)

    @pytest.mark.parametrize(
        ("key", "secret"),
        [("   ", "lk-api-secret-never-leaks"), ("lk-api-key", "   "), ("   ", "   ")],
    )
    async def test_blank_livekit_credentials_fail_before_launch(self, api_recorder, key, secret):
        session = self._session()
        with pytest.raises(SynthesiaError, match="LiveKit"):
            await session.start(
                self._FakeAgentSession(),
                self._FakeRoom(),
                livekit_url=self.LK_URL,
                livekit_api_key=key,
                livekit_api_secret=secret,
            )
        assert api_recorder.requests == []

    async def test_blank_livekit_url_fails_before_launch(self, api_recorder):
        session = self._session()
        with pytest.raises(SynthesiaError, match="LiveKit"):
            await session.start(
                self._FakeAgentSession(),
                self._FakeRoom(),
                livekit_url="   ",
                livekit_api_key=self.LK_KEY,
                livekit_api_secret=self.LK_SECRET,
            )
        assert api_recorder.requests == []

    async def test_livekit_credentials_from_env(self, api_recorder, instant_join, monkeypatch):
        monkeypatch.setenv("LIVEKIT_URL", self.LK_URL)
        monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "env-secret")
        room = self._FakeRoom()
        session = self._session()
        await session.start(self._FakeAgentSession(), room)

        claims = self._decode_jwt(api_recorder.requests[0].lk_token)
        assert claims["iss"] == "env-key"
        await session.aclose()

    async def test_custom_identity_in_minted_token(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._session(avatar_participant_identity=self.CUSTOM_IDENTITY)
        await self._start(session, room)

        claims = self._decode_jwt(api_recorder.requests[0].lk_token)
        assert claims["sub"] == self.CUSTOM_IDENTITY
        assert session.avatar_identity == self.CUSTOM_IDENTITY
        await session.aclose()

    async def test_custom_identity_drives_audio_destination(self, api_recorder, instant_join):
        room = self._FakeRoom()
        agent = self._FakeAgentSession()
        session = self._session(avatar_participant_identity=self.CUSTOM_IDENTITY)
        await self._start(session, room, agent)

        assert agent.output.audio.destination_identity == self.CUSTOM_IDENTITY
        await session.aclose()

    async def test_custom_identity_drives_swap_rpc(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = synthesia.AvatarSession(
            synthesia.AvatarConfig(avatar_ids=[self.ADA_ID, self.SECOND_ID]),
            api_key="syn-key",
            avatar_participant_identity=self.CUSTOM_IDENTITY,
        )
        await self._start(session, room)
        room.local_participant.rpc_response = json.dumps(
            {"status": "ok", "avatar_id": self.SECOND_ID}
        )

        await session.swap_avatar(self.SECOND_ID)

        assert room.local_participant.rpc_calls[0]["destination_identity"] == self.CUSTOM_IDENTITY
        await session.aclose()

    async def test_disconnect_handler_keys_on_custom_identity(
        self, api_recorder, instant_join, caplog
    ):
        room = self._FakeRoom()
        session = self._session(avatar_participant_identity=self.CUSTOM_IDENTITY)
        await self._start(session, room)

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.synthesia"):
            # A participant using the default identity is a different avatar; ignore it.
            default_p = type("P", (), {"identity": AVATAR_IDENTITY})()
            room.fire("participant_disconnected", default_p)
            assert caplog.text == ""
            assert session._teardown_task is None

            # This avatar leaving under its own identity is a crash.
            custom_p = type("P", (), {"identity": self.CUSTOM_IDENTITY})()
            room.fire("participant_disconnected", custom_p)
            await session._teardown_task

        assert caplog.text.count("avatar left the room unexpectedly") == 1

    async def test_track_handler_ignores_default_identity(self, api_recorder, instant_join, caplog):
        room = self._FakeRoom()
        session = self._session(avatar_participant_identity=self.CUSTOM_IDENTITY)
        await self._start(session, room)

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.synthesia"):
            publication = type("Pub", (), {"kind": rtc.TrackKind.KIND_VIDEO})()
            default_p = type("P", (), {"identity": AVATAR_IDENTITY})()
            room.fire("track_unpublished", publication, default_p)

        assert caplog.text == ""
        assert session._teardown_task is None
        await session.aclose()

    async def test_default_identity_when_omitted(self, api_recorder, instant_join):
        room = self._FakeRoom()
        agent = self._FakeAgentSession()
        session = self._session()
        await self._start(session, room, agent)

        assert session.avatar_identity == AVATAR_IDENTITY
        claims = self._decode_jwt(api_recorder.requests[0].lk_token)
        assert claims["sub"] == AVATAR_IDENTITY
        assert agent.output.audio.destination_identity == AVATAR_IDENTITY
        await session.aclose()

    @pytest.mark.parametrize("identity", ["", "   "])
    async def test_blank_identity_rejected_at_construction(self, identity):
        with pytest.raises(SynthesiaError, match="avatar_participant_identity"):
            self._session(avatar_participant_identity=identity)

    @pytest.mark.parametrize("name", ["", "   "])
    async def test_blank_name_rejected_at_construction(self, name):
        with pytest.raises(SynthesiaError, match="avatar_participant_name"):
            self._session(avatar_participant_name=name)

    async def test_custom_name_in_minted_token(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._session(avatar_participant_name="Host Avatar")
        await self._start(session, room)

        claims = self._decode_jwt(api_recorder.requests[0].lk_token)
        assert claims["name"] == "Host Avatar"
        await session.aclose()

    async def test_default_name_when_omitted(self, api_recorder, instant_join):
        room = self._FakeRoom()
        session = self._session()
        await self._start(session, room)

        claims = self._decode_jwt(api_recorder.requests[0].lk_token)
        assert claims["name"] == AVATAR_NAME
        await session.aclose()

    async def test_teardown_unregisters_handlers(self, api_recorder, instant_join):
        room = self._FakeRoom()
        agent = self._FakeAgentSession()
        session = self._session()
        await self._start(session, room, agent)
        await session.aclose()

        assert room.listener_count("disconnected") == 0
        assert room.listener_count("track_unpublished") == 0
        assert agent.listener_count("conversation_item_added") == 0
        assert agent.output.audio.closed is True
        assert session._audio_output is None


class TestUsageExample:
    """Developer-facing usage, mirroring the README example as living documentation.

    Constructs ``AvatarSession``, starts it before the ``AgentSession``, and
    observes the lifecycle logs. The call sequence and public surface here are
    exactly what a developer writes.
    """

    class _Output:
        def __init__(self):
            self.audio = None

        def replace_audio_tail(self, sink):
            self.audio = sink

    class _AgentSession:
        """Stand-in for ``livekit.agents.AgentSession``."""

        def __init__(self):
            self._started = False
            self.output = TestUsageExample._Output()
            self._handlers = {}

        def on(self, event, handler):
            self._handlers.setdefault(event, []).append(handler)

        def off(self, event, handler):
            if handler in self._handlers.get(event, []):
                self._handlers[event].remove(handler)

        def emit(self, *args, **kwargs):
            pass

        async def start(self, **kwargs):
            self._started = True

    class _LocalParticipant:
        identity = "my-voice-agent"

    class _Room:
        """Stand-in for the developer's connected ``rtc.Room``."""

        name = "my-room"

        def __init__(self):
            self._handlers = {}
            self.local_participant = TestUsageExample._LocalParticipant()

        def isconnected(self):
            return True

        def on(self, event, handler):
            self._handlers.setdefault(event, []).append(handler)

        def off(self, event, handler):
            if handler in self._handlers.get(event, []):
                self._handlers[event].remove(handler)

        def fire(self, event, *args):
            for handler in list(self._handlers.get(event, [])):
                handler(*args)

    @pytest.fixture(autouse=True)
    def fake_environment(self, monkeypatch):
        # The developer's Synthesia key and their own LiveKit project credentials,
        # set the same way a real agent's environment would have them.
        monkeypatch.setenv("SYNTHESIA_API_KEY", "syn_live_key")
        monkeypatch.setenv("LIVEKIT_URL", "wss://my-project.livekit.cloud")
        monkeypatch.setenv("LIVEKIT_API_KEY", "lk_key")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "lk_secret")

        async def _joined(**kwargs):
            return None

        monkeypatch.setattr("livekit.agents.utils.wait_for_participant", _joined)
        monkeypatch.setattr("livekit.agents.utils.wait_for_track_publication", _joined)

        class _FakeBackend:
            def __init__(self, **kwargs):
                pass

            async def start_session(self, request, *, conn_options=None):
                return StartSessionResponse(session_id="sess_abc")

        monkeypatch.setattr("livekit.plugins.synthesia.avatar.SynthesiaAPI", _FakeBackend)

        class _FakeAudioTransport:
            def __init__(self, room, *, destination_identity, **kwargs):
                self.destination_identity = destination_identity

            async def aclose(self):
                pass

        monkeypatch.setattr(
            "livekit.plugins.synthesia.avatar.DataStreamAudioOutput", _FakeAudioTransport
        )

    async def test_attach_an_avatar_to_a_voice_agent(self, caplog):
        # Construct the avatar with gallery avatar ids: the first is the active
        # avatar, any others are precomputed for a future mid-session swap. The
        # Synthesia API key is read from SYNTHESIA_API_KEY when api_key is not passed.
        avatar = synthesia.AvatarSession(
            synthesia.AvatarConfig(avatar_ids=["03cee7ec-ac90-45ec-8c20-74a399cf3dc4"])
        )

        session = self._AgentSession()
        room = self._Room()

        # Start the avatar BEFORE the agent session, passing the room. This wires the
        # agent's speech to the avatar participant.
        await avatar.start(session, room=room)
        await session.start(agent=object(), room=room)

        assert session.output.audio is not None
        assert session.output.audio.destination_identity == avatar.avatar_identity

        # When the room ends, the plugin logs it and tears down.
        with caplog.at_level(logging.INFO, logger="livekit.plugins.synthesia"):
            room.fire("disconnected")
            await avatar._teardown_task

        assert "avatar session ended" in caplog.text

    async def test_missing_key_raises_a_typed_error(self, monkeypatch):
        monkeypatch.delenv("SYNTHESIA_API_KEY", raising=False)
        with pytest.raises(synthesia.SynthesiaError):
            synthesia.AvatarSession(
                synthesia.AvatarConfig(avatar_ids=["03cee7ec-ac90-45ec-8c20-74a399cf3dc4"])
            )
