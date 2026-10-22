# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import patch

import pytest
from typing_extensions import Self

from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.plugins.agenthuman import AgentHumanException, AvatarSession
from livekit.plugins.agenthuman.api import AgentHumanAPI
from livekit.plugins.agenthuman.avatar import DEFAULT_AVATAR_ID

pytestmark = pytest.mark.unit

_NO_RETRY_DELAY = APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=1.0)


class _Response:
    def __init__(self, status: int, payload: Any = None) -> None:
        self.status = status
        self._payload = payload

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 400

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def text(self) -> str:
        return json.dumps(self._payload)

    async def json(self) -> Any:
        return self._payload


class _Session:
    """Stands in for aiohttp.ClientSession, replaying canned responses."""

    def __init__(self, *responses: _Response | Exception) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.requests.append({"url": url, **kwargs})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _api(session: _Session) -> AgentHumanAPI:
    return AgentHumanAPI(
        "ah-key",
        "https://api.agenthuman.test/v1/",
        conn_options=_NO_RETRY_DELAY,
        session=session,  # type: ignore[arg-type]
    )


async def test_create_session_sends_avatar_and_room() -> None:
    session = _Session(_Response(200, {"session": {"session_id": "sess_1"}}))

    session_id = await _api(session).create_session(
        avatar="avat_1",
        livekit_url="wss://lk.test",
        livekit_token="lk-token",
        aspect_ratio="3:4",
        extra_payload={"metadata": {"k": "v"}},
    )

    assert session_id == "sess_1"
    [request] = session.requests
    assert request["url"] == "https://api.agenthuman.test/v1/sessions"
    assert request["headers"]["x-api-key"] == "ah-key"
    assert request["json"] == {
        "avatar": "avat_1",
        "room": {"platform": "livekit", "url": "wss://lk.test", "token": "lk-token"},
        "aspect_ratio": "3:4",
        "metadata": {"k": "v"},
    }


async def test_create_session_omits_unset_aspect_ratio() -> None:
    session = _Session(_Response(200, {"session": {"session_id": "sess_1"}}))

    await _api(session).create_session(avatar="avat_1", livekit_url="wss://lk", livekit_token="t")

    assert "aspect_ratio" not in session.requests[0]["json"]


async def test_create_session_retries_server_errors() -> None:
    session = _Session(
        _Response(503, {"error": "busy"}),
        APIConnectionError("connection reset"),
        _Response(200, {"session": {"session_id": "sess_1"}}),
    )

    assert (
        await _api(session).create_session(
            avatar="avat_1", livekit_url="wss://lk", livekit_token="t"
        )
        == "sess_1"
    )
    assert len(session.requests) == 3  # 1 attempt + max_retry (2) retries


async def test_create_session_does_not_retry_client_errors() -> None:
    session = _Session(_Response(401, {"error": "invalid api key"}))

    with pytest.raises(APIConnectionError) as exc_info:
        await _api(session).create_session(
            avatar="avat_1", livekit_url="wss://lk", livekit_token="t"
        )

    assert not exc_info.value.retryable
    assert len(session.requests) == 1


async def test_create_session_gives_up_after_all_retries() -> None:
    session = _Session(*(_Response(500, {}) for _ in range(3)))

    with pytest.raises(APIConnectionError):
        await _api(session).create_session(
            avatar="avat_1", livekit_url="wss://lk", livekit_token="t"
        )
    assert len(session.requests) == 3


async def test_create_session_rejects_unexpected_response_without_leaking_it() -> None:
    session = _Session(_Response(200, {"access_token": "secret"}))

    with pytest.raises(AgentHumanException) as exc_info:
        await _api(session).create_session(
            avatar="avat_1", livekit_url="wss://lk", livekit_token="t"
        )

    assert "secret" not in str(exc_info.value)
    assert len(session.requests) == 1


def test_avatar_session_defaults_to_stock_avatar() -> None:
    with patch.dict(os.environ, {"AGENTHUMAN_API_KEY": "ah-key"}, clear=True):
        assert AvatarSession()._avatar == DEFAULT_AVATAR_ID


def test_avatar_session_rejects_both_avatar_kinds() -> None:
    with patch.dict(os.environ, {"AGENTHUMAN_API_KEY": "ah-key"}, clear=True):
        with pytest.raises(AgentHumanException, match="only one"):
            AvatarSession(avatar_id="avat_1", avatar_image_url="https://img.test/a.png")
    env = {"AGENTHUMAN_API_KEY": "ah-key", "AGENTHUMAN_AVATAR_ID": "avat_env"}
    with patch.dict(os.environ, env, clear=True), pytest.raises(AgentHumanException):
        AvatarSession(avatar_image_url="https://img.test/a.png")


def test_avatar_session_requires_api_key() -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(AgentHumanException, match="AGENTHUMAN_API_KEY"),
    ):
        AvatarSession(avatar_id="avat_1")


def test_avatar_session_reads_environment() -> None:
    env = {
        "AGENTHUMAN_API_KEY": "ah-key",
        "AGENTHUMAN_IMAGE_URL": "https://img.test/a.png",
        "AGENTHUMAN_API_URL": "https://api.agenthuman.test/v1",
    }
    with patch.dict(os.environ, env, clear=True):
        avatar = AvatarSession()

    assert avatar._avatar == "https://img.test/a.png"
    assert avatar._api_url == "https://api.agenthuman.test/v1"
    assert avatar.avatar_identity == "agenthuman-avatar-agent"
    assert avatar.provider == "agenthuman"


def test_avatar_session_arguments_override_environment() -> None:
    env = {"AGENTHUMAN_API_KEY": "env-key", "AGENTHUMAN_AVATAR_ID": "avat_env"}
    with patch.dict(os.environ, env, clear=True):
        avatar = AvatarSession(
            avatar_id="avat_arg", api_key="arg-key", avatar_participant_identity="my-avatar"
        )

    assert avatar._avatar == "avat_arg"
    assert avatar._api_key == "arg-key"
    assert avatar.avatar_identity == "my-avatar"
