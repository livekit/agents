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

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from typing_extensions import Self

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
from livekit.plugins.boson_avatar.api import BosonAvatarAPI
from livekit.plugins.boson_avatar.errors import BosonAvatarException

pytestmark = pytest.mark.unit


class _Response:
    def __init__(
        self,
        status: int,
        payload: object | None = None,
        *,
        headers: dict[str, str] | None = None,
        raw_text: str | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._payload = payload
        self._text = (
            raw_text if raw_text is not None else ("" if payload is None else json.dumps(payload))
        )

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 400

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def text(self) -> str:
        return self._text

    async def json(self, *, content_type: str | None = None) -> object:
        del content_type
        if self._payload is None:
            raise ValueError("empty body")
        return self._payload


class _Session:
    def __init__(self, outcomes: list[_Response | BaseException]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def request(self, method: str, url: str, **kwargs: object) -> _Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class BosonAvatarAPITest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()

    async def test_start_and_end_use_hosted_contract(self) -> None:
        session = _Session(
            [
                _Response(201, {"id": "avatar-session-1", "avatar_identity": "avatar-1"}),
                _Response(204),
            ]
        )
        client = BosonAvatarAPI(
            api_key="boson-key",
            api_url="https://avatar.example/v1/",
            conn_options=APIConnectOptions(max_retry=0, timeout=7),
            session=session,  # type: ignore[arg-type]
        )

        info = await client.start_session(
            avatar_id="asset-1",
            livekit_url="wss://tenant.livekit.cloud",
            livekit_room="room-1",
            livekit_token="signed-livekit-token",
            avatar_identity="avatar-1",
            publisher_identity="voice-1",
            width=640,
            height=480,
            max_duration_seconds=900,
            idempotency_key="idem-1",
        )
        await client.end_session(info.id)

        self.assertEqual(info.id, "avatar-session-1")
        self.assertEqual([call["method"] for call in session.calls], ["POST", "DELETE"])
        self.assertEqual(
            [call["url"] for call in session.calls],
            [
                "https://avatar.example/v1/sessions",
                "https://avatar.example/v1/sessions/avatar-session-1",
            ],
        )
        post = session.calls[0]
        self.assertEqual(post["headers"]["Authorization"], "Bearer boson-key")
        self.assertEqual(post["headers"]["Idempotency-Key"], "idem-1")
        self.assertEqual(
            post["json"],
            {
                "avatar_id": "asset-1",
                "transport": {
                    "type": "livekit",
                    "url": "wss://tenant.livekit.cloud",
                    "room_name": "room-1",
                    "participant_token": "signed-livekit-token",
                    "participant_identity": "avatar-1",
                    "publisher_identity": "voice-1",
                    "audio_source": "data_stream",
                },
                "output": {"width": 640, "height": 480},
                "max_duration_seconds": 900,
            },
        )

    async def test_retry_reuses_one_idempotency_key(self) -> None:
        session = _Session(
            [
                _Response(503, {"error": {"code": "busy"}}),
                _Response(201, {"id": "avatar-session-1", "avatar_identity": "avatar-1"}),
            ]
        )
        client = BosonAvatarAPI(
            api_key="boson-key",
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
            session=session,  # type: ignore[arg-type]
        )
        with patch("livekit.plugins.boson_avatar.api.asyncio.sleep", new=AsyncMock()):
            await client.start_session(
                avatar_id="asset-1",
                livekit_url="wss://tenant.livekit.cloud",
                livekit_room="room-1",
                livekit_token="signed-livekit-token",
                avatar_identity="avatar-1",
                publisher_identity="voice-1",
            )

        keys = [call["headers"]["Idempotency-Key"] for call in session.calls]
        self.assertEqual(len(keys), 2)
        self.assertTrue(keys[0])
        self.assertEqual(keys[0], keys[1])

    async def test_non_retryable_auth_error_preserves_status_and_request_id(
        self,
    ) -> None:
        session = _Session(
            [
                _Response(
                    401,
                    {"error": {"code": "invalid_api_key", "request_id": "req-1"}},
                )
            ]
        )
        client = BosonAvatarAPI(
            api_key="bad-key",
            conn_options=APIConnectOptions(max_retry=3),
            session=session,  # type: ignore[arg-type]
        )
        with self.assertRaises(APIStatusError) as raised:
            await client.start_session(
                avatar_id="asset-1",
                livekit_url="wss://tenant.livekit.cloud",
                livekit_room="room-1",
                livekit_token="signed-livekit-token",
                avatar_identity="avatar-1",
                publisher_identity="voice-1",
            )
        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(raised.exception.request_id, "req-1")
        self.assertEqual(len(session.calls), 1)

    async def test_transport_failure_is_wrapped_after_retries(self) -> None:
        session = _Session(
            [
                aiohttp.ClientConnectionError("offline"),
                asyncio.TimeoutError(),
            ]
        )
        client = BosonAvatarAPI(
            api_key="boson-key",
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
            session=session,  # type: ignore[arg-type]
        )
        with (
            patch("livekit.plugins.boson_avatar.api.asyncio.sleep", new=AsyncMock()),
            self.assertRaises(APIConnectionError),
        ):
            await client.end_session("avatar-session-1")
        self.assertEqual(len(session.calls), 2)

    async def test_rejects_missing_id_and_identity_mismatch(self) -> None:
        for payload in (
            {"avatar_identity": "avatar-1"},
            {"id": "avatar-session-1", "avatar_identity": "wrong-avatar"},
        ):
            with self.subTest(payload=payload):
                outcomes = [_Response(201, payload)]
                if payload.get("id"):
                    outcomes.append(_Response(204))
                session = _Session(outcomes)
                client = BosonAvatarAPI(
                    api_key="boson-key",
                    conn_options=APIConnectOptions(max_retry=0),
                    session=session,  # type: ignore[arg-type]
                )
                with self.assertRaises(BosonAvatarException):
                    await client.start_session(
                        avatar_id="asset-1",
                        livekit_url="wss://tenant.livekit.cloud",
                        livekit_room="room-1",
                        livekit_token="signed-livekit-token",
                        avatar_identity="avatar-1",
                        publisher_identity="voice-1",
                    )
                if payload.get("id"):
                    self.assertEqual([call["method"] for call in session.calls], ["POST", "DELETE"])

    def test_configuration_uses_documented_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BOSON_API_KEY": "env-key",
                "BOSON_AVATAR_API_URL": "https://env.example/",
            },
            clear=True,
        ):
            client = BosonAvatarAPI()
        self.assertEqual(client._api_key, "env-key")
        self.assertEqual(client._api_url, "https://env.example")

    def test_missing_api_key_fails_without_network_access(self) -> None:
        with self.assertRaisesRegex(BosonAvatarException, "BOSON_API_KEY"):
            BosonAvatarAPI()


if __name__ == "__main__":
    unittest.main()
