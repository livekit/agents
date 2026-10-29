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

import os
import unittest
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
from livekit.plugins.boson_avatar.api import (
    AvatarInfo,
    AvatarSessionInfo,
    AvatarSessionStartError,
    BosonAvatarAPI,
)
from livekit.plugins.boson_avatar.errors import BosonAvatarException

pytestmark = pytest.mark.unit


class _Reply:
    def __init__(self, status, payload=None, *, headers=None, raw_text=None, truncated=False):
        self.status = status
        self.payload = payload
        self.headers = headers or {}
        self.raw_text = raw_text
        self.truncated = truncated


class _Peer:
    def __init__(self, server, http, calls):
        self.server = server
        self.http = http
        self.calls = calls
        self.url = str(server.make_url("")).rstrip("/")

    async def close(self):
        await self.http.close()
        await self.server.close()


def _active_session(
    session_id: str = "avatar-session-1", avatar_identity: str = "avatar-1"
) -> dict[str, str]:
    return {
        "id": session_id,
        "object": "avatar.livekit.session",
        "status": "active",
        "avatar_identity": avatar_identity,
    }


class BosonAvatarAPITest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._peers = []

    async def asyncTearDown(self):
        for peer in self._peers:
            await peer.close()

    async def _session(self, outcomes):
        calls = []

        async def respond(request):
            call = {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
            }
            if request.can_read_body:
                call["json"] = await request.json()
            calls.append(call)
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                request.transport.close()
                return web.Response()
            if outcome.truncated:
                response = web.StreamResponse(
                    status=outcome.status,
                    headers={"Content-Type": "application/json", "Content-Length": "100"},
                )
                await response.prepare(request)
                await response.write(b"{")
                request.transport.close()
                return response
            if outcome.raw_text is not None:
                return web.Response(
                    status=outcome.status,
                    text=outcome.raw_text,
                    headers=outcome.headers,
                    content_type="application/json",
                )
            if outcome.payload is None:
                return web.Response(status=outcome.status, headers=outcome.headers)
            return web.json_response(
                outcome.payload, status=outcome.status, headers=outcome.headers
            )

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", respond)
        server = TestServer(app)
        await server.start_server()
        peer = _Peer(server, aiohttp.ClientSession(), calls)
        self._peers.append(peer)
        os.environ["BOSON_AVATAR_API_URL"] = peer.url + "/v1"
        return peer

    def setUp(self) -> None:
        self.env = patch.dict(
            os.environ,
            {"BOSON_AVATAR_API_URL": "https://avatar.test/v1"},
            clear=True,
        )
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()

    async def test_list_avatars_uses_provider_catalog(self) -> None:
        session = await self._session(
            [
                _Reply(
                    200,
                    {
                        "object": "avatar.list",
                        "data": [
                            {"avatar_id": "asset_demo", "name": "Claire"},
                            {"avatar_id": "asset_1", "name": "Emma"},
                        ],
                    },
                )
            ]
        )
        client = BosonAvatarAPI(
            api_key="boson-key",
            api_url=session.url + "/v1",
            conn_options=APIConnectOptions(max_retry=0),
            session=session.http,
        )

        avatars = await client.list_avatars()

        self.assertEqual(
            avatars,
            [
                AvatarInfo(avatar_id="asset_demo", name="Claire"),
                AvatarInfo(avatar_id="asset_1", name="Emma"),
            ],
        )
        self.assertEqual(session.calls[0]["method"], "GET")
        self.assertEqual(session.calls[0]["url"], session.url + "/v1/avatars")
        self.assertEqual(
            session.calls[0]["headers"]["Authorization"],
            "Bearer boson-key",
        )

    async def test_list_avatars_rejects_invalid_provider_responses(self) -> None:
        invalid_payloads = (
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
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                session = await self._session([_Reply(200, payload)])
                client = BosonAvatarAPI(
                    api_key="boson-key",
                    api_url=session.url + "/v1",
                    conn_options=APIConnectOptions(max_retry=0),
                    session=session.http,
                )
                with self.assertRaises(BosonAvatarException):
                    await client.list_avatars()

    async def test_start_and_end_use_hosted_contract(self) -> None:
        session = await self._session(
            [
                _Reply(201, _active_session()),
                _Reply(204),
            ]
        )
        client = BosonAvatarAPI(
            api_key="boson-key",
            api_url=session.url + "/v1/",
            conn_options=APIConnectOptions(max_retry=0),
            session=session.http,
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
                session.url + "/v1/sessions",
                session.url + "/v1/sessions/avatar-session-1",
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
        session = await self._session(
            [
                _Reply(
                    503,
                    {"error": {"code": "busy"}},
                    headers={"Retry-After": "5"},
                ),
                _Reply(201, _active_session()),
            ]
        )
        client = BosonAvatarAPI(
            api_key="boson-key",
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
            session=session.http,
        )
        with (
            patch("livekit.plugins.boson_avatar.api.asyncio.sleep", new=AsyncMock()) as sleep,
            self.assertLogs("livekit.plugins.boson_avatar", level="WARNING") as logs,
        ):
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
        sleep.assert_awaited_once_with(5.0)
        self.assertEqual(logs.records[0].error_type, "APIStatusError")
        self.assertEqual(logs.records[0].__dict__["lk.pii.path"], "/sessions")

    async def test_non_retryable_auth_error_preserves_status_and_request_id(
        self,
    ) -> None:
        session = await self._session(
            [
                _Reply(
                    401,
                    {"error": {"code": "invalid_api_key", "request_id": "req-1"}},
                )
            ]
        )
        client = BosonAvatarAPI(
            api_key="bad-key",
            conn_options=APIConnectOptions(max_retry=3),
            session=session.http,
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

    async def test_incomplete_delete_response_is_wrapped_after_retries(self) -> None:
        session = await self._session(
            [
                _Reply(200, truncated=True),
                _Reply(200, truncated=True),
            ]
        )
        client = BosonAvatarAPI(
            api_key="boson-key",
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
            session=session.http,
        )
        with (
            patch("livekit.plugins.boson_avatar.api.asyncio.sleep", new=AsyncMock()),
            self.assertRaises(APIConnectionError) as raised,
        ):
            await client.end_session("avatar-session-1")
        self.assertEqual(len(session.calls), 2)
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        self.assertTrue(raised.exception.__suppress_context__)

    async def test_rejects_missing_id_and_identity_mismatch(self) -> None:
        for payload in (
            {
                "object": "avatar.livekit.session",
                "status": "active",
                "avatar_identity": "avatar-1",
            },
            _active_session(avatar_identity="wrong-avatar"),
            {**_active_session(), "status": "pending"},
        ):
            with self.subTest(payload=payload):
                outcomes = [_Reply(201, payload)]
                if payload.get("id"):
                    outcomes.append(_Reply(204))
                session = await self._session(outcomes)
                client = BosonAvatarAPI(
                    api_key="boson-key",
                    conn_options=APIConnectOptions(max_retry=0),
                    session=session.http,
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

    async def test_compensation_log_does_not_expose_provider_error(self) -> None:
        provider_secret = "private-provider-response"
        session = await self._session(
            [
                _Reply(201, _active_session(avatar_identity="wrong-avatar")),
                _Reply(503, {"error": {"message": provider_secret}}),
            ]
        )
        client = BosonAvatarAPI(
            api_key="boson-key",
            conn_options=APIConnectOptions(max_retry=0),
            session=session.http,
        )

        with (
            self.assertLogs("livekit.plugins.boson_avatar", level="WARNING") as logs,
            self.assertRaises(AvatarSessionStartError) as raised,
        ):
            await client.start_session(
                avatar_id="asset-1",
                livekit_url="wss://tenant.livekit.cloud",
                livekit_room="room-1",
                livekit_token="signed-livekit-token",
                avatar_identity="avatar-1",
                publisher_identity="voice-1",
            )

        self.assertEqual(
            raised.exception.session_info,
            AvatarSessionInfo("avatar-session-1", "wrong-avatar"),
        )
        record = logs.records[0]
        self.assertIsNone(record.exc_info)
        self.assertEqual(record.error_type, "APIConnectionError")
        self.assertEqual(record.__dict__["lk.pii.session_id"], "avatar-session-1")
        self.assertNotIn(provider_secret, "\n".join(logs.output))

    async def test_rejects_non_contract_success_status_without_retry(self) -> None:
        session = await self._session([_Reply(202, _active_session())])
        client = BosonAvatarAPI(
            api_key="boson-key",
            conn_options=APIConnectOptions(max_retry=3),
            session=session.http,
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

        self.assertEqual(raised.exception.status_code, 202)
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(len(session.calls), 1)

    async def test_validates_json_delete_response(self) -> None:
        session = await self._session(
            [
                _Reply(
                    200,
                    {
                        "id": "avatar-session-1",
                        "object": "avatar.livekit.session",
                        "status": "active",
                    },
                )
            ]
        )
        client = BosonAvatarAPI(
            api_key="boson-key",
            conn_options=APIConnectOptions(max_retry=0),
            session=session.http,
        )

        with self.assertRaises(BosonAvatarException):
            await client.end_session("avatar-session-1")

        no_content = await self._session([_Reply(204)])
        client = BosonAvatarAPI(
            api_key="boson-key",
            conn_options=APIConnectOptions(max_retry=0),
            session=no_content.http,
        )
        await client.end_session("avatar-session-1")
        self.assertEqual(len(no_content.calls), 1)

    def test_invalid_api_urls_fail_without_network_access(self) -> None:
        invalid_urls = (
            "/",
            "avatar.example/v1",
            "ftp://avatar.example/v1",
            "https:///v1",
            "https://avatar.example/v1?region=us",
            "https://avatar.example/v1#sessions",
            "https://user:password@avatar.example/v1",
            "https://avatar.example:invalid/v1",
            "http://avatar.example/v1",
        )
        for api_url in invalid_urls:
            with (
                self.subTest(api_url=api_url),
                self.assertRaises(BosonAvatarException),
            ):
                BosonAvatarAPI(api_key="boson-key", api_url=api_url)

    def test_loopback_http_api_urls_are_allowed_for_local_development(self) -> None:
        for api_url in (
            "http://localhost:8400/v1/",
            "http://worker.localhost:8400/v1/",
            "http://LOCALHOST:8400/v1/",
            "http://Worker.LOCALHOST:8400/v1/",
            "http://127.0.0.1:8400/v1/",
            "http://[::1]:8400/v1/",
        ):
            with self.subTest(api_url=api_url):
                client = BosonAvatarAPI(api_key="boson-key", api_url=api_url)
                self.assertEqual(client._api_url, api_url.rstrip("/"))


if __name__ == "__main__":
    unittest.main()
