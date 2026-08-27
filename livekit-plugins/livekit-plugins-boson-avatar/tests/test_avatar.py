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
import base64
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest

from livekit import rtc
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.plugins.boson_avatar.api import AvatarSessionInfo
from livekit.plugins.boson_avatar.avatar import SAMPLE_RATE, AvatarSession
from livekit.plugins.boson_avatar.errors import BosonAvatarException

pytestmark = pytest.mark.unit


class _Room:
    name = "room-1"
    local_participant = SimpleNamespace(identity="voice-1")

    def isconnected(self) -> bool:
        return True


class _Output:
    def __init__(self) -> None:
        self.audio = None
        self.replaced = None

    def replace_audio_tail(self, output: object) -> None:
        self.replaced = output


class _AgentSession:
    def __init__(self, output: _Output | None = None) -> None:
        self.output = output or _Output()
        self._handlers: dict[str, list[object]] = {}

    def on(self, event: str, callback: object) -> None:
        self._handlers.setdefault(event, []).append(callback)

    def off(self, event: str, callback: object) -> None:
        handlers = self._handlers.get(event, [])
        if callback in handlers:
            handlers.remove(callback)

    def emit(self, event: str, value: object = None) -> None:
        for callback in list(self._handlers.get(event, [])):
            callback(value)  # type: ignore[operator]


def _jwt_claims(token: str) -> dict:
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


class AvatarSessionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()

    def test_constructor_requires_developer_configuration(self) -> None:
        with self.assertRaisesRegex(BosonAvatarException, "avatar_id"):
            AvatarSession(api_key="boson-key")
        with self.assertRaisesRegex(BosonAvatarException, "BOSON_API_KEY"):
            AvatarSession(avatar_id="asset-1")

    def test_constructor_validates_configuration(self) -> None:
        invalid = (
            {"width": 640},
            {"width": 0, "height": 640},
            {"width": True, "height": 640},
            {"width": 1.5, "height": 640},
            {"width": "wide", "height": 640},
            {"max_duration_seconds": -1},
            {"max_duration_seconds": 1.5},
            {"max_duration_seconds": 14_401},
            {"idempotency_key": ""},
            {"idempotency_key": 123},
            {"idempotency_key": "application-session-1"},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(BosonAvatarException):
                AvatarSession(avatar_id="asset-1", api_key="boson-key", **kwargs)

    def test_constructor_uses_environment(self) -> None:
        with patch.dict(
            os.environ,
            {"BOSON_API_KEY": "env-key", "BOSON_AVATAR_ID": "asset-env"},
            clear=True,
        ):
            session = AvatarSession()
        self.assertEqual(session._avatar_id, "asset-env")
        self.assertEqual(session.avatar_identity, "boson-avatar-agent")

    async def test_start_accepts_generic_agent_audio_and_mints_scoped_token(self) -> None:
        api_client = SimpleNamespace(
            start_session=AsyncMock(
                return_value=AvatarSessionInfo("provider-session-1", "avatar-1")
            ),
            end_session=AsyncMock(),
        )
        output = _Output()
        agent_session = _AgentSession(output)
        audio_sink = object()

        with (
            patch(
                "livekit.plugins.boson_avatar.avatar.BosonAvatarAPI",
                return_value=api_client,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.get_job_context",
                return_value=SimpleNamespace(
                    job=SimpleNamespace(id="AJ_test"),
                    local_participant_identity="voice-1",
                ),
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.DataStreamAudioOutput",
                return_value=audio_sink,
            ) as audio_output,
            patch.object(BaseAvatarSession, "start", new=AsyncMock()) as base_start,
            patch.object(BaseAvatarSession, "aclose", new=AsyncMock()),
        ):
            avatar = AvatarSession(
                avatar_id="asset-1",
                api_key="boson-key",
                width=640,
                height=480,
                max_duration_seconds=900,
                avatar_participant_identity="avatar-1",
                avatar_participant_name="Demo Avatar",
            )
            session_id = await avatar.start(
                agent_session,
                _Room(),  # type: ignore[arg-type]
                livekit_url="wss://tenant.livekit.cloud",
                livekit_api_key="livekit-key",
                livekit_api_secret="livekit-secret-with-enough-entropy",
            )

        self.assertEqual(session_id, "provider-session-1")
        self.assertEqual(avatar.session_id, "provider-session-1")
        base_start.assert_awaited_once()
        audio_output.assert_called_once_with(
            room=ANY,
            destination_identity="avatar-1",
            sample_rate=SAMPLE_RATE,
            wait_remote_track=rtc.TrackKind.KIND_AUDIO,
        )
        self.assertIs(output.replaced, audio_sink)
        call = api_client.start_session.await_args.kwargs
        self.assertEqual(call["publisher_identity"], "voice-1")
        self.assertEqual(call["avatar_id"], "asset-1")
        self.assertEqual(call["width"], 640)
        self.assertEqual(call["height"], 480)
        self.assertEqual(call["max_duration_seconds"], 900)
        self.assertEqual(
            call["idempotency_key"],
            "d2efd9c4-b705-5e73-9303-66ecad5bc551",
        )
        claims = _jwt_claims(call["livekit_token"])
        self.assertEqual(claims["sub"], "avatar-1")
        self.assertEqual(claims["name"], "Demo Avatar")
        self.assertEqual(claims["kind"], "agent")
        self.assertEqual(claims["video"]["room"], "room-1")
        self.assertTrue(claims["video"]["roomJoin"])
        self.assertFalse(claims["video"]["canSubscribe"])
        self.assertTrue(claims["video"]["canPublishData"])
        self.assertEqual(set(claims["video"]["canPublishSources"]), {"camera", "microphone"})
        self.assertEqual(claims["attributes"][ATTRIBUTE_PUBLISH_ON_BEHALF], "voice-1")

        with self.assertRaisesRegex(RuntimeError, "called twice"):
            await avatar.start(agent_session, _Room())  # type: ignore[arg-type]

    async def test_explicit_idempotency_key_overrides_livekit_job_default(self) -> None:
        api_client = SimpleNamespace(
            start_session=AsyncMock(
                return_value=AvatarSessionInfo("provider-session-1", "avatar-1")
            ),
            end_session=AsyncMock(),
        )

        with (
            patch(
                "livekit.plugins.boson_avatar.avatar.BosonAvatarAPI",
                return_value=api_client,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.get_job_context",
                return_value=SimpleNamespace(
                    job=SimpleNamespace(id="AJ_test"),
                    local_participant_identity="voice-1",
                ),
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.DataStreamAudioOutput",
                return_value=object(),
            ),
            patch.object(BaseAvatarSession, "start", new=AsyncMock()),
            patch.object(BaseAvatarSession, "aclose", new=AsyncMock()),
        ):
            avatar = AvatarSession(
                avatar_id="asset-1",
                api_key="boson-key",
                avatar_participant_identity="avatar-1",
                idempotency_key="123e4567-e89b-12d3-a456-426614174000",
            )
            await avatar.start(
                _AgentSession(),  # type: ignore[arg-type]
                _Room(),  # type: ignore[arg-type]
                livekit_url="wss://tenant.livekit.cloud",
                livekit_api_key="livekit-key",
                livekit_api_secret="livekit-secret-with-enough-entropy",
            )

        self.assertEqual(
            api_client.start_session.await_args.kwargs["idempotency_key"],
            "123e4567-e89b-12d3-a456-426614174000",
        )

    async def test_start_failure_ends_provider_session_and_base_session(self) -> None:
        api_client = SimpleNamespace(
            start_session=AsyncMock(
                return_value=AvatarSessionInfo("provider-session-1", "avatar-1")
            ),
            end_session=AsyncMock(),
        )
        output = _Output()
        output.replace_audio_tail = Mock(side_effect=RuntimeError("sink rejected"))
        agent_session = _AgentSession(output)

        with (
            patch(
                "livekit.plugins.boson_avatar.avatar.BosonAvatarAPI",
                return_value=api_client,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.get_job_context",
                return_value=None,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.DataStreamAudioOutput",
                return_value=object(),
            ),
            patch.object(BaseAvatarSession, "start", new=AsyncMock()),
            patch.object(BaseAvatarSession, "aclose", new=AsyncMock()) as base_close,
        ):
            avatar = AvatarSession(
                avatar_id="asset-1",
                api_key="boson-key",
                avatar_participant_identity="avatar-1",
            )
            with self.assertRaisesRegex(RuntimeError, "sink rejected"):
                await avatar.start(
                    agent_session,
                    _Room(),  # type: ignore[arg-type]
                    livekit_url="wss://tenant.livekit.cloud",
                    livekit_api_key="livekit-key",
                    livekit_api_secret="livekit-secret-with-enough-entropy",
                )
            await avatar.aclose()

        api_client.end_session.assert_awaited_once_with("provider-session-1")
        base_close.assert_awaited_once()
        self.assertIsNone(avatar.session_id)

    async def test_close_retries_api_error_and_base_cleanup_remains_idempotent(
        self,
    ) -> None:
        api_client = SimpleNamespace(
            start_session=AsyncMock(
                return_value=AvatarSessionInfo("provider-session-1", "avatar-1")
            ),
            end_session=AsyncMock(side_effect=[RuntimeError("provider unavailable"), None]),
        )
        agent_session = _AgentSession()

        with (
            patch(
                "livekit.plugins.boson_avatar.avatar.BosonAvatarAPI",
                return_value=api_client,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.get_job_context",
                return_value=None,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.DataStreamAudioOutput",
                return_value=object(),
            ),
            patch.object(BaseAvatarSession, "start", new=AsyncMock()),
            patch.object(BaseAvatarSession, "aclose", new=AsyncMock()) as base_close,
        ):
            avatar = AvatarSession(
                avatar_id="asset-1",
                api_key="boson-key",
                avatar_participant_identity="avatar-1",
            )
            await avatar.start(
                agent_session,
                _Room(),  # type: ignore[arg-type]
                livekit_url="wss://tenant.livekit.cloud",
                livekit_api_key="livekit-key",
                livekit_api_secret="livekit-secret-with-enough-entropy",
            )
            await avatar.aclose()
            await avatar.aclose()
            await avatar.aclose()

        self.assertEqual(api_client.end_session.await_count, 2)
        api_client.end_session.assert_awaited_with("provider-session-1")
        base_close.assert_awaited_once()
        self.assertIsNone(avatar.session_id)

    async def test_close_waits_for_inflight_start_and_compensates_session(self) -> None:
        create_entered = asyncio.Event()
        release_create = asyncio.Event()

        async def create_session(**_: object) -> AvatarSessionInfo:
            create_entered.set()
            await release_create.wait()
            return AvatarSessionInfo("provider-session-1", "avatar-1")

        api_client = SimpleNamespace(
            start_session=AsyncMock(side_effect=create_session),
            end_session=AsyncMock(),
        )
        agent_session = _AgentSession()

        with (
            patch(
                "livekit.plugins.boson_avatar.avatar.BosonAvatarAPI",
                return_value=api_client,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.get_job_context",
                return_value=None,
            ),
            patch.object(BaseAvatarSession, "start", new=AsyncMock()),
            patch.object(BaseAvatarSession, "aclose", new=AsyncMock()) as base_close,
        ):
            avatar = AvatarSession(
                avatar_id="asset-1",
                api_key="boson-key",
                avatar_participant_identity="avatar-1",
            )
            start_task = asyncio.create_task(
                avatar.start(
                    agent_session,  # type: ignore[arg-type]
                    _Room(),  # type: ignore[arg-type]
                    livekit_url="wss://tenant.livekit.cloud",
                    livekit_api_key="livekit-key",
                    livekit_api_secret="livekit-secret-with-enough-entropy",
                )
            )
            await create_entered.wait()
            close_task = asyncio.create_task(avatar.aclose())
            await asyncio.sleep(0)
            release_create.set()

            with self.assertRaisesRegex(RuntimeError, "closed while start"):
                await start_task
            await close_task
            await avatar.aclose()

        api_client.end_session.assert_awaited_once_with("provider-session-1")
        base_close.assert_awaited_once()
        self.assertIsNone(avatar.session_id)

    async def test_start_cancellation_waits_for_create_and_compensates_session(self) -> None:
        create_entered = asyncio.Event()
        release_create = asyncio.Event()

        async def create_session(**_: object) -> AvatarSessionInfo:
            create_entered.set()
            await release_create.wait()
            return AvatarSessionInfo("provider-session-1", "avatar-1")

        api_client = SimpleNamespace(
            start_session=AsyncMock(side_effect=create_session),
            end_session=AsyncMock(),
        )
        agent_session = _AgentSession()

        with (
            patch(
                "livekit.plugins.boson_avatar.avatar.BosonAvatarAPI",
                return_value=api_client,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.get_job_context",
                return_value=None,
            ),
            patch.object(BaseAvatarSession, "start", new=AsyncMock()),
            patch.object(BaseAvatarSession, "aclose", new=AsyncMock()) as base_close,
        ):
            avatar = AvatarSession(
                avatar_id="asset-1",
                api_key="boson-key",
                avatar_participant_identity="avatar-1",
            )
            start_task = asyncio.create_task(
                avatar.start(
                    agent_session,  # type: ignore[arg-type]
                    _Room(),  # type: ignore[arg-type]
                    livekit_url="wss://tenant.livekit.cloud",
                    livekit_api_key="livekit-key",
                    livekit_api_secret="livekit-secret-with-enough-entropy",
                )
            )
            await create_entered.wait()
            start_task.cancel()
            await asyncio.sleep(0)
            release_create.set()
            with self.assertRaises(asyncio.CancelledError):
                await start_task

        api_client.end_session.assert_awaited_once_with("provider-session-1")
        base_close.assert_awaited_once()
        self.assertIsNone(avatar.session_id)

    async def test_repeated_start_cancellation_keeps_cleanup_task_alive(self) -> None:
        create_entered = asyncio.Event()
        release_create = asyncio.Event()

        async def create_session(**_: object) -> AvatarSessionInfo:
            create_entered.set()
            await release_create.wait()
            return AvatarSessionInfo("provider-session-1", "avatar-1")

        api_client = SimpleNamespace(
            start_session=AsyncMock(side_effect=create_session),
            end_session=AsyncMock(),
        )
        agent_session = _AgentSession()

        with (
            patch(
                "livekit.plugins.boson_avatar.avatar.BosonAvatarAPI",
                return_value=api_client,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.get_job_context",
                return_value=None,
            ),
            patch.object(BaseAvatarSession, "start", new=AsyncMock()),
            patch.object(BaseAvatarSession, "aclose", new=AsyncMock()) as base_close,
        ):
            avatar = AvatarSession(
                avatar_id="asset-1",
                api_key="boson-key",
                avatar_participant_identity="avatar-1",
            )
            start_task = asyncio.create_task(
                avatar.start(
                    agent_session,  # type: ignore[arg-type]
                    _Room(),  # type: ignore[arg-type]
                    livekit_url="wss://tenant.livekit.cloud",
                    livekit_api_key="livekit-key",
                    livekit_api_secret="livekit-secret-with-enough-entropy",
                )
            )
            await create_entered.wait()
            start_task.cancel()
            for _ in range(10):
                if avatar._startup_cleanup_task is not None:
                    break
                await asyncio.sleep(0)
            cleanup_task = avatar._startup_cleanup_task
            assert cleanup_task is not None

            start_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await start_task
            release_create.set()
            await cleanup_task
            await avatar.aclose()

        api_client.end_session.assert_awaited_once_with("provider-session-1")
        base_close.assert_awaited_once()
        self.assertIsNone(avatar.session_id)

    async def test_cancellation_during_start_compensation_does_not_cancel_delete(
        self,
    ) -> None:
        delete_entered = asyncio.Event()
        release_delete = asyncio.Event()
        delete_completed: list[str] = []

        async def end_session(session_id: str) -> None:
            delete_entered.set()
            await release_delete.wait()
            delete_completed.append(session_id)

        api_client = SimpleNamespace(
            start_session=AsyncMock(
                return_value=AvatarSessionInfo("provider-session-1", "avatar-1")
            ),
            end_session=AsyncMock(side_effect=end_session),
        )
        output = _Output()
        output.replace_audio_tail = Mock(side_effect=RuntimeError("sink rejected"))
        agent_session = _AgentSession(output)

        with (
            patch(
                "livekit.plugins.boson_avatar.avatar.BosonAvatarAPI",
                return_value=api_client,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.get_job_context",
                return_value=None,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.DataStreamAudioOutput",
                return_value=object(),
            ),
            patch.object(BaseAvatarSession, "start", new=AsyncMock()),
            patch.object(BaseAvatarSession, "aclose", new=AsyncMock()) as base_close,
        ):
            avatar = AvatarSession(
                avatar_id="asset-1",
                api_key="boson-key",
                avatar_participant_identity="avatar-1",
            )
            start_task = asyncio.create_task(
                avatar.start(
                    agent_session,  # type: ignore[arg-type]
                    _Room(),  # type: ignore[arg-type]
                    livekit_url="wss://tenant.livekit.cloud",
                    livekit_api_key="livekit-key",
                    livekit_api_secret="livekit-secret-with-enough-entropy",
                )
            )
            await delete_entered.wait()
            cleanup_task = avatar._startup_cleanup_task
            assert cleanup_task is not None
            start_task.cancel()
            with self.assertRaisesRegex(RuntimeError, "sink rejected"):
                await start_task

            release_delete.set()
            await cleanup_task
            await avatar.aclose()

        self.assertEqual(delete_completed, ["provider-session-1"])
        api_client.end_session.assert_awaited_once_with("provider-session-1")
        base_close.assert_awaited_once()
        self.assertTrue(avatar._closed)
        self.assertIsNone(avatar.session_id)
        self.assertEqual(agent_session._handlers["close"], [])

    async def test_cancelling_close_does_not_cancel_owned_delete(self) -> None:
        delete_entered = asyncio.Event()
        release_delete = asyncio.Event()
        delete_completed: list[str] = []

        async def end_session(session_id: str) -> None:
            delete_entered.set()
            await release_delete.wait()
            delete_completed.append(session_id)

        api_client = SimpleNamespace(
            start_session=AsyncMock(
                return_value=AvatarSessionInfo("provider-session-1", "avatar-1")
            ),
            end_session=AsyncMock(side_effect=end_session),
        )
        agent_session = _AgentSession()

        with (
            patch(
                "livekit.plugins.boson_avatar.avatar.BosonAvatarAPI",
                return_value=api_client,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.get_job_context",
                return_value=None,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.DataStreamAudioOutput",
                return_value=object(),
            ),
            patch.object(BaseAvatarSession, "start", new=AsyncMock()),
            patch.object(BaseAvatarSession, "aclose", new=AsyncMock()) as base_close,
        ):
            avatar = AvatarSession(
                avatar_id="asset-1",
                api_key="boson-key",
                avatar_participant_identity="avatar-1",
            )
            await avatar.start(
                agent_session,  # type: ignore[arg-type]
                _Room(),  # type: ignore[arg-type]
                livekit_url="wss://tenant.livekit.cloud",
                livekit_api_key="livekit-key",
                livekit_api_secret="livekit-secret-with-enough-entropy",
            )

            close_task = asyncio.create_task(avatar.aclose())
            await delete_entered.wait()
            shutdown_task = avatar._shutdown_task
            assert shutdown_task is not None
            close_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await close_task

            release_delete.set()
            await shutdown_task
            await avatar.aclose()

        self.assertEqual(delete_completed, ["provider-session-1"])
        api_client.end_session.assert_awaited_once_with("provider-session-1")
        base_close.assert_awaited_once()
        self.assertTrue(avatar._closed)
        self.assertIsNone(avatar.session_id)

    async def test_agent_session_close_ends_provider_session(self) -> None:
        api_client = SimpleNamespace(
            start_session=AsyncMock(
                return_value=AvatarSessionInfo("provider-session-1", "avatar-1")
            ),
            end_session=AsyncMock(),
        )
        agent_session = _AgentSession()

        with (
            patch(
                "livekit.plugins.boson_avatar.avatar.BosonAvatarAPI",
                return_value=api_client,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.get_job_context",
                return_value=None,
            ),
            patch(
                "livekit.plugins.boson_avatar.avatar.DataStreamAudioOutput",
                return_value=object(),
            ),
            patch.object(BaseAvatarSession, "start", new=AsyncMock()),
            patch.object(BaseAvatarSession, "aclose", new=AsyncMock()) as base_close,
        ):
            avatar = AvatarSession(
                avatar_id="asset-1",
                api_key="boson-key",
                avatar_participant_identity="avatar-1",
            )
            await avatar.start(
                agent_session,  # type: ignore[arg-type]
                _Room(),  # type: ignore[arg-type]
                livekit_url="wss://tenant.livekit.cloud",
                livekit_api_key="livekit-key",
                livekit_api_secret="livekit-secret-with-enough-entropy",
            )
            agent_session.emit("close")
            assert avatar._agent_close_task is not None
            await avatar._agent_close_task

        api_client.end_session.assert_awaited_once_with("provider-session-1")
        base_close.assert_awaited_once()
        self.assertIsNone(avatar.session_id)


if __name__ == "__main__":
    unittest.main()
