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

    def test_constructor_validates_dimensions(self) -> None:
        invalid = (
            {"width": 640},
            {"width": 0, "height": 640},
            {"width": True, "height": 640},
            {"width": "wide", "height": 640},
            {"max_duration_seconds": -1},
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

    async def test_start_mints_scoped_token_and_installs_audio_output(self) -> None:
        api_client = SimpleNamespace(
            start_session=AsyncMock(
                return_value=AvatarSessionInfo("provider-session-1", "avatar-1")
            ),
            end_session=AsyncMock(),
        )
        output = _Output()
        agent_session = SimpleNamespace(output=output)
        audio_sink = object()

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
        claims = _jwt_claims(call["livekit_token"])
        self.assertEqual(claims["sub"], "avatar-1")
        self.assertEqual(claims["name"], "Demo Avatar")
        self.assertEqual(claims["kind"], "agent")
        self.assertEqual(claims["video"]["room"], "room-1")
        self.assertTrue(claims["video"]["roomJoin"])
        self.assertTrue(claims["video"]["canPublish"])
        self.assertTrue(claims["video"]["canSubscribe"])
        self.assertTrue(claims["video"]["canPublishData"])
        self.assertEqual(claims["attributes"][ATTRIBUTE_PUBLISH_ON_BEHALF], "voice-1")

        with self.assertRaisesRegex(RuntimeError, "called twice"):
            await avatar.start(agent_session, _Room())  # type: ignore[arg-type]

    async def test_start_failure_ends_provider_session_and_base_session(self) -> None:
        api_client = SimpleNamespace(
            start_session=AsyncMock(
                return_value=AvatarSessionInfo("provider-session-1", "avatar-1")
            ),
            end_session=AsyncMock(),
        )
        output = _Output()
        output.replace_audio_tail = Mock(side_effect=RuntimeError("sink rejected"))
        agent_session = SimpleNamespace(output=output)

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

    async def test_close_is_idempotent_and_base_cleanup_survives_api_error(
        self,
    ) -> None:
        api_client = SimpleNamespace(
            start_session=AsyncMock(
                return_value=AvatarSessionInfo("provider-session-1", "avatar-1")
            ),
            end_session=AsyncMock(side_effect=RuntimeError("provider unavailable")),
        )
        agent_session = SimpleNamespace(output=_Output())

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

        api_client.end_session.assert_awaited_once_with("provider-session-1")
        base_close.assert_awaited_once()
        self.assertIsNone(avatar.session_id)


if __name__ == "__main__":
    unittest.main()
