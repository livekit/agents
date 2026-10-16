from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.plugins.spatius import AvatarSession

pytestmark = pytest.mark.unit


def test_extra_params_are_copied() -> None:
    params = {"server_post_process": "false"}

    with patch.dict(
        os.environ,
        {
            "SPATIUS_API_KEY": "test-key",
            "SPATIUS_APP_ID": "test-app",
            "SPATIUS_AVATAR_ID": "test-avatar",
        },
    ):
        session = AvatarSession(extra_params=params)

    params["server_post_process"] = "true"

    assert session._extra_params == {"server_post_process": "false"}


async def test_extra_params_are_forwarded_to_sdk() -> None:
    sdk_session = MagicMock()
    sdk_session.init = AsyncMock()
    sdk_session.start = AsyncMock()
    sdk_session.close = AsyncMock()
    room = MagicMock()
    room.name = "test-room"
    room.local_participant.identity = "test-agent"
    agent_session = MagicMock()
    agent_session.output = MagicMock()

    with (
        patch.dict(
            os.environ,
            {
                "SPATIUS_API_KEY": "test-key",
                "SPATIUS_APP_ID": "test-app",
                "SPATIUS_AVATAR_ID": "test-avatar",
            },
        ),
        patch("livekit.plugins.spatius.avatar.BaseAvatarSession.start", new=AsyncMock()),
        patch("livekit.plugins.spatius.avatar.QueueAudioOutput") as queue_audio,
        patch(
            "livekit.plugins.spatius.avatar.new_avatar_session", return_value=sdk_session
        ) as create_sdk_session,
    ):
        queue_audio.return_value.start = AsyncMock()
        queue_audio.return_value.aclose = AsyncMock()
        queue_audio.return_value.on = MagicMock()
        session = AvatarSession(extra_params={"server_post_process": "false"}, sample_rate=24_000)
        await session.start(
            agent_session,
            room,
            livekit_url="wss://livekit.example.com",
            livekit_api_key="test-key-with-at-least-thirty-two-characters",
            livekit_api_secret="test-secret-with-at-least-thirty-two-characters",
        )

    assert create_sdk_session.call_args.kwargs["extra_params"] == {"server_post_process": "false"}
    await session.aclose()
