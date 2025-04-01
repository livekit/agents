from __future__ import annotations

import os
from collections.abc import Awaitable

import httpx

from livekit import api, rtc
from livekit.agents import (
    JobContext,
)
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.agents.voice.io import AudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF, RoomOutputOptions

API_URL_ENV_VAR = "BEY_API_URL"
"""
The environment variable name for the Beyond Presence API URL
"""

API_KEY_ENV_VAR = "BEY_API_KEY"
"""
The environment variable name for the Beyond Presence API key
"""


EGE_STOCK_AVATAR_ID = "b9be11b8-89fb-4227-8f86-4a881393cbdb"
"""
The ID of Ege's stock avatar
"""

_DEFAULT_API_URL = "https://api.bey.dev"

_AVATAR_AGENT_IDENTITY = "bey-avatar-agent"
_AVATAR_AGENT_NAME = "bey-avatar-agent"


class BeyAvatarSession:
    """A Beyond Presence avatar session"""

    def __init__(
        self,
        avatar_agent_joined_awaitable: Awaitable[None],
        local_agent_audio_output: AudioOutput,
        local_agent_room_output_options: RoomOutputOptions,
    ) -> None:
        """
        Args:
            avatar_agent_joined_awaitable: An awaitable that resolves when the avatar agent joins
                the room
            local_agent_audio_output: The audio sink for the local agent output audio to reach the
                avatar agent
            local_agent_room_output_options: The room output options for the local agent to write
                messages as the avatar agent
        """
        self._avatar_agent_joined_awaitable = avatar_agent_joined_awaitable
        self._local_agent_audio_output = local_agent_audio_output
        self._local_agent_room_output_options = local_agent_room_output_options

    async def wait_for_avatar_agent(self) -> None:
        """Wait for the avatar agent to join the room"""
        await self._avatar_agent_joined_awaitable

    @property
    def local_agent_audio_output(self) -> AudioOutput:
        """The audio sink for the local agent output audio to reach the avatar agent"""
        return self._local_agent_audio_output

    @property
    def local_agent_room_output_options(self) -> RoomOutputOptions:
        """The room output options for the local agent to write messages as the avatar agent"""
        return self._local_agent_room_output_options


class BeyException(Exception):
    """Exception for Beyond Presence errors"""


async def start_bey_avatar_session(
    ctx: JobContext,
    avatar_id: str = EGE_STOCK_AVATAR_ID,
) -> BeyAvatarSession:
    """
    Start a Beyond Presence avatar session

    Args:
        ctx: The LiveKit Agent job context
        avatar_id: The ID of the avatar to request

    Returns:
        The context for the Beyond Presence avatar session

    Raises:
        BeyException: If the Beyond Presence session fails to start
    """

    if (api_key := os.environ.get(API_KEY_ENV_VAR)) is None:
        raise BeyException(f"{API_KEY_ENV_VAR} environment variable not set")

    api_url = os.environ.get(API_URL_ENV_VAR, _DEFAULT_API_URL)

    livekit_avatar_token = (
        api.AccessToken()
        .with_kind("agent")
        .with_identity(_AVATAR_AGENT_IDENTITY)
        .with_name(_AVATAR_AGENT_NAME)
        .with_grants(api.VideoGrants(room_join=True, room=ctx.room.name))
        # allow the avatar agent to publish audio and video on behalf of your local agent
        .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: ctx.room.local_participant.identity})
        .to_jwt()
    )

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{api_url}/v1/session",
            headers={
                "x-api-key": api_key,
            },
            json={
                "avatar_id": avatar_id,
                "livekit_url": ctx._info.url,
                "livekit_token": livekit_avatar_token,
            },
        )
    if response.is_error:
        raise BeyException(f"Avatar session server responded with error: {response.text}")

    return BeyAvatarSession(
        avatar_agent_joined_awaitable=ctx.wait_for_participant(
            identity=_AVATAR_AGENT_IDENTITY, kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        ),
        local_agent_audio_output=DataStreamAudioOutput(
            ctx.room, destination_identity=_AVATAR_AGENT_IDENTITY
        ),
        local_agent_room_output_options=RoomOutputOptions(
            # avatar agent will speak to the user, so disable audio output on our end
            audio_enabled=False,
            transcription_enabled=True,
        ),
    )
