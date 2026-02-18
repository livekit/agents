from __future__ import annotations

import os

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    APIConnectOptions,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .api import DEFAULT_API_URL, KFLAPI
from .errors import KFLException
from .log import logger

SAMPLE_RATE = 24000
_AVATAR_AGENT_IDENTITY = "kfl-avatar-agent"
_AVATAR_AGENT_NAME = "kfl-avatar-agent"


class AvatarSession:
    """A KFL avatar session for the LiveKit Agents framework."""

    def __init__(
        self,
        *,
        persona_id: NotGivenOr[str] = NOT_GIVEN,
        persona_slug: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._conn_options = conn_options
        self._avatar_participant_identity = (
            avatar_participant_identity
            if utils.is_given(avatar_participant_identity)
            else _AVATAR_AGENT_IDENTITY
        )
        self._avatar_participant_name = (
            avatar_participant_name
            if utils.is_given(avatar_participant_name)
            else _AVATAR_AGENT_NAME
        )

        # Resolve persona
        self._persona_id = persona_id if utils.is_given(persona_id) else None
        self._persona_slug = persona_slug if utils.is_given(persona_slug) else None

        has_id = bool(self._persona_id)
        has_slug = bool(self._persona_slug)
        if has_id == has_slug:
            raise KFLException("Provide exactly one of persona_id or persona_slug")

        # Resolve API config
        api_url_val = (
            api_url if utils.is_given(api_url) else os.getenv("KFL_API_URL", DEFAULT_API_URL)
        )
        api_key_val = api_key if utils.is_given(api_key) else os.getenv("KFL_API_KEY")

        if not api_key_val:
            raise KFLException("KFL_API_KEY must be set by arguments or environment variables")

        self._api_url = api_url_val
        self._api_key = api_key_val

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise KFLException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        job_ctx = get_job_context()
        local_participant_identity = job_ctx.local_participant_identity

        # Mint a LiveKit token for the avatar worker with publish_on_behalf
        livekit_token = (
            api.AccessToken(
                api_key=livekit_api_key,
                api_secret=livekit_api_secret,
            )
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room.name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})
            .to_jwt()
        )

        # Call API to create a reservation and dispatch
        async with KFLAPI(
            api_key=self._api_key,
            api_url=self._api_url,
            conn_options=self._conn_options,
        ) as kfl_api:
            result = await kfl_api.create_plugin_session(
                persona_id=self._persona_id,
                persona_slug=self._persona_slug,
                room_name=room.name,
                livekit_url=livekit_url,
                livekit_token=livekit_token,
                source_participant_identity=local_participant_identity,
            )
            logger.debug(
                "KFL plugin session created: reservation_id=%s",
                result.get("reservation_id"),
            )

        # Redirect agent TTS audio to the avatar worker via DataStream
        agent_session.output.audio = DataStreamAudioOutput(
            room=room,
            destination_identity=self._avatar_participant_identity,
            sample_rate=SAMPLE_RATE,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO,
            clear_buffer_timeout=None,
        )
