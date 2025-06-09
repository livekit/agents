from __future__ import annotations

import os

import aiohttp

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

from .api import TavusAPI, TavusException
from .log import logger

SAMPLE_RATE = 24000
_AVATAR_AGENT_IDENTITY = "tavus-avatar-agent"
_AVATAR_AGENT_NAME = "tavus-avatar-agent"


class AvatarSession:
    """A Tavus avatar session"""

    def __init__(
        self,
        *,
        replica_id: NotGivenOr[str] = NOT_GIVEN,
        persona_id: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._http_session: aiohttp.ClientSession | None = None
        self._conn_options = conn_options

        self._persona_id = persona_id
        self._replica_id = replica_id
        self._api = TavusAPI(
            api_url=api_url,
            api_key=api_key,
            conn_options=conn_options,
            session=self._ensure_http_session(),
        )

        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = utils.http_context.http_session()

        return self._http_session

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
            raise TavusException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        try:
            job_ctx = get_job_context()
            decoded = job_ctx.decode_token(livekit_api_secret)
            local_participant_identity = decoded["sub"]
        except (RuntimeError, KeyError):
            if not room.isconnected():
                raise TavusException(
                    "local participant identity not found in token, and room is not connected"
                ) from None
            local_participant_identity = room.local_participant.identity

        livekit_token = (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            # allow the avatar agent to publish audio and video on behalf of your local agent
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})
            .to_jwt()
        )

        logger.debug("starting avatar session")
        await self._api.create_conversation(
            persona_id=self._persona_id,
            replica_id=self._replica_id,
            properties={"livekit_ws_url": livekit_url, "livekit_room_token": livekit_token},
        )

        agent_session.output.audio = DataStreamAudioOutput(
            room=room,
            destination_identity=self._avatar_participant_identity,
            sample_rate=SAMPLE_RATE,
        )
