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

from .api import DEFAULT_API_URL, AnamAPI
from .errors import AnamException
from .log import logger
from .types import PersonaConfig

SAMPLE_RATE = 24000
_AVATAR_AGENT_IDENTITY = "anam-avatar-agent"
_AVATAR_AGENT_NAME = "anam-avatar-agent"


class AvatarSession:
    """A Anam avatar session"""

    def __init__(
        self,
        *,
        persona_config: PersonaConfig,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._http_session: aiohttp.ClientSession | None = None
        self._conn_options = conn_options
        self.session_id: str | None = None
        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._persona_config: PersonaConfig = persona_config

        api_url_val = (
            api_url if utils.is_given(api_url) else os.getenv("ANAM_API_URL", DEFAULT_API_URL)
        )
        api_key_val = api_key if utils.is_given(api_key) else os.getenv("ANAM_API_KEY")

        if not api_key_val:
            raise AnamException("ANAM_API_KEY must be set by arguments or environment variables")

        self._api_url = api_url_val
        self._api_key = api_key_val

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
            raise AnamException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        job_ctx = get_job_context()
        local_participant_identity = job_ctx.local_participant_identity
        livekit_token = (
            api.AccessToken(
                api_key=livekit_api_key,
                api_secret=livekit_api_secret,
            )
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            # allow the avatar agent to publish audio and video on behalf of your local agent
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})
            .to_jwt()
        )
        async with AnamAPI(
            api_key=self._api_key, api_url=self._api_url, conn_options=self._conn_options
        ) as anam_api:
            session_token = await anam_api.create_session_token(
                persona_config=self._persona_config,
                livekit_url=livekit_url,
                livekit_token=livekit_token,
            )
            logger.debug("Anam session token created successfully.")

            logger.debug("Starting Anam engine session...")
            session_details = await anam_api.start_engine_session(
                session_token=session_token,
            )
            self.session_id = session_details.get("sessionId")

        agent_session.output.audio = DataStreamAudioOutput(
            room=room,
            destination_identity=self._avatar_participant_identity,
            sample_rate=SAMPLE_RATE,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO,
        )
