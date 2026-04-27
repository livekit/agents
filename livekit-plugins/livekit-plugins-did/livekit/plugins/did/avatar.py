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
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .api import DIDAPI
from .errors import DIDException
from .log import logger
from .types import AudioConfig

_AVATAR_AGENT_IDENTITY = "d-id-avatar-agent"
_AVATAR_AGENT_NAME = "d-id-avatar-agent"


class AvatarSession(BaseAvatarSession):
    """A D-ID avatar session"""

    def __init__(
        self,
        *,
        agent_id: str,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        audio_config: AudioConfig | None = None,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._http_session: aiohttp.ClientSession | None = None
        self._conn_options = conn_options
        self._agent_id = agent_id
        self._audio_config = audio_config or AudioConfig()
        self.session_id: str | None = None

        self._api = DIDAPI(
            api_url=api_url,
            api_key=api_key,
            conn_options=conn_options,
            session=self._ensure_http_session(),
        )

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
        await super().start(agent_session, room)

        _livekit_url = livekit_url if utils.is_given(livekit_url) else os.getenv("LIVEKIT_URL")
        _livekit_api_key = (
            livekit_api_key if utils.is_given(livekit_api_key) else os.getenv("LIVEKIT_API_KEY")
        )
        _livekit_api_secret = (
            livekit_api_secret
            if utils.is_given(livekit_api_secret)
            else os.getenv("LIVEKIT_API_SECRET")
        )
        if not _livekit_url or not _livekit_api_key or not _livekit_api_secret:
            raise DIDException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        job_ctx = get_job_context()
        local_participant_identity = job_ctx.local_participant_identity
        livekit_token = (
            api.AccessToken(api_key=_livekit_api_key, api_secret=_livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})
            .to_jwt()
        )

        logger.debug("starting avatar session")
        self.session_id = await self._api.join_session(
            agent_id=self._agent_id,
            transport={
                "provider": "livekit",
                "server_url": _livekit_url,
                "token": livekit_token,
                "room_name": room.name,
            },
            audio_config={"sample_rate": self._audio_config.sample_rate},
        )

        agent_session.output.audio = DataStreamAudioOutput(
            room=room,
            destination_identity=self._avatar_participant_identity,
            sample_rate=self._audio_config.sample_rate,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO,
        )
