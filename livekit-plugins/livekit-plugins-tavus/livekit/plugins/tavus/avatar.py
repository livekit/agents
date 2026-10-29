from __future__ import annotations

import aiohttp

import livekit.rtc as rtc
from livekit import api
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    VoiceSession,
    get_job_context,
    utils,
)
from livekit.agents.job import _server_credentials
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .api import TavusAPI, TavusException
from .log import logger

SAMPLE_RATE = 24000
_AVATAR_AGENT_IDENTITY = "tavus-avatar-agent"
_AVATAR_AGENT_NAME = "tavus-avatar-agent"


class AvatarSession(BaseAvatarSession):
    """A Tavus avatar session"""

    def __init__(
        self,
        *,
        face_id: NotGivenOr[str] = NOT_GIVEN,
        pal_id: NotGivenOr[str] = NOT_GIVEN,
        replica_id: NotGivenOr[str] = NOT_GIVEN,
        persona_id: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__()
        self._http_session: aiohttp.ClientSession | None = None
        self._conn_options = conn_options
        self.conversation_id: str | None = None
        # `replica_id`/`persona_id` are deprecated aliases for `face_id`/`pal_id`.
        self._pal_id = pal_id
        self._face_id = face_id
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

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "tavus"

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = utils.http_context.http_session()

        return self._http_session

    async def start(
        self,
        agent_session: VoiceSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        await super().start(agent_session, room)

        livekit_url, livekit_api_key, livekit_api_secret = _server_credentials(
            livekit_url, livekit_api_key, livekit_api_secret
        )
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise TavusException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        job_ctx = get_job_context()
        local_participant_identity = job_ctx.local_participant_identity
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
        self.conversation_id = await self._api.create_conversation(
            pal_id=self._pal_id,
            face_id=self._face_id,
            persona_id=self._persona_id,
            replica_id=self._replica_id,
            properties={"livekit_ws_url": livekit_url, "livekit_room_token": livekit_token},
        )

        self._attach_audio_output(
            agent_session,
            DataStreamAudioOutput(
                room=room,
                destination_identity=self._avatar_participant_identity,
                sample_rate=SAMPLE_RATE,
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
            ),
        )
