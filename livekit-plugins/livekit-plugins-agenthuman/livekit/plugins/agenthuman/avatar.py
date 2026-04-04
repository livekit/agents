from __future__ import annotations

import json
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

from .api import AgentHumanAPI, AgentHumanException
from .log import logger

_AVATAR_AGENT_IDENTITY = "agenthuman-avatar-agent"
_AVATAR_AGENT_NAME = "agenthuman-avatar-agent"


class AvatarSession:
    """An AgentHuman avatar session"""

    def __init__(
        self,
        *,
        avatar: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        aspect_ratio: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._http_session: aiohttp.ClientSession | None = None
        self._conn_options = conn_options
        self.session_id: str | None = None
        self._avatar = avatar
        self._aspect_ratio = aspect_ratio
        self._api = AgentHumanAPI(
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
            raise AgentHumanException(
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

        logger.debug("[agenthuman] starting avatar session")

        self.session_id = await self._api.create_session(
            avatar=self._avatar,
            aspect_ratio=self._aspect_ratio,
            livekit_room={"livekit_ws_url": livekit_url, "livekit_room_token": livekit_token},
        )

        @room.on("data_received")
        def on_data_received(data_packet: rtc.DataPacket) -> None:
            if data_packet.topic == "session.state":
                try:
                    payload = json.loads(data_packet.data.decode("utf-8"))
                    logger.info(
                        "[agenthuman] session.state received: state=%s reason=%s",
                        payload.get("state"),
                        payload.get("reason", ""),
                    )
                except Exception as e:
                    logger.warning("[agenthuman] session.state: failed to parse data packet: %s", e)

        agent_session.output.audio = DataStreamAudioOutput(
            room=room,
            destination_identity=self._avatar_participant_identity,
            sample_rate=16000,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO,
        )
