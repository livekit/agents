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

from .api import AkoolAPI, AkoolException
from .log import logger
from .schema import AvatarConfig

SAMPLE_RATE = 16000
_AVATAR_AGENT_IDENTITY = "akool-avatar-agent"
_AVATAR_AGENT_NAME = "akool-avatar-agent"

DEFAULT_AVATAR_CONFIG = AvatarConfig()


class AvatarSession:
    """A Akool avatar session"""

    def __init__(
        self,
        *,
        avatar_config: AvatarConfig = DEFAULT_AVATAR_CONFIG,
        client_id: NotGivenOr[str] = NOT_GIVEN,
        client_secret: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._http_session: aiohttp.ClientSession | None = None
        self._conn_options = conn_options
        self.session_id: str | None = None
        self._avatar_config = avatar_config

        self._client_id = client_id or os.getenv("AKOOL_CLIENT_ID")
        self._client_secret = client_secret or os.getenv("AKOOL_CLIENT_SECRET")
        if not self._client_id or not self._client_secret:
            raise AkoolException("AKOOL_CLIENT_ID and AKOOL_CLIENT_SECRET must be set")

        self._api = AkoolAPI(
            avatar_config=self._avatar_config,
            client_id=self._client_id,
            client_secret=self._client_secret,
            api_url=api_url,
            conn_options=conn_options,
            session=self._ensure_http_session(),
        )

        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME

    def get_avatar_participant_identity(self) -> str:
        return self._avatar_participant_identity

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
            raise AkoolException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        try:
            job_ctx = get_job_context()
            local_participant_identity = job_ctx.token_claims().identity
        except RuntimeError as e:
            if not room.isconnected():
                raise AkoolException("failed to get local participant identity") from e
            local_participant_identity = room.local_participant.identity

        logger.info(
            f"Starting avatar session for participant {local_participant_identity} in room {room.name}"
        )

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

        logger.debug("creating avatar session with akool API")
        try:
            session_detail = await self._api.create_session(
                livekit_url=livekit_url,
                livekit_token=livekit_token,
            )
            self.session_id = session_detail["_id"]
            logger.info(f"Avatar session created successfully, session_id: {self.session_id}")
        except AkoolException as e:
            logger.error(f"Failed to create avatar session: {e}")
            raise

        agent_session.output.audio = DataStreamAudioOutput(
            room=room,
            destination_identity=self._avatar_participant_identity,
            sample_rate=SAMPLE_RATE,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO,
        )

    async def aclose(self) -> None:
        """
        关闭 Avatar 会话并清理资源
        Close avatar session and cleanup resources
        """
        logger.info(f"Closing avatar session: {self.session_id}")

        # 如果有会话 ID，尝试调用关闭接口
        if self.session_id and self._api:
            try:
                await self._api.close_session(self.session_id)
            except Exception as e:
                logger.warning(f"Failed to close avatar session: {e}")

        # 关闭 HTTP 会话
        if self._http_session:
            try:
                await self._http_session.close()
            except Exception as e:
                logger.warning(f"Failed to close http session: {e}")
            finally:
                self._http_session = None

        # 重置会话 ID
        self.session_id = None
        logger.info("Avatar session closed successfully")
