from __future__ import annotations

import os
from dataclasses import asdict, dataclass

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

from .api import AvatarioAPI, AvatarioException
from .log import logger

SAMPLE_RATE = 24000
_AVATAR_AGENT_IDENTITY = "avatario-avatar-agent"
_AVATAR_AGENT_NAME = "avatario-avatar-agent"


class AvatarSession:
    """An Avatario avatar session"""

    @dataclass
    class VideoInfo:
        video_height: int = 720
        video_width: int = 1280
        custom_background_url: str | None = None

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        video_info: NotGivenOr[VideoInfo] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Initialize Avatario avatar session

        Args:
            avatar_id: The ID of avatar to use in the session. If not provided it will read
                        from the AVATARIO_AVATAR_ID environment variable.
                        IDs of our stock avatars can be found here:-
                        (https://avatario.ai/dashboard/9pqbj80f/avatars/stock)
            video_info: a dataclass containing information about the video resolution
                        and background of the avatar session.
            api_key: Your Avatario API key. If not provided, it will be read from
                     the AVATARIO_API_KEY environment variable.
            avatar_participant_identity: Identity of the avatario participant that will
                                         join the room. Defaults to "avatario-avatar-agent"
            avatar_participant_name: Name of the avatario participant that will join the
                                     room. Defaults to "avatario-avatar-agent"
            conn_options: Connection options for the aiohttp session.
        """
        self._http_session: aiohttp.ClientSession | None = None
        self._conn_options = conn_options
        video_info = video_info if utils.is_given(video_info) else self.VideoInfo()
        self._video_info = asdict(video_info)

        avatario_avatar_id = (
            avatar_id if utils.is_given(avatar_id) else os.getenv("AVATARIO_AVATAR_ID")
        )
        if not avatario_avatar_id:
            raise AvatarioException("AVATARIO_AVATAR_ID must be set")
        self._avatar_id = avatario_avatar_id

        avatario_api_key = api_key if utils.is_given(api_key) else os.getenv("AVATARIO_API_KEY")
        if not avatario_api_key:
            raise AvatarioException("AVATARIO_API_KEY must be set")
        self._api_key = avatario_api_key

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
        """Entrypoint to start the video avatar session"""
        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise AvatarioException(
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

        async with AvatarioAPI(
            api_key=self._api_key,
            avatar_id=self._avatar_id,
            video_info=self._video_info,
            conn_options=self._conn_options,
        ) as avatario_api:
            logger.debug("starting avatar session")
            await avatario_api.start_session(
                livekit_agent_identity=local_participant_identity,
                properties={
                    "url": livekit_url,
                    "token": livekit_token,
                },
            )

        agent_session.output.audio = DataStreamAudioOutput(
            room=room,
            destination_identity=self._avatar_participant_identity,
            sample_rate=SAMPLE_RATE,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO,
        )
