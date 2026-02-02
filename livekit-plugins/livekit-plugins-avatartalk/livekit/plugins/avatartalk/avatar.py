import os
from typing import Optional

from livekit import api, rtc
from livekit.agents import NOT_GIVEN, AgentSession, NotGivenOr, get_job_context
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.agents.voice.avatar import DataStreamAudioOutput

from .api import AvatarTalkAPI, AvatarTalkException
from .log import logger

_AVATAR_AGENT_IDENTITY = "avatartalk-agent"
_AVATAR_AGENT_NAME = "avatartalk-agent"
DEFAULT_AVATAR_NAME = "japanese_man"
DEFAULT_AVATAR_EMOTION = "expressive"
SAMPLE_RATE = 16000


class AvatarSession:
    """AvatarTalkAPI avatar session"""

    def __init__(
        self,
        *,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        avatar: NotGivenOr[str | None] = NOT_GIVEN,
        emotion: NotGivenOr[str | None] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str | None] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str | None] = NOT_GIVEN,
    ):
        self._avatartalk_api = AvatarTalkAPI(api_url, api_secret)
        self._avatar = avatar or (os.getenv("AVATARTALK_AVATAR") or DEFAULT_AVATAR_NAME)
        self._emotion = emotion or (os.getenv("AVATARTALK_EMOTION") or DEFAULT_AVATAR_EMOTION)
        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._agent_track = None

    def __generate_lk_token(
        self,
        livekit_api_key: str,
        livekit_api_secret: str,
        room: rtc.Room,
        participant_identity: str,
        participant_name: str,
        as_agent: bool = True,
        local_participant_identity: Optional[str] = None,
    ) -> str:
        token = api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
        token = token.with_identity(participant_identity)
        token = token.with_name(participant_name)
        token = token.with_grants(api.VideoGrants(room_join=True, room=room.name))

        if as_agent:
            token = token.with_kind("agent")

        if local_participant_identity is not None:
            token = token.with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})

        return token.to_jwt()

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str | None] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str | None] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise AvatarTalkException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set by arguments or environment variables"
            )

        session_task_mapping: dict[str, str] = {}

        job_ctx = get_job_context()

        async def _shutdown_session() -> None:
            if room.name not in session_task_mapping:
                return
            await self._avatartalk_api.stop_session(session_task_mapping[room.name])

        job_ctx.add_shutdown_callback(_shutdown_session)

        local_participant_identity = job_ctx.local_participant_identity
        livekit_token = self.__generate_lk_token(
            livekit_api_key,
            livekit_api_secret,
            room,
            self._avatar_participant_identity,
            self._avatar_participant_name,
            as_agent=True,
            local_participant_identity=local_participant_identity,
        )

        livekit_listener_token = self.__generate_lk_token(
            livekit_api_key,
            livekit_api_secret,
            room,
            "listener",
            "listener",
            as_agent=False,
            local_participant_identity=None,
        )

        logger.debug(
            "Starting Avatartalk agent session",
            extra={"avatar": self._avatar, "room_name": room.name},
        )
        try:
            resp = await self._avatartalk_api.start_session(
                livekit_url=livekit_url,
                avatar=self._avatar,
                emotion=self._emotion,
                room_name=room.name,
                livekit_listener_token=livekit_listener_token,
                livekit_room_token=livekit_token,
                agent_identity=local_participant_identity,
            )

            self.conversation_id = resp["task_id"]
            logger.debug(
                "Avatartalk agent session started",
                extra={
                    "avatar": self._avatar,
                    "emotion": self._emotion,
                    "room_name": room.name,
                    "task_id": self.conversation_id,
                },
            )
            session_task_mapping[room.name] = self.conversation_id

            agent_session.output.audio = DataStreamAudioOutput(
                room=room,
                destination_identity="listener",
                sample_rate=SAMPLE_RATE,
                # wait_remote_track=rtc.TrackKind.KIND_VIDEO,
            )
        except AvatarTalkException as e:
            logger.error(e)
