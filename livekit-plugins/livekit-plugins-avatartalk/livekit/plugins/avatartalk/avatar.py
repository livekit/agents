import os

from livekit import api, rtc
from livekit.agents import NOT_GIVEN, AgentSession, NotGivenOr, get_job_context
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput

from .api import AvatarTalkAPI, AvatarTalkException
from .log import logger

_AVATAR_AGENT_IDENTITY = "avatartalk-agent"
_AVATAR_AGENT_NAME = "avatartalk-agent"
DEFAULT_AVATAR_NAME = "japanese_man"
DEFAULT_AVATAR_EMOTION = "expressive"
SAMPLE_RATE = 16000


class AvatarSession(BaseAvatarSession):
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
        super().__init__()
        self._avatartalk_api = AvatarTalkAPI(api_url, api_secret)
        self._avatar = avatar or (os.getenv("AVATARTALK_AVATAR") or DEFAULT_AVATAR_NAME)
        self._emotion = emotion or (os.getenv("AVATARTALK_EMOTION") or DEFAULT_AVATAR_EMOTION)
        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._agent_track = None

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "avatartalk"

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str | None] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str | None] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        await super().start(agent_session, room)

        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url:
            raise AvatarTalkException(
                "livekit_url must be set by arguments or environment variables"
            )

        session_task_mapping: dict[str, str] = {}

        job_ctx = get_job_context()

        async def _shutdown_session() -> None:
            if room.name not in session_task_mapping:
                return
            await self._avatartalk_api.stop_session(session_task_mapping[room.name])

        job_ctx.add_shutdown_callback(_shutdown_session)

        local_participant_identity = job_ctx.local_participant_identity
        livekit_token = await job_ctx.create_participant_token(
            identity=self._avatar_participant_identity,
            name=self._avatar_participant_name,
            kind="agent",
            grants=api.VideoGrants(room_join=True, room=room.name),
            attributes={ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity},
            api_key=livekit_api_key or None,
            api_secret=livekit_api_secret or None,
        )

        livekit_listener_token = await job_ctx.create_participant_token(
            identity="listener",
            name="listener",
            grants=api.VideoGrants(room_join=True, room=room.name),
            api_key=livekit_api_key or None,
            api_secret=livekit_api_secret or None,
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

            agent_session.output.replace_audio_tail(
                DataStreamAudioOutput(
                    room=room,
                    destination_identity="listener",
                    sample_rate=SAMPLE_RATE,
                    # wait_remote_track=rtc.TrackKind.KIND_VIDEO,
                ),
            )
        except AvatarTalkException as e:
            logger.error(e)
