from __future__ import annotations

import os
from datetime import timedelta

from pydantic import BaseModel

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    http,
)
from livekit.agents.voice.avatar import (
    AvatarSession as BaseAvatarSession,
    AvatarStart,
    DataStreamAudioOutput,
)

DEFAULT_API_URL = "https://api.avatartalk.ai"
DEFAULT_AVATAR_NAME = "japanese_man"
DEFAULT_AVATAR_EMOTION = "expressive"
# the participant the provider listens to the agent's audio as
_LISTENER_IDENTITY = "listener"


class AvatarTalkException(Exception):
    """Exception for AvatarTalkAPI errors"""


class _Session(BaseModel):
    task_id: str | None


class AvatarSession(BaseAvatarSession):
    """AvatarTalkAPI avatar session"""

    sample_rate = 16000

    def __init__(
        self,
        *,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        avatar: NotGivenOr[str | None] = NOT_GIVEN,
        emotion: NotGivenOr[str | None] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(
            avatar_participant_identity=avatar_participant_identity or "avatartalk-agent",
            avatar_participant_name=avatar_participant_name,
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
            conn_options=conn_options,
        )
        self._api_url = api_url or DEFAULT_API_URL
        api_key = api_secret or os.getenv("AVATARTALK_API_KEY")
        if not api_key:
            raise AvatarTalkException(
                "AvatarTalk API key is required, either as argument or set"
                " AVATARTALK_API_KEY environment variable"
            )
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._avatar = avatar or os.getenv("AVATARTALK_AVATAR") or DEFAULT_AVATAR_NAME
        self._emotion = emotion or os.getenv("AVATARTALK_EMOTION") or DEFAULT_AVATAR_EMOTION
        self.conversation_id: str | None = None

    @property
    def provider(self) -> str:
        return "avatartalk"

    async def create_session(self, ctx: AvatarStart) -> str | None:
        # The listener token lives six hours, as before. The helper's default is one hour.
        listener_token = await ctx.participant_token(
            identity=_LISTENER_IDENTITY, name=_LISTENER_IDENTITY, ttl=timedelta(hours=6)
        )
        async with http.http_session().post(
            f"{self._api_url}/livekit/create-session",
            headers=self._headers,
            json={
                "livekit_url": ctx.livekit_url,
                "avatar": self._avatar,
                "emotion": self._emotion,
                "room_name": ctx.room.name,
                "room_token": await ctx.livekit_token(),
                "listener_token": listener_token,
                "agent_identity": ctx.agent_identity,
            },
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()
            self.conversation_id = _Session.model_validate_json(await response.text()).task_id
        return self.conversation_id

    async def end_session(self, session_id: str) -> None:
        async with http.http_session().delete(
            f"{self._api_url}/livekit/delete-session/{session_id}", headers=self._headers
        ) as response:
            await response.raise_for_status()

    def audio_output(self, ctx: AvatarStart) -> DataStreamAudioOutput:
        return DataStreamAudioOutput(
            room=ctx.room, destination_identity=_LISTENER_IDENTITY, sample_rate=self.sample_rate
        )
