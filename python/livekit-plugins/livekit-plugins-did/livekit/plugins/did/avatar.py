from __future__ import annotations

import os

from pydantic import BaseModel

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    http,
    utils,
)
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, AvatarStart

from .errors import DIDException
from .types import AudioConfig

DEFAULT_API_URL = "https://api.d-id.com"


class _Session(BaseModel):
    id: str


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
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(
            avatar_participant_identity=avatar_participant_identity,
            avatar_participant_name=avatar_participant_name,
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
            conn_options=conn_options,
        )
        self._agent_id = agent_id
        self.sample_rate = (audio_config or AudioConfig()).sample_rate
        self.session_id: str | None = None
        self._api_url = api_url if utils.is_given(api_url) else DEFAULT_API_URL
        key = api_key if utils.is_given(api_key) else os.getenv("DID_API_KEY")
        if not key:
            raise DIDException("DID_API_KEY must be set")
        self._api_key = key

    @property
    def provider(self) -> str:
        return "d-id"

    async def create_session(self, ctx: AvatarStart) -> str:
        async with http.http_session().post(
            f"{self._api_url}/v2/agents/{self._agent_id}/sessions/join",
            headers={"Authorization": f"Basic {self._api_key}"},
            json={
                "transport": {
                    "provider": "livekit",
                    "server_url": ctx.livekit_url,
                    "token": await ctx.livekit_token(),
                    "room_name": ctx.room.name,
                },
                "audio_config": {"sample_rate": self.sample_rate},
            },
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()
            self.session_id = _Session.model_validate_json(await response.text()).id
        return self.session_id
