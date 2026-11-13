from __future__ import annotations

import os

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    http,
)
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, AvatarStart

EGE_STOCK_AVATAR_ID = "b9be11b8-89fb-4227-8f86-4a881393cbdb"
_DEFAULT_API_URL = "https://api.bey.dev"


class BeyException(Exception):
    """Exception for Beyond Presence errors"""


class AvatarSession(BaseAvatarSession):
    """A Beyond Presence avatar session"""

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str | None] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
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
        self._avatar_id = avatar_id or EGE_STOCK_AVATAR_ID
        self._api_url = api_url or os.getenv("BEY_API_URL", _DEFAULT_API_URL)
        key = api_key or os.getenv("BEY_API_KEY")
        if not key:
            raise BeyException(
                "The api_key must be set either by passing api_key to the client or "
                "by setting the BEY_API_KEY environment variable"
            )
        self._api_key = key

    @property
    def provider(self) -> str:
        return "bey"

    async def create_session(self, ctx: AvatarStart) -> None:
        async with http.http_session().post(
            f"{self._api_url}/v1/session",
            headers={"x-api-key": self._api_key},
            json={
                "avatar_id": self._avatar_id,
                "livekit_url": ctx.livekit_url,
                "livekit_token": ctx.livekit_token(),
            },
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()
