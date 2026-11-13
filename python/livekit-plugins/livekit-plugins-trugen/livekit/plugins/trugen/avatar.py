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

DEFAULT_API_URL = "https://api.trugen.ai"
_DEFAULT_AVATAR_ID = "665a1170"


class TrugenException(Exception):
    """Exception for TruGen.AI errors"""


class AvatarSession(BaseAvatarSession):
    """TruGen Realtime Avatar Session"""

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
            avatar_participant_identity=avatar_participant_identity or "trugen-avatar",
            avatar_participant_name=avatar_participant_name or "Trugen Avatar",
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
            conn_options=conn_options,
        )
        self._avatar_id = avatar_id or _DEFAULT_AVATAR_ID
        self._api_url = api_url or DEFAULT_API_URL
        key = api_key or os.getenv("TRUGEN_API_KEY")
        if not key:
            raise TrugenException(
                "The api_key not found; set this by passing api_key to the client or "
                "by setting the TRUGEN_API_KEY environment variable"
            )
        self._api_key = key

    @property
    def provider(self) -> str:
        return "trugen"

    async def create_session(self, ctx: AvatarStart) -> None:
        async with http.http_session().post(
            f"{self._api_url}/v1/sessions",
            headers={"x-api-key": self._api_key},
            json={
                "avatar_id": self._avatar_id,
                "livekit_url": ctx.livekit_url,
                "livekit_token": ctx.livekit_token(),
            },
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()
