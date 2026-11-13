from __future__ import annotations

import os

from pydantic import BaseModel, JsonValue

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    http,
    utils,
)
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, AvatarStart

from .api import DEFAULT_API_URL, DEFAULT_FACE_ID, DEFAULT_PAL_ID, TavusException


class _Conversation(BaseModel):
    conversation_id: str


class AvatarSession(BaseAvatarSession):
    """A Tavus avatar session"""

    def __init__(
        self,
        *,
        face_id: NotGivenOr[str] = NOT_GIVEN,
        pal_id: NotGivenOr[str] = NOT_GIVEN,
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
        self.conversation_id: str | None = None
        key = api_key or os.getenv("TAVUS_API_KEY")
        if not key:
            raise TavusException("TAVUS_API_KEY must be set")
        self._api_key = key
        self._api_url = api_url or DEFAULT_API_URL
        pal_id = pal_id or os.getenv("TAVUS_PAL_ID") or ""
        face_id = face_id or os.getenv("TAVUS_FACE_ID") or ""
        if not pal_id:
            # the stock pal, with its stock face unless one is given
            pal_id, face_id = DEFAULT_PAL_ID, face_id or DEFAULT_FACE_ID
        self._pal_id = pal_id
        # a pal carries its own default face
        self._face_id = face_id

    @property
    def provider(self) -> str:
        return "tavus"

    async def create_session(self, ctx: AvatarStart) -> str:
        payload: dict[str, JsonValue] = {
            "pal_id": self._pal_id,
            "conversation_name": utils.shortuuid("lk_conversation_"),
            "properties": {
                "livekit_ws_url": ctx.livekit_url,
                "livekit_room_token": ctx.livekit_token(),
            },
        }
        if self._face_id:
            payload["face_id"] = self._face_id
        async with http.http_session().post(
            f"{self._api_url}/conversations",
            headers={"x-api-key": self._api_key},
            json=payload,
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()
            conversation = _Conversation.model_validate_json(await response.text())
        self.conversation_id = conversation.conversation_id
        return self.conversation_id
