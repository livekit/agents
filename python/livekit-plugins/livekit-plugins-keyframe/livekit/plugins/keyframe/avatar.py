from __future__ import annotations

import os

from pydantic import BaseModel, JsonValue

import livekit.rtc as rtc
from livekit import api
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    http,
    utils,
)
from livekit.agents.voice.avatar import (
    AvatarSession as BaseAvatarSession,
    AvatarStart,
    DataStreamAudioOutput,
)

from .errors import KeyframeException
from .log import logger
from .types import Emotion

DEFAULT_API_URL = "https://api.keyframelabs.com"
CONTROL_TOPIC = "lk.control"


class _Session(BaseModel):
    reservation_id: str | None = None


class AvatarSession(BaseAvatarSession):
    """A Keyframe avatar session for the LiveKit Agents framework."""

    def __init__(
        self,
        *,
        persona_id: NotGivenOr[str] = NOT_GIVEN,
        persona_slug: NotGivenOr[str] = NOT_GIVEN,
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
        self._persona: dict[str, JsonValue]
        if utils.is_given(persona_id) and not utils.is_given(persona_slug):
            self._persona = {"persona_id": persona_id}
        elif utils.is_given(persona_slug) and not utils.is_given(persona_id):
            self._persona = {"persona_slug": persona_slug}
        else:
            raise KeyframeException("Provide exactly one of persona_id or persona_slug")

        self._api_url = (
            api_url if utils.is_given(api_url) else os.getenv("KEYFRAME_API_URL", DEFAULT_API_URL)
        )
        api_key_val = api_key if utils.is_given(api_key) else os.getenv("KEYFRAME_API_KEY")
        if not api_key_val:
            raise KeyframeException(
                "KEYFRAME_API_KEY must be set by arguments or environment variables"
            )
        self._api_key = api_key_val

    @property
    def provider(self) -> str:
        return "keyframe"

    async def create_session(self, ctx: AvatarStart) -> str | None:
        grants = api.VideoGrants(
            room_join=True, room=ctx.room.name, can_publish=True, can_subscribe=True
        )
        async with http.http_session().post(
            f"{self._api_url}/v1/sessions/plugins/livekit",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                **self._persona,
                "room_name": ctx.room.name,
                "livekit_url": ctx.livekit_url,
                "livekit_token": await ctx.livekit_token(grants=grants),
                "source_participant_identity": ctx.agent_identity,
            },
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()
            return _Session.model_validate_json(await response.text()).reservation_id

    def audio_output(self, ctx: AvatarStart) -> DataStreamAudioOutput:
        return DataStreamAudioOutput(
            room=ctx.room,
            destination_identity=self.avatar_identity,
            sample_rate=self.sample_rate,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO,
            clear_buffer_timeout=None,
        )

    async def set_emotion(self, emotion: Emotion) -> None:
        """Set the avatar's emotional expression.

        Sends a control message to the avatar worker to change its facial
        expression and demeanor.

        Args:
            emotion: The emotion to express ('neutral', 'happy', 'sad', 'angry').
        """
        if self._room is None:
            logger.warning("set_emotion() called before start()")
            return

        await self._room.local_participant.publish_data(
            f"emotion:{emotion}".encode(),
            reliable=False,
            topic=CONTROL_TOPIC,
            destination_identities=[self.avatar_identity],
        )
