from __future__ import annotations

import json
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

from .log import logger

DEFAULT_API_URL = "https://api.dev.runwayml.com"
API_VERSION = "2024-11-06"


class RunwayException(Exception):
    """Exception for Runway errors"""


class _Session(BaseModel):
    id: str


class AvatarSession(BaseAvatarSession):
    """A Runway Characters avatar session.

    Creates a realtime session backed by Runway's avatar inference pipeline.
    The customer's LiveKit agent owns the conversational AI stack (STT, LLM, TTS);
    Runway provides the visual layer: audio in, avatar video out.
    """

    sample_rate = 16000

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str | None] = NOT_GIVEN,
        preset_id: NotGivenOr[str | None] = NOT_GIVEN,
        max_duration: NotGivenOr[int] = NOT_GIVEN,
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
        if not avatar_id and not preset_id:
            raise RunwayException("Either avatar_id or preset_id must be provided")
        if avatar_id and preset_id:
            raise RunwayException("Provide avatar_id or preset_id, not both")
        if avatar_id:
            self._avatar: dict[str, JsonValue] = {"type": "custom", "avatarId": str(avatar_id)}
        else:
            self._avatar = {"type": "runway-preset", "presetId": str(preset_id)}
        self._max_duration = max_duration

        self._api_url = api_url or os.getenv("RUNWAYML_BASE_URL", DEFAULT_API_URL)
        key = api_key or os.getenv("RUNWAYML_API_SECRET")
        if not key:
            raise RunwayException(
                "api_key must be set either by passing it to AvatarSession or "
                "by setting the RUNWAYML_API_SECRET environment variable"
            )
        self._headers = {"Authorization": f"Bearer {key}", "X-Runway-Version": API_VERSION}

    @property
    def provider(self) -> str:
        return "runway"

    async def create_session(self, ctx: AvatarStart) -> str:
        body: dict[str, JsonValue] = {
            "model": "gwm1_avatars",
            "avatar": self._avatar,
            "livekit": {
                "url": ctx.livekit_url,
                "token": ctx.livekit_token(),
                "roomName": ctx.room.name,
                "agentIdentity": ctx.agent_identity,
            },
        }
        if utils.is_given(self._max_duration):
            body["maxDuration"] = self._max_duration
        async with http.http_session().post(
            f"{self._api_url}/v1/realtime_sessions",
            headers=self._headers,
            json=body,
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()
            return _Session.model_validate_json(await response.text()).id

    async def end_session(self, session_id: str) -> None:
        room = self._room
        if room is not None and room.isconnected():
            # Runway ends a session told END_CALL as completed, rather than cancelled
            try:
                await room.local_participant.publish_data(
                    json.dumps({"type": "END_CALL"}).encode(),
                    reliable=True,
                    destination_identities=[self.avatar_identity],
                )
                return
            except Exception as e:
                logger.warning("ending the Runway session in the room failed", exc_info=e)
        # the room is gone: cancel the session so it stops billing before maxDuration
        async with http.http_session().delete(
            f"{self._api_url}/v1/realtime_sessions/{session_id}", headers=self._headers
        ) as response:
            await response.raise_for_status()
