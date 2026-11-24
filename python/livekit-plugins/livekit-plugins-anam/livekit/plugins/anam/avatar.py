from __future__ import annotations

import os

from pydantic import BaseModel, Field, JsonValue

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    http,
    utils,
)
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, AvatarStart

from .errors import AnamException
from .types import PersonaConfig, SessionOptions

DEFAULT_API_URL = "https://api.anam.ai"


class _Session(BaseModel):
    session_id: str | None = Field(default=None, alias="sessionId")


class AvatarSession(BaseAvatarSession):
    """An Anam avatar session"""

    def __init__(
        self,
        *,
        persona_config: PersonaConfig,
        session_options: NotGivenOr[SessionOptions] = NOT_GIVEN,
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
        self.session_id: str | None = None
        self._persona_config = persona_config
        self._session_options = session_options if utils.is_given(session_options) else None
        if self._session_options is not None and (
            (self._session_options.video_width is None)
            != (self._session_options.video_height is None)
        ):
            # Anam rejects a lone width or height with a 400
            raise ValueError("video_width and video_height must be set together (both or neither)")
        self._api_url = (
            api_url if utils.is_given(api_url) else os.getenv("ANAM_API_URL", DEFAULT_API_URL)
        )
        api_key_val = api_key if utils.is_given(api_key) else os.getenv("ANAM_API_KEY")
        if not api_key_val:
            raise AnamException("ANAM_API_KEY must be set by arguments or environment variables")
        self._api_key = api_key_val

    @property
    def provider(self) -> str:
        return "anam"

    async def create_session(self, ctx: AvatarStart) -> str | None:
        persona = self._persona_config
        persona_payload: dict[str, JsonValue] = {
            "type": "ephemeral",
            "name": persona.name,
            "avatarId": persona.avatarId,
            "llmId": "CUSTOMER_CLIENT_V1",
        }
        if persona.avatarModel:
            persona_payload["avatarModel"] = persona.avatarModel
        if persona.directorNotes is not None:
            # unset notes fall back to Anam's model and cue defaults
            notes = {k: v for k, v in vars(persona.directorNotes).items() if v is not None}
            if notes:
                persona_payload["directorNotes"] = notes

        payload: dict[str, JsonValue] = {
            "personaConfig": persona_payload,
            "environment": {
                "livekitUrl": ctx.livekit_url,
                "livekitToken": await ctx.livekit_token(),
            },
        }
        options: dict[str, JsonValue] = {}
        if (session_options := self._session_options) is not None:
            if session_options.video_width is not None:
                options["videoWidth"] = session_options.video_width
                options["videoHeight"] = session_options.video_height
            if session_options.show_ai_avatar_disclosure is not None:
                options["showAIAvatarDisclosure"] = session_options.show_ai_avatar_disclosure
        if options:
            payload["sessionOptions"] = options

        async with http.http_session().post(
            f"{self._api_url}/v1/engine/session",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()
            self.session_id = _Session.model_validate_json(await response.text()).session_id
        return self.session_id
