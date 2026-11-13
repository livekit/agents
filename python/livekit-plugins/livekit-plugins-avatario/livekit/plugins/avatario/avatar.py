from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from pydantic import JsonValue, TypeAdapter

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    http,
    utils,
)
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, AvatarStart

DEFAULT_API_URL = "https://avatario.ai/api/sdk"

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class AvatarioException(Exception):
    """Exception for Avatario errors"""


class AvatarSession(BaseAvatarSession):
    """An Avatario avatar session"""

    @dataclass
    class VideoInfo:
        video_height: int = 720
        video_width: int = 1280
        custom_background_url: str | None = None

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        video_info: NotGivenOr[VideoInfo] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Initialize Avatario avatar session

        Args:
            avatar_id: The ID of avatar to use in the session. If not provided it will read
                        from the AVATARIO_AVATAR_ID environment variable.
                        IDs of our stock avatars can be found here:-
                        (https://avatario.ai/dashboard/9pqbj80f/avatars/stock)
            video_info: a dataclass containing information about the video resolution
                        and background of the avatar session.
            api_url: The Avatario API's base URL.
            api_key: Your Avatario API key. If not provided, it will be read from
                     the AVATARIO_API_KEY environment variable.
            avatar_participant_identity: Identity of the avatario participant that will
                                         join the room. Defaults to "avatario-avatar-agent"
            avatar_participant_name: Name of the avatario participant that will join the
                                     room. Defaults to its identity.
            livekit_url: The URL the avatar joins through; ``LIVEKIT_URL`` by default.
            livekit_api_key: Signs the avatar's token; ``LIVEKIT_API_KEY`` by default.
            livekit_api_secret: Signs the avatar's token; ``LIVEKIT_API_SECRET`` by default.
            conn_options: Connection options for the Avatario API requests.
        """
        super().__init__(
            avatar_participant_identity=avatar_participant_identity,
            avatar_participant_name=avatar_participant_name,
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
            conn_options=conn_options,
        )
        self._video_info = asdict(video_info if utils.is_given(video_info) else self.VideoInfo())
        self._api_url = api_url if utils.is_given(api_url) else DEFAULT_API_URL

        avatario_avatar_id = (
            avatar_id if utils.is_given(avatar_id) else os.getenv("AVATARIO_AVATAR_ID")
        )
        if not avatario_avatar_id:
            raise AvatarioException("AVATARIO_AVATAR_ID must be set")
        self._avatar_id = avatario_avatar_id

        avatario_api_key = api_key if utils.is_given(api_key) else os.getenv("AVATARIO_API_KEY")
        if not avatario_api_key:
            raise AvatarioException("AVATARIO_API_KEY must be set")
        self._api_key = avatario_api_key

    @property
    def provider(self) -> str:
        return "avatario"

    async def create_session(self, ctx: AvatarStart) -> None:
        async with http.http_session().post(
            f"{self._api_url}/start-session",
            headers={"x-api-key": self._api_key},
            json={
                "avatario_face_id": self._avatar_id,
                "agent_id": ctx.agent_identity,
                "livekit": {"url": ctx.livekit_url, "token": ctx.livekit_token()},
                **self._video_info,
            },
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()
            _JSON_OBJECT.validate_json(await response.text())
