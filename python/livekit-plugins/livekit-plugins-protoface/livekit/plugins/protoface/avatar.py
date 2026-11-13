from __future__ import annotations

import os

from pydantic import BaseModel, JsonValue

from livekit import api
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    http,
    utils,
)
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, AvatarStart

from .errors import ProtofaceException
from .version import __version__

DEFAULT_API_URL = "https://api.protoface.com"
DEFAULT_STOCK_AVATAR_ID = "av_stock_001"


class _Session(BaseModel):
    id: str


class AvatarSession(BaseAvatarSession):
    """A Protoface avatar session for LiveKit Agents."""

    sample_rate = 16000

    def __init__(
        self,
        *,
        avatar_id: str = DEFAULT_STOCK_AVATAR_ID,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        max_duration_seconds: NotGivenOr[int] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Create a Protoface avatar session.

        Args:
            avatar_id: Protoface avatar ID to render. Defaults to the stable
                stock avatar ID `av_stock_001`.
            api_url: Protoface API base URL. Defaults to `PROTOFACE_API_URL` or
                the public Protoface API.
            api_key: Protoface API key. Defaults to `PROTOFACE_API_KEY`.
            max_duration_seconds: Optional maximum session duration. Protoface
                applies the lower of this value and the account plan limit.
            avatar_participant_identity: LiveKit participant identity for the
                avatar worker.
            avatar_participant_name: Display name for the avatar participant.
            livekit_url: LiveKit server URL. Defaults to `LIVEKIT_URL`.
            livekit_api_key: LiveKit API key. Defaults to `LIVEKIT_API_KEY`.
            livekit_api_secret: LiveKit API secret. Defaults to `LIVEKIT_API_SECRET`.
            conn_options: Timeout and retry options for Protoface API requests.

        Raises:
            ProtofaceException: If no API key is provided.
        """
        super().__init__(
            avatar_participant_identity=avatar_participant_identity,
            avatar_participant_name=avatar_participant_name,
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
            conn_options=conn_options,
        )
        key = api_key or os.getenv("PROTOFACE_API_KEY")
        if not key:
            raise ProtofaceException(
                "api_key must be set by passing it to AvatarSession or "
                "setting the PROTOFACE_API_KEY environment variable"
            )
        self._headers = {
            "Authorization": f"Bearer {key}",
            "User-Agent": f"livekit-plugins-protoface/{__version__}",
        }
        self._api_url = (api_url or os.getenv("PROTOFACE_API_URL") or DEFAULT_API_URL).rstrip("/")
        self._avatar_id = avatar_id
        self._max_duration_seconds = max_duration_seconds
        self.session_id: str | None = None
        """Protoface session ID after `start()` succeeds, otherwise `None`."""

    @property
    def provider(self) -> str:
        return "protoface"

    async def create_session(self, ctx: AvatarStart) -> str:
        grants = api.VideoGrants(
            room_join=True,
            room=ctx.room.name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        )
        body: dict[str, JsonValue] = {
            "avatar_id": self._avatar_id,
            "transport": {
                "type": "livekit",
                "url": ctx.livekit_url,
                "room_name": ctx.room.name,
                "worker_token": ctx.livekit_token(grants=grants),
                "worker_identity": ctx.avatar_identity,
                "audio_source": "data_stream",
            },
        }
        if utils.is_given(self._max_duration_seconds):
            body["max_duration_seconds"] = self._max_duration_seconds
        async with http.http_session().post(
            f"{self._api_url}/v1/sessions",
            headers=self._headers,
            json=body,
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()
            self.session_id = _Session.model_validate_json(await response.text()).id
        return self.session_id

    async def end_session(self, session_id: str) -> None:
        async with http.http_session().post(
            f"{self._api_url}/v1/sessions/{session_id}/end", headers=self._headers
        ) as response:
            await response.raise_for_status()
        self.session_id = None


__all__ = ["DEFAULT_STOCK_AVATAR_ID", "AvatarSession"]
