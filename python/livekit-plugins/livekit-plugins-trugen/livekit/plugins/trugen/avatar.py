from __future__ import annotations

import os

import livekit.rtc as rtc
from livekit import api
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    VoiceSession,
    get_job_context,
    http,
)
from livekit.agents.job import _server_credentials
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .log import logger

_BASE_API_URL = "https://api.trugen.ai"
_AVATAR_AGENT_IDENTITY = "trugen-avatar"
_AVATAR_AGENT_NAME = "Trugen Avatar"
_DEFAULT_AVATAR_ID = "665a1170"


class TrugenException(Exception):
    """Exception for TruGen.AI errors"""


class AvatarSession(BaseAvatarSession):
    """TruGen Realtime Avatar Session"""

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str | None] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str | None] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__()
        self._avatar_id = (
            _DEFAULT_AVATAR_ID if avatar_id is NOT_GIVEN or avatar_id is None else avatar_id
        )
        self._api_key = os.getenv("TRUGEN_API_KEY") if api_key is NOT_GIVEN else api_key
        if not self._api_key:
            raise TrugenException(
                "The api_key not found; set this by passing api_key to the client or "
                "by setting the TRUGEN_API_KEY environment variable"
            )
        if avatar_participant_identity is NOT_GIVEN or avatar_participant_identity is None:
            self._avatar_participant_identity = _AVATAR_AGENT_IDENTITY
        else:
            self._avatar_participant_identity = str(avatar_participant_identity)

        if avatar_participant_name is NOT_GIVEN or avatar_participant_name is None:
            self._avatar_participant_name = _AVATAR_AGENT_NAME
        else:
            self._avatar_participant_name = str(avatar_participant_name)
        self._http_session: http.HttpSession | None = None
        self._conn_options = conn_options

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "trugen"

    def _ensure_http_session(self) -> http.HttpSession:
        if self._http_session is None:
            self._http_session = http.http_session()

        return self._http_session

    async def start(
        self,
        agent_session: VoiceSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        await super().start(agent_session, room)

        livekit_url, livekit_api_key, livekit_api_secret = _server_credentials(
            livekit_url, livekit_api_key, livekit_api_secret
        )
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise TrugenException(
                "livekit_url, livekit_api_key, and livekit_api_secret not found; "
                "either pass them as arguments here or set environment variables."
            )

        job_ctx = get_job_context()
        local_participant_identity = job_ctx.local_participant_identity
        livekit_token = (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            # allow the avatar agent to publish audio and video on behalf of your local agent
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})
            .to_jwt()
        )

        logger.debug("Starting Realtime Avatar Session")
        await self._start_session(livekit_url, livekit_token)

        self._attach_audio_output(
            agent_session,
            DataStreamAudioOutput(
                room=room,
                destination_identity=self._avatar_participant_identity,
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
            ),
        )

    async def _start_session(self, livekit_url: str, livekit_token: str) -> None:
        api_key = self._api_key
        assert isinstance(api_key, str)
        session = self._ensure_http_session()

        async def request() -> None:
            async with session.post(
                f"{_BASE_API_URL}/v1/sessions",
                headers={"x-api-key": api_key},
                json={
                    "avatar_id": self._avatar_id,
                    "livekit_url": livekit_url,
                    "livekit_token": livekit_token,
                },
                conn_options=self._conn_options,
            ) as response:
                await response.raise_for_status()

        await http.retry_request(request, conn_options=self._conn_options)
