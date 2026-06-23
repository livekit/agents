from __future__ import annotations

import os

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    APIConnectOptions,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .api import ProtofaceAPI
from .errors import ProtofaceException
from .log import logger

DEFAULT_STOCK_AVATAR_ID = "av_stock_001"
SAMPLE_RATE = 16000
_AVATAR_AGENT_IDENTITY = "protoface-avatar-agent"
_AVATAR_AGENT_NAME = "protoface-avatar-agent"


class AvatarSession(BaseAvatarSession):
    """A Protoface avatar session for LiveKit Agents."""

    def __init__(
        self,
        *,
        avatar_id: str = DEFAULT_STOCK_AVATAR_ID,
        api_url: NotGivenOr[str | None] = NOT_GIVEN,
        api_key: NotGivenOr[str | None] = NOT_GIVEN,
        max_duration_seconds: NotGivenOr[int | None] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str | None] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str | None] = NOT_GIVEN,
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
            conn_options: Timeout and retry options for Protoface API requests.

        Raises:
            ProtofaceException: If no API key is provided.
        """
        super().__init__()
        self._avatar_id = avatar_id
        self._max_duration_seconds = max_duration_seconds
        self._avatar_participant_identity = _resolve_optional_string(
            avatar_participant_identity,
            _AVATAR_AGENT_IDENTITY,
        )
        self._avatar_participant_name = _resolve_optional_string(
            avatar_participant_name,
            _AVATAR_AGENT_NAME,
        )
        self._api = ProtofaceAPI(
            api_key=api_key,
            api_url=api_url,
            conn_options=conn_options,
        )
        self._session_id: str | None = None

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "protoface"

    @property
    def session_id(self) -> str | None:
        """Protoface session ID after `start()` succeeds, otherwise `None`."""
        return self._session_id

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str | None] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str | None] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        """Start the Protoface session and connect it to the LiveKit room.

        Args:
            agent_session: LiveKit agent session whose audio output should be
                routed through the avatar.
            room: Connected LiveKit room for the agent job.
            livekit_url: LiveKit server URL. Defaults to `LIVEKIT_URL`.
            livekit_api_key: LiveKit API key. Defaults to `LIVEKIT_API_KEY`.
            livekit_api_secret: LiveKit API secret. Defaults to
                `LIVEKIT_API_SECRET`.

        Raises:
            RuntimeError: If the same `AvatarSession` is started more than once.
            ProtofaceException: If required LiveKit credentials are missing or
                Protoface returns an invalid session response.
        """
        if self._session_id is not None:
            raise RuntimeError("AvatarSession.start() called twice; create a new AvatarSession.")

        await super().start(agent_session, room)

        livekit_url_value = _resolve_env_or_value(livekit_url, "LIVEKIT_URL")
        livekit_api_key_value = _resolve_env_or_value(livekit_api_key, "LIVEKIT_API_KEY")
        livekit_api_secret_value = _resolve_env_or_value(
            livekit_api_secret,
            "LIVEKIT_API_SECRET",
        )
        if not livekit_url_value or not livekit_api_key_value or not livekit_api_secret_value:
            raise ProtofaceException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        worker_token = self._mint_worker_token(
            room=room,
            livekit_api_key=livekit_api_key_value,
            livekit_api_secret=livekit_api_secret_value,
        )
        session = await self._api.start_session(
            avatar_id=self._avatar_id,
            transport={
                "type": "livekit",
                "url": livekit_url_value,
                "room_name": room.name,
                "worker_token": worker_token,
                "worker_identity": self._avatar_participant_identity,
                "audio_source": "data_stream",
            },
            max_duration_seconds=self._max_duration_seconds,
        )
        session_id = session.get("id")
        if not isinstance(session_id, str):
            raise ProtofaceException("Protoface API response missing session id")

        self._session_id = session_id
        logger.debug(
            "protoface session started",
            extra={"session_id": self._session_id, "avatar_id": self._avatar_id},
        )

        agent_session.output.replace_audio_tail(
            DataStreamAudioOutput(
                room=room,
                destination_identity=self._avatar_participant_identity,
                sample_rate=SAMPLE_RATE,
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
                clear_buffer_timeout=None,
            ),
        )

    async def aclose(self) -> None:
        """End the Protoface session and release LiveKit avatar resources."""
        session_id = self._session_id
        self._session_id = None
        try:
            if session_id is not None:
                try:
                    await self._api.end_session(session_id)
                except Exception:
                    logger.warning(
                        "failed to end protoface session",
                        extra={"session_id": session_id},
                        exc_info=True,
                    )
        finally:
            await super().aclose()

    def _mint_worker_token(
        self,
        *,
        room: rtc.Room,
        livekit_api_key: str,
        livekit_api_secret: str,
    ) -> str:
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            local_participant_identity = job_ctx.local_participant_identity
        elif room.isconnected():
            local_participant_identity = room.local_participant.identity
        else:
            raise ProtofaceException("failed to get local participant identity")

        if not local_participant_identity:
            raise ProtofaceException("failed to get local participant identity")

        grants = api.VideoGrants(
            room_join=True,
            room=room.name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        )
        return (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(grants)
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})
            .to_jwt()
        )


def _resolve_optional_string(value: NotGivenOr[str | None], default: str) -> str:
    if utils.is_given(value) and value:
        return value
    return default


def _resolve_env_or_value(value: NotGivenOr[str | None], env_name: str) -> str | None:
    if utils.is_given(value) and value is not None:
        return value
    return os.getenv(env_name)


__all__ = ["DEFAULT_STOCK_AVATAR_ID", "AvatarSession"]
