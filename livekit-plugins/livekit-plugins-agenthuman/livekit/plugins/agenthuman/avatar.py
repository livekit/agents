from __future__ import annotations

import json
import os
from typing import Any

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

from .api import DEFAULT_API_URL, AgentHumanAPI
from .errors import AgentHumanException
from .log import logger

SAMPLE_RATE = 16000
# public stock avatar, used when neither avatar_id nor avatar_image_url is given
DEFAULT_AVATAR_ID = "avat_01KMZJ3YN6BR83KR1FHP8VTHNX"
_AVATAR_AGENT_IDENTITY = "agenthuman-avatar-agent"
_AVATAR_AGENT_NAME = "agenthuman-avatar-agent"
_SESSION_STATE_TOPIC = "session.state"
# reasons the AgentHuman server sends when a session ends; anything else is logged as PII
_SESSION_END_REASONS = frozenset({"server", "user", "expiry"})


class AvatarSession(BaseAvatarSession):
    """An AgentHuman avatar session"""

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        avatar_image_url: NotGivenOr[str] = NOT_GIVEN,
        aspect_ratio: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            avatar_id: ID of an AgentHuman avatar (``avat_...``). Falls back to
                ``AGENTHUMAN_AVATAR_ID``, then to a public stock avatar when no image is given.
            avatar_image_url: Publicly accessible image to animate instead of a saved avatar.
                Falls back to ``AGENTHUMAN_IMAGE_URL``. Provide at most one of ``avatar_id``
                and ``avatar_image_url``.
            aspect_ratio: Aspect ratio of the avatar video: ``"4:3"`` (default), ``"3:4"``
                or ``"1:1"``.
            api_url: AgentHuman API URL. Falls back to ``AGENTHUMAN_API_URL``.
            api_key: AgentHuman API key. Falls back to ``AGENTHUMAN_API_KEY``.
            avatar_participant_identity: Identity of the avatar participant in the room.
            avatar_participant_name: Name of the avatar participant in the room.
            conn_options: Connection options for the AgentHuman API.
            **kwargs: Additional fields sent with the session request.
        """
        super().__init__()

        avatar_id = (
            avatar_id if utils.is_given(avatar_id) else os.getenv("AGENTHUMAN_AVATAR_ID", "")
        )
        avatar_image_url = (
            avatar_image_url
            if utils.is_given(avatar_image_url)
            else os.getenv("AGENTHUMAN_IMAGE_URL", "")
        )
        if avatar_id and avatar_image_url:
            raise AgentHumanException(
                "provide only one of avatar_id or avatar_image_url (or set "
                "AGENTHUMAN_AVATAR_ID or AGENTHUMAN_IMAGE_URL)"
            )
        # the API takes either in the same field
        self._avatar = avatar_id or avatar_image_url or DEFAULT_AVATAR_ID

        api_key_val = api_key if utils.is_given(api_key) else os.getenv("AGENTHUMAN_API_KEY")
        if not api_key_val:
            raise AgentHumanException(
                "AGENTHUMAN_API_KEY must be set by arguments or environment variables"
            )
        self._api_key = api_key_val
        self._api_url = (
            api_url if utils.is_given(api_url) else os.getenv("AGENTHUMAN_API_URL", DEFAULT_API_URL)
        )

        self._aspect_ratio = aspect_ratio
        self._conn_options = conn_options
        self._extra_payload: NotGivenOr[dict[str, Any]] = kwargs if kwargs else NOT_GIVEN
        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._session_id: str | None = None

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "agenthuman"

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def start(  # type: ignore[override]
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> str:
        await super().start(agent_session, room)

        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise AgentHumanException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
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

        room.on("data_received", self._on_data_received)

        # Route the agent's audio to the avatar before the session request, so speech
        # generated meanwhile is buffered until the avatar's video track arrives.
        agent_session.output.replace_audio_tail(
            DataStreamAudioOutput(
                room=room,
                destination_identity=self._avatar_participant_identity,
                sample_rate=SAMPLE_RATE,
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
                # the avatar reports when each reply is actually heard (lk.playback_started)
                wait_playback_start=True,
            ),
        )

        logger.debug("starting avatar session")
        async with AgentHumanAPI(
            api_key=self._api_key,
            api_url=self._api_url,
            conn_options=self._conn_options,
        ) as agenthuman_api:
            self._session_id = await agenthuman_api.create_session(
                avatar=self._avatar,
                livekit_url=livekit_url,
                livekit_token=livekit_token,
                aspect_ratio=self._aspect_ratio,
                extra_payload=self._extra_payload,
            )

        logger.debug("avatar session created", extra={"lk.pii.session_id": self._session_id})
        return self._session_id

    def _on_data_received(self, packet: rtc.DataPacket) -> None:
        if packet.topic != _SESSION_STATE_TOPIC:
            return
        if packet.participant is not None and packet.participant.identity != self.avatar_identity:
            return
        try:
            payload = json.loads(packet.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("failed to parse avatar session state")
            return

        state = payload.get("state")
        extra = {"state": state}
        if reason := payload.get("reason"):
            extra["reason" if reason in _SESSION_END_REASONS else "lk.pii.reason"] = reason
        if state in ("expired", "ended"):
            logger.warning("avatar session ended", extra=extra)
        else:
            logger.debug("avatar session state changed", extra=extra)

    async def aclose(self) -> None:
        if self._room is not None:
            self._room.off("data_received", self._on_data_received)
        await super().aclose()
