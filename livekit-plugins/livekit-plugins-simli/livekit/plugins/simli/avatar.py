from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit import api, rtc
from livekit.agents import (
    NOT_GIVEN,
    AgentSession,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .log import logger

SAMPLE_RATE = 16000
_AVATAR_AGENT_IDENTITY = "simli-avatar-agent"
_AVATAR_AGENT_NAME = "simli-avatar-agent"


@dataclass
class SimliConfig:
    """
    Args:
        api_key (str): Simli API Key
        face_id (str): Simli Face ID
        emotion_id (str):
            Emotion ID for Trinity Faces, defaults to happy_0.
            See https://docs.simli.com/emotions
        max_session_length (int):
            Absolute maximum session duration, avatar will disconnect after this time
            even if it's speaking.
        max_idle_time (int):
            Maximum duration the avatar is not speaking for before the avatar disconnects.
    """

    api_key: str
    face_id: str
    emotion_id: str = "92f24a0c-f046-45df-8df0-af7449c04571"
    max_session_length: int = 600
    max_idle_time: int = 30

    def create_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        result["faceId"] = f"{self.face_id}/{self.emotion_id}"
        result["handleSilence"] = True
        result["maxSessionLength"] = self.max_session_length
        result["maxIdleTime"] = self.max_idle_time
        return result


class AvatarSession:
    """A Simli avatar session"""

    def __init__(
        self,
        *,
        simli_config: SimliConfig,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        self._http_session: aiohttp.ClientSession | None = None
        self.conversation_id: str | None = None
        self._simli_config = simli_config
        self.api_url = api_url or "https://api.simli.ai"
        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._ensure_http_session()

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = utils.http_context.http_session()

        return self._http_session

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise Exception(
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

        logger.debug("starting avatar session")
        try:
            simli_session_token_request = await self._ensure_http_session().post(
                f"{self.api_url}/compose/token",
                json=self._simli_config.create_json(),
                headers={"x-simli-api-key": self._simli_config.api_key},
            )
            body = await simli_session_token_request.text()
            simli_session_token_request.raise_for_status()
        except Exception:
            logger.error(
                f"failed to create simli session token server returned {simli_session_token_request.status} and detail {body}"
            )
            return
        try:
            avatarConnectionRequest = await self._ensure_http_session().post(
                f"{self.api_url}/integrations/livekit/agents",
                json={
                    "session_token": (json.loads(body))["session_token"],
                    "livekit_token": livekit_token,
                    "livekit_url": livekit_url,
                },
            )
            body = await avatarConnectionRequest.text()
            avatarConnectionRequest.raise_for_status()
        except Exception:
            logger.error(
                f"failed to connect to simli avatar session server returned {avatarConnectionRequest.status} and detail {body}"
            )
            agent_session.output.audio = DataStreamAudioOutput(
                room=room,
                destination_identity=self._avatar_participant_identity,
                sample_rate=SAMPLE_RATE,
            )
