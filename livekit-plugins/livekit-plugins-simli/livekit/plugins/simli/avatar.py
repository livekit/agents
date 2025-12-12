from __future__ import annotations

import os
from abc import ABC
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, Union

import aiohttp

from livekit import api, rtc
from livekit.agents import (
    NOT_GIVEN,
    AgentSession,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.metrics import UsageCollector, VideoAvatarMetrics, log_metrics
from livekit.agents.voice.avatar import LatencyAudioOutput, attach_video_latency_listener
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .log import logger

TEvent = TypeVar("TEvent")

SAMPLE_RATE = 16000
_AVATAR_AGENT_IDENTITY = "simli-avatar-agent"
_AVATAR_AGENT_NAME = "simli-avatar-agent"


@dataclass
class SimliConfig:
    api_key: str
    face_id: str
    emotion_id: str = "92f24a0c-f046-45df-8df0-af7449c04571"
    max_session_length: int = 600
    max_idle_time: int = 30

    def create_json(self) -> dict[str, Any]:
        return {
            "apiKey": self.api_key,
            "faceId": f"{self.face_id}/{self.emotion_id}",
            "syncAudio": True,
            "handleSilence": True,
            "maxSessionLength": self.max_session_length,
            "maxIdleTime": self.max_idle_time,
        }


class AvatarSession(ABC, rtc.EventEmitter[Union[Literal["metrics_collected", "error"], TEvent]]):
    """A Simli avatar session with latency measurement"""

    def __init__(
        self,
        *,
        simli_config: SimliConfig,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        super().__init__()
        self._http_session: aiohttp.ClientSession | None = None
        self.conversation_id: str | None = None
        self._simli_config = simli_config
        self.api_url = api_url or "https://api.simli.ai"
        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._latency_tasks: set = set()
        self._ensure_http_session()
        self._usage_collector = UsageCollector()

        self.on("metrics_collected", self._on_metrics_collected)

    def _on_metrics_collected(self, ev: VideoAvatarMetrics):
        m = ev
        log_metrics(m)
        self._usage_collector.collect(m)

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
            raise Exception("livekit_url, livekit_api_key, and livekit_api_secret must be set")

        job_ctx = get_job_context()
        local_participant_identity = job_ctx.local_participant_identity
        livekit_token = (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})
            .to_jwt()
        )

        logger.debug("starting avatar session")
        simli_session_token = await self._ensure_http_session().post(
            f"{self.api_url}/startAudioToVideoSession", json=self._simli_config.create_json()
        )
        simli_session_token.raise_for_status()
        (
            await self._ensure_http_session().post(
                f"{self.api_url}/StartLivekitAgentsSession",
                json={
                    "session_token": (await simli_session_token.json())["session_token"],
                    "livekit_token": livekit_token,
                    "livekit_url": livekit_url,
                },
            )
        ).raise_for_status()

        audio_output = LatencyAudioOutput(
            room=room,
            destination_identity=self._avatar_participant_identity,
            sample_rate=SAMPLE_RATE,
        )
        agent_session.output.audio = audio_output

        attach_video_latency_listener(room, audio_output, self)

        logger.info("Avatar session started successfully")
