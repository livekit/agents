from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput
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


class AvatarSession(BaseAvatarSession):
    """A Simli avatar session"""

    def __init__(
        self,
        *,
        simli_config: SimliConfig,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__()
        self._conn_options = conn_options
        self._http_session: aiohttp.ClientSession | None = None
        self.conversation_id: str | None = None
        self._simli_config = simli_config
        self.api_url = api_url or "https://api.simli.ai"
        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._ensure_http_session()

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "simli"

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
        await super().start(agent_session, room)

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
            token_body = await self._post_with_retry(
                f"{self.api_url}/compose/token",
                payload=self._simli_config.create_json(),
                headers={"x-simli-api-key": self._simli_config.api_key},
            )
            session_token = json.loads(token_body)["session_token"]
        except APIStatusError as e:
            logger.error(
                "failed to create simli session token",
                extra={"status_code": e.status_code, "lk.pii.body": e.body},
            )
            return
        except (APIConnectionError, ValueError, KeyError, TypeError) as e:
            logger.error("failed to create simli session token", extra={"error": repr(e)})
            return

        try:
            await self._post_with_retry(
                f"{self.api_url}/integrations/livekit/agents",
                payload={
                    "session_token": session_token,
                    "livekit_token": livekit_token,
                    "livekit_url": livekit_url,
                },
            )
        except APIStatusError as e:
            logger.error(
                "failed to connect to simli avatar session",
                extra={"status_code": e.status_code, "lk.pii.body": e.body},
            )
            return
        except APIConnectionError as e:
            logger.error("failed to connect to simli avatar session", extra={"error": repr(e)})
            return

        agent_session.output.replace_audio_tail(
            DataStreamAudioOutput(
                room=room,
                destination_identity=self._avatar_participant_identity,
                sample_rate=SAMPLE_RATE,
            ),
        )

    async def _post_with_retry(
        self, url: str, *, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> str:
        """POST `payload` and return the response body, retrying 5xx and network errors.

        Raises:
            APIStatusError: If Simli returns a non-retryable error, or a retryable one
                persists after all retries
            APIConnectionError: If the request fails after all retries
        """
        session = self._ensure_http_session()
        # Keep the session's total limit: a timeout with only sock_connect has none.
        timeout = aiohttp.ClientTimeout(
            total=session.timeout.total, sock_connect=self._conn_options.timeout
        )
        for attempt in range(self._conn_options.max_retry + 1):
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                ) as response:
                    body = await response.text()
                    if not response.ok:
                        raise APIStatusError(
                            "Server returned an error", status_code=response.status, body=body
                        )
                    return body
            except APIStatusError as e:
                # A 4xx such as a bad API key will fail the same way every time.
                if not e.retryable:
                    raise
                logger.warning(
                    "failed to call simli api",
                    extra={"attempt": attempt + 1, "status_code": e.status_code},
                )
                if attempt >= self._conn_options.max_retry:
                    raise
                await asyncio.sleep(self._conn_options.retry_interval)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(
                    "failed to call simli api", extra={"attempt": attempt + 1, "error": str(e)}
                )
                if attempt >= self._conn_options.max_retry:
                    raise APIConnectionError("Failed to call Simli API after all retries") from e
                await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to call Simli API after all retries")
