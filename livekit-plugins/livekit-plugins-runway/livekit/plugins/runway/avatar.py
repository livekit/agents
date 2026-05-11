from __future__ import annotations

import asyncio
import json
import os
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
from livekit.agents.utils import is_given
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .log import logger

DEFAULT_API_URL = "https://api.dev.runwayml.com"
API_VERSION = "2024-11-06"
SAMPLE_RATE = 16000
_AVATAR_AGENT_IDENTITY = "runway-avatar-agent"
_AVATAR_AGENT_NAME = "runway-avatar-agent"


class RunwayException(Exception):
    """Exception for Runway errors"""


class AvatarSession(BaseAvatarSession):
    """A Runway Characters avatar session.

    Creates a realtime session backed by Runway's avatar inference pipeline.
    The customer's LiveKit agent owns the conversational AI stack (STT, LLM, TTS);
    Runway provides the visual layer — audio in, avatar video out.
    """

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
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        if not avatar_id and not preset_id:
            raise RunwayException("Either avatar_id or preset_id must be provided")
        if avatar_id and preset_id:
            raise RunwayException("Provide avatar_id or preset_id, not both")

        if avatar_id:
            self._avatar: dict[str, str] = {"type": "custom", "avatarId": str(avatar_id)}
        else:
            self._avatar = {"type": "runway-preset", "presetId": str(preset_id)}
        self._max_duration = max_duration

        self._api_url = api_url or os.getenv("RUNWAYML_BASE_URL", DEFAULT_API_URL)
        self._api_key = api_key or os.getenv("RUNWAYML_API_SECRET")
        if self._api_key is None:
            raise RunwayException(
                "api_key must be set either by passing it to AvatarSession or "
                "by setting the RUNWAYML_API_SECRET environment variable"
            )

        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._http_session: aiohttp.ClientSession | None = None
        self._conn_options = conn_options
        self._room: rtc.Room | None = None
        self._realtime_session_id: str | None = None
        self._end_session_task: asyncio.Task[None] | None = None

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
        self._room = room

        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise RunwayException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        job_ctx = get_job_context()
        self._local_participant_identity = job_ctx.local_participant_identity

        livekit_token = (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: self._local_participant_identity})
            .to_jwt()
        )

        logger.debug("starting Runway avatar session")
        await self._create_session(livekit_url, livekit_token, room.name)

        @agent_session.on("close")
        def _on_agent_session_close(_: Any) -> None:
            self._ensure_end_session_task()

        agent_session.output.audio = DataStreamAudioOutput(
            room=room,
            destination_identity=self._avatar_participant_identity,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO,
            sample_rate=SAMPLE_RATE,
        )

    async def _create_session(self, livekit_url: str, livekit_token: str, room_name: str) -> None:
        body: dict[str, object] = {
            "model": "gwm1_avatars",
            "avatar": self._avatar,
            "livekit": {
                "url": livekit_url,
                "token": livekit_token,
                "roomName": room_name,
                "agentIdentity": self._local_participant_identity,
            },
        }

        if is_given(self._max_duration):
            body["maxDuration"] = self._max_duration

        for attempt in range(self._conn_options.max_retry):
            try:
                async with self._ensure_http_session().post(
                    f"{self._api_url}/v1/realtime_sessions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "X-Runway-Version": API_VERSION,
                    },
                    json=body,
                    timeout=aiohttp.ClientTimeout(sock_connect=self._conn_options.timeout),
                ) as response:
                    if not response.ok:
                        text = await response.text()
                        raise APIStatusError(
                            "Runway API returned an error",
                            status_code=response.status,
                            body=text,
                        )
                    payload = await response.json()
                    session_id = payload.get("id") if isinstance(payload, dict) else None
                    if isinstance(session_id, str):
                        self._realtime_session_id = session_id
                    return

            except Exception as error:
                if isinstance(error, APIStatusError) and not error.retryable:
                    raise

                if isinstance(error, APIConnectionError):
                    logger.warning(
                        "failed to call Runway avatar API",
                        extra={"error": str(error)},
                    )
                else:
                    logger.exception("failed to call Runway avatar API")

                if attempt < self._conn_options.max_retry - 1:
                    await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to start Runway Avatar Session after all retries")

    def _ensure_end_session_task(self) -> asyncio.Task[None] | None:
        if self._end_session_task is not None:
            return self._end_session_task

        if self._room is None:
            return None

        self._end_session_task = asyncio.create_task(
            self._end_runway_realtime_session(self._room),
            name="runway_end_realtime_session",
        )
        return self._end_session_task

    async def _end_runway_realtime_session(self, room: rtc.Room) -> None:
        # Preferred path: data-channel END_CALL while the room is still connected.
        # The Runway worker handles this message and shuts the session down through
        # the normal "user ended call" lifecycle (COMPLETED, not CANCELLED).
        if room.isconnected():
            try:
                await room.local_participant.publish_data(
                    json.dumps({"type": "END_CALL"}).encode("utf-8"),
                    reliable=True,
                    destination_identities=[self._avatar_participant_identity],
                )
                logger.debug("sent Runway realtime session end call")
                return
            except Exception as exc:
                logger.warning(
                    "error ending Runway realtime session via data channel",
                    extra={"error": str(exc)},
                )

        # Fallback for hard shutdowns where the room is already disconnected
        # (e.g. aclose() registered as a job shutdown callback runs after
        # room.disconnect()): cancel the session via API so we don't keep
        # billing until maxDuration.
        await self._cancel_runway_realtime_session()

    async def _cancel_runway_realtime_session(self) -> None:
        session_id = self._realtime_session_id
        if session_id is None:
            logger.warning("could not cancel Runway realtime session; no session id available")
            return

        try:
            async with self._ensure_http_session().delete(
                f"{self._api_url}/v1/realtime_sessions/{session_id}",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "X-Runway-Version": API_VERSION,
                },
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as response:
                if response.ok:
                    logger.debug(
                        "cancelled Runway realtime session",
                        extra={"session_id": session_id},
                    )
                else:
                    logger.warning(
                        "could not cancel Runway realtime session",
                        extra={
                            "session_id": session_id,
                            "status": response.status,
                            "body": await response.text(),
                        },
                    )
        except Exception as exc:
            logger.warning(
                "error cancelling Runway realtime session",
                extra={"session_id": session_id, "error": str(exc)},
            )

    async def aclose(self) -> None:
        if end_session_task := self._ensure_end_session_task():
            await asyncio.shield(end_session_task)
