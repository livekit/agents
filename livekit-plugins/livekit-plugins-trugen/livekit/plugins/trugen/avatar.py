from __future__ import annotations

import asyncio
import os

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
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .log import logger

_BASE_API_URL = "https://api.trugen.ai"
_AVATAR_AGENT_IDENTITY = "trugen-avatar"
_AVATAR_AGENT_NAME = "Trugen Avatar"
_DEFAULT_AVATAR_ID = "7d881c1b"


class TrugenException(Exception):
    """Exception for TruGen.AI errors"""


class AvatarSession:
    """TruGen Realtime Avatar Session"""

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str | None] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
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
        self._http_session: aiohttp.ClientSession | None = None
        self._conn_options = conn_options

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
        if livekit_url is NOT_GIVEN:
            livekit_url = os.getenv("LIVEKIT_URL") or NOT_GIVEN
        if livekit_api_key is NOT_GIVEN:
            livekit_api_key = os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN
        if livekit_api_secret is NOT_GIVEN:
            livekit_api_secret = os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN

        if (
            livekit_url is NOT_GIVEN
            or livekit_api_key is NOT_GIVEN
            or livekit_api_secret is NOT_GIVEN
            or not livekit_url
            or not livekit_api_key
            or not livekit_api_secret
        ):
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

        agent_session.output.audio = DataStreamAudioOutput(
            room=room,
            destination_identity=self._avatar_participant_identity,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO,
        )

    async def _start_session(self, livekit_url: str, livekit_token: str) -> None:
        assert self._api_key is not None
        api_key = self._api_key
        assert isinstance(api_key, str)
        for i in range(self._conn_options.max_retry + 1):
            try:
                async with self._ensure_http_session().post(
                    f"{_BASE_API_URL}/v1/sessions",
                    headers={
                        "x-api-key": api_key,
                    },
                    json={
                        "avatar_id": self._avatar_id,
                        "livekit_url": livekit_url,
                        "livekit_token": livekit_token,
                    },
                    timeout=aiohttp.ClientTimeout(sock_connect=self._conn_options.timeout),
                ) as response:
                    if not response.ok:
                        text = await response.text()
                        raise APIStatusError(
                            "Server returned an error", status_code=response.status, body=text
                        )
                    return

            except Exception as e:
                if isinstance(e, APIStatusError):
                    logger.warning(
                        "API Error; Unable to trigger TruGen.AI API backend.",
                        extra={"status_code": e.status_code, "body": e.body},
                    )
                    if not e.retryable:
                        raise
                else:
                    logger.warning(
                        "API Error; Unable to trigger TruGen.AI API backend.",
                        extra={"error": str(e)},
                    )

                if i < self._conn_options.max_retry:
                    await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Max retries exhausted; Unable to start TruGen.AI Avatar Session.")
