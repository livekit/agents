from __future__ import annotations

import asyncio
import os

import aiohttp

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    APIConnectOptions,
    APIStatusError,
    NotGivenOr,
    utils,
)
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .log import logger

EGE_STOCK_AVATAR_ID = "b9be11b8-89fb-4227-8f86-4a881393cbdb"
_DEFAULT_API_URL = "https://api.bey.dev"

_AVATAR_AGENT_IDENTITY = "bey-avatar-agent"
_AVATAR_AGENT_NAME = "bey-avatar-agent"


class BeyException(Exception):
    """Exception for Beyond Presence errors"""


class AvatarSession:
    """A Beyond Presence avatar session"""

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str | None] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._avatar_id = avatar_id or EGE_STOCK_AVATAR_ID
        self._api_url = api_url or os.getenv("BEY_API_URL", _DEFAULT_API_URL)
        self._api_key = api_key or os.getenv("BEY_API_KEY")
        if self._api_key is None:
            raise BeyException(
                "The api_key must be set either by passing api_key to the client or "
                "by setting the BEY_API_KEY environment variable"
            )

        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
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
        livekit_url = livekit_url or os.getenv("LIVEKIT_URL")
        livekit_api_key = livekit_api_key or os.getenv("LIVEKIT_API_KEY")
        livekit_api_secret = livekit_api_secret or os.getenv("LIVEKIT_API_SECRET")
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise BeyException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        livekit_token = (
            api.AccessToken()
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            # allow the avatar agent to publish audio and video on behalf of your local agent
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: room.local_participant.identity})
            .to_jwt()
        )

        logger.debug("starting avatar session")
        await self._start_agent(livekit_url, livekit_token)

        logger.debug("waiting for avatar agent to join the room")
        await utils.wait_for_participant(room=room, identity=self._avatar_participant_identity)

        agent_session.output.audio = DataStreamAudioOutput(
            room=room,
            destination_identity=self._avatar_participant_identity,
        )

    async def _start_agent(self, livekit_url: str, livekit_token: str) -> None:
        assert self._api_key is not None

        for i in range(self._conn_options.max_retry):
            try:
                async with self._ensure_http_session().post(
                    f"{self._api_url}/v1/session",
                    headers={
                        "x-api-key": self._api_key,
                    },
                    json={
                        "avatar_id": self._avatar_id,
                        "livekit_url": livekit_url,
                        "livekit_token": livekit_token,
                    },
                    timeout=self._conn_options.timeout,
                ) as response:
                    if not response.ok:
                        text = await response.text()
                        logger.warning(
                            "failed to start avatar session",
                            extra={"status": response.status, "error": text, "tried": i + 1},
                        )
                        raise APIStatusError()
                    return
            except Exception:
                await asyncio.sleep(self._conn_options.retry_interval)

        raise APIStatusError(
            f"Failed to start Bey Avatar Session after {self._conn_options.max_retry} retries"
        )
