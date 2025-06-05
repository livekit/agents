from __future__ import annotations

import asyncio
import io
import os

import aiohttp
from PIL.Image import Image

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    NotGivenOr,
    utils,
)
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .log import logger

DEFAULT_API_URL = "https://api.hedra.com/public/livekit/v1/session"
_AVATAR_AGENT_IDENTITY = "hedra-avatar-agent"
_AVATAR_AGENT_NAME = "hedra-avatar-agent"


class HedraException(Exception):
    """Exception for Hedra errors"""


class AvatarSession:
    """A Hedra avatar session"""

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str | None] = NOT_GIVEN,
        avatar_image: NotGivenOr[Image] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._avatar_id = avatar_id
        self._avatar_image = avatar_image
        if not self._avatar_id and not self._avatar_image:
            raise HedraException("avatar_id or avatar_image must be provided")

        self._api_url = api_url or os.getenv("HEDRA_API_URL", DEFAULT_API_URL)
        self._api_key = api_key or os.getenv("HEDRA_API_KEY")
        if self._api_key is None:
            raise HedraException(
                "The api_key must be set either by passing api_key to the client or "
                "by setting the HEDRA_API_KEY environment variable"
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
        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise HedraException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        livekit_token = (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
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
        assert isinstance(self._api_url, str)

        data = aiohttp.FormData({"livekit_url": livekit_url, "livekit_token": livekit_token})

        if self._avatar_id:
            data.add_field("avatar_id", self._avatar_id)

        if self._avatar_image:
            img_byte_arr = io.BytesIO()
            self._avatar_image.save(img_byte_arr, format="JPEG", quality=95)
            img_byte_arr.seek(0)
            data.add_field(
                "avatar_image", img_byte_arr, filename="avatar.jpg", content_type="image/jpeg"
            )

        for i in range(self._conn_options.max_retry):
            try:
                async with self._ensure_http_session().post(
                    self._api_url,
                    headers={
                        "x-api-key": self._api_key,
                    },
                    data=data,
                    timeout=aiohttp.ClientTimeout(sock_connect=self._conn_options.timeout),
                ) as response:
                    if not response.ok:
                        text = await response.text()
                        raise APIStatusError(
                            "Server returned an error", status_code=response.status, body=text
                        )
                    return

            except Exception as e:
                if isinstance(e, APIConnectionError):
                    logger.warning("failed to call hedra avatar api", extra={"error": str(e)})
                else:
                    logger.exception("failed to call hedra avatar api")

                if i < self._conn_options.max_retry - 1:
                    await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to start Hedra Avatar Session after all retries")
