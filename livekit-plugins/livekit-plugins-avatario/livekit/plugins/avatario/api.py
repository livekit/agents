import asyncio
import os
from typing import Any, cast

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    NotGivenOr,
    utils,
)

from .log import logger


class AvatarioException(Exception):
    """Exception for Avatario errors"""


DEFAULT_API_URL = "https://avatario.ai/api/sdk"


class AvatarioAPI:
    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        video_info: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        if not avatar_id:
            raise AvatarioException("avatar_id must be set")
        self._avatar_id = avatar_id
        self._video_info = video_info
        self._api_key = api_key or os.getenv("AVATARIO_API_KEY")
        if self._api_key is None:
            raise AvatarioException("AVATARIO_API_KEY must be set")
        self._api_key = cast(str, self._api_key)

        self._conn_options = conn_options
        self._session = session or aiohttp.ClientSession()

    async def start_session(
        self,
        *,
        livekit_agent_identity: NotGivenOr[str] = NOT_GIVEN,
        properties: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        if not livekit_agent_identity:
            raise AvatarioException(
                "the identity of agent needs to be provided "
                "to ensure its proper communication with avatario backend"
            )

        properties = properties or {}

        payload = {
            "avatario_face_id": self._avatar_id,
            "agent_id": livekit_agent_identity,
            "livekit": properties,
        }

        if utils.is_given(self._video_info):
            payload.update(self._video_info)
        await self._post(payload)

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Make a POST request to the Avatario API with retry logic.

        Args:
            endpoint: API endpoint path (without leading slash)
            payload: JSON payload for the request

        Returns:
            Response data as a dictionary

        Raises:
            APIConnectionError: If the request fails after all retries
        """
        for i in range(self._conn_options.max_retry):
            try:
                async with self._session.post(
                    f"{DEFAULT_API_URL}/start-session",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self._api_key,
                    },
                    json=payload,
                    timeout=self._conn_options.timeout,
                ) as response:
                    if not response.ok:
                        text = await response.text()
                        raise APIStatusError(
                            "Server returned an error", status_code=response.status, body=text
                        )
                    return await response.json()
            except Exception as e:
                if isinstance(e, APIConnectionError):
                    logger.warning("failed to call Avatario API", extra={"error": str(e)})
                else:
                    logger.exception("failed to call avatario api")

                if i < self._conn_options.max_retry - 1:
                    await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to call Avatario API after all retries")
