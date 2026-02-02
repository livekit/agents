from __future__ import annotations

import asyncio
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
    """
    An asynchronous client for interacting with the Avatario API.

    This class handles authentication, request signing, and retries.
    """

    def __init__(
        self,
        api_key: str,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        video_info: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Initializes the AvatarioAPI client.
        """
        self._avatar_id = avatar_id
        self._api_key = api_key

        self._video_info = video_info
        self._conn_options = conn_options
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> AvatarioAPI:
        if self._owns_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        if self._owns_session and self._session:
            await self._session.close()

    async def start_session(
        self,
        *,
        livekit_agent_identity: NotGivenOr[str] = NOT_GIVEN,
        properties: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        """
        Initiate a session request

        Args:
            livekit_agent_identity: the participant identity of the agent in room.
            properties: A dictionary consisting of room url and token used by the
                        avatario participant to join the room.
        """
        if not utils.is_given(livekit_agent_identity):
            raise AvatarioException(
                "the identity of agent needs to be provided "
                "to ensure its proper communication with avatario backend"
            )

        properties = properties if utils.is_given(properties) else {}

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
            payload: JSON payload for the request

        Returns:
            Response data as a dictionary

        Raises:
            APIConnectionError: If the request fails after all retries
        """

        if self._session is None:
            raise RuntimeError("Session not initialized. Use 'async with AvatarioAPI(...)'.")
        last_exc: Exception | None = None
        for i in range(self._conn_options.max_retry):
            try:
                async with self._session.post(
                    f"{DEFAULT_API_URL}/start-session",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self._api_key,
                    },
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
                ) as response:
                    if not response.ok:
                        text = await response.text()
                        raise APIStatusError(
                            "Server returned an error",
                            status_code=response.status,
                            body=text,
                        )
                    data = await response.json()
                    return cast(dict[str, Any], data)
            except APIStatusError as e:
                last_exc = e
                if not e.retryable:
                    raise
                logger.warning("Avatario API returned error", extra={"error": str(e)})
            except Exception as e:
                last_exc = e
                logger.exception("failed to call Avatario API")

            if i < self._conn_options.max_retry - 1:
                await asyncio.sleep(self._conn_options.retry_interval)

        if isinstance(last_exc, APIStatusError):
            raise last_exc
        raise APIConnectionError("Failed to call Avatario API after all retries")
