from __future__ import annotations

import asyncio
import os
from typing import Any

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


class LemonSliceException(Exception):
    """Exception for LemonSlice errors"""


DEFAULT_API_URL = "https://lemonslice.com/api/liveai/sessions"


class LemonSliceAPI:
    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Initializes the LemonSliceAPI client.

        Args:
            api_key: Your LemonSlice API key. If not provided, it is read from
                    the LEMONSLICE_API_KEY environment variable.
            api_url: The base URL of the LemonSlice API.
            conn_options: Connection options for the aiohttp session.
            session: An optional existing aiohttp.ClientSession to use for requests.
        """
        ls_api_key = api_key if utils.is_given(api_key) else os.getenv("LEMONSLICE_API_KEY")
        if not ls_api_key:
            raise LemonSliceException("LEMONSLICE_API_KEY must be set")
        self._api_key = ls_api_key

        self._api_url = api_url or DEFAULT_API_URL
        self._conn_options = conn_options
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> LemonSliceAPI:
        if self._owns_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def start_agent_session(
        self,
        *,
        livekit_url: str,
        livekit_token: str,
        agent_id: NotGivenOr[str] = NOT_GIVEN,
        agent_image_url: NotGivenOr[str] = NOT_GIVEN,
        agent_prompt: NotGivenOr[str] = NOT_GIVEN,
        idle_timeout: NotGivenOr[int] = NOT_GIVEN,
        extra_payload: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> str:
        """
        Initiates a new LemonSlice agent session.

        Args:
            livekit_url: The LiveKit Cloud server URL.
            livekit_token: The LiveKit access token for the agent.
            agent_id: The ID of the LemonSlice agent to add to the session.
            agent_image_url: The URL of the image to use as the agent's avatar.
            agent_prompt: A prompt that subtly influences the avatar's movements and expressions.
            idle_timeout: The idle timeout, in seconds.
            extra_payload: Additional payload to include in the request.

        Returns:
            The unique session ID for the LemonSlice agent session.
        """
        if not utils.is_given(agent_id) and not utils.is_given(agent_image_url):
            raise LemonSliceException("Missing agent_id or agent_image_url")

        if utils.is_given(agent_id) and utils.is_given(agent_image_url):
            raise LemonSliceException("Only one of agent_id or agent_image_url can be provided")

        payload: dict[str, Any] = {
            "transport_type": "livekit",
            "properties": {
                "livekit_url": livekit_url,
                "livekit_token": livekit_token,
            },
        }

        if utils.is_given(agent_id):
            payload["agent_id"] = agent_id
        if utils.is_given(agent_image_url):
            payload["agent_image_url"] = agent_image_url
        if utils.is_given(agent_prompt):
            payload["agent_prompt"] = agent_prompt
        if utils.is_given(idle_timeout):
            payload["idle_timeout"] = idle_timeout
        if utils.is_given(extra_payload):
            payload.update(extra_payload)

        response_data = await self._post(payload)
        return response_data["session_id"]  # type: ignore

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Make a POST request to the LemonSlice API with retry logic.

        Args:
            payload: JSON payload for the request

        Returns:
            Response data as a dictionary

        Raises:
            APIConnectionError: If the request fails after all retries
        """
        session = self._session or aiohttp.ClientSession()
        try:
            for i in range(self._conn_options.max_retry + 1):
                try:
                    async with session.post(
                        self._api_url,
                        headers={
                            "Content-Type": "application/json",
                            "X-API-Key": self._api_key,
                        },
                        json=payload,
                        timeout=aiohttp.ClientTimeout(sock_connect=self._conn_options.timeout),
                    ) as response:
                        if not response.ok:
                            text = await response.text()
                            raise APIStatusError(
                                "Server returned an error", status_code=response.status, body=text
                            )
                        return await response.json()  # type: ignore
                except Exception as e:
                    if isinstance(e, APIStatusError) and not e.retryable:
                        raise APIConnectionError(
                            "Failed to call LemonSlice API with non-retryable error",
                            retryable=False,
                        ) from e

                    if isinstance(e, APIConnectionError):
                        logger.warning("failed to call LemonSlice api", extra={"error": str(e)})
                    else:
                        logger.exception("failed to call lemonslice api")

                    if i < self._conn_options.max_retry:
                        await asyncio.sleep(self._conn_options._interval_for_retry(i))
        finally:
            if not self._session:  # if we created the session, we close it
                await session.close()

        raise APIConnectionError("Failed to call LemonSlice API after all retries")
