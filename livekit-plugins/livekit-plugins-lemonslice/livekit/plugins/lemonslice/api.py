import asyncio
import os

import aiohttp

from typing import Any, Optional
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
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        ls_api_key = api_key or os.getenv("LEMONSLICE_API_KEY")
        if ls_api_key is None:
            raise LemonSliceException("LEMONSLICE_API_KEY must be set")
        self._api_key = ls_api_key

        self._api_url = api_url or DEFAULT_API_URL
        self._conn_options = conn_options
        self._session = session or aiohttp.ClientSession()

    async def start_agent_session(
        self,
        *,
        agent_id: NotGivenOr[str] = NOT_GIVEN,
        agent_image_url: NotGivenOr[str] = NOT_GIVEN,
        agent_prompt: NotGivenOr[str] = NOT_GIVEN,
        idle_timeout: NotGivenOr[int] = NOT_GIVEN,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_token: NotGivenOr[str] = NOT_GIVEN,
        extra_payload: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> str:

        if not utils.is_given(agent_id) and not utils.is_given(agent_image_url):
            raise LemonSliceException("Missing agent_id or agent_image_url")
        
        if utils.is_given(agent_id) and utils.is_given(agent_image_url):
            raise LemonSliceException("Only one of agent_id or agent_image_url can be provided")

        payload = {
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
        for i in range(self._conn_options.max_retry):
            try:
                async with self._session.post(
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
                        raise APIStatusError("Server returned an error", status_code=response.status, body=text)
                    return await response.json()  # type: ignore
            except Exception as e:
                if isinstance(e, APIConnectionError):
                    logger.warning("failed to call LemonSlice api", extra={"error": str(e)})
                else:
                    logger.exception("failed to call lemonslice api")

                if i < self._conn_options.max_retry - 1:
                    await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to call LemonSlice API after all retries")
