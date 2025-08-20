import asyncio
import os
from typing import Any, Optional

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


class TavusException(Exception):
    """Exception for Tavus errors"""


DEFAULT_API_URL = "https://tavusapi.com/v2"


class TavusAPI:
    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        tavus_api_key = api_key or os.getenv("TAVUS_API_KEY")
        if tavus_api_key is None:
            raise TavusException("TAVUS_API_KEY must be set")
        self._api_key = tavus_api_key

        self._api_url = api_url or DEFAULT_API_URL
        self._conn_options = conn_options
        self._session = session or aiohttp.ClientSession()

    async def create_conversation(
        self,
        *,
        replica_id: NotGivenOr[str] = NOT_GIVEN,
        persona_id: NotGivenOr[str] = NOT_GIVEN,
        properties: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        extra_payload: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> str:
        replica_id = replica_id or (os.getenv("TAVUS_REPLICA_ID") or NOT_GIVEN)
        if not replica_id:
            raise TavusException("TAVUS_REPLICA_ID must be set")

        persona_id = persona_id or (os.getenv("TAVUS_PERSONA_ID") or NOT_GIVEN)
        if not persona_id:
            # create a persona if not provided
            persona_id = await self.create_persona()

        properties = properties or {}
        payload = {
            "replica_id": replica_id,
            "persona_id": persona_id,
            "properties": properties,
        }
        if utils.is_given(extra_payload):
            payload.update(extra_payload)

        if "conversation_name" not in payload:
            payload["conversation_name"] = utils.shortuuid("lk_conversation_")

        response_data = await self._post("conversations", payload)
        return response_data["conversation_id"]  # type: ignore

    async def create_persona(
        self,
        name: NotGivenOr[str] = NOT_GIVEN,
        *,
        extra_payload: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> str:
        name = name or utils.shortuuid("lk_persona_")

        payload = {
            "persona_name": name,
            "pipeline_mode": "echo",
            "layers": {
                "transport": {"transport_type": "livekit"},
            },
        }

        if utils.is_given(extra_payload):
            payload.update(extra_payload)

        response_data = await self._post("personas", payload)
        return response_data["persona_id"]  # type: ignore

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Make a POST request to the Tavus API with retry logic.

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
                    f"{self._api_url}/{endpoint}",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self._api_key,
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
                if isinstance(e, APIConnectionError):
                    logger.warning("failed to call tavus api", extra={"error": str(e)})
                else:
                    logger.exception("failed to call tavus api")

                if i < self._conn_options.max_retry - 1:
                    await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to call Tavus API after all retries")
