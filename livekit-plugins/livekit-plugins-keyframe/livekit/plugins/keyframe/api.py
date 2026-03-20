from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
)

from .log import logger

DEFAULT_API_URL = "https://api.keyframelabs.com"


class KeyframeAPI:
    """Asynchronous client for the Keyframe Labs session API."""

    def __init__(
        self,
        api_key: str,
        api_url: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url
        self._conn_options = conn_options
        self._session = session
        self._own_session = session is None

    async def __aenter__(self) -> KeyframeAPI:
        if self._own_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        if self._own_session and self._session and not self._session.closed:
            await self._session.close()

    async def create_plugin_session(
        self,
        *,
        persona_id: str | None = None,
        persona_slug: str | None = None,
        room_name: str,
        livekit_url: str,
        livekit_token: str,
        source_participant_identity: str,
    ) -> dict[str, Any]:
        """Create a plugin session via POST /v1/sessions/plugins/livekit.

        Returns dict with reservation_id and avatar_participant_identity.
        """
        payload: dict[str, Any] = {
            "room_name": room_name,
            "livekit_url": livekit_url,
            "livekit_token": livekit_token,
            "source_participant_identity": source_participant_identity,
        }
        if persona_id:
            payload["persona_id"] = persona_id
        if persona_slug:
            payload["persona_slug"] = persona_slug

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        return await self._post("/v1/sessions/plugins/livekit", payload, headers)

    async def _post(
        self, endpoint: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        url = f"{self._api_url}{endpoint}"
        session = self._session or aiohttp.ClientSession()
        try:
            for i in range(self._conn_options.max_retry + 1):
                try:
                    async with session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(sock_connect=self._conn_options.timeout),
                    ) as response:
                        if not response.ok:
                            text = await response.text()
                            raise APIStatusError(
                                f"Keyframe API error for {url}: {response.status}",
                                status_code=response.status,
                                body=text,
                            )
                        return await response.json()  # type: ignore
                except Exception as e:
                    if isinstance(e, APIStatusError) and not e.retryable:
                        raise APIConnectionError(
                            f"Failed to call Keyframe API at {url} with non-retryable error",
                            retryable=False,
                        ) from e

                    if isinstance(e, APIConnectionError):
                        logger.warning("Failed to call Keyframe API", extra={"error": str(e)})
                    else:
                        logger.exception("Failed to call Keyframe API")

                    if i < self._conn_options.max_retry:
                        await asyncio.sleep(self._conn_options._interval_for_retry(i))
        finally:
            if not self._session:
                await session.close()

        raise APIConnectionError("Failed to call Keyframe API after all retries.")
