import asyncio
import logging
import os
from typing import Any, Optional

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    utils,
)

logger = logging.getLogger(__name__)


class LiveAvatarException(Exception):
    """Exception for LiveAvatar errors"""


DEFAULT_API_URL = "https://api.liveavatar.com/v1/sessions"


class LiveAvatarAPI:
    def __init__(
        self,
        api_key: str,
        *,
        api_url: str = DEFAULT_API_URL,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("LIVEAVATAR_API_KEY")
        if self._api_key is None:
            raise LiveAvatarException("api_key or LIVEAVATAR_API_KEY must be set")

        self._api_url = api_url or DEFAULT_API_URL
        self._conn_options = conn_options
        self._session = session or aiohttp.ClientSession()

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    async def create_streaming_session(
        self,
        *,
        livekit_url: str,
        livekit_token: str,
        room: rtc.Room,
        avatar_id: str,
        is_sandbox: bool = False,
    ) -> dict[str, Any]:
        """Create a new streaming session, return a session id"""

        livekit_config = {
            "livekit_room": room.name,
            "livekit_url": livekit_url,
            "livekit_client_token": livekit_token,
        }

        payload = {
            "mode": "CUSTOM",
            "avatar_id": avatar_id,
            "is_sandbox": is_sandbox,
            "livekit_config": livekit_config,
        }

        self._headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-API-KEY": self._api_key,
        }
        response_data = await self._post(endpoint="/token", payload=payload, headers=self._headers)
        return response_data

    async def start_streaming_session(self, session_id: str, session_token: str) -> dict[str, Any]:
        """Start the streaming session"""
        payload = {"session_id": session_id}
        headers = {"content-type": "application/json", "Authorization": f"Bearer {session_token}"}
        response_data = await self._post(endpoint="/start", payload=payload, headers=headers)
        return response_data

    async def stop_streaming_session(self, session_id: str, session_token: str) -> dict[str, Any]:
        """Stop the streaming session"""
        payload = {
            "session_id": session_id,
            "reason": "USER_DISCONNECTED",
        }
        headers = {"content-type": "application/json", "Authorization": f"Bearer {session_token}"}
        response_data = await self._post(endpoint="/stop", payload=payload, headers=headers)
        return response_data

    async def _post(
        self, *, endpoint: str, payload: dict[str, Any], headers: dict[str, Any]
    ) -> dict[str, Any]:
        url = self._api_url + endpoint
        for i in range(self._conn_options.max_retry):
            try:
                async with self._ensure_http_session().post(
                    url=url, headers=headers, json=payload
                ) as response:
                    if not response.ok:
                        text = await response.text()
                        raise APIStatusError(
                            f"Server returned an error for {url}: {response.status}",
                            status_code=response.status,
                            body=text,
                        )
                    return await response.json()  # type: ignore
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(
                    f"API request to {url} failed on attempt {i}",
                    extra={"error": str(e)},
                )
            except Exception:
                logger.exception("failed to call LiveAvatar API")

            if i < self._conn_options.max_retry - 1:
                await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to call LiveAvatar API after all retries.")
