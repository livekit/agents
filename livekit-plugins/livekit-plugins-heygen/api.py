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


class HeyGenException(Exception):
    """Exception for HeyGen errors"""


DEFAULT_API_URL = "https://api.heygen.com"


class HeyGenAPI:
    def __init__(
        self,
        api_key: str,
        *,
        api_url: str = DEFAULT_API_URL,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url or DEFAULT_API_URL
        self._conn_options = conn_options
        self._session = session or aiohttp.ClientSession()

    def _ensure_http_session(self):
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    async def create_streaming_session(
        self,
        *,
        livekit_url: str,
        livekit_token: str,
        room: rtc.Room,
        avatar_id: Optional[str] = None,
        quality: str = "high",
        version: str = "v2",
        video_encoding: str = "H264",
        voice: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a new streaming session, return a session id"""
        avatar_id = avatar_id or os.getenv("HEYGEN_AVATAR_ID", "default")

        env_quality = os.getenv("HEYGEN_STREAM_QUALITY")
        if env_quality:
            quality = env_quality.strip().lower()
        env_encoding = os.getenv("HEYGEN_VIDEO_ENCODING")
        if env_encoding:
            video_encoding = env_encoding.strip().upper()

        livekit_settings = {"room": room.name, "url": livekit_url, "token": livekit_token}

        payload = {
            "quality": quality,
            "avatar_id": avatar_id,
            "version": version,
            "video_encoding": video_encoding,
            "livekit_settings": livekit_settings,
        }

        self._headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": self._api_key,
        }
        response_data = await self._post(
            endpoint="/v1/streaming.new", payload=payload, headers=self._headers
        )

        if not response_data["data"].get("session_id"):
            raise HeyGenException("Unable to retrieve a session ID from API response")

        return response_data

    async def start_streaming_session(self, session_id: str) -> dict[str, Any]:
        """Start the streaming session"""
        payload = {"session_id": session_id}

        response_data = await self._post(
            endpoint="/v1/streaming.start", payload=payload, headers=self._headers
        )
        return response_data

    async def _post(self, *, endpoint: str, payload: dict[str, Any], headers: dict[str, Any]):
        url = self._api_url + endpoint
        if not headers:
            headers = self._headers
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
                f"API request to {url} failed on attempt",
                extra={"error": str(e)},
            )

        raise APIConnectionError("Failed to call HeyGen API after all retries.")
