from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from json import JSONDecodeError
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel, JsonValue, ValidationError

if TYPE_CHECKING:
    from .avatar import VideoQuality

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


class SessionToken(BaseModel):
    session_id: str
    session_token: str


class SessionStart(BaseModel):
    ws_url: str


_DataT = TypeVar("_DataT")


class SessionResponse(BaseModel, Generic[_DataT]):
    data: _DataT


class SessionStop(BaseModel):
    code: int


_ResponseT = TypeVar("_ResponseT", bound=BaseModel)


class LiveAvatarAPI:
    def __init__(
        self,
        api_key: str,
        *,
        api_url: str = DEFAULT_API_URL,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("LIVEAVATAR_API_KEY")
        if resolved_api_key is None:
            raise LiveAvatarException("api_key or LIVEAVATAR_API_KEY must be set")
        self._api_key = resolved_api_key

        self._api_url = api_url or DEFAULT_API_URL
        self._conn_options = conn_options
        self._session = session

    def http_session(self) -> aiohttp.ClientSession:
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
        video_quality: VideoQuality | None = None,
    ) -> SessionResponse[SessionToken]:
        """Create a new streaming session, return a session id"""

        livekit_config: dict[str, JsonValue] = {
            "livekit_room": room.name,
            "livekit_url": livekit_url,
            "livekit_client_token": livekit_token,
        }

        payload: dict[str, JsonValue] = {
            "mode": "LITE",
            "avatar_id": avatar_id,
            "is_sandbox": is_sandbox,
            "livekit_config": livekit_config,
        }

        if video_quality is not None:
            payload["video_quality"] = video_quality

        self._headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-API-KEY": self._api_key,
        }
        response_data = await self._post(
            endpoint="/token",
            payload=payload,
            headers=self._headers,
            response_type=SessionResponse[SessionToken],
        )
        return response_data

    async def start_streaming_session(
        self, session_id: str, session_token: str
    ) -> SessionResponse[SessionStart]:
        """Start the streaming session"""
        payload = {"session_id": session_id}
        headers = {"content-type": "application/json", "Authorization": f"Bearer {session_token}"}
        response_data = await self._post(
            endpoint="/start",
            payload=payload,
            headers=headers,
            response_type=SessionResponse[SessionStart],
        )
        return response_data

    async def stop_streaming_session(self, session_id: str, session_token: str) -> SessionStop:
        """Stop the streaming session"""
        payload = {
            "session_id": session_id,
            "reason": "USER_DISCONNECTED",
        }
        headers = {"content-type": "application/json", "Authorization": f"Bearer {session_token}"}
        response_data = await self._post(
            endpoint="/stop", payload=payload, headers=headers, response_type=SessionStop
        )
        return response_data

    async def _post(
        self,
        *,
        endpoint: str,
        payload: Mapping[str, JsonValue],
        headers: Mapping[str, str],
        response_type: type[_ResponseT],
    ) -> _ResponseT:
        url = self._api_url + endpoint
        for i in range(self._conn_options.max_retry + 1):
            try:
                async with self.http_session().post(
                    url=url, headers=headers, json=payload
                ) as response:
                    if not response.ok:
                        text = await response.text()
                        raise APIStatusError(
                            f"Server returned an error for {url}: {response.status}",
                            status_code=response.status,
                            body=text,
                        )
                    return response_type.model_validate(await response.json())
            except (ValidationError, JSONDecodeError, aiohttp.ContentTypeError) as exc:
                raise APIConnectionError(
                    "LiveAvatar returned an invalid response", retryable=False
                ) from exc
            except APIStatusError as e:
                if not e.retryable or i >= self._conn_options.max_retry:
                    raise
            except (TimeoutError, aiohttp.ClientError) as e:
                logger.warning(
                    f"API request to {url} failed on attempt {i}",
                    extra={"error": str(e)},
                )
            except Exception:
                logger.exception("failed to call LiveAvatar API")

            if i < self._conn_options.max_retry:
                await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to call LiveAvatar API after all retries.")
