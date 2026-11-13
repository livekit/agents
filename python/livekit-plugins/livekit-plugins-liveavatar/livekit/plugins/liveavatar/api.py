from __future__ import annotations

import os
from collections.abc import Mapping
from json import JSONDecodeError
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel, JsonValue, ValidationError

if TYPE_CHECKING:
    from .avatar import VideoQuality

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIError,
    http,
)


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
        session: http.HttpSession | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("LIVEAVATAR_API_KEY")
        if resolved_api_key is None:
            raise LiveAvatarException("api_key or LIVEAVATAR_API_KEY must be set")
        self._api_key = resolved_api_key

        self._api_url = api_url or DEFAULT_API_URL
        self._conn_options = conn_options
        self._session = session

    def http_session(self) -> http.HttpSession:
        if self._session is None:
            self._session = http.http_session()
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
        session = self.http_session()

        async def request() -> _ResponseT:
            async with session.post(
                url, headers=headers, json=payload, conn_options=self._conn_options
            ) as response:
                await response.raise_for_status()
                try:
                    return response_type.model_validate(await response.json())
                except (ValidationError, JSONDecodeError) as exc:
                    raise APIError(
                        "LiveAvatar returned an invalid response", retryable=False
                    ) from exc

        return await http.retry_request(request, conn_options=self._conn_options)
