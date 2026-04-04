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


class AgentHumanException(Exception):
    """Exception for AgentHuman errors"""


DEFAULT_AVATAR_ID = "avat_01KMZJ4WE5QCD9G3EK21QW8NJK"


class AgentHumanAPI:
    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        agenthuman_api_key = api_key or os.getenv("AGENTHUMAN_API_KEY")
        if agenthuman_api_key is None:
            raise AgentHumanException("AGENTHUMAN_API_KEY must be set")
        self._api_key = agenthuman_api_key

        self._api_url = "https://api.agenthuman.com/v1"
        self._conn_options = conn_options
        if session is not None:
            self._session = session
            self._owns_session = False
        else:
            self._session = aiohttp.ClientSession()
            self._owns_session = True

    async def aclose(self) -> None:
        if self._owns_session:
            await self._session.close()

    async def create_session(
        self,
        *,
        avatar: NotGivenOr[str] = NOT_GIVEN,
        aspect_ratio: NotGivenOr[str] = NOT_GIVEN,
        livekit_room: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        extra_payload: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> str:
        """
        Create a session with the AgentHuman API.

        Args:
            avatar: Avatar ID to use for the session
            aspect_ratio: Aspect ratio to use for the session
            livekit_room: LiveKit room configuration dictionary
            extra_payload: Additional payload to merge into the request

        Returns:
            Session ID string

        Raises:
            AgentHumanException: If avatar is not provided
            APIConnectionError: If the request fails
        """
        avatar = avatar or (os.getenv("AGENTHUMAN_AVATAR") or DEFAULT_AVATAR_ID)

        room_url = livekit_room.get("livekit_ws_url") if utils.is_given(livekit_room) else None
        room_token = (
            livekit_room.get("livekit_room_token") if utils.is_given(livekit_room) else None
        )
        payload = {
            "avatar": avatar,
            "room": {
                "platform": "livekit",
                "url": room_url,
                "token": room_token
            },
        }
        if utils.is_given(aspect_ratio):
            payload["aspect_ratio"] = aspect_ratio
        if utils.is_given(extra_payload):
            payload.update(extra_payload)

        last_exc: Exception | None = None
        for i in range(self._conn_options.max_retry):
            try:
                async with self._session.post(
                    f"{self._api_url}/sessions",
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
                    session_data = await response.json()
                    try:
                        return cast(str, session_data["session"]["session_id"])
                    except (KeyError, TypeError) as e:
                        raise AgentHumanException(
                            f"Unexpected API response structure: {session_data}"
                        ) from e
            except (APIStatusError, AgentHumanException):
                raise
            except Exception as e:
                last_exc = e
                if isinstance(e, APIConnectionError):
                    logger.warning(
                        "[agenthuman] failed to call agenthuman api", extra={"error": str(e)}
                    )
                else:
                    logger.exception("[agenthuman] failed to call agenthuman api")

                if i < self._conn_options.max_retry - 1:
                    await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError(
            "Failed to create AgentHuman session after all retries"
        ) from last_exc
