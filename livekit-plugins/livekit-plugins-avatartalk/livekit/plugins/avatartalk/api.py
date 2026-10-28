import logging
import os
from json import JSONDecodeError
from typing import TypeVar

import aiohttp
from pydantic import BaseModel, JsonValue, ValidationError

from livekit.agents import NOT_GIVEN, APIConnectionError, NotGivenOr

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.avatartalk.ai"


class AvatarTalkException(Exception):
    """Exception for AvatarTalkAPI errors"""


class _SessionResponse(BaseModel):
    task_id: str | None


class _StopResponse(BaseModel):
    pass


_ResponseT = TypeVar("_ResponseT", bound=BaseModel)


class AvatarTalkAPI:
    def __init__(
        self,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        self._api_url = api_url or DEFAULT_API_URL
        avatartalk_api_key = api_key or os.getenv("AVATARTALK_API_KEY")
        if avatartalk_api_key is None:
            raise AvatarTalkException(
                "AvatarTalk API key is required, either as argument or set"
                " AVATARTALK_API_KEY environment variable"
            )

        self._api_key = avatartalk_api_key
        self._headers = {"Authorization": f"Bearer {self._api_key}"}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        response_type: type[_ResponseT],
        json: dict[str, JsonValue] | None = None,
    ) -> _ResponseT:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method, f"{self._api_url}{path}", headers=self._headers, json=json
            ) as response:
                if response.ok:
                    try:
                        return response_type.model_validate(await response.json())
                    except (ValidationError, JSONDecodeError, aiohttp.ContentTypeError) as exc:
                        raise APIConnectionError(
                            "AvatarTalk returned an invalid response", retryable=False
                        ) from exc
                else:
                    r = await response.json()
                    raise AvatarTalkException(f"API request failed: {response.status} {r}")

    async def start_session(
        self,
        livekit_url: str,
        avatar: str,
        emotion: str,
        room_name: str,
        livekit_listener_token: str,
        livekit_room_token: str,
        agent_identity: str,
    ) -> _SessionResponse:
        return await self._request(
            "POST",
            "/livekit/create-session",
            response_type=_SessionResponse,
            json={
                "livekit_url": livekit_url,
                "avatar": avatar,
                "emotion": emotion,
                "room_name": room_name,
                "room_token": livekit_room_token,
                "listener_token": livekit_listener_token,
                "agent_identity": agent_identity,
            },
        )

    async def stop_session(self, task_id: str) -> _StopResponse:
        return await self._request(
            "DELETE", f"/livekit/delete-session/{task_id}", response_type=_StopResponse
        )
