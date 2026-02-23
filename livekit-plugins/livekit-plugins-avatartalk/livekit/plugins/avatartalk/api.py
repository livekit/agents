import logging
import os
from typing import Any

import aiohttp

from livekit.agents import NOT_GIVEN, NotGivenOr

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.avatartalk.ai"


class AvatarTalkException(Exception):
    """Exception for AvatarTalkAPI errors"""


class AvatarTalkAPI:
    def __init__(
        self,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
    ):
        self._api_url = api_url or DEFAULT_API_URL
        avatartalk_api_key = api_key or os.getenv("AVATARTALK_API_KEY")
        if avatartalk_api_key is None:
            raise AvatarTalkException(
                "AvatarTalk API key is required, either as argument or set"
                " AVATARTALK_API_KEY environment variable"
            )

        self._api_key = avatartalk_api_key
        self._headers = {"Authorization": f"Bearer {self._api_key}"}

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method, f"{self._api_url}{path}", headers=self._headers, **kwargs
            ) as response:
                if response.ok:
                    result: dict[str, Any] = await response.json()
                    return result
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
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/livekit/create-session",
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

    async def stop_session(self, task_id: str) -> dict[str, Any]:
        return await self._request("DELETE", f"/livekit/delete-session/{task_id}")
