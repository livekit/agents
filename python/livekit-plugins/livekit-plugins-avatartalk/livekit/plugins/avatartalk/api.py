import os
from typing import TypeVar

from pydantic import BaseModel, JsonValue, ValidationError

from livekit.agents import NOT_GIVEN, APIError, NotGivenOr, http

DEFAULT_API_URL = "https://api.avatartalk.ai"


class AvatarTalkException(Exception):
    """Exception for AvatarTalkAPI errors"""


class SessionResponse(BaseModel):
    task_id: str | None


class StopResponse(BaseModel):
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
        async with http.http_session().request(
            method, f"{self._api_url}{path}", headers=self._headers, json=json
        ) as response:
            await response.raise_for_status()
            text = await response.text()
            try:
                return response_type.model_validate_json(text)
            except ValidationError as exc:
                raise APIError(
                    "AvatarTalk returned an invalid response", kind="protocol", detail=text
                ) from exc

    async def start_session(
        self,
        livekit_url: str,
        avatar: str,
        emotion: str,
        room_name: str,
        livekit_listener_token: str,
        livekit_room_token: str,
        agent_identity: str,
    ) -> SessionResponse:
        return await self._request(
            "POST",
            "/livekit/create-session",
            response_type=SessionResponse,
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

    async def stop_session(self, task_id: str) -> StopResponse:
        return await self._request(
            "DELETE", f"/livekit/delete-session/{task_id}", response_type=StopResponse
        )
