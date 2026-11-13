from __future__ import annotations

from json import JSONDecodeError
from types import TracebackType

from pydantic import BaseModel, JsonValue, ValidationError

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIError,
    http,
)

DEFAULT_API_URL = "https://api.keyframelabs.com"


class SessionResponse(BaseModel):
    reservation_id: str | None = None


class KeyframeAPI:
    """Asynchronous client for the Keyframe Labs session API."""

    def __init__(
        self,
        api_key: str,
        api_url: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: http.HttpSession | None = None,
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url
        self._conn_options = conn_options
        self._session = session

    async def __aenter__(self) -> KeyframeAPI:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    async def create_plugin_session(
        self,
        *,
        persona_id: str | None = None,
        persona_slug: str | None = None,
        room_name: str,
        livekit_url: str,
        livekit_token: str,
        source_participant_identity: str,
    ) -> SessionResponse:
        """Create a plugin session via POST /v1/sessions/plugins/livekit.

        Returns the reservation identity accepted by the provider.
        """
        payload: dict[str, JsonValue] = {
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
        self, endpoint: str, payload: dict[str, JsonValue], headers: dict[str, str]
    ) -> SessionResponse:
        url = f"{self._api_url}{endpoint}"
        session = self._session or http.http_session()

        async def request() -> SessionResponse:
            async with session.post(
                url, headers=headers, json=payload, conn_options=self._conn_options
            ) as response:
                await response.raise_for_status()
                try:
                    return SessionResponse.model_validate(await response.json())
                except (ValidationError, JSONDecodeError) as exc:
                    raise APIError(
                        "Keyframe returned an invalid response", retryable=False
                    ) from exc

        return await http.retry_request(request, conn_options=self._conn_options)
