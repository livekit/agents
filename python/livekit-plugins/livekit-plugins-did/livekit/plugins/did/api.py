import os
from json import JSONDecodeError

from pydantic import BaseModel, JsonValue, ValidationError

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    APIError,
    NotGivenOr,
    http,
    utils,
)

from .errors import DIDException

DEFAULT_API_URL = "https://api.d-id.com"


class _JoinSessionResponse(BaseModel):
    id: str


class DIDAPI:
    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: http.HttpSession | None = None,
    ) -> None:
        did_api_key = api_key if utils.is_given(api_key) else os.getenv("DID_API_KEY")
        if not did_api_key:
            raise DIDException("DID_API_KEY must be set")
        self._api_key = did_api_key

        self._api_url = api_url if utils.is_given(api_url) else DEFAULT_API_URL
        self._conn_options = conn_options
        self._session = session

    async def join_session(
        self,
        *,
        agent_id: str,
        transport: dict[str, JsonValue],
        audio_config: dict[str, JsonValue],
    ) -> str:
        """Dispatch a D-ID avatar worker into the room.

        Returns the session id.
        """
        payload: dict[str, JsonValue] = {
            "transport": transport,
            "audio_config": audio_config,
        }
        response_data = await self._post(f"v2/agents/{agent_id}/sessions/join", payload)
        return response_data.id

    async def _post(self, endpoint: str, payload: dict[str, JsonValue]) -> _JoinSessionResponse:
        url = f"{self._api_url}/{endpoint}"
        session = self._session or http.http_session()

        async def request() -> _JoinSessionResponse:
            async with session.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Basic {self._api_key}",
                },
                json=payload,
                conn_options=self._conn_options,
            ) as response:
                await response.raise_for_status()
                try:
                    return _JoinSessionResponse.model_validate(await response.json())
                except (ValidationError, JSONDecodeError) as exc:
                    raise APIError("D-ID returned an invalid response", retryable=False) from exc

        return await http.retry_request(request, conn_options=self._conn_options)
