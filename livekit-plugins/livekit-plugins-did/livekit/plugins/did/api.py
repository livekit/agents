import asyncio
import os

import aiohttp
from pydantic import BaseModel, JsonValue

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    NotGivenOr,
    utils,
)
from livekit.agents.utils import http_context

from .errors import DIDException
from .log import logger

DEFAULT_API_URL = "https://api.d-id.com"


class JoinSessionResponse(BaseModel):
    id: str


class DIDAPI:
    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        did_api_key = api_key if utils.is_given(api_key) else os.getenv("DID_API_KEY")
        if not did_api_key:
            raise DIDException("DID_API_KEY must be set")
        self._api_key = did_api_key

        self._api_url = api_url if utils.is_given(api_url) else DEFAULT_API_URL
        self._conn_options = conn_options
        self._session = session or aiohttp.ClientSession()

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

    async def _post(self, endpoint: str, payload: dict[str, JsonValue]) -> JoinSessionResponse:
        url = f"{self._api_url}/{endpoint}"
        num_attempts = self._conn_options.max_retry + 1
        for attempt in range(num_attempts):
            try:
                async with http_context.request(
                    self._session,
                    "POST",
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Basic {self._api_key}",
                    },
                    json=payload,
                    conn_options=self._conn_options,
                ) as response:
                    if not response.ok:
                        text = await response.text()
                        raise APIStatusError(
                            "D-ID API error",
                            status_code=response.status,
                            body=text,
                        )
                    return JoinSessionResponse.model_validate(await response.json())
            except (ValueError, aiohttp.ContentTypeError) as exc:
                raise APIConnectionError(
                    "Did returned an invalid response", retryable=False
                ) from exc
            except APIStatusError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(
                    f"D-ID API request failed (attempt {attempt + 1}/{num_attempts})",
                    extra={"error": str(e)},
                )
                if attempt == num_attempts - 1:
                    raise APIConnectionError(f"Failed to connect to D-ID API at {url}") from e
                await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError(f"Failed to connect to D-ID API at {url}")
