import asyncio
import os
from typing import Any, Optional

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    NotGivenOr,
)

from .log import logger
from .schema import AvatarConfig, CreateSessionRequest, Credentials


class AkoolException(Exception):
    """Exception for Akool errors"""


DEFAULT_API_URL = "https://openapi.akool.com/api/open"


class AkoolAPI:
    def __init__(
        self,
        avatar_config: AvatarConfig,
        client_id: NotGivenOr[str] = NOT_GIVEN,
        client_secret: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._avatar_config = avatar_config
        self._client_id = client_id or os.getenv("AKOOL_CLIENT_ID")
        self._client_secret = client_secret or os.getenv("AKOOL_CLIENT_SECRET")
        if not self._client_id or not self._client_secret:
            raise AkoolException("AKOOL_CLIENT_ID and AKOOL_CLIENT_SECRET must be set")
        self._api_url = api_url or DEFAULT_API_URL
        self._conn_options = conn_options
        self._session = session or aiohttp.ClientSession()
        self._access_token = None

    async def _get_access_token(self) -> str:
        """
        Get an access token from the Akool API.
        https://docs.akool.com/authentication/usage
        """
        url = f"{self._api_url}/v3/getToken"
        payload = {"clientId": self._client_id, "clientSecret": self._client_secret}
        response_data = await self._post(url, payload)
        if response_data["code"] != 1000:
            raise AkoolException(f"failed to get access token, error code: {response_data['code']}")
        logger.info(f"get_access_token response: {response_data}")
        return response_data["token"]

    async def create_session(self, livekit_url: str, livekit_token: str) -> str:
        """
        https://docs.akool.com/ai-tools-suite/live-avatar#create-session
        """
        url = f"{self._api_url}/v4/liveAvatar/session/create"
        payload = CreateSessionRequest(
            stream_type="livekit",
            credentials=Credentials(
                livekit_url=livekit_url,
                livekit_token=livekit_token,
            ),
            **self._avatar_config.model_dump(exclude_none=True),
        ).model_dump(exclude_none=True)
        logger.info(f"create_session payload: {payload}")
        response_data = await self._post(url, payload, need_token=True)
        logger.info(f"create_session response: {response_data}")
        return response_data["data"]  # type: ignore

    async def close_session(self, session_id: str) -> None:
        """
        Close avatar session
        https://docs.akool.com/ai-tools-suite/live-avatar#close-session
        """
        url = f"{self._api_url}/v4/liveAvatar/session/close"
        payload = {"session_id": session_id}
        logger.info(f"close_session payload: {payload}")
        response_data = await self._post(url, payload, need_token=True)
        logger.info(f"close_session response: {response_data}")

    async def _post(self, url: str, payload: dict[str, Any], need_token=False) -> dict[str, Any]:
        """
        Make a POST request to the Akool API with retry logic.

        Args:
            endpoint: API endpoint path (without leading slash)
            payload: JSON payload for the request

        Returns:
            Response data as a dictionary

        Raises:
            APIConnectionError: If the request fails after all retries
        """
        headers = {"Content-Type": "application/json"}
        if need_token:
            if not self._access_token:
                self._access_token = await self._get_access_token()
            headers["Authorization"] = f"Bearer {self._access_token}"

        for i in range(self._conn_options.max_retry):
            try:
                async with self._session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(sock_connect=self._conn_options.timeout),
                ) as response:
                    if not response.ok:
                        text = await response.text()
                        raise APIStatusError(
                            "Server returned an error", status_code=response.status, body=text
                        )
                    return await response.json()  # type: ignore
            except Exception as e:
                if isinstance(e, APIConnectionError):
                    logger.warning("failed to call akool api", extra={"error": str(e)})
                else:
                    logger.exception("failed to call akool api")

                if i < self._conn_options.max_retry - 1:
                    await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to call Akool API after all retries")
