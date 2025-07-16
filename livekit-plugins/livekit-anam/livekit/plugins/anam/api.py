import asyncio
import os
from typing import Any, Dict, Literal, Optional

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

class AnamException(Exception):
    """Custom exception for Anam API errors."""

DEFAULT_API_URL = "https://api.anam.ai"

class AnamAPI:
    """
    An asynchronous client for interacting with the Anam API.

    This class handles authentication, request signing, and retries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = DEFAULT_API_URL,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """
        Initializes the AnamAPI client.

        Args:
            api_key: Your Anam API key. If not provided, it will be read from
                     the ANAM_API_KEY environment variable.
            api_url: The base URL of the Anam API.
            conn_options: Connection options for the aiohttp session.
            session: An optional existing aiohttp.ClientSession to use for requests.
        """
        self._api_key = api_key or os.getenv("ANAM_API_KEY")
        if not self._api_key:
            raise AnamException("ANAM_API_KEY must be provided or set as an environment variable.")

        self._api_url = api_url
        self._conn_options = conn_options
        self._session = session
        self._own_session = session is None

    async def __aenter__(self):
        if self._own_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._own_session and self._session:
            await self._session.close()

    async def create_session_token(
        self,
        persona_config: Dict[str, Any],
        livekit_url: str,
        livekit_token: str
    ) -> str:
        """
        Creates a session token to authorize starting an engine session.

        Returns:
            The created session token (a JWT string).
        """
        payload = {
            "personaConfig": {
                "type": "ephemeral", 
                **persona_config,
                "llmId": "CUSTOMER_CLIENT_V1"
            },
        }
        payload["environment"] = {
            "livekitUrl": livekit_url,
            "livekitToken": livekit_token,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}", # Use API Key here
            "Content-Type": "application/json",
        }
        response_data = await self._post("/v1/auth/session-token", payload, headers)

        
        session_token = response_data.get("sessionToken")
        if not session_token:
            raise AnamException("Failed to retrieve sessionToken from API response.")
        return session_token

    async def start_engine_session(
        self,
        session_token: str,
    ) -> Dict[str, Any]:
        """
        Starts the engine session using a previously created session token.

        Args:
            session_token: The temporary token from create_session_token.
            livekit_url: The URL of the LiveKit instance.
            livekit_token: The access token for the LiveKit room.

        Returns:
            The session details, including sessionId and engine host info.
        """
        headers = {
            "Authorization": f"Bearer {session_token}", # Use Session Token here
            "Content-Type": "application/json",
        }
        return await self._post("/v1/engine/session", {}, headers)

    async def _post(self, endpoint: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Internal method to make a POST request with retry logic.
        """
        url = f"{self._api_url}{endpoint}"
        session = self._session or aiohttp.ClientSession()
        try:
            for attempt in range(self._conn_options.max_retry):
                try:
                    async with session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(sock_connect=self._conn_options.timeout),
                    ) as response:
                        if not response.ok:
                            text = await response.text()
                            raise APIStatusError(
                                f"Server returned an error for {url}: {response.status}",
                                status_code=response.status,
                                body=text,
                            )
                        return await response.json()
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(
                        f"API request to {url} failed on attempt {attempt + 1}",
                        extra={"error": str(e)},
                    )
                    if attempt >= self._conn_options.max_retry - 1:
                        raise APIConnectionError(f"Failed to connect to Anam API at {url}") from e
                    await asyncio.sleep(self._conn_options.retry_interval)
        finally:
            if not self._session: # if we created the session, we close it
                await session.close()
        
        raise APIConnectionError("Failed to call Anam API after all retries.")