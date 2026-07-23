from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
)

from .log import logger
from .types import PersonaConfig, SessionOptions

DEFAULT_API_URL = "https://api.anam.ai"


class AnamAPI:
    """
    An asynchronous client for interacting with the Anam API.

    This class handles authentication, request signing, and retries.
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
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
        self._api_key = api_key
        self._api_url = api_url
        self._conn_options = conn_options
        self._session = session
        self._own_session = session is None

    async def __aenter__(self) -> AnamAPI:
        if self._own_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        if self._own_session and self._session:
            await self._session.close()

    async def start_session(
        self,
        persona_config: PersonaConfig,
        livekit_url: str,
        livekit_token: str,
        session_options: SessionOptions | None = None,
    ) -> dict[str, Any]:
        """
        Starts an engine session.

        Args:
            session_options: Optional per-session output options (e.g. explicit
                video dimensions) forwarded to Anam as ``sessionOptions``. When
                ``None``, Anam uses the avatar model's default output.

        Returns:
            The session details, including sessionId and engine host info.
        """
        persona_config_payload: dict[str, Any] = {
            "type": "ephemeral",
            "name": persona_config.name,
            "avatarId": persona_config.avatarId,
            "llmId": "CUSTOMER_CLIENT_V1",
        }

        if persona_config.avatarModel:
            persona_config_payload["avatarModel"] = persona_config.avatarModel

        if persona_config.directorNotes is not None:
            # drop unset (None) ones so Anam falls back to its model/cue defaults.
            director_notes = {
                k: v for k, v in vars(persona_config.directorNotes).items() if v is not None
            }
            if director_notes:
                persona_config_payload["directorNotes"] = director_notes

        payload: dict[str, Any] = {
            "personaConfig": persona_config_payload,
        }
        payload["environment"] = {
            "livekitUrl": livekit_url,
            "livekitToken": livekit_token,
        }

        if session_options is not None and (
            session_options.video_width is not None or session_options.video_height is not None
        ):
            # Anam's public API speaks camelCase pixel dimensions and wants them
            # as a matched pair: it rejects a lone width/height (and any
            # unsupported pair) with an HTTP 400, surfaced below as
            # APIStatusError, rather than downgrading. Fail fast on a half pair
            # rather than round-tripping a 400.
            if session_options.video_width is None or session_options.video_height is None:
                raise ValueError(
                    "video_width and video_height must be set together (both or neither)"
                )
            payload["sessionOptions"] = {
                "videoWidth": session_options.video_width,
                "videoHeight": session_options.video_height,
            }

        headers = {
            "Authorization": f"Bearer {self._api_key}",  # Use API Key here
            "Content-Type": "application/json",
        }
        return await self._post("/v1/engine/session", payload, headers)

    async def _post(
        self, endpoint: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
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
                        return await response.json()  # type: ignore
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(
                        f"API request to {url} failed on attempt {attempt + 1}",
                        extra={"error": str(e)},
                    )
                    if attempt >= self._conn_options.max_retry - 1:
                        raise APIConnectionError(f"Failed to connect to Anam API at {url}") from e
                    await asyncio.sleep(self._conn_options.retry_interval)
        finally:
            if not self._session:  # if we created the session, we close it
                await session.close()

        raise APIConnectionError("Failed to call Anam API after all retries.")
