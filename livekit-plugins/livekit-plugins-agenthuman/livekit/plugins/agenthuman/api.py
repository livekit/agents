from __future__ import annotations

import asyncio
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

from .errors import AgentHumanException
from .log import logger

DEFAULT_API_URL = "https://api.agenthuman.com/v1"


class AgentHumanAPI:
    """Asynchronous client for the AgentHuman session API."""

    def __init__(
        self,
        api_key: str,
        api_url: str = DEFAULT_API_URL,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url.rstrip("/")
        self._conn_options = conn_options
        self._session = session
        self._own_session = session is None

    async def __aenter__(self) -> AgentHumanAPI:
        if self._own_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        if self._own_session and self._session and not self._session.closed:
            await self._session.close()

    async def create_session(
        self,
        *,
        avatar: str,
        livekit_url: str,
        livekit_token: str,
        aspect_ratio: NotGivenOr[str] = NOT_GIVEN,
        extra_payload: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> str:
        """Create an AgentHuman session that joins the given LiveKit room.

        Args:
            avatar: An avatar ID (``avat_...``) or a publicly accessible image URL.
            livekit_url: LiveKit server URL the avatar connects to.
            livekit_token: Access token the avatar joins the room with.
            aspect_ratio: Aspect ratio of the avatar video, e.g. ``"3:4"``.
            extra_payload: Additional fields merged into the request body.

        Returns:
            The AgentHuman session ID.
        """
        payload: dict[str, Any] = {
            "avatar": avatar,
            "room": {"platform": "livekit", "url": livekit_url, "token": livekit_token},
        }
        if utils.is_given(aspect_ratio):
            payload["aspect_ratio"] = aspect_ratio
        if utils.is_given(extra_payload):
            payload.update(extra_payload)

        data = await self._post("/sessions", payload)
        try:
            return cast(str, data["session"]["session_id"])
        except (KeyError, TypeError):
            # the response also carries session credentials, so report its shape only
            keys = sorted(data) if isinstance(data, dict) else type(data).__name__
            raise AgentHumanException(
                f"unexpected AgentHuman API response, missing session.session_id (got {keys})"
            ) from None

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> Any:
        url = f"{self._api_url}{endpoint}"
        headers = {"Content-Type": "application/json", "x-api-key": self._api_key}
        session = self._session or aiohttp.ClientSession()
        try:
            for i in range(self._conn_options.max_retry + 1):
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
                                f"AgentHuman API error for {url}: {response.status}",
                                status_code=response.status,
                                body=text,
                            )
                        return await response.json()
                except Exception as e:
                    if isinstance(e, APIStatusError) and not e.retryable:
                        raise APIConnectionError(
                            f"Failed to call AgentHuman API at {url} with non-retryable error",
                            retryable=False,
                        ) from e

                    if isinstance(e, APIConnectionError):
                        logger.warning("Failed to call AgentHuman API", extra={"error": str(e)})
                    else:
                        logger.exception("Failed to call AgentHuman API")

                    if i < self._conn_options.max_retry:
                        await asyncio.sleep(self._conn_options._interval_for_retry(i))
        finally:
            if not self._session:
                await session.close()

        raise APIConnectionError("Failed to call AgentHuman API after all retries.")
