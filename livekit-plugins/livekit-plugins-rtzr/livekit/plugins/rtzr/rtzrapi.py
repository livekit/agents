from __future__ import annotations

import logging
import os
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class RTZRAPIError(Exception):
    """Base exception for RTZR API errors."""

    pass


class RTZRConnectionError(RTZRAPIError):
    """Exception raised when connection to RTZR API fails."""

    pass


class RTZRStatusError(RTZRAPIError):
    """Exception raised when RTZR API returns an error status."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class RTZRTimeoutError(RTZRAPIError):
    """Exception raised when RTZR API request times out."""

    pass


DEFAULT_SAMPLE_RATE = 8000


class RTZROpenAPIClient:
    """RTZR OpenAPI Client for authentication and WebSocket streaming.

    This is an independent SDK client that can be used without livekit dependencies.
    It supports both manual session management and async context manager usage.

    Example:
        # Manual session management
        client = RTZROpenAPIClient()
        token = await client.get_token()
        ws = await client.connect_websocket(config)
        await client.close()

        # Context manager (recommended)
        async with RTZROpenAPIClient() as client:
            token = await client.get_token()
            ws = await client.connect_websocket(config)
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self.client_id = client_id or os.environ.get("RTZR_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("RTZR_CLIENT_SECRET")

        if not (self.client_id and self.client_secret):
            raise ValueError("RTZR_CLIENT_ID and RTZR_CLIENT_SECRET must be set")

        self._http_session = http_session
        self._owns_session = http_session is None  # Track if we own the session
        self._token: dict[str, Any] | None = None
        self._api_base = "https://openapi.vito.ai"
        self._ws_base = "wss://" + self._api_base.split("://", 1)[1]

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def get_token(self) -> str:
        """Get a valid access token, refreshing if necessary."""
        if self._token is None or self._token["expire_at"] < time.time() - 3600:
            await self._refresh_token()
        return self._token["access_token"]

    async def _refresh_token(self) -> None:
        """Refresh the access token."""
        sess = self._ensure_http_session()
        url = f"{self._api_base}/v1/authenticate"

        try:
            async with sess.post(
                url, data={"client_id": self.client_id, "client_secret": self.client_secret}
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                self._token = data
                logger.debug("Successfully refreshed RTZR access token")
        except aiohttp.ClientResponseError as e:
            logger.error("RTZR authentication failed: %s %s", e.status, e.message)
            raise RTZRStatusError(
                message=f"Authentication failed: {e.message}",
                status_code=e.status,
            ) from e
        except aiohttp.ClientError as e:
            logger.error("RTZR authentication connection error: %s", e)
            raise RTZRConnectionError("Failed to authenticate with RTZR API") from e

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session is available."""
        if not self._http_session:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def close(self) -> None:
        """Close the HTTP session if we own it."""
        if self._http_session and self._owns_session:
            await self._http_session.close()
            self._http_session = None

    async def connect_websocket(
        self, config: dict[str, str], headers: dict[str, str] | None = None
    ) -> aiohttp.ClientWebSocketResponse:
        """Connect to the streaming WebSocket endpoint."""
        # Build URL like reference client
        query_string = "&".join(f"{k}={v}" for k, v in config.items())
        url = f"{self._ws_base}/v1/transcribe:streaming?{query_string}"

        # Prepare headers
        token = await self.get_token()
        ws_headers = {"Authorization": f"bearer {token}"}
        if headers:
            ws_headers.update(headers)

        session = self._ensure_http_session()

        try:
            ws = await session.ws_connect(url, headers=ws_headers)
            logger.debug("Connected to RTZR WebSocket at %s", url)
            return ws
        except aiohttp.ClientResponseError as e:
            logger.error("RTZR WebSocket connection failed: %s %s", e.status, e.message)
            raise RTZRStatusError(
                message=f"WebSocket connection failed: {e.message}",
                status_code=e.status,
            ) from e
        except aiohttp.ClientError as e:
            logger.error("RTZR WebSocket client error: %s", e)
            raise RTZRConnectionError("WebSocket connection failed") from e

    def build_config(
        self,
        model_name: str = "sommers_ko",
        domain: str = "CALL",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        encoding: str = "LINEAR16",
        epd_time: float = 0.5,
        noise_threshold: float = 0.60,
        active_threshold: float = 0.80,
        use_punctuation: bool = False,
    ) -> dict[str, str]:
        """Build configuration dictionary for WebSocket connection."""
        config = {
            "model_name": model_name,
            "domain": domain,
            "sample_rate": str(sample_rate),
            "encoding": encoding,
            "epd_time": str(epd_time),
            "noise_threshold": str(noise_threshold),
            "active_threshold": str(active_threshold),
            "use_punctuation": "true" if use_punctuation else "false",
        }

        return config
