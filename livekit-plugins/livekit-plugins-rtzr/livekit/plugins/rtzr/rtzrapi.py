from __future__ import annotations

import asyncio
import logging
import os
import time
import urllib.parse
from collections.abc import Iterable
from types import TracebackType
from typing import TypedDict

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


class _Token(TypedDict):
    access_token: str
    expire_at: float


def _format_keywords(keywords: Iterable[str | tuple[str, float]]) -> str:
    formatted: list[str] = []
    keyword_list: list[str | tuple[str, float]] = list(keywords)
    if len(keyword_list) > 100:
        raise ValueError("RTZR keyword boosting supports up to 100 keywords")

    for item in keyword_list:
        if isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError("RTZR keyword boosting tuples must be (keyword, boost)")
            word, boost = item
            if not isinstance(word, str):
                raise ValueError("RTZR keyword boosting keywords must be strings")
            if not isinstance(boost, (int, float)):
                raise ValueError("RTZR keyword boost must be a number")
            if not word:
                raise ValueError("RTZR keyword boosting keywords must be non-empty")
            if len(word) > 20:
                raise ValueError("RTZR keyword boosting keywords must be <= 20 chars")
            boost_value = float(boost)
            if boost_value < -5.0 or boost_value > 5.0:
                raise ValueError("RTZR keyword boost must be between -5.0 and 5.0")
            formatted.append(f"{word}:{boost_value}")
            continue

        if not isinstance(item, str):
            raise ValueError("RTZR keyword boosting items must be strings or (keyword, boost)")

        keyword = item.strip()
        if not keyword:
            raise ValueError("RTZR keyword boosting keywords must be non-empty")

        if ":" in keyword:
            word, boost_str = keyword.rsplit(":", 1)
            if not word:
                raise ValueError("RTZR keyword boosting keywords must be non-empty")
            if len(word) > 20:
                raise ValueError("RTZR keyword boosting keywords must be <= 20 chars")
            try:
                boost = float(boost_str)
            except ValueError as exc:
                raise ValueError("RTZR keyword boost must be a number") from exc
            if boost < -5.0 or boost > 5.0:
                raise ValueError("RTZR keyword boost must be between -5.0 and 5.0")
            formatted.append(f"{word}:{boost}")
            continue

        if len(keyword) > 20:
            raise ValueError("RTZR keyword boosting keywords must be <= 20 chars")
        formatted.append(keyword)

    return ",".join(formatted)


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
        self.client_id = client_id or os.environ.get("RTZR_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("RTZR_CLIENT_SECRET")

        if not (self.client_id and self.client_secret):
            raise ValueError("RTZR_CLIENT_ID and RTZR_CLIENT_SECRET must be set")

        self._http_session = http_session
        self._owns_session = http_session is None  # Track if we own the session
        self._token: _Token | None = None
        self._token_lock = asyncio.Lock()
        self._api_base = "https://openapi.vito.ai"
        self._ws_base = "wss://" + self._api_base.split("://", 1)[1]

    async def __aenter__(self) -> RTZROpenAPIClient:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def get_token(self) -> str:
        """Get a valid access token, refreshing if necessary (30 mins prior to expiry)."""
        async with self._token_lock:
            # Refresh if token is missing or expires within the next 30 minutes (1800 seconds)
            if self._token is None or time.time() + 1800 >= self._token["expire_at"]:
                await self._refresh_token()

            if self._token is None:
                raise RTZRAPIError("Failed to obtain RTZR access token")
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
                if not isinstance(data, dict):
                    raise RTZRStatusError("Invalid token response payload")
                access_token = data.get("access_token")
                expire_at = data.get("expire_at")
                if not isinstance(access_token, str) or not isinstance(expire_at, (int, float)):
                    raise RTZRStatusError("Invalid token response payload")
                self._token = {"access_token": access_token, "expire_at": float(expire_at)}
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
            # We explicitly configure a TCPConnector though it defaults to TCP_NODELAY=True
            connector = aiohttp.TCPConnector(enable_cleanup_closed=True)
            self._http_session = aiohttp.ClientSession(connector=connector)
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
        # Build URL with properly encoded query string
        query_string = urllib.parse.urlencode(config, safe=":,")
        url = f"{self._ws_base}/v1/transcribe:streaming?{query_string}"

        # Prepare headers
        token = await self.get_token()
        ws_headers = {"Authorization": f"Bearer {token}"}
        if headers:
            ws_headers.update(headers)

        session = self._ensure_http_session()

        try:
            # heartbeat=15.0 sends a ping every 15s to keep NAT/firewalls open
            # and detect dead peers quickly
            ws = await session.ws_connect(url, headers=ws_headers, heartbeat=15.0)
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
        use_itn: bool = True,
        use_disfluency_filter: bool = False,
        use_profanity_filter: bool = False,
        use_punctuation: bool = False,
        keywords: Iterable[str | tuple[str, float]] | None = None,
        language: str = "ko",
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
            "use_itn": "true" if use_itn else "false",
            "use_disfluency_filter": "true" if use_disfluency_filter else "false",
            "use_profanity_filter": "true" if use_profanity_filter else "false",
            "use_punctuation": "true" if use_punctuation else "false",
        }
        if model_name == "whisper":
            config["language"] = language

        if keywords:
            config["keywords"] = _format_keywords(keywords)

        return config
