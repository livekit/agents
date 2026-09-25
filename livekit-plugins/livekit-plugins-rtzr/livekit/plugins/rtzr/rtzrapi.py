from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Iterable
from types import TracebackType
from typing import TypedDict
from urllib.parse import urlencode

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
            if not math.isfinite(boost_value) or not -5.0 <= boost_value <= 5.0:
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
            except ValueError:
                raise ValueError("RTZR keyword boost must be a number") from None
            if not math.isfinite(boost) or not -5.0 <= boost <= 5.0:
                raise ValueError("RTZR keyword boost must be between -5.0 and 5.0")
            formatted.append(f"{word}:{boost}")
            continue

        if len(keyword) > 20:
            raise ValueError("RTZR keyword boosting keywords must be <= 20 chars")
        formatted.append(keyword)

    for keyword in formatted:
        word = keyword.split(":", 1)[0]
        if not word.strip() or any(c != " " and not "가" <= c <= "힣" for c in word):
            raise ValueError("RTZR keywords must contain only Korean syllables and spaces")
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
        self._api_base = os.getenv("RTZR_API_BASE", "https://openapi.vito.ai").rstrip("/")
        self._ws_base = os.getenv(
            "RTZR_WEBSOCKET_URL",
            self._api_base.replace("https://", "wss://", 1).replace("http://", "ws://", 1),
        ).rstrip("/")

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
        """Get a valid access token, refreshing 30 minutes before expiry."""
        async with self._token_lock:
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
            raise RTZRStatusError("RTZR authentication failed", status_code=e.status) from None
        except aiohttp.ClientError:
            raise RTZRConnectionError("Failed to authenticate with RTZR API") from None

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session is available."""
        if not self._http_session:
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
        query_string = urlencode(config, safe=":,")
        url = f"{self._ws_base}/v1/transcribe:streaming?{query_string}"

        session = self._ensure_http_session()
        custom_auth = any(key.lower() == "authorization" for key in (headers or {}))
        for attempt in range(2):
            token = await self.get_token()
            ws_headers = {"Authorization": f"Bearer {token}"}
            if headers:
                ws_headers.update(headers)
            try:
                return await session.ws_connect(url, headers=ws_headers, heartbeat=15.0)
            except aiohttp.ClientResponseError as e:
                if e.status == 401 and attempt == 0 and not custom_auth:
                    async with self._token_lock:
                        # Another connection may already have replaced the rejected token.
                        if self._token is not None and self._token["access_token"] == token:
                            self._token = None
                    continue
                raise RTZRStatusError(
                    "RTZR WebSocket connection failed", status_code=e.status
                ) from None
            except aiohttp.ClientError:
                raise RTZRConnectionError("RTZR WebSocket connection failed") from None
        raise AssertionError("WebSocket retry loop exhausted")

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
