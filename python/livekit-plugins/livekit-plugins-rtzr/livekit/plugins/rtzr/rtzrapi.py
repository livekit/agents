from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from types import TracebackType
from typing import cast
from urllib.parse import urlencode

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from livekit.agents import APIConnectOptions, APIError, http
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)


DEFAULT_SAMPLE_RATE = 8000


class _Token(TypedDict):
    access_token: str
    expire_at: float


_TOKEN = TypeAdapter(_Token)


def _format_keywords(keywords: Iterable[object]) -> str:
    formatted: list[str] = []
    keyword_list = list(keywords)
    if len(keyword_list) > 100:
        raise ValueError("RTZR keyword boosting supports up to 100 keywords")

    for item in keyword_list:
        if isinstance(item, tuple):
            pair = cast(tuple[object, ...], item)
            if len(pair) != 2:
                raise ValueError("RTZR keyword boosting tuples must be (keyword, boost)")
            word, boost = pair
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

    Provider requests use the shared transport phase limits. The client supports
    both manual session management and async context manager usage.

    Example:
        async with RTZROpenAPIClient() as client:
            target = await client.streaming_target(client.build_config())
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        http_session: http.HttpSession | None = None,
    ) -> None:
        self.client_id = client_id or os.environ.get("RTZR_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("RTZR_CLIENT_SECRET")

        if not (self.client_id and self.client_secret):
            raise ValueError("RTZR_CLIENT_ID and RTZR_CLIENT_SECRET must be set")

        self._http_session = http_session
        self._token: _Token | None = None
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

    async def get_token(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> str:
        """Get a valid access token, refreshing if necessary."""
        token = self._token
        if token is None or token["expire_at"] < time.time() - 3600:
            token = self._token = await self._fetch_token(conn_options)
        return token["access_token"]

    async def _fetch_token(self, conn_options: APIConnectOptions) -> _Token:
        sess = self._ensure_http_session()
        url = f"{self._api_base}/v1/authenticate"

        form = {"client_id": self.client_id or "", "client_secret": self.client_secret or ""}
        async with sess.post(
            url,
            conn_options=conn_options,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=urlencode(form),
        ) as resp:
            await resp.raise_for_status()
            body = await resp.json()
        try:
            token = _TOKEN.validate_python(body, strict=True)
        except ValidationError as e:
            raise APIError(
                "RTZR sent an invalid token response", kind="protocol", detail=str(body)
            ) from e
        logger.debug("Successfully refreshed RTZR access token")
        return token

    def _ensure_http_session(self) -> http.HttpSession:
        """Ensure HTTP session is available."""
        if not self._http_session:
            self._http_session = http.http_session()
        return self._http_session

    async def close(self) -> None:
        """Nothing to release: the HTTP session belongs to the caller or the job."""

    async def streaming_target(
        self,
        config: dict[str, str],
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> http.SocketTarget:
        """Where the streaming WebSocket connects, with a fresh token."""
        query_string = "&".join(f"{k}={v}" for k, v in config.items())
        token = await self.get_token(conn_options=conn_options)
        return http.SocketTarget(
            f"{self._ws_base}/v1/transcribe:streaming?{query_string}",
            headers={"Authorization": f"bearer {token}"},
        )

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
        keywords: Iterable[str | tuple[str, float]] | None = None,
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

        if keywords:
            config["keywords"] = _format_keywords(keywords)

        return config
