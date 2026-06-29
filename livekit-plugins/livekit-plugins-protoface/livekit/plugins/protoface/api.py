from __future__ import annotations

import asyncio
import os
from typing import Any

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    NotGivenOr,
    utils,
)

from .errors import ProtofaceException
from .log import logger
from .version import __version__

DEFAULT_API_URL = "https://api.protoface.com"
_USER_AGENT = f"livekit-plugins-protoface/{__version__}"


class ProtofaceAPI:
    """Async client for the Protoface session API."""

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str | None] = NOT_GIVEN,
        api_url: NotGivenOr[str | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Protoface API client.

        Args:
            api_key: Protoface API key. Defaults to the `PROTOFACE_API_KEY`
                environment variable.
            api_url: Protoface API base URL. Defaults to `PROTOFACE_API_URL` or
                `https://api.protoface.com`.
            conn_options: Timeout and retry settings for API calls.
            session: Optional caller-owned HTTP session. When omitted, the
                client uses LiveKit's shared HTTP session.

        Raises:
            ProtofaceException: If no API key is passed and `PROTOFACE_API_KEY`
                is not set.
        """
        self._api_key = _resolve_optional_string(api_key, "PROTOFACE_API_KEY")
        if not self._api_key:
            raise ProtofaceException(
                "api_key must be set by passing it to ProtofaceAPI or "
                "setting the PROTOFACE_API_KEY environment variable"
            )

        api_url_value = _resolve_optional_string(api_url, "PROTOFACE_API_URL")
        self._api_url = (api_url_value or DEFAULT_API_URL).rstrip("/")
        self._conn_options = conn_options
        self._session = session

    async def start_session(
        self,
        *,
        avatar_id: str,
        transport: dict[str, Any],
        max_duration_seconds: NotGivenOr[int | None] = NOT_GIVEN,
    ) -> dict[str, Any]:
        """Create a hosted Protoface avatar session.

        Args:
            avatar_id: Protoface avatar ID to render.
            transport: Protoface transport configuration. The LiveKit Agents
                plugin uses `audio_source="data_stream"`.
            max_duration_seconds: Optional maximum session duration. Protoface
                applies the lower of this value and the account plan limit.

        Returns:
            The decoded Protoface session object.

        Raises:
            APIConnectionError: If a retryable API or network error persists
                after all retry attempts.
            APIStatusError: If Protoface returns a non-retryable error response.
        """
        body: dict[str, Any] = {"avatar_id": avatar_id, "transport": transport}
        if utils.is_given(max_duration_seconds) and max_duration_seconds is not None:
            body["max_duration_seconds"] = max_duration_seconds

        return await self._json("POST", "/v1/sessions", json=body)

    async def end_session(self, session_id: str) -> dict[str, Any]:
        """Request a graceful end for a hosted Protoface session.

        Args:
            session_id: Protoface session ID returned by `start_session()`.

        Returns:
            The decoded Protoface response body.

        Raises:
            APIConnectionError: If a retryable API or network error persists
                after all retry attempts.
            APIStatusError: If Protoface returns a non-retryable error response.
        """
        return await self._json("POST", f"/v1/sessions/{session_id}/end")

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    async def _json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        request_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
            **(headers or {}),
        }
        url = f"{self._api_url}{path}"
        error: Exception | None = None

        for attempt in range(self._conn_options.max_retry + 1):
            try:
                async with self._ensure_http_session().request(
                    method,
                    url,
                    json=json,
                    headers=request_headers,
                    timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
                ) as response:
                    payload = await _read_payload(response)
                    if response.ok:
                        if not isinstance(payload, dict):
                            raise APIStatusError(
                                "Protoface API returned a non-object JSON response",
                                status_code=response.status,
                                body=payload,
                                retryable=False,
                            )
                        return payload

                    raise APIStatusError(
                        "Protoface API returned an error",
                        status_code=response.status,
                        body=payload,
                    )
            except asyncio.TimeoutError as exc:
                error = APITimeoutError()
                error.__cause__ = exc
            except aiohttp.ClientError as exc:
                error = APIConnectionError()
                error.__cause__ = exc
            except APIStatusError as exc:
                if not exc.retryable:
                    raise
                error = exc

            if attempt == self._conn_options.max_retry:
                break

            logger.warning(
                "protoface api request failed, retrying",
                extra={"attempt": attempt + 1, "method": method, "path": path},
            )
            await asyncio.sleep(self._conn_options._interval_for_retry(attempt))

        raise APIConnectionError("Failed to call Protoface API after all retries.") from error


def _resolve_optional_string(value: NotGivenOr[str | None], env_name: str) -> str | None:
    if utils.is_given(value) and value is not None:
        return value
    return os.getenv(env_name)


async def _read_payload(response: aiohttp.ClientResponse) -> object:
    text = await response.text()
    if not text:
        return {}

    try:
        return await response.json(content_type=None)
    except ValueError:
        return {"raw": text}


__all__ = ["DEFAULT_API_URL", "ProtofaceAPI"]
