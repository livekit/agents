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

from .errors import FaceModeAPIError
from .log import logger
from .models import LiveKitRoom, SessionDetails, SessionRequest
from .version import __version__

DEFAULT_API_URL = "https://api.facemode.io/api"
_USER_AGENT = f"livekit-plugins-facemode/{__version__}"


class FaceModeAPI:
    """Async client for the FaceMode session API."""

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str | None] = NOT_GIVEN,
        api_url: NotGivenOr[str | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a FaceMode API client.

        Args:
            api_key: FaceMode API key. Defaults to the `FACEMODE_API_KEY`
                environment variable.
            api_url: FaceMode API base URL. Defaults to `FACEMODE_API_URL` or
                `https://api.facemode.io/api`.
            conn_options: Timeout and retry settings for API calls.
            session: Optional caller-owned HTTP session. When omitted, the
                client uses LiveKit's shared HTTP session.

        Raises:
            FaceModeAPIError: If no API key is passed and `FACEMODE_API_KEY`
                is not set.
        """
        self._api_key = _resolve_optional_string(api_key, "FACEMODE_API_KEY")
        if not self._api_key:
            raise FaceModeAPIError(
                "api_key must be set by passing it to FaceModeAPI or "
                "setting the FACEMODE_API_KEY environment variable"
            )

        api_url_value = _resolve_optional_string(api_url, "FACEMODE_API_URL")
        self._api_url = (api_url_value or DEFAULT_API_URL).rstrip("/")
        self._conn_options = conn_options
        self._session = session

    async def start_session(
        self,
        *,
        avatar_id: str,
        room: LiveKitRoom,
        room_name: str = "",
        input_provider: str | None = None,
    ) -> SessionDetails:
        """Create a hosted FaceMode avatar session.

        Args:
            avatar_id: FaceMode avatar ID to render.
            room: LiveKit room credentials the avatar worker joins with.
            room_name: Exact LiveKit room name, persisted server-side.
            input_provider: Optional upstream speech input provider pinned for
                the session.

        Returns:
            The parsed FaceMode session details.

        Raises:
            APIConnectionError: If a retryable API or network error persists
                after all retry attempts.
            APIStatusError: If FaceMode returns a non-retryable error response.
            FaceModeAPIError: If the response cannot be parsed.
        """
        payload = SessionRequest(
            avatar_id=avatar_id,
            room=room,
            room_name=room_name,
            input_provider=input_provider,
        ).to_payload()
        data = await self._json("POST", "/sessions", json=payload)
        return _parse_session(data, context="session response")

    async def get_session(
        self,
        session_id: str,
        *,
        fallback: SessionDetails | None = None,
    ) -> SessionDetails:
        """Fetch the current state of a FaceMode session.

        Args:
            session_id: FaceMode session ID returned by `start_session()`.
            fallback: Optional previous session details used to fill fields the
                status response omits.

        Returns:
            The parsed FaceMode session details.

        Raises:
            APIConnectionError: If a retryable API or network error persists
                after all retry attempts.
            APIStatusError: If FaceMode returns a non-retryable error response.
            FaceModeAPIError: If the response cannot be parsed.
        """
        data = await self._json("GET", f"/sessions/{session_id}")
        return _parse_session(data, context="ingestion status response", fallback=fallback)

    async def reconnect_session(
        self,
        session_id: str,
        *,
        room: LiveKitRoom,
        fallback: SessionDetails | None = None,
    ) -> SessionDetails:
        """Request a fresh one-time ingestion WebSocket token for a session.

        Args:
            session_id: FaceMode session ID returned by `start_session()`.
            room: LiveKit room credentials of the session.
            fallback: Optional previous session details used to fill fields the
                reconnect response omits.

        Returns:
            The parsed FaceMode session details with refreshed ingestion
            credentials.

        Raises:
            APIConnectionError: If a retryable API or network error persists
                after all retry attempts.
            APIStatusError: If FaceMode returns a non-retryable error response.
            FaceModeAPIError: If the response cannot be parsed.
        """
        data = await self._json(
            "POST",
            f"/sessions/{session_id}/reconnect",
            json={"room": room.to_payload()},
        )
        if isinstance(data, dict):
            data.setdefault("sessionId", session_id)
        return _parse_session(data, context="reconnect response", fallback=fallback)

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
        if json is not None:
            request_headers["Content-Type"] = "application/json"
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
                                "FaceMode API returned a non-object JSON response",
                                status_code=response.status,
                                body=payload,
                                retryable=False,
                            )
                        return payload

                    raise APIStatusError(
                        "FaceMode API returned an error",
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
                "facemode api request failed, retrying",
                extra={"attempt": attempt + 1, "method": method, "path": path},
            )
            await asyncio.sleep(self._conn_options._interval_for_retry(attempt))

        raise APIConnectionError("Failed to call FaceMode API after all retries.") from error


def _parse_session(
    data: dict[str, Any],
    *,
    context: str,
    fallback: SessionDetails | None = None,
) -> SessionDetails:
    try:
        return SessionDetails.from_api(data, fallback=fallback)
    except ValueError as error:
        raise FaceModeAPIError(f"FaceMode {context} was incomplete") from error


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


__all__ = ["DEFAULT_API_URL", "FaceModeAPI"]
