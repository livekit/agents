import asyncio
import os
import warnings
from typing import Any

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

from .log import logger


class TavusException(Exception):
    """Exception for Tavus errors"""


DEFAULT_API_URL = "https://tavusapi.com/v2"
# Stock face used when the caller provides neither a face nor a pal.
DEFAULT_FACE_ID = "r72f7f7f7c8b"


def _resolve_renamed_arg(
    new_value: NotGivenOr[str],
    deprecated_value: NotGivenOr[str],
    *,
    deprecated_name: str,
    new_name: str,
) -> NotGivenOr[str]:
    # Prefer the new arg; fall back to the deprecated alias and warn only when it's used.
    if deprecated_value and not new_value:
        warnings.warn(
            f"`{deprecated_name}` is deprecated, use `{new_name}` instead",
            DeprecationWarning,
            stacklevel=3,
        )
    return new_value or deprecated_value


def _deprecated_env(deprecated_name: str, new_name: str) -> str | None:
    # Read a deprecated env var, warning if it's set so callers migrate to `new_name`.
    value = os.getenv(deprecated_name)
    if value:
        warnings.warn(
            f"`{deprecated_name}` is deprecated, use `{new_name}` instead",
            DeprecationWarning,
            stacklevel=3,
        )
    return value


class TavusAPI:
    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        tavus_api_key = api_key or os.getenv("TAVUS_API_KEY")
        if tavus_api_key is None:
            raise TavusException("TAVUS_API_KEY must be set")
        self._api_key = tavus_api_key

        self._api_url = api_url or DEFAULT_API_URL
        self._conn_options = conn_options
        self._session = session or aiohttp.ClientSession()

    async def create_conversation(
        self,
        *,
        face_id: NotGivenOr[str] = NOT_GIVEN,
        pal_id: NotGivenOr[str] = NOT_GIVEN,
        replica_id: NotGivenOr[str] = NOT_GIVEN,
        persona_id: NotGivenOr[str] = NOT_GIVEN,
        properties: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        extra_payload: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> str:
        # `replica_id`/`persona_id` are deprecated aliases for `face_id`/`pal_id`.
        face_id = _resolve_renamed_arg(
            face_id, replica_id, deprecated_name="replica_id", new_name="face_id"
        )
        pal_id = _resolve_renamed_arg(
            pal_id, persona_id, deprecated_name="persona_id", new_name="pal_id"
        )

        face_id = (
            face_id
            or os.getenv("TAVUS_FACE_ID")
            or _deprecated_env("TAVUS_REPLICA_ID", "TAVUS_FACE_ID")
            or NOT_GIVEN
        )
        pal_id = (
            pal_id
            or os.getenv("TAVUS_PAL_ID")
            or _deprecated_env("TAVUS_PERSONA_ID", "TAVUS_PAL_ID")
            or NOT_GIVEN
        )

        if not pal_id:
            # no pal to reuse, so create one — falling back to the default face
            pal_id = await self.create_pal(default_face_id=face_id or DEFAULT_FACE_ID)

        properties = properties or {}
        payload: dict[str, Any] = {"pal_id": pal_id, "properties": properties}
        # send face_id only when given; otherwise the pal's default_face_id is used
        if face_id:
            payload["face_id"] = face_id
        if utils.is_given(extra_payload):
            payload.update(extra_payload)

        if "conversation_name" not in payload:
            payload["conversation_name"] = utils.shortuuid("lk_conversation_")

        response_data = await self._post("conversations", payload)
        return response_data["conversation_id"]  # type: ignore

    async def create_pal(
        self,
        name: NotGivenOr[str] = NOT_GIVEN,
        *,
        default_face_id: str,
        extra_payload: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> str:
        name = name or utils.shortuuid("lk_pal_")

        payload = {
            "pal_name": name,
            "default_face_id": default_face_id,
            "pipeline_mode": "echo",
            "layers": {
                "transport": {"transport_type": "livekit"},
            },
        }

        if utils.is_given(extra_payload):
            payload.update(extra_payload)

        response_data = await self._post("pals", payload)
        return response_data["pal_id"]  # type: ignore

    async def create_persona(
        self,
        name: NotGivenOr[str] = NOT_GIVEN,
        *,
        extra_payload: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> str:
        # Deprecated: use create_pal(). Kept on the legacy /v2/personas endpoint.
        warnings.warn(
            "`create_persona` is deprecated, use `create_pal` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        name = name or utils.shortuuid("lk_persona_")

        payload = {
            "persona_name": name,
            "pipeline_mode": "echo",
            "layers": {
                "transport": {"transport_type": "livekit"},
            },
        }

        if utils.is_given(extra_payload):
            payload.update(extra_payload)

        response_data = await self._post("personas", payload)
        return response_data["persona_id"]  # type: ignore

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Make a POST request to the Tavus API with retry logic.

        Args:
            endpoint: API endpoint path (without leading slash)
            payload: JSON payload for the request

        Returns:
            Response data as a dictionary

        Raises:
            APIConnectionError: If the request fails after all retries
        """
        for i in range(self._conn_options.max_retry):
            try:
                async with self._session.post(
                    f"{self._api_url}/{endpoint}",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self._api_key,
                    },
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
                    logger.warning("failed to call tavus api", extra={"error": str(e)})
                else:
                    logger.exception("failed to call tavus api")

                if i < self._conn_options.max_retry - 1:
                    await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to call Tavus API after all retries")
