import asyncio
import os
from json import JSONDecodeError
from typing import TypeVar

import aiohttp
from pydantic import BaseModel, JsonValue, ValidationError

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    NotGivenOr,
    utils,
)
from livekit.agents.utils import http_context

from .log import logger


class TavusException(Exception):
    """Exception for Tavus errors"""


DEFAULT_API_URL = "https://tavusapi.com/v2"
# Stock Tavus PAL. Use create_pal() to create a PAL with the appearance you'd like.
DEFAULT_PAL_ID = "pb87e71797da"
# Stock Tavus face ("Lucy - Home", phoenix-4.5) used with DEFAULT_PAL_ID when no face is given.
DEFAULT_FACE_ID = "r4067604db72"


def _coalesce_with_deprecated(
    new_value: NotGivenOr[str],
    deprecated_value: NotGivenOr[str],
    *,
    deprecated_name: str,
    new_name: str,
) -> NotGivenOr[str]:
    # Prefer the new arg; fall back to the deprecated alias and warn only when it's used.
    if deprecated_value and not new_value:
        logger.warning(f"`{deprecated_name}` is deprecated, use `{new_name}` instead")
    return new_value or deprecated_value


def _deprecated_env(deprecated_name: str, new_name: str) -> str | None:
    # Read a deprecated env var, warning if it's set so callers migrate to `new_name`.
    value = os.getenv(deprecated_name)
    if value:
        logger.warning(f"`{deprecated_name}` is deprecated, use `{new_name}` instead")
    return value


class _ConversationResponse(BaseModel):
    conversation_id: str


class _PalResponse(BaseModel):
    pal_id: str


class _PersonaResponse(BaseModel):
    persona_id: str


_ResponseT = TypeVar("_ResponseT", bound=BaseModel)


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
        properties: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN,
        extra_payload: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN,
    ) -> str:
        # `replica_id`/`persona_id` are deprecated aliases for `face_id`/`pal_id`.
        face_id = _coalesce_with_deprecated(
            face_id, replica_id, deprecated_name="replica_id", new_name="face_id"
        )
        pal_id = _coalesce_with_deprecated(
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

        # extra_payload overrides the payload wholesale, so a pal named there counts
        # as user-supplied when deciding whether to fill in the stock face.
        extra = dict(extra_payload) if utils.is_given(extra_payload) else {}
        if not pal_id and not extra.get("pal_id"):
            # no pal supplied — use the default stock pal and, unless overridden, its stock face
            pal_id = DEFAULT_PAL_ID
            face_id = face_id or DEFAULT_FACE_ID

        properties = properties if utils.is_given(properties) else {}
        payload: dict[str, JsonValue] = {"properties": properties}
        if utils.is_given(pal_id):
            payload["pal_id"] = pal_id
        # a user-supplied pal carries its own default face, so only send face_id when we have one
        if utils.is_given(face_id) and face_id:
            payload["face_id"] = face_id
        payload.update(extra)

        if "conversation_name" not in payload:
            payload["conversation_name"] = utils.shortuuid("lk_conversation_")

        response_data = await self._post(
            "conversations", payload, response_type=_ConversationResponse
        )
        return response_data.conversation_id

    async def create_pal(
        self,
        name: NotGivenOr[str] = NOT_GIVEN,
        *,
        default_face_id: str,
        extra_payload: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN,
    ) -> str:
        name = name if utils.is_given(name) and name else utils.shortuuid("lk_pal_")

        payload: dict[str, JsonValue] = {
            "pal_name": name,
            "default_face_id": default_face_id,
            "pipeline_mode": "echo",
            "layers": {
                "transport": {"transport_type": "livekit"},
            },
        }

        if utils.is_given(extra_payload):
            payload.update(extra_payload)

        response_data = await self._post("pals", payload, response_type=_PalResponse)
        return response_data.pal_id

    async def create_persona(
        self,
        name: NotGivenOr[str] = NOT_GIVEN,
        *,
        extra_payload: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN,
    ) -> str:
        # Deprecated: use create_pal(). Kept on the legacy /v2/personas endpoint.
        logger.warning("`create_persona` is deprecated, use `create_pal` instead")
        name = name if utils.is_given(name) and name else utils.shortuuid("lk_persona_")

        payload: dict[str, JsonValue] = {
            "persona_name": name,
            "pipeline_mode": "echo",
            "layers": {
                "transport": {"transport_type": "livekit"},
            },
        }

        if utils.is_given(extra_payload):
            payload.update(extra_payload)

        response_data = await self._post("personas", payload, response_type=_PersonaResponse)
        return response_data.persona_id

    async def _post(
        self, endpoint: str, payload: dict[str, JsonValue], *, response_type: type[_ResponseT]
    ) -> _ResponseT:
        """
        Make a POST request to the Tavus API with retry logic.

        Args:
            endpoint: API endpoint path (without leading slash)
            payload: JSON payload for the request

        Returns:
            Response data as a dictionary

        Raises:
            APIStatusError: If Tavus returns a non-retryable error, or a retryable one
                persists after all retries
            APIConnectionError: If the request fails after all retries
        """
        async with self._conn_options.total_deadline():
            for attempt in range(self._conn_options.max_retry + 1):
                try:
                    async with http_context.request(
                        self._session,
                        "POST",
                        f"{self._api_url}/{endpoint}",
                        headers={
                            "Content-Type": "application/json",
                            "x-api-key": self._api_key,
                        },
                        json=payload,
                        conn_options=self._conn_options,
                    ) as response:
                        if not response.ok:
                            text = await response.text()
                            raise APIStatusError(
                                "Server returned an error", status_code=response.status, body=text
                            )
                        try:
                            return response_type.model_validate(await response.json())
                        except (ValidationError, JSONDecodeError, aiohttp.ContentTypeError) as e:
                            # Tavus already accepted the POST; retrying could create a duplicate.
                            raise APIConnectionError(
                                "Tavus returned an invalid response", retryable=False
                            ) from e
                except APIStatusError as e:
                    # A 4xx such as a bad API key will fail the same way every time.
                    if not e.retryable:
                        raise
                    logger.warning(
                        "failed to call tavus api",
                        extra={"attempt": attempt + 1, "status_code": e.status_code},
                    )
                    if attempt >= self._conn_options.max_retry:
                        raise
                    await asyncio.sleep(self._conn_options.retry_interval)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(
                        "failed to call tavus api", extra={"attempt": attempt + 1, "error": str(e)}
                    )
                    if attempt >= self._conn_options.max_retry:
                        raise APIConnectionError(
                            "Failed to call Tavus API after all retries"
                        ) from e
                    await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to call Tavus API after all retries")
