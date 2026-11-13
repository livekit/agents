import os
from typing import TypeVar

from pydantic import BaseModel, JsonValue, ValidationError

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    APIError,
    NotGivenOr,
    http,
    utils,
)


class TavusException(Exception):
    """Exception for Tavus errors"""


DEFAULT_API_URL = "https://tavusapi.com/v2"
# Stock Tavus PAL. Use create_pal() to create a PAL with the appearance you'd like.
DEFAULT_PAL_ID = "pb87e71797da"
# Stock Tavus face ("Lucy - Home", phoenix-4.5) used with DEFAULT_PAL_ID when no face is given.
DEFAULT_FACE_ID = "r4067604db72"


class _ConversationResponse(BaseModel):
    conversation_id: str


class _PalResponse(BaseModel):
    pal_id: str


_ResponseT = TypeVar("_ResponseT", bound=BaseModel)


class TavusAPI:
    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: http.HttpSession | None = None,
    ) -> None:
        tavus_api_key = api_key or os.getenv("TAVUS_API_KEY")
        if tavus_api_key is None:
            raise TavusException("TAVUS_API_KEY must be set")
        self._api_key = tavus_api_key

        self._api_url = api_url or DEFAULT_API_URL
        self._conn_options = conn_options
        self._session = session or http.HttpSession()

    async def create_conversation(
        self,
        *,
        face_id: NotGivenOr[str] = NOT_GIVEN,
        pal_id: NotGivenOr[str] = NOT_GIVEN,
        properties: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN,
        extra_payload: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN,
    ) -> str:
        face_id = face_id or os.getenv("TAVUS_FACE_ID") or NOT_GIVEN
        pal_id = pal_id or os.getenv("TAVUS_PAL_ID") or NOT_GIVEN

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
            APIStatusError: If Tavus returns a non-retryable error
            APIConnectionError: If the request fails after all retries
        """

        async def request() -> _ResponseT:
            async with self._session.post(
                f"{self._api_url}/{endpoint}",
                headers={"Content-Type": "application/json", "x-api-key": self._api_key},
                json=payload,
                conn_options=self._conn_options,
            ) as response:
                await response.raise_for_status()
                text = await response.text()
                try:
                    return response_type.model_validate_json(text)
                except ValidationError as exc:
                    raise APIError(
                        "Tavus returned an invalid response", kind="protocol", detail=text
                    ) from exc

        return await http.retry_request(request, conn_options=self._conn_options)
