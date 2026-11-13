from __future__ import annotations

from types import TracebackType

from pydantic import JsonValue, TypeAdapter

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    APIError,
    NotGivenOr,
    http,
    utils,
)

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class AvatarioException(Exception):
    """Exception for Avatario errors"""


DEFAULT_API_URL = "https://avatario.ai/api/sdk"


class AvatarioAPI:
    """
    An asynchronous client for interacting with the Avatario API.

    This class handles authentication and retries.
    """

    def __init__(
        self,
        api_key: str,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        video_info: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: http.HttpSession | None = None,
    ) -> None:
        """
        Initializes the AvatarioAPI client.
        """
        self._avatar_id = avatar_id
        self._api_key = api_key

        self._video_info = video_info
        self._conn_options = conn_options
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> AvatarioAPI:
        if self._owns_session:
            self._session = http.HttpSession()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._owns_session and self._session:
            await self._session.aclose()

    async def start_session(
        self,
        *,
        livekit_agent_identity: NotGivenOr[str] = NOT_GIVEN,
        properties: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN,
    ) -> None:
        """
        Initiate a session request

        Args:
            livekit_agent_identity: the participant identity of the agent in room.
            properties: A dictionary consisting of room url and token used by the
                        avatario participant to join the room.
        """
        if not utils.is_given(livekit_agent_identity):
            raise AvatarioException(
                "the identity of agent needs to be provided "
                "to ensure its proper communication with avatario backend"
            )

        properties = properties if utils.is_given(properties) else {}

        payload: dict[str, object] = {
            "avatario_face_id": self._avatar_id,
            "agent_id": livekit_agent_identity,
            "livekit": properties,
        }

        if utils.is_given(self._video_info):
            payload.update(self._video_info)
        await self._post(payload)

    async def _post(self, payload: dict[str, object]) -> dict[str, JsonValue]:
        """
        Make a POST request to the Avatario API with retry logic.

        Args:
            payload: JSON payload for the request

        Returns:
            Response data as a dictionary

        Raises:
            APIError: If the request fails after all retries
        """
        session = self._session
        if session is None:
            raise RuntimeError("Session not initialized. Use 'async with AvatarioAPI(...)'.")

        async def request() -> dict[str, JsonValue]:
            async with session.post(
                f"{DEFAULT_API_URL}/start-session",
                headers={"Content-Type": "application/json", "x-api-key": self._api_key},
                json=payload,
                conn_options=self._conn_options,
            ) as response:
                await response.raise_for_status()
                try:
                    return _JSON_OBJECT.validate_python(await response.json())
                except ValueError as exc:
                    raise APIError(
                        "Avatario returned an invalid response", retryable=False
                    ) from exc

        return await http.retry_request(request, conn_options=self._conn_options)
