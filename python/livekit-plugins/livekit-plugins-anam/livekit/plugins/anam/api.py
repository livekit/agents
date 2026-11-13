from __future__ import annotations

from types import TracebackType

from pydantic import BaseModel, Field, JsonValue, ValidationError

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIError,
    http,
)

from .types import PersonaConfig, SessionOptions

DEFAULT_API_URL = "https://api.anam.ai"


class SessionResponse(BaseModel):
    session_id: str | None = Field(default=None, alias="sessionId")


class AnamAPI:
    """
    An asynchronous client for interacting with the Anam API.

    This class handles authentication and retries.
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: http.HttpSession | None = None,
    ) -> None:
        """
        Initializes the AnamAPI client.

        Args:
            api_key: Your Anam API key. If not provided, it will be read from
                     the ANAM_API_KEY environment variable.
            api_url: The base URL of the Anam API.
            conn_options: Connection options for the requests.
            session: An optional HttpSession to use for requests, else the job's.
        """
        self._api_key = api_key
        self._api_url = api_url
        self._conn_options = conn_options
        self._session = session

    async def __aenter__(self) -> AnamAPI:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    async def start_session(
        self,
        persona_config: PersonaConfig,
        livekit_url: str,
        livekit_token: str,
        session_options: SessionOptions | None = None,
    ) -> SessionResponse:
        """
        Starts an engine session.

        Args:
            session_options: Optional per-session output options (e.g. explicit
                video dimensions) forwarded to Anam as ``sessionOptions``. When
                ``None``, Anam uses the avatar model's default output.

        Returns:
            The session details, including sessionId and engine host info.
        """
        persona_config_payload: dict[str, JsonValue] = {
            "type": "ephemeral",
            "name": persona_config.name,
            "avatarId": persona_config.avatarId,
            "llmId": "CUSTOMER_CLIENT_V1",
        }

        if persona_config.avatarModel:
            persona_config_payload["avatarModel"] = persona_config.avatarModel

        if persona_config.directorNotes is not None:
            # drop unset (None) ones so Anam falls back to its model/cue defaults.
            director_notes = {
                k: v for k, v in vars(persona_config.directorNotes).items() if v is not None
            }
            if director_notes:
                persona_config_payload["directorNotes"] = director_notes

        payload: dict[str, JsonValue] = {
            "personaConfig": persona_config_payload,
        }
        payload["environment"] = {
            "livekitUrl": livekit_url,
            "livekitToken": livekit_token,
        }

        session_options_payload: dict[str, JsonValue] = {}
        if session_options is not None and (
            session_options.video_width is not None or session_options.video_height is not None
        ):
            # Anam's public API speaks camelCase pixel dimensions and wants them
            # as a matched pair: it rejects a lone width/height (and any
            # unsupported pair) with an HTTP 400, surfaced below as
            # APIStatusError, rather than downgrading. Fail fast on a half pair
            # rather than round-tripping a 400.
            if session_options.video_width is None or session_options.video_height is None:
                raise ValueError(
                    "video_width and video_height must be set together (both or neither)"
                )
            session_options_payload = {
                "videoWidth": session_options.video_width,
                "videoHeight": session_options.video_height,
            }

        if session_options is not None and session_options.show_ai_avatar_disclosure is not None:
            session_options_payload["showAIAvatarDisclosure"] = (
                session_options.show_ai_avatar_disclosure
            )

        if session_options_payload:
            payload["sessionOptions"] = session_options_payload

        headers = {
            "Authorization": f"Bearer {self._api_key}",  # Use API Key here
            "Content-Type": "application/json",
        }
        return await self._post("/v1/engine/session", payload, headers)

    async def _post(
        self, endpoint: str, payload: dict[str, JsonValue], headers: dict[str, str]
    ) -> SessionResponse:
        url = f"{self._api_url}{endpoint}"
        session = self._session or http.http_session()

        async def request() -> SessionResponse:
            async with session.post(
                url, headers=headers, json=payload, conn_options=self._conn_options
            ) as response:
                await response.raise_for_status()
                text = await response.text()
                try:
                    return SessionResponse.model_validate_json(text)
                except ValidationError as exc:
                    raise APIError(
                        "Anam returned an invalid response", kind="protocol", detail=text
                    ) from exc

        return await http.retry_request(request, conn_options=self._conn_options)
