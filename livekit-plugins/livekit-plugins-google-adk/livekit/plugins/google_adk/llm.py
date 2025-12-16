"""Google ADK LLM Plugin for LiveKit Agents Framework."""

import logging
import time
from typing import Any

import aiohttp

from livekit.agents import llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .llm_stream import LLMStream

logger = logging.getLogger(__name__)


class LLM(llm.LLM):
    """
    Google ADK (Agent Development Kit) LLM integration for LiveKit Agents.

    This plugin allows LiveKit voice agents to leverage Google ADK's sophisticated
    orchestration capabilities including:
    - Multi-agent coordination
    - Complex workflow management
    - MCP tool integration
    - Native telemetry and observability
    - Session management

    Example:
        ```python
        from livekit.plugins.google_adk import LLM as GoogleADK

        adk_llm = GoogleADK(
            api_base_url="http://localhost:8000",
            app_name="my-agent",
            user_id="user_123",
        )
        ```
    """

    def __init__(
        self,
        *,
        api_base_url: str = "http://localhost:8000",
        app_name: str,
        user_id: str,
        model: str = "google-adk",
        session_id: str | None = None,
        use_room_name_as_session: bool = False,
        use_participant_identity_as_session: bool = False,
        auto_create_session: bool = True,
        request_timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Google ADK LLM.

        Args:
            api_base_url: Base URL of the ADK server
            app_name: ADK application name
            user_id: User identifier for ADK session
            model: Model identifier (for tracking purposes)
            session_id: Optional existing session ID to reuse
            use_room_name_as_session: Use LiveKit room name as ADK session ID
            use_participant_identity_as_session: Use LiveKit participant identity as session ID
            auto_create_session: Automatically create session if not provided
            request_timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to base LLM class
        """
        super().__init__(**kwargs)
        self._api_base_url = api_base_url.rstrip("/")
        self._app_name = app_name
        self._user_id = user_id
        self._model = model
        self._session_id = session_id
        self._use_room_name = use_room_name_as_session
        self._use_participant_identity = use_participant_identity_as_session
        self._auto_create_session = auto_create_session
        self._request_timeout = request_timeout
        self._client_session: aiohttp.ClientSession | None = None
        self._room_name: str | None = None
        self._participant_identity: str | None = None

    @property
    def model(self) -> str:
        """Get the model name/identifier."""
        return self._model

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "google-adk"

    def set_livekit_context(
        self, *, room_name: str | None = None, participant_identity: str | None = None
    ) -> None:
        """
        Set LiveKit context for session ID generation.

        This is called automatically by the agent if using room-based or participant-based sessions.

        Args:
            room_name: LiveKit room name
            participant_identity: LiveKit participant identity
        """
        self._room_name = room_name
        self._participant_identity = participant_identity

        if self._use_room_name and room_name and not self._session_id:
            self._session_id = f"room-{room_name}"
            logger.info(f"[ADK] Using room-based session: {self._session_id}")

        elif self._use_participant_identity and participant_identity and not self._session_id:
            self._session_id = f"participant-{participant_identity}"
            logger.info(f"[ADK] Using participant-based session: {self._session_id}")

    async def _ensure_client_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp client session exists."""
        if self._client_session is None or self._client_session.closed:
            self._client_session = aiohttp.ClientSession()
        return self._client_session

    async def _create_session(self) -> str:
        """Create a new ADK session."""
        session_id = f"session-{int(time.time() * 1000)}"
        url = f"{self._api_base_url}/apps/{self._app_name}/users/{self._user_id}/sessions/{session_id}"

        logger.info(f"[ADK] Creating session: {session_id}")

        client = await self._ensure_client_session()
        try:
            async with client.post(
                url,
                json={},
                timeout=aiohttp.ClientTimeout(total=self._request_timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"Failed to create ADK session (status {resp.status}): {error_text}"
                    )
                data = await resp.json()
                logger.info(f"[ADK] Session created successfully: {data}")
                return session_id
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to connect to ADK server: {e}") from e

    async def _ensure_session(self) -> str:
        """Ensure session exists, creating if necessary."""
        if self._session_id is None:
            if not self._auto_create_session:
                raise RuntimeError("No session_id provided and auto_create_session is False")
            self._session_id = await self._create_session()
        return self._session_id

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        """
        Stream chat completion from Google ADK.

        Args:
            chat_ctx: Chat context containing conversation history
            tools: Optional list of function tools (forwarded to ADK)
            conn_options: API connection options
            **kwargs: Additional arguments

        Returns:
            LLMStream that yields ChatChunk objects with streamed content
        """
        # Try to get LiveKit context from chat_ctx if available
        # Check if first message has extra data with room/participant info
        if chat_ctx.items and not self._session_id:
            first_item = chat_ctx.items[0]
            if hasattr(first_item, "extra") and first_item.extra:
                room_name = first_item.extra.get("room_name")
                participant_identity = first_item.extra.get("participant_identity")
                if room_name or participant_identity:
                    self.set_livekit_context(
                        room_name=room_name, participant_identity=participant_identity
                    )

        # Get or create session ID
        if self._session_id is None:
            if not self._auto_create_session:
                raise RuntimeError("No session_id provided and auto_create_session is False")
            # Mark for lazy creation
            session_id = f"__pending__{int(time.time() * 1000)}"
        else:
            session_id = self._session_id

        # Create and return the LLMStream
        return LLMStream(
            llm_instance=self,
            api_base_url=self._api_base_url,
            app_name=self._app_name,
            user_id=self._user_id,
            session_id=session_id,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        """Close the LLM and cleanup resources."""
        if self._client_session and not self._client_session.closed:
            await self._client_session.close()
            self._client_session = None
