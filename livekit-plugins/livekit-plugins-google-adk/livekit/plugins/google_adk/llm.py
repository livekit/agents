"""Google ADK LLM Plugin for LiveKit Agents Framework."""

import logging
import time
from typing import Any

import aiohttp
from typing_extensions import override
from yarl import URL

from livekit.agents import llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .llm_stream import LLMStream

logger = logging.getLogger(__name__)


class LLM(llm.LLM):
    """
    Google ADK LLM integration for LiveKit Agents.

    Provides text-only streaming LLM support via Google ADK server.
    Tools and system prompts are managed by ADK, not LiveKit.
    """

    provider = "google-adk"

    def __init__(
        self,
        *,
        api_base_url: str,
        app_name: str,
        user_id: str,
        model: str = "google-adk",
        session_id: str | None = None,
        auto_create_session: bool = True,
        request_timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Google ADK LLM.

        Args:
            api_base_url: URL of the ADK server
            app_name: ADK application name (must match ADK config)
            user_id: User identifier for session management
            model: Model identifier for tracking (default: "google-adk")
            session_id: Explicit session ID to reuse (optional)
            auto_create_session: Auto-create session if not provided (default: True)
            request_timeout: HTTP request timeout in seconds (default: 30.0)
            **kwargs: Additional arguments passed to base LLM class
        """
        super().__init__(**kwargs)
        self._api_base_url = api_base_url.rstrip("/")
        self._app_name = app_name
        self._user_id = user_id
        self._model = model
        self._session_id = session_id
        self._auto_create_session = auto_create_session
        self._request_timeout = request_timeout
        self._client_session: aiohttp.ClientSession | None = None

        logger.info(
            "[ADK] Google ADK LLM initialized - instructions parameter is NOT passed to ADK. "
            "ADK manages prompts internally through its own configuration."
        )

    @property
    def model(self) -> str:
        """Return the model identifier."""
        return self._model

    async def _ensure_client_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP client session."""
        if self._client_session is None or self._client_session.closed:
            self._client_session = aiohttp.ClientSession()
        return self._client_session

    async def _create_session(self) -> str:
        """
        Create a new ADK session.

        Returns:
            The created session ID

        Raises:
            RuntimeError: If session creation fails
        """
        session_id = f"session-{int(time.time() * 1000)}"
        # Use yarl URL builder to handle encoding automatically
        url = (
            URL(self._api_base_url)
            / "apps"
            / self._app_name
            / "users"
            / self._user_id
            / "sessions"
            / session_id
        )

        client = await self._ensure_client_session()
        async with client.post(
            url,
            json={},
            timeout=aiohttp.ClientTimeout(total=self._request_timeout),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(await resp.text())

        logger.info("[ADK] Created session %s", session_id)
        return session_id

    async def _ensure_session(self) -> str:
        """
        Ensure session exists, creating if necessary.

        Returns:
            The session ID

        Raises:
            RuntimeError: If no session_id and auto_create_session is False
        """
        if self._session_id is None:
            if not self._auto_create_session:
                raise RuntimeError("No session_id provided")
            self._session_id = await self._create_session()
        return self._session_id

    @override
    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: None = None,  # type: ignore[override]
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "LLMStream":
        """
        Stream chat completion from Google ADK.

        Note: parallel_tool_calls, tool_choice, and extra_kwargs are accepted
        for API compatibility but are not used by ADK (tools are configured in ADK).
        """
        # These parameters are accepted for base class compatibility
        # but are not used since ADK manages tools internally
        _ = parallel_tool_calls, tool_choice, extra_kwargs

        return LLMStream(
            llm_instance=self,
            api_base_url=self._api_base_url,
            app_name=self._app_name,
            user_id=self._user_id,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )

    @override
    async def aclose(self) -> None:
        """Close the LLM and cleanup resources."""
        if self._client_session and not self._client_session.closed:
            await self._client_session.close()
            self._client_session = None
