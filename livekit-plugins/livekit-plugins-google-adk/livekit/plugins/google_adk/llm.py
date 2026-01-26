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
        return self._model

    async def _ensure_client_session(self) -> aiohttp.ClientSession:
        if self._client_session is None or self._client_session.closed:
            self._client_session = aiohttp.ClientSession()
        return self._client_session

    async def _create_session(self) -> str:
        session_id = f"session-{int(time.time() * 1000)}"
        url = (
            f"{self._api_base_url}/apps/{self._app_name}"
            f"/users/{self._user_id}/sessions/{session_id}"
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
        if self._session_id is None:
            if not self._auto_create_session:
                raise RuntimeError("No session_id provided")
            self._session_id = await self._create_session()
        return self._session_id

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **_: Any,
    ) -> LLMStream:
        """
        IMPORTANT:
        chat() MUST return an LLMStream, not a coroutine.
        LiveKit uses: `async with llm.chat(...)`
        """

        return LLMStream(
            llm_instance=self,
            api_base_url=self._api_base_url,
            app_name=self._app_name,
            user_id=self._user_id,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        if self._client_session and not self._client_session.closed:
            await self._client_session.close()
            self._client_session = None
