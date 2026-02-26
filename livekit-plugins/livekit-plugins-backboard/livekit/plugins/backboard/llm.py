"""
Backboard.io LLM plugin for LiveKit Agents.

Implements LiveKit's llm.LLM / llm.LLMStream interface to integrate
Backboard's thread-based conversation API (with persistent memory
and RAG) into a LiveKit Agents voice pipeline.

Backboard manages its own conversation history via thread_id,
so this plugin extracts only the latest user message from ChatContext
and lets Backboard handle context, memory retrieval, and RAG.

Frame Flow:
    Audio -> STT -> SpeechEvent -> AgentSession -> ChatContext
      -> BackboardLLM.chat() -> BackboardLLMStream._run()
      -> ChatChunk tokens -> TTS -> Audio
"""

import json
import os
import uuid
from typing import Any, Optional

import httpx
from livekit.agents import llm, utils
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    CompletionUsage,
    Tool,
)

try:
    from livekit.agents.types import (
        DEFAULT_API_CONNECT_OPTIONS,
        APIConnectOptions,
        NOT_GIVEN,
        NotGivenOr,
    )
except ImportError:
    from livekit.agents import (
        DEFAULT_API_CONNECT_OPTIONS,
        APIConnectOptions,
        NOT_GIVEN,
        NotGivenOr,
    )

from .session import SessionStore

logger = utils.log.logger

_DEFAULT_BASE_URL = "https://app.backboard.io/api"


class BackboardLLM(llm.LLM):
    """
    LiveKit Agents LLM plugin that routes inference through Backboard.io.

    Backboard provides:
    - Persistent memory across conversations
    - RAG over uploaded documents
    - Thread-based context management
    - 1,800+ LLM backends via a single API

    Args:
        api_key: Backboard API key. Falls back to ``BACKBOARD_API_KEY`` env var.
        base_url: Backboard API base URL. Defaults to ``https://app.backboard.io/api``.
        assistant_id: Backboard assistant ID. Required for thread creation.
        user_id: User identity for thread management. Defaults to ``"default"``.
        llm_provider: LLM provider name (e.g. ``"openai"``, ``"anthropic"``, ``"xai"``).
        model_name: Model name (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-5-20250929"``).
        memory: Memory mode per message — ``"auto"`` (read+write), ``"readonly"`` (read only),
            or ``None`` (disabled). Defaults to ``"auto"``.
        session_store: Optional custom ``SessionStore`` for thread management.

    Example::

        from livekit.agents import AgentSession
        from livekit.plugins import backboard

        session = AgentSession(
            llm=backboard.LLM(
                assistant_id="your-assistant-id",
                llm_provider="openai",
                model_name="gpt-4o",
            ),
            stt=...,
            tts=...,
        )
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = _DEFAULT_BASE_URL,
        assistant_id: str = "",
        user_id: str = "default",
        llm_provider: str = "openai",
        model_name: str = "gpt-4o",
        memory: Optional[str] = "auto",
        session_store: Optional[SessionStore] = None,
    ) -> None:
        super().__init__()
        self._api_key = api_key or os.environ.get("BACKBOARD_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Backboard API key is required. Set BACKBOARD_API_KEY environment "
                "variable or pass api_key to BackboardLLM()."
            )

        self._base_url = base_url
        self._assistant_id = assistant_id
        self._user_id = user_id
        self._llm_provider = llm_provider
        self._model_name = model_name
        self._memory = memory
        self._session_store = session_store or SessionStore(
            api_key=self._api_key,
            base_url=self._base_url,
            assistant_id=self._assistant_id,
        )
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "backboard"

    def set_user_id(self, user_id: str) -> None:
        """Update the current user identity for thread management."""
        self._user_id = user_id

    def set_assistant_id(self, assistant_id: str) -> None:
        """Update the assistant ID and propagate to the session store."""
        self._assistant_id = assistant_id
        self._session_store.set_assistant_id(assistant_id)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30, connect=5),
            )
        return self._client

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "BackboardLLMStream":
        return BackboardLLMStream(
            llm_instance=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            client=self._get_client(),
            api_key=self._api_key,
            base_url=self._base_url,
            user_id=self._user_id,
            assistant_id=self._assistant_id,
            llm_provider=self._llm_provider,
            model_name=self._model_name,
            memory=self._memory,
            session_store=self._session_store,
        )

    async def aclose(self) -> None:
        """Clean up HTTP client and session store."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        await self._session_store.aclose()


# Public alias matching LiveKit plugin conventions (e.g. openai.LLM)
LLM = BackboardLLM


class BackboardLLMStream(llm.LLMStream):
    """
    Streaming response from Backboard API.

    Parses Backboard's SSE (Server-Sent Events) format and emits
    ``ChatChunk`` objects into the LiveKit Agents pipeline.
    """

    def __init__(
        self,
        *,
        llm_instance: BackboardLLM,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        client: httpx.AsyncClient,
        api_key: str,
        base_url: str,
        user_id: str,
        assistant_id: str,
        llm_provider: str,
        model_name: str,
        memory: Optional[str],
        session_store: SessionStore,
    ) -> None:
        super().__init__(
            llm_instance, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options
        )
        self._client = client
        self._api_key = api_key
        self._base_url = base_url
        self._user_id = user_id
        self._assistant_id = assistant_id
        self._llm_provider = llm_provider
        self._model_name = model_name
        self._memory = memory
        self._session_store = session_store

    def _extract_user_message(self) -> str:
        """
        Extract the latest user message from ChatContext.

        Backboard manages its own conversation history via thread_id,
        so we only need the most recent user utterance. Falls back to
        developer/system instructions (used by ``generate_reply()``).
        """
        for msg in reversed(self._chat_ctx.messages()):
            if msg.role == "user" and msg.text_content:
                return msg.text_content
        for msg in reversed(self._chat_ctx.messages()):
            if msg.role in ("developer", "system") and msg.text_content:
                return msg.text_content
        return ""

    async def _run(self) -> None:
        """
        Stream a response from the Backboard API.

        1. Extracts the latest user message from ChatContext
        2. Resolves thread_id via SessionStore
        3. Streams from Backboard's SSE endpoint
        4. Emits ChatChunk objects for the LiveKit pipeline
        """
        user_message = self._extract_user_message()
        if not user_message:
            logger.warning("No user message found in ChatContext")
            self._event_ch.send_nowait(
                ChatChunk(
                    id=str(uuid.uuid4()),
                    delta=ChoiceDelta(
                        role="assistant",
                        content="I didn't catch that. Could you please repeat?",
                    ),
                )
            )
            return

        thread_id = await self._session_store.get_or_create_thread(self._user_id)
        logger.debug(f"Streaming from thread {thread_id} for user {self._user_id}")

        request_id = str(uuid.uuid4())
        total_tokens = 0

        data: dict[str, str] = {
            "content": user_message,
            "llm_provider": self._llm_provider,
            "model_name": self._model_name,
            "stream": "true",
        }
        if self._memory:
            data["memory"] = self._memory

        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/threads/{thread_id}/messages",
                headers={
                    "X-API-Key": self._api_key,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data=data,
            ) as response:
                response.raise_for_status()

                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk

                    while "\n\n" in buffer:
                        event, buffer = buffer.split("\n\n", 1)

                        for line in event.split("\n"):
                            if not line.startswith("data: "):
                                continue

                            payload = line[6:]
                            if payload == "[DONE]":
                                return

                            try:
                                parsed = json.loads(payload)
                            except json.JSONDecodeError:
                                continue

                            event_type = parsed.get("type")

                            if event_type == "content_streaming":
                                content = parsed.get("content")
                                if content:
                                    total_tokens += 1
                                    self._event_ch.send_nowait(
                                        ChatChunk(
                                            id=request_id,
                                            delta=ChoiceDelta(
                                                role="assistant",
                                                content=content,
                                            ),
                                        )
                                    )

                            elif event_type in ("message_complete", "run_ended"):
                                usage = CompletionUsage(
                                    completion_tokens=parsed.get(
                                        "output_tokens", total_tokens
                                    ),
                                    prompt_tokens=parsed.get("input_tokens", 0),
                                    total_tokens=parsed.get(
                                        "total_tokens", total_tokens
                                    ),
                                )
                                self._event_ch.send_nowait(
                                    ChatChunk(id=request_id, usage=usage)
                                )
                                return

                            elif event_type == "error":
                                error_msg = parsed.get("error", "Unknown error")
                                logger.error(f"Backboard stream error: {error_msg}")
                                self._event_ch.send_nowait(
                                    ChatChunk(
                                        id=request_id,
                                        delta=ChoiceDelta(
                                            role="assistant",
                                            content=f"Error: {error_msg}",
                                        ),
                                    )
                                )
                                return

        except httpx.TimeoutException as e:
            logger.error(f"Backboard timeout: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"Backboard HTTP {e.response.status_code}: {e}")
            raise
        except Exception as e:
            logger.error(f"Backboard stream error: {e}")
            raise
