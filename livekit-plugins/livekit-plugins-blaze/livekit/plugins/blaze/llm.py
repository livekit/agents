"""
Blaze LLM Plugin for LiveKit Voice Agent

LLM plugin that interfaces with Blaze's chatbot service.

API Endpoint: POST /v1/voicebot-call/{bot_id}/chat-conversion-stream
Input: JSON array of messages [{ "role": "user"|"assistant", "content": str }]
Output: SSE streaming with data: { "content": str } format
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from typing import Optional, Dict, Any, List

import httpx
from livekit.agents import llm, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

from .log import logger
from ._config import BlazeConfig


class LLM(llm.LLM):
    """
    Blaze LLM Plugin for conversational AI.

    Interfaces with Blaze's chatbot service for voice agent conversations.

    Args:
        bot_id: Blaze bot identifier (required).
        api_url: Base URL for the chat service. If not provided,
                 reads from BLAZE_API_URL environment.
        auth_token: Bearer token for authentication. If not provided,
                    reads from BLAZE_API_TOKEN environment.
        deep_search: Enable deep search mode for enhanced retrieval.
        agentic_search: Enable agentic search capabilities.
        enable_tools: Enable tool/function calling.
        demographics: Optional user demographics dict (gender, age).
        timeout: Request timeout in seconds (default: 60.0).
        config: Optional BlazeConfig for centralized configuration.

    Example:
        >>> from livekit.plugins import blaze
        >>>
        >>> # Basic usage
        >>> llm = blaze.LLM(bot_id="my-bot-123")
        >>>
        >>> # With search features
        >>> llm = blaze.LLM(
        ...     bot_id="my-bot-123",
        ...     deep_search=True,
        ...     agentic_search=True,
        ...     demographics={"gender": "female", "age": 30}
        ... )
        >>>
        >>> # Using shared config
        >>> config = blaze.BlazeConfig(api_url="https://api.blaze.vn")
        >>> llm = blaze.LLM(config=config, bot_id="support-bot")
    """

    def __init__(
        self,
        *,
        bot_id: str,
        api_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        deep_search: bool = False,
        agentic_search: bool = False,
        enable_tools: bool = False,
        demographics: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        config: Optional[BlazeConfig] = None,
    ) -> None:
        super().__init__()

        # Load configuration
        self._config = config or BlazeConfig()

        # Resolve settings
        self._api_url = api_url or self._config.api_url
        self._bot_id = bot_id
        self._auth_token = auth_token or self._config.auth_token
        self._deep_search = deep_search
        self._agentic_search = agentic_search
        self._enable_tools = enable_tools
        self._demographics = demographics
        self._timeout = timeout or self._config.llm_timeout

        # Build chat URL
        self._chat_url = f"{self._api_url}/v1/voicebot-call/{bot_id}/chat-conversion-stream"

        # Shared HTTP client for connection pooling
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=5.0)
        )

        logger.info(f"BlazeLLM initialized: url={self._api_url}, bot_id={bot_id}")

    @property
    def provider(self) -> str:
        """Returns the provider name."""
        return "Blaze"

    @property
    def bot_id(self) -> str:
        """Returns the configured bot ID."""
        return self._bot_id

    async def aclose(self) -> None:
        """Close the shared HTTP client and release connections."""
        await self._client.aclose()
        await super().aclose()

    def update_options(
        self,
        *,
        deep_search: Optional[bool] = None,
        agentic_search: Optional[bool] = None,
        enable_tools: Optional[bool] = None,
        demographics: Optional[Dict[str, Any]] = None,
        auth_token: Optional[str] = None,
    ) -> None:
        """
        Update LLM options at runtime.

        Args:
            deep_search: Enable/disable deep search
            agentic_search: Enable/disable agentic search
            enable_tools: Enable/disable tool calling
            demographics: Update user demographics
            auth_token: New authentication token
        """
        if deep_search is not None:
            self._deep_search = deep_search
        if agentic_search is not None:
            self._agentic_search = agentic_search
        if enable_tools is not None:
            self._enable_tools = enable_tools
        if demographics is not None:
            self._demographics = demographics
        if auth_token is not None:
            self._auth_token = auth_token

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: Optional[List[llm.Tool]] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Optional[llm.ToolChoice] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "LLMStream":
        """
        Start a streaming chat completion.

        Args:
            chat_ctx: Chat context containing message history
            tools: List of tools (not used by Blaze LLM)
            conn_options: API connection options (retry, timeout)
            parallel_tool_calls: Parallel tool calls (not used)
            tool_choice: Tool choice (not used)
            extra_kwargs: Extra kwargs (not used)

        Returns:
            LLMStream that yields response chunks
        """
        if tools:
            logger.warning(
                "Blaze LLM does not support function calling. "
                "%d tool(s) provided will be ignored: %s",
                len(tools),
                ", ".join(t.name for t in tools),
            )

        return LLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class LLMStream(llm.LLMStream):
    """Streaming LLM implementation for Blaze chatbot."""

    def __init__(
        self,
        llm_instance: LLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: List[llm.Tool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            llm=llm_instance,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._llm = llm_instance
        self._chat_ctx = chat_ctx

    def _convert_messages(self) -> List[Dict[str, str]]:
        """Convert chat context messages to Blaze format.

        ChatRole is Literal['system', 'user', 'assistant'] — string comparisons.
        ChatMessage.content is list[ChatContent]; use text_content to get the string form.

        System messages are collected and merged into a single context
        message prepended to the conversation, preserving their original order.
        """
        messages: List[Dict[str, str]] = []
        system_parts: List[str] = []

        for msg in self._chat_ctx.messages:
            text = msg.text_content
            if not text:
                continue
            if msg.role == "system":
                system_parts.append(text)
            elif msg.role == "user":
                messages.append({"role": "user", "content": text})
            elif msg.role == "assistant":
                messages.append({"role": "assistant", "content": text})

        # Merge all system messages and prepend as unified context
        if system_parts:
            system_text = "\n\n".join(system_parts)
            messages.insert(
                0, {"role": "user", "content": f"[System Instructions]\n{system_text}"}
            )

        return messages

    async def _run(self) -> None:
        """Execute the chat completion and yield response chunks."""
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        try:
            messages = self._convert_messages()

            if not messages:
                logger.warning("[%s] No messages to send to chatbot", request_id)
                return

            # Build URL with query parameters using httpx for proper encoding
            query_params: Dict[str, str] = {
                "is_voice_call": "true",
                "use_tool_based": "true",
            }
            if self._llm._deep_search:
                query_params["deep_search"] = "true"
            if self._llm._agentic_search:
                query_params["agentic_search"] = "true"

            # Add demographics if available
            if self._llm._demographics:
                gender = self._llm._demographics.get("gender")
                if gender and gender != "unknown":
                    query_params["gender"] = str(gender)
                age = self._llm._demographics.get("age")
                if age is not None:  # Allow age=0
                    query_params["age"] = str(age)

            url = str(httpx.URL(self._llm._chat_url, params=query_params))

            # Prepare headers
            headers: Dict[str, str] = {"Content-Type": "application/json"}
            if self._llm._auth_token:
                headers["Authorization"] = f"Bearer {self._llm._auth_token}"

            logger.info(
                "[%s] LLM chat request: %d messages, bot=%s",
                request_id, len(messages), self._llm._bot_id,
            )

            # Make streaming request with retry on transient failures
            # Use shared client from LLM instance (connection pooling)
            conn_options = self._conn_options

            full_response = ""
            for attempt in range(conn_options.max_retry + 1):
                full_response = ""  # Reset on each attempt
                try:
                    async with self._llm._client.stream(
                        "POST",
                        url,
                        json=messages,
                        headers=headers,
                    ) as response:
                        if response.status_code >= 500:
                            error_text = (await response.aread()).decode(errors="replace")
                            if attempt < conn_options.max_retry:
                                delay = conn_options.retry_interval * (2 ** attempt)
                                jitter = delay * 0.1 * random.random()
                                logger.warning(
                                    "[%s] LLM attempt %d/%d failed (%d). "
                                    "Retrying in %.1fs…",
                                    request_id, attempt + 1,
                                    conn_options.max_retry + 1,
                                    response.status_code, delay,
                                )
                                await asyncio.sleep(delay + jitter)
                                continue
                            raise LLMError(
                                f"Chatbot service error: {response.status_code}",
                                status_code=response.status_code,
                            )

                        if response.status_code != 200:
                            error_text = (await response.aread()).decode(errors="replace")
                            logger.error(
                                "[%s] LLM error %d: %s",
                                request_id, response.status_code, error_text,
                            )
                            raise LLMError(
                                f"Chatbot service error: {response.status_code}",
                                status_code=response.status_code,
                            )

                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue

                            # Handle SSE format
                            if line.startswith("data: "):
                                data_str = line[6:]

                                if data_str.strip() == "[DONE]":
                                    logger.debug(
                                        "[%s] Stream completed with [DONE]",
                                        request_id,
                                    )
                                    break

                                try:
                                    data = json.loads(data_str)
                                    content = self._extract_content(data)
                                    if content:
                                        full_response += content
                                        chunk = llm.ChatChunk(
                                            id=request_id,
                                            delta=llm.ChoiceDelta(
                                                role="assistant",
                                                content=content,
                                            ),
                                        )
                                        self._event_ch.send_nowait(chunk)
                                except json.JSONDecodeError:
                                    logger.warning(
                                        "[%s] Failed to parse SSE data: %s",
                                        request_id, data_str[:100],
                                    )
                                    continue

                            # Handle raw JSON lines format
                            else:
                                try:
                                    data = json.loads(line)
                                    content = self._extract_content(data)
                                    if content:
                                        full_response += content
                                        chunk = llm.ChatChunk(
                                            id=request_id,
                                            delta=llm.ChoiceDelta(
                                                role="assistant",
                                                content=content,
                                            ),
                                        )
                                        self._event_ch.send_nowait(chunk)
                                except json.JSONDecodeError:
                                    logger.warning(
                                        "[%s] Failed to parse JSON line: %s",
                                        request_id, line[:100],
                                    )
                                    continue

                    break  # Success — exit retry loop

                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    if attempt < conn_options.max_retry:
                        delay = conn_options.retry_interval * (2 ** attempt)
                        jitter = delay * 0.1 * random.random()
                        logger.warning(
                            "[%s] LLM network error (attempt %d/%d): %s. "
                            "Retrying in %.1fs…",
                            request_id, attempt + 1,
                            conn_options.max_retry + 1, e, delay,
                        )
                        await asyncio.sleep(delay + jitter)
                    else:
                        raise LLMError(f"LLM network error: {e}") from e

            latency = time.monotonic() - start_time
            logger.info(
                "[%s] LLM chat completed: %d chars, latency=%.3fs",
                request_id, len(full_response), latency,
            )

        except LLMError:
            raise
        except Exception as e:
            latency = time.monotonic() - start_time
            logger.error(
                "[%s] LLM chat failed after %.3fs: %s", request_id, latency, e
            )
            raise LLMError(f"LLM chat failed: {str(e)}") from e

    def _extract_content(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract text content from various response formats."""
        # Format: {"content": "..."}
        if "content" in data:
            return data.get("content", "")

        # Format: {"text": "..."}
        if "text" in data:
            return data.get("text", "")

        # Format: {"delta": {"text": "..."}}
        if "delta" in data:
            delta = data.get("delta", {})
            if isinstance(delta, dict) and "text" in delta:
                return delta.get("text", "")

        return None


class LLMError(Exception):
    """Exception raised when LLM service encounters an error."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
