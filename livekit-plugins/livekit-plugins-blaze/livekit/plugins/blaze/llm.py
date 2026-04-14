"""
Blaze LLM Plugin for LiveKit Voice Agent

LLM plugin that interfaces with Blaze's chatbot service.

API Endpoint: POST /v1/voicebot-call/{bot_id}/chat-conversion-stream
Input: JSON array of messages [{ "role": "user"|"assistant", "content": str }]
Output: SSE streaming with data: { "content": str } format

Supports:
  - Streaming chat completion via SSE
  - Deep search and agentic search modes
  - User demographics for personalization
  - Function/tool calling (when enabled and supported by the bot)
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

import httpx

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    NotGivenOr,
    llm,
)

from ._config import BlazeConfig
from .log import logger


class LLM(llm.LLM):
    """
    Blaze LLM Plugin for conversational AI.

    Interfaces with Blaze's chatbot service for voice agent conversations.
    Supports streaming chat, demographics-based personalization, and
    function calling.

    Args:
        bot_id: Blaze bot identifier (required).
        api_url: Base URL for the chat service.
        auth_token: Bearer token for authentication.
        deep_search: Enable deep search mode.
        agentic_search: Enable agentic search mode.
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
        >>> # With function calling
        >>> llm = blaze.LLM(bot_id="my-bot", enable_tools=True)
    """

    def __init__(
        self,
        *,
        bot_id: str,
        api_url: str | None = None,
        auth_token: str | None = None,
        deep_search: bool = False,
        agentic_search: bool = False,
        enable_tools: bool = False,
        demographics: dict[str, Any] | None = None,
        timeout: float | None = None,
        config: BlazeConfig | None = None,
    ) -> None:
        super().__init__()

        self._config = config or BlazeConfig()
        self._api_url = api_url or self._config.api_url
        self._bot_id = bot_id
        self._auth_token = auth_token or self._config.api_token
        self._deep_search = deep_search
        self._agentic_search = agentic_search
        self._enable_tools = enable_tools
        self._demographics = demographics
        self._timeout = timeout if timeout is not None else self._config.llm_timeout
        self._chat_url = f"{self._api_url}/v1/voicebot-call/{bot_id}/chat-conversion-stream"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=5.0))

        logger.info("BlazeLLM initialized: url=%s, bot_id=%s", self._api_url, bot_id)

    @property
    def provider(self) -> str:
        return "Blaze"

    @property
    def bot_id(self) -> str:
        return self._bot_id

    async def aclose(self) -> None:
        await self._client.aclose()
        await super().aclose()

    def update_options(
        self,
        *,
        deep_search: bool | None = None,
        agentic_search: bool | None = None,
        enable_tools: bool | None = None,
        demographics: dict[str, Any] | None = None,
        auth_token: str | None = None,
    ) -> None:
        """Update LLM options at runtime."""
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
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        """Start a streaming chat completion.

        Args:
            chat_ctx: Chat context containing message history.
            tools: List of tools available for function calling.
                   Requires ``enable_tools=True`` in constructor.
            conn_options: API connection options (retry, timeout).
            parallel_tool_calls: Not used by Blaze LLM.
            tool_choice: Not used by Blaze LLM.
            extra_kwargs: Not used by Blaze LLM.

        Returns:
            LLMStream that yields response chunks (text and/or tool calls).
        """
        if tools and not self._enable_tools:
            logger.warning(
                "Tools provided but enable_tools=False. "
                "%d tool(s) will be ignored: %s",
                len(tools),
                ", ".join(t.id for t in tools),
            )
            tools = None

        return LLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class LLMStream(llm.LLMStream):
    """Streaming LLM implementation for Blaze chatbot.

    Handles both plain text responses and function call responses.

    Function call response format from Blaze API::

        data: {"tool_calls": [{"id": "call_xxx", "function": {"name": "fn", "arguments": "{...}"}}]}
        data: {"content": "text response"}
        data: [DONE]
    """

    def __init__(
        self,
        llm_instance: LLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            llm=llm_instance,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._blaze_llm = llm_instance

    def _convert_messages(self) -> list[dict[str, str]]:
        """Convert chat context messages to Blaze format.

        System/developer messages are SKIPPED because the Blaze chatapp already
        loads the voicebot prompt from the database. Sending them again would
        cause double-prompting and format conflicts.

        Function call results are included as user messages so the LLM can
        continue the conversation after tool execution.
        """
        messages: list[dict[str, str]] = []

        for msg in self._chat_ctx.messages():
            text = msg.text_content
            if not text:
                continue
            if msg.role in ("system", "developer"):
                continue
            elif msg.role == "user":
                messages.append({"role": "user", "content": text})
            elif msg.role == "assistant":
                clean = re.sub(r"<img>[^<]*</img>", "", text, flags=re.IGNORECASE).strip()
                if clean:
                    messages.append({"role": "assistant", "content": clean})

        return messages

    def _build_tools_param(self) -> list[dict[str, Any]] | None:
        """Serialize tools into the format expected by Blaze API.

        Returns None if no tools are configured.

        Blaze API accepts OpenAI-compatible tool format::

            [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
        """
        if not self._tools:
            return None

        tool_defs: list[dict[str, Any]] = []
        for tool in self._tools:
            # FunctionTool has callable_function with description and parameters
            if hasattr(tool, "callable_function"):
                fn = tool.callable_function
                tool_def: dict[str, Any] = {
                    "type": "function",
                    "function": {
                        "name": fn.name,
                    },
                }
                if fn.description:
                    tool_def["function"]["description"] = fn.description
                if fn.parameters:
                    tool_def["function"]["parameters"] = fn.parameters
                tool_defs.append(tool_def)
            else:
                # Generic tool — use id as name
                tool_defs.append({
                    "type": "function",
                    "function": {"name": tool.id},
                })

        return tool_defs if tool_defs else None

    async def _run(self) -> None:
        """Execute the chat completion and yield response chunks.

        Handles both text content and function/tool call responses.
        Retry logic is handled by the base class ``_main_task()``.
        """
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        messages = self._convert_messages()
        if not messages:
            logger.warning("[%s] No messages to send to chatbot", request_id)
            return

        blaze = self._blaze_llm
        query_params: dict[str, str] = {
            "is_voice_call": "true",
            "agent_stream": "true",
            "use_tool_based": "true" if blaze._enable_tools else "false",
        }
        if blaze._deep_search:
            query_params["deep_search"] = "true"
        if blaze._agentic_search:
            query_params["agentic_search"] = "true"
        if blaze._demographics:
            gender = blaze._demographics.get("gender")
            if gender and gender != "unknown":
                query_params["gender"] = str(gender)
            age = blaze._demographics.get("age")
            if age is not None:
                query_params["age"] = str(age)

        url = str(httpx.URL(blaze._chat_url, params=query_params))

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if blaze._auth_token:
            headers["Authorization"] = f"Bearer {blaze._auth_token}"

        # Build request body
        body: dict[str, Any] = {"messages": messages}
        tools_param = self._build_tools_param()
        if tools_param:
            body["tools"] = tools_param

        logger.info(
            "[%s] LLM chat request: %d messages, bot=%s, tools=%d",
            request_id,
            len(messages),
            blaze._bot_id,
            len(self._tools),
        )

        full_response = ""
        try:
            async with blaze._client.stream(
                "POST",
                url,
                json=messages if not tools_param else body,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_text = (await response.aread()).decode(errors="replace")
                    raise APIStatusError(
                        f"Chatbot service error {response.status_code}: {error_text}",
                        status_code=response.status_code,
                        request_id=request_id,
                        body=error_text,
                    )

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    data_str = line[6:] if line.startswith("data: ") else line

                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("[%s] Failed to parse data: %s", request_id, data_str[:100])
                        continue

                    # Handle tool calls
                    tool_calls = self._extract_tool_calls(data)
                    if tool_calls:
                        chunk = llm.ChatChunk(
                            id=request_id,
                            delta=llm.ChoiceDelta(
                                role="assistant",
                                tool_calls=tool_calls,
                            ),
                        )
                        self._event_ch.send_nowait(chunk)
                        continue

                    # Handle text content
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

        except httpx.TimeoutException as e:
            raise APITimeoutError(f"LLM request timed out: {e}") from e
        except httpx.NetworkError as e:
            raise APIConnectionError(f"LLM network error: {e}") from e
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError(f"LLM connection error: {e}") from e

        latency = time.monotonic() - start_time
        logger.info(
            "[%s] LLM chat completed: %d chars, latency=%.3fs",
            request_id,
            len(full_response),
            latency,
        )

    def _extract_content(self, data: dict[str, Any]) -> str | None:
        """Extract text content from various response formats."""
        if "content" in data and data["content"] is not None:
            return str(data["content"])
        if "text" in data and data["text"] is not None:
            return str(data["text"])
        if "delta" in data:
            delta = data.get("delta", {})
            if isinstance(delta, dict) and delta.get("text") is not None:
                return str(delta["text"])
            if isinstance(delta, dict) and delta.get("content") is not None:
                return str(delta["content"])
        return None

    def _extract_tool_calls(self, data: dict[str, Any]) -> list[llm.FunctionToolCall]:
        """Extract function tool calls from the response data.

        Supports OpenAI-compatible format::

            {"tool_calls": [{"id": "call_xxx", "function": {"name": "fn", "arguments": "{...}"}}]}

        Also supports delta format::

            {"delta": {"tool_calls": [...]}}
        """
        raw_calls = data.get("tool_calls")
        if raw_calls is None and "delta" in data:
            delta = data.get("delta", {})
            if isinstance(delta, dict):
                raw_calls = delta.get("tool_calls")

        if not raw_calls or not isinstance(raw_calls, list):
            return []

        result: list[llm.FunctionToolCall] = []
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function", {})
            if not isinstance(fn, dict):
                continue
            name = fn.get("name", "")
            if not name:
                continue
            arguments = fn.get("arguments", "")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            call_id = call.get("id", f"call_{uuid.uuid4().hex[:8]}")

            result.append(
                llm.FunctionToolCall(
                    name=name,
                    arguments=arguments,
                    call_id=call_id,
                )
            )

        return result
