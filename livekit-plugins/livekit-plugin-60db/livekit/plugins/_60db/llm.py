from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    llm,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .client import _get_default_api_key
from .log import logger

# Load .env.local from current directory
load_dotenv(Path(".env.local"))

_DEFAULT_API_URL = "https://api.60db.ai/v1/chat/completions"


class LLM(llm.LLM):
    """60db.ai HTTP-based LLM provider for LiveKit Agents."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        ws_url: str | None = None,
        model: str = "qcall/slm-3b-int4",
        top_k: int | None = None,
        chat_template_kwargs: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()

        self._api_key = api_key or _get_default_api_key() or os.getenv("SIXTY_DB_API_KEY", "")
        self._api_url = ws_url or os.getenv("SIXTY_DB_LLM_URL", "") or _DEFAULT_API_URL
        self._model = model
        self._top_k = top_k
        self._chat_template_kwargs = chat_template_kwargs
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens

        if not self._api_key:
            raise ValueError(
                "60db API key is required. Set SIXTY_DB_API_KEY env var or pass api_key argument."
            )

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        logger.info("60db LLM: initialized with model=%s, api_url=%s", self._model, self._api_url)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "60db"

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
        return LLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra_kwargs if is_given(extra_kwargs) else {},
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class LLMStream(llm.LLMStream):
    """SSE-based streaming LLM implementation for 60db.ai."""

    def __init__(
        self,
        llm: LLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._llm_instance: LLM = llm
        self._extra_kwargs = extra_kwargs

    async def _run(self) -> None:
        # Convert chat context to OpenAI format
        messages, _ = self._chat_ctx.to_provider_format("openai")

        # Parse tools
        tool_ctx = llm.ToolContext(self._tools)
        tool_schemas = tool_ctx.parse_function_tools("openai")

        # Build request body
        body: dict[str, Any] = {
            "model": self._llm_instance._model,
            "messages": messages,
        }

        if tool_schemas:
            body["tools"] = tool_schemas

        if self._llm_instance._top_k is not None:
            body["top_k"] = self._llm_instance._top_k

        if self._llm_instance._chat_template_kwargs is not None:
            body["chat_template_kwargs"] = self._llm_instance._chat_template_kwargs

        if self._llm_instance._temperature is not None:
            body["temperature"] = self._llm_instance._temperature

        if self._llm_instance._top_p is not None:
            body["top_p"] = self._llm_instance._top_p

        if self._llm_instance._max_tokens is not None:
            body["max_tokens"] = self._llm_instance._max_tokens

        # Merge any extra kwargs
        body.update(self._extra_kwargs)

        # Tool call accumulation state
        tool_call_id: str | None = None
        tool_call_name: str | None = None
        tool_call_arguments: str | None = None

        try:
            async with self._llm_instance._client.stream(
                "POST",
                self._llm_instance._api_url,
                json=body,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue

                    if not line.startswith("data: "):
                        continue

                    data_str = line[len("data: ") :]

                    if data_str == "[DONE]":
                        # Flush any pending tool call
                        if tool_call_id is not None:
                            self._event_ch.send_nowait(
                                llm.ChatChunk(
                                    id="",
                                    delta=llm.ChoiceDelta(
                                        role="assistant",
                                        content="",
                                        tool_calls=[
                                            llm.FunctionToolCall(
                                                name=tool_call_name or "",
                                                arguments=tool_call_arguments or "",
                                                call_id=tool_call_id,
                                            )
                                        ],
                                    ),
                                )
                            )
                            tool_call_id = None
                            tool_call_name = None
                            tool_call_arguments = None
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("60db LLM: failed to parse SSE data: %s", data_str)
                        continue

                    # Skip non-choice messages (e.g. chat_id, done)
                    if "choices" not in data:
                        continue

                    chunk_id = data.get("id", "")

                    for choice in data["choices"]:
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason")

                        # Handle tool calls
                        tool_calls = delta.get("tool_calls")
                        if tool_calls:
                            for tc in tool_calls:
                                fn = tc.get("function", {})
                                tc_id = tc.get("id")

                                # New tool call — flush previous if any
                                if tc_id and tool_call_id and tc_id != tool_call_id:
                                    self._event_ch.send_nowait(
                                        llm.ChatChunk(
                                            id=chunk_id,
                                            delta=llm.ChoiceDelta(
                                                role="assistant",
                                                content="",
                                                tool_calls=[
                                                    llm.FunctionToolCall(
                                                        name=tool_call_name or "",
                                                        arguments=tool_call_arguments or "",
                                                        call_id=tool_call_id,
                                                    )
                                                ],
                                            ),
                                        )
                                    )

                                if fn.get("name"):
                                    # Start of a new tool call
                                    tool_call_id = tc_id or tool_call_id or ""
                                    tool_call_name = fn["name"]
                                    tool_call_arguments = fn.get("arguments", "") or ""
                                elif fn.get("arguments"):
                                    # Continuation of arguments
                                    if tool_call_arguments is None:
                                        tool_call_arguments = ""
                                    tool_call_arguments += fn["arguments"]

                        # On finish_reason, flush pending tool call
                        if finish_reason and tool_call_id is not None:
                            self._event_ch.send_nowait(
                                llm.ChatChunk(
                                    id=chunk_id,
                                    delta=llm.ChoiceDelta(
                                        role="assistant",
                                        content=delta.get("content") or "",
                                        tool_calls=[
                                            llm.FunctionToolCall(
                                                name=tool_call_name or "",
                                                arguments=tool_call_arguments or "",
                                                call_id=tool_call_id,
                                            )
                                        ],
                                    ),
                                )
                            )
                            tool_call_id = None
                            tool_call_name = None
                            tool_call_arguments = None
                            continue

                        # Emit text content
                        content = delta.get("content")
                        if content and tool_call_id is None:
                            self._event_ch.send_nowait(
                                llm.ChatChunk(
                                    id=chunk_id,
                                    delta=llm.ChoiceDelta(
                                        role="assistant",
                                        content=content,
                                    ),
                                )
                            )

                    # Handle usage data from the chunk level
                    usage = data.get("usage")
                    if usage:
                        self._event_ch.send_nowait(
                            llm.ChatChunk(
                                id=data.get("id", ""),
                                usage=llm.CompletionUsage(
                                    completion_tokens=usage.get("completion_tokens", 0),
                                    prompt_tokens=usage.get("prompt_tokens", 0),
                                    total_tokens=usage.get("total_tokens", 0),
                                ),
                            )
                        )

        except httpx.TimeoutException as e:
            raise APITimeoutError() from e
        except httpx.HTTPStatusError as e:
            raise APIConnectionError(
                f"60db LLM: HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except Exception as e:
            raise APIConnectionError(f"60db LLM: connection error: {e}") from e
