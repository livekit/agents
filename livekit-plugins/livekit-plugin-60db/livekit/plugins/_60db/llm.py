from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
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

# request fields LLMStream builds from validated inputs; extra_kwargs must not
# be able to replace them
_RESERVED_BODY_FIELDS = frozenset({"model", "messages", "stream", "tools"})


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

        # never send the bearer key (or conversation content) to an arbitrary
        # plaintext destination — https/wss anywhere, http/ws only for localhost
        parsed_url = urlparse(self._api_url)
        if parsed_url.scheme not in ("http", "https", "ws", "wss"):
            raise ValueError(
                f"60db LLM: unsupported API URL scheme {parsed_url.scheme!r} in {self._api_url!r}"
            )
        if parsed_url.scheme in ("http", "ws") and parsed_url.hostname not in (
            "localhost",
            "127.0.0.1",
            "::1",
        ):
            raise ValueError(
                "60db LLM: plaintext API URL is only allowed for localhost; "
                "use an https:// URL to avoid exposing the API key and chat content"
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
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
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
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._llm_instance: LLM = llm
        self._parallel_tool_calls = parallel_tool_calls
        self._tool_choice = tool_choice
        self._extra_kwargs = extra_kwargs
        # chunks are emitted to the caller as they arrive; a retry after output
        # was sent would duplicate speech or tool execution, so let the
        # framework mark errors non-retryable once output has left the stream
        self._retry_on_chunk_sent = False

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
            "stream": True,
        }

        if tool_schemas:
            body["tools"] = tool_schemas

        if is_given(self._tool_choice):
            body["tool_choice"] = self._tool_choice

        if is_given(self._parallel_tool_calls):
            body["parallel_tool_calls"] = self._parallel_tool_calls

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

        # Merge any extra kwargs. Reserved fields built from validated inputs
        # can't be replaced, and explicitly given tool options win — but
        # extra_kwargs may still supply tool options the dedicated parameters
        # didn't set
        for key, value in self._extra_kwargs.items():
            if key in _RESERVED_BODY_FIELDS:
                logger.warning("60db LLM: ignoring reserved extra_kwarg %r", key)
                continue
            if key in body:
                continue
            body[key] = value

        # Tool call accumulation state
        tool_call_id: str | None = None
        tool_call_name: str | None = None
        tool_call_arguments: str | None = None

        try:
            async with self._llm_instance._client.stream(
                "POST",
                self._llm_instance._api_url,
                json=body,
                timeout=httpx.Timeout(self._conn_options.timeout, connect=15.0),
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
                        # don't log the payload — malformed model output can
                        # carry generated text or tool arguments
                        logger.warning(
                            "60db LLM: failed to parse SSE data (length=%d)", len(data_str)
                        )
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
            status_code = e.response.status_code
            body: object = None
            try:
                # the response is streamed, so the body must be read before use
                await e.response.aread()
                try:
                    body = e.response.json()
                except Exception:
                    body = e.response.text
            except Exception:
                pass  # body is optional context for the error
            # APIStatusError marks 4xx (except 408/429/499) as non-retryable, so
            # invalid requests/auth no longer burn through the retry budget
            raise APIStatusError(
                f"60db LLM: HTTP {status_code}",
                status_code=status_code,
                request_id=e.response.headers.get("x-request-id"),
                body=body,
            ) from e
        except Exception as e:
            raise APIConnectionError(f"60db LLM: connection error: {e}") from e
