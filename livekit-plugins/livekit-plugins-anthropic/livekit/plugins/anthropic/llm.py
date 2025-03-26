# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Literal

import httpx

import anthropic
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, llm
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import FunctionTool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .models import ChatModels
from .utils import to_chat_ctx, to_fnc_ctx


@dataclass
class _LLMOptions:
    model: str | ChatModels
    user: NotGivenOr[str]
    temperature: NotGivenOr[float]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice | Literal["auto", "required", "none"]]
    caching: NotGivenOr[Literal["ephemeral"]]
    top_k: NotGivenOr[int]
    max_tokens: NotGivenOr[int]
    """If set to "ephemeral", the system prompt, tools, and chat history will be cached."""


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "claude-3-5-sonnet-20241022",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        user: NotGivenOr[str] = NOT_GIVEN,
        client: anthropic.AsyncClient | None = None,
        top_k: NotGivenOr[int] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice | Literal["auto", "required", "none"]] = NOT_GIVEN,
        caching: NotGivenOr[Literal["ephemeral"]] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Anthropic LLM.

        ``api_key`` must be set to your Anthropic API key, either using the argument or by setting
        the ``ANTHROPIC_API_KEY`` environmental variable.

        model (str | ChatModels): The model to use. Defaults to "claude-3-5-sonnet-20241022".
        api_key (str | None): The Anthropic API key. Defaults to the ANTHROPIC_API_KEY environment variable.
        base_url (str | None): The base URL for the Anthropic API. Defaults to None.
        user (str | None): The user for the Anthropic API. Defaults to None.
        client (anthropic.AsyncClient | None): The Anthropic client to use. Defaults to None.
        temperature (float | None): The temperature for the Anthropic API. Defaults to None.
        parallel_tool_calls (bool | None): Whether to parallelize tool calls. Defaults to None.
        tool_choice (Union[ToolChoice, Literal["auto", "required", "none"]] | None): The tool choice for the Anthropic API. Defaults to "auto".
        caching (Literal["ephemeral"] | None): If set to "ephemeral", caching will be enabled for the system prompt, tools, and chat history.
        """  # noqa: E501

        super().__init__()

        self._opts = _LLMOptions(
            model=model,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            caching=caching,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        anthropic_api_key = api_key if is_given(api_key) else os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("Anthropic API key is required")

        self._client = anthropic.AsyncClient(
            api_key=anthropic_api_key,
            base_url=base_url if is_given(base_url) else None,
            http_client=httpx.AsyncClient(
                timeout=5.0,
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                    keepalive_expiry=120,
                ),
            ),
        )

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice | Literal["auto", "required", "none"]] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        extra = {}

        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        if is_given(self._opts.user):
            extra["user"] = self._opts.user

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature

        if is_given(self._opts.top_k):
            extra["top_k"] = self._opts.top_k

        extra["max_tokens"] = self._opts.max_tokens if is_given(self._opts.max_tokens) else 1024

        if tools:
            extra["tools"] = to_fnc_ctx(tools, self._opts.caching)
            tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice
            if is_given(tool_choice):
                anthropic_tool_choice: dict[str, Any] | None = {"type": "auto"}
                if isinstance(tool_choice, ToolChoice):
                    if tool_choice.type == "function":
                        anthropic_tool_choice = {
                            "type": "tool",
                            "name": tool_choice.name,
                        }
                elif isinstance(tool_choice, str):
                    if tool_choice == "required":
                        anthropic_tool_choice = {"type": "any"}
                    elif tool_choice == "none":
                        extra["tools"] = []
                        anthropic_tool_choice = None
                if anthropic_tool_choice is not None:
                    parallel_tool_calls = (
                        parallel_tool_calls
                        if is_given(parallel_tool_calls)
                        else self._opts.parallel_tool_calls
                    )
                    if is_given(parallel_tool_calls):
                        anthropic_tool_choice["disable_parallel_tool_use"] = not parallel_tool_calls
                    extra["tool_choice"] = anthropic_tool_choice

        anthropic_ctx, system_message = to_chat_ctx(chat_ctx, id(self), caching=self._opts.caching)

        if system_message:
            extra["system"] = [system_message]

        stream = self._client.messages.create(
            messages=anthropic_ctx,
            model=self._opts.model,
            stream=True,
            **extra,
        )

        return LLMStream(
            self,
            anthropic_stream=stream,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        anthropic_stream: Awaitable[anthropic.AsyncStream[anthropic.types.RawMessageStreamEvent]],
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool] | None,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._awaitable_anthropic_stream = anthropic_stream
        self._anthropic_stream: (
            anthropic.AsyncStream[anthropic.types.RawMessageStreamEvent] | None
        ) = None

        # current function call that we're waiting for full completion (args are streamed)
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None

        self._request_id: str = ""
        self._ignoring_cot = False  # ignore chain of thought
        self._input_tokens = 0
        self._cache_creation_tokens = 0
        self._cache_read_tokens = 0
        self._output_tokens = 0

    async def _run(self) -> None:
        retryable = True
        try:
            if not self._anthropic_stream:
                self._anthropic_stream = await self._awaitable_anthropic_stream

            async with self._anthropic_stream as stream:
                async for event in stream:
                    chat_chunk = self._parse_event(event)
                    if chat_chunk is not None:
                        self._event_ch.send_nowait(chat_chunk)
                        retryable = False

                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=self._request_id,
                        usage=llm.CompletionUsage(
                            completion_tokens=self._output_tokens,
                            prompt_tokens=self._input_tokens,
                            total_tokens=self._input_tokens
                            + self._output_tokens
                            + self._cache_creation_tokens
                            + self._cache_read_tokens,
                            cache_creation_input_tokens=self._cache_creation_tokens,
                            cache_read_input_tokens=self._cache_read_tokens,
                        ),
                    )
                )
        except anthropic.APITimeoutError as e:
            raise APITimeoutError(retryable=retryable) from e
        except anthropic.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
            ) from e
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    def _parse_event(self, event: anthropic.types.RawMessageStreamEvent) -> llm.ChatChunk | None:
        if event.type == "message_start":
            self._request_id = event.message.id
            self._input_tokens = event.message.usage.input_tokens
            self._output_tokens = event.message.usage.output_tokens
            if event.message.usage.cache_creation_input_tokens:
                self._cache_creation_tokens = event.message.usage.cache_creation_input_tokens
            if event.message.usage.cache_read_input_tokens:
                self._cache_read_tokens = event.message.usage.cache_read_input_tokens
        elif event.type == "message_delta":
            self._output_tokens += event.usage.output_tokens
        elif event.type == "content_block_start":
            if event.content_block.type == "tool_use":
                self._tool_call_id = event.content_block.id
                self._fnc_name = event.content_block.name
                self._fnc_raw_arguments = ""
        elif event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                text = delta.text

                if self._tools is not None:
                    # anthropic may inject COC when using functions
                    if text.startswith("<thinking>"):
                        self._ignoring_cot = True
                    elif self._ignoring_cot and "</thinking>" in text:
                        text = text.split("</thinking>")[-1]
                        self._ignoring_cot = False

                if self._ignoring_cot:
                    return None

                return llm.ChatChunk(
                    id=self._request_id,
                    delta=llm.ChoiceDelta(content=text, role="assistant"),
                )
            elif delta.type == "input_json_delta":
                assert self._fnc_raw_arguments is not None
                self._fnc_raw_arguments += delta.partial_json

        elif event.type == "content_block_stop":
            if self._tool_call_id is not None:
                assert self._fnc_name is not None
                assert self._fnc_raw_arguments is not None

                chat_chunk = llm.ChatChunk(
                    id=self._request_id,
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            llm.FunctionToolCall(
                                arguments=self._fnc_raw_arguments or "",
                                name=self._fnc_name or "",
                                call_id=self._tool_call_id or "",
                            )
                        ],
                    ),
                )
                self._tool_call_id = self._fnc_raw_arguments = self._fnc_name = None
                return chat_chunk

        return None
