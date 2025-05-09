# Copyright 2025 LiveKit, Inc.
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

import llama_api_client
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
    temperature: NotGivenOr[float]
    top_k: NotGivenOr[int]
    max_completion_tokens: NotGivenOr[int]


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "Llama-4-Maverick-17B-128E-Instruct-FP8",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: llama_api_client.AsyncLlamaAPIClient | None = None,
        top_k: NotGivenOr[int] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Llama LLM.

        ``api_key`` must be set to your MetaLlama API key, either using the argument or by setting
        the ``LLAMA_API_KEY`` environmental variable.

        model (str | ChatModels): The model to use. Defaults to "Llama-4-Maverick-17B-128E-Instruct-FP8".
        api_key (str, optional): The Meta Llama API key. Defaults to the LLAMA_API_KEY environment variable.
        base_url (str, optional): The base URL for the Llama API. Defaults to None.
        client (llama_api_client.AsyncLlamaAPIClient | None): The Llama client to use. Defaults to None.
        top_k (int, optional): The top K for the Llama API. Defaults to None.
        max_completion_tokens (int, optional): The max tokens for the Llama API. Defaults to None.
        temperature (float, optional): The temperature for the Llama API. Defaults to None.
        parallel_tool_calls (bool, optional): Whether to parallelize tool calls. Defaults to None.
        tool_choice (ToolChoice, optional): The tool choice for the Llama API. Defaults to "auto".
        """

        super().__init__()

        self._opts = _LLMOptions(
            model=model,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            top_k=top_k,
            max_completion_tokens=max_completion_tokens,
        )
        llama_api_key = (
            api_key if is_given(api_key) else os.environ.get("LLAMA_API_KEY")
        )
        if not llama_api_key:
            raise ValueError("Llama API key is required")

        self._client = client or llama_api_client.AsyncLlamaAPIClient(
            api_key=llama_api_key,
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
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        extra = {}

        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature

        if is_given(self._opts.top_k):
            extra["top_k"] = self._opts.top_k

        extra["max_completion_tokens"] = (
            self._opts.max_completion_tokens
            if is_given(self._opts.max_completion_tokens)
            else 1024
        )

        if tools:
            extra["tools"] = to_fnc_ctx(tools)
            tool_choice = (
                tool_choice if is_given(tool_choice) else self._opts.tool_choice
            )
            if is_given(tool_choice):
                llama_tool_choice: dict[str, Any] | None = {"type": "auto"}
                if (
                    isinstance(tool_choice, dict)
                    and tool_choice.get("type") == "function"
                ):
                    llama_tool_choice = {
                        "type": "tool",
                        "name": tool_choice["function"]["name"],
                    }
                elif isinstance(tool_choice, str):
                    if tool_choice == "required":
                        llama_tool_choice = {"type": "any"}
                    elif tool_choice == "none":
                        extra["tools"] = []
                        llama_tool_choice = None
                if llama_tool_choice is not None:
                    parallel_tool_calls = (
                        parallel_tool_calls
                        if is_given(parallel_tool_calls)
                        else self._opts.parallel_tool_calls
                    )
                    if is_given(parallel_tool_calls):
                        llama_tool_choice["disable_parallel_tool_use"] = (
                            not parallel_tool_calls
                        )
                    extra["tool_choice"] = llama_tool_choice

        llama_ctx = to_chat_ctx(chat_ctx)

        stream = self._client.chat.completions.create(
            messages=llama_ctx,
            model=self._opts.model,
            stream=True,
            **extra,
        )

        # return LLMStream(
        #     self,
        #     llama_stream=stream,
        #     chat_ctx=chat_ctx,
        #     tools=tools,
        #     conn_options=conn_options,
        # )
