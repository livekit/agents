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

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import urlparse

import httpx

import openai
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, llm
from livekit.agents.llm import ToolChoice, utils as llm_utils
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types import ReasoningEffort
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    completion_create_params,
)
from openai.types.chat.chat_completion_chunk import Choice

from .log import logger
from .models import (
    CerebrasChatModels,
    ChatModels,
    DeepSeekChatModels,
    OctoChatModels,
    PerplexityChatModels,
    TelnyxChatModels,
    TogetherChatModels,
    XAIChatModels,
    _supports_reasoning_effort,
)
from .utils import AsyncAzureADTokenProvider, to_fnc_ctx



Verbosity = Literal["low", "medium", "high"]
APIMode = Literal["chat", "response"]


@dataclass
class _LLMOptions:
    model: str | ChatModels
    user: NotGivenOr[str]
    safety_identifier: NotGivenOr[str]
    prompt_cache_key: NotGivenOr[str]
    temperature: NotGivenOr[float]
    top_p: NotGivenOr[float]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice]
    store: NotGivenOr[bool]
    metadata: NotGivenOr[dict[str, str]]
    max_completion_tokens: NotGivenOr[int]
    service_tier: NotGivenOr[str]
    reasoning_effort: NotGivenOr[ReasoningEffort]
    verbosity: NotGivenOr[Verbosity]
    api_mode: APIMode
    enable_caching: bool


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "gpt-4.1",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        store: NotGivenOr[bool] = NOT_GIVEN,
        metadata: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        service_tier: NotGivenOr[str] = NOT_GIVEN,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        verbosity: NotGivenOr[Verbosity] = NOT_GIVEN,
        api_mode: APIMode = "chat",
        enable_caching: bool = False,
        _provider_fmt: NotGivenOr[str] = NOT_GIVEN,
        _strict_tool_schema: bool = True,
    ) -> None:
        """
        Create a new instance of OpenAI LLM.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.

        Args:
            api_mode: API mode to use - "chat" for Chat API or "response" for Responses API (default: "chat")
            enable_caching: Enable context caching for Responses API mode (default: False)
        """
        super().__init__()

        if not is_given(reasoning_effort) and _supports_reasoning_effort(model):
            reasoning_effort = "minimal"

        self._opts = _LLMOptions(
            model=model,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            store=store,
            metadata=metadata,
            max_completion_tokens=max_completion_tokens,
            service_tier=service_tier,
            reasoning_effort=reasoning_effort,
            safety_identifier=safety_identifier,
            prompt_cache_key=prompt_cache_key,
            top_p=top_p,
            verbosity=verbosity,
            api_mode=api_mode,
            enable_caching=enable_caching,
        )
        self._provider_fmt = _provider_fmt or "openai"
        self._strict_tool_schema = _strict_tool_schema
        # For Response API session caching - stores the last response_id
        self._previous_response_id: str | None = None
        self._client = client or openai.AsyncClient(
            api_key=api_key if is_given(api_key) else None,
            base_url=base_url if is_given(base_url) else None,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=timeout
                if timeout
                else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

    @property
    def model(self) -> str:
        """Get the model name for this LLM instance."""
        return self._opts.model

    def set_previous_response_id(self, response_id: str) -> None:
        """Set the previous response ID for session caching (Responses API only)."""
        self._previous_response_id = response_id
        logger.debug(f"Updated previous_response_id to: {response_id}")

    def clear_previous_response_id(self) -> None:
        """Clear the previous response ID."""
        self._previous_response_id = None
        logger.debug("Cleared previous_response_id")

    async def warmup_system_prompt(self, system_prompt: str) -> str:
        """
        预缓存 system prompt，返回带缓存的 response_id。

        这个方法会发送一个只包含 system prompt 的请求到 Response API，
        启用缓存，并返回 response_id。后续的对话可以直接使用这个 ID，
        从而命中 system prompt 的缓存。

        Args:
            system_prompt: 要缓存的系统提示词

        Returns:
            response_id: 带缓存的 response ID

        Raises:
            ValueError: 如果不是 Response API 模式或未启用缓存
        """
        if self._opts.api_mode != "response":
            raise ValueError("warmup_system_prompt only works in 'response' API mode")

        if not self._opts.enable_caching:
            raise ValueError("warmup_system_prompt requires enable_caching=True")

        logger.info("Warming up system prompt cache...")

        # 构建请求参数
        request_params = {
            "model": self._opts.model,
            "input": [{"role": "system", "content": system_prompt}],
            "stream": False,  # 不需要流式输出
            "extra_body": {
                "caching": {"type": "enabled"},
                "thinking": {"type": "disabled"}
            }
        }

        # 添加其他配置参数
        if is_given(self._opts.temperature):
            request_params["temperature"] = self._opts.temperature

        if is_given(self._opts.max_completion_tokens):
            request_params["max_completion_tokens"] = self._opts.max_completion_tokens

        try:
            # 调用 Response API
            response = await self._client.responses.create(**request_params)

            # 提取 response_id
            response_id = response.id

            # 自动设置为 previous_response_id
            self.set_previous_response_id(response_id)

            logger.info(f"System prompt cached successfully with response_id: {response_id}")

            # 检查缓存命中情况
            if hasattr(response, 'usage'):
                usage = response.usage
                input_tokens = getattr(usage, 'input_tokens', 0)
                logger.info(f"System prompt tokens: {input_tokens}")

            return response_id

        except Exception as e:
            logger.error(f"Failed to warmup system prompt: {e}")
            raise

    async def warmup_conversation_history(self, system_prompt: str, messages: list[dict]) -> str:
        """
        预缓存完整的对话历史（system prompt + 历史消息），返回带缓存的 response_id。

        用于恢复会话时,将 system prompt 和历史对话一起缓存,
        这样后续对话既能命中 system prompt 缓存,也能命中历史对话缓存。

        Args:
            system_prompt: 系统提示词
            messages: 历史消息列表,格式: [{"role": "user", "content": "..."}, ...]

        Returns:
            response_id: 带缓存的 response ID

        Raises:
            ValueError: 如果不是 Response API 模式或未启用缓存
        """
        if self._opts.api_mode != "response":
            raise ValueError("warmup_conversation_history only works in 'response' API mode")

        if not self._opts.enable_caching:
            raise ValueError("warmup_conversation_history requires enable_caching=True")

        logger.info(f"Warming up conversation history cache with {len(messages)} messages...")

        # 构建完整的输入（system prompt + 历史消息）
        input_messages = [{"role": "system", "content": system_prompt}]
        input_messages.extend(messages)

        # 构建请求参数
        request_params = {
            "model": self._opts.model,
            "input": input_messages,
            "stream": False,  # 不需要流式输出
            "extra_body": {
                "caching": {"type": "enabled"},
                "thinking": {"type": "disabled"}
            }
        }

        # 添加其他配置参数
        if is_given(self._opts.temperature):
            request_params["temperature"] = self._opts.temperature

        if is_given(self._opts.max_completion_tokens):
            request_params["max_completion_tokens"] = self._opts.max_completion_tokens

        try:
            # 调用 Response API
            response = await self._client.responses.create(**request_params)

            # 提取 response_id
            response_id = response.id

            # 自动设置为 previous_response_id
            self.set_previous_response_id(response_id)

            logger.info(f"✅ Conversation history cached successfully with response_id: {response_id}")

            # 检查缓存命中情况
            if hasattr(response, 'usage'):
                usage = response.usage
                input_tokens = getattr(usage, 'input_tokens', 0)
                cached_tokens = getattr(getattr(usage, 'input_tokens_details', None), 'cached_tokens', 0)
                logger.info(f"Total input tokens: {input_tokens}, Cached tokens: {cached_tokens}")

            return response_id

        except Exception as e:
            logger.error(f"Failed to warmup conversation history: {e}")
            raise


    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[
            completion_create_params.ResponseFormat | type[llm_utils.ResponseFormatT]
        ] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        extra = {}
        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        if is_given(self._opts.metadata):
            extra["metadata"] = self._opts.metadata

        if is_given(self._opts.user):
            extra["user"] = self._opts.user

        if is_given(self._opts.max_completion_tokens):
            extra["max_completion_tokens"] = self._opts.max_completion_tokens

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature

        if is_given(self._opts.service_tier):
            extra["service_tier"] = self._opts.service_tier

        if is_given(self._opts.reasoning_effort):
            extra["reasoning_effort"] = self._opts.reasoning_effort

        if is_given(self._opts.safety_identifier):
            extra["safety_identifier"] = self._opts.safety_identifier

        if is_given(self._opts.prompt_cache_key):
            extra["prompt_cache_key"] = self._opts.prompt_cache_key

        if is_given(self._opts.top_p):
            extra["top_p"] = self._opts.top_p

        if is_given(self._opts.verbosity):
            extra["verbosity"] = self._opts.verbosity

        parallel_tool_calls = (
            parallel_tool_calls if is_given(parallel_tool_calls) else self._opts.parallel_tool_calls
        )
        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice  # type: ignore
        if is_given(tool_choice):
            oai_tool_choice: ChatCompletionToolChoiceOptionParam
            if isinstance(tool_choice, dict):
                oai_tool_choice = {
                    "type": "function",
                    "function": {"name": tool_choice["function"]["name"]},
                }
                extra["tool_choice"] = oai_tool_choice
            elif tool_choice in ("auto", "required", "none"):
                oai_tool_choice = tool_choice
                extra["tool_choice"] = oai_tool_choice

        if is_given(response_format):
            extra["response_format"] = llm_utils.to_openai_response_format(response_format)  # type: ignore

        return LLMStream(
            self,
            model=self._opts.model,
            provider_fmt=self._provider_fmt,
            strict_tool_schema=self._strict_tool_schema,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
            api_mode=self._opts.api_mode,
            enable_caching=self._opts.enable_caching,
            previous_response_id=self._previous_response_id,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        model: str | ChatModels,
        provider_fmt: str,
        strict_tool_schema: bool,
        client: openai.AsyncClient,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
        api_mode: APIMode = "chat",
        enable_caching: bool = False,
        previous_response_id: str | None = None,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._provider_fmt = provider_fmt
        self._strict_tool_schema = strict_tool_schema
        self._client = client
        self._llm = llm
        self._extra_kwargs = extra_kwargs
        self._api_mode = api_mode
        self._enable_caching = enable_caching
        self._previous_response_id = previous_response_id
        self._current_response_id: str | None = None

    async def _run(self) -> None:
        # current function call that we're waiting for full completion (args are streamed)
        # (defined inside the _run method to make sure the state is reset for each run/attempt)
        self._oai_stream: openai.AsyncStream[ChatCompletionChunk] | None = None
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None
        self._tool_index: int | None = None
        retryable = True

        try:
            if self._api_mode == "response":
                await self._run_response_api()
            else:
                await self._run_chat_api()

        except openai.APITimeoutError:
            raise APITimeoutError(retryable=retryable) from None
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
                retryable=retryable,
            ) from None
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    async def _run_chat_api(self) -> None:
        """Run using standard Chat Completions API"""
        retryable = True
        chat_ctx, _ = self._chat_ctx.to_provider_format(format=self._provider_fmt)
        fnc_ctx = (
            to_fnc_ctx(self._tools, strict=self._strict_tool_schema)
            if self._tools
            else openai.NOT_GIVEN
        )

        if not self._tools:
            # remove tool_choice from extra_kwargs if no tools are provided
            self._extra_kwargs.pop("tool_choice", None)

        self._oai_stream = stream = await self._client.chat.completions.create(
            messages=cast(list[ChatCompletionMessageParam], chat_ctx),
            tools=fnc_ctx,
            model=self._model,
            stream_options={"include_usage": True},
            stream=True,
            timeout=httpx.Timeout(self._conn_options.timeout),
            **self._extra_kwargs,
        )

        thinking = asyncio.Event()
        async with stream:
            async for chunk in stream:
                for choice in chunk.choices:
                    chat_chunk = self._parse_choice(chunk.id, choice, thinking)
                    if chat_chunk is not None:
                        retryable = False
                        self._event_ch.send_nowait(chat_chunk)

                if chunk.usage is not None:
                    retryable = False
                    tokens_details = chunk.usage.prompt_tokens_details
                    cached_tokens = tokens_details.cached_tokens if tokens_details else 0
                    chunk = llm.ChatChunk(
                        id=chunk.id,
                        usage=llm.CompletionUsage(
                            completion_tokens=chunk.usage.completion_tokens or 0,
                            prompt_tokens=chunk.usage.prompt_tokens or 0,
                            prompt_cached_tokens=cached_tokens or 0,
                            total_tokens=chunk.usage.total_tokens or 0,
                        ),
                    )
                    self._event_ch.send_nowait(chunk)

    async def _run_response_api(self) -> None:
        """Run using Responses API with caching support"""
        retryable = True

        # Build the input for Responses API
        chat_ctx_dict = self._chat_ctx.to_dict()

        # Response API 的核心优势：自动上下文管理
        # - 如果有 previous_response_id：只发送最新的用户输入
        # - 如果没有 previous_response_id：发送完整的上下文（首次调用或重置会话）
        input_messages = []

        if self._previous_response_id is not None:
            # 后续调用：只发送最新的消息
            # Response API 会自动从 previous_response_id 获取历史上下文
            items = chat_ctx_dict.get("items", [])
            if items:
                # 只取最后一条消息（通常是用户的新输入）
                last_item = items[-1]
                if last_item["type"] == "message":
                    role = last_item["role"]
                    content = last_item["content"]

                    if isinstance(content, str):
                        msg = {"role": role, "content": content}
                    elif isinstance(content, list) and len(content) == 1 and isinstance(content[0], str):
                        msg = {"role": role, "content": content[0]}
                    else:
                        msg = {"role": role, "content": str(content) if not isinstance(content, list) else content}

                    input_messages.append(msg)
                    logger.debug(f"Using session cache, sending only last message: {msg}")
        else:
            # 首次调用：发送完整的上下文
            for item in chat_ctx_dict.get("items", []):
                if item["type"] == "message":
                    role = item["role"]
                    content = item["content"]

                    if isinstance(content, str):
                        msg = {"role": role, "content": content}
                    elif isinstance(content, list):
                        if len(content) == 1 and isinstance(content[0], str):
                            msg = {"role": role, "content": content[0]}
                        else:
                            msg = {"role": role, "content": content}
                    else:
                        msg = {"role": role, "content": str(content)}

                    input_messages.append(msg)

            logger.debug(f"First call or cache reset, sending {len(input_messages)} messages")

        # Prepare request parameters - copy extra_kwargs but exclude unsupported params
        request_params = {
            "model": self._model,
            "input": input_messages,
            "stream": True,
        }

        # Handle response_format conversion for Response API
        # Response API uses 'text' parameter instead of 'response_format'
        response_format = self._extra_kwargs.get('response_format')
        has_json_schema = False
        if response_format:
            # Convert response_format to text parameter for Response API
            if hasattr(response_format, 'type'):
                format_type = response_format.type
            elif isinstance(response_format, dict) and 'type' in response_format:
                format_type = response_format['type']
            else:
                format_type = 'json_object'

            request_params['text'] = {
                'format': {
                    'type': format_type
                }
            }
            has_json_schema = True

        # Copy extra_kwargs but exclude parameters not supported by Response API
        unsupported_params = {'response_format', 'stream_options'}
        for key, value in self._extra_kwargs.items():
            if key not in unsupported_params:
                request_params[key] = value

        # Add caching configuration
        # Note: caching is not supported with JSON schema in Response API
        if self._enable_caching and not has_json_schema:
            if "extra_body" not in request_params:
                request_params["extra_body"] = {}
            request_params["extra_body"]["caching"] = {"type": "enabled"}
        elif self._enable_caching and has_json_schema:
            logger.warning(
                "Caching is not supported with JSON schema in Response API. "
                "Disabling caching for this request."
            )

        # Add previous_response_id for session caching
        if self._previous_response_id is not None:
            request_params["previous_response_id"] = self._previous_response_id
            logger.debug(f"Using previous_response_id: {self._previous_response_id}")

        # Add tools if provided
        if self._tools:
            fnc_ctx = to_fnc_ctx(self._tools, strict=self._strict_tool_schema)
            request_params["tools"] = fnc_ctx


        # Call Responses API
        try:
            self._oai_stream = stream = await self._client.responses.create(**request_params)
        except Exception as e:
            logger.error(
                f"Response API call failed",
                extra={
                    "error": str(e),
                }
            )
            raise

        thinking = asyncio.Event()
        async with stream:
            async for chunk in stream:
                # Response API 返回的是事件流（Event Stream）
                event_type = type(chunk).__name__

                # Extract response_id from ResponseCreatedEvent
                if event_type == 'ResponseCreatedEvent' and hasattr(chunk, 'response'):
                    response = chunk.response
                    if hasattr(response, 'id'):
                        self._current_response_id = response.id
                        # 重要：不在这里设置 previous_response_id
                        # 只有在 ResponseCompletedEvent 时才设置，避免使用被中断的无效 ID
                        logger.debug(f"Got response_id: {self._current_response_id}")

                # 处理文本增量事件
                elif event_type == 'ResponseTextDeltaEvent':
                    # ResponseTextDeltaEvent 包含增量文本
                    if hasattr(chunk, 'delta'):
                        text = chunk.delta
                        if text:
                            retryable = False
                            chat_chunk = llm.ChatChunk(
                                id=self._current_response_id or "",
                                delta=llm.ChoiceDelta(content=text, role="assistant"),
                            )
                            self._event_ch.send_nowait(chat_chunk)
                # 处理完成事件，获取 usage 信息
                elif event_type == 'ResponseCompletedEvent':
                    if hasattr(chunk, 'response'):
                        response = chunk.response

                        # 核心修复：只有在 Response 成功完成时才更新 previous_response_id
                        # 这样可以避免使用被中断的无效 response_id
                        if hasattr(response, 'id'):
                            if isinstance(self._llm, LLM):
                                self._llm.set_previous_response_id(response.id)
                            logger.debug(f"✅ Response completed, updated previous_response_id: {response.id}")

                        if hasattr(response, 'usage') and response.usage is not None:
                            retryable = False
                            usage = response.usage
                            tokens_details = getattr(usage, 'input_tokens_details', None)
                            cached_tokens = getattr(tokens_details, 'cached_tokens', 0) if tokens_details else 0

                            usage_chunk = llm.ChatChunk(
                                id=self._current_response_id or "",
                                usage=llm.CompletionUsage(
                                    completion_tokens=getattr(usage, 'output_tokens', 0),
                                    prompt_tokens=getattr(usage, 'input_tokens', 0),
                                    prompt_cached_tokens=cached_tokens or 0,
                                    total_tokens=getattr(usage, 'total_tokens', 0),
                                ),
                            )
                            self._event_ch.send_nowait(usage_chunk)
                # 处理工具调用事件（如果需要）
                # 可以根据需要添加其他事件类型的处理

    def _parse_choice(
        self, id: str, choice: Choice, thinking: asyncio.Event
    ) -> llm.ChatChunk | None:
        delta = choice.delta

        # https://github.com/livekit/agents/issues/688
        # the delta can be None when using Azure OpenAI (content filtering)
        if delta is None:
            return None

        if delta.tool_calls:
            for tool in delta.tool_calls:
                if not tool.function:
                    continue

                call_chunk = None
                if self._tool_call_id and tool.id and tool.index != self._tool_index:
                    call_chunk = llm.ChatChunk(
                        id=id,
                        delta=llm.ChoiceDelta(
                            role="assistant",
                            content=delta.content,
                            tool_calls=[
                                llm.FunctionToolCall(
                                    arguments=self._fnc_raw_arguments or "",
                                    name=self._fnc_name or "",
                                    call_id=self._tool_call_id or "",
                                )
                            ],
                        ),
                    )
                    self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None

                if tool.function.name:
                    self._tool_index = tool.index
                    self._tool_call_id = tool.id
                    self._fnc_name = tool.function.name
                    self._fnc_raw_arguments = tool.function.arguments or ""
                elif tool.function.arguments:
                    self._fnc_raw_arguments += tool.function.arguments  # type: ignore

                if call_chunk is not None:
                    return call_chunk

        if choice.finish_reason in ("tool_calls", "stop") and self._tool_call_id:
            call_chunk = llm.ChatChunk(
                id=id,
                delta=llm.ChoiceDelta(
                    role="assistant",
                    content=delta.content,
                    tool_calls=[
                        llm.FunctionToolCall(
                            arguments=self._fnc_raw_arguments or "",
                            name=self._fnc_name or "",
                            call_id=self._tool_call_id or "",
                        )
                    ],
                ),
            )
            self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None
            return call_chunk

        delta.content = llm_utils.strip_thinking_tokens(delta.content, thinking)

        if not delta.content:
            return None

        return llm.ChatChunk(
            id=id,
            delta=llm.ChoiceDelta(content=delta.content, role="assistant"),
        )
