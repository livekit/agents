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

"""OpenAI Responses API support for LiveKit Agents.

This module provides support for the OpenAI Responses API, which is a newer
stateful API that combines features from Chat Completions and Assistants APIs.

Azure OpenAI deployments that only support the Responses API endpoint
(e.g., `/openai/responses`) can use this module.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx

import openai
from livekit.agents import llm
from livekit.agents._exceptions import APIConnectionError, APIStatusError, APITimeoutError
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import (
    FunctionTool,
    RawFunctionTool,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .utils import AsyncAzureADTokenProvider

lk_oai_debug = int(os.getenv("LK_OPENAI_DEBUG", 0))


@dataclass
class _ResponsesLLMOptions:
    model: str
    temperature: NotGivenOr[float]
    max_output_tokens: NotGivenOr[int]
    top_p: NotGivenOr[float]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice]
    truncation: NotGivenOr[Literal["auto", "disabled"]]


class ResponsesLLM(llm.LLM):
    """LLM implementation using the OpenAI Responses API.

    The Responses API is a newer stateful API from OpenAI/Azure that provides
    enhanced capabilities over the traditional Chat Completions API.

    Use `ResponsesLLM.with_azure()` to create an instance for Azure OpenAI
    deployments that use the Responses API endpoint.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        truncation: NotGivenOr[Literal["auto", "disabled"]] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of OpenAI Responses API LLM.

        Args:
            model: The model to use (e.g., "gpt-4o", "gpt-5.1-codex-mini")
            api_key: OpenAI API key
            base_url: Base URL for the API
            client: Pre-configured OpenAI async client
            temperature: Sampling temperature (0-2)
            max_output_tokens: Maximum tokens in the response
            top_p: Top-p sampling parameter
            parallel_tool_calls: Allow parallel tool execution
            tool_choice: Tool selection strategy
            truncation: Context truncation strategy
            timeout: HTTP timeout settings
            max_retries: Maximum retry attempts
        """
        super().__init__()

        self._opts = _ResponsesLLMOptions(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            truncation=truncation,
        )

        self._client = client or openai.AsyncClient(
            api_key=api_key if is_given(api_key) else None,
            base_url=base_url if is_given(base_url) else None,
            max_retries=max_retries if is_given(max_retries) else 0,
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
        return self._opts.model

    @property
    def provider(self) -> str:
        return self._client._base_url.netloc.decode("utf-8")

    @staticmethod
    def with_azure(
        *,
        model: str = "gpt-4o",
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        truncation: NotGivenOr[Literal["auto", "disabled"]] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
    ) -> ResponsesLLM:
        """Create a ResponsesLLM instance configured for Azure OpenAI.

        This method creates an LLM that uses the Azure OpenAI Responses API endpoint.

        Args:
            model: The deployment model name
            azure_endpoint: Azure OpenAI endpoint URL
            azure_deployment: Azure deployment name
            api_version: API version (e.g., "2025-04-01-preview")
            api_key: Azure API key (or set AZURE_OPENAI_API_KEY env var)
            azure_ad_token: Azure AD token for authentication
            azure_ad_token_provider: Async token provider for Azure AD
            organization: OpenAI organization ID
            project: OpenAI project ID
            base_url: Override base URL
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            top_p: Top-p sampling
            parallel_tool_calls: Allow parallel tool calls
            tool_choice: Tool selection strategy
            truncation: Truncation strategy
            timeout: HTTP timeout

        Returns:
            Configured ResponsesLLM instance for Azure
        """
        azure_client = openai.AsyncAzureOpenAI(
            max_retries=0,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout
            if timeout
            else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
        )  # type: ignore

        return ResponsesLLM(
            model=model,
            client=azure_client,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            truncation=truncation,
        )

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> ResponsesLLMStream:
        extra = {}
        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature

        if is_given(self._opts.max_output_tokens):
            extra["max_output_tokens"] = self._opts.max_output_tokens

        if is_given(self._opts.top_p):
            extra["top_p"] = self._opts.top_p

        if is_given(self._opts.truncation):
            extra["truncation"] = self._opts.truncation

        parallel_tool_calls = (
            parallel_tool_calls if is_given(parallel_tool_calls) else self._opts.parallel_tool_calls
        )
        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        tool_choice_val = tool_choice if is_given(tool_choice) else self._opts.tool_choice
        if is_given(tool_choice_val):
            if isinstance(tool_choice_val, dict):
                extra["tool_choice"] = {
                    "type": "function",
                    "name": tool_choice_val["function"]["name"],
                }
            elif tool_choice_val in ("auto", "required", "none"):
                extra["tool_choice"] = tool_choice_val

        return ResponsesLLMStream(
            self,
            model=self._opts.model,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )


class ResponsesLLMStream(llm.LLMStream):
    """Stream implementation for the OpenAI Responses API."""

    def __init__(
        self,
        llm_instance: ResponsesLLM,
        *,
        model: str,
        client: openai.AsyncClient,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm_instance, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._client = client
        self._extra_kwargs = extra_kwargs
        # Track function call state during streaming
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str = ""

    async def _run(self) -> None:
        retryable = True

        try:
            # Convert chat context to Responses API input format
            input_messages = _chat_ctx_to_responses_input(self._chat_ctx)

            # Build tools for Responses API
            tools_param = _build_responses_tools(self._tools) if self._tools else None

            if lk_oai_debug:
                from livekit.agents.log import logger

                logger.debug(
                    "responses.create",
                    extra={
                        "input": input_messages,
                        "tools": tools_param,
                        "model": self._model,
                    },
                )

            # Remove tool_choice if no tools provided
            if not self._tools:
                self._extra_kwargs.pop("tool_choice", None)

            # Call the Responses API with streaming
            stream = await self._client.responses.create(
                model=self._model,
                input=cast(Any, input_messages),
                tools=cast(Any, tools_param) if tools_param else openai.NOT_GIVEN,
                stream=True,
                timeout=httpx.Timeout(self._conn_options.timeout),
                **self._extra_kwargs,
            )

            response_id = ""
            async for event in stream:
                chunk = self._parse_stream_event(event, response_id)
                if chunk is not None:
                    retryable = False
                    response_id = chunk.id
                    self._event_ch.send_nowait(chunk)

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

    def _parse_stream_event(self, event: Any, response_id: str) -> llm.ChatChunk | None:
        """Parse a Responses API stream event into a ChatChunk."""
        event_type = getattr(event, "type", None)

        # Get response ID from various event types
        if hasattr(event, "response") and hasattr(event.response, "id"):
            response_id = event.response.id
        elif hasattr(event, "id"):
            response_id = event.id

        if not response_id:
            response_id = "responses_stream"

        # Handle text delta events
        if event_type == "response.output_text.delta":
            delta_text = getattr(event, "delta", "")
            if delta_text:
                return llm.ChatChunk(
                    id=response_id,
                    delta=llm.ChoiceDelta(role="assistant", content=delta_text),
                )

        # Handle text done events (for final content)
        elif event_type == "response.output_text.done":
            # Text is complete, no need to emit anything extra
            pass

        # Handle function call arguments delta
        elif event_type == "response.function_call_arguments.delta":
            delta_args = getattr(event, "delta", "")
            if delta_args:
                self._fnc_raw_arguments += delta_args

        # Handle function call arguments done
        elif event_type == "response.function_call_arguments.done":
            # Arguments complete, function call will be emitted on output_item.done
            pass

        # Handle output item added (start of a new output item like function call)
        elif event_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if item:
                item_type = getattr(item, "type", None)
                if item_type == "function_call":
                    self._tool_call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                    self._fnc_name = getattr(item, "name", None)
                    self._fnc_raw_arguments = ""

        # Handle output item done (completion of function call or message)
        elif event_type == "response.output_item.done":
            item = getattr(event, "item", None)
            if item:
                item_type = getattr(item, "type", None)
                if item_type == "function_call":
                    call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                    name = getattr(item, "name", None)
                    arguments = getattr(item, "arguments", None) or self._fnc_raw_arguments

                    if call_id and name:
                        chunk = llm.ChatChunk(
                            id=response_id,
                            delta=llm.ChoiceDelta(
                                role="assistant",
                                tool_calls=[
                                    llm.FunctionToolCall(
                                        arguments=arguments,
                                        name=name,
                                        call_id=call_id,
                                    )
                                ],
                            ),
                        )
                        # Reset state
                        self._tool_call_id = None
                        self._fnc_name = None
                        self._fnc_raw_arguments = ""
                        return chunk

        # Handle response completed event (contains usage info)
        elif event_type == "response.completed":
            response = getattr(event, "response", None)
            if response:
                usage = getattr(response, "usage", None)
                if usage:
                    # Get cached tokens from input_tokens_details object
                    cached_tokens = 0
                    input_details = getattr(usage, "input_tokens_details", None)
                    if input_details:
                        cached_tokens = getattr(input_details, "cached_tokens", 0)

                    return llm.ChatChunk(
                        id=response_id,
                        usage=llm.CompletionUsage(
                            completion_tokens=getattr(usage, "output_tokens", 0),
                            prompt_tokens=getattr(usage, "input_tokens", 0),
                            prompt_cached_tokens=cached_tokens,
                            total_tokens=getattr(usage, "total_tokens", 0),
                        ),
                    )

        # Handle content part text delta (alternative text streaming format)
        elif event_type == "response.content_part.delta":
            delta = getattr(event, "delta", None)
            if delta:
                text = getattr(delta, "text", None)
                if text:
                    return llm.ChatChunk(
                        id=response_id,
                        delta=llm.ChoiceDelta(role="assistant", content=text),
                    )

        # Handle text delta events (another format)
        elif event_type == "response.text.delta":
            delta_text = getattr(event, "delta", "")
            if delta_text:
                return llm.ChatChunk(
                    id=response_id,
                    delta=llm.ChoiceDelta(role="assistant", content=delta_text),
                )

        return None


def _chat_ctx_to_responses_input(chat_ctx: ChatContext) -> list[dict[str, Any]]:
    """Convert a ChatContext to Responses API input format.

    The Responses API expects an array of input items with a specific structure.
    """
    messages: list[dict[str, Any]] = []

    for item in chat_ctx.items:
        if item.type == "message":
            msg = _message_to_responses_input(item)
            if msg:
                messages.append(msg)
        elif item.type == "function_call":
            # Function calls are part of assistant output, handled differently in Responses API
            messages.append(
                {
                    "type": "function_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                }
            )
        elif item.type == "function_call_output":
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": item.output,
                }
            )

    return messages


def _message_to_responses_input(msg: llm.ChatMessage) -> dict[str, Any] | None:
    """Convert a ChatMessage to Responses API input format."""
    role = msg.role

    # Map roles appropriately
    if role == "developer":
        role = "developer"
    elif role == "system":
        role = "system"
    elif role == "assistant":
        role = "assistant"
    else:
        role = "user"

    content_parts: list[dict[str, Any]] = []
    text_content = ""

    for content in msg.content:
        if isinstance(content, str):
            if text_content:
                text_content += "\n"
            text_content += content
        elif isinstance(content, llm.ImageContent):
            content_parts.append(_image_to_responses_input(content))

    # Build the message
    if not content_parts:
        # Simple text message
        return {
            "role": role,
            "content": text_content,
        }
    else:
        # Multi-part content
        if text_content:
            content_parts.insert(0, {"type": "input_text", "text": text_content})
        return {
            "role": role,
            "content": content_parts,
        }


def _image_to_responses_input(image: llm.ImageContent) -> dict[str, Any]:
    """Convert an ImageContent to Responses API format."""
    img = llm.utils.serialize_image(image)

    if img.external_url:
        return {
            "type": "input_image",
            "image_url": img.external_url,
            "detail": img.inference_detail,
        }

    assert img.data_bytes is not None
    b64_data = base64.b64encode(img.data_bytes).decode("utf-8")
    return {
        "type": "input_image",
        "image_url": f"data:{img.mime_type};base64,{b64_data}",
        "detail": img.inference_detail,
    }


def _build_responses_tools(
    tools: list[FunctionTool | RawFunctionTool],
) -> list[dict[str, Any]]:
    """Build tools in Responses API format."""
    responses_tools: list[dict[str, Any]] = []

    for tool in tools:
        if is_raw_function_tool(tool):
            info = get_raw_function_info(tool)
            responses_tools.append(
                {
                    "type": "function",
                    "name": info.raw_schema.get("name", ""),
                    "description": info.raw_schema.get("description", ""),
                    "parameters": info.raw_schema.get("parameters", {}),
                }
            )
        elif is_function_tool(tool):
            schema = llm.utils.build_strict_openai_schema(tool)
            # Convert from chat completions format to responses format
            func_def = schema.get("function", {})
            responses_tools.append(
                {
                    "type": "function",
                    "name": func_def.get("name", ""),
                    "description": func_def.get("description", ""),
                    "parameters": func_def.get("parameters", {}),
                    "strict": func_def.get("strict", True),
                }
            )

    return responses_tools
