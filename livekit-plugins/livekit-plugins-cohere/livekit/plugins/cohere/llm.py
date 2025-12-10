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
from dataclasses import dataclass
from typing import Any, cast

import httpx

import cohere
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, llm
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .models import ChatModels
from .utils import to_fnc_ctx


@dataclass
class _LLMOptions:
    model: str | ChatModels
    temperature: NotGivenOr[float]
    max_tokens: NotGivenOr[int]
    top_p: NotGivenOr[float]
    top_k: NotGivenOr[int]
    frequency_penalty: NotGivenOr[float]
    presence_penalty: NotGivenOr[float]
    tool_choice: NotGivenOr[ToolChoice]


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "command-r-plus-08-2024",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: cohere.AsyncClient | None = None,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        top_k: NotGivenOr[int] = NOT_GIVEN,
        frequency_penalty: NotGivenOr[float] = NOT_GIVEN,
        presence_penalty: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Cohere LLM.

        ``api_key`` must be set to your Cohere API key, either using the argument or by setting
        the ``COHERE_API_KEY`` environmental variable.

        Args:
            model (str | ChatModels): The model to use. Defaults to "command-r-plus-08-2024".
            api_key (str, optional): The Cohere API key. Defaults to the COHERE_API_KEY environment variable.
            base_url (str, optional): The base URL for the Cohere API. Defaults to None.
            client (cohere.AsyncClient | None): The Cohere client to use. Defaults to None.
            temperature (float, optional): The temperature for the Cohere API. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to None.
            top_p (float, optional): The top-p value for nucleus sampling. Defaults to None.
            top_k (int, optional): The top-k value for top-k sampling. Defaults to None.
            frequency_penalty (float, optional): The frequency penalty. Defaults to None.
            presence_penalty (float, optional): The presence penalty. Defaults to None.
            tool_choice (ToolChoice, optional): The tool choice for the Cohere API. Defaults to "auto".
        """

        super().__init__()

        self._opts = _LLMOptions(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tool_choice=tool_choice,
        )

        cohere_api_key = api_key if is_given(api_key) else os.environ.get("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("Cohere API key is required")

        self._client = client or cohere.AsyncClient(
            api_key=cohere_api_key,
            base_url=base_url if is_given(base_url) else None,
            httpx_client=httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, read=60.0),  # Longer timeout for streaming
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                    keepalive_expiry=120,
                ),
            ),
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Cohere"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
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

        if is_given(self._opts.max_tokens):
            extra["max_tokens"] = self._opts.max_tokens

        if is_given(self._opts.top_p):
            extra["p"] = self._opts.top_p

        if is_given(self._opts.top_k):
            extra["k"] = self._opts.top_k

        if is_given(self._opts.frequency_penalty):
            extra["frequency_penalty"] = self._opts.frequency_penalty

        if is_given(self._opts.presence_penalty):
            extra["presence_penalty"] = self._opts.presence_penalty

        if tools:
            extra["tools"] = to_fnc_ctx(tools)
            tool_choice = (
                cast(ToolChoice, tool_choice) if is_given(tool_choice) else self._opts.tool_choice
            )
            if is_given(tool_choice):
                if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                    # Cohere doesn't support specific function selection in the same way
                    # We'll use "auto" and let the model decide
                    pass
                elif isinstance(tool_choice, str):
                    if tool_choice == "required":
                        # Cohere will automatically use tools when available
                        pass
                    elif tool_choice == "none":
                        extra["tools"] = []

        openai_ctx, extra_data = chat_ctx.to_provider_format(format="openai")
        messages = openai_ctx

        # Handle system message - Cohere uses 'preamble' instead of system messages
        system_message = None
        if extra_data is None:
            # Extract system messages from the messages list
            filtered_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    if system_message is None:
                        system_message = msg["content"]
                    else:
                        system_message += "\n" + msg["content"]
                else:
                    filtered_messages.append(msg)
            messages = filtered_messages

        # Convert messages to Cohere format
        cohere_messages: list[dict[str, str]] = []
        current_message = ""

        for msg in messages:
            if msg.get("role") == "user":
                current_message = msg.get("content", "")
                if len(cohere_messages) > 0:
                    cohere_messages.append({"role": "USER", "message": current_message})
            elif msg.get("role") == "assistant":
                assistant_content = msg.get("content", "")
                if assistant_content:  # Only add non-empty assistant messages
                    cohere_messages.append({"role": "CHATBOT", "message": assistant_content})

        # Ensure we have a valid message to send
        if not current_message:
            current_message = "Hello"  # Fallback message

        # Remove the last user message from history since it goes in the message parameter
        chat_history: list[dict[str, str]] = []
        if cohere_messages:
            # Keep all but the last user message in history
            for i, msg in enumerate(cohere_messages):
                if i < len(cohere_messages) - 1 or msg["role"] != "USER":
                    chat_history.append(msg)

        # Convert to proper Cohere message format
        cohere_chat_history: list[cohere.UserMessage | cohere.ChatbotMessage] = []
        for msg in chat_history:
            if msg["role"] == "USER":
                cohere_chat_history.append(cohere.UserMessage(message=msg["message"]))
            elif msg["role"] == "CHATBOT":
                cohere_chat_history.append(cohere.ChatbotMessage(message=msg["message"]))

        stream = self._client.chat_stream(
            model=self._opts.model,
            chat_history=cohere_chat_history,
            message=current_message,
            preamble=system_message,
            **extra,
        )

        return LLMStream(
            self,
            cohere_stream=stream,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        cohere_stream: Any,  # Cohere streaming response
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._cohere_stream = cohere_stream

        # current function call that we're waiting for full completion
        self._tool_calls: list[dict] = []
        self._current_tool_call: dict | None = None

        self._request_id: str = ""
        self._input_tokens = 0
        self._output_tokens = 0

    async def _run(self) -> None:
        retryable = True
        try:
            async for event in self._cohere_stream:
                chat_chunk = self._parse_event(event)
                if chat_chunk is not None:
                    self._event_ch.send_nowait(chat_chunk)
                    retryable = False

            # Send final usage information
            if self._input_tokens > 0 or self._output_tokens > 0:
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=self._request_id,
                        usage=llm.CompletionUsage(
                            completion_tokens=self._output_tokens,
                            prompt_tokens=self._input_tokens,
                            total_tokens=self._input_tokens + self._output_tokens,
                        ),
                    )
                )
        except cohere.TooManyRequestsError as e:
            raise APIStatusError(
                str(e),
                status_code=429,
                request_id=getattr(e, "request_id", None),
                body=getattr(e, "body", None),
            ) from e
        except cohere.BadRequestError as e:
            raise APIStatusError(
                str(e),
                status_code=400,
                request_id=getattr(e, "request_id", None),
                body=getattr(e, "body", None),
            ) from e
        except cohere.UnauthorizedError as e:
            raise APIStatusError(
                str(e),
                status_code=401,
                request_id=getattr(e, "request_id", None),
                body=getattr(e, "body", None),
            ) from e
        except cohere.ForbiddenError as e:
            raise APIStatusError(
                str(e),
                status_code=403,
                request_id=getattr(e, "request_id", None),
                body=getattr(e, "body", None),
            ) from e
        except cohere.NotFoundError as e:
            raise APIStatusError(
                str(e),
                status_code=404,
                request_id=getattr(e, "request_id", None),
                body=getattr(e, "body", None),
            ) from e
        except cohere.UnprocessableEntityError as e:
            raise APIStatusError(
                str(e),
                status_code=422,
                request_id=getattr(e, "request_id", None),
                body=getattr(e, "body", None),
            ) from e
        except cohere.InternalServerError as e:
            raise APIStatusError(
                str(e),
                status_code=500,
                request_id=getattr(e, "request_id", None),
                body=getattr(e, "body", None),
            ) from e
        except httpx.TimeoutException as e:
            raise APITimeoutError(retryable=retryable) from e
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    def _parse_event(self, event: Any) -> llm.ChatChunk | None:
        # Handle Cohere streaming events
        if hasattr(event, "event_type"):
            if event.event_type == "stream-start":
                if hasattr(event, "generation_id"):
                    self._request_id = event.generation_id
                return None

            elif event.event_type == "text-generation":
                if hasattr(event, "text"):
                    return llm.ChatChunk(
                        id=self._request_id,
                        delta=llm.ChoiceDelta(content=event.text, role="assistant"),
                    )

            elif event.event_type == "tool-calls-generation":
                if hasattr(event, "tool_calls"):
                    tool_calls: list[llm.FunctionToolCall] = []
                    for tool_call in event.tool_calls:
                        tool_calls.append(
                            llm.FunctionToolCall(
                                arguments=str(tool_call.parameters)
                                if hasattr(tool_call, "parameters")
                                else "",
                                name=tool_call.name if hasattr(tool_call, "name") else "",
                                call_id=getattr(tool_call, "id", str(len(tool_calls))),
                            )
                        )

                    if tool_calls:
                        return llm.ChatChunk(
                            id=self._request_id,
                            delta=llm.ChoiceDelta(
                                role="assistant",
                                tool_calls=tool_calls,
                            ),
                        )

            elif event.event_type == "stream-end":
                if hasattr(event, "response"):
                    response = event.response
                    if hasattr(response, "meta") and hasattr(response.meta, "tokens"):
                        tokens = response.meta.tokens
                        if hasattr(tokens, "input_tokens"):
                            self._input_tokens = tokens.input_tokens
                        if hasattr(tokens, "output_tokens"):
                            self._output_tokens = tokens.output_tokens
                return None

        # Fallback for simple text streaming
        elif hasattr(event, "text"):
            return llm.ChatChunk(
                id=self._request_id,
                delta=llm.ChoiceDelta(content=event.text, role="assistant"),
            )

        return None
