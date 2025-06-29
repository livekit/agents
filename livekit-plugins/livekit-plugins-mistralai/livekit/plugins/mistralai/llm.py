from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from mistralai import AssistantMessage, ChatCompletionChoice, Mistral, SystemMessage, UserMessage

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, llm
from livekit.agents.llm import (
    LLM,
    ChatChunk,
    ChatContext,
    LLMStream,
    ToolChoice,
    utils as llm_utils,
)
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .models import ChatModels


def to_async_stream_mistral_format(chat_ctx: ChatContext):
    """
    Custom function to change livekit ChatContext Object to
    Mistral injectable AsyncStreaming Object (ChatCompletionStreamRequestMessages)
    """
    if isinstance(chat_ctx, ChatContext):
        messages = chat_ctx.to_dict().get("items")  # transform chatContext to dict for processing
        print("OK")
        messages_mistral = []
        for element in messages:
            if element["role"] == "assistant":
                assistant_msg = AssistantMessage(content=element["content"][0])
                messages_mistral.append(assistant_msg)
            elif element["role"] == "user":
                user_msg = UserMessage(content=element["content"][0])
                messages_mistral.append(user_msg)
            elif element["role"] == "system":
                system_msg = SystemMessage(content=element["content"][0])
                messages_mistral.append(system_msg)

    return messages_mistral


@dataclass
class _LLMOptions:
    model: str
    temperature: NotGivenOr[float]
    max_completion_tokens: NotGivenOr[int]


# Mistral LLM Class
class MistralLLM(LLM):
    def __init__(
        self,
        model: str | ChatModels = "ministral-8b-2410",
        api_key: str | None = None,
        client: Mistral | None = None,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        _provider_fmt: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        super().__init__()
        self._opts = _LLMOptions(
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        self._provider_fmt = _provider_fmt or "mistralai"
        self._client = Mistral(api_key=api_key or os.environ["MISTRAL_API_KEY"])

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[type[llm_utils.ResponseFormatT]] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> MistralLLMStream:
        extra = {}

        if is_given(self._opts.max_completion_tokens):
            extra["max_completion_tokens"] = self._opts.max_completion_tokens

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature

        return MistralLLMStream(
            self,
            model=self._opts.model,
            client=self._client,
            chat_ctx=chat_ctx,
            provider_fmt=self._provider_fmt,
        )


# Mistral LLM STREAM
class MistralLLMStream(LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        model: str | ChatModels,
        provider_fmt: str,
        client: Mistral,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._client = client
        self._llm = llm
        self._provider_fmt = provider_fmt

    async def _run(self) -> None:
        # current function call that we're waiting for full completion (args are streamed)
        # (defined inside the _run method to make sure the state is reset for each run/attempt)
        self._mai_stream: None = None
        retryable = True

        try:
            messages = to_async_stream_mistral_format(chat_ctx=self._chat_ctx)
            async_response = await self._client.chat.stream_async(
                messages=messages,
                model=self._model,
            )
            print("Streaming Start")
            async for chunk in async_response:
                for choice in chunk.data.choices:
                    chat_chunk = self._parse_choice(chunk.data.id, choice)
                    if chat_chunk is not None:
                        retryable = False
                        self._event_ch.send_nowait(chat_chunk)

        except APITimeoutError:
            raise APITimeoutError(retryable=retryable) from None
        except APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
                retryable=retryable,
            ) from None
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    def _parse_choice(self, id: str, choice: ChatCompletionChoice) -> ChatChunk | None:
        delta = choice.delta

        if delta is None:  # 1st case delta is None
            return None

        call_chunk = None

        call_chunk = llm.ChatChunk(
            id=id, delta=llm.ChoiceDelta(role="assistant", content=delta.content)
        )

        if call_chunk is not None:
            return call_chunk

        if choice.finish_reason == "stop":
            call_chunk = llm.ChatChunk(
                id=id, delta=llm.ChoiceDelta(role="assistant", content=delta.content)
            )

            return call_chunk

        delta.content = llm_utils.strip_thinking_tokens(delta.content)

        if not delta.content:
            return None

        return llm.ChatChunk(
            id=id,
            delta=llm.ChoiceDelta(content=delta.content, role="assistant"),
        )
