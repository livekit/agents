from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

import httpx

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, llm
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
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
from mistralai import ChatCompletionChoice, ChatCompletionStreamRequestMessages, Mistral

from .models import ChatModels


@dataclass
class _LLMOptions:
    model: str
    temperature: NotGivenOr[float]
    max_completion_tokens: NotGivenOr[int]


# Mistral LLM Class
class LLM(llm.LLM):
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
        self._client = Mistral(api_key=api_key or os.environ.get("MISTRAL_API_KEY"))

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
    ) -> LLMStream:
        extra: dict[str, Any] = {}

        if is_given(self._opts.max_completion_tokens):
            extra["max_completion_tokens"] = self._opts.max_completion_tokens

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature

        return LLMStream(
            self,
            model=self._opts.model,
            client=self._client,
            chat_ctx=chat_ctx,
            provider_fmt=self._provider_fmt,
            extra_kwargs=extra,
        )


# Mistral LLM STREAM
class LLMStream(llm.LLMStream):
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
        extra_kwargs: dict[str, Any],
    ) -> None:
        if tools is None:
            tools = []
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._client = client
        self._llm = llm
        self._provider_fmt = provider_fmt
        self._extra_kwargs = extra_kwargs

    async def _run(self) -> None:
        # current function call that we're waiting for full completion (args are streamed)
        # (defined inside the _run method to make sure the state is reset for each run/attempt)
        retryable = True

        try:
            messages_mistral = self._chat_ctx.to_provider_format(format=self._provider_fmt)
            async_response = await self._client.chat.stream_async(
                messages=cast(list[ChatCompletionStreamRequestMessages], messages_mistral),
                model=self._model,
                **self._extra_kwargs,
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
        # 1) get the streaming delta
        delta = getattr(choice, "delta", None)
        if not (delta and delta.content):
            return None

        return llm.ChatChunk(
            id=id,
            delta=llm.ChoiceDelta(content=delta.content, role="assistant"),
        )
