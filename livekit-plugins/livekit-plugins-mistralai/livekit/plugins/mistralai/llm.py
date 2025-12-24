from __future__ import annotations

import json
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
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given, shortuuid
from mistralai import (
    ChatCompletionStreamRequestMessagesTypedDict,
    CompletionResponseStreamChoice,
    Mistral,
    ToolTypedDict,
)

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
    ) -> None:
        super().__init__()
        self._opts = _LLMOptions(
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        self._client = Mistral(api_key=api_key or os.environ.get("MISTRAL_API_KEY"))

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "MistralAI"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
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

        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        if is_given(tool_choice):
            extra["tool_choice"] = tool_choice

        if is_given(response_format):
            extra["response_format"] = response_format

        return LLMStream(
            self,
            model=self._opts.model,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )


# Mistral LLM STREAM
class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm_v: LLM,
        *,
        model: str | ChatModels,
        client: Mistral,
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm_v, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._client = client
        self._llm = llm_v
        self._extra_kwargs = extra_kwargs
        self._tool_ctx = llm.ToolContext(tools)

    async def _run(self) -> None:
        # current function call that we're waiting for full completion (args are streamed)
        # (defined inside the _run method to make sure the state is reset for each run/attempt)
        retryable = True

        try:
            messages, _ = self._chat_ctx.to_provider_format(format="mistralai")
            tools = self._tool_ctx.parse_function_tools("openai", strict=True)

            async_response = await self._client.chat.stream_async(
                messages=cast(list[ChatCompletionStreamRequestMessagesTypedDict], messages),
                tools=cast(list[ToolTypedDict], tools),
                model=self._model,
                timeout_ms=int(self._conn_options.timeout * 1000),
                **self._extra_kwargs,
            )
            async for ev in async_response:
                if not ev.data.choices:
                    continue
                choice = ev.data.choices[0]
                chat_chunk = self._parse_choice(ev.data.id, choice)
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

    def _parse_choice(self, id: str, choice: CompletionResponseStreamChoice) -> ChatChunk | None:
        chunk = llm.ChatChunk(id=id)
        if choice.delta.content and isinstance(choice.delta.content, str):
            # TODO: support thinking chunks
            chunk.delta = llm.ChoiceDelta(content=choice.delta.content, role="assistant")

        if choice.delta.tool_calls:
            if not chunk.delta:
                chunk.delta = llm.ChoiceDelta(role="assistant")

            for tool in choice.delta.tool_calls:
                arguments = tool.function.arguments
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                call_id = tool.id or shortuuid("tool_call_")

                chunk.delta.tool_calls.append(
                    llm.FunctionToolCall(
                        name=tool.function.name, arguments=arguments, call_id=call_id
                    )
                )
        return chunk if chunk.delta else None
