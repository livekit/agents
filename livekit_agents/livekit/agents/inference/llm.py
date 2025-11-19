from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Literal, Union, cast

import httpx
import openai
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionPredictionContentParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)
from openai.types.chat.chat_completion_chunk import Choice
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params import Metadata
from typing_extensions import TypedDict

from .. import llm
from .._exceptions import APIConnectionError, APIStatusError, APITimeoutError
from ..llm import ToolChoice, utils as llm_utils
from ..llm.chat_context import ChatContext
from ..llm.tool_context import (
    FunctionTool,
    RawFunctionTool,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
from ._utils import create_access_token

lk_oai_debug = int(os.getenv("LK_OPENAI_DEBUG", 0))


OpenAIModels = Literal[
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-oss-120b",
]

GoogleModels = Literal[
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.0-flash",
    "google/gemini-2.0-flash-lite",
]

QwenModels = Literal["qwen/qwen3-235b-a22b-instruct"]

KimiModels = Literal["moonshotai/kimi-k2-instruct"]

DeepSeekModels = Literal["deepseek-ai/deepseek-v3"]

LLMModels = Union[OpenAIModels, GoogleModels, QwenModels, KimiModels, DeepSeekModels]


class ChatCompletionOptions(TypedDict, total=False):
    frequency_penalty: float | None
    logit_bias: dict[str, int] | None
    logprobs: bool | None
    max_completion_tokens: int | None
    max_tokens: int | None
    metadata: Metadata | None
    modalities: list[Literal["text", "audio"]] | None
    n: int | None
    parallel_tool_calls: bool
    prediction: ChatCompletionPredictionContentParam | None
    presence_penalty: float | None
    prompt_cache_key: str
    reasoning_effort: ReasoningEffort | None
    safety_identifier: str
    seed: int | None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None
    stop: str | None | list[str] | None
    store: bool | None
    temperature: float | None
    top_logprobs: int | None
    top_p: float | None
    user: str
    verbosity: Literal["low", "medium", "high"] | None
    web_search_options: completion_create_params.WebSearchOptions

    # livekit-typed arguments
    tool_choice: ToolChoice
    # TODO(theomonnomn): support repsonse format
    # response_format: completion_create_params.ResponseFormat


DEFAULT_BASE_URL = "https://agent-gateway.livekit.cloud/v1"


@dataclass
class _LLMOptions:
    model: LLMModels | str
    provider: str | None
    base_url: str
    api_key: str
    api_secret: str
    extra_kwargs: ChatCompletionOptions | dict[str, Any]


class LLM(llm.LLM):
    def __init__(
        self,
        model: LLMModels | str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        extra_kwargs: ChatCompletionOptions | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        lk_base_url = (
            base_url if base_url else os.environ.get("LIVEKIT_INFERENCE_URL", DEFAULT_BASE_URL)
        )

        lk_api_key = (
            api_key
            if api_key
            else os.getenv("LIVEKIT_INFERENCE_API_KEY", os.getenv("LIVEKIT_API_KEY", ""))
        )
        if not lk_api_key:
            raise ValueError(
                "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable"
            )

        lk_api_secret = (
            api_secret
            if api_secret
            else os.getenv("LIVEKIT_INFERENCE_API_SECRET", os.getenv("LIVEKIT_API_SECRET", ""))
        )
        if not lk_api_secret:
            raise ValueError(
                "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental variable"
            )

        self._opts = _LLMOptions(
            model=model,
            provider=provider,
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
            extra_kwargs=extra_kwargs or {},
        )
        self._client = openai.AsyncClient(
            api_key=create_access_token(self._opts.api_key, self._opts.api_secret),
            base_url=self._opts.base_url,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50, max_keepalive_connections=50, keepalive_expiry=120
                ),
            ),
        )

    @classmethod
    def from_model_string(cls, model: str) -> LLM:
        """Create a LLM instance from a model string"""
        return cls(model)

    @property
    def model(self) -> str:
        """Get the model name for this LLM instance."""
        return self._opts.model

    @property
    def provider(self) -> str:
        return "livekit"

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

        parallel_tool_calls = (
            parallel_tool_calls
            if is_given(parallel_tool_calls)
            else self._opts.extra_kwargs.get("parallel_tool_calls", NOT_GIVEN)
        )
        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        extra_tool_choice = self._opts.extra_kwargs.get("tool_choice", NOT_GIVEN)
        tool_choice = tool_choice if is_given(tool_choice) else extra_tool_choice  # type: ignore
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

        extra.update(self._opts.extra_kwargs)

        self._client.api_key = create_access_token(self._opts.api_key, self._opts.api_secret)
        return LLMStream(
            self,
            model=self._opts.model,
            provider=self._opts.provider,
            strict_tool_schema=True,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM | llm.LLM,
        *,
        model: LLMModels | str,
        provider: str | None = None,
        strict_tool_schema: bool,
        client: openai.AsyncClient,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
        provider_fmt: str = "openai",  # used internally for chat_ctx format
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._provider = provider
        self._provider_fmt = provider_fmt
        self._strict_tool_schema = strict_tool_schema
        self._client = client
        self._llm = llm
        self._extra_kwargs = extra_kwargs

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
            chat_ctx, _ = self._chat_ctx.to_provider_format(format=self._provider_fmt)
            fnc_ctx = (
                to_fnc_ctx(self._tools, strict=self._strict_tool_schema)
                if self._tools
                else openai.NOT_GIVEN
            )
            if lk_oai_debug:
                tool_choice = self._extra_kwargs.get("tool_choice", NOT_GIVEN)
                logger.debug(
                    "chat.completions.create",
                    extra={
                        "fnc_ctx": fnc_ctx,
                        "tool_choice": tool_choice,
                        "chat_ctx": chat_ctx,
                    },
                )
            if not self._tools:
                # remove tool_choice from extra_kwargs if no tools are provided
                self._extra_kwargs.pop("tool_choice", None)

            if self._provider:
                headers = self._extra_kwargs.setdefault("extra_headers", {})
                headers["X-LiveKit-Inference-Provider"] = self._provider

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
                                completion_tokens=chunk.usage.completion_tokens,
                                prompt_tokens=chunk.usage.prompt_tokens,
                                prompt_cached_tokens=cached_tokens or 0,
                                total_tokens=chunk.usage.total_tokens,
                            ),
                        )
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


def to_fnc_ctx(
    fnc_ctx: list[llm.FunctionTool | llm.RawFunctionTool], *, strict: bool = True
) -> list[ChatCompletionToolParam]:
    tools: list[ChatCompletionToolParam] = []
    for fnc in fnc_ctx:
        if is_raw_function_tool(fnc):
            info = get_raw_function_info(fnc)
            tools.append(
                {
                    "type": "function",
                    "function": info.raw_schema,  # type: ignore
                }
            )
        elif is_function_tool(fnc):
            schema = (
                llm.utils.build_strict_openai_schema(fnc)
                if strict
                else llm.utils.build_legacy_openai_schema(fnc)
            )
            tools.append(schema)  # type: ignore

    return tools
